"""
The original paper:
https://arxiv.org/pdf/1606.00915.pdf
This architecture comes from this implementation:
https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow/blob/master/model.py
"""
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import array_ops
import util_functions
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNet
from bohaoCustom import uabMakeNetwork as network
import uabRepoPaths
import imageio
import util_functions
import uabUtilreader
import uabDataReader

class DeeplabV2(uabMakeNetwork_UNet.UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'DeeplabV2'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        self.input_size = self.inputs[x_name].shape[1:3]

        self.encoding = self.build_encoder(x_name)
        self.pred = self.build_decoder(self.encoding)
        self.output_size = self.pred.shape[1:3]

        self.output = tf.image.resize_bilinear(tf.nn.softmax(self.pred), self.input_size)

    def make_loss(self, y_name, loss_type='xent', lamb_BGC= 0.2, **kwargs):
        # TODO loss type IoU
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_resize = tf.image.resize_nearest_neighbor(self.inputs[y_name], self.output_size)
            y_flat = tf.reshape(tf.squeeze(y_resize, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])

            if loss_type == 'xent':
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
            elif loss_type == 'BGC':
                y = y_resize
                pred_real = tf.reshape(self.pred[:self.source_control[0], :, :, :], [-1, self.class_num])
                pred_fake = tf.reshape(self.pred[self.source_control[0]:, :, :, :], [-1, self.class_num])

                y_real = tf.reshape(tf.squeeze(y[:self.source_control[0], :, :, :], axis=[3]), [-1, ])
                y_fake = tf.reshape(tf.squeeze(y[self.source_control[0]:, :, :, :], axis=[3]), [-1, ])

                indices_real = tf.squeeze(tf.where(tf.less_equal(y_real, self.class_num - 1)), 1)
                indices_fake = tf.squeeze(tf.where(tf.less_equal(y_fake, self.class_num - 1)), 1)
                pred_real = tf.gather(pred_real, indices_real)
                pred_fake = tf.gather(pred_fake, indices_fake)

                gt_real = tf.gather(y_real, indices_real)
                gt_fake = tf.gather(y_fake, indices_fake)

                self.loss_real = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_real, labels=gt_real))
                self.loss_fake = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_fake, labels=gt_fake))

                self.loss = self.loss_real + lamb_BGC * self.loss_fake
            else:
                # focal loss: this comes from
                # https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
                if 'alpha' not in kwargs:
                    kwargs['alpha'] = 0.25
                if 'gamma' not in kwargs:
                    kwargs['gamma'] = 2
                gt = tf.one_hot(gt, depth=2, dtype=tf.float32)
                sigmoid_p = tf.nn.sigmoid(prediction)
                zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
                pos_p_sub = array_ops.where(gt >= sigmoid_p, gt - sigmoid_p, zeros)
                neg_p_sub = array_ops.where(gt > zeros, zeros, sigmoid_p)
                per_entry_cross_ent = - kwargs['alpha'] * (pos_p_sub ** kwargs['gamma']) * tf.log(tf.clip_by_value(
                    sigmoid_p, 1e-8, 1.0)) - (1- kwargs['alpha']) * (neg_p_sub ** kwargs['gamma']) * tf.log(
                    tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
                self.loss = tf.reduce_sum(per_entry_cross_ent)

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None:
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs/n_train*self.bs))
            start_step = gs - int(start_epoch*n_train/self.bs)
        else:
            start_epoch = 0
            start_step = 0

        cross_entropy_valid_min = np.inf
        iou_valid_max = 0
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs):
                X_batch, y_batch = train_reader.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            cross_entropy_valid_mean = []
            iou_valid_mean = np.zeros(2)
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run([self.pred, self.loss, self.loss_iou],
                                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                                 self.inputs[y_name]: y_batch_val,
                                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Validation cross entropy: {:.3f}, duration: {:.3f}'.format(cross_entropy_valid_mean,
                                                                                  duration))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean})
            valid_iou_summary = sess.run(valid_iou_summary_op,
                                         feed_dict={self.valid_iou: iou_valid_mean})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)
            summary_writer.add_summary(valid_iou_summary, self.global_step_value)
            if valid_iou:
                if iou_valid_mean > iou_valid_max:
                    iou_valid_max = iou_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            else:
                if cross_entropy_valid_mean < cross_entropy_valid_min:
                    cross_entropy_valid_min = cross_entropy_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            if image_summary is not None:
                pred_valid = sess.run(tf.image.resize_bilinear(pred_valid, self.input_size))
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            fineTune=False, valid_iou=False, best_model=True):
        if gpu is not None:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if isTrain:
            coord = tf.train.Coordinator()
            with tf.Session(config=self.config) as sess:
                # init model
                init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                # load model
                if pretrained_model_dir is not None:
                    if layers2load is not None:
                        self.load_weights(pretrained_model_dir, layers2load)
                    else:
                        if not fineTune:
                            restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
                            loader = tf.train.Saver(var_list=restore_var)
                            self.load(pretrained_model_dir, sess, loader, epoch=load_epoch_num)
                        else:
                            self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val

    def build_encoder(self, x_name):
        print("-----------build encoder: deeplab initial-----------")
        outputs = self._start_block(x_name)
        print("after start block:", outputs.shape)
        outputs = self._bottleneck_resblock(outputs, 256, '2a', identity_connection=False)
        outputs = self._bottleneck_resblock(outputs, 256, '2b')
        outputs = self._bottleneck_resblock(outputs, 256, '2c')
        print("after block1:", outputs.shape)
        outputs = self._bottleneck_resblock(outputs, 512, '3a', half_size=True, identity_connection=False)
        for i in range(1, 4):
            outputs = self._bottleneck_resblock(outputs, 512, '3b%d' % i)
        print("after block2:", outputs.shape)
        outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4a', identity_connection=False)
        for i in range(1, 23):
            outputs = self._dilated_bottle_resblock(outputs, 1024, 2, '4b%d' % i)
        print("after block3:", outputs.shape)
        outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5a', identity_connection=False)
        outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5b')
        outputs = self._dilated_bottle_resblock(outputs, 2048, 4, '5c')
        print("after block4:", outputs.shape)
        return outputs

    def build_decoder(self, encoding):
        print("-----------build decoder-----------")
        outputs = self._ASPP(encoding, self.class_num, [6, 12, 18, 24])
        #outputs = tf.image.resize_bilinear(outputs, self.input_size)
        print("after aspp block:", outputs.shape)
        return outputs

    # blocks
    def _start_block(self, x_name):
        outputs = self._conv2d(self.inputs[x_name], 7, 64, 2, name='conv1')
        outputs = self._batch_norm(outputs, name='bn_conv1', is_training=False, activation_fn=tf.nn.relu)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        return outputs

    def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
        first_s = 2 if half_size else 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, first_s, name='res%s_branch1' % name)
            o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='res%s_branch2a' % name)
        o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='res%s_branch2b' % name)
        o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
        o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='res%s' % name)
        # relu
        outputs = self._relu(outputs, name='res%s_relu' % name)
        return outputs

    def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, 1, name='res%s_branch1' % name)
            o_b1 = self._batch_norm(o_b1, name='bn%s_branch1' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='res%s_branch2a' % name)
        o_b2a = self._batch_norm(o_b2a, name='bn%s_branch2a' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='res%s_branch2b' % name)
        o_b2b = self._batch_norm(o_b2b, name='bn%s_branch2b' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='res%s_branch2c' % name)
        o_b2c = self._batch_norm(o_b2c, name='bn%s_branch2c' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='res%s' % name)
        # relu
        outputs = self._relu(outputs, name='res%s_relu' % name)
        return outputs

    def _ASPP(self, x, num_o, dilations):
        o = []
        for i, d in enumerate(dilations):
            o.append(self._dilated_conv2d(x, 3, num_o, d, name='fc1_voc12_c%d' % i, biased=True))
        return self._add(o, name='fc1_voc12')

        # layers

    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
        """
        Conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name):
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            s = [1, stride, stride, 1]
            o = tf.nn.conv2d(x, w, s, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
        """
        Dilated conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name):
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _relu(self, x, name):
        return tf.nn.relu(x, name=name)

    def _add(self, x_l, name):
        return tf.add_n(x_l, name=name)

    def _max_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

    def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
        # For a small batch size, it is better to keep
        # the statistics of the BN layers (running means and variances) frozen,
        # and to not update the values provided by the pre-trained model by setting is_training=False.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name) as scope:
            o = tf.contrib.layers.batch_norm(
                x,
                scale=True,
                activation_fn=activation_fn,
                is_training=is_training,
                trainable=trainable,
                scope=scope)
            return o


class DeeplabV3(DeeplabV2):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'DeeplabV3'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        self.input_size = self.inputs[x_name].shape[1:3]

        self.encoding = self.build_encoder(x_name)
        self.pred = self.build_decoder(self.encoding)
        self.output_size = self.pred.shape[1:3]

        self.output = tf.image.resize_bilinear(tf.nn.softmax(self.pred), self.input_size)

    def build_encoder(self, x_name):
        print("-----------build encoder-----------" )
        scope_name = 'resnet_v1_101'
        with tf.variable_scope(scope_name) as scope:
            outputs = self._start_block('conv1', x_name)
            print("after start block:", outputs.shape)
            with tf.variable_scope('block1') as scope:
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_1', identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')
                print("after block1:", outputs.shape)
            with tf.variable_scope('block2') as scope:
                outputs = self._bottleneck_resblock(outputs, 512, 'unit_1', half_size=True, identity_connection=False)
                for i in range(2, 5):
                    outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)
                print("after block2:", outputs.shape)
            with tf.variable_scope('block3') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1', identity_connection=False)
                num_layers_block3 = 23
                for i in range(2, num_layers_block3 + 1):
                    outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)
                print("after block3:", outputs.shape)
            with tf.variable_scope('block4') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')
                print("after block4:", outputs.shape)
                return outputs

    def build_decoder(self, encoding):
        print("-----------build decoder-----------")
        with tf.variable_scope('decoder') as scope:
            outputs = self._ASPP(encoding, self.class_num, [6, 12, 18, 24])
            print("after aspp block:", outputs.shape)
            return outputs

            # blocks

    def _start_block(self, name, x_name):
        outputs = self._conv2d(self.inputs[x_name], 7, 64, 2, name=name)
        outputs = self._batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
        outputs = self._max_pool2d(outputs, 3, 2, name='pool1')
        return outputs

    def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
        first_s = 2 if half_size else 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, first_s, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False,
                                    activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, first_s, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2b = self._conv2d(o_b2a, 3, num_o / 4, 1, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = self._conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name)
            o_b1 = self._batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False,
                                    activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = self._conv2d(x, 1, num_o / 4, 1, name='%s/bottleneck_v1/conv1' % name)
        o_b2a = self._batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2b = self._dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name)
        o_b2b = self._batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False,
                                 activation_fn=tf.nn.relu)

        o_b2c = self._conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name)
        o_b2c = self._batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = self._add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = self._relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _ASPP(self, x, num_o, dilations):
        o = []
        for i, d in enumerate(dilations):
            o.append(self._dilated_conv2d(x, 3, num_o, d, name='aspp/conv%d' % (i + 1), biased=True))
        return self._add(o, name='aspp/add')

        # layers

    def _conv2d(self, x, kernel_size, num_o, stride, name, biased=False):
        """
        Conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            s = [1, stride, stride, 1]
            o = tf.nn.conv2d(x, w, s, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _dilated_conv2d(self, x, kernel_size, num_o, dilation_factor, name, biased=False):
        """
        Dilated conv2d without BN or relu.
        """
        num_x = x.shape[self.channel_axis].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
            return o

    def _relu(self, x, name):
        return tf.nn.relu(x, name=name)

    def _add(self, x_l, name):
        return tf.add_n(x_l, name=name)

    def _max_pool2d(self, x, kernel_size, stride, name):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(x, k, s, padding='SAME', name=name)

    def _batch_norm(self, x, name, is_training, activation_fn, trainable=False):
        # For a small batch size, it is better to keep
        # the statistics of the BN layers (running means and variances) frozen,
        # and to not update the values provided by the pre-trained model by setting is_training=False.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.
        # Set trainable = False to remove them from trainable_variables.
        with tf.variable_scope(name + '/BatchNorm') as scope:
            o = tf.contrib.layers.batch_norm(
                x,
                scale=True,
                activation_fn=activation_fn,
                is_training=is_training,
                trainable=trainable,
                scope=scope)
            return o

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False,  isOneShotTest=False,img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            fineTune=False, valid_iou=False, best_model=True):
        if gpu is not None:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if isTrain:
            coord = tf.train.Coordinator()
            with tf.Session(config=self.config) as sess:
                # init model
                init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                # load model
                if pretrained_model_dir is not None:
                    if layers2load is not None:
                        self.load_weights(pretrained_model_dir, layers2load)
                    else:
                        if not fineTune:
                            restore_var = [v for v in tf.global_variables() if 'resnet_v1' in v.name and
                                       not 'Adam' in v.name]
                            loader = tf.train.Saver(var_list=restore_var)
                            self.load(pretrained_model_dir, sess, loader, epoch=load_epoch_num)
                        else:
                            self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        elif isOneShotTest:
            coord = tf.train.Coordinator()
            with tf.Session(config=self.config) as sess:
                # init model
                init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
                sess.run(init)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                # load model
                self.load(continue_dir, sess, saver, epoch=load_epoch_num)
                print('Successfully load model')
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.one_shot_test('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, test_reader=train_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
        else:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val

    def one_shot_test(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
                      test_reader=None,
                      image_summary=None, verb_step=100, save_epoch=5,
                      img_mean=np.array((0, 0, 0), dtype=np.float32),
                      continue_dir=None, valid_iou=False, ds_name='deepglobe'):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)
        self.model_name = continue_dir.split('/')[-1]
        score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
        if not os.path.exists(score_save_dir):
            os.makedirs(score_save_dir)
        with open(os.path.join(score_save_dir, 'result.txt'), 'w'):
            pass
        cross_entropy_valid_min = np.inf
        iou_valid_max = 0
        count = 1
        iou_record = np.zeros(2)
        for epoch in range(1):
            start_time = time.time()
            for step in range(0, n_train, self.bs):
                print('-'*10+'Begin evaluate on batch'+str(step)+'-'*10)
                X_batch_test, y_batch_test = test_reader.readerAction(sess)
                pred_map_test, iou_test = sess.run([self.output, self.loss_iou],
                                                     feed_dict={self.inputs[x_name]: X_batch_test,
                                                                self.inputs[y_name]: y_batch_test,
                                                                self.trainable: False})
                iou_record += iou_test

        mean_iou_record = iou_record[0] / iou_record[1]
        print('Overall mean IoU={:.3f}'.format(mean_iou_record))
        print('We are using this model', self.model_name)

    def evaluate_e_city(self, rgb_list, gt_list, rgb_dir, gt_dir, input_size, tile_size, batch_size, img_mean,
                        model_dir, gpu=None, save_result=True, save_result_parent_dir=None, show_figure=False,
                        verb=True, ds_name='default', load_epoch_num=None, best_model=True, test_mode=None):
        if show_figure:
            import matplotlib.pyplot as plt

        if save_result:
            self.model_name = model_dir.split('/')[-1]
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir,
                                              self.model_name, ds_name)
            if not os.path.exists(score_save_dir):
                os.makedirs(score_save_dir)
            with open(os.path.join(score_save_dir, 'result.txt'), 'w'):
                pass

        iou_record = []
        mid_iou_record = []
        iou_return = {}
        count = 0
        for file_name, file_name_truth in zip(rgb_list, gt_list):
            count = count + 1
            tile_name = file_name_truth.split('_')[0]
            if verb:
                print('Evaluating {} ... '.format(tile_name))
            start_time = time.time()

            # prepare the reader
            reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                    dataInds=[0],
                                                    nChannels=3,
                                                    parentDir=rgb_dir,
                                                    chipFiles=[file_name],
                                                    chip_size=input_size,
                                                    tile_size=tile_size,
                                                    batchSize=batch_size,
                                                    block_mean=img_mean,
                                                    overlap=self.get_overlap(),
                                                    padding=np.array((self.get_overlap() / 2, self.get_overlap() / 2)),
                                                    isTrain=False)
            rManager = reader.readManager

            # run the model
            pred = self.run(pretrained_model_dir=model_dir,
                            test_reader=rManager,
                            tile_size=tile_size,
                            patch_size=input_size,
                            gpu=gpu, load_epoch_num=load_epoch_num, best_model=best_model)

            truth_label_img = imageio.imread(os.path.join(gt_dir, file_name_truth))
            pred_for_iou = pred

            iou = util_functions.iou_metric(truth_label_img, pred_for_iou, divide_flag=True)
            iou_record.append(iou)
            iou_return[tile_name] = iou

            duration = time.time() - start_time
            if verb:
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0] / iou[1], duration))
            mid_iou_record.append(iou)
            if count % 5 == 0:
                mean_iou = np.sum(np.array(mid_iou_record)[:, 0]) / np.sum(np.array(mid_iou_record)[:, 1])
                print(str(count) + ':  ' + 'Overall mean IoU={:.3f}'.format(mean_iou))
                mid_iou_record = []

            # save results
            if save_result:
                pred_save_dir = os.path.join(score_save_dir, 'pred')
                if not os.path.exists(pred_save_dir):
                    os.makedirs(pred_save_dir)
                imageio.imsave(os.path.join(pred_save_dir, tile_name + '.png'), pred.astype(np.uint8))
                with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                    file.write('{} {}\n'.format(tile_name, iou))

            if show_figure:
                plt.figure(figsize=(12, 4))
                ax1 = plt.subplot(121)
                ax1.imshow(truth_label_img)
                plt.title('Truth')
                ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
                ax2.imshow(pred)
                plt.title('pred')
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0],
                                                                 iou[0] / iou[1]))
                plt.show()

        iou_record = np.array(iou_record)
        mean_iou = np.sum(iou_record[:, 0]) / np.sum(iou_record[:, 1])
        print('Overall mean IoU={:.3f}'.format(mean_iou))
        if save_result:
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir, self.model_name,
                                              ds_name)
            with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                file.write('{}'.format(mean_iou))

        return iou_return

class DeeplabV3_MTinput(DeeplabV3):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, source_num=2, source_name=None):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'DeeplabV3'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None
        self.source_num = source_num
        if source_name is None or len(source_name) != source_num:
            self.source_name = ['D{}'.format(i) for i in range(self.source_num)]
        else:
            self.source_name = source_name

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False, valid_source_idx=0):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None:
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs/n_train*self.bs))
            start_step = gs - int(start_epoch*n_train/self.bs)
        else:
            start_epoch = 0
            start_step = 0

        cross_entropy_valid_min = np.inf
        iou_valid_max = 0
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs):
                X_batch_total = []
                y_batch_total = []
                for s in range(self.source_num):
                    X_batch, y_batch = train_reader[s].readerAction(sess)
                    X_batch_total.append(X_batch)
                    y_batch_total.append(y_batch)
                X_batch_total = np.concatenate(X_batch_total, axis=0)
                y_batch_total = np.concatenate(y_batch_total, axis=0)
                #X_batch, y_batch = train_reader[s].readerAction(sess)
                #X_batch, y_batch = train_reader.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch_total,
                                                                self.inputs[y_name]: y_batch_total,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch_total,
                                                                                       self.inputs[y_name]: y_batch_total,
                                                                                       self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, step, step_cross_entropy))

            # validation
            cross_entropy_valid_mean = []
            iou_valid_mean = np.zeros(2)
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run([self.pred, self.loss, self.loss_iou],
                                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                                 self.inputs[y_name]: y_batch_val,
                                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Validation cross entropy: {:.3f}, duration: {:.3f}'.format(cross_entropy_valid_mean,
                                                                                  duration))
            valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op,
                                                   feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean})
            valid_iou_summary = sess.run(valid_iou_summary_op,
                                         feed_dict={self.valid_iou: iou_valid_mean})
            summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)
            summary_writer.add_summary(valid_iou_summary, self.global_step_value)
            if valid_iou:
                if iou_valid_mean > iou_valid_max:
                    iou_valid_max = iou_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            else:
                if cross_entropy_valid_mean < cross_entropy_valid_min:
                    cross_entropy_valid_min = cross_entropy_valid_mean
                    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                    saver.save(sess, '{}/best_model.ckpt'.format(self.ckdir))

            if image_summary is not None:
                pred_valid = sess.run(tf.image.resize_bilinear(pred_valid, self.input_size))
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)