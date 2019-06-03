import os
import re
import time
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import array_ops
import util_functions
import uabUtilreader
import uabDataReader
import uabRepoPaths
from bohaoCustom import uabMakeNetwork as network
from skimage.measure import label
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.misc import imread


class UnetModel(network.Network):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'Unet'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None
        self.n_train = 0
        self.n_valid = 0

    def make_ckdir(self, ckdir, patch_size, par_dir=None):
        if type(patch_size) is list:
            patch_size = patch_size[0]
        # make unique directory for save
        dir_name = '{}_PS{}_BS{}_EP{}_LR{}_DS{}_DR{}_SFN{}'.\
            format(self.model_name, patch_size, self.bs, self.epochs, self.lr, self.ds, self.dr, self.sfn)
        if par_dir is None:
            self.ckdir = os.path.join(ckdir, dir_name)
        else:
            self.ckdir = os.path.join(ckdir, par_dir, dir_name)

    def create_graph(self, x_name, class_num):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', dropout=self.dropout_rate)
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, dropout=self.dropout_rate)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, dropout=self.dropout_rate)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, dropout=self.dropout_rate)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, dropout=self.dropout_rate)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, dropout=self.dropout_rate)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        self.output = tf.nn.softmax(self.pred)

    def load_weights(self, ckpt_dir, layers2load):
        # this is different from network.load()
        # this function only loads specified layers
        layers_list = []
        if isinstance(layers2load, str):
            layers2load = [int(a) for a in layers2load.split(',')]
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        try:
            latest_check_point = tf.train.latest_checkpoint(ckpt_dir)
            tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)
            print('loaded {}'.format(latest_check_point))
        except tf.errors.NotFoundError:
            with open(os.path.join(ckpt_dir, 'checkpoint'), 'r') as f:
                ckpts = f.readlines()
            ckpt_file_name = ckpts[0].split('/')[-1].strip().strip('\"')
            latest_check_point = os.path.join(ckpt_dir, ckpt_file_name)
            tf.contrib.framework.init_from_checkpoint(latest_check_point, load_dict)
            print('loaded {}'.format(latest_check_point))

    def restore_model(self,sess):
        # automatically restore last saved model if checkpoint exists
        if tf.train.latest_checkpoint(self.ckdir): 

            self.load(self.ckdir,sess)

            with open(os.path.join(self.ckdir,'checkpoint'),'r') as f:
                model_checkpoint_path = f.readline().split('/')[-1]
            buf = [int(i) for i in re.findall(r"\d+", model_checkpoint_path)]
            if len(buf) == 1:
                start_step = buf[0]+1
                self.start_epoch = int(np.floor(start_step/(8000/self.bs)))
            elif len(buf) == 2:
                self.start_epoch = buf[0]+1
                start_step = buf[1]+1
        else:
            self.start_epoch,start_step = [0,0]
            
        sess.run(self.global_step.assign(start_step))
        self.global_step_value = self.global_step.eval()
        print('restoring model from epoch %d step %d'%(self.start_epoch,self.global_step_value))

    def make_learning_rate(self, n_train):
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                        tf.cast(n_train/self.bs * self.ds, tf.int32),
                                                        self.dr, staircase=True)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        # TODO loss type IoU
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
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
            elif loss_type == 'regularization':
                l2_loss = tf.losses.get_regularization_loss()
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))+ 0.5*l2_loss

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

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step)
            else:
                print('Train parameters in scope:')
                for layer in train_var_filter:
                    print(layer)
                train_vars = tf.trainable_variables()
                var_list = []
                for var in train_vars:
                    if var.name.split('/')[0] in train_var_filter:
                        var_list.append(var)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step,
                                                                                     var_list=var_list)

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train_config(self, x_name, y_name, n_train, n_valid, patch_size, ckdir, loss_type='regularization', train_var_filter=None,
                     hist=False, par_dir=None, **kwargs):
        self.make_loss(y_name, loss_type, **kwargs)
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size, par_dir)
        self.make_summary(hist)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.n_train = n_train
        self.n_valid = n_valid

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

        if continue_dir is not None and os.path.exists(continue_dir):
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
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def test(self, x_name, sess, test_iterator):
        result = []
        for X_batch in test_iterator:
            pred = sess.run(self.output, feed_dict={self.inputs[x_name]: X_batch,
                                                    self.trainable: False})
            result.append(pred)
        result = np.vstack(result)
        return result

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 0

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True):
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
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val

    def evaluate(self, rgb_list, gt_list, rgb_dir, gt_dir, input_size, tile_size, batch_size, img_mean,
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
        iou_return = {}
        for file_name, file_name_truth in zip(rgb_list, gt_list):
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
                                                    padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
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
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0]/iou[1], duration))

            # save results
            if save_result:
                pred_save_dir = os.path.join(score_save_dir, 'pred')
                if not os.path.exists(pred_save_dir):
                    os.makedirs(pred_save_dir)
                imageio.imsave(os.path.join(pred_save_dir, tile_name+'.png'), pred.astype(np.uint8))
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
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0], iou[0]/iou[1]))
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

    def evaluate_multi_criteria(self, rgb_list, gt_list, rgb_dir, gt_dir, input_size, tile_size, batch_size, img_mean,
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
        m_iou_record = []
        accuracy_record = []
        precision_record = []
        recall_record = []
        performance_score_record = []
        F1_record = []
        iou_return = {}
        for file_name, file_name_truth in zip(rgb_list, gt_list):
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
                                                    padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
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
            img_area = pred_for_iou.shape[0] * pred_for_iou.shape[1]
            TP = np.sum(truth_label_img*pred_for_iou)
            FP = np.sum(np.where(pred_for_iou == 1, 1, 0)*np.where(truth_label_img != 1, 1, 0))
            TN = np.sum((1-truth_label_img)*(1-pred_for_iou))
            FN = np.sum(np.where(pred_for_iou == 0, 1, 0)*np.where(truth_label_img != 0, 1, 0))
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            iou = util_functions.iou_metric(truth_label_img, pred_for_iou, divide_flag=True)
            accuracy_record.append(np.sum(np.where(pred_for_iou == truth_label_img, 1, 0))/img_area)
            precision_record.append(precision)
            recall_record.append(recall)
            F1_record.append(2*((precision * recall)/(precision+recall)))
            performance_score_record.append(np.mean([TP/(TP+FN), TN/(TN+FP)]))
            m_iou_record.append(np.mean([iou[0]/iou[1], np.sum((1-truth_label_img)*(1-pred_for_iou))/np.sum(np.where(((1-truth_label_img)+(1-pred_for_iou)) >  0, 1, 0))]))
            iou_record.append(iou)
            iou_return[tile_name] = iou

            duration = time.time() - start_time
            if verb:
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0]/iou[1], duration))

            # save results
            if save_result:
                pred_save_dir = os.path.join(score_save_dir, 'pred')
                if not os.path.exists(pred_save_dir):
                    os.makedirs(pred_save_dir)
                imageio.imsave(os.path.join(pred_save_dir, tile_name+'.png'), pred.astype(np.uint8))
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
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0], iou[0]/iou[1]))
                plt.show()

        iou_record = np.array(iou_record)
        mean_iou = np.sum(iou_record[:, 0]) / np.sum(iou_record[:, 1])
        mean_t_iou = mean_iou
        mean_m_iou = np.mean(m_iou_record)
        mean_accuracy = np.mean(accuracy_record)
        mean_precision = np.mean(precision_record)
        mean_recall = np.mean(recall_record)
        mean_F1 = np.mean(F1_record)
        mean_performance_score = np.mean(performance_score_record)
        print('Overall mean IoU={:.3f}'.format(mean_iou))
        print('Overall mean mIoU={:.3f}'.format(mean_m_iou))
        print('Overall mean Accuracy={:.3f}'.format(mean_accuracy))
        print('Overall mean Precision={:.3f}'.format(mean_precision))
        print('Overall mean Recall={:.3f}'.format(mean_recall))
        print('Overall mean F1={:.3f}'.format(mean_F1))
        print('Overall mean Performance Score={:.3f}'.format(mean_performance_score))

        if save_result:
            if save_result_parent_dir is None:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, self.model_name, ds_name)
            else:
                score_save_dir = os.path.join(uabRepoPaths.evalPath, save_result_parent_dir, self.model_name,
                                              ds_name)
            with open(os.path.join(score_save_dir, 'result.txt'), 'a+') as file:
                file.write('{}'.format(mean_iou))

        return iou_return,mean_t_iou, mean_m_iou, mean_accuracy, mean_precision, mean_recall, mean_F1, mean_performance_score


class UnetModelCrop(UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetCrop'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                           padding='valid', dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                           padding='valid', dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                           padding='valid', dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                           padding='valid', dropout=self.dropout_rate)
        self.encoding = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        # upsample
        up6 = self.crop_upsample_concat(self.encoding, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        self.output = tf.nn.softmax(self.pred)

    def make_loss(self, y_name, loss_type='regularization', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
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
            elif loss_type == 'regularization':
                l2_loss = tf.losses.get_regularization_loss()
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)) + 0.1*l2_loss
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
                pos_p_sub = array_ops.where(gt > sigmoid_p, gt - sigmoid_p, zeros)
                neg_p_sub = array_ops.where(gt > zeros, zeros, sigmoid_p)
                per_entry_cross_ent = - kwargs['alpha'] * (pos_p_sub ** kwargs['gamma']) * tf.log(tf.clip_by_value(
                    sigmoid_p, 1e-8, 1.0)) - (1- kwargs['alpha']) * (neg_p_sub ** kwargs['gamma']) * tf.log(
                    tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
                self.loss = tf.reduce_sum(per_entry_cross_ent)

    def load_weights_append_first_layer(self, ckpt_dir, layers2load, conv1_weight, check_weight=False):
        # this functino load weights from pretrained model and add extra filters to first layer
        layers_list = []
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id == 1:
                layers_list.append('layerconv1/conv_1/bias:0')
                layers_list.append('layerconv1/bn_1')
                layers_list.append('layerconv1/conv_2')
                layers_list.append('layerconv1/bn_2')
                continue
            elif layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)

        layerconv1_kernel = tf.trainable_variables()[0]
        assign_op = layerconv1_kernel.assign(conv1_weight)
        with tf.Session() as sess:
            sess.run(assign_op)
            weight = sess.run(layerconv1_kernel)

        if check_weight:
            import matplotlib.pyplot as plt
            _, _, c_num, _ = weight.shape
            for i in range(c_num):
                plt.subplot(321+i)
                plt.imshow(weight[:, :, i, :].reshape((16, 18)))
                plt.colorbar()
                plt.title(i)
            plt.show()

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 184

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, isOneShotTest=False,img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True):
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
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir, valid_iou=valid_iou)
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
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
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
                                                    padding=np.array((self.get_overlap()/2, self.get_overlap()/2)),
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
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0]/iou[1], duration))
            mid_iou_record.append(iou)
            if count % 5 == 0:
                mean_iou = np.sum(np.array(mid_iou_record)[:, 0]) / np.sum(np.array(mid_iou_record)[:, 1])
                print(str(count)+':  ' +   'Overall mean IoU={:.3f}'.format(mean_iou))
                mid_iou_record = []

            # save results
            if save_result:
                pred_save_dir = os.path.join(score_save_dir, 'pred')
                if not os.path.exists(pred_save_dir):
                    os.makedirs(pred_save_dir)
                imageio.imsave(os.path.join(pred_save_dir, tile_name+'.png'), pred.astype(np.uint8))
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
                plt.suptitle('{} Results on {} IoU={:3f}'.format(self.model_name, file_name_truth.split('_')[0], iou[0]/iou[1]))
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


class UnetModelCrop_MTratio(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, source_num=2, source_control=[4, 1]): # Now we only support 2 source
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetCrop'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None
        self.source_num = source_num
        self.source_control = source_control

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

        if continue_dir is not None and os.path.exists(continue_dir):
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
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def make_loss(self, y_name, loss_type='BGC', lamb_BGC= 0.2, **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w - self.get_overlap(),
                                                       h - self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
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
            elif loss_type == 'regularization':
                l2_loss = tf.losses.get_regularization_loss()
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)) + 0.1*l2_loss
            elif loss_type == 'BGC':

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


                self.loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_real, labels=gt_real))
                self.loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_fake, labels=gt_fake))

                self.loss = self.loss_real + lamb_BGC * self.loss_fake
                # self.loss = self.loss_real + lamb_BGC * self.loss_fake # does this one affect the learning rate ?

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
                pos_p_sub = array_ops.where(gt > sigmoid_p, gt - sigmoid_p, zeros)
                neg_p_sub = array_ops.where(gt > zeros, zeros, sigmoid_p)
                per_entry_cross_ent = - kwargs['alpha'] * (pos_p_sub ** kwargs['gamma']) * tf.log(tf.clip_by_value(
                    sigmoid_p, 1e-8, 1.0)) - (1- kwargs['alpha']) * (neg_p_sub ** kwargs['gamma']) * tf.log(
                    tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
                self.loss = tf.reduce_sum(per_entry_cross_ent)


class UnetModelPredict(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetPredict'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                           padding='valid', dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                           padding='valid', dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                           padding='valid', dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                           padding='valid', dropout=self.dropout_rate)
        self.encoding = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        # upsample
        up6 = self.crop_upsample_concat(self.encoding, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)
        output_flat = tf.reshape(self.pred[:, :, :, 1], [self.bs, 388 * 388])
        self.building_pred = tf.layers.dense(output_flat, 1, name='building_pred', activation=None)

    def make_loss(self, y_name, y_name_2, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

            self.building_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs[y_name_2],
                                                                                        logits=self.building_pred))

    def make_update_ops(self, x_name, y_name, y_name_2):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('inputs', self.inputs[y_name_2])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_learning_rate(self, n_train):
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                        tf.cast(n_train/self.bs * self.ds, tf.int32),
                                                        self.dr, staircase=True)

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                 seg_optm = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                global_step=self.global_step)
                 clf_optm = tf.train.AdamOptimizer(self.learning_rate * 0.1).minimize(self.building_loss,
                                                                                      global_step=None)
                 self.optimizer = [seg_optm, clf_optm]

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('Classify Loss', self.building_loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train_config(self, x_name, y_name, y_name_2, n_train, n_valid, patch_size, ckdir, loss_type='xent',
                     train_var_filter=None, hist=False, par_dir=None, **kwargs):
        self.make_loss(y_name, y_name_2, loss_type, **kwargs)
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name, y_name_2)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size, par_dir)
        self.make_summary(hist)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.n_train = n_train
        self.n_valid = n_valid

    def train(self, x_name, y_name, y_name_2, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_building=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
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
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                X_batch, _, building_truth = train_reader_building.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[1], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch,
                                                                self.inputs[y_name_2]: building_truth,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = sess.run([self.pred, self.loss, self.summary],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.inputs[y_name_2]: building_truth,
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
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def run(self, train_reader=None, train_reader_building=None, valid_reader=None, test_reader=None,
            pretrained_model_dir=None, layers2load=None, isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32),
            verb_step=100, save_epoch=5, gpu=None, tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1,
            continue_dir=None, load_epoch_num=None, valid_iou=False, best_model=True):
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
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', 'Y2', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               train_reader_building=train_reader_building,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val


class UnetModelGAN(UnetModelCrop):
    @staticmethod
    def make_list(x):
        if type(x) is int or type(x) is float:
            x = [x for _ in range(3)]
        elif type(x) is list:
            pass
        else:
            # try take it as a string
            x = [float(a) for a in str(x).split(',')]
        return x

    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'UnetGAN'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.fake_logit = None
        self.true_logit = None
        self.fake_logit_ = None
        self.d_loss = None
        self.g_loss = None

    def load_weights(self, ckpt_dir, layers2load, load_final_layer=False):
        # this is different from network.load()
        # this function only loads specified layers
        layers_list = []
        if isinstance(layers2load, str):
            layers2load = [int(a) for a in layers2load.split(',')]
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        if load_final_layer:
            layers_list.append('final')

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        try:
            latest_check_point = tf.train.latest_checkpoint(ckpt_dir)
            tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)
            print('loaded {}'.format(latest_check_point))
        except tf.errors.NotFoundError:
            with open(os.path.join(ckpt_dir, 'checkpoint'), 'r') as f:
                ckpts = f.readlines()
            ckpt_file_name = ckpts[0].split('/')[-1].strip().strip('\"')
            latest_check_point = os.path.join(ckpt_dir, ckpt_file_name)
            tf.contrib.framework.init_from_checkpoint(latest_check_point, load_dict)
            print('loaded {}'.format(latest_check_point))

    def make_encoder(self, x_name):
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                           padding='valid', dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn * 2, sfn * 2], self.trainable, name='conv2',
                                           padding='valid', dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn * 4, sfn * 4], self.trainable, name='conv3',
                                           padding='valid', dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn * 8, sfn * 8], self.trainable, name='conv4',
                                           padding='valid', dropout=self.dropout_rate)
        pool5 = self.conv_conv_pool(pool4, [sfn * 16, sfn * 16], self.trainable, name='conv5', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        # upsample
        up6 = self.crop_upsample_concat(pool5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn * 8, sfn * 8], self.trainable, name='up6', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn * 4, sfn * 4], self.trainable, name='up7', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn * 2, sfn * 2], self.trainable, name='up8', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        return conv9

    def make_discriminator(self, y, sfn=2, reuse=False):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1, pool1 = self.conv_conv_pool(y, [sfn, sfn], self.trainable, name='conv1',
                                               padding='valid', dropout=self.dropout_rate)
            conv2, pool2 = self.conv_conv_pool(pool1, [sfn * 2, sfn * 2], self.trainable, name='conv2',
                                               padding='valid', dropout=self.dropout_rate)
            conv3, pool3 = self.conv_conv_pool(pool2, [sfn * 4, sfn * 4], self.trainable, name='conv3',
                                               padding='valid', dropout=self.dropout_rate)
            conv4, pool4 = self.conv_conv_pool(pool3, [sfn * 8, sfn * 8], self.trainable, name='conv4',
                                               padding='valid', dropout=self.dropout_rate)
            conv5, pool5 = self.conv_conv_pool(pool4, [sfn * 16, sfn * 16], self.trainable, name='conv5',
                                               padding='valid', dropout=self.dropout_rate)
            conv6, pool6 = self.conv_conv_pool(pool5, [sfn * 16, sfn * 16], self.trainable, name='conv6',
                                               padding='valid', dropout=self.dropout_rate)
            flat = tf.reshape(pool6, shape=[self.bs, 2 * 2 * sfn * 16])
            return self.fc_fc(flat, [500, 100, 1], self.trainable, name='fc_final', activation=None, dropout=False)

    def create_graph(self, names, class_num, start_filter_num=32):
        self.class_num = class_num

        with tf.variable_scope('Encoder'):
            conv9 = self.make_encoder(names[0])

            self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
            self.output = tf.nn.softmax(self.pred)

        # self.hard_label = tf.cast(tf.expand_dims(tf.argmax(self.output, axis=-1, name='hard_label'), axis=-1), tf.float32)
        with tf.variable_scope('Discriminator'):
            # self.fake_logit = self.make_discriminator(self.hard_label, sfn=start_filter_num//4, reuse=False)
            _, w, h, _ = self.inputs[names[1]].get_shape().as_list()
            true_y = tf.cast(tf.image.resize_image_with_crop_or_pad(self.inputs[names[1]], w - self.get_overlap(),
                                                                    h - self.get_overlap()), tf.float32)
            self.true_logit = self.make_discriminator(true_y, sfn=start_filter_num//4, reuse=False)

            self.fake_logit_ = self.make_discriminator(tf.expand_dims(self.output[:, :, :, 1], axis=-1),
                                                       sfn=start_filter_num//4, reuse=True)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

        with tf.variable_scope('adv_loss'):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logit,
                                                        labels=tf.ones([self.bs, 1])))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                        labels=tf.zeros([self.bs, 1])))
            self.d_loss = 0.5 * d_loss_real + 0.5 * d_loss_fake
            self.g_loss = -0.5 * tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                        labels=tf.zeros([self.bs, 1])))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_learning_rate(self, n_train):
        self.learning_rate = []
        for i in range(3):
            self.learning_rate.append(tf.train.exponential_decay(self.lr[i], self.global_step,
                                                                 tf.cast(n_train/self.bs * self.ds[i], tf.int32),
                                                                 self.dr[i], staircase=True))

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            t_vars = tf.trainable_variables()
            e_vars = [var for var in t_vars if 'Encoder' in var.name]
            d_vars = [var for var in t_vars if 'Discriminator' in var.name]
            if train_var_filter is None:
                seg_optm = tf.train.AdamOptimizer(self.learning_rate[0]).\
                    minimize(self.loss, var_list=e_vars, global_step=self.global_step)
                g_optm = tf.train.AdamOptimizer(self.learning_rate[1]).\
                    minimize(self.g_loss, var_list=e_vars, global_step=None)
                d_optm = tf.train.AdamOptimizer(self.learning_rate[2]). \
                    minimize(self.d_loss, var_list=d_vars, global_step=None)
                self.optimizer = [seg_optm, g_optm, d_optm]

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('Generator Loss', self.g_loss)
        tf.summary.scalar('Discriminator Loss', self.d_loss)
        tf.summary.scalar('learning rate seg', self.learning_rate[0])
        tf.summary.scalar('learning rate g', self.learning_rate[1])
        tf.summary.scalar('learning rate d', self.learning_rate[2])
        self.summary = tf.summary.merge_all()

    def make_ckdir(self, ckdir, patch_size, par_dir=None):
        if type(patch_size) is list:
            patch_size = patch_size[0]
        # make unique directory for save
        dir_name = '{}_PS{}_BS{}_EP{}_LR{}_DS{}_DR{}'.\
            format(self.model_name, patch_size, self.bs, self.epochs,
                   '_'.join([str(a) for a in self.lr]),
                   '_'.join([str(a) for a in self.ds]),
                   '_'.join([str(a) for a in self.dr]))
        if par_dir is None:
            self.ckdir = os.path.join(ckdir, dir_name)
        else:
            self.ckdir = os.path.join(ckdir, par_dir, dir_name)

    def train_config(self, x_name, y_name, n_train, n_valid, patch_size, ckdir, loss_type='xent',
                     train_var_filter=None, hist=False, par_dir=None, **kwargs):
        self.make_loss(y_name, loss_type, **kwargs)
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size, par_dir)
        self.make_summary(hist)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.n_train = n_train
        self.n_valid = n_valid

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_d_loss_summary_op = tf.summary.scalar('d_loss_validation', self.valid_d_loss)
        valid_g_loss_summary_op = tf.summary.scalar('g_loss_validation', self.valid_g_loss)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
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
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                X_batch, _ = train_reader_target.readerAction(sess)
                _, y_batch = train_reader_source.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[1], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch,
                                                                self.inputs[y_name]: y_batch,
                                                                self.trainable: True})

                X_batch, _ = train_reader_target.readerAction(sess)
                _, y_batch = train_reader_source.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[2], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch,
                                                                self.inputs[y_name]: y_batch,
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
            d_loss_valid_mean = []
            g_loss_valid_mean = []
            iou_valid_mean = np.zeros(2)
            X_batch_val, y_batch_val, pred_valid = None, None, None
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run([self.pred, self.loss, self.loss_iou],
                                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                                 self.inputs[y_name]: y_batch_val,
                                                                                 self.trainable: False})
                _, y_batch_val_target = train_reader_source.readerAction(sess)
                d_loss_valid, g_loss_valid = sess.run([self.d_loss, self.g_loss],
                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                 self.inputs[y_name]: y_batch_val_target,
                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                d_loss_valid_mean.append(d_loss_valid)
                g_loss_valid_mean.append(g_loss_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            d_loss_valid_mean = np.mean(d_loss_valid_mean)
            g_loss_valid_mean = np.mean(g_loss_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Val xent: {:.3f}, g_loss: {:.3f}, d_loss: {:.3f}, duration: {:.3f}'.
                      format(cross_entropy_valid_mean, d_loss_valid_mean, g_loss_valid_mean, duration))
            valid_summaries = sess.run([valid_cross_entropy_summary_op, valid_iou_summary_op,
                                        valid_d_loss_summary_op, valid_g_loss_summary_op],
                                       feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean,
                                                  self.valid_iou: iou_valid_mean,
                                                  self.valid_d_loss: d_loss_valid_mean,
                                                  self.valid_g_loss: g_loss_valid_mean})
            for i in range(4):
                summary_writer.add_summary(valid_summaries[i], self.global_step_value)
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
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def run(self, train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
            test_reader=None, pretrained_model_dir=None, layers2load=None, isTrain=False,
            img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True):
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
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               train_reader_source=train_reader_source, train_reader_target=train_reader_target,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val


class UnetModelGAN_V2(UnetModelGAN):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'UnetGAN_V2'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.fake_logit = None
        self.true_logit = None
        self.fake_logit_ = None
        self.d_loss = None
        self.g_loss = None

    def make_discriminator(self, y, sfn=4, reuse=False):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1 = self.conv_conv_pool(y, [sfn], self.trainable, name='conv1', kernal_size=(5, 5), conv_stride=(2, 2),
                                        padding='valid', dropout=self.dropout_rate, pool=False)
            conv2 = self.conv_conv_pool(conv1, [sfn], self.trainable, name='conv2', kernal_size=(5, 5),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv3 = self.conv_conv_pool(conv2, [sfn * 2], self.trainable, name='conv3', kernal_size=(5, 5),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv4 = self.conv_conv_pool(conv3, [sfn * 2], self.trainable, name='conv4', kernal_size=(3, 3),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv5 = self.conv_conv_pool(conv4, [sfn * 4], self.trainable, name='conv5', kernal_size=(3, 3),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            conv6 = self.conv_conv_pool(conv5, [sfn * 4], self.trainable, name='conv6', kernal_size=(3, 3),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False)
            flat = tf.reshape(conv6, shape=[self.bs, 4 * 4 * sfn * 4])
            return self.fc_fc(flat, [100, 1], self.trainable, name='fc_final', activation=None, dropout=False)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

        with tf.variable_scope('adv_loss'):
            d_loss_fake_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit_,
                                                                                   labels=tf.zeros([self.bs, 1])))
            d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit_,
                                                                                   labels=tf.ones([self.bs, 1])))
            d_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            self.g_loss = d_loss_fake_1
            self.d_loss = d_loss_fake_0 + d_loss_real_1

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                t_vars = tf.trainable_variables()
                e_vars = [var for var in t_vars if 'Encoder' in var.name]
                d_vars = [var for var in t_vars if 'Discriminator' in var.name]
                seg_optm = tf.train.AdamOptimizer(self.learning_rate[0], name='Adam_Seg').\
                    minimize(self.loss, var_list=e_vars, global_step=self.global_step)
                g_optm = tf.train.AdamOptimizer(self.learning_rate[1], name='Adam_g').\
                    minimize(self.g_loss, var_list=e_vars, global_step=None)
                d_optm = tf.train.AdamOptimizer(self.learning_rate[2], name='Adam_d').\
                    minimize(self.d_loss, var_list=d_vars, global_step=None)
                self.optimizer = [seg_optm, g_optm, d_optm]

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('Generator Loss', self.g_loss)
        tf.summary.scalar('Discriminator Loss', self.d_loss)
        tf.summary.scalar('learning rate g', self.learning_rate[0])
        tf.summary.scalar('learning rate d', self.learning_rate[1])
        self.summary = tf.summary.merge_all()

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_d_loss_summary_op = tf.summary.scalar('d_loss_validation', self.valid_d_loss)
        valid_g_loss_summary_op = tf.summary.scalar('g_loss_validation', self.valid_g_loss)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
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
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                X_batch, _ = train_reader_target.readerAction(sess)
                _, y_batch = train_reader_source.readerAction(sess)
                sess.run([self.optimizer[2]], feed_dict={self.inputs[x_name]: X_batch,
                                                         self.inputs[y_name]: y_batch,
                                                         self.trainable: True})

                X_batch, _ = train_reader_target.readerAction(sess)
                sess.run([self.optimizer[1]], feed_dict={self.inputs[x_name]: X_batch,
                                                         self.inputs[y_name]: y_batch,
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
            d_loss_valid_mean = []
            g_loss_valid_mean = []
            iou_valid_mean = np.zeros(2)
            X_batch_val, y_batch_val, pred_valid = None, None, None
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run([self.pred, self.loss, self.loss_iou],
                                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                                 self.inputs[y_name]: y_batch_val,
                                                                                 self.trainable: False})
                _, y_batch_val_target = train_reader_source.readerAction(sess)
                d_loss_valid, g_loss_valid = sess.run([self.d_loss, self.g_loss],
                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                 self.inputs[y_name]: y_batch_val_target,
                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                d_loss_valid_mean.append(d_loss_valid)
                g_loss_valid_mean.append(g_loss_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            d_loss_valid_mean = np.mean(d_loss_valid_mean)
            g_loss_valid_mean = np.mean(g_loss_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Val xent: {:.3f}, g_loss: {:.3f}, d_loss: {:.3f}, duration: {:.3f}'.
                      format(cross_entropy_valid_mean, d_loss_valid_mean, g_loss_valid_mean, duration))
            valid_summaries = sess.run([valid_cross_entropy_summary_op, valid_iou_summary_op,
                                        valid_d_loss_summary_op, valid_g_loss_summary_op],
                                       feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean,
                                                  self.valid_iou: iou_valid_mean,
                                                  self.valid_d_loss: d_loss_valid_mean,
                                                  self.valid_g_loss: g_loss_valid_mean})
            for i in range(4):
                summary_writer.add_summary(valid_summaries[i], self.global_step_value)
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
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)


class UnetModelGAN_V3(UnetModelGAN_V2):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, pad=40):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'UnetGAN_V3'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.pad = pad
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 4, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.refine = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def make_encoder(self, x_name):
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], False, name='conv1',
                                           padding='valid', dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn * 2, sfn * 2], False, name='conv2',
                                           padding='valid', dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn * 4, sfn * 4], False, name='conv3',
                                           padding='valid', dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn * 8, sfn * 8], False, name='conv4',
                                           padding='valid', dropout=self.dropout_rate)
        pool5 = self.conv_conv_pool(pool4, [sfn * 16, sfn * 16], False, name='conv5', pool=False,
                                    padding='valid', dropout=self.dropout_rate)

        # upsample
        up6 = self.crop_upsample_concat(pool5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn * 8, sfn * 8], False, name='up6', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn * 4, sfn * 4], False, name='up7', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn * 2, sfn * 2], False, name='up8', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], False, name='up9', pool=False,
                                    padding='valid', dropout=self.dropout_rate)
        return conv9

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        tf.add_to_collection('outputs', self.refine)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    @ staticmethod
    def res_block(input_, n_filter, name):
        res = input_
        conv = input_
        with tf.variable_scope('res{}'.format(name)):
            for i in range(2):
                conv = tf.layers.conv2d(conv, n_filter, (3, 3), name='res{}_conv_{}'.format(name, i))
                conv = tf.layers.batch_normalization(conv, name='res{}_batchnorm_{}'.format(name, i))
                conv = tf.nn.relu(conv, name='res{}_relu_{}'.format(name, i))
            res = res[:, 2:-2, 2:-2, :]
            conv = tf.add(res, conv, name='res{}_add'.format(name))
        return conv

    @staticmethod
    def trans_2d_block(input_, n_filter, name):
        with tf.variable_scope('trans_{}'.format(name)):
            input_ = tf.layers.conv2d_transpose(input_, n_filter, (3, 3), strides=(2, 2), padding='SAME',
                                                name='trans_{}_conv'.format(name))
            input_ = tf.layers.batch_normalization(input_, name='trans_{}_batchnorm'.format(name))
            input_ = tf.nn.relu(input_, name='trans_{}_relu'.format(name))
        return input_

    def create_graph(self, names, class_num, start_filter_num=32):
        self.class_num = class_num

        conv9 = self.make_encoder(names[0])
        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

        self.output = tf.nn.softmax(self.pred)
        self.hard_label = tf.cast(tf.expand_dims(tf.argmax(self.output, axis=-1, name='hard_label'), axis=-1),
                                  tf.float32)
        tf.stop_gradient(self.hard_label)

        with tf.variable_scope('Attn'):
            self.refine = self.make_attn(self.hard_label)

        with tf.variable_scope('Discriminator'):
            _, w, h, _ = self.inputs[names[1]].get_shape().as_list()
            true_y = tf.cast(tf.image.resize_image_with_crop_or_pad(self.inputs[names[1]], w - self.get_overlap(),
                                                                    h - self.get_overlap()), tf.float32)
            self.true_logit = self.make_discriminator(true_y, sfn=start_filter_num//4, reuse=False)

            self.fake_logit = self.make_discriminator(self.refine, sfn=start_filter_num//4, reuse=True)

    def make_attn(self, pred):
        orig = pred
        padding = tf.constant([[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        pred = tf.pad(pred, padding, 'REFLECT', name='reflect_pad')
        pred = self.conv_conv_pool(pred, [self.sfn], self.trainable, 'conv_block_1', (9, 9), (1, 1), pool=False,
                                   activation=tf.nn.relu)
        pred = self.conv_conv_pool(pred, [self.sfn * 2], self.trainable, 'conv_block_2', (3, 3), (2, 2), pool=False,
                                   activation=tf.nn.relu)
        pred = self.conv_conv_pool(pred, [self.sfn * 4], self.trainable, 'conv_block_3', (3, 3), (2, 2), pool=False,
                                   activation=tf.nn.relu)
        for i in range(5):
            pred = self.res_block(pred, self.sfn * 4, str(i))

        pred = self.trans_2d_block(pred, self.sfn * 2, '1')
        pred = self.trans_2d_block(pred, self.sfn, '2')
        pred = self.conv_conv_pool(tf.concat([pred, orig], axis=-1), [1], self.trainable, '3', (9, 9), pool=False,
                                   activation=tf.nn.sigmoid)
        return pred

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.refine, [-1, ])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])

            pred = tf.to_int32(pred_flat > 0.5)
            intersect = tf.cast(tf.reduce_sum(y_flat * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(y_flat), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(y_flat * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_flat,
                                                                               labels=tf.to_float(y_flat)))

        with tf.variable_scope('adv_loss'):
            d_loss_fake_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.zeros([self.bs, 1])))
            d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            d_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            self.g_loss = d_loss_fake_1
            self.d_loss = d_loss_fake_0 + d_loss_real_1

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                t_vars = tf.trainable_variables()
                e_vars = [var for var in t_vars if 'Attn' in var.name]
                d_vars = [var for var in t_vars if 'Discriminator' in var.name]
                seg_optm = tf.train.AdamOptimizer(self.learning_rate[0], name='Adam_Seg').\
                    minimize(self.loss, var_list=e_vars, global_step=self.global_step)
                g_optm = tf.train.AdamOptimizer(self.learning_rate[1], name='Adam_g').\
                    minimize(self.g_loss, var_list=e_vars, global_step=None)
                d_optm = tf.train.AdamOptimizer(self.learning_rate[2], name='Adam_d').\
                    minimize(self.d_loss, var_list=d_vars, global_step=None)
                self.optimizer = [seg_optm, g_optm, d_optm]

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_d_loss_summary_op = tf.summary.scalar('d_loss_validation', self.valid_d_loss)
        valid_g_loss_summary_op = tf.summary.scalar('g_loss_validation', self.valid_g_loss)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
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
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                X_batch, _ = train_reader_target.readerAction(sess)
                _, y_batch = train_reader_source.readerAction(sess)
                sess.run([self.optimizer[2]], feed_dict={self.inputs[x_name]: X_batch,
                                                         self.inputs[y_name]: y_batch,
                                                         self.trainable: True})

                X_batch, _ = train_reader_target.readerAction(sess)
                sess.run([self.optimizer[1]], feed_dict={self.inputs[x_name]: X_batch,
                                                         self.inputs[y_name]: y_batch,
                                                         self.trainable: True})

                if self.global_step_value % verb_step == 0:
                    step_cross_entropy, step_summary = sess.run([self.loss, self.summary],
                                                                feed_dict={self.inputs[x_name]: X_batch,
                                                                           self.inputs[y_name]: y_batch,
                                                                           self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            cross_entropy_valid_mean = []
            d_loss_valid_mean = []
            g_loss_valid_mean = []
            iou_valid_mean = np.zeros(2)
            X_batch_val, y_batch_val, pred_valid, refine_valid = None, None, None, None
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, refine_valid, cross_entropy_valid, iou_valid = sess.run(
                    [self.pred, self.refine, self.loss, self.loss_iou], feed_dict={self.inputs[x_name]: X_batch_val,
                                                                                   self.inputs[y_name]: y_batch_val,
                                                                                   self.trainable: False})
                _, y_batch_val_target = train_reader_source.readerAction(sess)
                d_loss_valid, g_loss_valid = sess.run([self.d_loss, self.g_loss],
                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                 self.inputs[y_name]: y_batch_val_target,
                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                d_loss_valid_mean.append(d_loss_valid)
                g_loss_valid_mean.append(g_loss_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            d_loss_valid_mean = np.mean(d_loss_valid_mean)
            g_loss_valid_mean = np.mean(g_loss_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Val xent: {:.3f}, g_loss: {:.3f}, d_loss: {:.3f}, duration: {:.3f}'.
                      format(cross_entropy_valid_mean, d_loss_valid_mean, g_loss_valid_mean, duration))
            valid_summaries = sess.run([valid_cross_entropy_summary_op, valid_iou_summary_op,
                                        valid_d_loss_summary_op, valid_g_loss_summary_op],
                                       feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean,
                                                  self.valid_iou: iou_valid_mean,
                                                  self.valid_d_loss: d_loss_valid_mean,
                                                  self.valid_g_loss: g_loss_valid_mean})
            for i in range(4):
                summary_writer.add_summary(valid_summaries[i], self.global_step_value)
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
                valid_image_summary = sess.run(
                    valid_image_summary_op, feed_dict={
                        self.valid_images: image_summary(X_batch_val[:, 92:-92, 92:-92, :3],
                                                         y_batch_val[:, 92:-92, 92:-92, :],
                                                         pred_valid, refine_valid, img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)

    def run(self, train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
            test_reader=None, pretrained_model_dir=None, layers2load=None, isTrain=False,
            img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True):
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
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               train_reader_source=train_reader_source, train_reader_target=train_reader_target,
                               image_summary=self.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val

    def test(self, x_name, sess, test_iterator):
        result = []
        for X_batch in test_iterator:
            pred = sess.run(self.refine, feed_dict={self.inputs[x_name]: X_batch,
                                                    self.trainable: False})
            result.append(np.concatenate([1-pred, pred], axis=-1))
        result = np.vstack(result)
        return result

    @staticmethod
    def image_summary(image, truth, prediction, refine, img_mean=np.array((0, 0, 0), dtype=np.float32)):
        truth_img = util_functions.decode_labels(truth)

        prediction = util_functions.pad_prediction(image, prediction)
        pred_labels = util_functions.get_pred_labels(prediction)
        pred_img = util_functions.decode_labels(pred_labels)

        refine_img = util_functions.decode_labels(np.rint(refine))
        return np.concatenate([image + img_mean, truth_img, pred_img, refine_img], axis=2)


class UnetModelGAN_V3Shrink(UnetModelGAN_V3):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, pad=24):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'UnetGAN_V3Shrink'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.pad = pad
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 4, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.refine = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def make_attn(self, pred):
        orig = pred
        padding = tf.constant([[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        pred = tf.pad(pred, padding, 'REFLECT', name='reflect_pad')
        pred = self.conv_conv_pool(pred, [self.sfn // 2], self.trainable, 'conv_block_1', (9, 9), (1, 1), pool=False,
                                   activation=tf.nn.relu)
        pred = self.conv_conv_pool(pred, [self.sfn], self.trainable, 'conv_block_2', (3, 3), (2, 2), pool=False,
                                   activation=tf.nn.relu)
        pred = self.conv_conv_pool(pred, [self.sfn * 2], self.trainable, 'conv_block_3', (3, 3), (2, 2), pool=False,
                                   activation=tf.nn.relu)
        for i in range(3):
            pred = self.res_block(pred, self.sfn * 2, str(i))

        pred = self.trans_2d_block(pred, self.sfn, '1')
        pred = self.trans_2d_block(pred, self.sfn // 2, '2')
        pred = self.conv_conv_pool(tf.concat([pred, orig], axis=-1), [1], self.trainable, '3', (9, 9), pool=False,
                                   activation=tf.nn.sigmoid)
        return pred


class UnetModelGAN_V3ShrinkRGB(UnetModelGAN_V3Shrink):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, pad=24):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'UnetGAN_V3ShrinkRGB'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.pad = pad
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 4, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.refine = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def make_attn(self, rgb, pred):
        orig = pred
        rgb = rgb[:, 92:-92, 92:-92, :]
        padding = tf.constant([[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        channel = []
        for i in range(3):
            channel.append(rgb[:, :, :, 0] * pred[:, :, :, 0])
        pred = tf.stack(channel, axis=-1)
        pred = tf.pad(pred, padding, 'REFLECT', name='reflect_pad')
        pred = self.conv_conv_pool(pred, [self.sfn // 2], self.trainable, 'conv_block_1', (9, 9), (1, 1), pool=False,
                                   activation=tf.nn.relu)
        pred = self.conv_conv_pool(pred, [self.sfn], self.trainable, 'conv_block_2', (3, 3), (2, 2), pool=False,
                                   activation=tf.nn.relu)
        pred = self.conv_conv_pool(pred, [self.sfn * 2], self.trainable, 'conv_block_3', (3, 3), (2, 2), pool=False,
                                   activation=tf.nn.relu)
        for i in range(3):
            pred = self.res_block(pred, self.sfn * 2, str(i))

        pred = self.trans_2d_block(pred, self.sfn, '1')
        pred = self.trans_2d_block(pred, self.sfn // 2, '2')
        pred = self.conv_conv_pool(tf.concat([pred, orig], axis=-1), [1], self.trainable, '3', (9, 9), pool=False,
                                   activation=tf.nn.sigmoid)
        return pred

    def create_graph(self, names, class_num, start_filter_num=32):
        self.class_num = class_num

        conv9 = self.make_encoder(names[0])
        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

        self.output = tf.nn.softmax(self.pred)
        self.hard_label = tf.cast(tf.expand_dims(tf.argmax(self.output, axis=-1, name='hard_label'), axis=-1),
                                  tf.float32)
        tf.stop_gradient(self.hard_label)

        with tf.variable_scope('Attn'):
            self.refine = self.make_attn(self.inputs[names[0]], self.hard_label)

        with tf.variable_scope('Discriminator'):
            _, w, h, _ = self.inputs[names[1]].get_shape().as_list()
            true_y = tf.cast(tf.image.resize_image_with_crop_or_pad(self.inputs[names[1]], w - self.get_overlap(),
                                                                    h - self.get_overlap()), tf.float32)
            self.true_logit = self.make_discriminator(true_y, sfn=start_filter_num//4, reuse=False)

            self.fake_logit = self.make_discriminator(self.refine, sfn=start_filter_num//4, reuse=True)


class UnetModelGAN_V4RGB(UnetModelGAN_V3ShrinkRGB):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, pad=24):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'UnetGAN_V4'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.pad = pad
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 4, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.refine = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def make_discriminator(self, y, sfn=4, reuse=False):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1, pool1 = self.conv_conv_pool(y, [sfn], self.trainable, name='conv1', kernal_size=(3, 3),
                                               conv_stride=(2, 2), padding='valid', pool_size=(2, 2),
                                               dropout=self.dropout_rate, pool=True)
            conv2, pool2 = self.conv_conv_pool(pool1, [sfn], self.trainable, name='conv2', kernal_size=(3, 3),
                                               conv_stride=(2, 2), padding='valid', pool_size=(2, 2),
                                               dropout=self.dropout_rate, pool=True)
            conv3, pool3 = self.conv_conv_pool(pool2, [sfn * 2], self.trainable, name='conv3', kernal_size=(3, 3),
                                               pool_size=(2, 2), conv_stride=(1, 1), padding='valid',
                                               dropout=self.dropout_rate, pool=True)
            conv4, pool4 = self.conv_conv_pool(pool3, [sfn * 2], self.trainable, name='conv4', kernal_size=(3, 3),
                                               pool_size=(2, 2), conv_stride=(1, 1), padding='valid',
                                               dropout=self.dropout_rate, pool=True)
            flat = tf.reshape(pool4, shape=[self.bs, 4 * 4 * sfn * 2])
            return self.fc_fc(flat, [100, 1], self.trainable, name='fc_final', activation=None, dropout=False)


class UnetModelGAN_V5RGB(UnetModelGAN_V4RGB):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, pad=24, train_unet=False):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'UnetGAN_V5'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.train_unet = train_unet
        self.pad = pad
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 4, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.refine = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                t_vars = tf.trainable_variables()
                e_vars = [var for var in t_vars if 'Attn' in var.name]
                d_vars = [var for var in t_vars if 'Discriminator' in var.name]
                if self.train_unet:
                    seg_vars = [var for var in t_vars if 'Discriminator' not in var.name]
                else:
                    seg_vars = e_vars
                seg_optm = tf.train.AdamOptimizer(self.learning_rate[0], name='Adam_Seg').\
                    minimize(self.loss, var_list=seg_vars, global_step=self.global_step)
                g_optm = tf.train.AdamOptimizer(self.learning_rate[1], name='Adam_g').\
                    minimize(self.g_loss, var_list=e_vars, global_step=None)
                d_optm = tf.train.AdamOptimizer(self.learning_rate[2], name='Adam_d').\
                    minimize(self.d_loss, var_list=d_vars, global_step=None)
                self.optimizer = [seg_optm, g_optm, d_optm]

    def make_discriminator(self, y, sfn=4, reuse=False):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1 = self.conv_conv_pool(y, [sfn // 2], self.trainable, name='conv1', kernal_size=(5, 5), conv_stride=(2, 2),
                                        padding='valid', dropout=self.dropout_rate, pool=False, bn=True,
                                        activation=tf.nn.leaky_relu)
            conv2 = self.conv_conv_pool(conv1, [sfn], self.trainable, name='conv2', kernal_size=(5, 5),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False,
                                        bn=True, activation=tf.nn.leaky_relu)
            conv3 = self.conv_conv_pool(conv2, [sfn * 2], self.trainable, name='conv3', kernal_size=(5, 5),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False,
                                        bn=True, activation=tf.nn.leaky_relu)
            conv4 = self.conv_conv_pool(conv3, [sfn * 4], self.trainable, name='conv4', kernal_size=(5, 5),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False,
                                        bn=True, activation=tf.nn.leaky_relu)
            conv5 = self.conv_conv_pool(conv4, [sfn * 8], self.trainable, name='conv5', kernal_size=(5, 5),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False,
                                        bn=True, activation=tf.nn.leaky_relu)
            conv6 = self.conv_conv_pool(conv5, [sfn * 16], self.trainable, name='conv6', kernal_size=(5, 5),
                                        conv_stride=(2, 2), padding='valid', dropout=self.dropout_rate, pool=False,
                                        bn=True, activation=tf.nn.leaky_relu)
            flat = tf.reshape(conv6, shape=[self.bs, 3 * 3 * sfn * 16])
            return self.fc_fc(flat, [128, 1], self.trainable, name='fc_final', activation=None, dropout=False)


class SSAN(UnetModelGAN_V4RGB):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, lada=2):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.lr = self.make_list(learn_rate)
        self.ds = self.make_list(decay_step)
        self.dr = self.make_list(decay_rate)
        self.name = 'SSAN'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.lada = lada
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_d_loss = tf.placeholder(tf.float32, [])
        self.valid_g_loss = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(
            tf.uint8, shape=[None, input_size[0] - self.get_overlap(), (input_size[1] - self.get_overlap()) * 3, 3],
            name='validation_images')
        self.update_ops = None
        self.config = None
        self.hard_label = None
        self.fake_logit = None
        self.true_logit = None
        self.d_loss = None
        self.g_loss = None

    def get_overlap(self):
        return 0

    def make_encoder(self, x_name):
        C = 8
        dilation_rate = [1, 1, 2, 4, 8, 16, 1]
        filters = [2*C, 2*C, 4*C, 8*C, 16*C, 32*C, 32*C]
        input_ = self.inputs[x_name]
        with tf.variable_scope('encoder'):
            for i in range(7):
                input_ = tf.layers.conv2d(input_, filters[i], kernel_size=(3, 3), padding='SAME',
                                          dilation_rate=(dilation_rate[i], dilation_rate[i]), name='conv{}'.format(i))
                input_ = tf.layers.batch_normalization(input_, training=self.trainable, name='bn_{}'.format(i))
                input_ = tf.nn.relu(input_, name='relu_{}'.format(i))
        return tf.layers.conv2d(input_, self.class_num, kernel_size=(1, 1), padding='SAME', name='conv8')

    def make_discriminator(self, y, sfn=4, reuse=False):
        with tf.variable_scope('discr', reuse=reuse):
            # downsample
            conv1, pool1 = self.conv_conv_pool(y, [96, 128, 128], self.trainable, name='conv1', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv2, pool2 = self.conv_conv_pool(pool1, [128, 128], self.trainable, name='conv2', kernal_size=(3, 3),
                                               conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate,
                                               pool=True, bn=False, activation=tf.nn.relu)
            conv3 = self.conv_conv_pool(pool2, [256], self.trainable, name='conv3', kernal_size=(3, 3),
                                        conv_stride=(1, 1), padding='valid', dropout=self.dropout_rate, pool=False,
                                        bn=True, activation=tf.nn.relu)
            conv4 = tf.layers.conv2d(conv3, 2, kernel_size=(3, 3), name='layerconv4')
            flat = tf.reshape(conv4, shape=[self.bs, 48*48*2])
            return self.fc_fc(flat, [1], self.trainable, name='fc_final', activation=None, dropout=False)

    def create_graph(self, names, class_num, start_filter_num=32):
        self.class_num = class_num

        conv9 = self.make_encoder(names[0])
        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

        with tf.variable_scope('Discriminator'):
            true_y = tf.cast(self.inputs[names[1]], tf.float32)
            orig_rgb = self.inputs[names[0]] * true_y
            pred_rgb = self.inputs[names[0]] * tf.expand_dims(self.output[:, :, :, 1], axis=-1)
            self.true_logit = self.make_discriminator(orig_rgb, sfn=start_filter_num//4, reuse=False)
            self.fake_logit = self.make_discriminator(pred_rgb, sfn=start_filter_num//4, reuse=True)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)

            pred = tf.argmax(prediction, axis=-1, output_type=tf.int32)
            intersect = tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            union = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred), tf.float32)
            self.loss_iou = tf.convert_to_tensor([intersect, union])
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

        with tf.variable_scope('adv_loss'):
            d_loss_fake_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.zeros([self.bs, 1])))
            d_loss_fake_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            d_loss_real_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logit,
                                                                                   labels=tf.ones([self.bs, 1])))
            self.g_loss = self.lada * d_loss_fake_1 + self.loss
            self.d_loss = d_loss_fake_0 + d_loss_real_1

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                t_vars = tf.trainable_variables()
                e_vars = [var for var in t_vars if 'encoder' in var.name]
                d_vars = [var for var in t_vars if 'Discriminator' in var.name]
                g_optm = tf.train.AdamOptimizer(self.learning_rate[0], name='Adam_g').\
                    minimize(self.g_loss, var_list=e_vars, global_step=self.global_step)
                d_optm = tf.train.AdamOptimizer(self.learning_rate[1], name='Adam_d').\
                    minimize(self.d_loss, var_list=d_vars, global_step=None)
                self.optimizer = [g_optm, d_optm]

    def make_learning_rate(self, n_train):
        self.learning_rate = []
        for i in range(2):
            self.learning_rate.append(tf.train.exponential_decay(self.lr[i], self.global_step,
                                                                 tf.cast(n_train/self.bs * self.ds[i], tf.int32),
                                                                 self.dr[i], staircase=True))

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, train_reader_source=None, train_reader_target=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_d_loss_summary_op = tf.summary.scalar('d_loss_validation', self.valid_d_loss)
        valid_g_loss_summary_op = tf.summary.scalar('g_loss_validation', self.valid_g_loss)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)

        if continue_dir is not None and os.path.exists(continue_dir):
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
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                X_batch, _ = train_reader_target.readerAction(sess)
                _, y_batch = train_reader_source.readerAction(sess)
                sess.run([self.optimizer[1]], feed_dict={self.inputs[x_name]: X_batch,
                                                         self.inputs[y_name]: y_batch,
                                                         self.trainable: True})

                if self.global_step_value % verb_step == 0:
                    step_cross_entropy, step_summary = sess.run([self.loss, self.summary],
                                                                feed_dict={self.inputs[x_name]: X_batch,
                                                                           self.inputs[y_name]: y_batch,
                                                                           self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            cross_entropy_valid_mean = []
            d_loss_valid_mean = []
            g_loss_valid_mean = []
            iou_valid_mean = np.zeros(2)
            X_batch_val, y_batch_val, pred_valid = None, None, None
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid = sess.run(
                    [self.pred, self.loss, self.loss_iou], feed_dict={self.inputs[x_name]: X_batch_val,
                                                                      self.inputs[y_name]: y_batch_val,
                                                                      self.trainable: False})
                _, y_batch_val_target = train_reader_source.readerAction(sess)
                d_loss_valid, g_loss_valid = sess.run([self.d_loss, self.g_loss],
                                                      feed_dict={self.inputs[x_name]: X_batch_val,
                                                                 self.inputs[y_name]: y_batch_val_target,
                                                                 self.trainable: False})
                cross_entropy_valid_mean.append(cross_entropy_valid)
                d_loss_valid_mean.append(d_loss_valid)
                g_loss_valid_mean.append(g_loss_valid)
                iou_valid_mean += iou_valid
            cross_entropy_valid_mean = np.mean(cross_entropy_valid_mean)
            d_loss_valid_mean = np.mean(d_loss_valid_mean)
            g_loss_valid_mean = np.mean(g_loss_valid_mean)
            iou_valid_mean = iou_valid_mean[0] / iou_valid_mean[1]
            duration = time.time() - start_time
            if valid_iou:
                print('Validation IoU: {:.3f}, duration: {:.3f}'.format(iou_valid_mean, duration))
            else:
                print('Val xent: {:.3f}, g_loss: {:.3f}, d_loss: {:.3f}, duration: {:.3f}'.
                      format(cross_entropy_valid_mean, d_loss_valid_mean, g_loss_valid_mean, duration))
            valid_summaries = sess.run([valid_cross_entropy_summary_op, valid_iou_summary_op,
                                        valid_d_loss_summary_op, valid_g_loss_summary_op],
                                       feed_dict={self.valid_cross_entropy: cross_entropy_valid_mean,
                                                  self.valid_iou: iou_valid_mean,
                                                  self.valid_d_loss: d_loss_valid_mean,
                                                  self.valid_g_loss: g_loss_valid_mean})
            for i in range(4):
                summary_writer.add_summary(valid_summaries[i], self.global_step_value)
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
                valid_image_summary = sess.run(
                    valid_image_summary_op, feed_dict={
                        self.valid_images: util_functions.image_summary(X_batch_val, y_batch_val, pred_valid, img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)


class UnetModelCropSplit(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetCropSplit'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None
        self.config = None

    def create_graph(self, x_name, class_num, start_filter_num=64):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        with tf.device("/gpu:0"):
            conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                               padding='valid', dropout=self.dropout_rate)
            conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                               padding='valid', dropout=self.dropout_rate)
            conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                               padding='valid', dropout=self.dropout_rate)
            conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                               padding='valid', dropout=self.dropout_rate)
            self.encoding = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                        padding='valid', dropout=self.dropout_rate)

        # upsample
        with tf.device("/gpu:1"):
            up6 = self.crop_upsample_concat(self.encoding, conv4, 8, name='6')
            conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                        padding='valid', dropout=self.dropout_rate)
            up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
            conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                        padding='valid', dropout=self.dropout_rate)
            up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
            conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                        padding='valid', dropout=self.dropout_rate)
            up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
            conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                        padding='valid', dropout=self.dropout_rate)

            self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
            self.output = tf.nn.softmax(self.pred)

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step,
                                                                                     colocate_gradients_with_ops=True)
            else:
                print('Train parameters in scope:')
                for layer in train_var_filter:
                    print(layer)
                train_vars = tf.trainable_variables()
                var_list = []
                for var in train_vars:
                    if var.name.split('/')[0] in train_var_filter:
                        var_list.append(var)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step,
                                                                                     var_list=var_list,
                                                                                     colocate_gradients_with_ops=True)


class UnetModelMoreCrop(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetMoreCrop'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = self.sfn

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, padding='valid')

        _, w, h, _ = conv9.get_shape().as_list()
        crop9 = tf.image.resize_image_with_crop_or_pad(conv9, w-40, h-40)

        self.pred = tf.layers.conv2d(crop9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 224



#%% UnetModelCropWeighted has loss function that assigns weights to imbalanced classes
class UnetModelCropWeighted(UnetModelCrop):
    
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,batch_size=5, start_filter_num=32, class_weights = [1,1]):
        super(UnetModelCropWeighted, self).__init__( inputs, trainable, input_size, model_name, dropout_rate,learn_rate, decay_step, decay_rate, epochs,batch_size, start_filter_num)
        self.cweights = tf.constant([w/class_weights[0] for w in class_weights], dtype = tf.float32)
        self.name = 'UnetCropWeighted'
        self.model_name = self.get_unique_name(model_name)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        # TODO loss type IoU
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction = tf.gather(pred_flat, indices)
            weights = tf.gather(self.cweights,gt)
            self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=prediction, labels=gt, weights=weights))



class UnetModel_Appendix(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetCropAppend'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.update_ops = None

    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_pool(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1', padding='valid')
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn*2, sfn*2], self.trainable, name='conv2', padding='valid')
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn*4, sfn*4], self.trainable, name='conv3', padding='valid')
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn*8, sfn*8], self.trainable, name='conv4', padding='valid')
        conv5 = self.conv_conv_pool(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False, padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False, padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False, padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False, padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, padding='valid')

        conv10 = tf.layers.conv2d(conv9, sfn, (1, 1), name='second_final', padding='same')
        self.pred = tf.layers.conv2d(conv10, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)


class ResUnetModel_Crop(UnetModelCrop):
    def create_graph(self, x_name, class_num, start_filter_num=32):
        self.class_num = class_num
        sfn = start_filter_num

        # downsample
        conv1, pool1 = self.conv_conv_identity_pool_crop(self.inputs[x_name], [sfn, sfn], self.trainable, name='conv1',
                                                         padding='valid')
        conv2, pool2 = self.conv_conv_identity_pool_crop(pool1, [sfn*2, sfn*2], self.trainable, name='conv2',
                                                         padding='valid')
        conv3, pool3 = self.conv_conv_identity_pool_crop(pool2, [sfn*4, sfn*4], self.trainable, name='conv3',
                                                         padding='valid')
        conv4, pool4 = self.conv_conv_identity_pool_crop(pool3, [sfn*8, sfn*8], self.trainable, name='conv4',
                                                         padding='valid')
        conv5 = self.conv_conv_identity_pool_crop(pool4, [sfn*16, sfn*16], self.trainable, name='conv5', pool=False,
                                                  padding='valid')

        # upsample
        up6 = self.crop_upsample_concat(conv5, conv4, 8, name='6')
        conv6 = self.conv_conv_identity_pool_crop(up6, [sfn*8, sfn*8], self.trainable, name='up6', pool=False,
                                                  padding='valid')
        up7 = self.crop_upsample_concat(conv6, conv3, 32, name='7')
        conv7 = self.conv_conv_identity_pool_crop(up7, [sfn*4, sfn*4], self.trainable, name='up7', pool=False,
                                                  padding='valid')
        up8 = self.crop_upsample_concat(conv7, conv2, 80, name='8')
        conv8 = self.conv_conv_identity_pool_crop(up8, [sfn*2, sfn*2], self.trainable, name='up8', pool=False,
                                                  padding='valid')
        up9 = self.crop_upsample_concat(conv8, conv1, 176, name='9')
        conv9 = self.conv_conv_identity_pool_crop(up9, [sfn, sfn], self.trainable, name='up9', pool=False,
                                                  padding='valid')

        self.pred = tf.layers.conv2d(conv9, class_num, (1, 1), name='final', activation=None, padding='same')
        self.output = tf.nn.softmax(self.pred)


class UnetModelTrilabel(UnetModelCrop):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'UnetTrilabel'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = [tf.placeholder(tf.float32, []), tf.placeholder(tf.float32, [])]
        self.valid_iou = [tf.placeholder(tf.float32, []), tf.placeholder(tf.float32, [])]
        self.valid_images = [tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                             input_size[1] * 3, 3], name='validation_images_0'),
                             tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                             input_size[1] * 3, 3], name='validation_images_1')
                             ]
        self.update_ops = None
        self.config = None

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        with tf.variable_scope('loss'):
            # background, class_1, class_2
            pred_class_1 = tf.concat([self.pred[:, :, :, 0], self.pred[:, :, :, 1]], axis=-1)
            pred_class_2 = tf.concat([self.pred[:, :, :, 0], self.pred[:, :, :, 2]], axis=-1)
            pred_flat_1 = tf.reshape(pred_class_1, [-1, 2])
            pred_flat_2 = tf.reshape(pred_class_2, [-1, 2])
            _, w, h, _ = self.inputs[y_name].get_shape().as_list()
            y = tf.image.resize_image_with_crop_or_pad(self.inputs[y_name], w-self.get_overlap(), h-self.get_overlap())
            y_flat = tf.reshape(tf.squeeze(y, axis=[3]), [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(y_flat, self.class_num - 1)), 1)
            gt = tf.gather(y_flat, indices)
            prediction_1 = tf.gather(pred_flat_1, indices)
            prediction_2 = tf.gather(pred_flat_2, indices)

            pred_1 = tf.argmax(prediction_1, axis=-1, output_type=tf.int32)
            pred_2 = tf.argmax(prediction_2, axis=-1, output_type=tf.int32)
            intersect_1 = tf.cast(tf.reduce_sum(gt * pred_1), tf.float32)
            union_1 = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred_1), tf.float32) \
                    - tf.cast(tf.reduce_sum(gt * pred_1), tf.float32)
            intersect_2 = tf.cast(tf.reduce_sum(gt * pred_2), tf.float32)
            union_2 = tf.cast(tf.reduce_sum(gt), tf.float32) + tf.cast(tf.reduce_sum(pred_2), tf.float32) \
                      - tf.cast(tf.reduce_sum(gt * pred_2), tf.float32)
            self.loss_iou = [tf.convert_to_tensor([intersect_1, union_1]), tf.convert_to_tensor([intersect_2, union_2])]

            if loss_type == 'xent':
                self.loss = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_1, labels=gt)),
                             tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction_2, labels=gt))]
            else:
                pass

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):
            if train_var_filter is None:
                self.optimizer = [tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss[0],
                                                                                     global_step=self.global_step),
                                  tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss[1],
                                                                                      global_step=None)]
            else:
                # TODO make optms into list
                print('Train parameters in scope:')
                for layer in train_var_filter:
                    print(layer)
                train_vars = tf.trainable_variables()
                var_list = []
                for var in train_vars:
                    if var.name.split('/')[0] in train_var_filter:
                        var_list.append(var)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step,
                                                                                     var_list=var_list)

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy 0', self.loss[0])
        tf.summary.scalar('Cross Entropy 1', self.loss[1])
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def run(self, train_reader_1=None, train_reader_2=None, valid_reader_1=None, valid_reader_2=None, test_reader=None,
            pretrained_model_dir=None,  layers2load=None, isTrain=False,
            img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5,
            gpu=None, tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True):
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
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir, valid_iou=valid_iou)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            pad = self.get_overlap()
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify_shrink(result,
                                                          [tile_size[0] + pad, tile_size[1] + pad],
                                                          tile_size,
                                                          patch_size,
                                                          [patch_size[0] - pad, patch_size[1] - pad],
                                                          overlap=pad)
            return util_functions.get_pred_labels(image_pred) * truth_val

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader_0=None, train_reader_1=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False):
        # define summary operations
        valid_cross_entropy_summary_op = [tf.summary.scalar('xent_validation_0', self.valid_cross_entropy[0]),
                                          tf.summary.scalar('xent_validation_1', self.valid_cross_entropy[1])]
        valid_iou_summary_op = [tf.summary.scalar('iou_validation_0', self.valid_iou[0]),
                                tf.summary.scalar('iou_validation_1', self.valid_iou[1])]
        valid_image_summary_op = [tf.summary.image('Validation_images_summary_0', self.valid_images[0],
                                                   max_outputs=10),
                                  tf.summary.image('Validation_images_summary_1', self.valid_images[1],
                                                   max_outputs=10)
                                  ]

        if continue_dir is not None and os.path.exists(continue_dir):
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
                X_batch, y_batch = train_reader_0.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[0], self.global_step],
                                                     feed_dict={self.inputs[x_name]:X_batch,
                                                                self.inputs[y_name]:y_batch,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = \
                        sess.run([self.pred, self.loss[0], self.summary],
                                 feed_dict={self.inputs[x_name]: X_batch,
                                            self.inputs[y_name]: y_batch,
                                            self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy_0 = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
                X_batch, y_batch = train_reader_1.readerAction(sess)
                _, self.global_step_value = sess.run([self.optimizer[1], self.global_step],
                                                     feed_dict={self.inputs[x_name]: X_batch,
                                                                self.inputs[y_name]: y_batch,
                                                                self.trainable: True})
                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary = \
                        sess.run([self.pred, self.loss[1], self.summary],
                                 feed_dict={self.inputs[x_name]: X_batch,
                                            self.inputs[y_name]: y_batch,
                                            self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy_1 = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
            # validation
            for i in range(2):
                cross_entropy_valid_mean = []
                iou_valid_mean = np.zeros(2)
                for step in range(0, n_valid, self.bs):
                    X_batch_val, y_batch_val = valid_reader[i].readerAction(sess)
                    pred_valid, cross_entropy_valid, iou_valid = sess.run([self.pred, self.loss[i], self.loss_iou[i]],
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
                valid_cross_entropy_summary = sess.run(valid_cross_entropy_summary_op[i],
                                                       feed_dict={self.valid_cross_entropy[i]: cross_entropy_valid_mean})
                valid_iou_summary = sess.run(valid_iou_summary_op,
                                             feed_dict={self.valid_iou[i]: iou_valid_mean})
                summary_writer.add_summary(valid_cross_entropy_summary, self.global_step_value)
                summary_writer.add_summary(valid_iou_summary, self.global_step_value)

                if image_summary is not None:
                    valid_image_summary = sess.run(valid_image_summary_op[i],
                                                   feed_dict={self.valid_images[i]:
                                                                  image_summary(X_batch_val[:,:,:,:3], y_batch_val, pred_valid,
                                                                                img_mean)})
                    summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)


class Vgg16:
    def __init__(self):
        self.parameters = []

    def saver(self):
        return tf.train.Saver()

    def maxpool(self, name, input_data):
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name=name)
        return out

    def conv(self, name, input_data, out_channel, is_train):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for index, variables in enumerate(self.parameters):
            sess.run(variables.assign(weights[keys[index]]))
        print("Vgg loading pre-trained weights ")


class UnetModel_Optical_NN(network.Network):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, weights_lr_scale=None, pretrained_weights=None):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = 'Unet'
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 2, 3], name='validation_images')
        self.update_ops = None
        self.config = None
        self.n_train = 0
        self.n_valid = 0
        self.weights_lr_scale = weights_lr_scale
        self.pretrained_weights = pretrained_weights

    def make_ckdir(self, ckdir, patch_size, par_dir=None):
        if type(patch_size) is list:
            patch_size = patch_size[0]
        # make unique directory for save
        dir_name = '{}_PS{}_BS{}_EP{}_LR{}_DS{}_DR{}_SFN{}'. \
            format(self.model_name, patch_size, self.bs, self.epochs, self.lr, self.ds, self.dr, self.sfn)
        if par_dir is None:
            self.ckdir = os.path.join(ckdir, dir_name)
        else:
            self.ckdir = os.path.join(ckdir, par_dir, dir_name)

    def create_graph(self, x_name, class_num, light_source_num=3, is_normalized=True, is_RGB_input=True, experiment_setup='Normal', conv_mode=False):
        self.class_num = class_num
        self.light_source_num = light_source_num
        sfn = self.sfn
        # self.coe_array = tf.Variable(tf.truncated_normal(
        #     shape=[self.light_source_num], mean=0.0, stddev=1.0), name='light_source_weights')
        # self.coe_array = tf.Variable(tf.constant(0.5, shape=[self.light_source_num])
        #                        , name='light_source_weights')
        # coe_arr = np.zeros(75)
        # coe_arr[[12, 37, 62]] = 1
        if self.pretrained_weights is not None:
            weights_csv = pd.read_csv(self.pretrained_weights, header=None)
            coe_arr = weights_csv.values.reshape(75)
        else:
            pass
            # coe_arr = np.random.normal(loc=0, scale=1.0, size=None)
            # coe_arr = 0.01 * (coe_arr - np.mean(coe_arr))
            # small disturbation and zero-mean
            coe_arr = 0.01 * np.ones(75)
        self.coe_array = tf.Variable(coe_arr, name='light_source_weights', dtype=tf.float32)
        self.light_source_weights = self.coe_array
        if is_normalized:
            mean, var = tf.nn.moments(self.coe_array, axes=[0])
            self.coe_array = (self.coe_array - mean)
        components = []
        self.light_source_weights_normalized = self.coe_array

        for i in range(self.light_source_num):
            components.append(
                tf.scalar_mul(tf.gather(tf.cast(self.coe_array, tf.float64), i),
                              tf.gather(tf.cast(self.inputs[x_name], tf.float64), i, axis=3)))
        if is_RGB_input:
            sumed_input = tf.concat([tf.cast(
                tf.expand_dims(tf.reduce_sum(tf.convert_to_tensor(components[:int(light_source_num / 3)]), axis=0),
                               axis=3), tf.float32),
                                     tf.cast(tf.expand_dims(tf.reduce_sum(tf.convert_to_tensor(
                                         components[int(light_source_num / 3):int(2 * light_source_num / 3)]), axis=0),
                                                            axis=3), tf.float32),
                                     tf.cast(tf.expand_dims(tf.reduce_sum(tf.convert_to_tensor(
                                         components[int(2 * light_source_num / 3):int(3 * light_source_num / 3)]),
                                                                          axis=0), axis=3), tf.float32)]
                                    , axis=3)
        else:
            sumed_input = tf.cast(
                tf.expand_dims(tf.reduce_sum(tf.convert_to_tensor(components), axis=0), axis=3), tf.float32)
        if experiment_setup == 'Normal':
            pass
        elif experiment_setup == '13special':
            sumed_input = tf.concat([tf.cast(
                tf.expand_dims(self.inputs[x_name][:, :, :, 12],
                               axis=3), tf.float32),
                tf.cast(
                    tf.expand_dims(self.inputs[x_name][:, :, :, 37],
                                   axis=3), tf.float32),
                tf.cast(
                    tf.expand_dims(self.inputs[x_name][:, :, :, 62],
                                   axis=3), tf.float32)]
                                    , axis=3)
        elif experiment_setup == 'Average_all':
            sumed_input = tf.concat([tf.cast(
                tf.expand_dims(tf.reduce_mean(self.inputs[x_name][:, :, :, :25], axis=3),
                               axis=3), tf.float32),
                tf.cast(
                    tf.expand_dims(tf.reduce_mean(self.inputs[x_name][:, :, :, 25:50], axis=3),
                                   axis=3), tf.float32),
                tf.cast(
                    tf.expand_dims(tf.reduce_mean(self.inputs[x_name][:, :, :, 50:75], axis=3),
                                   axis=3), tf.float32)]
                , axis=3)
        # downsample
        # self.see_gradients = tf.gradients(sumed_input, self.coe_array)
        self.see_gradients = tf.constant([1, 2])
        if conv_mode:
            pass
        # downsample
        conv1, pool1 = self.conv_conv_pool(sumed_input, [sfn, sfn], self.trainable, name='conv1',
                                           dropout=self.dropout_rate)
        conv2, pool2 = self.conv_conv_pool(pool1, [sfn * 2, sfn * 2], self.trainable, name='conv2',
                                           dropout=self.dropout_rate)
        conv3, pool3 = self.conv_conv_pool(pool2, [sfn * 4, sfn * 4], self.trainable, name='conv3',
                                           dropout=self.dropout_rate)
        conv4, pool4 = self.conv_conv_pool(pool3, [sfn * 8, sfn * 8], self.trainable, name='conv4',
                                           dropout=self.dropout_rate)
        conv5 = self.conv_conv_pool(pool4, [sfn * 16, sfn * 16], self.trainable, name='conv5', pool=False,
                                    dropout=self.dropout_rate)

        # upsample
        up6 = self.upsample_concat(conv5, conv4, name='6')
        conv6 = self.conv_conv_pool(up6, [sfn * 8, sfn * 8], self.trainable, name='up6', pool=False,
                                    dropout=self.dropout_rate)
        up7 = self.upsample_concat(conv6, conv3, name='7')
        conv7 = self.conv_conv_pool(up7, [sfn * 4, sfn * 4], self.trainable, name='up7', pool=False,
                                    dropout=self.dropout_rate)
        up8 = self.upsample_concat(conv7, conv2, name='8')
        conv8 = self.conv_conv_pool(up8, [sfn * 2, sfn * 2], self.trainable, name='up8', pool=False,
                                    dropout=self.dropout_rate)
        up9 = self.upsample_concat(conv8, conv1, name='9')
        conv9 = self.conv_conv_pool(up9, [sfn, sfn], self.trainable, name='up9', pool=False, dropout=self.dropout_rate)
        up9_ex = self.upsample_to_same_concat(conv9, sumed_input, name='9_ex')
        conv9_ex = self.conv_conv_pool(up9_ex, [sfn, sfn], self.trainable, name='up9_ex', pool=False, dropout=self.dropout_rate)
        self.pred = tf.layers.conv2d(conv9_ex, class_num, (1, 1), name='final', activation=None, padding='same',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        self.output = tf.nn.softmax(self.pred)

    def load_weights(self, ckpt_dir, layers2load):
        # this is different from network.load()
        # this function only loads specified layers
        layers_list = []
        if isinstance(layers2load, str):
            layers2load = [int(a) for a in layers2load.split(',')]
        for layer_id in layers2load:
            assert 1 <= layer_id <= 9
            if layer_id <= 5:
                prefix = 'layerconv'
            else:
                prefix = 'layerup'
            layers_list.append('{}{}'.format(prefix, layer_id))

        load_dict = {}
        for layer_name in layers_list:
            feed_layer = layer_name + '/'
            load_dict[feed_layer] = feed_layer
        try:
            latest_check_point = tf.train.latest_checkpoint(ckpt_dir)
            tf.contrib.framework.init_from_checkpoint(ckpt_dir, load_dict)
            print('loaded {}'.format(latest_check_point))
        except tf.errors.NotFoundError:
            with open(os.path.join(ckpt_dir, 'checkpoint'), 'r') as f:
                ckpts = f.readlines()
            ckpt_file_name = ckpts[0].split('/')[-1].strip().strip('\"')
            latest_check_point = os.path.join(ckpt_dir, ckpt_file_name)
            tf.contrib.framework.init_from_checkpoint(latest_check_point, load_dict)
            print('loaded {}'.format(latest_check_point))

    def restore_model(self, sess):
        # automatically restore last saved model if checkpoint exists
        if tf.train.latest_checkpoint(self.ckdir):

            self.load(self.ckdir, sess)

            with open(os.path.join(self.ckdir, 'checkpoint'), 'r') as f:
                model_checkpoint_path = f.readline().split('/')[-1]
            buf = [int(i) for i in re.findall(r"\d+", model_checkpoint_path)]
            if len(buf) == 1:
                start_step = buf[0] + 1
                self.start_epoch = int(np.floor(start_step / (8000 / self.bs)))
            elif len(buf) == 2:
                self.start_epoch = buf[0] + 1
                start_step = buf[1] + 1
        else:
            self.start_epoch, start_step = [0, 0]

        sess.run(self.global_step.assign(start_step))
        self.global_step_value = self.global_step.eval()
        print('restoring model from epoch %d step %d' % (self.start_epoch, self.global_step_value))

    def make_learning_rate(self, n_train):
        self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                        tf.cast(n_train / self.bs * self.ds, tf.int32),
                                                        self.dr, staircase=True)

    def make_loss(self, y_name, loss_type='xent', **kwargs):
        # TODO loss type IoU
        with tf.variable_scope('loss'):
            pred_flat = tf.reshape(self.pred, [-1, self.class_num])
            y_flat = tf.reshape(tf.squeeze(self.inputs[y_name], axis=[3]), [-1, ])
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
            elif loss_type == 'regularization':
                l2_loss = tf.losses.get_regularization_loss()
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)) + 0.5 * l2_loss

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
                    sigmoid_p, 1e-8, 1.0)) - (1 - kwargs['alpha']) * (neg_p_sub ** kwargs['gamma']) * tf.log(
                    tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
                self.loss = tf.reduce_sum(per_entry_cross_ent)

    def make_update_ops(self, x_name, y_name):
        tf.add_to_collection('inputs', self.inputs[x_name])
        tf.add_to_collection('inputs', self.inputs[y_name])
        tf.add_to_collection('outputs', self.pred)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def make_optimizer(self, train_var_filter):
        with tf.control_dependencies(self.update_ops):

            t_vars = tf.trainable_variables()
            w_vars = [var for var in t_vars if 'light_source_weights' in var.name]
            other_vars = [var for var in t_vars if 'light_source_weights' not in var.name]
            if self.weights_lr_scale is not None:
                optimizer_w = tf.train.AdamOptimizer(self.learning_rate*self.weights_lr_scale).minimize(self.loss,
                                                                                     global_step=None, var_list=w_vars)
                optimizer_other = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step, var_list=other_vars)
                self.optimizer = tf.group(optimizer_w, optimizer_other)
                # stop
                return 0

            if train_var_filter is None:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step)

            else:
                print('Train parameters in scope:')
                for layer in train_var_filter:
                    print(layer)
                train_vars = tf.trainable_variables()
                var_list = []
                for var in train_vars:
                    if var.name.split('/')[0] in train_var_filter:
                        var_list.append(var)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                     global_step=self.global_step,
                                                                                     var_list=var_list)

    def make_summary(self, hist=False):
        if hist:
            tf.summary.histogram('Predicted Prob', tf.argmax(tf.nn.softmax(self.pred), 1))
        tf.summary.scalar('Cross Entropy', self.loss)
        tf.summary.scalar('learning rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train_config(self, x_name, y_name, n_train, n_valid, patch_size, ckdir, loss_type='regularization',
                     train_var_filter=None,
                     hist=False, par_dir=None, **kwargs):
        self.make_loss(y_name, loss_type, **kwargs)
        self.make_learning_rate(n_train)
        self.make_update_ops(x_name, y_name)
        self.make_optimizer(train_var_filter)
        self.make_ckdir(ckdir, patch_size, par_dir)
        self.make_summary(hist)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.n_train = n_train
        self.n_valid = n_valid

    def train(self, x_name, y_name, n_train, sess, summary_writer, n_valid=1000,
              train_reader=None, valid_reader=None,
              image_summary=None, verb_step=100, save_epoch=5,
              img_mean=np.array((0, 0, 0), dtype=np.float32),
              continue_dir=None, valid_iou=False, name_for_coe='Normal', obsolete_training=False):
        # define summary operations
        valid_cross_entropy_summary_op = tf.summary.scalar('xent_validation', self.valid_cross_entropy)
        valid_iou_summary_op = tf.summary.scalar('iou_validation', self.valid_iou)
        valid_image_summary_op = tf.summary.image('Validation_images_summary', self.valid_images,
                                                  max_outputs=10)
        cur_initial = True
        if continue_dir is not None and os.path.exists(continue_dir):
            self.load(continue_dir, sess)
            gs = sess.run(self.global_step)
            start_epoch = int(np.ceil(gs / n_train * self.bs))
            start_step = gs - int(start_epoch * n_train / self.bs)
        else:
            start_epoch = 0
            start_step = 0

        cross_entropy_valid_min = np.inf
        iou_valid_max = 0
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            for step in range(start_step, n_train, self.bs):
                X_batch, y_batch = train_reader.readerAction(sess)
                if obsolete_training:
                    pass
                else:
                    _, self.global_step_value = sess.run([self.optimizer, self.global_step],
                                                         feed_dict={self.inputs[x_name]: X_batch,
                                                                    self.inputs[y_name]: y_batch,
                                                                    self.trainable: True})


                if self.global_step_value % verb_step == 0:
                    pred_train, step_cross_entropy, step_summary, gradients = sess.run([self.pred, self.loss, self.summary, self.see_gradients],
                                                                            feed_dict={self.inputs[x_name]: X_batch,
                                                                                       self.inputs[y_name]: y_batch,
                                                                                       self.trainable: False})
                    summary_writer.add_summary(step_summary, self.global_step_value)
                    print('Epoch {:d} step {:d}\tcross entropy = {:.3f}'.
                          format(epoch, self.global_step_value, step_cross_entropy))
                    # print('See gradients: ', gradients)
            # validation
            cross_entropy_valid_mean = []
            iou_valid_mean = np.zeros(2)
            for step in range(0, n_valid, self.bs):
                X_batch_val, y_batch_val = valid_reader.readerAction(sess)
                pred_valid, cross_entropy_valid, iou_valid, coe_array  = sess.run([self.pred, self.loss, self.loss_iou, self.coe_array],
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
            for index, each_coe in enumerate(coe_array):
                print('Current coefficient of light for source {}: {:.3f}'.format(index, each_coe))
            if cur_initial is True:
                np.savetxt(name_for_coe+'_initial_light_source_coefficient_1.csv', coe_array, delimiter=',')
                cur_initial = False
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
                valid_image_summary = sess.run(valid_image_summary_op,
                                               feed_dict={self.valid_images:
                                                              image_summary(X_batch_val[:, :, :, :3], y_batch_val,
                                                                            pred_valid,
                                                                            img_mean)})
                summary_writer.add_summary(valid_image_summary, self.global_step_value)

            if epoch % save_epoch == 0:
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
                saver.save(sess, '{}/model_{}.ckpt'.format(self.ckdir, epoch), global_step=self.global_step)
        np.savetxt(name_for_coe+'_normalized_final_light_source_coefficient_1.csv', coe_array, delimiter=',')

    def test(self, x_name, sess, test_iterator):
        result = []
        for X_batch in test_iterator:
            pred = sess.run(self.output, feed_dict={self.inputs[x_name]: X_batch,
                                                    self.trainable: False})
            result.append(pred)
        result = np.vstack(result)
        return result

    def get_overlap(self):
        # TODO calculate the padding pixels
        return 0

    def run(self, train_reader=None, valid_reader=None, test_reader=None, pretrained_model_dir=None, layers2load=None,
            isTrain=False, img_mean=np.array((0, 0, 0), dtype=np.float32), verb_step=100, save_epoch=5, gpu=None,
            tile_size=(5000, 5000), patch_size=(572, 572), truth_val=1, continue_dir=None, load_epoch_num=None,
            valid_iou=False, best_model=True, name_for_coe='Normal'):
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
                        self.load(pretrained_model_dir, sess, saver, epoch=load_epoch_num)
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)
                try:
                    train_summary_writer = tf.summary.FileWriter(self.ckdir, sess.graph)
                    self.train('X', 'Y', self.n_train, sess, train_summary_writer,
                               n_valid=self.n_valid, train_reader=train_reader, valid_reader=valid_reader,
                               image_summary=util_functions.image_summary_only_pred_and_GT, img_mean=img_mean,
                               verb_step=verb_step, save_epoch=save_epoch, continue_dir=continue_dir,
                               valid_iou=valid_iou, name_for_coe=name_for_coe)
                finally:
                    coord.request_stop()
                    coord.join(threads)
                    saver.save(sess, '{}/model.ckpt'.format(self.ckdir), global_step=self.global_step)
        else:
            if self.config is None:
                self.config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=self.config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                self.load(pretrained_model_dir, sess, epoch=load_epoch_num, best_model=best_model)
                self.model_name = pretrained_model_dir.split('/')[-1]
                result = self.test('X', sess, test_reader)
            image_pred = uabUtilreader.un_patchify(result, tile_size, patch_size)
            return util_functions.get_pred_labels(image_pred) * truth_val

    def evaluate(self, rgb_list, gt_list, rgb_dir, gt_dir, input_size, tile_size, batch_size, img_mean,
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
        iou_return = {}
        for file_name, file_name_truth in zip(rgb_list, gt_list):
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
            # if test_mode == 'B & Bg':
            #     pred_for_iou[np.where(pred_for_iou == 2)] = 0
            #
            # elif test_mode == 'R & Bg':
            #     # pass  # no need to do anything for the label
            #     # truth_label_img[np.where(truth_label_img == 1)] = 2
            #     pred_for_iou[np.where(pred_for_iou == 1)] = 0
            #     pred_for_iou[np.where(pred_for_iou == 2)] = 1
            # max_pred = np.max(truth_label_img)

            iou = util_functions.iou_metric(truth_label_img, pred_for_iou, divide_flag=True)
            iou_record.append(iou)
            iou_return[tile_name] = iou

            duration = time.time() - start_time
            if verb:
                print('{} mean IoU={:.3f}, duration: {:.3f}'.format(tile_name, iou[0] / iou[1], duration))

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

    def iou(img1, img2):
        return np.sum(np.multiply(img1, img2)) / np.sum(img1 * img2 > 0)


    def aji(gt, pred):
        if gt.max() == 0:
            return 0
        gt = label(gt, neighbors=8, background=0)
        pred = label(pred, neighbors=8, background=0)
        intersect, union, ghost = [], [], []
        not_ghost_id = set()
        for i in range(1, np.max(gt) + 1):
            unit_map = np.where(gt == i, 1, 0)
            iou_list = []
            overlap_id_list = list(set(np.multiply(unit_map, pred).flatten()) - set([0]))
            if overlap_id_list:
                for cell_id in overlap_id_list:
                    iou_list.append(iou(unit_map, np.where(pred == cell_id, 1, 0)))
                max_id = overlap_id_list[np.argmax(iou_list)]
                intersect.append(np.sum(np.multiply(unit_map, np.where(pred == max_id, 1, 0))))
                union.append(np.sum((unit_map + np.where(pred == max_id, 1, 0)) > 0))
                not_ghost_id |= set(overlap_id_list)
            else:
                intersect.append(0)
                union.append(np.sum(unit_map))

        # compute ghost
        if len(not_ghost_id) != np.max(pred):
            for j in set(range(1, np.max(pred) + 1)) - not_ghost_id:
                ghost.append(np.sum(np.where(pred == j, 1, 0)))
        return np.sum(intersect) / (np.sum(union) - np.sum(ghost))