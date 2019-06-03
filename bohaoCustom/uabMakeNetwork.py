import os
from glob import glob
import tensorflow as tf


class Network(object):
    def __init__(self, inputs, trainable, dropout_rate=None,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5):
        self.total_epoch = tf.placeholder(tf.float32, shape=None, name='total_epoch')
        self.current_epoch = tf.placeholder(tf.float32, shape=None, name='current_epoch')
        self.inputs = inputs
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        self.name = 'network'
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        self.global_step_value = 0
        self.lr = learn_rate
        self.ds = decay_step
        self.dr = decay_rate
        self.epochs = epochs
        self.bs = batch_size
        self.class_num = []
        self.loss = []
        self.loss_iou = []
        self.optimizer = []
        self.pred = []
        self.output = []
        self.summary = []
        self.ckdir = []
        self.model_name = []

    def create_graph(self, **kwargs):
        raise NotImplementedError('Must be implemented by the subclass')

    def make_ckdir(self, ckdir, patch_size, par_dir=None):
        if type(patch_size) is list:
            patch_size = patch_size[0]
        # make unique directory for save
        dir_name = '{}_PS{}_BS{}_EP{}_LR{}_DS{}_DR{}'.\
            format(self.model_name, patch_size, self.bs, self.epochs, self.lr, self.ds, self.dr)
        if par_dir is None:
            self.ckdir = os.path.join(ckdir, dir_name)
        else:
            self.ckdir = os.path.join(ckdir, par_dir, dir_name)

    def load(self, model_path, sess, saver=None, epoch=None, best_model=False):
        # this can only be called after create_graph()
        # loads all weights in a graph
        if saver is None:
            saver = tf.train.Saver(var_list=tf.global_variables())
        if os.path.exists(model_path) and tf.train.get_checkpoint_state(model_path):
            if epoch is None:
                best_model_path = glob(os.path.join(model_path, 'best_model.ckpt*.index'))
                if len(best_model_path) > 0 and best_model:
                    best_model_name = best_model_path[0][:-6]
                    saver.restore(sess, best_model_name)
                    print('loaded {}'.format(best_model_name))
                else:
                    try:
                        latest_check_point = tf.train.latest_checkpoint(model_path)
                        saver.restore(sess, latest_check_point)
                        print('loaded {}'.format(latest_check_point))
                    except (tf.errors.NotFoundError, ValueError):
                        saver = tf.train.Saver(var_list=[i for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                                                         if 'save' not in i.name])
                        with open(os.path.join(model_path, 'checkpoint'), 'r') as f:
                            ckpts = f.readlines()
                        ckpt_file_name = ckpts[0].split('/')[-1].strip().strip('\"')
                        latest_check_point = os.path.join(model_path, ckpt_file_name)
                        print('-'*20+latest_check_point+'-'*20)
                        saver.restore(sess, latest_check_point)
                        print('loaded {}'.format(latest_check_point))

            else:
                ckpt_file_name = glob(os.path.join(model_path, 'model_{}.ckpt*.index'.format(epoch)))
                ckpt_file_name = ckpt_file_name[0][:-6]
                saver.restore(sess, ckpt_file_name)
                print('loaded {}'.format(ckpt_file_name))
        else:
            saver.restore(sess, model_path)
            print('loaded {}'.format(model_path))

    def get_unique_name(self, suffix):
        if len(suffix) > 0:
            return '{}_{}'.format(self.name, suffix)
        else:
            return self.name

    def conv_conv_pool(self, input_, n_filters, training, name, kernal_size=(3, 3),
                       conv_stride=(1, 1), pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                       activation=tf.nn.relu, padding='same', bn=True, dropout=None, reuse=False):
        net = input_

        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, kernal_size, activation=None, strides=conv_stride,
                                       padding=padding, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                       name='conv_{}'.format(i + 1), reuse=reuse)
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1), reuse=reuse)
                net = activation(net, name='relu_{}'.format(name, i + 1))
                if dropout is not None:
                    net = tf.layers.dropout(net, rate=self.dropout_rate, training=training,
                                            name='drop_{}'.format(name, i + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool

    def conv_conv_identity_pool_crop(self, input_, n_filters, training, name, kernal_size=(3, 3),
                                     pool=True, pool_size=(2, 2), pool_stride=(2, 2),
                                     activation=tf.nn.relu, padding='same', bn=True, dropout=None):
        net = input_
        _, w, h, _ = input_.get_shape().as_list()
        with tf.variable_scope('layer{}'.format(name)):
            input_conv = tf.layers.conv2d(net, n_filters[-1], kernal_size, activation=None,
                                          padding='same', name='conv_skip')
            input_conv = tf.layers.batch_normalization(input_conv, training=training, name='bn_skip')

            for i, F in enumerate(n_filters):
                net = tf.layers.conv2d(net, F, kernal_size, activation=None,
                                       padding=padding, name='conv_{}'.format(i + 1))
                if bn:
                    net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                net = activation(net, name='relu_{}'.format(name, i + 1))
                if dropout is not None:
                    net = tf.layers.dropout(net, rate=self.dropout_rate, training=training,
                                            name='drop_{}'.format(name, i + 1))

            # identity connection
            if padding == 'valid':
                input_conv = tf.image.resize_image_with_crop_or_pad(input_conv, w-2*len(n_filters), h-2*len(n_filters))
            net = tf.add(input_conv, net)
            net = activation(net, name='relu_{}'.format(name, len(n_filters) + 1))

            if pool is False:
                return net

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
            return net, pool

    def concat(self, input_a, input_b, training, name):
        with tf.variable_scope('layer{}'.format(name)):
            inputA_norm = tf.layers.batch_normalization(input_a, training=training, name='bn')
            return tf.concat([inputA_norm, input_b], axis=-1, name='concat_{}'.format(name))

    def upsampling_2D(self, tensor, name, size=(2, 2)):
        H, W, _ = tensor.get_shape().as_list()[1:]  # first dim is batch num
        H_multi, W_multi = size
        target_H = H * H_multi
        target_W = W * W_multi

        return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name='upsample_{}'.format(name))

    def upsampling_2D_bilinear(self, tensor, name, size=(2, 2)):
        H, W, _ = tensor.get_shape().as_list()[1:]  # first dim is batch num
        H_multi, W_multi = size
        target_H = H * H_multi
        target_W = W * W_multi

        return tf.image.resize_bilinear(tensor, (target_H, target_W), name='upsample_bilinear_{}'.format(name))

    def upsample_concat(self, input_a, input_b, name, size=(2, 2)):
        upsample = self.upsampling_2D(input_a, size=size, name=name)
        return tf.concat([upsample, input_b], axis=-1, name='concat_{}'.format(name))

    def upsample_to_same_concat(self, input_a, input_b, name, size=(2, 2)):
        target_H, target_W, _ = input_b.get_shape().as_list()[1:]
        upsample = tf.image.resize_nearest_neighbor(input_a, (target_H, target_W), name='upsample_{}'.format(name))
        # upsample = self.upsampling_2D(input_a, size=size, name=name)
        return tf.concat([upsample, input_b], axis=-1, name='concat_{}'.format(name))

    def crop_upsample_concat(self, input_a, input_b, margin, name):
        with tf.variable_scope('crop_upsample_concat'):
            _, w, h, _ = input_b.get_shape().as_list()
            input_b_crop = tf.image.resize_image_with_crop_or_pad(input_b, w-margin, h-margin)
            return self.upsample_concat(input_a, input_b_crop, name)

    def fc_fc(self, input_, n_filters, training, name, activation=tf.nn.relu, dropout=True, reuse=False):
        net = input_
        with tf.variable_scope('layer{}'.format(name)):
            for i, F in enumerate(n_filters):
                net = tf.layers.dense(net, F, activation=None, reuse=reuse, name='dense_{}'.format(i+1))
                if activation is not None:
                    net = activation(net, name='relu_{}'.format(name, i + 1))
                if dropout:
                    net = tf.layers.dropout(net, rate=self.dropout_rate, training=training, name='drop_{}'.format(name, i + 1))
        return net

    def Wide_Resnet_block(self,
                          input_,
                          unit_num,
                          in_channels,
                          channels,  #n_filters are replaced by a list of channels
                          training,
                          name,   # kernel size is fixed, no need to input kernel size
                          conv_stride=(1, 1),
                          pool=True,
                          pool_size=(2, 2),
                          pool_stride=(2, 2),
                          activation=tf.nn.relu,
                          padding='same',
                          bn=True,
                          dropout=None):
        ''''
               Created on Tue Nov 27 20:40:11 2018

               @author: Fanjie Kong

               Wider ResNet with pre-activation (identity mapping) blocks
               Parameters
               ----------
               unit_num : int
                   Number of residual units for one block
               in_channels : int
                   Number of input channel
               channels : list of int
                    Number of channels in the internal feature maps. Can either have two or three elements.

        '''
        net = input_
        is_bottleneck = (len(channels) == 3)
        need_proj_conv = (in_channels != channels[-1])
        with tf.variable_scope('layer{}'.format(name)):
            for i in range(unit_num):
                net = tf.layers.batch_normalization(net, training=training, name='bn_{}'.format(i+1))
                res = net  # res equals the last net
                # Check parameters for inconsistencies
                if len(channels) != 2 and len(channels) != 3:
                    raise ValueError("channels must contain either two or three values")
                # Basic Wide_Resnet Block
                if not is_bottleneck:
                    net = tf.layers.conv2d(net, channels[0], (3, 3), activation=None, strides=conv_stride,
                                           padding=padding, name='conv1_{}'.format(i+1))
                    if bn:
                        net = tf.layers.batch_normalization(net, training=training, name='bn1_{}'.format(i+1))
                    net = activation(net, name='relu1_{}'.format(name, i+1))
                    if dropout is not None:
                        net = tf.layers.dropout(net, rate=self.dropout_rate, training=training,
                                                name='drop1_{}'.format(name, i+1))
                    net = tf.layers.conv2d(net, channels[1], (3, 3), activation=None, strides=conv_stride,
                                           padding=padding, name='conv2_{}'.format(i+1))

                else:
                    # Bottleneck one
                    net = tf.layers.conv2d(net, channels[0], (1, 1), activation=None, strides=conv_stride,
                                          padding=padding, name='conv1_{}'.format(i+1))
                    if bn:
                        net = tf.layers.batch_normalization(net, training=training, name='bn1_{}'.format(i+1))
                    net = activation(net, name='relu1_{}'.format(name, i+1))
                    if dropout is not None:
                        net = tf.layers.dropout(net, rate=self.dropout_rate, training=training,
                                                name='drop1_{}'.format(name, i+1))
                    # Conv2
                    net = tf.layers.conv2d(net, channels[1], (3, 3), activation=None, strides=conv_stride,
                                           padding=padding, name='conv2_{}'.format(i+1))
                    if bn:
                        net = tf.layers.batch_normalization(net, training=training, name='bn2_{}'.format(i+1))
                    net = activation(net, name='relu2_{}'.format(name, i+1))
                    if dropout is not None:
                        net = tf.layers.dropout(net, rate=self.dropout_rate, training=training,
                                                name='drop2_{}'.format(name, i+1))
                    net = tf.layers.conv2d(net, channels[2], (1, 1), activation=None, strides=conv_stride,
                                           padding=padding, name='conv3_{}'.format(i+1))
                if need_proj_conv:
                    res = tf.layers.conv2d(res, channels[-1], (1, 1), activation=None, strides=conv_stride,
                                            padding=padding, name='proj_conv_{}'.format(i+1))
                    need_proj_conv = False # just do this one time
                net = tf.add(res, net, name='res_add_{}'.format(i+1))

            pool = tf.layers.max_pooling2d(net, pool_size, strides=pool_stride, name='pool_{}'.format(name))
        return net, pool
