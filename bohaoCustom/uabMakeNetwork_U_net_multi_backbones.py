"""
Created on 06/01/2019
Fanjie Kong

Reference github:
https://github.com/qubvel/segmentation_models
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
from bohaoCustom.segmentation_models import Unet
from bohaoCustom.segmentation_models.segmentation_models.backbones import get_preprocessing
from bohaoCustom.segmentation_models.segmentation_models.losses import bce_jaccard_loss
from bohaoCustom.segmentation_models.segmentation_models.metrics import iou_score

class Unet_MTbackbones(uabMakeNetwork_UNet.UnetModel):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, backbone_name='resnet34'):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = backbone_name + 'U_net'
        self.backbone = backbone_name
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
        self.update_ops = None
        self.config = None

    def create_graph(self, x_name, class_num):
        print('-'*10 + 'The backbone is '+self.backbone+10*'-')
        self.class_num = class_num
        model = Unet(self.backbone, classes=self.class_num, activation=None, encoder_weights='imagenet')
        self.pred = model(self.inputs[x_name])
        self.output = tf.nn.softmax(self.pred)

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


class Unet_MTbackbones_MTratio(Unet_MTbackbones):
    def __init__(self, inputs, trainable, input_size, model_name='', dropout_rate=0.2,
                 learn_rate=1e-4, decay_step=60, decay_rate=0.1, epochs=100,
                 batch_size=5, start_filter_num=32, backbone_name='resnet34',source_num=2, source_control=[4, 1]):
        network.Network.__init__(self, inputs, trainable, dropout_rate,
                                 learn_rate, decay_step, decay_rate, epochs, batch_size)
        self.name = backbone_name + 'U_net'
        self.backbone = backbone_name
        self.model_name = self.get_unique_name(model_name)
        self.sfn = start_filter_num
        self.learning_rate = None
        self.valid_cross_entropy = tf.placeholder(tf.float32, [])
        self.valid_iou = tf.placeholder(tf.float32, [])
        self.valid_images = tf.placeholder(tf.uint8, shape=[None, input_size[0],
                                                            input_size[1] * 3, 3], name='validation_images')
        self.channel_axis = 3
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