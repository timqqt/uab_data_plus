"""
Created on 06/01/2019
Fanjie Kong
"""
import sys
sys.path.append('/hpchome/collinslab/fk43/FanjieKong/scripts/uab')
import os
import time
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_U_net_multi_backbones as uabMakeNetwork_U_net     # you can also import other models from bohaoCustom

# experiment settings
chip_size = (288, 288)          # image will be extracted to this size
tile_size = (5000, 5000)        # size of the original image
chip_size2 = (288, 288)          # image will be extracted to this size
tile_size2 = (572, 572)
batch_size = 7                  # mini-batch size
class_num = 2
learn_rate = 5e-5               # learning rate
decay_step = 50                 # learn rate dacay after 60 epochs
decay_rate = 0.1                # learn rate decay to 0.1*before
epochs = 100                    # total number of epochs to run
start_filter_num = 32           # the number of filters at the first layer
n_train = 3000                  # number of samples per epoch
n_valid = 800                   # number of samples every validation step
GPU = 0                         # which gpu to use, remember to set to None if you don't know which one to use
source_num = 2
source_control = [6, 1]
model_name = 'Joint_training_syn_and_inria_with_d_and_e_resUnet_ratio_6_1_1'
pretrained_model_dir = None
backbone = 'resnet34'

# make network
# define place holder
X = tf.placeholder(tf.float32, shape=[None, chip_size[0], chip_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, chip_size[0], chip_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')         # This controls if you'll update weights or not
                                                    # Set this True when training
model = uabMakeNetwork_U_net.Unet_MTbackbones_MTratio({'X': X, 'Y': y},
                                          trainable=mode,                       # control if you're training or not
                                          input_size=chip_size,                 # input size to NN, same as extracted
                                          model_name=model_name,                                      # patch size
                                          batch_size=batch_size,                # mini-batch size
                                          learn_rate=learn_rate,                # learning rate
                                          decay_step=decay_step,                # learn rate decay after 60 epochs
                                          decay_rate=decay_rate,                # learn rate decay to 0.1*before
                                          epochs=epochs,
                                          source_num=source_num,
                                          backbone_name=backbone,
                                          start_filter_num=start_filter_num)    # number of filters at the first layer
model.create_graph('X', class_num=class_num)                                            # TensorFlow will now draw the graph


####Inira 1 ########
# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info
print(blCol.readMetadata())                         # now inria collection has 4 channels, the last one is GT with (0,1)

# extract patches
extrObj = uab_DataHandlerFunctions.\
    uabPatchExtr([0, 1, 2, 4],                              # extract all 4 channels
                 cSize=chip_size,                           # patch size as 572*572
                 numPixOverlap=int(model.get_overlap()),    # overlap as 184
                 extSave=['jpg', 'jpg', 'jpg', 'png'],      # save rgb files as jpg and gt as png
                 isTrain=True,                              # the extracted patches are used for training
                 gtInd=3,                                   # gt is the 4th(count from 0) in the list of indices
                 pad=int(model.get_overlap() / 2))          # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
# use first 5 tiles for validation
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(11, 37)],filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck',  'vienna_s'])
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 11)], filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck',  'vienna_s'])

with tf.name_scope('image_loader'):
    # GT has no mean to subtract, append a 0 for block mean
    dataReader_train1 = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      source_control[0], dataAug='flip,rotate',
                                                      block_mean=np.append([0], img_mean))
    # no augmentation needed for validation
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_valid, chip_size, tile_size,
                                                      batch_size, dataAug=' ', block_mean=np.append([0], img_mean))

#### Synthetic #####
# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('deepglobe')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info
print(blCol.readMetadata())                         # now inria collection has 4 channels, the last one is GT with (0,1)

# extract patches
extrObj = uab_DataHandlerFunctions.\
    uabPatchExtr([0, 1, 2, 4],                              # extract all 4 channels
                 cSize=chip_size,                           # patch size as 572*572
                 numPixOverlap=int(model.get_overlap()),    # overlap as 184
                 extSave=['jpg', 'jpg', 'jpg', 'png'],      # save rgb files as jpg and gt as png
                 isTrain=True,                              # the extracted patches are used for training
                 gtInd=3,                                   # gt is the 4th(count from 0) in the list of indices
                 pad=int(model.get_overlap() / 2))          # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
# use first 5 tiles for validation

#file_list_train_Syn= uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 5000)], filter_list = ['Shanghai', 'Paris', 'Vegas', 'Khartoum','random_city_s_d_', 'random_city_s_e_'])
file_list_train_Syn= uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 5000)], filter_list = ['Shanghai', 'Paris', 'Vegas', 'Khartoum'])


##############################

with tf.name_scope('image_loader'):
    # GT has no mean to subtract, append a 0 for block mean
    dataReader_train2 = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train_Syn, chip_size2, tile_size2,
                                                       source_control[1], dataAug='flip,rotate',
                                                       block_mean=np.append([0], img_mean))

# train
start_time = time.time()

model.train_config('X', 'Y', n_train, n_valid, chip_size, uabRepoPaths.modelPath, loss_type='xent', mode='B & Bg')
model.run(train_reader=[dataReader_train1, dataReader_train2],
          valid_reader=dataReader_valid,
          pretrained_model_dir=pretrained_model_dir,        # train from scratch, no need to load pre-trained model
          isTrain=True,
          img_mean=img_mean,
          verb_step=100,                    # print a message every 100 step(sample)
          save_epoch=5,                     # save the model every 5 epochs
          gpu=GPU,
          tile_size=tile_size,
          patch_size=chip_size)

duration = time.time() - start_time
print('duration {:.2f} hours'.format(duration/60/60))
