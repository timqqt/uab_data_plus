"""
Created on 06/01/2019
Fanjie Kong
"""
import sys
import os
import numpy as np
import time
sys.path.append('/hpchome/collinslab/fk43/FanjieKong/scripts/uab')
import tensorflow as tf
import uabCrossValMaker
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import uab_collectionFunctions
# from bohaoCustom import uabMakeNetwork_dual_TernausV2 as uabMakeNetwork_UnetMTL
from bohaoCustom import uabMakeNetwork_DeepLabV2     # you can also import other models from bohaoCustom

# settings
decay_step = 50                 # learn rate dacay after 60 epochs
decay_rate = 0.1                # learn rate decay to 0.1*before
GPU = 0                         # which gpu to use, remember to set to None if you don't know which one to use
batch_size = 20                  # mini-batch size, not necessarily equal to the batch size in training
input_size = (286, 286)         # input size to NN, same as extracted patch size, no need to be the same as training
tile_size = (572, 572)        # size of the building image
chip_size = (286, 286)          # image will be extracted to this size
n_train = 2000                  # number of samples per epoch
n_valid = 1000                  # number of samples every validation step
source_num = 1
epochs = 100                    # total number of epochs to run
learn_rate = 2e-4               # learning rate

tf.reset_default_graph()        # reset the graph before you start
# this is where I have my pretrained model
model_dir = r'/work/fk43/Models/DeeplabV3_fair_setting_Joint_training_syn_and_inria_without_d_e_Deeplabv2_ratio_6_1_1_PS(286, 286)_BS7_EP100_LR5e-05_DS50_DR0.1_SFN32'
# make the model, same as training
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')

model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X': X, 'Y': y},
                                          trainable=mode,                       # control if you're training or not
                                          input_size=input_size,                 # input size to NN, same as extracted
                                          batch_size=batch_size,                # mini-batch size
                                          learn_rate=learn_rate,                # learning rate
                                          decay_step=decay_step,                # learn rate decay after 60 epochs
                                          decay_rate=decay_rate,                # learn rate decay to 0.1*before
                                          epochs=epochs)    # number of filters at the first layer

model.create_graph('X', class_num=2)    
# ------------------------------------------Dataset DeepGlobe Testing Set---------------------------------------------#
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
                 cSize=input_size,                           # patch size as 572*572
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
file_list_test_Vegas = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(2310, 3850)],  filter_list = ['Shanghai', 'Khartoum', 'Paris', 'random'])
file_list_test_Shanghai = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(2749, 4581)],  filter_list = ['Vegas', 'Khartoum', 'Paris', 'random'])
file_list_test_Paris= uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(688, 1145)],  filter_list = ['Shanghai', 'Khartoum', 'Vegas', 'random'])
file_list_test_Khartoum= uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(607, 1012)],  filter_list = ['Shanghai', 'Paris', 'Vegas', 'random'])

file_list_valid = list(np.concatenate((file_list_test_Vegas, file_list_test_Shanghai, file_list_test_Paris, file_list_test_Khartoum), axis=0))

with tf.name_scope('image_loader'):
    # no augmentation needed for validation
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_valid, chip_size, tile_size,
                                                      batch_size, dataAug=' ', block_mean=np.append([0], img_mean))


print(len(file_list_valid))
# assert False
# train
start_time = time.time()

model.train_config('X', 'Y', len(file_list_valid), len(file_list_valid), chip_size, uabRepoPaths.modelPath, loss_type='xent', mode='B & Bg')
model.run(train_reader=dataReader_valid,
          continue_dir=model_dir,        # train from scratch, no need to load pre-trained model
          isOneShotTest=True,
          img_mean=img_mean,
          verb_step=100,                    # print a message every 100 step(sample)
          save_epoch=5,                     # save the model every 5 epochs
          gpu=GPU,
          tile_size=tile_size,
          patch_size=chip_size)


duration = time.time() - start_time
print('duration {:.2f} hours'.format(duration/60/60))





