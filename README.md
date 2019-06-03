# Framework for Image Segmentation/Classification of Satellite Images
1. [Installation of Tensorflow and dependencies](https://github.com/bohaohuang/aml-docs/blob/master/install_tensorflow.ipynb)
2. Examples
    1. [Make collections](./%5Dexamples/examplescript_SetCollectionAndProcessTiles.ipynb)
    2. [Extract patches](./%5Dexamples/examplescript_extractPatches.py)
    3. [Train CNN](./%5Dexamples/examplescript_train_unet_inria.py)
    4. Test CNN: [single image](./%5Dexamples/examplescript_test_pretrained_model.ipynb), [image batches](./%5Dexamples/examplescript_test_pretrained_model_inria.py)
    5. [Train MTL CNN](%5Dexamples/examplescript_train_unet_inria_road_trilabel.py)
3. Supported Network
    1. [U-Net](bohaoCustom/uabMakeNetwork_UNet_with%20changes.py)
    2. [FRRN](./bohaoCustom/uabMakeNetwork_FRRN.py)
    3. [DeepLab V2](./bohaoCustom/uabMakeNetwork_DeepLabV2.py)

# TODO
- [X] uabDataReader: none queue iterator GT order
- [X] uabDataReader: none queue iterator data aug
- [X] Keep track of best model during training
- [X] Optimize model.evaluate()

# Known Bugs
- [X] Loading a model to fine tune will cause problem by calling run() with pretrained_model_dir set in bohaoCustom/uabMakeNetwork_Unet.py
- [X] Unnecessary patches are extracted when input size equals tile size at testing
- [X] Redundant reader initialization in model.evaluate()
- [X] Continue training has some errors in loading functions
- [ ] model.load_weights() reads full absolute path