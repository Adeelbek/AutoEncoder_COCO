# Shallow and Deep AutoEncoder Models with Various Input Images (32x32) and (255x255)
These are two auto-encoder models with various size of hidden layers. In this project, we also utilized MS COCO dataset for training auto-encoder. To be able to use MS COCO for train auto-encoder in pytorch environment, we modefied the standard annotaion files to make them match only two classes, background and object classes. Overall number of images in COCO set is 1.65 M. User can freely change the size input images and add additional CNN layers to see the hidden layers effect on final performance of Auto-encoder. Currently, we applied only to loss (criterion) functions (MSE and KL Convergence) to calculate reconstruction loss. In the next steps, we plan to add SSIM loss calculation function to provide significant difference between normal and abnormal loss values. 
# Difference of models
The one who want to utilize shallow auto-encoder (3 hidden-layers) with CIFAR10 (32x32) images, please use `CIFARAutoencoder` class inside 'utility_classes.py' file and download pretrained weights for various latent size from following links:
 - pretrained weights for `latent_dim=64` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_64.ckpt >here</a>
 - pretrained weights for `latent_dim=128` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_128.ckpt>here</a>
 - pretrained weights for `latent_dim=256` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_256.ckpt>here</a>
 - pretrained weights for `latent_dim=384` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_384.ckpt>here</a><br>

The one who wants to use deep auto-encoder (6 hidden layers) with MS COCO HD images, please use `COCOAutoencoder` class inside 'utility_classes.py' file and download modified COCO dataset from <a href=https://drive.google.com/file/d/11XYpqGEJMCphKiD6z3_NKrj5CRLwg8S9/view?usp>here</a>. Pretrained weights for COCO dataset have not been provided yet. However, user can freely start training of deep model from scratch for various latent dimensions starting from `latent_dim = 128,256,384,512` so on. As longer the latent dimension as higher the performance of auto-encoder. 
# Reconstruction loss for different latent dimensions
The experiment over shallow autoencoder model proves that larger latent dimension improves reconstruction ability of auto-encoder and reduces the loss function. Result of experiment can be seen in following graph. 
<img src="https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/Figure_1.png" width=510>
# Running the project
To run auto-encoder, user needs to select the data which will be used to train auto-encoder (whether shallow or deep model). Default data name (`data_name`) in `main.py` file is set `CIFAR`, (`data_name=CIFAR`). To be able to fully explore deep auto-encoder model, please change `data_name` to `COCO` as `data_name=COCO` and run `python main.py` in home directory `../../Autoencoder/`. Followings are environmental dependencies:
```
 - pytorch>=1.7
 - python>=3.8
 - pytorch_lightning>=1.5
 - cython
```
**Remark** *If you face any issue while inferencing or training on `COCO` based deep auto-encoder, please try to run `jupyter-notebook` file in the same environment. If you keep getting errors, then try to make sure you have the same dependencies as shown above list*.  
