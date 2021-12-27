# Shallow and Deep AutoEncoder Models with Various Input Images (32x32) and (255x255)
These are two auto-encoder models with various size of hidden layers. In this project, we also utilized MS COCO dataset for training auto-encoder. To be able to use MS COCO for train auto-encoder in pytorch environment, we modefied the standard annotaion files to make them match only two classes, background and object classes. Overall number of images in COCO set is 1.65 M. User can freely change the size input images and add additional CNN layers to see the hidden layers effect on final performance of Auto-encoder. Currently, we applied only to loss (criterion) functions (MSE and KL Convergence) to calculate reconstruction loss. In the next steps, we plan to add SSIM loss calculation function to provide significant difference between normal and abnormal loss values. 
# Difference of models
The one who wnat to utilize auto-encoder with CIFAR10 (32x32) images, please use `CIFAR_AutoEncoder` class and download pretrained weights for various latent size from following links:
 - pretrained weights for `latent_dim=64` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_64.ckpt >here</a>
 - pretrained weights for `latent_dim=128` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_128.ckpt>here</a>
 - pretrained weights for `latent_dim=256` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_256.ckpt>here</a>
 - pretrained weights for `latent_dim=384` is <a href=https://github.com/Adeelbek/AutoEncoder_COCO/releases/download/AutoEncoder_COCO/cifar10_384.ckpt>here</a>
