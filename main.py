import os
import urllib.request
from urllib.error import HTTPError
#plotting tools
import matplotlib.pyplot as plt
from pytorch_lightning.core.hooks import CheckpointHooks
import seaborn as sns
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgb
from PIL import Image
from torch.random import seed
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision 
from torchvision.datasets import CIFAR10, CocoDetection
from torchvision import transforms
from pycocotools.coco import COCO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0")
from utility_funcs import *
from utility_classes import *
pl.seed_everything(42)
CIFARDATASET_PATH = "data/"
COCODATASET_PATH = "/home/odilbek/Desktop/Pytorch_Exer/data/COCO/"
CHECKPOINT_PATH = 'saved_models/Avtoencoder/'

def visualize_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 4.5))
    plt.title("Reconstructed from %i latents" % (model.hparams.latent_dim))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

def plot_graph(latent_dims, val_scores):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(latent_dims, val_scores, "--", 
             color="#000", marker="*", 
             markeredgecolor="#000", 
             markerfacecolor="y", 
             markersize=16)
    plt.xscale("log")
    plt.xticks(latent_dims, labels=latent_dims)
    plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    plt.xlabel("Latent dimensionality")
    plt.ylabel("Reconstruction error")
    plt.minorticks_off()
    plt.ylim(0, 100)
    plt.show()

def compare_imgs(img1, img2, title_prefix=""):
    loss = F.mse_loss(img1, img2, reduction="sum")
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), 
                                       nrow=2, normalize=True, 
                                       range=(-1,1))
    grid = grid.permute(1,2,0)
    plt.figure(figsize=(4,2))
    plt.title(f"{title_prefix} Loss : {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model_params = dict()
    model_dict = dict()
    data_name = 'COCO'
    if data_name == 'COCO':
        model_params = {'data_path' : PATH_TO_DATA, 
                       'ann_path' : PATH_TO_ANN, 
                       'bgbbox_path' : PATH_TO_BGBBOX,
                       'util_data' : 100000,
                       'train_chunk' : 80000,
                       'val_chunk' : 15000,
                       'test_chunk' : 5000,
                       'batch_size' : 100, 
                       'num_workers' : 4, 
                       'checkpoint' : CHECKPOINT_PATH, 
                       'latent_dims' : [384, 512],
                       'seed' : 42, 
                       'normal_value' : 0.5,
                       'num_imgs' : 8,
                       }
    elif data_name == 'CIFAR':
        model_params = {'data_path' : CIFARDATASET_PATH,
                        'train_chunk' : 45000,
                        'val_chunk' : 5000,
                        'batch_size' : 256,
                        'num_workers' : 4,
                        'normal_value' : 0.5,
                        'checkpoint' : CHECKPOINT_PATH,
                        'latent_dims' : [64, 128, 256, 384],
                        'seed' : 42,
                        'normal_value' : 0.5,
                        'num_imgs' : 8 
                        }

    dparams = {'pretrained_weights' : ["cifar10_64.ckpt", 
                                       "cifar10_128.ckpt", 
                                       "cifar10_256.ckpt", 
                                       "cifar10_384.ckpt"],
                'checkpoint' : CHECKPOINT_PATH,
                'base_url' : "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial9/",
               }
               
    
    #train_loader, val_loader, test_loader = None
    if data_name == 'CIFAR':
        download_pretrained_weights(dparams)
        train_loader, val_loader, test_loader = cifar10_dataset_loader(model_params)
        for dim in model_params['latent_dims']:
            model_cifar, result_cifar = train_cifar_autoencoder(dim,model_params)
            model_dict[dim] = {"model" : model_cifar, "result" : result_cifar}
        _latent_dims = sorted(k for k in model_dict)
        val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in _latent_dims]
        plot_graph(_latent_dims, val_scores) 
        input_images = get_train_images(model_params)
        for latent in model_dict:
            visualize_reconstructions(model_dict[latent]["model"], input_images)  
    elif data_name == 'COCO':
        train_loader, val_loader, test_loader = coco_dataset_loader(model_params)
        for lat_dim in model_params['latent_dims']:
            model_coco, result_coco = train_coco_autoencoder(lat_dim, model_params)
            model_dict[lat_dim] = {"model" : model_coco, "result" : result_coco}
    else:
        print("No data ID is indicated")





