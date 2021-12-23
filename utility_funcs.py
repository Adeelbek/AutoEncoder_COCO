import os
import PIL
import urllib.request
from urllib.error import HTTPError
import torch
from torch.serialization import save
from torchvision import transforms 
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision.datasets import CIFAR10, CocoDetection
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from utility_classes import *
device = torch.device("cuda:0")



# download pretrained weights from github repos
def download_pretrained_weights(download_params=dict()):
    download_params['pretrained_weights'] = ["cifar10_64.ckpt", "cifar10_128.ckpt", "cifar10_256.ckpt", "cifar10_384.ckpt"]
    os.makedirs(download_params['checkpoint'], exist_ok=True)
    for file_name in download_params['pretrained_weights']:
        file_path = os.path.join(download_params['checkpoint'], file_name)
        if not os.path.isfile(file_path):
            file_url = download_params['base_url'] + file_name
            print("downloading %s..." %file_url)
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the files manually,"
                " or contact the author with the full output including the following error:\n",
                e,)

# loading cifar10 train, val and test dataset 
def cifar10_dataset_loader(cifar_train_params): #saved_path="", batch_size=256, num_workers=4, seed=42, normal_value=0.5):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((cifar_train_params['normal_value'],),
                                                        (cifar_train_params['normal_value'],))])
    #load training dataset 
    train_dataset = CIFAR10(root=cifar_train_params['data_path'], train=True, transform=transform, download=True)
    pl.seed_everything(cifar_train_params['seed'])
    train_set, val_set = data.random_split(train_dataset, [45000, 5000])
    #load testing dataset
    test_set = CIFAR10(root=cifar_train_params['data_path'], train=False, transform=transform, download=True)
    # define a set of dataloaders 
    train_loader = DataLoader(train_set, 
                              batch_size=cifar_train_params['batch_size'], 
                              shuffle=True, 
                              drop_last=True, 
                              pin_memory=True, 
                              num_workers=cifar_train_params['num_workers'])
    val_loader = DataLoader(val_set, 
                            batch_size=cifar_train_params['batch_size'], 
                            shuffle=False, drop_last=False, 
                            num_workers=cifar_train_params['num_workers'])

    test_loader = DataLoader(test_set, 
                             batch_size=cifar_train_params['batch_size'], 
                             shuffle=False, 
                             drop_last=False, 
                             num_workers=cifar_train_params['num_workers'])
    return train_loader, val_loader, test_loader

def get_train_images(cifar_train_params):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((cifar_train_params['normal_value'],),
                                                         (cifar_train_params['normal_value'],))])
    train_dataset = CIFAR10(root=cifar_train_params['data_path'], train=True, transform=transform, download=True)
    return torch.stack([train_dataset[i][0] for i in range(cifar_train_params['num_imgs'])], dim=0)

# load coco train val and test dataset
def coco_dataset_loader(coco_train_params):#path_to_data="", path_to_ann="", path_to_bgbbox="", util_data=100000, train_chunk=80000, val_chunk=15000, test_chunk=5000, batch_size=100, num_workers=4):
    coco_dataset = CocoClsDataset(img_dir=coco_train_params['data_path'], # takes at least 10 sec to execute
                          ann_file=coco_train_params['ann_path'],
                          bg_bboxes_file=coco_train_params['bgbbox_path'])    
    train_data, _ = torch.utils.data.random_split(coco_dataset, [coco_train_params['util_data'], (len(coco_dataset)-coco_train_params['util_data'])])
    train_set, val_set = torch.utils.data.random_split(train_data, [coco_train_params['train_chunk'], 
                                                      (coco_train_params['util_data']-coco_train_params['train_chunk'])])
    val_set, test_set = torch.utils.data.random_split(val_set, [coco_train_params['val_chunk'], coco_train_params['test_chunk']])
    print(f"Number of training data {len(train_set)}\nNumber of validation data {len(val_set)}\nNumber of test data {len(test_set)}\n")
    train_loader = DataLoader(train_set, 
                              batch_size=coco_train_params['batch_size'], 
                              shuffle=True, 
                              drop_last=True, 
                              pin_memory=True, 
                              num_workers=coco_train_params['num_workers'])
    val_loader = DataLoader(val_set, 
                            batch_size=coco_train_params['batch_size'], 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers=coco_train_params['num_workers'])
    test_loader = DataLoader(test_set, 
                             batch_size=coco_train_params['batch_size'], 
                             shuffle=False, 
                             drop_last=False, 
                             num_workers=coco_train_params['num_workers'])
    return train_loader, val_loader, test_loader

# run autoencoder model using cifar10 dataset 
def train_cifar_autoencoder(latent_dim,cifar_train_params):
    train_loader, val_loader, test_loader = cifar10_dataset_loader(cifar_train_params)
    trainer = pl.Trainer(default_root_dir=os.path.join(cifar_train_params['checkpoint'],"cifar10_%i" %latent_dim),
                         gpus = 1, 
                         max_epochs=500, 
                         callbacks=[ModelCheckpoint(save_weights_only = True),
                                    GenerateCallback(get_train_images(cifar_train_params), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None 
    
    pretrained_filename = os.path.join(cifar_train_params['checkpoint'], "cifar10_%i.ckpt" %latent_dim)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = CifarAutoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = CifarAutoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

# run autoencoder model to experiment with normalized and classless coco detection dataset

def train_coco_autoencoder(latent_dim, coco_train_params):
    print("Latent Dimension is ", latent_dim)
    train_loader, val_loader, test_loader = coco_dataset_loader(coco_train_params)
    
    trainer = pl.Trainer(default_root_dir=os.path.join(coco_train_params['checkpoint'], "coco80_%i" %latent_dim),
                         gpus=1, 
                         max_epochs=300, 
                         callbacks=[ModelCheckpoint(save_weights_only=True), 
                                    GenerateCallback(get_train_images(coco_train_params), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    
    pretrained_filename = os.path.join(coco_train_params['checkpoint'], "coco80_%i.ckpt" %latent_dim)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading ...")
        model = CocoAutoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = CocoAutoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Testing the model on validation data
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test":test_result, "val":val_result}
    return model, result







