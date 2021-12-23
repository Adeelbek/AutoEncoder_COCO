import os
from PIL import Image
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
device = torch.device("cuda:0")

PATH_TO_DATA = '/home/odilbek/Desktop/Pytorch_Exer/data/COCO/train2017'
PATH_TO_ANN = '/home/odilbek/Desktop/Pytorch_Exer/data/COCO/annotations/instances_train2017.json'
PATH_TO_BGBBOX = '/home/odilbek/Desktop/Pytorch_Exer/data/COCO/annotations/coco_train_bg_bboxes.log'

"""
COCO Data loader class which decreases the number of classes and adds
background classes
"""
class CocoClsDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, bg_bboxes_file):
        self.ann_file = PATH_TO_ANN
        self.img_dir = PATH_TO_DATA
        self.coco = COCO(self.ann_file)
        self.bg_bboxes_file = bg_bboxes_file
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
        cat_ids = self.coco.getCatIds()
        categories = self.coco.dataset['categories']
        self.id2cat = dict()
        for category in categories:
            self.id2cat[category['id']] = category['name']
        self.id2cat[0] = 'background'
        self.id2label = {category['id']:label + 1 for label, category in enumerate(categories)}
        self.id2label[0] = 0
        self.label2id = {v:k for v,k in self.id2label.items()}
        tmp_ann_ids = self.coco.getAnnIds()
        self.ann_ids = []
        for ann_id in tmp_ann_ids:
            ann = self.coco.loadAnns([ann_id])[0]
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            if ann['area'] <= 0 or w < 1 or h < 1 or ann['iscrowd']:
                continue
            self.ann_ids.append(ann_id)
        self.bg_anns = self._load_bg_anns()
        self._cal_num_dict()
        print('total_length of dataset:', len(self))
        
    def _cal_num_dict(self):
        self.num_dict = {}
        for ann_id in self.ann_ids:
            ann = self.coco.loadAnns([ann_id])[0]
            cat = self.id2cat[ann['category_id']]
            num = self.num_dict.get(cat, 0)
            self.num_dict[cat] = num + 1
        self.num_dict['background'] = len(self.bg_anns)
    
    def _load_bg_anns(self):
        assert os.path.exists(self.bg_bboxes_file)
        bg_anns = []
        with open(self.bg_bboxes_file, 'r') as f:
            line = f.readline()
            while line:
                if line.strip() == '':
                    break
                file_name, num = line.strip().split()
                for _ in range(int(num)):
                    bbox = f.readline()
                    bbox = bbox.strip().split()
                    bbox = [float(i) for i in bbox]
                    w = bbox[2] - bbox[0] + 1
                    h = bbox[3] - bbox[1] + 1
                    bbox[2], bbox[3] = w, h
                    ann = dict(
                        file_name=file_name,
                        bbox=bbox)
                    bg_anns.append(ann)
                line = f.readline()
        return bg_anns
    
    def __len__(self):
        return len(self.ann_ids) + len(self.bg_anns)


    def __getitem__(self, idx):
        if idx < len(self.ann_ids):
            ann = self.coco.loadAnns([self.ann_ids[idx]])[0]

            cat_id = ann['category_id']
            label = self.id2label[cat_id]

            img_meta = self.coco.loadImgs(ann['image_id'])[0]
            img_path = os.path.join(self.img_dir, img_meta['file_name'])
        else:
            ann = self.bg_anns[idx - len(self.ann_ids)]

            label = 0

            img_path = os.path.join(self.img_dir, ann['file_name'])

        img = Image.open(img_path).convert('RGB')
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        img = img.crop((x, y, x + w - 1, y + h - 1))

        # save_img = img.resize((224, 224), Image.BILINEAR)
        # save_img.save('test.jpg')

        try:
            img = self.transform(img)
        except:
            print(img.mode)
            exit(0)
        if label != 0:
            label = 1
        tmp_label = torch.zeros(2)
        tmp_label[label] = 1
        return img, tmp_label

"""
Encoder class that use CIFAR10 dataset (32x32 pxls) as an input images 
"""
class CifarEncoder(nn.Module):
    def __init__(self, 
                num_input_channels : int, 
                base_channel_size : int, 
                latent_dim : int, 
                act_fn : object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # org_size X org_size
            #nn.BatchNorm2d(c_hid), # can be used but from small input images it's not necessary 
            act_fn(), 
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), # dimension won't change 
            act_fn(), 
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # org_size/2 X org_size/2
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1), # dimension won't change
            act_fn(), 
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # org_size/4 X org_size/4
            act_fn(), 
            nn.Flatten(), 
            nn.Linear(2*16*c_hid, latent_dim)
        )
        
    def forward(self,x):
        return self.net(x)

"""
Decoder class that produces CIFAR10 dataset size (32x32 pxls) in the end
"""
class CifarDecoder(nn.Module):
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid), 
            act_fn()
        )
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 ==> 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(), 
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), #8x8==>16x16
            act_fn(), 
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), 
            act_fn(), 
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), #16x16==>32x32
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

"""
Autoencoder uses CIFAR10 dataset for train validation and test
"""
class CifarAutoencoder(pl.LightningModule):
    def __init__(self, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 encoder_class : object = CifarEncoder,
                 decoder_class : object = CifarDecoder, 
                 num_input_channels : int = 3,
                 width : int = 32, 
                 height : int = 32
        ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, batch, mode="mse"):
        x, _ = batch
        x_hat = self.forward(x)
        if mode == "mse":
            loss = F.mse_loss(x, x_hat, reduction="none")
        else:
            loss = F.kl_div(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode = 'min', 
                                                         factor=0.2, 
                                                         patience=20, 
                                                         min_lr=5e-5)
        return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"val_loss"}
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
        return loss
"""
Encoder that takes a modefoed COCO dataset as an input to extract more representation from bigger input images (64x64, 128x128, 256x256)
"""


class CocoEncoder(nn.Module):
    
    def __init__(self, 
                num_input_channels : int, 
                base_channel_size : int, 
                latent_dim : int, 
                act_fn : object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 256x256 ==> 128x128
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), 
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2), # 128x128 ==> 64x64
            act_fn(), 
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), # 
            act_fn(), 
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2), # 64x64 ==> 32x32
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),#
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 ==> 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), # dimension won't change
            act_fn(), 
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 ==> 8x8
            act_fn(), 
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1), #
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 ==> 4x4
            act_fn(),
            nn.Flatten(), 
            nn.Linear(2*16*c_hid, latent_dim)
        )
        
    def forward(self,x):
        return self.net(x)

"""
Decoder which decodes encoded large images from high definition COCO dataset 
"""
class CocoDecoder(nn.Module):
    def __init__(self, 
                 num_input_channels : int, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 act_fn : object = nn.GELU):
        
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid), 
            act_fn()
        )
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4==>8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1), 
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 ==> 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(), 
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 ==> 32x32
            act_fn(), 
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1), 
            act_fn(), 
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 32x32 ==> 64x64
            act_fn(), 
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 64x64 ==> 128x128
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), #128x128 ==> 256x256
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
"""
CocoAutoencoder to combine encoder and decoder operation 
"""
class CocoAutoencoder(pl.LightningModule):
    def __init__(self, 
                 base_channel_size : int, 
                 latent_dim : int, 
                 encoder_class : object = CocoEncoder,
                 decoder_class : object = CocoDecoder, 
                 num_input_channels : int = 3,
                 width : int = 256, 
                 height : int = 256
        ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, batch, mode="mse"):
        x, _ = batch
        x_hat = self.forward(x)
        if mode == "mse":
            loss = F.mse_loss(x, x_hat, reduction="none")
        else:
            loss = F.kl_div(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode = 'min', 
                                                         factor=0.2, 
                                                         patience=20, 
                                                         min_lr=5e-5)
        return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"val_loss"}
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
        return loss



"""
The class which generates a callback from the entire process
"""
class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs
        
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            imgs = torch.stack([input_imgs, reconst_imgs], dim = 1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)

