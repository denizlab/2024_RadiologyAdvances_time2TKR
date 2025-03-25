# ==============================================================================
# Copyright (C) 2024 Ozkan Cigdem, Shengjia Chen, Chaojie Zhang, 
# Kyunghyun Cho, Richard Kijowski, Cem M Deniz

# This file is part of 2024_RadiologyAdvances_time2TKR
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
import argparse
import cv2
import numpy as np
import pandas as pd
import os
import random
from collections import defaultdict
import skimage.transform
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torch.backends.cudnn as cudnn
from timm.scheduler import create_scheduler
from timm.utils import get_state_dict
from timm.utils import ModelEmaV2 as ModelEma
from pathlib import Path
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

from augmentation import *
import twist_utils
from objective import *
from resnet_3d_custom import get_resnet3d

parser = argparse.ArgumentParser('Self-Supervised', add_help=False)
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--dim', default=1000, type=int)
parser.add_argument('--hid_dim', default=4096, type=int) 
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--warmup_epochs', default=100, type=int)
parser.add_argument('--save_every_n_epoch', default=50, type=int)
parser.add_argument('--num_workers', default=40, type=int)
parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--output_dir', default='./Results/', help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--seed', default=2022, type=int)


parser.add_argument('--tau', type=float, default=1.0, metavar='LR')
parser.add_argument('--EPS', type=float, default=1e-5, help='episillon')
parser.add_argument('--amp', default=0, type=int)
parser.add_argument('--crops_interact_style', type=str, default='sparse')

#OAI data related
parser.add_argument('--train_df_path', type=str, default= '/gpfs/data/denizlab/Users/cz2064/OAI_CL_test/TWIST/Metadata/OAI_Progression_CaseControl/SAG_IW_TSE/SAG-IW-TSE_unsupervised_train.csv')
parser.add_argument('--val_df_path', type=str, default= '/gpfs/data/denizlab/Users/cz2064/OAI_CL_test/TWIST/Metadata/OAI_Progression_CaseControl/SAG_IW_TSE/SAG-IW-TSE_unsupervised_val.csv')
parser.add_argument('--t2map_images', type=bool, default=False)
parser.add_argument('--num_slices_to_load', type=int, default=36)
parser.add_argument('--image_size_h', type=int, default=384)
parser.add_argument('--image_size_w', type=int, default=384)
parser.add_argument('--image_size_d', type=int, default=36)
parser.add_argument('--local_crops_number', type=int, default=6)
parser.add_argument('--min1', type=float, default=0.9, metavar='LR')
parser.add_argument('--max1', type=float, default=1.3, metavar='LR')
parser.add_argument('--min2', type=float, default=0.3, metavar='LR')
parser.add_argument('--max2', type=float, default=0.7, metavar='LR')
parser.add_argument('--aug_degrees', type=int, default=10)
parser.add_argument('--aug_GaussianNoise', type=bool, default=False)
parser.add_argument('--aug_GaussianBlur', type=bool, default=True)
parser.add_argument('--aug_RandomFlip', type=bool, default=True)
parser.add_argument('--aug_MultiCrop', type=bool, default=False)
parser.add_argument('--local_crop_size_h', type=int, default=128)
parser.add_argument('--local_crop_size_w', type=int, default=128)
parser.add_argument('--local_crop_size_d', type=int, default=12)


args = parser.parse_args()

args.global_crops_scale = (args.min1, args.max1)
args.local_crops_scale = (args.min2, args.max2)
args.image_size = (args.image_size_d, args.image_size_h, args.image_size_w)
args.local_crop_size = (args.local_crop_size_d, args.local_crop_size_h, args.local_crop_size_w)




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None
setup_seed(args.seed)


class OaiDataSet_Unsupervised(Dataset):
    def __init__(self, args, data_df_path, augmentation=True, train=True, labels_to_load=[]):
        self.args = args
        self.augmentation = augmentation
        self.labels_to_load = labels_to_load
        self.data_df = pd.read_csv(data_df_path)
        self.train = train

    def __getitem__(self, index):
        args = self.args
        image_path = self.data_df.iloc[index]['File_Name']
        if args.t2map_images:
            image = read_t2mapping_hdf5(image_path, args.num_slices_to_load)
        else:
            image = read_iw_tse_hdf5(image_path, args.num_slices_to_load)
        
        label = []
        for label_name in self.labels_to_load:
            label.append(self.data_df.iloc[index][label_name])
        
        crops_list = []
        if self.augmentation:
            if args.aug_MultiCrop and self.train: # 2 global transforms and N local transforms
                global_transform_1 = RandomRescale2D(args.global_crops_scale)
                global_transform_2 = RandomRescale3D(args.global_crops_scale)
                local_transform = RandomRescale3D(args.local_crops_scale)
                
                crop_1 = global_transform_1(image)
                crop_1 = SimpleRotate(args.aug_degrees)(crop_1)
                crop_1 = RandomCrop(args.image_size)(crop_1)
                
                crop_2 = global_transform_1(image)
                crop_2 = SimpleRotate(args.aug_degrees)(crop_2)
                crop_2 = RandomCrop(args.image_size)(crop_2)
                crops_list.append(crop_1)
                crops_list.append(crop_2)
                
                crop_n = local_transform(image)
                crop_n = SimpleRotate(args.aug_degrees)(crop_n)
                for crop_num in range(args.local_crops_number):
                    crop_n = RandomCrop(args.local_crop_size)(crop_n)
                    crops_list.append(crop_n)

            else:
                global_transform_1 = RandomRescale2D(args.global_crops_scale)
                global_transform_2 = RandomRescale3D(args.global_crops_scale)
                crop_1 = global_transform_1(image)
                crop_1 = SimpleRotate(args.aug_degrees)(crop_1)
                crop_1 = RandomCrop(args.image_size)(crop_1)
                crop_2 = global_transform_1(image)
                crop_2 = SimpleRotate(args.aug_degrees)(crop_2)
                crop_2 = RandomCrop(args.image_size)(crop_2)
                crops_list.append(crop_1)
                crops_list.append(crop_2)

            transform_list = []
            transform_list.append(Standardize())
            if args.aug_RandomFlip:
                transform_list.append(RandomFlip())
            if args.aug_GaussianBlur:
                transform_list.append(GaussianBlur())
        
            transform_list.append(ToTensor())
            
            if args.aug_GaussianNoise:
                transform_list.append(GaussianNoise())
            transform_composed = transforms.Compose(transform_list)
            
            return_list = []
            for each_crop in crops_list:
                new_crop = transform_composed(each_crop)
                return_list.append(new_crop)
            
        else: # no augmentation
            val_transform = CenterCrop(args.image_size)
            crop = val_transform(image)
            crops_list.append(crop)
            transform_list = []
            transform_list.append(Standardize())
            transform_list.append(ToTensor())
            transform_composed = transforms.Compose(transform_list)
            return_list = []
            for each_crop in crops_list:
                new_crop = transform_composed(each_crop)
                return_list.append(new_crop)

        return return_list, torch.FloatTensor(label), torch.LongTensor([index])

    def __len__(self):
        return len(self.data_df)

    
    
class resnet_model(nn.Module):
    def __init__(self):
        super(resnet_model, self).__init__()
        self.backbone = get_resnet3d('TSE')
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc = nn.Linear(in_features=256, out_features=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),x.size(1))
        x = self.fc(x)
        return x
resnet = resnet_model() 


class ProjectionHead(nn.Module):
    def __init__(self, args, feature_dim=2048):
        super(ProjectionHead, self).__init__()

        norm = nn.BatchNorm1d(args.dim)

        batchnorm = nn.BatchNorm1d

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, args.hid_dim, bias=True),
            batchnorm(args.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=0.0),

            nn.Linear(args.hid_dim, args.hid_dim, bias=True),
            batchnorm(args.hid_dim),
            nn.ReLU(),
        )

        last_linear = nn.Linear(args.hid_dim, args.dim, bias=True)
        self.last_linear = last_linear
        self.norm = norm

    def reg_gnf(self, grad):
        self.gn_f = grad.abs().mean().item()

    def reg_gnft(self, grad):
        self.gn_ft = grad.abs().mean().item()

    def forward(self, x):
        x = self.projection_head(x)
        f = self.last_linear(x)
        ft = self.norm(f)
        if self.train and x.requires_grad:
            f.register_hook(self.reg_gnf)
            ft.register_hook(self.reg_gnft)
        self.f_column_std = f.std(dim=0, unbiased=False).mean()
        self.f_row_std    = f.std(dim=1, unbiased=False).mean()
        self.ft_column_std = ft.std(dim=0, unbiased=False).mean()
        self.ft_row_std    = ft.std(dim=1, unbiased=False).mean()
        return ft
    
class TWIST(nn.Module):
    def __init__(self, args):
        super(TWIST, self).__init__()
        self.backbone = resnet_model()
        self.feature_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()
        self.projection_heads = ProjectionHead(args,feature_dim=self.feature_dim)
    
    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        out = self.projection_heads(output)
        return out

    def backbone_weights(self):
        return self.backbone.state_dict()
    
    
    
    
train_loader = DataLoader(OaiDataSet_Unsupervised(args=args, data_df_path=args.train_df_path, train=True),\
                          batch_size=args.batch_size, num_workers=args.num_workers, \
                          pin_memory=args.pin_mem, shuffle=True, drop_last=True)
val_loader = DataLoader(OaiDataSet_Unsupervised(args=args, data_df_path=args.val_df_path, \
                                                train=False, augmentation=True),\
                        batch_size=args.batch_size, num_workers=args.num_workers, \
                        pin_memory=args.pin_mem, shuffle=False, drop_last=True)

device = torch.device(args.device)
model = TWIST(args)
model.to(device);



class WarmupSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1
    
optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = WarmupSchedule(optimizer, warmup_steps=args.warmup_epochs)
criterion = EntLoss(args, 0.0, 1.0, pqueue=None)

def train_one_epoch(args, model, criterion, data_loader, optimizer, scheduler, device, epoch, set_training_mode=True, logfn=None):
    setup_seed(args.seed+epoch)
    model.train(set_training_mode);
    with open(logfn,'a') as file0:
        print("---------------------------Epoch "+str(epoch)+"---------------------------",file=file0)
    real_labels, pred_labels = [], []
    print_freq, iteration = 1000, 0
    train_loss_list = []
    for i, (imgs, real_label, img_index) in enumerate(data_loader):
        optimizer.zero_grad()
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        feat = model(imgs)
        if not args.aug_MultiCrop:
            args.local_crops_number = 0
        all_feats = feat.chunk(2+args.local_crops_number)
        all_probs = [torch.nn.functional.softmax(f/args.tau, dim=-1) for f in all_feats]
        n_views = len(all_feats)
        
        loss, all_loss = {}, []
        for i1 in range(2):
            for i2 in range(i1+1, n_views):
                all_loss.append(criterion(all_feats[i1], all_feats[i2], use_queue=False))
        for k in all_loss[0].keys():
            loss[k] = sum([single_loss[k]/len(all_loss) for single_loss in all_loss])

        loss['final'].backward()
        optimizer.step()
        train_loss_list.append(loss['final'].item())   
        iteration = iteration + 1
        if (iteration!=0) and (iteration%(print_freq/args.batch_size)==0):
            with open(logfn,'a') as file0:
                print('eh:'+str(loss['eh'].item())+' he:'+str(loss['he'].item())+' final:'+str(loss['final'].item()),file=file0)
    train_loss = np.mean(train_loss_list)
    scheduler.step()
    with open(logfn,'a') as file0:
        print('Epoch {} train loss: {:6.8f}'.format(epoch, train_loss), file=file0)
        
    return None


def eval_one_epoch(args, model, data_loader, device, optimizer, scheduler, epoch, \
                   set_training_mode=False, logfn=None, best_val_loss = np.inf):
    setup_seed(args.seed)
    model.train(set_training_mode);
    
    val_loss_list = []
    for i, (imgs, real_label, img_index) in enumerate(data_loader):
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        feat = model(imgs)
        all_feats = feat.chunk(2)
        all_probs = [torch.nn.functional.softmax(f/args.tau, dim=-1) for f in all_feats]
        n_views = len(all_feats)
        
        loss, all_loss = {}, []
        for i1 in range(2):
            for i2 in range(i1+1, n_views):
                all_loss.append(criterion(all_feats[i1], all_feats[i2], use_queue=False))
        for k in all_loss[0].keys():
            loss[k] = sum([single_loss[k]/len(all_loss) for single_loss in all_loss])
            
        val_loss_list.append(loss['final'].item())   
            
    val_loss = np.mean(val_loss_list)
        
    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(args.output_dir,'checkpoint_best_loss.pth')
        save_dict = {
                    'model': model.state_dict(),
                    'backbone': model.backbone_weights(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler,
                    'epoch': epoch,
                    'args': args
        }
        twist_utils.save_on_master(save_dict, checkpoint_path)
    with open(logfn,'a') as file0:
        print('Epoch {}  val  loss: {:6.8f} | best loss: {:6.8f}'.format(epoch, val_loss, best_val_loss), file=file0)
        
    return best_val_loss


best_val_loss = np.inf
output_dir = Path(args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logfn=os.path.join(output_dir, 'detail_log.txt')


print(f"Start training for {args.epochs} epochs")
start_time = time.time()
start_date = datetime.datetime.now()
for epoch in range(args.epochs):
    
    train_stats = train_one_epoch(args,model, criterion, train_loader, optimizer, scheduler, device, epoch, \
                                  set_training_mode=True, logfn=os.path.join(output_dir, 'detail_log.txt'))
    
    best_val_loss = eval_one_epoch(args, model, val_loader, device, optimizer, scheduler, epoch, set_training_mode=False, \
                                  logfn=os.path.join(output_dir, 'detail_log.txt'), best_val_loss = best_val_loss)
    
    checkpoint_path = os.path.join(args.output_dir,'checkpoint.pth')
    save_dict = {
                'model': model.state_dict(),
                'backbone': model.backbone_weights(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
                'epoch': epoch,
                'args': args}
    twist_utils.save_on_master(save_dict, checkpoint_path)
    total_time = time.time() - start_time
    total_days = (datetime.datetime.now() - start_date).days
    with open(logfn,'a') as file0:
        print('Training time {}d {}'.format(str(total_days),str(datetime.timedelta(seconds=int(total_time)))),file=file0)

        



