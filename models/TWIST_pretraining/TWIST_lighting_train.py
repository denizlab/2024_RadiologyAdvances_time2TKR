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
import os
import warnings
from lars import *
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.strategies.ddp import DDPStrategy

from augmentation import *
from objective import *
from resnet_3d_custom import get_resnet3d


def cli_main(args):
    if not isinstance(args, dict):
        args_dict = vars(args)
    else:
        args_dict = args


    # Environment
    CHECKPOINT_PATH = os.environ.get(
        "PATH_CHECKPOINT", "saved_model/"
    )
    save_name = "TWIST"
    
    NUM_WORKERS = os.cpu_count()
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)

    # Setting the seed
    pl.seed_everything(args.seed)
    
    # data
    data_module = OaiDataSet_Unsueprvised_DataModule(args)
    args.train_len = len(data_module.train_dataloader())
    # model
    model = TWISTModule(args)
    
    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(project="SSL_Training", entity="oai_fighter", save_dir=args.save_dir)

    # Log hyperparameters
    wandb_logger.log_hyperparams(args_dict)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        gpus=1,
        # gpus=1 if str(device) == "cuda:0" else 0,
        accumulate_grad_batches=4,
        min_epochs=50,
        max_epochs=args.epochs,
        # auto_scale_batch_size = "binsearch",
        callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_total_loss", save_top_k=5),
                   LearningRateMonitor("epoch")],
        strategy="ddp",
        logger=wandb_logger
        # strategy=DDPStrategy(find_unused_parameters=False)
    )

    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = False
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # fit
    trainer.fit(model, datamodule=data_module)
    
    
class OaiDataSet_Unsueprvised_DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.args = args
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_mem
                  
    def train_dataloader(self):
        train_dataset = OaiDataSet_Unsupervised(
            self.args, data_df_path=self.args.train_df_path, augmentation=True, train=True, labels_to_load=[]
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, \
                        pin_memory=self.pin_memory, shuffle=False, drop_last=True)
    
    def val_dataloader(self):
        val_dataset = OaiDataSet_Unsupervised(
            self.args, data_df_path=self.args.val_df_path, augmentation=True, train=False, labels_to_load=[]
        )
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, \
                        pin_memory=self.pin_memory, shuffle=False, drop_last=True)
    
               
class OaiDataSet_Unsupervised(Dataset):
    def __init__(self, args, data_df_path, augmentation=True, train=True, labels_to_load=[]):
        self.args = args
        self.augmentation = augmentation
        self.labels_to_load = labels_to_load
        self.data_dir = args.data_dir
        self.data_df = pd.read_csv(data_df_path)
        self.train = train
    
    def __getitem__(self, index):
        args = self.args

        image_path = os.path.join(self.data_dir, self.data_df.iloc[index]['File_Name'])
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

class WarmupSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1
    
class TWISTModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hparams.update(vars(args))
        self.automatic_optimization = True
        self.save_hyperparameters()
        self.backbone = resnet_model()
        self.feature_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()
        self.projection_heads = ProjectionHead(args,feature_dim=self.feature_dim)
        self.loss_module = EntLoss(args, pqueue=None)
        self.apply(self.init_weights)
        self.param_weights = []
        self.param_biases = []
        self.get_model_parameters()
        args.local_crops_number = 0 if not args.aug_MultiCrop else 6
        self.iteration = 0
        
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
            torch.nn.init.xavier_uniform(m.weight,gain=1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        
    def forward(self, imgs):
        return self.model(imgs)
    
    def get_model_parameters(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print('{} is not optimized'.format(name))
                continue
            skip = ['pos_embed', 'cls_token', 'dist_token']
            if len(param.shape) == 1 or name.endswith(".bias") or sum([sk in name for sk in skip]):
                print('{} has been excluded for weight decay'.format(name))
                self.param_biases.append(param)
            else:
                self.param_weights.append(param)
                
    def adjust_learning_rate(self, step):
        warmup_steps = self.args.warmup_epochs * self.args.train_len
        max_steps = self.args.epochs * self.args.train_len
        base_lr = 1.0
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
            self.optimizers().optimizer.param_groups[0]['lr'] = lr * self.args.lr
            self.optimizers().optimizer.param_groups[1]['lr'] = lr * self.args.lr / self.args.lr_wbr
        else:
            step -= warmup_steps
            cosann_mme_steps = max_steps - warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / cosann_mme_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
            self.optimizers().optimizer.param_groups[0]['lr'] = lr * self.args.lr
            self.optimizers().optimizer.param_groups[1]['lr'] = lr * self.args.lr / self.args.lr_wbr
    
    def model(self, x):
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
    
    def configure_optimizers(self):
        if self.args.optim == 'sgd':
            bias_weight_decay = 0.0 if self.args.exclude_bias_weight_decay else self.args.weight_decay
            parameters = [{'params': self.param_weights, 'weight_decay': self.args.weight_decay}, 
                        {'params': self.param_biases,  'weight_decay': bias_weight_decay}]
            optimizer = torch.optim.SGD(parameters, lr=0, momentum=0.9, weight_decay=self.args.weight_decay)
        elif self.args.optim in ['lars', 'lars_oss']:
            bias_weight_decay = 0.0 if self.args.exclude_bias_weight_decay else self.args.weight_decay
            parameters = [{'params': self.param_weights, 'weight_decay': self.args.weight_decay, 'lars_exclude': False}, 
                            {'params': self.param_biases,  'weight_decay': bias_weight_decay, 'lars_exclude': True}]
            optimizer = LARS_OPENSELF(parameters, lr=0, weight_decay=self.args.weight_decay, momentum=0.9)
        elif self.args.optim == 'adamw':
            bias_weight_decay = 0.0 if self.args.exclude_bias_weight_decay else self.args.weight_decay
            parameters = [{'params': self.param_weights, 'weight_decay': self.args.weight_decay}, 
                            {'params': self.param_biases,  'weight_decay': bias_weight_decay}]
            optimizer = torch.optim.AdamW(parameters)

        # parameters = [{'params': self.param_weights, 'weight_decay': self.args.weight_decay, 'lars_exclude': False}, 
        #             {'params': self.param_biases,  'weight_decay': 0.0, 'lars_exclude': True}]
        # optimizer = LARS_OPENSELF(parameters, lr=0, weight_decay= self.args.weight_decay, momentum=0.9)
        return optimizer
    
    # def on_train_start(self):
        # if self.logger is not None:
        #     self.logger.log_hyperparams(self.hparams)
        #     self.logger.save()
        
    def training_step(self, batch, batch_idx):
        imgs, label, img_index = batch
        epoch = self.current_epoch
        
        self.adjust_learning_rate(epoch*self.args.train_len + batch_idx)
        feat = self.forward(imgs)
        
        all_feats = feat.chunk(2+self.args.local_crops_number)
        all_probs = [torch.nn.functional.softmax(f/self.args.tau, dim=-1) for f in all_feats]
        n_views = len(all_feats)
        
        loss, all_loss = {}, []
        for i1 in range(2):
            for i2 in range(i1+1, n_views):
                all_loss.append(self.loss_module(all_feats[i1], all_feats[i2], use_queue=False))
        for k in all_loss[0].keys():
            loss[k] = sum([single_loss[k]/len(all_loss) for single_loss in all_loss])
        
        self.log("train_total_loss",loss['final'].item(), on_step=True, on_epoch=True)
        self.log("train_kl_loss", loss['kl'].item(), on_step=False, on_epoch=True)
        self.log("train_eh_loss", loss['eh'].item(), on_step=False, on_epoch=True)
        self.log("train_he_loss", loss['he'].item(), on_step=False, on_epoch=True)
        
        return loss['final']
    
    def validation_step(self, batch, batch_idx):
        # set mode to validation
        imgs, label, img_indx = batch
        feat = self.forward(imgs)
        all_feats = feat.chunk(2)
        all_probs = [torch.nn.functional.softmax(f/self.args.tau, dim=-1) for f in all_feats]
        n_views = len(all_feats)
        
        loss, all_loss = {}, []
        for i1 in range(2):
            for i2 in range(i1+1, n_views):
                all_loss.append(self.loss_module(all_feats[i1], all_feats[i2], use_queue=False))
        for k in all_loss[0].keys():
            loss[k] = sum([single_loss[k]/len(all_loss) for single_loss in all_loss])
        
        self.log("val_total_loss",loss['final'].item(), on_step=False, on_epoch=True)
        self.log("val_kl_loss", loss['kl'].item(), on_step=False, on_epoch=True)
        self.log("val_eh_loss", loss['eh'].item(), on_step=False, on_epoch=True)
        self.log("val_he_loss", loss['he'].item(), on_step=False, on_epoch=True)

        return torch.stack(all_probs)

    # def validation_step_end(self, batch_parts):
    #     return torch.stack([batch_parts[0], batch_parts[1]])
    
    # def validation_epoch_end(self, validation_stp_outputs):
    #     all_probs = torch.stack(validation_stp_outputs)
    #     all_probs = all_probs.view(-1,self.args.dim)
    #     prob_save_path = os.path.join(self.logger.log_dir, 'probs')
    #     os.makedirs(prob_save_path, exist_ok=True)
    #     torch.save(all_probs, os.path.join(prob_save_path, f'epoch{self.current_epoch}_prob.pt'))
    #     figure = self.create_heatmap(all_probs.cpu())
    #     self.logger.experiment.add_figure("prob_heatmap", figure, global_step=self.global_step)
        
    # def create_heatmap(self, data):
    #     s = sns.heatmap(data, annot=False, cbar=False, xticklabels=True, yticklabels=True)
    #     return s.get_figure()
        
def build_args():
    parser = argparse.ArgumentParser('Self-Supervised', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--dim', default=1000, type=int)
    parser.add_argument('--hid_dim', default=4096, type=int) 
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument("--optim",  choices=['adamw', 'lars', 'sgd'], type=str, default="adamw")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', default=1000, type=int) 
    parser.add_argument('--warmup_epochs', default=100, type=int)
    parser.add_argument('--save_every_n_epoch', default=50, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--output_dir', default='./Results/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--lam1', type=float, default=0.0, metavar='LR')
    parser.add_argument('--lam2', type=float, default=1.0, metavar='LR')
    parser.add_argument('--tau', type=float, default=1.0, metavar='LR')
    parser.add_argument('--EPS', type=float, default=1e-5, help='episillon')
    parser.add_argument('--amp', default=0, type=int)
    parser.add_argument('--crops_interact_style', type=str, default='sparse')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    
    #OAI data related
    parser.add_argument("--contrast", type=str, default="SAG_IW_TSE", choices=['SAG_IW_TSE','COR_IW_TSE','SAG_3D_DESS'])
    parser.add_argument("--default_save_dir", type=str, default="./")
    parser.add_argument("--default_data_dir", type=str, default="/gpfs/data/denizlab/Datasets/OAI/")
    parser.add_argument('--data_dir', type=str, default = 'path/to/data/directory')
    parser.add_argument('--train_df_path', type=str, default = 'path/to/train/csv')
    parser.add_argument('--val_df_path', type=str, default = 'path/to/val/csv')
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
    parser.add_argument('--aug_MultiCrop', type=bool, default=True)
    parser.add_argument('--local_crop_size_h', type=int, default=128)
    parser.add_argument('--local_crop_size_w', type=int, default=128)
    parser.add_argument('--local_crop_size_d', type=int, default=12)

    # Model parameters
    parser.add_argument('--exclude_bias_weight_decay', type=int, default=1)
    
    args = parser.parse_args()
    
    if "DESS" in args.train_df_path:
        args.image_size_d = 160
        args.num_slices_to_load = 160
    elif "TSE" in args.train_df_path:
        args.image_size_d = 160
        args.num_slices_to_load = 36
        
    args.data_dir = os.path.join(args.default_data_dir, args.contrast)
    args.save_dir = os.path.join(args.default_save_dir, args.contrast)
    args.global_crops_scale = (args.min1, args.max1)
    args.local_crops_scale = (args.min2, args.max2)
    args.image_size = (args.image_size_d, args.image_size_h, args.image_size_w)
    args.local_crop_size = (args.local_crop_size_d, args.local_crop_size_h, args.local_crop_size_w) 
    
    args.lr = 0.05 * args.batch_size  / 256.0
    args.lr_sl = 0.05 * args.batch_size  / 256.0
    args.lr_wbr = 1.0
    args.weight_decay = 1.5e-6
    args.weight_decay_end = 1.5e-6
    args.warmup_epochs = 50
    
    return args
    
def run_cli():
    args = build_args()
    cli_main(args)
    
if __name__ == "__main__":
    run_cli()