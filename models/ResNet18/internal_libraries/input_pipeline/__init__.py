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
import h5py
from internal_libraries import XrayDataLoader, losses_and_metrics
from torchvision import transforms
import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torch.nn.functional as F

######################################################################################################################################################
class XrayFromCSV(data.Dataset):
    def __init__(self, csv_file, image_size, crop_size, left_right='Left', mode='Train', normalize=True,
                 image_type='Xray', image_root_path='/gpfs/data/denizlab/Datasets/OAI/Radiographs/'
                                                    'annotation_hg_2022-10-03_CombindIndividualRuns/data/', 
                 sigma=torch.sqrt(torch.tensor(4)), bin_step =4, MRIsequence = 'DESS'):
        
        # image_root_path='/gpfs/data/denizlab/Datasets/OAI/Radiographs/annotation_hg_2021-06-29_12-53-31/data/'):
        super(XrayFromCSV, self).__init__()
        self.root_dir = image_root_path
        self.csv_df = pd.read_csv(csv_file, index_col=0)
        self.crop_size = crop_size
        self.image_size = image_size
        self.mode = mode
        self.left_right = left_right
        self.test_df = None
        self.normalize = normalize
        self.image_type = image_type
        self.sigma = sigma
        self.bin_step = bin_step
        # self.dataset = self.data_process().iloc[[1, 2, 3, 4],]# 5, 6, 7, 8, 9, 10, 11, 12],]
        self.dataset = self.data_process()
        self.MRIsequence = MRIsequence

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_item = self.dataset.iloc[idx].to_numpy()
        location = data_item[0]
        data_item[1:] = data_item[1:].astype(float)
        month = 12 * data_item[1]  / 365 
        month = 12 + month  # moving label from [0-108] months to [12-120]. We subtract 12 after prediction 

        # Here the bin size for discretization is 4 and variance=4. The data will be discretized as 8,12,16, ..., 116,120,124.
        y_dist, y_set = losses_and_metrics.label_to_normal_dist(month, sigma=self.sigma, bin_step=self.bin_step)
        
        ############################################################################################################
        
        if self.image_type == 'Xray':
            im_batch = image_loader_Xray(location, self.image_size, self.crop_size, 'identity', self.mode, self.normalize)  ## If it is pretrained Resnet, change 'CC' to 'identity'. Pretrained Resnet has 3 channels
        elif self.image_type == 'MRI':
            try:
                im_batch = image_loader_MRI(location, self.image_size, self.crop_size, 'identity', self.mode,
                                           self.normalize, self.MRIsequence)                
            except:
                return data_item
        # return im_batch, (data_item[0], (y_dist, y_set, data_item[1]))
    
        return im_batch, (data_item[0], (y_dist, y_set, data_item[1]))

    def data_process(self):
        intermediate_df = None
        if self.left_right == 'Left':
            intermediate_df = self.csv_df[self.csv_df['Knee'] == 1]
            # intermediate_df = intermediate_df[['Left_TKR_Days', 'File_Name']]
            # intermediate_df.rename(columns={'File_Name': 'Location',
            #                                 'TKR_Days': 'Days_to_tkr'}, inplace=True)
        elif self.left_right == 'Right':
            intermediate_df = self.csv_df[self.csv_df['Knee'] == 2]
            # intermediate_df = intermediate_df[['Right_TKR_Days', 'File_Name']]
            # intermediate_df.rename(columns={'File_Name': 'Location',
            #                                 'TKR_Days': 'Days_to_tkr'}, inplace=True)
        else:
            intermediate_df_l = self.csv_df[['Left_TKR_Days', 'File_Name']]
            intermediate_df_l.rename(columns={'File_Name': 'Location',
                                              'Left_TKR_Days': 'Days_to_tkr'}, inplace=True)
            intermediate_df_r = self.csv_df[['Right_TKR_Days', 'File_Name']]
            intermediate_df_r.rename(columns={'File_Name': 'Location',
                                              'Right_TKR_Days': 'Days_to_tkr'}, inplace=True)
            intermediate_df = pd.concat([intermediate_df_l, intermediate_df_r]) # row concatination
        intermediate_df['Location'] = self.root_dir + intermediate_df['Location'].astype(str)
        intermediate_df.dropna(inplace=True) #remove the NaN or None columns/rows
        intermediate_df = intermediate_df.iloc[:, [1, 0]]
        return intermediate_df
######################################################################################################################################################
#####################################################################################################################################################
def image_loader_Xray(img_name, image_size, crop_size, read_model='identity', mode='val', normalize=True):
    f = h5py.File(img_name, 'r')
    image = f.get('data')[()]
    image = image[..., np.newaxis]
    image_size_file = image.shape[0]
    f.close()
    
    transRGB = XrayDataLoader.ToRGB() if read_model != "CC" else XrayDataLoader.Identity()  #Convert 2D Xrays into 3D image
    # transRGB = XrayDataLoader.ToRGB()  #Convert 2D Xrays into 3D image

    transResize = XrayDataLoader.Identity() if image_size == image_size_file else XrayDataLoader.Resize(image_size)
    data_transforms = transforms_applied(transResize, crop_size, transRGB)
    image = image.astype('float32')
    image = data_transforms[mode](image)
    if normalize:
        means = image.mean(axis=(1, 2))
        std = image.std(axis=(1, 2))
        normalize = transforms.Normalize(means, std)
        image = normalize(image)
    image = image.float()
    # image = image.unsqueeze(0) #For 2D Resnet, we dont need that. We use 1 channel input and added this channel above as "image = image[..., np.newaxis]". 
    return image 

######################################################################################################################################################
def image_loader_MRI(img_name, image_size, crop_size, read_model='identity', mode='val', normalize=True, MRIsequence='IW-TSE'):
    f = h5py.File(img_name, 'r')
    image = f.get('data')[()]
    image_size_file = image.shape[0]
    f.close()

    if MRIsequence == 'IW-TSE' :
        if image.shape[0] < 50: # For preprocessed data, we need to permute the coordinates. Original preprocessed data has 37x384x384.
            image = image.transpose(1, 2, 0)# Now we have 384x384x37 
            image_size_file = image.shape[0]

        if image.shape[2] > 36:
            image = image[:, :,:36]
        transRGB = XrayDataLoader.ToRGB() if read_model != "identity" else XrayDataLoader.Identity()
        transResize = XrayDataLoader.Identity() if image_size == image_size_file else XrayDataLoader.Resize(image_size)
        data_transforms = transforms_applied(transResize, crop_size, transRGB)
        image = image.astype('float32')
        image = data_transforms[mode](image)
        if normalize:
            means = image.mean(axis=(1, 2))
            std = image.std(axis=(1, 2))
            normalize = transforms.Normalize(means, std)
            image = normalize(image)
        if image.shape[0] < 36:
            base_shape = image.shape[0]
            image = F.pad(image, (0, 0, 0, 0, 0, 36 - base_shape), value=0)
        image = image.float()
        image = image.unsqueeze(0)
        return image
        
    elif MRIsequence == 'COR-TSE' :
        if image.shape[2] > 36:
            image = image[:, :,:36]
        transRGB = XrayDataLoader.ToRGB() if read_model != "identity" else XrayDataLoader.Identity()
        transResize = XrayDataLoader.Identity() if image_size == image_size_file else XrayDataLoader.Resize(image_size)
        data_transforms = transforms_applied(transResize, crop_size, transRGB)
        image = image.astype('float32')
        image = data_transforms[mode](image)
        if normalize:
            means = image.mean(axis=(1, 2))
            std = image.std(axis=(1, 2))
            normalize = transforms.Normalize(means, std)
            image = normalize(image)
        if image.shape[0] < 36:
            base_shape = image.shape[0]
            image = F.pad(image, (0, 0, 0, 0, 0, 36 - base_shape), value=0)
        image = image.float()
        image = image.unsqueeze(0)
        return image
    
    elif MRIsequence == 'DESS':
        if len(image.shape) == 4: # Masked data has size of 384x384x160x7. Get the concatenated ROIs in the 0th dimension.
            image = image[:,:,:,0] # Now we have 384x384x160

        if image.shape[2] > 160:
            image = image[:, :,:160]
        transRGB = XrayDataLoader.ToRGB() if read_model != "identity" else XrayDataLoader.Identity()
        transResize = XrayDataLoader.Identity() if image_size == image_size_file else XrayDataLoader.Resize(image_size)
        data_transforms = transforms_applied(transResize, crop_size, transRGB)
        image = image.astype('float32')
        image = data_transforms[mode](image)
        # print(image.shape)
        if normalize:
            means = image.mean(axis=(1, 2))
            std = image.std(axis=(1, 2))
            normalize = transforms.Normalize(means, std)
            image = normalize(image)
        if image.shape[0] < 160:
            base_shape = image.shape[0]
            image = F.pad(image, (0, 0, 0, 0, 0, 160 - base_shape), value=0)
        image = image.float()
        # print(image.shape)
        image = image.unsqueeze(0)
        return image

######################################################################################################################################################
######################################################################################################################################################
  
def transforms_applied(transResize, crop_size, transRGB):
    data_transforms = {
        'Train': transforms.Compose([
        transResize,
        XrayDataLoader.RandomCrop(crop_size),
        transRGB,
        XrayDataLoader.ToTensor(),
        XrayDataLoader.RandomHorizontalFlip(),
    ]),
        'Validation': transforms.Compose([
        transResize,
        XrayDataLoader.CenterCrop(crop_size),
        transRGB,
        XrayDataLoader.ToTensor(),
    ]),
        'Validation5Crop': transforms.Compose([
        transResize,
        transRGB,
        XrayDataLoader.ToTensor(),
    ]),
    }
    return data_transforms
