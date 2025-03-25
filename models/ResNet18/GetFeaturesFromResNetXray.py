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
import warnings
import torch
import torchvision.models as tvmodels
from torch.utils.data import DataLoader
from internal_libraries import input_pipeline
from pytorch_lightning.utilities import seed
import random
import pandas as pd
import torch.nn as nn
import numpy
torch.cuda.empty_cache()

warnings.filterwarnings('ignore')
def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)

# check pytorch version and setting up the GPU
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #use cpu or gpu
print(device)
################# Xray ################
MRIsequence='XRay'
##########################################################################################################################
learning_rate_selected = 1e-4 # [1e-4, 1e-6]
weight_decay_selected = 1e-4  # [1e-4, 5e-5]
batch_size_train = 1 # [2(ResNet50), 4(ResNet34)]
sigma_std = torch.sqrt(torch.tensor(4))                                          # Standard deviation for discretization
sigma_std_1 = torch.sqrt(torch.tensor(2))                                          # Standard deviation for discretization
bin_step_no = torch.tensor(4)                                                     # No of bins (discretized)
number_of_bins = int(((108 + 2*sigma_std*sigma_std) / bin_step_no + 1).item())   # No of outputs(bins)
##########################################################################################################################
train_dataset = input_pipeline.XrayFromCSV('Xray_Train_SUBSET.csv', 1024, 768, sigma=sigma_std, bin_step= bin_step_no, 
                                        left_right='Both_legs', mode='Train', normalize=True, image_type='Xray', 
                                        image_root_path='/gpfs/data/denizlab/Datasets/OAI/Radiographs/annotation_hg_2023-11-16/data/',MRIsequence = 'Xray')
validation_dataset = input_pipeline.XrayFromCSV('Xray_Validation_SUBSET.csv', 1024, 768, sigma=sigma_std, bin_step= bin_step_no, 
                                        left_right='Both_legs', mode='Validation', normalize=True, image_type='Xray', 
                                        image_root_path='/gpfs/data/denizlab/Datasets/OAI/Radiographs/annotation_hg_2023-11-16/data/',MRIsequence = 'Xray')
test_dataset = input_pipeline.XrayFromCSV('Xray_Test_SUBSET.csv', 1024, 768, sigma=sigma_std, bin_step= bin_step_no, 
                                        left_right='Both_legs', mode='Validation', normalize=True, image_type='Xray', 
                                        image_root_path='/gpfs/data/denizlab/Datasets/OAI/Radiographs/annotation_hg_2023-11-16/data/',MRIsequence = 'Xray')
########################################################################################################################
########################################################################################################################
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=40, prefetch_factor=256,
                        drop_last=True, worker_init_fn=numpy.random.seed(int(42)))

val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=40, prefetch_factor=256,
                        drop_last=True, worker_init_fn=numpy.random.seed(int(42)))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=40, prefetch_factor=256,
                        drop_last=True, worker_init_fn=numpy.random.seed(int(42)))
########################################################################################################################
# Load the pretrained ResNet18 model
pretrained_resnet18 = tvmodels.resnet18(pretrained=True)
pretrained_resnet18.to(device)
pretrained_resnet18.eval()

# Extract features up to the second-to-last layer
featuresResNet = nn.Sequential(*list(pretrained_resnet18.children())[:-2])

# Define datasets
dataset_type = {
    'train': train_loader,
    'validation':val_loader,
    'test':test_loader
}

# Initialize a list to collect all dataframes
dataset_names = []

# Process each dataset
for key_dataset, loader in dataset_type.items():
    print(f"Processing {key_dataset} dataset")
    resnet_features, all_locations = [], []
    
    # Extract features without gradients
    with torch.no_grad():
        for loaded_data in loader:
            data = loaded_data[0].to(device)  # Get the image
            location = loaded_data[1][0][0]  # Get the location of the data
            outputs = featuresResNet(data)   # Extract features
            resnet_features.append(outputs)
            all_locations.append(location)
    
    # Concatenate features
    resnet_features = torch.cat(resnet_features, dim=0)
    
    # Apply adaptive average pooling and flatten features
    avg_pool = nn.AdaptiveAvgPool2d(1)
    x_ResNet = avg_pool(resnet_features).view(resnet_features.size(0), -1)
    
    # Convert features to numpy and create a DataFrame
    x_ResNet_np = x_ResNet.cpu().numpy()
    df_Resnet = pd.DataFrame(x_ResNet_np)
    
    # Extract file names
    split_strings = [s.split('/')[-1] for s in all_locations]
    split_strings_series = pd.Series(split_strings)
    
    # Add metadata columns
    df_Resnet.insert(0, "File_Name", split_strings_series)
    df_Resnet.insert(0, 'TrainValTest', key_dataset)
    
    # Add to the list of datasets
    dataset_names.append(df_Resnet)


# Concatenate all datasets into a single DataFrame
result_df = pd.concat(dataset_names, ignore_index=True)

# Rename columns to include MRI sequence information
result_df.columns = list(result_df.columns[:2]) + [MRIsequence + str(col) for col in result_df.columns[2:]]

# Save the final DataFrame to a CSV file
csv_path = 'Pretrained2DResnet18_XRay.csv'
result_df.to_csv(csv_path, index=False)

print(f"Feature extraction completed. Features saved to: {csv_path}")

