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
import torch
import numpy as np

def modified_KLDivLoss(y_pred, y_true):
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')
    y2 = y_true + 1e-8
    loss = loss_func(y_pred, y2) 
    return {'Final loss': loss}
               
def summary_writer_dict(loss_dict, writer, step, mode='/train'):
    if writer is not None:
        for key in loss_dict.keys():
            try:
                writer.add_scalar(key + mode, loss_dict[key].item(), step)
            except:
                writer.add_scalar(key + mode, loss_dict[key], step)
#######################################################################################################################################################################################
#Our months are between 0-108. We shifted them by 12. So, updated months are between [12-120]. 
    # When we select sigma=sqrt(4), our labels need to be between [8-124]. 
        #   Number of bins: M= (124-8)/bin_step+1
        #   For bin_step=4, M= 30 bins. Labelset should be between [8-124]. Variance=4 so for borders, to have the distribution, [12-4, 120+4] becomes [8,124]: labelset = np.array([i * bin_step + 8 for i in range(int(120 / bin_step))]) 
    # When we select sigma=sqrt(12), our labels(months) [12-120] need to be between [12-12, 120+12] = [0-132]. 
        #   Number of bins: M= (132-0)/bin_step+1
        #   For bin_step=1, M= 132 bins. Labelset should be between [0-132].  labelset = np.array([i * bin_step for i in range(int(132 / bin_step))])  
#######################################################################################################################################################################################
def label_to_normal_dist(label, sigma=2, bin_step=4, within_year=108, shift_month=12):
    # labelset = np.array([i * bin_step + 8 for i in range(int(120 / bin_step))]) #Here we get bin_step=4, var=4 . So M=120/4=30bins. Here sigma =sqrt(4).
    labelset = np.array([i * bin_step + (shift_month-sigma*sigma) for i in range(int((within_year + 2*sigma*sigma) / bin_step + 1))]) #Our data starts from 12 and has a range of 120-12=108. We subtract var from 12 and add var to 120.So, range becomes:108+2*var(sigma*sigma)

    dis = np.exp(-1/2. * np.power((labelset - label)/sigma, 2))
    dis = dis / dis.sum()
    return dis, labelset
##########################################################################################################################
