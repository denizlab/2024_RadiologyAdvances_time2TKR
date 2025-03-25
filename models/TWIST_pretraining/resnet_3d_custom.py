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
# -*- coding: utf-8 -*-
"""
Constructors for 3D resnets
"""
from functools import partial

# import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    'ResNet3D', 'resnet3d10', 'resnet3d18', 'resnet3d34', 'resnet3d50', 'resnet3d101',
    'resnet3d152', 'resnet3d200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def resolve_norm_layer_3d(planes, norm_class, num_groups=1):
    if norm_class.lower() == "batch":
        return nn.BatchNorm3d(planes)
    if norm_class.lower() == "group":
        return nn.GroupNorm(num_groups, planes)
    raise NotImplementedError(
        f"norm_class must be batch or group, but {norm_class} was given"
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, norm_class='batch', downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, norm_class='batch', downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = resolve_norm_layer_3d(planes, norm_class, groups)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = resolve_norm_layer_3d(planes * 4, norm_class, groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 norm_class='batch',
                 groups=8,
                 shortcut_type='B',
                 in_channels=1,
                 inplanes=64,
                 conv1_kernel=(7,7,7),
                 conv1_stride=(1,2,2),
                 pool1_stride=(2,2,2),
                 conv2=None):
        super(ResNet3D, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv3d(
            in_channels,
            inplanes,
            kernel_size=conv1_kernel,
            stride=conv1_stride,
            padding=(1, 3, 3), 
            bias=False)
        self.bn1 = resolve_norm_layer_3d(inplanes, norm_class, groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=pool1_stride, padding=1)
        
        if conv2 is None:
            self.conv2 = None 
        else:
            self.conv2 = nn.Conv3d(
                inplanes,
                inplanes,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False)
        
        self.layer1 = self._make_layer(block, inplanes, layers[0],
                                       shortcut_type, norm_class, groups)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1],
                                       shortcut_type, norm_class, groups, stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2],
                                       shortcut_type, norm_class, groups,
                                       stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3],
                                       shortcut_type, norm_class, groups, stride=2) 
        
        self.num_filter_last_seq = inplanes * 8 * block.expansion
        self.initialize()

    def initialize(self):
        for m in self.modules():
            self._layer_init(m)

    @staticmethod
    def _layer_init(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def _make_layer(self, block, planes, blocks, shortcut_type, norm_class='batch', groups=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), resolve_norm_layer_3d(planes * block.expansion, norm_class, groups))

        layers = []
        layers.append(block(self.inplanes, planes, stride, groups=groups, norm_class=norm_class, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
#         print(x.shape)
        x = self.conv1(x)
#         print("after conv1", x.shape)
        x = self.bn1(x)
#         print("after bn1", x.shape)
        x = self.relu(x)
        x = self.maxpool(x)
#         print("after maxpool", x.shape)
        if self.conv2 is not None:
            x = self.conv2(x)
#             print("after conv2", x.shape)
            
        x = self.layer1(x)
#         print("after 1 ", x.shape)
        x = self.layer2(x)
#         print("after 2", x.shape)
        x = self.layer3(x)
#         print("after 3", x.shape)
        x = self.layer4(x)
#         print("last size", x.shape)

#         x = self.avgpool(x)

#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet3d10(**kwargs):
    """Constructs a ResNet3D-10 model.
    """
    model = ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet3d18(**kwargs):
    """Constructs a ResNet3D-18 model.
    """
    model = ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet3d34(**kwargs):
    """Constructs a ResNet3D-34 model.
    """
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet3d50(**kwargs):
    """Constructs a ResNet3D-50 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet3d101(**kwargs):
    """Constructs a ResNet3D-101 model.
    """
    model = ResNet3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet3d152(**kwargs):
    """Constructs a ResNet3D-152 model.
    """
    model = ResNet3D(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet3d200(**kwargs):
    """Constructs a ResNet3D-200 model.
    """
    model = ResNet3D(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

def DESS(**kwargs):
    """Constructs a ResNet3D-200 model.
    """
    model = ResNet3D(
        BasicBlock, 
        [2, 2, 2, 2], 
        inplanes=32,
        conv1_kernel=(7,7,7),
        conv1_stride=(2,2,2),
        pool1_stride=(2,2,2),
        conv2=True,
        **kwargs
    )
    return model


def TSE(**kwargs):
    """Constructs a ResNet3D-TSE model.
    """
    model = ResNet3D(
        BasicBlock, 
        [2, 2, 2, 2], 
        inplanes=32,
        conv1_kernel=(3,7,7),
        conv1_stride=(2,2,2),
        pool1_stride=(1,2,2),
        conv2=True,
        **kwargs
    )
    return model

def get_resnet3d(size=18, **kwargs):
    if size == 10:
        return resnet3d10(**kwargs)
    elif size == 18:
        return resnet3d18(**kwargs)
    elif size == 34:
        return resnet3d34(**kwargs)
    elif size == 50:
        return resnet3d50(**kwargs)
    elif size == 101:
        return resnet3d101(**kwargs)
    elif size == 152:
        return resnet3d152(**kwargs)
    elif size == 200:
        return resnet3d200(**kwargs)
    elif size == 'DESS':
        return DESS(**kwargs)
    elif size == 'TSE':
        return TSE(**kwargs)
    raise KeyError(size)

    
    
class GlobalPool3D(nn.Module):

    def __init__(self, global_pool, lse_gamma=None):
        super(GlobalPool3D, self).__init__()
        self.global_pool = global_pool
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))
#         self.exp_pool = ExpPool()
#         self.linear_pool = LinearPool()
#         self.lse_pool = LogSumExpPool(gamma=lse_gamma)

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, feat_map):
        if self.global_pool == 'AVG':
            return self.avgpool(feat_map)
        elif self.global_pool == 'MAX':
            return self.maxpool(feat_map)
        elif self.global_pool == 'AVG_MAX':
            a = self.avgpool(feat_map)
            b = self.maxpool(feat_map)
            return torch.cat((a, b), 1)
#         elif self.global_pool == 'AVG_MAX_LSE':
#             a = self.avgpool(feat_map)
#             b = self.maxpool(feat_map)
#             c = self.lse_pool(feat_map)
#             return torch.cat((a, b, c), 1)
#         elif self.global_pool == 'EXP':
#             return self.exp_pool(feat_map)
#         elif self.global_pool == 'LINEAR':
#             return self.linear_pool(feat_map)
#         elif self.global_pool == 'LSE':
#             return self.lse_pool(feat_map)
        else:
            raise Exception('Unknown pooling type : {}'
                            .format(self.global_pool))
