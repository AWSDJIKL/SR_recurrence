# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/3/4 14:47
# @Author  : LINYANZHEN
# @File    : resnet.py
from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable


class Resnet(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(Resnet, self).__init__()
        resnet_features = models.resnet50(pretrained=True).features
        modules = [m for m in resnet_features]
        self.resnet = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.requires_grad_(False)


    def forward(self, sr, hr):
        def _forward(x):
            # x = self.sub_mean(x)
            x = self.resnet(x)
            return x

        resnet_sr = _forward(sr)
        with torch.no_grad():
            resnet_hr = _forward(hr.detach())
        loss = nn.MSELoss()(resnet_sr, resnet_hr)

        return loss
