# -*- coding: utf-8 -*-
'''
各类要复现的模型
'''
# @Time    : 2021/12/3 11:59
# @Author  : LINYANZHEN
# @File    : models.py
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(ResBlock, self).__init__()
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.conv3x3_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)
        x = self.batchnorm(x)
        return x + x_in


class Upsample_Block(nn.Module):
    def __init__(self):
        super(Upsample_Block, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.residual1 = ResBlock(64, 64, (1, 1))
        self.residual2 = ResBlock(64, 64, (1, 1))
        self.residual3 = ResBlock(64, 64, (1, 1))
        self.residual4 = ResBlock(64, 64, (1, 1))
        self.upsample1 = Upsample_Block()
        self.upsample2 = Upsample_Block()
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU()
        self.bicubic = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        x_in = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        # x_in = self.bicubic(x_in)
        # x = x + x_in
        return x
