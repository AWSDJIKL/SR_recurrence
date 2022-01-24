# -*- coding: utf-8 -*-
'''
亚像素卷积
'''
# @Time    : 2022/1/23 15:03
# @Author  : LINYANZHEN
# @File    : ESPCN.py

import torch.nn as nn

def make_model(args):
    return ESPCN(args)

class ESPCN(nn.Module):
    def __init__(self, args):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 3 * (args.scale ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(args.scale)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x
