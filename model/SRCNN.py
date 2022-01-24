# -*- coding: utf-8 -*-
'''
卷积超分辨率神经网络开山之作
'''
# @Time    : 2022/1/23 15:03
# @Author  : LINYANZHEN
# @File    : SRCNN.py

import torch.nn as nn


def make_model(args):
    return SRCNN(args)


class SRCNN(nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=args.scale)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, (9, 9), padding=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 3, (5, 5), padding=(2, 2)),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.cnn(x)
        return x
