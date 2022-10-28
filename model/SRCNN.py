# -*- coding: utf-8 -*-
'''
卷积超分辨率神经网络开山之作
'''
# @Time    : 2022/1/23 15:03
# @Author  : LINYANZHEN
# @File    : SRCNN.py

import torch.nn as nn


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("SRCNN use PMG")
    else:
        print("SRCNN")
    return SRCNN(args)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.relu(x)
        return x


class SRCNN(nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()
        self.support_PMG = True
        self.upsample = nn.Upsample(scale_factor=args.scale)
        self.cnn = nn.Sequential(
            ConvBlock(3, 64, (9, 9), (4, 4)),
            ConvBlock(64, 32, (1, 1), (0, 0)),
            ConvBlock(32, 3, (5, 5), (2, 2), False),
        )
        channel_adaptive = [
            nn.Conv2d(64, 3, (3, 3), padding=(1, 1)),
            nn.Conv2d(32, 3, (3, 3), padding=(1, 1)),
        ]
        self.channel_adaptive = nn.Sequential(*channel_adaptive)

    def forward(self, x, step=2):
        x = self.upsample(x)
        for i in range(step + 1):
            x = self.cnn[i](x)
        if step < 2:
            x = self.channel_adaptive[step](x)
        return x
