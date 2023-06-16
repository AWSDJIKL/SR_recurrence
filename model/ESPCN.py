# -*- coding: utf-8 -*-
'''
亚像素卷积
'''
# @Time    : 2022/1/23 15:03
# @Author  : LINYANZHEN
# @File    : ESPCN.py

import torch.nn as nn


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("ESPCN use PMG")
    else:
        print("ESPCN")
    return ESPCN(args)


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


class ESPCN(nn.Module):
    def __init__(self, args):
        super(ESPCN, self).__init__()
        self.support_PMG = True
        # self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        # self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        # self.conv3 = nn.Conv2d(32, 3 * (args.scale ** 2), (3, 3), (1, 1), (1, 1))

        body = [
            ConvBlock(3, 64, (5, 5), (2, 2)),
            ConvBlock(64, 32, (3, 3), (1, 1)),
            ConvBlock(32, 3 * (args.scale ** 2), (3, 3), (1, 1), False)
        ]
        self.body = nn.Sequential(*body)
        channel_adaptive = [
            nn.Conv2d(64, 3 * (args.scale ** 2), (3, 3), padding=(1, 1)),
            nn.Conv2d(32, 3 * (args.scale ** 2), (3, 3), padding=(1, 1)),
        ]
        self.channel_adaptive = nn.Sequential(*channel_adaptive)
        self.pixel_shuffle = nn.PixelShuffle(args.scale)
        self.part = args.part

    def forward(self, x, step=-1):
        # for i in range(step + 1):
        #     x = self.body[i](x)
        for i in range(self.part[step]):
            x = self.body[i](x)
        if 0 <= step < 2:
            x = self.channel_adaptive[step](x)
        x = self.pixel_shuffle(x)
        return x
