# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2023/1/7 21:34
# @Author  : LINYANZHEN
# @File    : IMDNP.py
import torch
import torch.nn as nn
from model import common


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("IMDN use PMG")
    else:
        print("IMDN")
    return IMDNP(args)


class IMDB(nn.Module):
    def __init__(self):
        super(IMDB, self).__init__()
        self.conv3_1 = common.default_conv(64, 64, 3)
        self.conv3_2 = common.default_conv(48, 64, 3)
        self.conv3_3 = common.default_conv(48, 64, 3)
        self.conv3_4 = common.default_conv(48, 16, 3)
        self.conv1x1 = common.default_conv(64, 64, 1)

        self.leakyRelu = nn.LeakyReLU()

    def forward(self, x):
        res = x
        x = self.conv3_1(x)
        x = self.leakyRelu(x)
        # channel split
        s1 = x[:, :16, :, :]
        x = x[:, 16:, :, :]

        x = self.conv3_2(x)
        x = self.leakyRelu(x)
        # channel split
        s2 = x[:, :16, :, :]
        x = x[:, 16:, :, :]

        x = self.conv3_3(x)
        x = self.leakyRelu(x)
        # channel split
        s3 = x[:, :16, :, :]
        x = x[:, 16:, :, :]

        x = self.conv3_4(x)
        x = self.leakyRelu(x)
        x = torch.cat([s1, s2, s3, x], 1)
        x = self.conv1x1(x)
        return x


class IMDNP(nn.Module):
    def __init__(self, args):
        super(IMDNP, self).__init__()
        self.support_PMG = True
        self.head = nn.Sequential(*[common.default_conv(3, 64, 3),
                                    nn.LeakyReLU()])
        self.body = nn.Sequential(*[IMDB() for _ in range(8)])
        self.conv3 = nn.Sequential(*[common.default_conv(64, 64, 3),
                                     nn.LeakyReLU()])
        self.tail = nn.Sequential(*[nn.Sequential(*[common.default_conv(64, 3 * 2 * 2, 3),
                                                    nn.LeakyReLU(),
                                                    nn.PixelShuffle(2)]),
                                    nn.Sequential(*[common.default_conv(64, 3 * 4 * 4, 3),
                                                    nn.LeakyReLU(),
                                                    nn.PixelShuffle(4)])
                                    ])

    def forward(self, x, step=1):
        x = self.head(x)
        res = x
        for i in range(step + 1):
            for j in range(4):
                x = self.body[2 * i + j](x)
        x = self.conv3(x)
        x += res
        x = self.tail[step](x)
        return x
