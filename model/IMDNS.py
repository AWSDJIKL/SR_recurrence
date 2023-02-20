# -*- coding: utf-8 -*-
'''
对比实验，验证分多少个阶段更合适
'''
# @Time    : 2023/1/28 19:47
# @Author  : LINYANZHEN
# @File    : IMDNS.py
import torch
import torch.nn as nn
from model import common


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("IMDNS use PMG")
    else:
        print("IMDNS")
    return IMDNS(args)


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


class IMDNS(nn.Module):
    def __init__(self, args):
        super(IMDNS, self).__init__()
        self.support_PMG = True
        self.head = nn.Sequential(*[common.default_conv(3, 64, 3),
                                    nn.LeakyReLU()])
        self.body = nn.Sequential(*[IMDB() for _ in range(8)])
        self.conv3 = nn.Sequential(*[common.default_conv(64, 64, 3),
                                     nn.LeakyReLU()])
        self.tail = nn.Sequential(*[common.default_conv(64, 3 * args.scale * args.scale, 3),
                                    nn.LeakyReLU(),
                                    nn.PixelShuffle(args.scale)])
        self.part = args.part

    def forward(self, x, step=-1):
        x = self.head(x)
        res = x
        for i in range(self.part[step]):
            x = self.body[i](x)
        x = self.conv3(x)
        x += res
        x = self.tail(x)
        return x
