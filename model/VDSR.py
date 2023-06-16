# -*- coding: utf-8 -*-
'''

'''
import torch
# @Time    : 2022/10/28 11:51
# @Author  : LINYANZHEN
# @File    : VDSR.py
import torch.nn as nn
from model import common


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("VDSR use PMG")
    else:
        print("VDSR")
    return VDSR(args)


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, args):
        super(VDSR, self).__init__()
        self.args = args
        self.support_PMG = True
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        modules_body = [Conv_ReLU_Block() for _ in range(18)]
        self.body = nn.Sequential(*modules_body)
        self.upsample = nn.Upsample(scale_factor=args.scale)
        self.part = args.part

    def forward(self, x, step=-1):
        x = self.upsample(x)
        residual = x
        x = self.conv_in(x)
        x = self.relu(x)
        for i in range(self.part[step]):
            x = self.body[i](x)
        # for i in range(step + 1):
        #     for j in range(6):
        #         x = self.body[6 * i + j](x)
        x = self.conv_out(x)
        x = torch.add(x, residual)
        return x
