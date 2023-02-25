# -*- coding: utf-8 -*-
'''

'''
# @Time : 2023/2/20 10:03
# @Author : LINYANZHEN
# @File : __init__.py.py

import torch
import torch.nn as nn
from . import block as B


def make_model(args, parent=False):
    print("prepare model")
    if args.is_PMG:
        print("HNCT use PMG")
    else:
        print("HNCT")
    return HNCT(args)


class Cascade(nn.Module):
    def __init__(self, ):
        super(Cascade, self).__init__()
        self.conv1 = B.conv_layer(50, 50, kernel_size=1)
        self.conv3 = B.conv_layer(50, 50, kernel_size=3)
        self.conv5 = B.conv_layer(50, 50, kernel_size=5)
        self.c = B.conv_block(50 * 4, 50, kernel_size=1, act_type='lrelu')

    def forward(self, x):
        conv5 = self.conv5(x)
        extra = x + conv5
        conv3 = self.conv3(extra)
        extra = x + conv3
        conv1 = self.conv1(extra)
        cat = torch.cat([conv5, conv3, conv1, x], dim=1)
        input = self.c(cat)
        return input


class HNCT(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(HNCT, self).__init__()
        self.support_PMG = True
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.body = nn.Sequential(*[B.HBCT(in_channels=nf) for _ in range(4)])
        self.channel_adaptive = nn.Sequential(
            *[B.conv_block(nf * i, nf * 4, kernel_size=1, act_type="lrelu") for i in range(1, 4)])
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, input, step=3):
        x = self.fea_conv(input)
        res = x
        hbct_out = []
        for i in range(step + 1):
            x = self.body[i](x)
            hbct_out.append(x)
        hbct_out = torch.cat(hbct_out, dim=1)
        if step < 3:
            hbct_out = self.channel_adaptive[step](hbct_out)
        out_B = self.c(hbct_out)
        out_lr = self.LR_conv(out_B) + res
        output = self.upsampler(out_lr)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
