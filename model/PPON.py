# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/10/25 21:59
# @Author  : LINYANZHEN
# @File    : PPON.py

from model.common import conv_layer, Upsampler, default_conv, MeanShift
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


def make_model(args, parent=False):
    print("prepare model")
    if args.is_PMG:
        print("PPON_PMG")
    else:
        print("PPON")
    return PPON(args)


def activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class _ResBlock_32(nn.Module):
    def __init__(self, nc=64):
        super(_ResBlock_32, self).__init__()
        self.c1 = conv_layer(nc, nc, 3, 1, 1)
        self.d1 = conv_layer(nc, nc // 2, 3, 1, 1)  # rate=1
        self.d2 = conv_layer(nc, nc // 2, 3, 1, 2)  # rate=2
        self.d3 = conv_layer(nc, nc // 2, 3, 1, 3)  # rate=3
        self.d4 = conv_layer(nc, nc // 2, 3, 1, 4)  # rate=4
        self.d5 = conv_layer(nc, nc // 2, 3, 1, 5)  # rate=5
        self.d6 = conv_layer(nc, nc // 2, 3, 1, 6)  # rate=6
        self.d7 = conv_layer(nc, nc // 2, 3, 1, 7)  # rate=7
        self.d8 = conv_layer(nc, nc // 2, 3, 1, 8)  # rate=8
        self.act = activation('lrelu')
        self.c2 = conv_layer(nc * 4, nc, 1, 1, 1)  # 256-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d5 = self.d5(output1)
        d6 = self.d6(output1)
        d7 = self.d7(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        add4 = add3 + d5
        add5 = add4 + d6
        add6 = add5 + d7
        add7 = add6 + d8

        combine = torch.cat([d1, add1, add2, add3, add4, add5, add6, add7], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2.mul(0.2)

        return output


class RRBlock_32(nn.Module):
    def __init__(self, nb):
        super(RRBlock_32, self).__init__()
        body = nn.ModuleList()
        for i in range(nb):
            body.append(_ResBlock_32())
        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        return out.mul(0.2) + x


class PPON(nn.Module):
    def __init__(self, args):
        super(PPON, self).__init__()
        self.args = args
        n_feats = 64
        self.nbs = [3, 3, 3, 3]
        scale = args.scale
        self.head = conv_layer(args.n_colors, 64, 3)
        modules_body = nn.ModuleList()
        for i in range(4):
            modules_body.append(RRBlock_32(self.nbs[i]))
        modules_tail = [
            conv_layer(n_feats, n_feats, 3),
            Upsampler(default_conv, scale, n_feats, act=False),
            conv_layer(n_feats, args.n_colors, 3)]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x, step=None):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x

        conv_out = []
        if step:
            for i in range((step + 1)):
                x = self.body[i](x)
        else:
            x = self.body(x)
        x = self.tail(res)
        x = self.add_mean(x)
        return x
