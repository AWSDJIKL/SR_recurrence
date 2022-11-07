# -*- coding: utf-8 -*-
'''

'''
import math

# @Time    : 2022/10/25 21:59
# @Author  : LINYANZHEN
# @File    : PPON.py

from model.common import conv_layer, Upsampler, default_conv, MeanShift
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from collections import OrderedDict


def make_model(args, parent=False):
    print("prepare model")
    if args.is_PMG:
        print("PPON use PMG")
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


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def upconv_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1, act_type='relu'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest')
    conv = conv_layer(in_channels, out_channels, kernel_size, stride)
    act = activation(act_type)
    return sequential(upsample, conv, act)


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
        self.support_PMG = True
        self.args = args
        n_feats = 64
        scale = args.scale
        self.fea_conv = conv_layer(3, n_feats, kernel_size=3)  # common
        rb_blocks = [RRBlock_32(3) for _ in range(24)]
        self.LR_conv = conv_layer(n_feats, n_feats, kernel_size=3)
        act_type = "lrelu"
        upsample_block = upconv_block
        n_upscale = int(math.log(scale, 2))
        if scale == 3:
            n_upscale = 1
        if scale == 3:
            upsampler = upsample_block(n_feats, n_feats, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(n_feats, n_feats, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = conv_block(n_feats, n_feats, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(n_feats, 3, kernel_size=3, norm_type=None, act_type=None)
        self.body = nn.Sequential(*rb_blocks)
        self.CRM = sequential(*upsampler, HR_conv0, HR_conv1)  # recon content

    def forward(self, x, step=3):
        # x = self.sub_mean(x)
        x = self.fea_conv(x)
        res = x
        if step:
            for i in range((step + 1)):
                for j in range(6):
                    x = self.body[6 * i + j](x)
        else:
            x = self.body(x)
        x = self.LR_conv(x)
        res = x + res
        x = self.CRM(res)
        # x = self.add_mean(x)
        return x
