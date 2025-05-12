# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2024/8/12 11:29
# @Author  : LINYANZHEN
# @File    : IMDN_plus_APMG.py
import torch
import torch.nn as nn
from collections import OrderedDict


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("IMDN_plus_APMG use APMG")
    else:
        print("IMDN_plus_APMG")
    return IMDN_plus_APMG(args)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


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


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class IMDB_plus(nn.Module):
    def __init__(self, in_channels, distillation_rate=1 / 6):
        super(IMDB_plus, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)  # 6
        self.c1 = conv_layer(in_channels, in_channels, 3)  # 36 --> 36
        self.c2 = conv_layer(self.distilled_channels * 5, self.distilled_channels * 5, 3)  # 30 --> 30
        self.c3 = conv_layer(self.distilled_channels * 4, self.distilled_channels * 4, 3)  # 24 --> 24
        self.c4 = conv_layer(self.distilled_channels * 3, self.distilled_channels * 3, 3)  # 18 --> 18
        self.c5 = conv_layer(self.distilled_channels * 2, self.distilled_channels * 2, 3)  # 12 --> 12
        self.c6 = conv_layer(self.distilled_channels * 1, self.distilled_channels * 1, 3)  # 6 --> 6
        self.act = nn.SiLU()
        self.c7 = conv_layer(self.distilled_channels * 6, in_channels, 1)

    def load_weight(self, block, load_mode="momentum", m=0.998):
        if load_mode == "direct":
            self.c1.weight = block.c1.weight
            self.c1.bias = block.c1.bias
            self.c2.weight = block.c2.weight
            self.c2.bias = block.c2.bias
            self.c3.weight = block.c3.weight
            self.c3.bias = block.c3.bias
            self.c4.weight = block.c4.weight
            self.c4.bias = block.c4.bias
            self.c5.weight = block.c5.weight
            self.c5.bias = block.c5.bias
            self.c6.weight = block.c6.weight
            self.c6.bias = block.c6.bias
            self.c7.weight = block.c7.weight
            self.c7.bias = block.c7.bias
        elif load_mode == "momentum":
            self.c1.weight = torch.nn.Parameter(m * self.c1.weight + (1 - m) * block.c1.weight)
            self.c1.bias = torch.nn.Parameter(m * self.c1.bias + (1 - m) * block.c1.bias)
            self.c2.weight = torch.nn.Parameter(m * self.c2.weight + (1 - m) * block.c2.weight)
            self.c2.bias = torch.nn.Parameter(m * self.c2.bias + (1 - m) * block.c2.bias)
            self.c3.weight = torch.nn.Parameter(m * self.c3.weight + (1 - m) * block.c3.weight)
            self.c3.bias = torch.nn.Parameter(m * self.c3.bias + (1 - m) * block.c3.bias)
            self.c4.weight = torch.nn.Parameter(m * self.c4.weight + (1 - m) * block.c4.weight)
            self.c4.bias = torch.nn.Parameter(m * self.c4.bias + (1 - m) * block.c4.bias)
            self.c5.weight = torch.nn.Parameter(m * self.c5.weight + (1 - m) * block.c5.weight)
            self.c5.bias = torch.nn.Parameter(m * self.c5.bias + (1 - m) * block.c5.bias)
            self.c6.weight = torch.nn.Parameter(m * self.c6.weight + (1 - m) * block.c6.weight)
            self.c6.bias = torch.nn.Parameter(m * self.c6.bias + (1 - m) * block.c6.bias)
            self.c7.weight = torch.nn.Parameter(m * self.c7.weight + (1 - m) * block.c7.weight)
            self.c7.bias = torch.nn.Parameter(m * self.c7.bias + (1 - m) * block.c7.bias)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))  # 36 --> 36
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.distilled_channels * 5),
                                                 dim=1)  # 6, 30
        out_c2 = self.act(self.c2(remaining_c1))  # 30 --> 30
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.distilled_channels * 4),
                                                 dim=1)  # 6, 24
        out_c3 = self.act(self.c3(remaining_c2))  # 24 --> 24
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.distilled_channels * 3),
                                                 dim=1)  # 6, 18
        out_c4 = self.act(self.c4(remaining_c3))  # 18 --> 18
        distilled_c4, remaining_c4 = torch.split(out_c4, (self.distilled_channels, self.distilled_channels * 2),
                                                 dim=1)  # 6, 12
        out_c5 = self.act(self.c5(remaining_c4))  # 12 --> 12
        distilled_c5, remaining_c5 = torch.split(out_c5, (self.distilled_channels, self.distilled_channels),
                                                 dim=1)  # 6, 6
        out_c6 = self.act(self.c6(remaining_c5))  # 6 --> 6

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4, distilled_c5, out_c6], dim=1)
        out_fused = self.c7(out) + x
        return out_fused


class IMDN_plus_APMG(nn.Module):
    def __init__(self, args):
        super(IMDN_plus_APMG, self).__init__()
        self.support_PMG = True
        self.part = args.part
        in_nc = 3
        self.nf = 36
        self.max_nb = 16
        if args.is_PMG:
            self.now_nb = 1
        else:
            self.now_nb = 16
        out_nc = 3
        rb_blocks = [IMDB_plus(self.nf) for _ in range(self.now_nb)]
        self.conv3 = conv_layer(self.nf, self.nf, kernel_size=3)
        upsample_block = pixelshuffle_block
        self.head = conv_layer(in_nc, self.nf, kernel_size=3)
        self.body = nn.Sequential(*rb_blocks)
        self.momentum_body = nn.Sequential(*rb_blocks)

        self.tail = upsample_block(self.nf, out_nc, upscale_factor=args.scale)

    def grow(self):
        if self.now_nb < self.max_nb:
            rb_blocks = [IMDB_plus(self.nf) for _ in range(self.now_nb * 2)]
            momentum_rb_blocks = [IMDB_plus(self.nf) for _ in range(self.now_nb * 2)]
            # 根据已有的层计算权重
            for i in range(self.now_nb):
                rb_blocks[2 * i - 1].load_weight(self.momentum_body[i], "direct")
                rb_blocks[2 * i].load_weight(self.momentum_body[i], "direct")
                momentum_rb_blocks[2 * i - 1].load_weight(self.momentum_body[i], "direct")
                momentum_rb_blocks[2 * i].load_weight(self.momentum_body[i], "direct")
            self.body = nn.Sequential(*rb_blocks)
            self.momentum_body = nn.Sequential(*momentum_rb_blocks)
            self.now_nb = self.now_nb * 2

    def renew_momentum(self):
        for i in range(self.now_nb):
            self.momentum_body[i].load_weight(self.body[i], "momentum")

    def forward(self, x):
        x = self.head(x)
        res = x
        x = self.body(x)
        x = self.conv3(x)
        x += res
        x = self.tail(x)
        return x
