# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/10/24 21:51
# @Author  : LINYANZHEN
# @File    : ConvSR.py
from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


def make_model(args, parent=False):
    print("prepare model")
    if args.is_PMG:
        print("ConvSR")
    else:
        print("ConvSR")
    return ConvSR(args)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, depth):
        super(ConvBlock, self).__init__()
        self.body = nn.Sequential(
            *[Block(dim=dim) for i in range(depth)]
        )

    def forward(self, x):
        input = x
        x = self.body(x)
        x = input + x
        return x


class ConvSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ConvSR, self).__init__()
        self.support_PMG = True
        n_feats = 64

        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)

        self.depths = [3, 3, 9, 3]
        self.piece = len(args.crop_piece) + 1
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in self.depths:
            modules_body.append(Block(n_feats))

        channel_adaptive = [
            nn.Conv2d(n_feats * (i + 1), n_feats * (4 + 1), 1, padding=0, stride=1)
            for i in range(1, self.piece)
        ]

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats * (4 + 1), n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.channel_adaptive = nn.Sequential(*channel_adaptive)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, step=None):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = x

        conv_out = []
        if step:
            for i in range((step + 1)):
                x = self.body[i](x)
                conv_out.append(x)
            conv_out.append(res)
            res = torch.cat(conv_out, 1)
            if step < self.piece:
                res = self.channel_adaptive[step](res)
        else:
            for i in range(4):
                x = self.body[i](x)
                conv_out.append(x)
            conv_out.append(res)
            res = torch.cat(conv_out, 1)
        x = self.tail(res)
        # x = self.add_mean(x)
        return x
