# -*- coding: utf-8 -*-
'''
1.在渐进式的基础上融入通道注意力和空间注意力
2.使用ConvNeXt改进卷积核
'''
# @Time    : 2022/1/6 14:19
# @Author  : LINYANZHEN
# @File    : GSACA.py
import math

import torch
import torch.nn as nn
from model import common
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.nn.functional as F


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("GSACA_PMG")
    else:
        print("GSACA")
    return GSACA(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca = self.avg_pool(x)
        ca = self.conv_du(ca)

        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avgout, maxout], dim=1)
        sa = self.sigmoid(self.conv2d(sa))

        return (x * ca) + sa


# SpatialAttention（SA）Layer
class SALayer(nn.Module):
    def __init__(self):
        super(SALayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avgout, maxout], dim=1)
        sa = self.sigmoid(self.conv2d(sa))
        return x * sa


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class MSRB(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = CALayer(n_feats)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        # output = self.ca(output)
        output += x
        return output


class FMSRB(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(FMSRB, self).__init__()
        self.conv3_1_1 = nn.Conv2d(n_feats, n_feats, (3, 1), (1, 1), (1, 0))
        self.conv3_1_2 = nn.Conv2d(n_feats, n_feats, (1, 3), (1, 1), (0, 1))
        self.conv5_1_1 = nn.Conv2d(n_feats, n_feats, (5, 1), (1, 1), (2, 0))
        self.conv5_1_2 = nn.Conv2d(n_feats, n_feats, (1, 5), (1, 1), (0, 2))

        self.down_sample_1 = nn.Conv2d(n_feats * 2, n_feats, (1, 1))
        self.down_sample_2 = nn.Conv2d(n_feats * 2, n_feats, (1, 1))

        self.conv3_2_1 = nn.Conv2d(n_feats, n_feats, (3, 1), (1, 1), (1, 0))
        self.conv3_2_2 = nn.Conv2d(n_feats, n_feats, (1, 3), (1, 1), (0, 1))
        self.conv5_2_1 = nn.Conv2d(n_feats, n_feats, (5, 1), (1, 1), (2, 0))
        self.conv5_2_2 = nn.Conv2d(n_feats, n_feats, (1, 5), (1, 1), (0, 2))

        self.down_sample_3 = nn.Conv2d(n_feats * 2, n_feats, (1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.ca = CALayer(n_feats)

    def forward(self, x):
        res = x

        s1 = self.conv3_1_1(x)
        s1 = self.conv3_1_2(s1)
        s1 = self.relu(s1)
        p1 = self.conv5_1_1(x)
        p1 = self.conv5_1_2(p1)
        p1 = self.relu(p1)

        s1_p1 = self.down_sample_1(torch.cat([s1, p1], 1))
        p1_s1 = self.down_sample_2(torch.cat([p1, s1], 1))

        s2 = self.conv3_2_1(s1_p1)
        s2 = self.conv3_2_2(s2)
        s2 = self.relu(s2)
        p2 = self.conv5_2_1(p1_s1)
        p2 = self.conv5_2_2(p2)
        p2 = self.relu(p2)

        output = self.down_sample_3(torch.cat([s2, p2], 1))
        output = self.ca(output)
        output += res

        return output


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            # RCAB(
            #     conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            MSRB()
            # FMSRB()
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class GSACA(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(GSACA, self).__init__()
        self.support_PMG = True
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        act = nn.ReLU(True)
        self.is_PMG = args.is_PMG
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # 预处理的一部分，减去整个数据集的均值
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(4)]

        self.body = nn.Sequential(*modules_body)

        modules_tail = [nn.Sequential(*[
            # conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]) for i in range(4)]
        self.tail = nn.Sequential(*modules_tail)

        self.mid_conv = conv(n_feats, n_feats, kernel_size)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)

    def forward(self, x, step=3):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in range(step + 1):
            x = self.body[i](x)
        x = self.mid_conv(x)
        x += res
        x = self.tail[step](x)
        x = self.add_mean(x)
        return x
