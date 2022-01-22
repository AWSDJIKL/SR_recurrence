# -*- coding: utf-8 -*-
'''
在渐进式的基础上融入通道注意力和空间注意力
'''
# @Time    : 2022/1/6 14:19
# @Author  : LINYANZHEN
# @File    : GSACA.py
import math

import torch
import torch.nn as nn
from model import common


class MHSA(nn.Module):
    def __init__(self, channels, width, heigh):
        super(MHSA, self).__init__()
        self.query = nn.Conv2d(channels, channels, (1, 1))
        self.key = nn.Conv2d(channels, channels, (1, 1))
        self.value = nn.Conv2d(channels, channels, (1, 1))

        # self.rel_h = nn.Parameter(torch.randn([1, 3, 1, heigh]), requires_grad=True)
        # self.rel_w = nn.Parameter(torch.randn([1, 3, width, 1]), requires_grad=True)
        # self.width = width
        # self.heigh = heigh
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, w, h = x.size()

        # w_linear = nn.Linear(self.width * 3, w * 3).to("cuda:1")
        # h_linear = nn.Linear(self.heigh * 3, h * 3).to("cuda:1")

        # print("{} {} {} {}".format(b, c, w, h))
        q = self.query(x).view(b, c, -1)
        k = self.key(x).view(b, c, -1)
        v = self.value(x).view(b, c, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)
        # content_position = (self.rel_h + self.rel_w).view(1, c, -1).permute(0, 2, 1)
        # content_position = torch.matmul(content_position, q)

        # energy = content_content + content_position
        # attention = self.softmax(energy)
        attention = self.softmax(content_content)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(b, c, w, h)

        return out + x


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

        return x * sa * ca


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
        # modules_body.append(SALayer())
        modules_body.append(CALayer(n_feat, reduction))
        # modules_body.append(MHSA(n_feat, 48, 48))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
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
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # 预处理的一部分，减去整个数据集的均值
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(4)]

        self.mid_conv = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [
            # conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*[nn.Sequential(*modules_tail) for _ in range(4)])

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
