# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/10/28 12:07
# @Author  : LINYANZHEN
# @File    : EDSR.py

import torch.nn as nn
from model import common


def make_model(args):
    print("prepare model")
    if args.is_PMG:
        print("EDSR use PMG")
    else:
        print("EDSR")
    return EDSR(args)


class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.support_PMG = True
        n_resblocks = 16  # 16
        n_feats = 64  # 64
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.part = args.part

    def forward(self, x, step=-1):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in range(self.part[step]):
            res = self.body[i](res)
        # for i in range(step + 1):
        #     for j in range(4):
        #         res = self.body[4 * i + j](res)
        res = self.body[-1](res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
