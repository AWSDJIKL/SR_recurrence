# -*- coding: utf-8 -*-
'''
https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf
多尺度残差网络，改造为PMG版本
'''
# @Time    : 2022/6/23 21:40
# @Author  : LINYANZHEN
# @File    : MSRN_PMG.py
from model import common
import torch
import torch.nn as nn


def make_model(args, parent=False):
    print("prepare model")
    if args.is_PMG:
        print("MSRN_PMG")
    else:
        print("MSRN")
    return MSRN_PMG(args)


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

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class MSRN_PMG(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MSRN_PMG, self).__init__()
        self.support_PMG = True
        n_feats = 64
        n_blocks = 8
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)

        self.n_blocks = n_blocks

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                MSRB(n_feats=n_feats))

        channel_adaptive = [
            nn.Conv2d(n_feats * (self.n_blocks // 4 + 1), n_feats * (self.n_blocks + 1), 1, padding=0, stride=1),
            nn.Conv2d(n_feats * ((self.n_blocks // 4) * 2 + 1), n_feats * (self.n_blocks + 1), 1, padding=0, stride=1),
            nn.Conv2d(n_feats * ((self.n_blocks // 4) * 3 + 1), n_feats * (self.n_blocks + 1), 1, padding=0, stride=1),
        ]

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats * (self.n_blocks + 1), n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.channel_adaptive = nn.Sequential(*channel_adaptive)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, step=3):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x

        MSRB_out = []
        # 将8个block分为4个部分
        for i in range((step + 1) * 2):
            x = self.body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)
        res = torch.cat(MSRB_out, 1)
        if step < 3:
            res = self.channel_adaptive[step](res)
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
