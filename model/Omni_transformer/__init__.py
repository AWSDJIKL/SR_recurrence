# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2023/11/4 22:21
# @Author  : LINYANZHEN
# @File    : __init__.py


import torch
import torch.nn as nn


def make_model(args, parent=False):
    print("prepare model")
    if args.is_PMG:
        print("Omni_transformer use PMG")
    else:
        print("Omni_transformer")
    return Omni_transformer(args)


class Omni_transformer(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
