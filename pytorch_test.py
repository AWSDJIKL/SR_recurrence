# -*- coding: utf-8 -*-
'''

'''
# @Time : 2022/11/7 17:35
# @Author : LINYANZHEN
# @File : pytorch_test.py
import torch
import torch.nn as nn

a = torch.tensor([[[[1, 1],
                    [2, 3]],
                   [[3, 4],
                    [3, 5]],
                   [[5, 7],
                    [7, 7]]]]).float()
print(a)
print(a.size())
a_mean = nn.AdaptiveAvgPool2d(1)(a)
print(a_mean)
a = torch.pow(a - a_mean, 2)
print(a)
# print(a.size())
a_mean = nn.AdaptiveAvgPool2d(1)(a)
print(a_mean)
a_sqrt = torch.sqrt(a_mean)
print(a_sqrt)
