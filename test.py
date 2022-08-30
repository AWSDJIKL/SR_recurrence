# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/23 10:42
# @Author  : LINYANZHEN
# @File    : test.py
import matplotlib.pyplot as plt
import numpy as np
import torchsummary
import model.MSRN
from option import args
import torch
import torch.nn.functional as F

k = 32
p = 0
s = 8

tensor = torch.range(0, 10 * 3 * 512 * 512 - 1).view(10, 3, 512, 512)
# print(tensor)

unfold = F.unfold(tensor, kernel_size=(k, k), stride=s, padding=p).permute(0, 2, 1)
b, c, l = unfold.size()
b = (b * c * l) // (3 * k * k)
print(b)
unfold = unfold.reshape(b, 3, k, k)
print(unfold)
print(unfold.shape)
