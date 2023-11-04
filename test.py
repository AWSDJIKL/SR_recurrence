# -*- coding: utf-8 -*-
'''

'''
import re

import PIL.Image
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
import imageio
from PIL import Image
import torchvision.transforms
import pandas as pd

if __name__ == '__main__':
    print(args.part)
    # 打开文件并读取每一行
    with open('checkpoint/x4_IMDN_plus_PMG_1_1_stride1.0/option.txt', 'r') as f:
        lines = f.readlines()
    # 创建空字典，并将每一行转换为键值对存入字典
    args_dict = {}
    for line in lines:
        key, value = line.split(' ', 1)
        args_dict[key] = value
    args.part = args_dict['part']
    # args.part = eval(args_dict['part'])
    print(type(args.part))