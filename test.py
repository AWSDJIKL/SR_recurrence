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
    x = torch.randn(1, 64, 4, 0)
    print(x)
    # with open("test.txt", "r", encoding="utf8") as file:
    #     place_num_dict = {}
    #     for line in file.readlines():
    #         places_pattern = re.compile("[\u4e00-\u9fa5]+")
    #         num_pattern = re.compile("[0-9]+")
    #         places = places_pattern.findall(line)
    #         num = num_pattern.findall(line)
    #         # print(places)
    #         # print(num)
    #
    #         for i in range(len(places)):
    #             if places[i] not in place_num_dict.keys():
    #                 place_num_dict[places[i]] = int(num[i])
    #             else:
    #                 place_num_dict[places[i]] += int(num[i])
    #     df=pd.DataFrame(columns=place_num_dict.keys())
    #     # print(df)
    #     sum=0
    #     for k,v in place_num_dict.items():
    #         sum+=v
    #         print(k,v)
    #         df[k]=v
    #     # print(df)
    #     print(sum)
