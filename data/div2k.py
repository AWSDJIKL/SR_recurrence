# -*- coding: utf-8 -*-
'''
DIV2K数据集
'''
# @Time    : 2021/12/19 10:55
# @Author  : LINYANZHEN
# @File    : div2k.py
import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data


class DIV2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        self.repeat = args.test_every // (len(self.lr_list) // args.batch_size)

    def _scan(self):
        '''
        将所有的图片路径打包成列表
        :return:
        '''
        lr_list = []
        # 当需要同时训练多个放大倍率时需要的
        hr_list = []
        for i in range(1, 801):
            if self.args.data_type.find('npy') >= 0 and not self.args.data_type.find('reset') >= 0:
                lr_list.append(os.path.join(self.lr_dir, "{:0>4}x4.npy".format(i)))
                hr_list.append(os.path.join(self.hr_dir, "{:0>4}.npy".format(i)))
            else:
                lr_list.append(os.path.join(self.lr_dir, "{:0>4}x4.png".format(i)))
                hr_list.append(os.path.join(self.hr_dir, "{:0>4}.png".format(i)))

        return lr_list, hr_list

    def _set_filesystem(self):
        if self.args.data_type.find('img') >= 0 or self.args.data_type.find('reset') >= 0:
            self.hr_dir = "dataset/DIV2K_train_HR/DIV2K_train_HR"
            self.lr_dir = "dataset/DIV2K_train_LR/DIV2K_train_LR_bicubic/X4"
        else:
            self.hr_dir = "dataset/DIV2K_train_HR/DIV2K_train_HR/npy"
            self.lr_dir = "dataset/DIV2K_train_LR/DIV2K_train_LR_bicubic/X4/npy"

    def __len__(self):
        return len(self.lr_list) * self.repeat

    def _get_index(self, index):
        return index % len(self.lr_list)
