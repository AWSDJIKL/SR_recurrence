# -*- coding: utf-8 -*-
'''
DIV2k+Flickr2K
'''
# @Time : 2022/10/31 15:48
# @Author : LINYANZHEN
# @File : df2k.py
import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data


class DF2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DF2K, self).__init__(args, train)
        self.repeat = args.test_every // (len(self.lr_list) // args.batch_size)

    def _scan(self):
        '''
        将所有的图片路径打包成列表
        :return:
        '''
        lr_list = []
        # 当需要同时训练多个放大倍率时需要的
        hr_list = []
        for i in range(800):
            if self.args.data_type.find('npy') >= 0 and not self.args.data_type.find('reset') >= 0:
                lr_list.append(os.path.join(self.lr_dir, "{}.npy".format(i)))
                hr_list.append(os.path.join(self.hr_dir, "{}.npy".format(i)))
            else:
                lr_list.append(os.path.join(self.lr_dir, "{}.png".format(i)))
                hr_list.append(os.path.join(self.hr_dir, "{}.png".format(i)))
        for i in range(2650):
            if self.args.data_type.find('npy') >= 0 and not self.args.data_type.find('reset') >= 0:
                lr_list.append(os.path.join(self.lr_dir, "{}.npy".format(i)))
                hr_list.append(os.path.join(self.hr_dir, "{}.npy".format(i)))
            else:
                lr_list.append(os.path.join(self.lr_dir, "{}.png".format(i)))
                hr_list.append(os.path.join(self.hr_dir, "{}.png".format(i)))


        return lr_list, hr_list

    def _set_filesystem(self):
        self.div_hr_dir = "dataset/DIV2K_train_HR/x{}/hr".format(self.args.scale)
        self.div_lr_dir = "dataset/DIV2K_train_HR/x{}/lr".format(self.args.scale)
        self.flickr_hr_dir = "dataset/Flickr2K/x{}/hr".format(self.args.scale)
        self.flickr_lr_dir = "dataset/Flickr2K/x{}/lr".format(self.args.scale)

    def __len__(self):
        # return len(self.lr_list) * self.repeat
        return len(self.lr_list)

    def _get_index(self, index):
        # return index % len(self.lr_list)
        return index
