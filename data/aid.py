# -*- coding: utf-8 -*-
'''
AID数据集
'''
# @Time    : 2024/6/30 15:32
# @Author  : LINYANZHEN
# @File    : aid.py
import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data


class AID(srdata.SRData):
    def __init__(self, args, train=True):
        super(AID, self).__init__(args, train)
        self.repeat = args.test_every // (len(self.lr_list) // args.batch_size)

    def _scan(self):
        '''
        将所有的图片路径打包成列表
        :return:
        '''
        lr_list = []
        # 当需要同时训练多个放大倍率时需要的
        hr_list = []
        if self.args.data_type.find('npy') >= 0 and not self.args.data_type.find('reset') >= 0:
            for root, dirs, files in os.walk(self.lr_dir):
                for file in files:
                    if file[-3:] == "npy":
                        lr_list.append(os.path.join(self.lr_dir,file))
            for root, dirs, files in os.walk(self.hr_dir):
                for file in files:
                    if file[-3:] == "npy":
                        hr_list.append(os.path.join(self.hr_dir,file))
        else:
            for root, dirs, files in os.walk(self.lr_dir):
                for file in files:
                    if file[-3:] == "png":
                        lr_list.append(os.path.join(self.lr_dir,file))
            for root, dirs, files in os.walk(self.hr_dir):
                for file in files:
                    if file[-3:] == "png":
                        hr_list.append(os.path.join(self.hr_dir,file))
        # for i in range(800):
        #     if self.args.data_type.find('npy') >= 0 and not self.args.data_type.find('reset') >= 0:
        #         lr_list.append(os.path.join(self.lr_dir, "{}.npy".format(i)))
        #         hr_list.append(os.path.join(self.hr_dir, "{}.npy".format(i)))
        #     else:
        #         lr_list.append(os.path.join(self.lr_dir, "{}.png".format(i)))
        #         hr_list.append(os.path.join(self.hr_dir, "{}.png".format(i)))

        return lr_list, hr_list

    def _set_filesystem(self):
        self.hr_dir = "dataset/AID/x{}/hr".format(self.args.scale)
        self.lr_dir = "dataset/AID/x{}/lr".format(self.args.scale)

    def __len__(self):
        # return len(self.lr_list) * self.repeat
        return len(self.lr_list)

    def _get_index(self, index):
        # return index % len(self.lr_list)
        return index