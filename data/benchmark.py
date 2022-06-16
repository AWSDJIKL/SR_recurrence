# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/19 10:56
# @Author  : LINYANZHEN
# @File    : benchmark.py
import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data


class Benchmark(srdata.SRData):
    def __init__(self, args, train=False):
        super(Benchmark, self).__init__(args, train, benchmark=True)
        self.args = args

    def _scan(self):
        lr_list = []
        hr_list = []
        if self.args.test_set == "Set5":
            for i in range(1, 6):
                lr_list.append(os.path.join(self.dir_lr, "img_{:0>3}_SRF_{}_LR.png".format(i, self.args.scale)))
                hr_list.append(os.path.join(self.dir_hr, "img_{:0>3}_SRF_{}_HR.png".format(i, self.args.scale)))
        elif self.args.test_set == "Set14":
            for i in range(1, 15):
                lr_list.append(os.path.join(self.dir_lr, "img_{:0>3}_SRF_{}_LR.png".format(i, self.args.scale)))
                hr_list.append(os.path.join(self.dir_hr, "img_{:0>3}_SRF_{}_HR.png".format(i, self.args.scale)))
        elif self.args.test_set == "BSD100":
            for i in range(1, 101):
                lr_list.append(os.path.join(self.dir_lr, "img_{:0>3}_SRF_{}_LR.png".format(i, self.args.scale)))
                hr_list.append(os.path.join(self.dir_hr, "img_{:0>3}_SRF_{}_HR.png".format(i, self.args.scale)))
        elif self.args.test_set == "Urban100":
            for i in range(1, 101):
                lr_list.append(os.path.join(self.dir_lr, "img_{:0>3}_SRF_{}_LR.png".format(i, self.args.scale)))
                hr_list.append(os.path.join(self.dir_hr, "img_{:0>3}_SRF_{}_HR.png".format(i, self.args.scale)))
        return lr_list, hr_list

    def _set_filesystem(self):
        if self.args.test_set == "Set5":
            self.dir_hr = "dataset/set5/Set5/image_SRF_4"
            self.dir_lr = "dataset/set5/Set5/image_SRF_4"
        elif self.args.test_set == "Set14":
            self.dir_hr = "dataset/set14/Set14/image_SRF_4"
            self.dir_lr = "dataset/set14/Set14/image_SRF_4"
        elif self.args.test_set == "BSD100":
            self.dir_hr = "dataset/BSD100_SR/image_SRF_4"
            self.dir_lr = "dataset/BSD100_SR/image_SRF_4"
        elif self.args.test_set == "Urban100":
            self.dir_hr = "dataset/Urban100_SR/image_SRF_4"
            self.dir_lr = "dataset/Urban100_SR/image_SRF_4"
