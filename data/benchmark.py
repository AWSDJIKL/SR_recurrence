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
        dataset_size = 0
        if self.args.test_set == "Set5":
            dataset_size = 5
        elif self.args.test_set == "Set14":
            dataset_size = 14
        elif self.args.test_set == "BSD500":
            dataset_size = 500
        elif self.args.test_set in ["BSD100", "Urban100"]:
            dataset_size = 100

        for i in range(dataset_size):
            lr_list.append(os.path.join(self.lr_dir, "{}.png".format(i)))
            hr_list.append(os.path.join(self.hr_dir, "{}.png".format(i)))

        return lr_list, hr_list

    def _set_filesystem(self):
        if self.args.test_set in ["Set5", "Set14", "BSD100", "Urban100"]:
            self.lr_dir = "dataset/{}/x{}/lr".format(self.args.test_set, self.args.scale)
            self.hr_dir = "dataset/{}/x{}/hr".format(self.args.test_set, self.args.scale)
        # if self.args.test_set == "Set5":
        #     self.lr_dir = "dataset/Set5/x{}/lr".format(self.args.scale)
        #     self.hr_dir = "dataset/Set5/x{}/hr".format(self.args.scale)
        # elif self.args.test_set == "Set14":
        #     self.lr_dir = "dataset/Set14/x{}/lr".format(self.args.scale)
        #     self.hr_dir = "dataset/Set14/x{}/hr".format(self.args.scale)
        # elif self.args.test_set == "BSD500":
        #     self.lr_dir = "dataset/BSD500/x{}/lr".format(self.args.scale)
        #     self.hr_dir = "dataset/BSD500/x{}/hr".format(self.args.scale)
        # elif self.args.test_set == "Urban100":
        #     self.lr_dir = "dataset/Urban100_SR/x{}/lr".format(self.args.scale)
        #     self.hr_dir = "dataset/Urban100_SR/x{}/hr".format(self.args.scale)
