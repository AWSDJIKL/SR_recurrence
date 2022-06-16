# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/3/14 21:55
# @Author  : LINYANZHEN
# @File    : test.py
import os
import shutil

from data import common

import numpy as np
# import scipy.misc as misc
import imageio
import torch
import torch.utils.data as data
import tqdm


class test_set(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.scale = args.scale
        self.img_list = []
        for root, dirs, files in os.walk("img_test/test_set"):
            for file in files:
                self.img_list.append(os.path.join(root, file))

    def __getitem__(self, idx):
        lr, hr, img_name = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, img_name

    def __len__(self):
        return len(self.img_list)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        hr = imageio.imread(self.img_list[idx])
        img_name = os.path.splitext(os.path.split(self.img_list[idx])[-1])[0]
        lr = hr
        return lr, hr, img_name

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale
        ih, iw = lr.shape[0:2]
        hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    def clean_npy_dir(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
