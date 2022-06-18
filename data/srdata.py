# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/19 10:52
# @Author  : LINYANZHEN
# @File    : srdata.py
import os
import shutil

from data import common

import numpy as np
# import scipy.misc as misc
import imageio
import torch
import torch.utils.data as data
import tqdm
from PIL import Image
from torchvision import transforms


class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem()

        if args.data_type == 'img' or benchmark:  # 选择直接读取图片；当数据集是验证集时，也选择直接读取图片
            # 直接读取图片
            self.lr_list, self.hr_list = self._scan()
        elif args.data_type.find('npy') >= 0:
            # 将图片转为numpy数组并保存为.npy文件，加速读取
            self.lr_list, self.hr_list = self._scan()
            if args.data_type.find('reset') >= 0:
                print('Preparing numpy files')
                lr_save_dir = os.path.join(self.lr_dir, "npy")
                hr_save_dir = os.path.join(self.hr_dir, "npy")
                self.clean_npy_dir(lr_save_dir)
                self.clean_npy_dir(hr_save_dir)
                lr_list = []
                hr_list = []
                # 遍历图片，保存为npy文件
                for index, (lr_path, hr_path) in enumerate(zip(self.lr_list, self.hr_list)):
                    lr = imageio.imread(lr_path)
                    hr = imageio.imread(hr_path)
                    lr_npy_path = os.path.join(lr_save_dir, "{:0>4}x4.npy".format(index + 1))
                    hr_npy_path = os.path.join(hr_save_dir, "{:0>4}.npy".format(index + 1))
                    np.save(lr_npy_path, lr)
                    np.save(hr_npy_path, hr)
                    lr_list.append(lr_npy_path)
                    hr_list.append(hr_npy_path)
                self.lr_list = lr_list
                self.hr_list = hr_list
        else:
            print('Please define data type')

    def _scan(self):
        '''
        定义每个数据集的lr，hr路径
        :return:
        '''
        raise NotImplementedError

    def _set_filesystem(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, img_name = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, img_name

    def __len__(self):
        return len(self.lr_list)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.lr_list[idx]
        hr = self.hr_list[idx]
        img_name = os.path.splitext(os.path.split(hr)[-1])[0]
        if self.args.data_type == 'img' or self.benchmark:
            lr = imageio.imread(lr)
            hr = imageio.imread(hr)
        elif self.args.data_type.find('npy') >= 0:
            lr = np.load(lr)
            hr = np.load(hr)
        return lr, hr, img_name

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale
        if self.train:
            # 是训练集，因为图片大小不一样，放不进同一个batch，需要对图片进行切割
            lr, hr = common.get_patch(lr, hr, patch_size, scale)
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            # 将hr切割，保证sr图片大小与hr匹配
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    def clean_npy_dir(self, dir):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
