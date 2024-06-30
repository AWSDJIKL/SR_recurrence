# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/19 10:56
# @Author  : LINYANZHEN
# @File    : common.py
import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms


def get_patch(img_in, img_tar, patch_size, scale):
    '''

    :param img_in:
    :param img_tar:
    :param patch_size:
    :param scale:
    :return:
    '''
    ih, iw = img_in.shape[:2]

    tp = patch_size
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar


def crop_img(img, patch_size):
    ih, iw = img.shape[:2]
    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)
    img = img[iy:iy + patch_size, ix:ix + patch_size, :]
    return img


def set_channel(l, n_channel):
    '''
    调整图片的通道数，使其符合要求
    :param l: 原始图片
    :param n_channel: 要求的通道数
    :return:
    '''

    def _set_channel(img):
        # 只有2维，即只有h,w，增加通道维度
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


def np2Tensor(l, rgb_range):
    '''
    将numpy数组转为tensor
    :param l:
    :param rgb_range:
    :return:
    '''

    def _np2Tensor(img):
        # 首先转换维度，从(h,w,c)转为(c,h,w)
        # ascontiguousarray：将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        # 转tensor
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def add_noise(x, noise='.'):
    '''
    添加噪声
    :param x:
    :param noise:
    :return:
    '''
    if noise != '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(l, hflip=True, rot=True):
    '''
    图像增强，对图片随机旋转或者翻转
    :param l:
    :param hflip:
    :param rot:
    :return:
    '''
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]
