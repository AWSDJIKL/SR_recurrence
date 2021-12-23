# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/3 11:59
# @Author  : LINYANZHEN
# @File    : utils.py
import shutil

import torch
import torchvision
from PIL import Image
import os
import math
import time
import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision import transforms
import imageio
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


class Checkpoint():
    def __init__(self, args, model, experiment_name):
        self.args = args
        self.log = torch.Tensor()
        self.model = model
        self.loss_list = []
        self.psnr_list = []
        self.best_psnr = 0
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        self.checkpoint_dir = os.path.join("checkpoint", experiment_name)
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.mkdir(self.checkpoint_dir)
        self.log_file_path = os.path.join(self.checkpoint_dir, "log.txt")
        # 先写开头部分
        with open(self.log_file_path, "w") as f:
            f.write(now + "\n")

    def record_epoch(self, epoch, epoch_loss, epoch_psnr):
        if epoch_psnr > self.best_psnr:
            self.best_psnr = epoch_psnr
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best.pth"))
            print("模型已保存")
        self.write_log("epoch :{}".format(epoch))
        self.write_log("loss :{}".format(epoch_loss))
        self.write_log("psnr :{}    best psnr:{}".format(epoch_psnr, self.best_psnr))

    def save_final(self):
        self.plot_loss()
        self.plot_psnr()
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "final.pth"))

    def write_log(self, log):
        print(log)
        with open(self.log_file_path, "a") as f:
            f.write(log)
            f.write("\n")

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_list, 'b', label='loss')
        plt.legend()
        plt.grid()
        plt.savefig('{}/loss.png'.format(self.checkpoint_dir), dpi=256)
        plt.close()

    def plot_psnr(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.psnr_list, 'b', label='psnr')
        plt.legend()
        plt.grid()
        plt.title('best psnr=%5.2f' % self.best_psnr)
        plt.savefig('{}/psnr.png'.format(self.checkpoint_dir), dpi=256)
        plt.close()


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calculate_psnr(sr, hr, scale, rgb_range, benchmark=False):
    '''
    计算两张图片之间的PSNR误差

    :param img1:
    :param img2:
    :return:
    '''

    if len(sr.size()) < 4:
        sr = sr.unsqueeze(0)
    if len(hr.size()) < 4:
        hr = hr.unsqueeze(0)
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        # print(diff)
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

    # return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2)).item()


def calculate_ssim(img1, img2, kernel_size=11):
    '''

    :param img1:
    :param img2:
    :param kernel_size: 滑动窗口大小
    :return:
    '''

    def create_kernel(kernel_size, channel):
        '''
        创建计算核并分配权重
        :param kernel_size:
        :return:
        '''
        # 仅计算平均数
        kernel = torch.Tensor([[
            [[1 for i in range(kernel_size)] for i in range(kernel_size)]
            for i in range(channel)] for i in range(channel)]).cuda()
        kernel /= kernel.sum()
        return kernel

    k1 = 0.01
    k2 = 0.03
    if torch.max(img1) > 128:
        max = 255
    else:
        max = 1
    if torch.min(img1) < -0.5:
        min = -1
    else:
        min = 0
    l = max - min
    c1 = (k1 * l) ** 2
    c2 = (k2 * l) ** 2
    (channel, h, w) = img1.size()
    kernel = create_kernel(kernel_size, channel)
    # print(kernel)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
    # 计算均值
    mean1 = F.conv2d(img1, weight=kernel, stride=1, padding=0)
    mean2 = F.conv2d(img2, weight=kernel, stride=1, padding=0)
    # print(img1.size())
    # print(mean1.size())
    # 计算方差,利用公式dx=e(x^2)-e(x)^2
    variance1 = F.conv2d(img1 ** 2, weight=kernel, stride=1, padding=0) - mean1 ** 2
    variance2 = F.conv2d(img2 ** 2, weight=kernel, stride=1, padding=0) - mean2 ** 2
    # 计算协方差
    covariance = F.conv2d(img1 * img2, weight=kernel, stride=1, padding=0) - (mean1 * mean2)

    ssim = torch.mean(((2 * mean1 * mean2 + c1) * (2 * covariance + c2)) / (
            (mean1 ** 2 + mean2 ** 2 + c1) * (variance1 + variance2 + c2)))
    return ssim


def crop_img(img, img_size, n):
    # print(img.size())
    patch_size = img_size // n
    img_list = []
    for i in range(n):
        for j in range(n):
            img_list.append(img[..., i * patch_size:(i + 1) * patch_size,
                            j * patch_size:(j + 1) * patch_size])
    out = torch.cat(img_list, 0)
    # print(out.size())
    return out


def time_format(second):
    m, s = divmod(second, 60)
    m = round(m)
    s = round(s)
    if m < 60:
        return "{}m{}s".format(m, s)
    else:
        h, m = divmod(m, 60)
        h = round(h)
        m = round(m)
    if h < 24:
        return "{}h{}m{}s".format(h, m, s)
    else:
        d, h = divmod(h, 24)
        d = round(d)
        h = round(h)
    return "{}d{}h{}m{}s".format(d, h, m, s)
