# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/3 11:59
# @Author  : LINYANZHEN
# @File    : utils.py
import random
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
        self.model = model
        self.checkpoint_dir = os.path.join("checkpoint", experiment_name)
        self.model_checkpoint_dir = os.path.join(self.checkpoint_dir, "model")
        self.log_file_path = os.path.join(self.checkpoint_dir, "log.txt")
        self.loss_log = os.path.join(self.checkpoint_dir, "loss.npy")
        self.psnr_log = os.path.join(self.checkpoint_dir, "psnr.npy")
        if args.load_checkpoint:
            print("正在加载保存点")
            # 加载保存点
            self.model_checkpoint_dir = os.path.join(self.checkpoint_dir, "model")
            log_dict = np.load(os.path.join(self.checkpoint_dir, "log_dict.npy"), allow_pickle=True)
            self.best_psnr = log_dict.item().get("best_psnr")
            self.loss_list = log_dict.item().get("loss_list")
            self.psnr_list = log_dict.item().get("psnr_list")
        else:
            print("正在初始化")
            self.loss_list = []
            self.psnr_list = []
            self.best_psnr = 0
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if os.path.exists(self.checkpoint_dir):
                shutil.rmtree(self.checkpoint_dir)
            os.mkdir(self.checkpoint_dir)
            os.mkdir(self.model_checkpoint_dir)
            # 先写开头部分
            with open(self.log_file_path, "w") as f:
                f.write(now + "\n")
        # 记录配置文件
        with open(os.path.join(self.checkpoint_dir, "option.txt"), "w") as f:
            for parameter in dir(args):
                if parameter[0] == "_":
                    continue
                f.write("{} {}\n".format(parameter, getattr(args, parameter)))

    def record_epoch(self, epoch, epoch_loss, epoch_psnr, optimizer, scheduler):
        if epoch_psnr > self.best_psnr:
            self.best_psnr = epoch_psnr
            torch.save(self.model.state_dict(), os.path.join(self.model_checkpoint_dir, "best.pth".format(epoch)))
        torch.save(self.model.state_dict(), os.path.join(self.model_checkpoint_dir, "final.pth".format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pth"))
        torch.save(scheduler.state_dict(), os.path.join(self.checkpoint_dir, "scheduler.pth"))
        self.write_log("epoch :{}".format(epoch))
        self.write_log("loss :{}".format(epoch_loss))
        self.write_log("psnr :{}    best psnr:{}".format(epoch_psnr, self.best_psnr))
        self.loss_list.append(epoch_loss)
        self.psnr_list.append(epoch_psnr)
        self.plot_loss()
        self.plot_psnr()
        log_dict = {
            "best_psnr": self.best_psnr,
            "loss_list": self.loss_list,
            "psnr_list": self.psnr_list,
        }
        np.save(os.path.join(self.checkpoint_dir, "log_dict.npy"), log_dict)

    def save_final(self):
        self.plot_loss()
        self.plot_psnr()
        # torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "final.pth"))

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
            for i in range(channel)] for i in range(channel)]).to("cuda:0")
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
    (batch_size, channel, h, w) = img1.size()
    kernel = create_kernel(kernel_size, channel)
    # print(kernel)
    # img1 = img1.unsqueeze(0)
    # img2 = img2.unsqueeze(0)
    img1 = img1.to("cuda:0")
    img2 = img2.to("cuda:0")
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
            (mean1 ** 2 + mean2 ** 2 + c1) * (variance1 + variance2 + c2))).item()
    return ssim


def save_img(tensor, save_dir, name):
    normalized = tensor[0].data.mul(255 / 255)
    # permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
    ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
    # 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
    imageio.imsave(os.path.join(save_dir, "{}.png".format(name)), ndarr)


def crop_img(img, img_size, n, stride=None, padding=0):
    # print(img.size())
    k = img_size // n
    if not stride:
        stride = k
    unfold = F.unfold(img, kernel_size=(k, k), stride=stride, padding=padding).permute(0, 2, 1)
    b, c, l = unfold.size()
    b = (b * c * l) // (3 * k * k)
    out = unfold.reshape(b, 3, k, k)
    # patch_size = img_size // n
    # img_list = []
    # for i in range(n):
    #     for j in range(n):
    #         img_list.append(img[..., i * patch_size:(i + 1) * patch_size,
    #                         j * patch_size:(j + 1) * patch_size])
    #
    # # random.shuffle(img_list)
    # out = torch.cat(img_list, 0)
    # # print(out.size())
    return out


def jigsaw_generator(lr, hr, lr_size, hr_size, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    lr_block_size = lr_size // n
    hr_block_size = hr_size // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws_lr = lr.clone()
    jigsaws_hr = hr.clone()
    for i in range(rounds):
        x, y = l[i]
        # lr
        temp = jigsaws_lr[..., 0:lr_block_size, 0:lr_block_size].clone()
        jigsaws_lr[..., 0:lr_block_size, 0:lr_block_size] = jigsaws_lr[..., x * lr_block_size:(x + 1) * lr_block_size,
                                                            y * lr_block_size:(y + 1) * lr_block_size].clone()
        jigsaws_lr[..., x * lr_block_size:(x + 1) * lr_block_size, y * lr_block_size:(y + 1) * lr_block_size] = temp
        # hr
        temp = jigsaws_hr[..., 0:hr_block_size, 0:hr_block_size].clone()
        jigsaws_hr[..., 0:hr_block_size, 0:hr_block_size] = jigsaws_hr[..., x * hr_block_size:(x + 1) * hr_block_size,
                                                            y * hr_block_size:(y + 1) * hr_block_size].clone()
        jigsaws_hr[..., x * hr_block_size:(x + 1) * hr_block_size, y * hr_block_size:(y + 1) * hr_block_size] = temp

    return jigsaws_lr, jigsaws_hr


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
