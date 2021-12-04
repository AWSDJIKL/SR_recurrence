# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/3 11:59
# @Author  : LINYANZHEN
# @File    : utils.py
import torch
from PIL import Image
import os
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision import transforms


def calculate_psnr(img1, img2):
    '''
    计算两张图片之间的PSNR误差

    :param img1:
    :param img2:
    :return:
    '''
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2)).item()


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


def calculate_ssim(img1, img2, kernel_size=11):
    '''

    :param img1:
    :param img2:
    :param kernel_size: 滑动窗口大小
    :return:
    '''
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


def tensor_to_image(tensor):
    # tr_mean, tr_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # mu = torch.Tensor(tr_mean).view(-1, 1, 1).cuda()
    # sigma = torch.Tensor(tr_std).view(-1, 1, 1).cuda()
    # img = transforms.ToPILImage()((tensor * sigma + mu).clamp(0, 1))
    img = transforms.ToPILImage()(tensor)
    return img


def test_model(model, test_image_path, upscale_factor, save_name):
    '''
    测试模型效果

    :param model: 要测试的模型
    :param test_image_path: 用于测试的图片的位置
    :param upscale_factor: 放大倍数
    :return:
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    origin_image = Image.open(test_image_path).convert('RGB')
    img_name, suffix = os.path.splitext(test_image_path)
    image_width = (origin_image.width // upscale_factor) * upscale_factor
    image_height = (origin_image.height // upscale_factor) * upscale_factor

    hr_image = origin_image.resize((image_width, image_height), resample=Image.BICUBIC)
    lr_image = origin_image.resize((image_width // upscale_factor, image_height // upscale_factor),
                                   resample=Image.BICUBIC)

    x = Variable(ToTensor()(lr_image)).to(device).unsqueeze(0)  # 补上batch_size那一维
    y = Variable(ToTensor()(hr_image)).to(device)

    with torch.no_grad():
        # out = model(x).clip(0, 1).squeeze()
        out = model(x).clip(0, 1)
        out = out.squeeze()
    psnr = calculate_psnr(y, out)
    out_y, _, _ = transforms.ToPILImage()(out).convert('YCbCr').split()
    hr_y, _, _ = hr_image.convert('YCbCr').split()
    # print(out_y.size)
    # print(hr_y.size)
    ssim = calculate_ssim(Variable(ToTensor()(hr_y)).to(device),
                          Variable(ToTensor()(out_y)).to(device))
    print('{} PSNR: {}'.format(save_name, psnr))
    print('{} SSIM: {}'.format(save_name, ssim))
    out = tensor_to_image(out)
    out.save(img_name + '_{}_x{}'.format(save_name, upscale_factor) + suffix)
    return
