# -*- coding: utf-8 -*-
'''

'''
import PIL.Image
# @Time    : 2021/12/23 10:42
# @Author  : LINYANZHEN
# @File    : test.py
import matplotlib.pyplot as plt
import numpy as np
import torchsummary
import model.MSRN
from option import args
import torch
import torch.nn.functional as F
import imageio
from PIL import Image
import torchvision.transforms

model = model.MSRN.MSRN(args)
model.parameters()
model.load_state_dict(torch.load("checkpoint/x16_MSRN_1_L1/model/final.pth"))

img=Image.open("img_test/HR.png").convert("RGB")
lr = img.resize((32, 32),Image.BICUBIC)
lr.save("img_test/LR.png")
lr = imageio.imread("img_test/LR.png")
np_transpose = np.ascontiguousarray(lr.transpose((2, 0, 1)))
# 转tensor
tensor = torch.from_numpy(np_transpose).float()
tensor.mul_(255 / 255)
lr_tensor = tensor.unsqueeze(0)
sr = model(lr_tensor)
normalized = sr[0].data.mul(255 / 255)
# permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
# 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
imageio.imsave("img_test/SR.png", ndarr)
