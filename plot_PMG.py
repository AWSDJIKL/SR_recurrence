# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/4/21 22:32
# @Author  : LINYANZHEN
# @File    : plot_PMG.py
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import utils
import PIL.Image as Image
from torchvision import transforms
import os

#
# def save_img(img_list, save_path):
#     for i in range(0, len(img_list), 3):
#         r = img_list[i + 0].unsqueeze(0)
#         g = img_list[i + 1].unsqueeze(0)
#         b = img_list[i + 2].unsqueeze(0)
#         img = torch.cat([r, g, b])
#         img = torchvision.transforms.functional.to_pil_image(img)
#         img.save(os.path.join(save_path, "{}.png".format(i)))
#
#
# if __name__ == '__main__':
#     img = Image.open("dataset/Set5/Set5/image_SRF_4/img_001_SRF_4_HR.png").convert('RGB')
#     tensor = transforms.ToTensor()(img)
#     origin = torchvision.transforms.functional.to_pil_image(tensor)
#     origin.save("img_test/PMG/origin.png")
#     x8 = utils.crop_img(tensor, 512, 8)
#     x4 = utils.crop_img(tensor, 512, 4)
#     x2 = utils.crop_img(tensor, 512, 2)
#     save_img(x8, "img_test/PMG/x8")
#     save_img(x4, "img_test/PMG/x4")
#     save_img(x2, "img_test/PMG/x2")

normal = np.load("checkpoint/x4_MSRN_1_L1/log_dict.npy", allow_pickle=True).item().get("psnr_list")
pmg = np.load("checkpoint/x4_MSRN_PMG_PMG_1_L1/log_dict.npy", allow_pickle=True).item().get("psnr_list")
plt.figure(dpi=400, figsize=(16, 9))
plt.title("each epoch PSNR compare in training", fontsize=25)
plt.plot(normal, color="black", label="normal")
plt.plot(pmg, color="blue", label="SRPMG")
plt.legend(prop={"size": 25})
plt.axhline(y=pmg[230], color='red', linestyle='dashed')
plt.axvline(x=230, color='red', linestyle='dashed')
plt.xlabel("epoch",fontsize=25)
plt.ylabel("PSNR",fontsize=25)
plt.savefig("PSNR_curve.png")
