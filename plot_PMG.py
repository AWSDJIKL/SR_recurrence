# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/4/21 22:32
# @Author  : LINYANZHEN
# @File    : plot_PMG.py
import torchvision

import utils
import PIL.Image as Image
from torchvision import transforms
import os


def save_img(img_list, save_path):
    i = 0
    for img in img_list:
        img=torchvision.transforms.functional.to_pil_image(img)
        img.save(os.path.join(save_path, "{}.png".format(i)))
        i += 1


if __name__ == '__main__':
    img = Image.open("dataset/Set5/Set5/image_SRF_4/img_001_SRF_4_HR.png").convert('RGB')
    tensor = transforms.ToTensor()(img)
    origin = torchvision.transforms.functional.to_pil_image(tensor)
    origin.save("img_test/PMG/origin.png")
    x16 = utils.crop_img(tensor, 512, 16)
    x8 = utils.crop_img(tensor, 512, 8)
    x4 = utils.crop_img(tensor, 512, 4)
    save_img(x16, "img_test/PMG/x16")
    save_img(x8, "img_test/PMG/x8")
    save_img(x4, "img_test/PMG/x4")
