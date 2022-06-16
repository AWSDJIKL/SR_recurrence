# -*- coding: utf-8 -*-
'''
加载模型，并测试不同测试集上的PSNR平均值
'''
# @Time    : 2022/3/13 20:42
# @Author  : LINYANZHEN
# @File    : test_model.py
import os
import shutil
import imageio
import numpy as np
import torch
import utils
import model
from option import args
import data.benchmark
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F


def model_test(model, test_loader):
    device = "cuda:0"
    model = model.to(device)
    psnr = 0
    with torch.no_grad():
        model.eval()
        for (lr, hr, img_name) in test_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            sr = utils.quantize(sr, args.rgb_range)
            psnr += utils.calculate_psnr(sr, hr, args.scale, args.rgb_range)
    psnr /= len(test_loader)
    return psnr


def test_bicubic(test_loader, scale=4):
    psnr = 0
    for (lr, hr, img_name) in test_loader:
        sr = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
        sr = utils.quantize(sr, args.rgb_range)
        psnr += utils.calculate_psnr(sr, hr, args.scale, args.rgb_range)
        # for img, name in zip([lr, hr, sr], ["LR", "HR", "bicubic_{}".format(img_name)]):
        #     normalized = img[0].data.mul(255 / args.rgb_range)
        #     # permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
        #     ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        #     # 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
        #     imageio.imsave(os.path.join("img_test/bicubic", "{}.png".format(name)), ndarr)
    psnr /= len(test_loader)
    return psnr


if __name__ == '__main__':
    test_set_list = [
        "Set5",
        "Set14",
        "BSD100",
        "Urban100"
    ]
    # 各个模型名字与路径名
    model_list = {
        "bicubic": "",
        "SRCNN_1_L1": "SRCNN",
        "FSRCNN_1_L1": "FSRCNN",
        "ESPCN_1_L1": "ESPCN",
        "RCAN_1_L1": "RCAN",
        "MSRN_1_L1": "MSRN",
        "MSARN_1_L1": "MSARN",
        "MSARN_1_L1+1e-3_VGG": "MSARN",
        "MSARN_PMG_1_L1": "MSARN",
        "MSARN_PMG_1_L1+1e-3_VGG": "MSARN",
    }
    # 新建一个dataframe保存每个模型的结果
    df = pd.DataFrame({"model": model_list.keys()})
    for test_set_name in test_set_list:
        args.test_set = test_set_name
        test_set = data.benchmark.Benchmark(args)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        psnr_list = []
        # 逐个导入模型并测试超分辨率结果，保存到文件夹中
        for model_path, model_name in model_list.items():
            print(model_path)
            if model_path == "bicubic":
                psnr = test_bicubic(test_loader)
            else:
                model_path = os.path.join("img_test/test_model", model_path, "model/best.pth")
                state_dict = torch.load(model_path)
                test_model = model.get_model(model_name, args)
                test_model.load_state_dict(state_dict)
                psnr = model_test(test_model, test_loader)
            print("test_set:{}  model:{}  psnr:{}".format(test_set_name, model_path, psnr))
            psnr_list.append(psnr)
        df.insert(len(df.columns), test_set_name, psnr_list)
    df.to_csv("img_test/psnr.csv", index=False)
