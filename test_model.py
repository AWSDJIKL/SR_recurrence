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
import ssim

SSIM = ssim.SSIM(data_range=255)


def model_test(model, test_loader):
    device = "cuda:0"
    model = model.to(device)
    psnr = 0
    ssim = 0
    with torch.no_grad():
        model.eval()
        for (lr, hr, img_name) in test_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            sr = utils.quantize(sr, args.rgb_range)
            psnr += utils.calculate_psnr(sr, hr, args.scale, args.rgb_range)
            # ssim += utils.calculate_ssim(sr, hr)
            ssim += SSIM(sr, hr).item()
    psnr /= len(test_loader)
    ssim /= len(test_loader)
    # lpips /= len(test_loader)
    return psnr, ssim


def test_bicubic(test_loader, scale=4):
    psnr = 0
    ssim = 0
    for (lr, hr, img_name) in test_loader:
        sr = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
        sr = utils.quantize(sr, args.rgb_range)
        psnr += utils.calculate_psnr(sr, hr, args.scale, args.rgb_range)
        # ssim += utils.calculate_ssim(sr, hr)
        ssim += SSIM(sr, hr).item()
        # psnr += PSNR(sr, hr)
        # ssim += SSIM(sr, hr)
        # for img, name in zip([lr, hr, sr], ["LR", "HR", "bicubic_{}".format(img_name)]):
        #     normalized = img[0].data.mul(255 / args.rgb_range)
        #     # permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
        #     ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        #     # 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
        #     imageio.imsave(os.path.join("img_test/bicubic", "{}.png".format(name)), ndarr)
    psnr /= len(test_loader)
    ssim /= len(test_loader)
    return psnr, ssim


if __name__ == '__main__':
    test_set_list = [
        "Set5",
        "Set14",
        "BSD100",
        "Urban100"
    ]
    # 各个模型名字与路径名
    model_list = {
        # "bicubic": "",
        # "SRCNN_1_L1": "SRCNN",
        # "FSRCNN_1_L1": "FSRCNN",
        # "ESPCN_1_L1": "ESPCN",
        # "RCAN_1_L1": "RCAN",
        "MSRN_1_L1": "MSRN",
        # "MSRN_PMG_1_L1": "MSRN_PMG",
        # "MSRN_OLD_PMG_OLD_PMG_1_L1": "MSRN_OLD_PMG",
        # "MSRN_PMG_PMG_no_crop_1_L1": "MSRN_PMG",
        # "MSRN_PMG_PMG_1_L1": "MSRN_PMG",
    }
    # 新建一个dataframe保存每个模型的结果

    for scale_factor in [4,8]:
        args.scale = scale_factor
        print(args.scale)
        df = pd.DataFrame({"model": model_list.keys()})
        for test_set_name in test_set_list:
            args.test_set = test_set_name
            test_set = data.benchmark.Benchmark(args)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
            psnr_list = []
            ssim_list = []
            # 逐个导入模型并测试超分辨率结果，保存到文件夹中
            for model_path, model_name in model_list.items():
                print(model_path)
                if model_path == "bicubic":
                    psnr, ssim = test_bicubic(test_loader, scale=scale_factor)
                else:
                    model_path = "checkpoint/x{}_{}/model/final.pth".format(scale_factor, model_path)
                    state_dict = torch.load(model_path)
                    test_model = model.get_model(model_name, args)
                    test_model.load_state_dict(state_dict)
                    psnr, ssim = model_test(test_model, test_loader)
                print("test_set:{}  model:{}  psnr:{} ssim:{}".format(test_set_name, model_path, psnr, ssim,
                                                                      ))
                psnr_list.append(psnr)
                ssim_list.append(ssim)
            df.insert(len(df.columns), test_set_name + "_PSNR", psnr_list)
            df.insert(len(df.columns), test_set_name + "_SSIM", ssim_list)
        df.to_csv("checkpoint/x{}_test_result.csv".format(scale_factor), index=False)
