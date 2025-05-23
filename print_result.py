# -*- coding: utf-8 -*-
'''
读取模型，输出超分辨率结果

'''
# @Time    : 2022/3/6 15:57
# @Author  : LINYANZHEN
# @File    : print_result.py
import os
import shutil

import imageio
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import utils
import model
from option import args
import data.benchmark
import imageio
from PIL import Image, ImageDraw, ImageFont
import ssim

SSIM = ssim.SSIM(data_range=255)


def blow_up_details(input_image, pos, size, upscale_factor):
    '''
    将图像指定位置的细节放大指定倍数，绘制与图像的左下角并输出到指定路径
    :param input_image: 输入图像
    :param pos: 放大位置的左上角坐标（图像左上角为（0，0）点）
    :param size: 放大区域大小
    :param upscale_factor: 放大倍数
    :return:
    '''
    print("图像大小：{}".format(input_image.size))
    im_box = input_image.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))
    im_box = im_box.resize((size[0] * upscale_factor, size[1] * upscale_factor))
    h0 = input_image.size[1] - size[1] * upscale_factor
    input_image.paste(im_box, (0, h0))
    im_draw = ImageDraw.Draw(input_image)
    im_draw.line((pos[0], pos[1], pos[0] + size[0], pos[1]), width=1, fill=(255, 0, 0))
    im_draw.line((pos[0], pos[1] + size[1], pos[0] + size[0], pos[1] + size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((pos[0], pos[1], pos[0], pos[1] + size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((pos[0] + size[0], pos[1], pos[0] + size[0], pos[1] + size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((0, h0, size[0] * upscale_factor, h0), width=1, fill=(255, 0, 0))
    im_draw.line((0, input_image.size[1], size[0] * upscale_factor, input_image.size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((0, h0, 0, input_image.size[1]), width=1, fill=(255, 0, 0))
    im_draw.line((size[0] * upscale_factor, h0, size[0] * upscale_factor, input_image.size[1]), width=1,
                 fill=(255, 0, 0))
    return input_image


def crop_img(input_img, pos, size, upscale_factor):
    im_box = input_img.crop((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]))
    # im_box = im_box.resize((size[0] * upscale_factor, size[1] * upscale_factor))
    return im_box


def image_concat(image_list, image_text_list, output_path):
    '''
    :param image_list:
    :param image_text_list:
    :param output_path:
    :return:
    '''
    target_size = [0, 0]
    x_interval = 20
    y_interval = 50
    target_size[0] += (image_list[0].size[0] + x_interval) * len(image_list) - x_interval
    target_size[1] += image_list[0].size[1] + y_interval
    # 构建画布
    output_image = Image.new('RGB', (target_size[0], target_size[1]), color=(255, 255, 255))
    draw = ImageDraw.Draw(output_image)
    # 设置字体，字号大小
    font = ImageFont.truetype("fonts/arial.ttf", 20)
    x, y = 0, 0
    for i in range(len(image_list)):
        output_image.paste(image_list[i], (x, y))
        # draw.text((x, image_list[0].size[1]), image_text_list[i], (0, 0, 0), font=font)
        x += image_list[0].size[0] + x_interval
    output_image.save(output_path)


def model_test(model, test_set, index, save_path, save_name, upscale_factor=4):
    device = "cuda:0"
    lr, hr, img_name = test_set[index]
    lr = lr.to(device).unsqueeze(0)
    hr = hr.unsqueeze(0)
    model = model.to(device)
    sr = model(lr).to("cpu")
    sr = utils.quantize(sr, args.rgb_range)
    # print(sr.size())
    # print(hr.size())
    psnr = utils.calculate_psnr(sr, hr, args.scale, args.rgb_range)
    ssim = SSIM(sr, hr)
    with open(os.path.join(save_path, "log.csv"), "a") as file:
        file.write("{},{:.2f},{:.4f}\n".format(save_name, psnr, ssim))
    for img, name in zip([lr, hr, sr], ["LR", "HR", save_name]):
        normalized = img[0].data.mul(255 / args.rgb_range)
        # permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        # 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
        imageio.imsave(os.path.join(save_path, "{}.png".format(name)), ndarr)


def test_bicubic(test_set, index, save_path):
    lr, hr, img_name = test_set[index]
    lr = lr.unsqueeze(0)
    hr = hr.unsqueeze(0)
    sr = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
    sr = utils.quantize(sr, args.rgb_range)
    psnr = utils.calculate_psnr(sr, hr, args.scale, args.rgb_range)
    ssim = SSIM(sr, hr)
    with open(os.path.join(save_path, "log.csv"), "a") as file:
        file.write("bicubic,{:.2f},{:.4f}\n".format(psnr, ssim))
    for img, name in zip([lr, hr, sr], ["LR", "HR", "bicubic"]):
        normalized = img[0].data.mul(255 / args.rgb_range)
        # permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        # 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
        imageio.imsave(os.path.join(save_path, "{}.png".format(name)), ndarr)
    return psnr


def test_multi_scale_bicubic(test_set, index, save_path):
    scale_list = [2, 4, 8, 16]
    for scale in scale_list:
        lr, hr, img_name = test_set[index]
        hr = hr.unsqueeze(0)
        width = (hr.size()[-2] // scale) * scale
        height = (hr.size()[-1] // scale) * scale
        hr = transforms.CenterCrop((width, height))(hr)
        lr = F.interpolate(hr, scale_factor=1 / scale, mode="bicubic", align_corners=False)
        sr = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
        sr = utils.quantize(sr, args.rgb_range)
        psnr = utils.calculate_psnr(sr, hr, args.scale, args.rgb_range)
        with open(os.path.join(save_path, "log.csv"), "a") as file:
            file.write("bicubic_{},{}\n".format(scale, psnr))
        for img, name in zip([lr, hr, sr], ["LR", "HR", "bicubic_{}".format(scale)]):
            normalized = img[0].data.mul(255 / args.rgb_range)
            # permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            # 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
            imageio.imsave(os.path.join(save_path, "{}.png".format(name)), ndarr)


if __name__ == '__main__':
    # # 各个模型名字与路径名
    # model_list = {
    #     "bicubic": "",
    #     "x4_SRCNN_384": "SRCNN",
    #     "x4_SRCNN_384_PMG_16_6_1_stride1.0": "SRCNN",
    #     "x4_ESPCN_384": "ESPCN",
    #     "x4_ESPCN_384_PMG_16_6_1_stride1.0": "ESPCN",
    #     "x4_RFDN_384": "RFDN",
    #     "x4_RFDN_PMG_384_6_4_2_1_stride1.0": "RFDN",
    #     "x4_IMDN_plus_384": "IMDN_plus",
    #     "x4_IMDN_plus_384_PMG_16_12_6_3_1_stride1.0": "IMDN_plus",
    #     "x4_ESRT_384": "ESRT",
    #     "x4_ESRT_384_PMG_8_1_stride1.0": "ESRT",
    #     "x4_SwinIR_384": "SwinIR",
    #     "x4_SwinIR_384_PMG_6_4_2_1_stride1.0": "SwinIR",
    #     "x4_Swin2SR_384": "Swin2SR",
    #     "x4_Swin2SR_384_PMG_6_4_2_1_stride1.0": "Swin2SR",
    #     "x4_HAT_384": "HAT",
    #     "x4_HAT_384_PMG_6_3_2_1_stride1.0": "HAT",
    # }
    # # args.test_set = "Set5"
    # # args.test_set = "Set14"
    # # args.test_set = "BSD100"
    # args.test_set = "Urban100"
    # # args.test_set = "img_test"
    # args.scale = 4
    # # index = [1, 3, 28,30]
    # index = [30]
    # # index = [0]
    # test_set = data.benchmark.Benchmark(args)
    # # for i in range(len(test_set)):
    # for i in index:
    #     img_path = test_set[i][-1]
    #     img_name, suffix = os.path.splitext(img_path)
    #     img_name = os.path.split(img_name)[-1]
    #     save_path = os.path.join("img_test", img_name)
    #     # 在img_test文件夹下建立一个与图片同名的文件夹保存结果
    #     if os.path.exists(save_path):
    #         shutil.rmtree(save_path)
    #     os.mkdir(save_path)
    #     # 新建log
    #     with open(os.path.join(save_path, "log.csv"), "w") as file:
    #         file.write("model,PSNR,SSIM\n")
    #     # 逐个导入模型并测试超分辨率结果，保存到文件夹中
    #     for model_path, model_name in model_list.items():
    #         print(model_path)
    #         if model_path == "bicubic":
    #             test_bicubic(test_set, i, save_path)
    #         else:
    #             # state_dict_path = os.path.join("img_test/test_model", model_path, "model/final.pth")
    #             state_dict = torch.load("checkpoint/{}/model/final.pth".format(model_path), map_location="cuda:0")
    #             with open('checkpoint/{}/option.txt'.format(model_path), 'r') as f:
    #                 lines = f.readlines()
    #             args_dict = {}
    #             for line in lines:
    #                 key, value = line.split(' ', 1)
    #                 args_dict[key] = value
    #             args.part = eval(args_dict['part'])
    #             test_model = model.get_model(model_name, args)
    #             test_model.load_state_dict(state_dict)
    #             model_test(test_model, test_set, i, save_path, model_path)

    # 对文件夹内每张图片都放大细节
    img_list = {
        "img_test/1": [860, 520, 80, 80],
        "img_test/3": [640, 300, 80, 80],
        "img_test/28": [560, 200, 80, 80],
        "img_test/30": [700, 400, 80, 80],
    }
    save_path = "img_test/blow_up_details"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    for img_folder, pos_size in img_list.items():
        # img_folder = "img_test/img_001_SRF_4_HR"
        # 在img_test中建立一个文件夹
        sub_save_path = os.path.join(save_path, img_folder[9:])
        if os.path.exists(sub_save_path):
            shutil.rmtree(sub_save_path)
        os.mkdir(sub_save_path)
        for root, dirs, files in os.walk(img_folder):
            for file in files:
                if file[-3:] == "png" and file[:2] != "LR":
                    print(file)
                    img = Image.open(os.path.join(root, file))
                    crop = crop_img(img, pos_size[:2], pos_size[2:], 2)
                    crop.save(os.path.join(sub_save_path, "{}_crop.png".format(file[:-4])))
                    blow = blow_up_details(img, pos_size[:2], pos_size[2:], 2)
                    blow.save(os.path.join(sub_save_path, "{}_blow_up.png".format(file[:-4])))
