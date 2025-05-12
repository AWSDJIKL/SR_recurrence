# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2024/12/9 23:13
# @Author  : LINYANZHEN
# @File    : lam_model.py

import cv2
import torch
import numpy as np
from option import args
from LAM.lam_utils import Tensor2PIL, PIL2Tensor
from LAM.SISR_BackProp import GaussianBlurPath
from LAM.SISR_BackProp import attribution_objective, Path_gradient
from LAM.SISR_BackProp import saliency_map_PG as saliency_map
from LAM.attributes import attr_grad
from LAM.lam_utils import cv2_to_pil, pil_to_cv2
from LAM.lam_utils import vis_saliency, vis_saliency_kde, grad_abs_norm, prepare_images, make_pil_grid
import model
import os.path

model_list = {
    "x4_SRCNN_384": "SRCNN",
    "x4_SRCNN_384_PMG_16_6_1_stride1.0": "SRCNN",
    "x4_ESPCN_384": "ESPCN",
    "x4_ESPCN_384_PMG_16_6_1_stride1.0": "ESPCN",
    "x4_RFDN_384": "RFDN",
    "x4_RFDN_PMG_384_6_4_2_1_stride1.0": "RFDN",
    "x4_IMDN_plus_384": "IMDN_plus",
    "x4_IMDN_plus_384_PMG_16_12_6_3_1_stride1.0": "IMDN_plus",
    "x4_ESRT_384": "ESRT",
    "x4_ESRT_384_PMG_8_1_stride1.0": "ESRT",
    "x4_SwinIR_384": "SwinIR",
    "x4_SwinIR_384_PMG_6_4_2_1_stride1.0": "SwinIR",
    "x4_Swin2SR_384": "Swin2SR",
    "x4_Swin2SR_384_PMG_6_4_2_1_stride1.0": "Swin2SR",
    "x4_HAT_384": "HAT",
    "x4_HAT_384_PMG_6_3_2_1_stride1.0": "HAT",
}


def lam(model, save_name, test_img, w, h, window_size=16):
    # window_size = 16  # Define windoes_size of D
    # img_lr, img_hr = prepare_images("dataset/lam/{}.png".format(test_img))  # Change this image name
    # l = 9
    img_lr, img_hr = prepare_images("dataset/Urban100/x4/hr/28.png".format(test_img))  # Change this image name
    w = 560
    h = 200
    l = 10
    window_size = 80
    tensor_lr = PIL2Tensor(img_lr)[:3]
    tensor_hr = PIL2Tensor(img_hr)[:3]
    cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2)
    cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)

    # w = 125  # The x coordinate of your select patch, 125 as an example
    # h = 160  # The y coordinate of your select patch, 160 as an example

    # And check the red box
    # Is your selected patch this one? If not, adjust the `w` and `h`.

    draw_img = pil_to_cv2(img_hr)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)

    sigma = 1.2
    fold = 50
    alpha = 0.5

    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), model, attr_objective,
                                                                              gaus_blur_path_func)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
         saliency_image_abs,
         blend_abs_and_input,
         blend_kde_and_input,
         # Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))
         ]
    )

    if not os.path.exists("img_test"):
        os.mkdir("img_test")
    if not os.path.exists("img_test/lam"):
        os.mkdir("img_test/lam")
    if not os.path.exists("img_test/lam/{}".format(test_img)):
        os.mkdir("img_test/lam/{}".format(test_img))
    pil.save("img_test/lam/{}/{}.png".format(test_img, save_name))
    print("lam result saved")


if __name__ == '__main__':

    test_img = "7"
    # 加载模型，这一部分重写，根据需求加载自己的模型
    for model_path, model_name in model_list.items():
        with open('checkpoint/{}/option.txt'.format(model_path), 'r') as f:
            lines = f.readlines()
        args_dict = {}
        for line in lines:
            key, value = line.split(' ', 1)
            args_dict[key] = value
        args.part = eval(args_dict['part'])
        save_name = model_path
        model_path = "checkpoint/{}/model/final.pth".format(model_path)
        state_dict = torch.load(model_path, map_location="cuda:0")
        test_model = model.get_model(model_name, args)
        test_model.load_state_dict(state_dict)
        print("load model:{}".format(model_path))
        lam(test_model, save_name, test_img, 125, 160)
