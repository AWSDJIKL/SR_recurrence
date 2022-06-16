# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/23 10:42
# @Author  : LINYANZHEN
# @File    : test.py
import matplotlib.pyplot as plt
import numpy as np

# log = np.load("img_test/test_model/MSARN_1_L1/log_dict.npy",allow_pickle=True)
# log_PMG = np.load("img_test/test_model/MSARN_PMG_1_L1/log_dict.npy",allow_pickle=True)
#
# psnr=log.item()["psnr_list"]
# psnr_PMG=log_PMG.item()["psnr_list"]
#
# plt.figure(figsize=(10, 5))
# plt.plot(psnr, 'b', label='Normal')
# plt.plot(psnr_PMG, 'r', label='Use progressive training')
# plt.legend()
# plt.grid()
# plt.savefig('PSNR.png', dpi=400)
# plt.close()

x = 121
x = (x // 4) * 4
print(x)
