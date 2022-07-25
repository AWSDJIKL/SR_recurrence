# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/23 10:42
# @Author  : LINYANZHEN
# @File    : test.py
import matplotlib.pyplot as plt
import numpy as np

log = np.load("checkpoint/x4_MSRN_OLD_PMG_OLD_PMG_1_L1/log_dict.npy",allow_pickle=True)
log_PMG = np.load("checkpoint/x4_MSRN_PMG_PMG_1_L1/log_dict.npy",allow_pickle=True)

psnr=log.item()["psnr_list"]
psnr_PMG=log_PMG.item()["psnr_list"]

plt.figure(figsize=(10, 5))
plt.plot(psnr, 'b', label='Progressive')
plt.plot(psnr_PMG, 'r', label='SRPMG')
plt.legend()
plt.grid()
plt.savefig('PSNR_curve.png', dpi=400)
plt.close()

# x = 121
# x = (x // 4) * 4
# print(x)
