# -*- coding: utf-8 -*-
'''
Charbonnier损失函数，近似L1函数
'''
# @Time    : 2022/2/15 11:49
# @Author  : LINYANZHEN
# @File    : Charbonnier.py
import torch
import torch.nn as nn


class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
