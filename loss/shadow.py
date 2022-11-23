# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/11/8 22:50
# @Author  : LINYANZHEN
# @File    : shadow.py
import torch
import torch.nn as nn


class Shadow(nn.Module):
    def __init__(self, loss=nn.MSELoss()):
        super(Shadow, self).__init__()
        self.loss = loss

    def forward(self, outputs, y):
        loss = []
        for output in outputs:
            loss.append(self.loss(output, y).unsqueeze(0))
        loss = torch.cat(loss,0).mean()
        return loss
