# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/9/22 20:58
# @Author  : LINYANZHEN
# @File    : spl.py
import torch
import torch.nn as nn


class SPL(nn.Module):
    def __init__(self, loss=nn.MSELoss()):
        super(SPL, self).__init__()
        self.loss = loss

    def forward(self, out, y, learn_percent=0.5):
        loss = []
        for i in range(out.size()[0]):
            loss.append(self.loss(out[i], y[i]))
        index = torch.topk(torch.tensor(loss), int(len(loss) * learn_percent)).indices
        result = loss[index[0]]
        for i in index[1:]:
            result += loss[i]
        return result
