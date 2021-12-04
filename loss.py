# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/3 12:17
# @Author  : LINYANZHEN
# @File    : loss.py
import torch
import torch.nn as nn
import torchvision.models


class vgg_loss(nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.PerceptualModel = nn.Sequential(*list(vgg.features)[:9])
        self.PerceptualModel.eval()
        self.PerceptualModel.cuda()

    def forward(self, pred, y):
        perceptual_pred = self.PerceptualModel(pred)
        perceptual_y = self.PerceptualModel(y)
        perceptual_loss = torch.nn.MSELoss()(perceptual_pred, perceptual_y)
        loss = perceptual_loss
        return loss
