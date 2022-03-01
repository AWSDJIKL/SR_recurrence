# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/20 11:12
# @Author  : LINYANZHEN
# @File    : __init__.py
import torch
import torch.nn as nn
from importlib import import_module


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.loss_name = args.loss_name
        self.is_PMG = args.is_PMG
        print('Preparing loss function:')
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss_name.split('+'):
            # 该损失所占的权重*损失类型
            weight, loss_type = loss.split('_')
            loss_function = None
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == "Charbonnier":
                module = import_module("loss.Charbonnier")
                loss_function = getattr(module, "L1_Charbonnier_loss")()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])
        # self.loss_weight = nn.ParameterList([nn.Parameter(torch.Tensor([0.5])) for i in range(4)])

    def forward(self, sr, hr, step=0):
        # if self.is_PMG:
        #     # print(sr.size())
        #     # print(hr.size())
        #     loss = self.loss[step]["function"](sr, hr)
        #     loss = self.loss[step]["weight"] * loss
        #     return loss
        # else:
        #     losses = []
        #     for i, l in enumerate(self.loss):
        #         if l['function'] is not None:
        #             loss = l['function'](sr, hr)
        #             effective_loss = l['weight'] * loss
        #             losses.append(effective_loss)
        #     loss_sum = sum(losses)
        #     return loss_sum
        losses = []

        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
        loss_sum = sum(losses)
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        return self.loss_module
