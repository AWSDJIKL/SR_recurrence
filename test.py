# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/23 10:42
# @Author  : LINYANZHEN
# @File    : test.py
from torchsummary import summary
from model import RCAN
from option import args
import torch

# model = RCAN.make_model(args)
# summary(model.to("cuda:0"), input_size=(3, 192, 192), batch_size=1)
x = torch.Tensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]]])
print(x.size())
print(torch.nn.Softmax(dim=0)(x))
