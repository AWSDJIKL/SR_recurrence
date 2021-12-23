# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/23 10:42
# @Author  : LINYANZHEN
# @File    : test.py
from torchsummary import summary
from model import RCAN
from option import args

model = RCAN.make_model(args)
summary(model.to("cuda:0"), input_size=(3, 512, 512), batch_size=1)
