# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/1/23 15:27
# @Author  : LINYANZHEN
# @File    : __init__.py

from importlib import import_module


def get_model(model_name, args):
    model = import_module("model." + model_name)
    return model.make_model(args)
