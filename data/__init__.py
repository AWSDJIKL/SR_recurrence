# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/19 10:51
# @Author  : LINYANZHEN
# @File    : __init__.py
from importlib import import_module
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.train_loader = None
        module_train = import_module('data.' + args.train_set.lower())
        trainset = getattr(module_train, args.train_set)(args)
        self.train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = None
        if args.test_set in ['Set5', 'Set14', 'B100', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False)
            self.test_loader = DataLoader(testset, batch_size=1, shuffle=False)
