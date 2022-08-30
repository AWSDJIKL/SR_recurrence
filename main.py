# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2021/12/19 23:14
# @Author  : LINYANZHEN
# @File    : main.py
import random

import numpy as np
import torch
# from model import RCAN, GradualSR, GSACA
import Trainer
import data
from option import args
import loss
import model
# import PMGTrainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    setup_seed(args.seed)
    m = model.get_model(args.model_name, args)
    loader = data.Data(args)
    loss = loss.Loss(args)
    trainer = Trainer.Trainer(args, m, loader, loss)
    while not trainer.is_finsh():
        trainer.train_and_test()
