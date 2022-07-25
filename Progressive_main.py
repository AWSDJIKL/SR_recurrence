
import random

import numpy as np
import torch
import data
from option import args
import loss
import model
import ProgressiveTrainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(args.seed)
    m = model.get_model(args.model_name, args)
    loader = data.ProgressiveData(args)
    loss = loss.Loss(args)
    trainer = ProgressiveTrainer.ProgressiveTrainer(args, m, loader, loss)
    while not trainer.is_finsh():
        trainer.train_and_test()
