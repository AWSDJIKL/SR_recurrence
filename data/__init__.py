
from importlib import import_module
from torch.utils.data import DataLoader
import copy


class Data:
    def __init__(self, args):
        self.train_loader = None
        module_train = import_module('data.' + args.train_set.lower())
        trainset = getattr(module_train, args.train_set)(args)
        self.train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = None
        if args.test_set in ['Set5', 'Set14', 'BSD100', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False)
            self.test_loader = DataLoader(testset, batch_size=1, shuffle=False)


class ProgressiveData:
    def __init__(self, args):
        x2_args = copy.deepcopy(args)
        x4_args = copy.deepcopy(args)
        x8_args = copy.deepcopy(args)
        x2_args.scale = 2
        x8_args.scale = 8
        print(x2_args.scale)
        print(x4_args.scale)

        self.train_loader = []
        module_train = import_module('data.' + args.train_set.lower())
        x2_trainset = getattr(module_train, args.train_set)(x2_args)
        self.train_loader.append(DataLoader(x2_trainset, batch_size=args.batch_size, shuffle=True))
        x4_trainset = getattr(module_train, args.train_set)(x4_args)
        self.train_loader.append(DataLoader(x4_trainset, batch_size=args.batch_size, shuffle=True))
        x8_trainset = getattr(module_train, args.train_set)(x8_args)
        self.train_loader.append(DataLoader(x8_trainset, batch_size=args.batch_size, shuffle=True))
        self.test_loader = None
        if args.test_set in ['Set5', 'Set14', 'BSD100', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False)
            self.test_loader = DataLoader(testset, batch_size=1, shuffle=False)
