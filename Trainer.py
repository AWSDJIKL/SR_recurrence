# -*- coding: utf-8 -*-
'''
训练器
'''
# @Time    : 2021/12/18 15:50
# @Author  : LINYANZHEN
# @File    : Trainer.py
import shutil

import imageio
import torch
from torch.autograd import Variable
# from tqdm import tqdm
import tqdm
import data
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import utils


class Trainer():
    def __init__(self, args, model, loader: data.Data, loss):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.model = model
        self.is_PMG = args.is_PMG
        self.loss = loss
        # 是否使用半精度
        self.half_precision = args.half_precision
        self.device = args.device
        if self.half_precision:
            self.model.half()
            self.loss.half()
        self.model.to(self.device)
        self.loss.to(self.device)

        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        if self.is_PMG and model.support_PMG:
            experiment_name = model.__class__.__name__ + "_PMG_" + loss.loss_name
        else:
            args.is_PMG = False
            self.is_PMG = False
            experiment_name = model.__class__.__name__ + "_" + loss.loss_name
        self.checkpoint = utils.Checkpoint(args, self.model, experiment_name)
        if args.load_checkpoint:
            # 加载上一次的模型，优化器和学习率调整器
            self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint.checkpoint_dir, "optimizer.pth")))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint.checkpoint_dir, "scheduler.pth")))
            last_epoch = self.scheduler.last_epoch
            checkpoint = torch.load(os.path.join(self.checkpoint.checkpoint_dir, "model/final.pth"))
            self.model.load_state_dict(checkpoint)

    def train_and_test(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        self.checkpoint.write_log("[Epoch {}]\tLearning rate: {}".format(epoch, lr))
        # 将模型设置为训练模式
        self.model.train()
        epoch_loss = 0
        epoch_psnr = 0
        progress = tqdm.tqdm(self.train_loader, total=len(self.train_loader))
        for (lr, hr, img_name) in progress:
            lr = self.prepare(lr)
            hr = self.prepare(hr)
            if self.is_PMG:
                lr_size = self.args.patch_size // self.args.scale
                hr_size = self.args.patch_size
                lr_list = []
                hr_list = []
                for n in [16, 8, 4]:
                    print(1)
                    # for n in [64, 16, 4]:
                    lr_list.append(utils.crop_img(lr, lr_size, n))
                    hr_list.append((utils.crop_img(hr, hr_size, n)))
                    # jigsaws_lr, jigsaws_hr = utils.jigsaw_generator(lr, hr, lr_size, hr_size, n)
                    # lr_list.append(jigsaws_lr)
                    # hr_list.append(jigsaws_hr)
                lr_list.append(lr)
                hr_list.append(hr)
                for i, lr_rate in zip(range(4), [1, 1, 1, 2]):
                    # print(lr_list[i].device)
                    # print(next(self.model.parameters()).device)
                    self.optimizer.zero_grad()
                    sr = self.model(lr_list[i], i)
                    # loss = self.loss(sr, hr_list[i], i)
                    loss = self.loss(sr, hr_list[i]) * lr_rate
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item() / 4
            else:
                self.optimizer.zero_grad()
                sr = self.model(lr)
                # self.save_sr_result([lr, hr, sr], img_name[0], epoch, self.checkpoint.checkpoint_dir)
                # return
                loss = self.loss(sr, hr)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
        epoch_loss /= len(self.train_loader)
        with torch.no_grad():
            self.model.eval()
            for (lr, hr, img_name) in self.test_loader:
                lr = self.prepare(lr)
                hr = self.prepare(hr)
                if self.is_PMG:
                    sr = self.model(lr, 3)
                else:
                    sr = self.model(lr)
                # print(sr)
                sr = utils.quantize(sr, self.args.rgb_range)
                self.save_sr_result([lr, hr, sr], img_name[0], epoch, self.checkpoint.checkpoint_dir)
                psnr = utils.calculate_psnr(sr, hr, self.args.scale, self.args.rgb_range)
                epoch_psnr += psnr
        epoch_psnr /= len(self.test_loader)
        self.scheduler.step()
        self.checkpoint.record_epoch(epoch, epoch_loss, epoch_psnr, self.optimizer, self.scheduler)

    def prepare(self, tensor):
        if self.half_precision:
            tensor = tensor.half()
        tensor = tensor.to(self.device)
        return tensor

    def save_sr_result(self, img_list, img_name, epoch, checkpoint_dir):
        '''
        保存超分辨率结果
        :param img_list: [lr,hr,sr]
        :param img_name: 图片的名字
        :param epoch: 当前是第几个epoch
        :param checkpoint_dir: 保存路径
        :return:
        '''
        save_dir = os.path.join(checkpoint_dir, img_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for img, name in zip(img_list, ["LR", "HR", str(epoch)]):
            normalized = img[0].data.mul(255 / self.args.rgb_range)
            # permute：交换维度函数，为了适应numpy，从(c,h,w)改为(h,w,c)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            # 使用imageio.imsave函数保存，一共两个参数：路径，numpy数组(图片)
            imageio.imsave(os.path.join(save_dir, "{}.png".format(name)), ndarr)

    def make_optimizer(self):
        # filter函数(判断函数，列表)，过滤列表中的所有项，然后返回能通过函数的项
        # 此处表示提取模型中所有可以训练的参数项
        trainable = filter(lambda x: x.requires_grad, self.model.parameters())
        # loss_trainable = filter(lambda x: x.requires_grad, self.loss.parameters())
        if self.args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': self.args.momentum}
        elif self.args.optimizer == 'ADAM':
            optimizer_function = optim.Adam
            kwargs = {
                'betas': (self.args.beta1, self.args.beta2),
                'eps': self.args.epsilon
            }
        elif self.args.optimizer == 'RMSprop':
            optimizer_function = optim.RMSprop
            kwargs = {'eps': self.args.epsilon}

        kwargs['lr'] = self.args.lr
        kwargs['weight_decay'] = self.args.weight_decay

        return optimizer_function(trainable, **kwargs)
        # return optimizer_function([{"params": trainable}, {"params": loss_trainable}], **kwargs)

    def make_scheduler(self):
        if self.args.decay_type == 'step':
            scheduler = lrs.StepLR(
                self.optimizer,
                step_size=self.args.lr_decay,
                gamma=self.args.gamma
            )
        elif self.args.decay_type.find('step') >= 0:
            milestones = self.args.decay_type.split('_')
            milestones.pop(0)
            milestones = list(map(lambda x: int(x), milestones))
            scheduler = lrs.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=self.args.gamma
            )

        return scheduler

    def is_finsh(self):
        epoch = self.scheduler.last_epoch + 1
        if epoch > self.args.epoch:
            self.checkpoint.save_final()
            return True
        else:
            return False
