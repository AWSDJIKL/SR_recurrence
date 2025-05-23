# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2024/8/21 23:42
# @Author  : LINYANZHEN
# @File    : transformer_trainer.py
import shutil
from datetime import datetime

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
import time


class Trainer():
    def __init__(self, args, model, loader: data.Data, loss):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.model = model
        self.is_PMG = args.is_PMG
        self.is_crop = args.is_crop
        self.growth_stage = 0
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
        now_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.is_PMG and model.support_PMG:
            crop_piece = "_".join(map(str, args.crop_piece))
            experiment_name = "x{}_{}_APMG_{}_stride{}".format(args.scale, model.__class__.__name__, crop_piece,
                                                               args.stride)
        else:
            args.is_PMG = False
            self.is_PMG = False
            experiment_name = "x{}_{}".format(args.scale, model.__class__.__name__)
        # experiment_name = "{}_x{}_{}".format(now_time, args.scale, model.__class__.__name__)
        self.checkpoint = utils.Checkpoint(args, self.model, experiment_name)
        if args.load_checkpoint:
            # 加载上一次的模型，优化器和学习率调整器
            self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint.checkpoint_dir, "optimizer.pth")))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint.checkpoint_dir, "scheduler.pth")))
            last_epoch = self.scheduler.last_epoch
            checkpoint = torch.load(os.path.join(self.checkpoint.checkpoint_dir, "model/final.pth"))
            self.model.load_state_dict(checkpoint)

    def train_and_test(self):
        start_time = time.time()
        epoch = self.scheduler.last_epoch + 1

        # 根据epoch判断是否增长网络
        if self.is_PMG:
            if epoch == self.args.growth_schedule[self.growth_stage]:
                self.checkpoint.write_log("网络增长")
                self.growth_stage += 1
                self.model.grow()
                self.optimizer = self.make_optimizer()
                self.checkpoint.write_log("Growth stage: {}".format(self.growth_stage))

        learn_percent = 0.5 + 0.1 * (epoch // 40)
        lr = self.scheduler.get_last_lr()[0]
        self.checkpoint.write_log(
            "[Epoch {}/{}]\tLearning rate: {}".format(epoch, self.args.epoch, lr))
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
                if self.is_crop:
                    n = self.args.crop_piece[self.growth_stage]
                    if n > 1:
                        lr = utils.crop_img(lr, lr_size, n, self.args.stride)
                        hr = utils.crop_img(hr, hr_size, n, self.args.stride)
                self.optimizer.zero_grad()
                sr = self.model(lr)
                loss = self.loss(sr, hr)
                loss.backward()
                self.optimizer.step()

                # 更新动量网络
                self.model.cpu()
                self.model.renew_momentum()
                self.model.to(self.device)

                epoch_loss += loss.item() / len(self.args.crop_piece)
            else:
                self.optimizer.zero_grad()
                sr = self.model(lr)
                # self.save_sr_result([lr, hr, sr], img_name[0], epoch, self.checkpoint.checkpoint_dir)
                # return
                loss = self.loss(sr, hr)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                # break
        epoch_loss /= len(self.train_loader)
        with torch.no_grad():
            self.model.eval()
            # print("test")
            for (lr, hr, img_name) in self.test_loader:
                lr = self.prepare(lr)
                hr = self.prepare(hr)
                sr = self.model(lr)
                # print(sr)
                sr = utils.quantize(sr, self.args.rgb_range)
                #
                if self.args.save_epoch_result:
                    self.save_sr_result([lr, hr, sr], img_name[0], epoch, self.checkpoint.checkpoint_dir)
                psnr = utils.calculate_psnr(sr, hr, self.args.scale, self.args.rgb_range)
                epoch_psnr += psnr
        epoch_psnr /= len(self.test_loader)
        self.scheduler.step()
        self.checkpoint.record_epoch(epoch, epoch_loss, epoch_psnr, self.optimizer, self.scheduler)
        end_time = time.time()
        total_time = end_time - start_time
        self.checkpoint.write_log("epoch time={}s".format(total_time))

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
        else:
            print("Not supported optimizer: {}.".format(self.args.optimizer))
            raise NotImplementedError
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
