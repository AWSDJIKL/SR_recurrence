import shutil

import imageio
import torch
import tqdm
import data
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import utils


class ProgressiveTrainer():
    def __init__(self, args, model, loader: data.ProgressiveData, loss):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.model = model
        self.is_PMG = args.is_PMG
        self.loss = loss
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
            experiment_name = "x{}_{}_progressive_{}".format(args.scale, model.__class__.__name__, loss.loss_name)
        else:
            args.is_PMG = False
            self.is_PMG = False
            experiment_name = "x{}_{}_{}".format(args.scale, model.__class__.__name__, loss.loss_name)
        self.checkpoint = utils.Checkpoint(args, self.model, experiment_name)
        if args.load_checkpoint:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint.checkpoint_dir, "optimizer.pth")))
            self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint.checkpoint_dir, "scheduler.pth")))
            checkpoint = torch.load(os.path.join(self.checkpoint.checkpoint_dir, "model/final.pth"))
            self.model.load_state_dict(checkpoint)

    def train_and_test(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        self.checkpoint.write_log("[Epoch {}/{}]\tLearning rate: {}".format(epoch, self.args.epoch, lr))
        self.model.train()
        epoch_loss = 0
        epoch_psnr = 0
        stages = self.args.scale // 2
        for i in range(stages):
            progress = tqdm.tqdm(self.train_loader[i], total=len(self.train_loader[i]))
            for (lr, hr, img_name) in progress:
                lr = self.prepare(lr)
                hr = self.prepare(hr)
                self.optimizer.zero_grad()
                sr = self.model(lr, i)
                loss = self.loss(sr, hr)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
        epoch_loss /= len(self.train_loader[0])
        with torch.no_grad():
            self.model.eval()
            for (lr, hr, img_name) in self.test_loader:
                lr = self.prepare(lr)
                hr = self.prepare(hr)
                sr = self.model(lr)
                sr = utils.quantize(sr, self.args.rgb_range)
                if self.args.save_epoch_result:
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
        save_dir = os.path.join(checkpoint_dir, img_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for img, name in zip(img_list, ["LR", "HR", str(epoch)]):
            normalized = img[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            imageio.imsave(os.path.join(save_dir, "{}.png".format(name)), ndarr)

    def make_optimizer(self):
        trainable = filter(lambda x: x.requires_grad, self.model.parameters())
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
