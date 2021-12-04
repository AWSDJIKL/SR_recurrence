# -*- coding: utf-8 -*-
'''

'''

# @Time    : 2021/12/3 11:58
# @Author  : LINYANZHEN
# @File    : train.py
import os
import shutil
import time
import matplotlib.pyplot as plt
import torch
import tqdm
import utils
import models
import datasets
import loss
from PIL import Image


def train_and_val(model, train_loader, val_loader, criterion, optimizer, epoch, experiment_name):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    save_path = os.path.join("checkpoint", experiment_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    test_img_path = os.path.join(save_path, "test.png")
    Image.open("img_test/test.png").save(test_img_path)
    psnr_list = []
    loss_list = []
    best_psnr = 0
    for i in range(epoch):
        epoch_psnr = 0
        epoch_loss = 0
        count = 0
        epoch_start_time = time.time()
        print("epoch[{}/{}]".format(i, epoch))
        model.train()
        progress = tqdm.tqdm(train_loader, total=len(train_loader))
        for (x, y) in progress:
            # if index == (len(train_loader) - 1) and i == (epoch - 1):
            #     config.print_grad = True
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            optimizer.zero_grad()
            # print(x.size())
            # print(out.size())
            # print(y.size())
            loss = criterion(out, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            count += len(x)
            x = x.cpu()
            y = y.cpu()
            out = out.cpu()
            loss.cpu()
            torch.cuda.empty_cache()

        epoch_loss /= count
        count = 0
        model.eval()
        with torch.no_grad():
            # progress = tqdm.tqdm(val_loader, total=len(train_loader))
            for index, (x, y) in enumerate(val_loader, 0):
                x = x.cuda()
                y = y.cuda()
                out = model(x)
                epoch_psnr += utils.calculate_psnr(y, out)
                count += len(x)
                x = x.cpu()
                y = y.cpu()
                out = out.cpu()
                torch.cuda.empty_cache()
        epoch_psnr /= count
        psnr_list.append(epoch_psnr)
        loss_list.append(epoch_loss)

        save_name = "{}".format(i)
        utils.test_model(model, test_img_path, 4, save_name)
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            # 保存psnr最优模型
            torch.save(model.state_dict(),
                       os.path.join(save_path, "{}_best.pth".format(experiment_name)))
            print("模型已保存")

        print("psnr:{}  best psnr:{}".format(epoch_psnr, best_psnr))
        print("loss:{}".format(epoch_loss))
        print("  用时:{}min".format((time.time() - epoch_start_time) / 60))
    # 保存最后一个epoch的模型，作为比对
    torch.save(model.state_dict(),
               os.path.join(save_path, "{}_final_epoch.pth".format(experiment_name)))
    print("模型已保存")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_list, 'b', label='psnr')
    plt.legend()
    plt.grid()
    plt.title('best psnr=%5.2f' % best_psnr)
    plt.savefig(os.path.join(save_path, 'psnr.jpg'), dpi=256)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, 'r', label='loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, 'loss.jpg'), dpi=256)
    plt.close()


if __name__ == '__main__':
    train_loader, val_loader = datasets.get_super_resolution_dataloader()
    model = models.EnhanceNet()
    criterion = loss.vgg_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
