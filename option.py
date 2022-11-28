# -*- coding: utf-8 -*-
'''
训练配置
'''
# @Time    : 2021/12/18 12:14
# @Author  : LINYANZHEN
# @File    : option.py
import argparse

parser = argparse.ArgumentParser(description='Super Resolution framework')
# dataset setting
parser.add_argument("--train_set", type=str, default="DIV2K", help="train set name")
parser.add_argument("--test_set", type=str, default="Set5", help="test set name")
parser.add_argument("--scale", type=int, default=4, help="")
parser.add_argument("--data_type", type=str, default="npy", help="")
parser.add_argument("--batch_size", type=int, default=4, help="")
parser.add_argument("--patch_size", type=int, default=384, help="")
parser.add_argument("--rgb_range", type=int, default=255, help="")
parser.add_argument("--n_color", type=int, default=3, help="")
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
# model setting
parser.add_argument("--model_name", type=str, default="RFDN", help="")
parser.add_argument("--is_PMG", type=lambda x: x.lower() == 'true', default=False, help="")
parser.add_argument("--is_crop", type=lambda x: x.lower() == 'true', default=True, help="")
parser.add_argument("--crop_piece", nargs='+', type=int, default=[6, 4, 3, 1], help="")
parser.add_argument("--stride", type=float, default=3, help="")

parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=False,
                    help='subtract pixel mean from the input')

parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# train setting
parser.add_argument("--epoch", type=int, default=1000, help="")
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument("--seed", type=int, default=100, help="")
parser.add_argument("--device", type=str, default="cuda:0",
                    help="use which device to train")
parser.add_argument("--half_precision", type=lambda x: x.lower() == 'true', default=False,
                    help="use half precision")

# Optimization setting
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
# Loss setting
parser.add_argument('--loss_name', type=str, default='1_L1', help='loss function configuration')
# parser.add_argument('--loss_name', type=str, default='1_shadow', help='loss function configuration')
# parser.add_argument('--loss_name', type=str, default='1_spl', help='loss function configuration')
# parser.add_argument('--loss_name', type=str, default='1_L1+1e-3_VGG', help='loss function configuration')

# 是否加载上一次的保存点
parser.add_argument('--load_checkpoint', type=lambda x: x.lower() == 'true', default=False,
                    help='load checkpoint to continue train')
parser.add_argument('--save_epoch_result', type=lambda x: x.lower() == 'true', default=False,
                    help='save every epoch sr result ')
args = parser.parse_args()
