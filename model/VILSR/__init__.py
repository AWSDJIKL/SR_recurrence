# -*- coding: utf-8 -*-
'''
以ESRT为骨架，尝试将ViL运用到SR中
'''
# @Time    : 2024/8/14 23:02
# @Author  : LINYANZHEN
# @File    : __init__.py


from . import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from .non import NONLocalBlock2D
from .tools import extract_image_patches, \
    reduce_mean, reduce_sum, same_padding, reverse_patches

# from .transformer import drop_path, DropPath, PatchEmbed, Mlp, MLABlock
# from .position import PositionEmbeddingLearned, PositionEmbeddingSine
import pdb
import math
import einops
from .vision_lstm2 import ViLBlockPair, LayerNorm
from .vision_lstm_util import VitPatchEmbed, VitPosEmbed2d


def make_model(args):
    # inpu = torch.randn(1, 3, 320, 180).cpu()
    # flops, params = profile(RTC(upscale).cpu(), inputs=(inpu,))
    # print(params)
    # print(flops)
    print("prepare model")
    if args.is_PMG:
        print("VILSR use APMG")
    else:
        print("VILSR")
    return VILSR(args, upscale=args.scale)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
        super(one_conv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate, inchanels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)

    def forward(self, x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output  # torch.cat((x,output),1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                           padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class one_module(nn.Module):
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats // 2, 3)
        self.layer2 = one_conv(n_feats, n_feats // 2, 3)
        # self.layer3 = one_conv(n_feats, n_feats//2,3)
        self.layer4 = BasicConv(n_feats, n_feats, 3, 1, 1)
        self.alise = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
        self.atten = CALayer(n_feats)
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)
        self.weight3 = common.Scale(1)
        self.weight4 = common.Scale(1)
        self.weight5 = common.Scale(1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # pdb.set_trace()
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2), self.weight3(x1)], 1))))
        return self.weight4(x) + self.weight5(x4)


class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = one_module(n_feats)
        self.decoder_low = one_module(n_feats)  # nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_high = one_module(n_feats)
        self.alise = one_module(n_feats)
        self.alise2 = BasicConv(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        for i in range(5):
            x2 = self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x


class Un(nn.Module):
    def __init__(self, n_feats, wn, pool_stride=8):
        super(Un, self).__init__()
        self.pool_stride = pool_stride
        self.encoder1 = Updownblock(n_feats)
        self.encoder2 = Updownblock(n_feats)
        self.encoder3 = Updownblock(n_feats)
        self.reduce = common.default_conv(3 * n_feats, n_feats, 3)
        self.weight2 = common.Scale(1)
        self.weight1 = common.Scale(1)
        # transformer部分
        # self.attention = MLABlock(n_feat=n_feats, dim=288)
        # 尝试直接替换为vision_lstm
        # 需要先压成一维
        self.attention = ViLBlockPair(dim=n_feats)

        self.alise = common.default_conv(n_feats, n_feats, 3)

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=n_feats,
            num_channels=n_feats,
            resolution=(96, 96),
            patch_size=pool_stride,
        )

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=n_feats)

        self.downsample = nn.Conv2d(n_feats, n_feats, kernel_size=pool_stride, stride=pool_stride)

        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * pool_stride * pool_stride, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=pool_stride))

    def forward(self, x):
        # out = self.encoder3(self.encoder2(self.encoder1(x)))
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        # out = x3
        b, c, h, w = x3.shape
        out = self.reduce(torch.cat([x1, x2, x3], dim=1))

        # flatten to 1d

        # print("0", out.size())  # [1, 32, 96, 96]
        # embed patches
        # 原版有一个池化极大降低了现存占用
        out = self.downsample(out)
        # print("1", out.size())
        out = einops.rearrange(out, "b c ... -> b ... c")
        # print("2", out.size())
        out = einops.rearrange(out, "b ... d -> b (...) d")
        # print("3", out.size())
        out = self.attention(out)

        # seqlen_h, seqlen_w = self.patch_embed.seqlens
        # 还原
        # 先还原为原本的大小

        out = einops.rearrange(
            out,
            "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
            seqlen_h=h // self.pool_stride,
            seqlen_w=w // self.pool_stride
        )
        # print("4", out.size())
        out = self.upsample(out)
        # print("5", out.size())
        # out = out.permute(0, 2, 1)
        # out = reverse_patches(out, (h, w), (3, 3), 1, 1)

        out = self.alise(out)
        # print(x.size())
        return self.weight1(x) + self.weight2(out)


class VILSR(nn.Module):
    def __init__(self, args, upscale=4, conv=common.default_conv):
        super(VILSR, self).__init__()

        self.support_PMG = False

        # 权重归一化层，但貌似没用上
        wn = lambda x: torch.nn.utils.weight_norm(x)
        n_feats = 192
        n_blocks = 12
        kernel_size = 3
        scale = upscale  # args.scale[0] #gaile
        print(scale)
        act = nn.ReLU(True)
        # self.up_sample = F.interpolate(scale_factor=2, mode='nearest')
        self.n_blocks = n_blocks
        self.args = args
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # ################################
        # # define head module
        # # 头部，一个简单的3x3卷积层
        # modules_head = [conv(3, n_feats, kernel_size)]
        #
        # # define body module
        # modules_body = nn.ModuleList()
        # for i in range(n_blocks):
        #     modules_body.append(
        #         Un(n_feats=n_feats, wn=wn, pool_stride=self.args.pool_stride))
        #
        # # define tail module
        # # 尾部，图像重建模块
        # modules_tail = [
        #
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, 3, kernel_size)]
        #
        # self.up = nn.Sequential(common.Upsampler(conv, scale, n_feats, act=False),
        #                         BasicConv(n_feats, 3, 3, 1, 1))
        # self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        # self.tail = nn.Sequential(*modules_tail)
        # self.reduce = conv(n_blocks * n_feats, n_feats, kernel_size)
        # ####################################

        self.pool_stride = args.pool_stride
        self.head = conv(3, n_feats, kernel_size)

        self.patch_embed = VitPatchEmbed(
            dim=n_feats,
            num_channels=n_feats,
            resolution=(96, 96),
            patch_size=args.pool_stride,
        )
        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=n_feats)

        self.downsample = nn.Conv2d(n_feats, n_feats, kernel_size=args.pool_stride, stride=args.pool_stride)

        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * args.pool_stride * args.pool_stride, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=args.pool_stride))

        # merge two blocks into a blockpair to keep depth equal to the depth of transformers
        # useful to keep layer-wise lr decay implementations consistent with transformers
        # self.blocks = nn.ModuleList(
        #     [
        #         ViLBlockPair(dim=n_feats)
        #         # ViLBlockPair(
        #         #     dim=dim,
        #         #     drop_path=dpr[i],
        #         #     conv_kind=conv_kind,
        #         #     seqlens=self.patch_embed.seqlens,
        #         #     proj_bias=proj_bias,
        #         #     norm_bias=norm_bias,
        #         # )
        #         for i in range(n_blocks)
        #     ],
        # )

        self.block1 = nn.ModuleList(
            [
                ViLBlockPair(dim=n_feats)
                for i in range(6)
            ],
        )
        self.block2 = nn.ModuleList(
            [
                ViLBlockPair(dim=n_feats)
                for i in range(6)
            ],
        )
        self.block3 = nn.ModuleList(
            [
                ViLBlockPair(dim=n_feats)
                for i in range(6)
            ],
        )
        self.block4 = nn.ModuleList(
            [
                ViLBlockPair(dim=n_feats)
                for i in range(6)
            ],
        )

        self.norm = LayerNorm(n_feats, bias=True, eps=1e-6)
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x1, x2=None, test=False):
        # ###########################
        # # x1 = self.sub_mean(x1)
        # x1 = self.head(x1)
        # res2 = x1
        # # res2 = x2
        # body_out = []
        # for i in range(self.n_blocks):
        #     x1 = self.body[i](x1)
        #     body_out.append(x1)
        # res1 = torch.cat(body_out, 1)
        # res1 = self.reduce(res1)
        #
        # x1 = self.tail(res1)
        # x1 = self.up(res2) + x1
        # # x1 = self.add_mean(x1)
        # # x2 = self.tail(res2)
        # return x1
        # ###################################

        out = self.head(x1)
        b, c, h, w = out.shape
        out = self.downsample(out)
        # print("1", out.size())
        out = einops.rearrange(out, "b c ... -> b ... c")
        # print("2", out.size())
        out = einops.rearrange(out, "b ... d -> b (...) d")

        # # apply blocks
        # for i in range(self.n_blocks):
        #     out = self.blocks[i](out)
        # out = self.norm(out)
        # out = einops.rearrange(
        #     out,
        #     "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
        #     seqlen_h=h // self.pool_stride,
        #     seqlen_w=w // self.pool_stride
        # )

        for block in self.block1:
            out = block(out)
        out = einops.rearrange(
            out,
            "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
            seqlen_h=h // self.pool_stride,
            seqlen_w=w // self.pool_stride
        )
        x1 = out

        out = einops.rearrange(out, "b c ... -> b ... c")
        out = einops.rearrange(out, "b ... d -> b (...) d")
        for block in self.block2:
            out = block(out)
        out = einops.rearrange(
            out,
            "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
            seqlen_h=h // self.pool_stride,
            seqlen_w=w // self.pool_stride
        )
        x2 = out

        out = einops.rearrange(out, "b c ... -> b ... c")
        out = einops.rearrange(out, "b ... d -> b (...) d")
        for block in self.block3:
            out = block(out)
        out = einops.rearrange(
            out,
            "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
            seqlen_h=h // self.pool_stride,
            seqlen_w=w // self.pool_stride
        )
        x3 = out

        out = einops.rearrange(out, "b c ... -> b ... c")
        out = einops.rearrange(out, "b ... d -> b (...) d")
        for block in self.block4:
            out = block(out)
        out = einops.rearrange(
            out,
            "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
            seqlen_h=h // self.pool_stride,
            seqlen_w=w // self.pool_stride
        )
        x4 = out

        out = x1 + x2 + x3 + x4

        # print("4", out.size())
        out = self.upsample(out)
        out = self.tail(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        # MSRB_out = []from model import common
