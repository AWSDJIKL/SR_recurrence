# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2025/5/1 8:54
# @Author  : LINYANZHEN
# @File    : AdaptivePositionEmbedding.py
import torch.nn as nn
import torch
import torch.nn.functional as F
from model import shiftconv


class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class AdaptivePositionEmbedding(nn.Module):
    def __init__(self, embed_dim):
        '''
        自适应位置编码，从输入特征中提取位置编码
        Args:
            embed_dim:
        '''
        super(AdaptivePositionEmbedding, self).__init__()

        self.embed_dim = embed_dim
        # self.safm = SAFM(embed_dim)

        self.scb = shiftconv.Block(embed_dim, kernel_size=(24, 3))

        # self.elab = ELAB.ELAB(embed_dim, embed_dim)

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.ca_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 16, 1, bias=False),
        #                              nn.ReLU(),
        #                              nn.Conv2d(embed_dim // 16, embed_dim, 1, bias=False))
        # self.sa_conv = nn.Conv2d(2, 1, 3, 1, 1)
        # self.relu = nn.ReLU()
        self.condition_embedding = nn.Sequential(
            *[nn.Conv2d(embed_dim, embed_dim, 7, 1, 3),
              nn.ReLU(),
              nn.Conv2d(embed_dim, embed_dim, 5, 1, 2),
              nn.ReLU(),
              nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
              nn.ReLU()]
        )

    def forward(self, x, H, W):
        B, N, C = x.shape

        pos_emb = x.transpose(1, 2).view(B, C, H, W)
        # pos_emb = torch.ones(1, C, H, W).to(x.device)

        # pos_emb = self.condition_embedding(pos_emb)
        pos_emb = self.safm(pos_emb)
        # pos_emb=self.elab(pos_emb)

        pos_emb = pos_emb.flatten(2).transpose(1, 2)
        return pos_emb


# class AdaptivePositionEmbedding(nn.Module):
#     def __init__(self, embed_dim, kernel_size=1, dilation=1):
#         '''
#         自适应位置编码，从输入特征中提取位置编码
#         Args:
#             embed_dim:
#         '''
#         super(AdaptivePositionEmbedding, self).__init__()
#
#         self.embed_dim = embed_dim
#         padding = int(dilation * (kernel_size - 1) / 2)
#         # 31.73
#         # self.condition_embedding = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         # 31.81 (hat stride[3,3,3] dilation[3,3,3])
#
#         # self.condition_embedding = nn.Sequential(
#         #     *[nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, padding, dilation),
#         #       nn.ReLU(),
#         #       nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, padding, dilation),
#         #       nn.ReLU(),
#         #       nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, padding, dilation),
#         #       nn.ReLU()]
#         # )
#         # 31.81 (hat stride[3,3,3] dilation[1,3,1])
#         self.condition_embedding = nn.Sequential(
#             *[nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, 1, 1),
#               nn.ReLU(),
#               nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, padding, dilation),
#               nn.ReLU(),
#               nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, 1, 1),
#               nn.ReLU()]
#         )
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         feat_token = x
#         x = self.condition_embedding(x)
#         x = x + feat_token
#         x = x.flatten(2).transpose(1, 2)
#         return x


# class AdaptivePositionEmbedding(nn.Module):
#     def __init__(self, embed_dim, sigma=2, M=10, num_freq=64, max_freq=8):
#         '''
#         自适应位置编码，从输入特征中提取位置编码
#         Args:
#             embed_dim:
#         '''
#         super(AdaptivePositionEmbedding, self).__init__()
#
#         self.embed_dim = embed_dim
#         self.sigma = sigma
#         self.M = M
#         self.num_freq = num_freq
#         self.max_freq = max_freq
#         # 生成频率范围（指数或线性增长）
#         # self.freqs = torch.linspace(0, max_freq, num_freq) * (2 * math.pi)
#         # 或者使用指数增长：
#         # self.freqs = 2 ** torch.linspace(0, max_freq, num_freq) * (2 * math.pi)
#         # 可学习频率参数（对数空间初始化）
#         self.log_freqs = nn.Parameter(torch.linspace(0, math.log(max_freq), num_freq), requires_grad=True)  # 32.3621
#         # print('sigma:', sigma, 'M:', M)
#         self.c = nn.Sequential(*[
#             nn.Linear(2 + 4 * self.num_freq, embed_dim),
#             nn.ReLU()])
#         self.pos_mlp = nn.Sequential(*[
#             # nn.Linear(2 + 4 * self.num_freq, embed_dim),
#             # nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim * 16),
#             nn.ReLU(),
#             nn.Linear(embed_dim * 16, embed_dim * 32),
#             nn.ReLU(),
#             nn.Linear(embed_dim * 32, embed_dim * 16),
#             nn.ReLU(),
#             nn.Linear(embed_dim * 16, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#         ])
#
#         # # 可学习频率参数（对数空间初始化）
#         # self.log_freqs = nn.Parameter(torch.linspace(0, math.log(max_freq), num_freq), requires_grad=True)  # 32.3621 num_freq=64, max_freq=8
#         # # print('sigma:', sigma, 'M:', M)
#         # self.pos_mlp = nn.Sequential(*[
#         #     nn.Linear(2 + 4 * self.num_freq, embed_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(embed_dim, embed_dim * 16),
#         #     nn.ReLU(),
#         #     nn.Linear(embed_dim * 16, embed_dim * 32),
#         #     nn.ReLU(),
#         #     nn.Linear(embed_dim * 32, embed_dim * 16),
#         #     nn.ReLU(),
#         #     nn.Linear(embed_dim * 16, embed_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(embed_dim, embed_dim),
#         #     nn.ReLU(),
#         # ])
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         # 获取可学习频率（指数转换确保正值）
#         freqs = torch.exp(self.log_freqs) * (2 * math.pi)  # (num_freq,)
#         # 生成归一化坐标网格
#         x_coords = torch.linspace(0, 1, W, device=x.device)
#         y_coords = torch.linspace(0, 1, H, device=x.device)
#         x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')  # 创建二维网格
#         # 生成傅里叶特征
#         x_f = x_grid.reshape(-1)  # (H*W,)
#         y_f = y_grid.reshape(-1)  # (H*W,)
#         # 对每个频率计算sin和cos
#         fourier_feats = []
#         for freq in freqs:
#             fourier_feats.extend([
#                 torch.sin(freq * x_f),
#                 torch.cos(freq * x_f),
#                 torch.sin(freq * y_f),
#                 torch.cos(freq * y_f)
#             ])
#         fourier_feats = torch.stack(fourier_feats, dim=1)  # (H*W, 4*num_freq)
#         # 拼接原始坐标
#         raw_coords = torch.stack([x_f, y_f], dim=1)  # (H*W, 2)
#         fourier_feats = torch.cat([raw_coords, fourier_feats], dim=1)  # (H*W, 2 + 4*num_freq)
#         # 扩展为批次维度并返回
#         fourier_feats = fourier_feats.unsqueeze(0)  # (1, H*W, d)
#         # print(fourier_feats.shape)
#         # if batch_size > 1:
#         #     fourier_feats = fourier_feats.repeat(batch_size, 1, 1)  # (B, H*W, d)
#         # 使用多层感知机对傅里叶特征进行处理
#         # fourier_feats = self.pos_mlp(fourier_feats + self.c(x))
#         fourier_feats = self.pos_mlp(x + self.c(fourier_feats))
#         return fourier_feats
#
#         # # 从高斯分布中随机采样频率矩阵 B
#         # B_matrix = torch.randn(self.M, C, device=x.device) * self.sigma
#         #
#         # # 计算傅里叶特征
#         # # input_tensor: (B, N, C) -> (B, N, 1, C)
#         # # B_matrix: (num_features, C) -> (1, 1, num_features, C)
#         # input_tensor = x
#         # input_tensor = input_tensor  # (B, N, 1, C)
#         # B_matrix = B_matrix.unsqueeze(0)  # (1, 1, num_features, C)
#         #
#         # # 计算 2πBv
#         # freq_product = 2 * torch.pi * torch.matmul(input_tensor, B_matrix.transpose(-1, -2))  # (B, N, num_features)
#         #
#         # # 计算 cos 和 sin
#         # cos_features = torch.cos(freq_product)  # (B, N, num_features)
#         # sin_features = torch.sin(freq_product)  # (B, N, num_features)
#         #
#         # # 合并 cos 和 sin 特征
#         # fourier_features = torch.cat([cos_features, sin_features], dim=-1)  # (B, N, 2 * num_features)
#         # fourier_features = self.pos_mlp(fourier_features)
#         # return fourier_features


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x
