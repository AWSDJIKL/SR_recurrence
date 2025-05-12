# -*- coding: utf-8 -*-
'''
SRFormer: Permuted Self-Attention for Single Image Super-Resolution

'''
# @Time    : 2025/3/14 15:56
# @Author  : LINYANZHEN
# @File    : SRFormer.py
# model: SRFormer
# SRFormer: Permuted Self-Attention for Single Image Super-Resolution

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
# from .arch_util import to_2tuple, trunc_normal_
import torch.nn.functional as F
import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger
from model.AdaptivePositionEmbedding import AdaptivePositionEmbedding

# to_2tuple = _ntuple(2)


def make_model(args):
    # inpu = torch.randn(1, 3, 320, 180).cpu()
    # flops, params = profile(RTC(upscale).cpu(), inputs=(inpu,))
    # print(params)
    # print(flops)
    print("prepare model")
    if args.is_PMG:
        print("SRFormer use PMG")
    else:
        print("SRFormer")
    return SRFormer(args, upscale=args.scale)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


class emptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class dwconv(nn.Module):
    def __init__(self, hidden_features):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.before_add = emptyModule()
        self.after_add = emptyModule()
        self.dwconv = dwconv(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = self.before_add(x)
        x = x + self.dwconv(x, x_size)
        x = self.after_add(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class PSA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.permuted_window_size = (window_size[0] // 2, window_size[1] // 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.permuted_window_size[0] - 1) * (2 * self.permuted_window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise aligned relative position index for each token inside the window
        coords_h = torch.arange(self.permuted_window_size[0])
        coords_w = torch.arange(self.permuted_window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.permuted_window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.permuted_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.permuted_window_size[1] - 1
        aligned_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        aligned_relative_position_index = aligned_relative_position_index.reshape \
            (self.permuted_window_size[0], self.permuted_window_size[1], 1, 1,
             self.permuted_window_size[0] * self.permuted_window_size[1]).repeat(1, 1, 2, 2, 1) \
            .permute(0, 2, 1, 3, 4).reshape(4 * self.permuted_window_size[0] * self.permuted_window_size[1],
                                            self.permuted_window_size[0] * self.permuted_window_size[1])  # FN*FN,WN*WN
        self.register_buffer('aligned_relative_position_index', aligned_relative_position_index)
        # compresses the channel dimension of KV
        self.kv = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        # compress the channel dimension of KV :(num_windows*b, num_heads, n//4, c//num_heads)
        kv = self.kv(x).reshape(b_, self.permuted_window_size[0], 2, self.permuted_window_size[1], 2, 2,
                                c // 4).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, n // 4, 2, self.num_heads,
                                                                             c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # keep the channel dimension of Q: (num_windows*b, num_heads, n, c//num_heads)
        q = self.q(x).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (num_windows*b, num_heads, n, n//4)

        relative_position_bias = self.relative_position_bias_table[self.aligned_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.permuted_window_size[0] * self.permuted_window_size[1],
            -1)  # (n, n//4)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, n, n//4)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n // 4) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n // 4)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.permuted_window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        # calculate flops for 1 window with token length of n
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 1.5 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n / 4
        #  x = (attn @ v)
        flops += self.num_heads * n * n / 4 * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class PSA_Block(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.permuted_window_size = window_size // 2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)

        self.attn = PSA(
            dim,
            window_size=_ntuple(2)(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

        # emptyModule for Power Spectrum Based Evaluation
        self.after_norm1 = emptyModule()
        self.after_attention = emptyModule()
        self.residual_after_attention = emptyModule()
        self.after_norm2 = emptyModule()
        self.after_mlp = emptyModule()
        self.residual_after_mlp = emptyModule()

    def calculate_mask(self, x_size):
        # calculate mask for original windows
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # calculate mask for permuted windows
        h, w = x_size
        permuted_window_mask = torch.zeros((1, h // 2, w // 2, 1))  # 1 h w 1
        h_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size,
                                                                -self.shift_size // 2),
                    slice(-self.shift_size // 2, None))
        w_slices = (slice(0, -self.permuted_window_size), slice(-self.permuted_window_size,
                                                                -self.shift_size // 2),
                    slice(-self.shift_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                permuted_window_mask[:, h, w, :] = cnt
                cnt += 1

        permuted_windows = window_partition(permuted_window_mask, self.permuted_window_size)
        permuted_windows = permuted_windows.view(-1, self.permuted_window_size * self.permuted_window_size)
        # calculate attention mask
        attn_mask = mask_windows.unsqueeze(2) - permuted_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = self.after_norm1(x)
        x = x.view(b, h, w, c)

        # cyclic shift
        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)
        x = self.after_attention(x)
        # FFN
        x = shortcut + self.drop_path(x)
        x = self.residual_after_attention(x)
        x = self.residual_after_mlp(
            x + self.drop_path(self.after_mlp(self.mlp(self.after_norm2(self.norm2(x)), x_size))))

        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        # norm1
        flops += self.dim * h * w
        # W-MSA/SW-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * 25
        # norm2
        flops += self.dim * h * w
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,ape=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.ape = ape
        if self.ape:
            self.adaptive_position_embedding = AdaptivePositionEmbedding(dim)
        # build blocks
        self.blocks = nn.ModuleList([
            PSA_Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
            if self.ape and i == 0:
                # print("ape")
                x = self.adaptive_position_embedding(x, x_size[0], x_size[1])
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PSA_Group(nn.Module):
    """Residual Swin Transformer Block (PSA_Group).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',ape=False):
        super(PSA_Group, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,ape=ape)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.before_PSA_Group_conv = emptyModule()
        self.after_PSA_Group_conv = emptyModule()
        self.after_PSA_Group_Residual = emptyModule()

    def forward(self, x, x_size):
        return self.after_PSA_Group_Residual(self.after_PSA_Group_conv(self.patch_embed(
            self.conv(self.patch_unembed(self.before_PSA_Group_conv(self.residual_group(x, x_size)), x_size)))) + x)

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, window_size=22, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if img_size % window_size != 0:
            img_size = img_size + (window_size - img_size % window_size)
        img_size = _ntuple(2)(img_size)
        patch_size = _ntuple(2)(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, window_size=24, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if img_size % window_size != 0:
            img_size = img_size + (window_size - img_size % window_size)
        img_size = _ntuple(2)(img_size)
        patch_size = _ntuple(2)(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


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
#             nn.Linear(embed_dim,2 + 4 * self.num_freq),
#             nn.ReLU()])
#         self.pos_mlp = nn.Sequential(*[
#             nn.Linear(2 + 4 * self.num_freq, embed_dim),
#             nn.ReLU(),
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
#         fourier_feats = self.pos_mlp(fourier_feats + self.c(x))
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


@ARCH_REGISTRY.register()
class SRFormer(nn.Module):
    r""" SRFormer
        A PyTorch implement of : `SRFormer: Permuted Self-Attention for Single Image Super-Resolution`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 args,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(SRFormer, self).__init__()

        self.args = args
        self.support_PMG = True
        self.part = args.part
        self.ape = args.ape
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        # self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            window_size=window_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            window_size=window_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # trunc_normal_(self.absolute_pos_embed, std=.02)
            print("use ape")
            # self.adaptive_position_embedding = nn.ModuleList(
            #     [AdaptivePositionEmbedding(embed_dim, 4 * (i + 1), 10 * (i + 1), num_freq=32 * (4 - i)) for i in        # 32.46
            #      range(self.num_layers)])
            # self.adaptive_position_embedding = nn.ModuleList(
            #     [AdaptivePositionEmbedding(embed_dim, num_freq=4 * (i + 1), max_freq=10 * (i + 1)) for i in
            #      range(self.num_layers)])

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Permuted Self Attention Group  (PSA_Group)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PSA_Group(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,ape=self.ape)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, step=-1):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # for layer in self.layers:
        #     x = layer(x, x_size)

        for i in range(self.part[step]):
            # if self.ape:
            #     pos_emb = x
            #     pos_emb = self.adaptive_position_embedding[i](pos_emb, x_size[0], x_size[1])
            #     x = x + pos_emb
            x = self.layers[i](x, x_size)


        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, step=-1):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, step)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        # elif self.upsampler == 'pixelshuffledirect':
        #     # for lightweight SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.upsample(x)
        # elif self.upsampler == 'nearest+conv':
        #     # for real-world SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.conv_before_upsample(x)
        #     x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #     x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #     x = self.conv_last(self.lrelu(self.conv_hr(x)))
        # else:
        #     # for image denoising and JPEG compression artifact reduction
        #     x_first = self.conv_first(x)
        #     res = self.conv_after_body(self.forward_features(x_first)) + x_first
        #     x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 9 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
