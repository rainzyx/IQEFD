"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from HAM import HAMBlock



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

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
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv 深度卷积
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        # x=SE_Block(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


import torch
from torch import nn


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.  分类头的缩放因子
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()


        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)) # [N, C, H, W]
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm1 = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norm2 = nn.LayerNorm(dims[-1], eps=1e-6)
        # self.head1 = nn.Linear(dims[-1], num_classes)
        self.head1 = nn.Linear(dims[-1], num_classes)
        self.head2 = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)  # 初始化
        self.head1.weight.data.mul_(head_init_scale)
        self.head1.bias.data.mul_(head_init_scale)
        self.head2.weight.data.mul_(head_init_scale)
        self.head2.bias.data.mul_(head_init_scale)

        self.necks = nn.ModuleList()
        d_r = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]
        c = 0
        # d = [768,384,192,192,384,768]
        d = [1024,512,256,256,512,1024]
        for i in range(6):
            neck = nn.Sequential(
                *[Block(dim=d[i], drop_rate=d_r[c + j],layer_scale_init_value=layer_scale_init_value)
                  for j in range(1)]
            )
            self.necks.append(neck)

        # 存储三个上采样模块
        self.ups = nn.ModuleList()
        for i in range(3):
            up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))

            self.ups.append(up)

        self.convs = nn.ModuleList()
        self.hams = nn.ModuleList()
        self.hams1 = nn.ModuleList()
        self.hams_out = HAMBlock(1024)

        # input_chans = [768,768,384,192,192,384]
        # out_chans = [384,192,96,96,192,384]

        input_chans = [1024, 1024, 512, 256, 256, 512]
        out_chans = [512, 256, 128, 128, 256, 512]
        ham1_chans = [128, 256, 512]
        k_s = [1,1,1,3,3,3]
        st_d = [1,1,1,2,2,2]
        pd = [0,0,0,1,1,1]
        for i in range(6):
            conv = nn.Sequential(nn.Conv2d(input_chans[i], out_chans[i], kernel_size=k_s[i], stride=st_d[i], padding=pd[i]))
            self.convs.append(conv)
        for i in range(3):
            ham = nn.Sequential(HAMBlock(out_chans[i]).to('cuda'))
            ham1 = nn.Sequential(HAMBlock(ham1_chans[i]).to('cuda'))
            self.hams.append(ham)
            self.hams1.append(ham1)

# 以上新加D
    # 初始化权重，如果 m 是 Conv2d 或 Linear 层，则使用截断正态分布初始化其权重（标准差 0.2），并将偏置设为 0。
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out_list = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i < 3:
                x = self.cbams1[i](x)
            out_list.append(x)

        l = list(reversed(out_list))
        x = l[0]
        l1 = []

        for i in range(6):
            x = self.convs[i](x)
            if i < 3:
                x = self.cbams[i](x)
            if i < 3:
                l1.append(x)
                x = self.ups[i](x)
                x = torch.cat((x, l[i+1]), 1)
            else:
                l1 = list(reversed(l1))
                x = torch.cat((x,l1[i-3]), 1)
            x = self.necks[i](x)
        b = x
        # x=torch.cat((x, l[0]), 1)
        x = l[0] + x
        HAM_1536 = HAMBlock(1024).to('cuda')
        x = HAM_1536(x)  # 8,1024,8,8 -> 8,1024,8,8
        x = self.cbams_out(x)
        # x = l[0] + x
        return self.norm1(x.mean([-2, -1])),self.norm2(l[0].mean([-2, -1])),l[0],b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0,out0,convnext_feature,final_feature = self.forward_features(x)  # new
        # x0.shape: [8,1024]
        x = self.head1(x0)
        out = self.head2(out0)

        return x,out,x0,out0,convnext_feature,final_feature


def convnext_tiny(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_small(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_base(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def convnext_large(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model