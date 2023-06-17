import torch
import torch.nn as nn
from network.model_utils import LearnedSinusoidalPosEmb, activation_function
import ocnn
from ocnn.utils import scatter_add
from ocnn.octree import Octree
import numpy as np


class OctreeGroupNorm(torch.nn.Module):
    def __init__(self, num_groups: int, num_channels: int, nempty: bool = False):
        super().__init__()

        self.eps = 1e-5
        self.nempty = nempty
        self.G = num_groups
        self.C = num_channels
        assert self.C % self.G == 0
        self.C_in_G = self.C//self.G
        self.weights = torch.nn.Parameter(torch.Tensor(1, num_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        r''''''
        batch_size = octree.batch_size

        batch_id = octree.batch_id(depth, self.nempty)

        ones = data.new_ones([data.shape[0], 1])

        count = scatter_add(ones, batch_id, dim=0,
                            dim_size=batch_size) * self.C_in_G

        norm = 1.0 / (count + self.eps)
        mean = scatter_add(data, batch_id, dim=0, dim_size=batch_size) * norm

        mean = mean.reshape(batch_size, self.G, self.C_in_G).sum(-1,
                                                                 keepdim=True).repeat(1, 1, self.C_in_G).reshape(batch_size, self.C)

        out = data - mean[batch_id]

        var = scatter_add(out * out, batch_id, dim=0,
                          dim_size=batch_size) * norm
        var = var.reshape(batch_size, self.G, self.C_in_G).sum(-1,
                                                               keepdim=True).repeat(1, 1, self.C_in_G).reshape(batch_size, self.C)
        inv_std = 1.0 / (var + self.eps).sqrt()

        out = out * inv_std[batch_id]

        out = out * self.weights + self.bias
        return out


class OctreeResBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck: int = 4, nempty: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck
        channelb = int(out_channels / bottleneck)
        self.conv1a_norm = normalization(channelb)
        self.conv3_norm = normalization(channelb)

        self.conv1x1a = ocnn.modules.Conv1x1(in_channels, channelb)
        self.conv3x3 = ocnn.nn.OctreeConv(
            channelb, channelb, kernel_size=[3], nempty=nempty, use_bias=True)
        self.conv1x1b = ocnn.modules.Conv1x1(channelb, out_channels)
        self.conv1x1c = ocnn.modules.Conv1x1(
            in_channels, out_channels) if self.in_channels != self.out_channels else our_Identity()

        self.silu = activation_function()

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        # print(data.shape)

        conv1 = self.silu(self.conv1a_norm(self.conv1x1a(data), octree, depth))
        conv2 = self.silu(self.conv3_norm(
            self.conv3x3(conv1, octree, depth), octree, depth))
        out = self.conv1x1b(conv2) + self.conv1x1c(data)

        return out


def normalization(channels):
    num_groups = min(32, channels)
    return OctreeGroupNorm(num_groups=num_groups, num_channels=channels, nempty=True)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class our_Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.up_pool = ocnn.nn.OctreeUpsample(method="nearest", nempty=True)
        if use_conv:
            self.conv = ocnn.nn.OctreeConv(
                channels, channels, kernel_size=[3], nempty=True, use_bias=True)

    def forward(self, x, octree, depth):
        x = self.up_pool(x, octree, depth)

        if self.use_conv:
            x = self.conv(x, octree, depth + 1)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = ocnn.nn.OctreeConv(
                channels, channels, kernel_size=[3], stride=2, nempty=True, use_bias=True)
        else:
            self.op = ocnn.nn.OctreeMaxPool(nempty=True)

    def forward(self, x, octree, depth):
        return self.op(x, octree, depth)


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, emb_dim, condition_input=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            activation_function(),
            nn.Linear(emb_dim, dim_out)
        )
        self.condition_input = condition_input
        if self.condition_input:
            self.condition_mlp = nn.Linear(emb_dim, dim_out)
        self.block1_norm = normalization(dim_in)
        self.block2_norm = normalization(dim_out)
        self.block1 = ocnn.nn.OctreeConv(
            dim_in, dim_out, kernel_size=[3], nempty=True, use_bias=True)
        self.block2 = zero_module(ocnn.nn.OctreeConv(
            dim_out, dim_out, kernel_size=[3], nempty=True, use_bias=True))

        self.silu = activation_function()
        self.res_conv = ocnn.modules.Conv1x1(
            dim_in, dim_out, use_bias=True) if dim_in != dim_out else our_Identity()

    def forward(self, x, time_emb, octree, depth, conditon=None):
        h = self.silu(self.block1_norm(x, octree, depth))

        h = self.block1(h, octree, depth)

        batch_size = time_emb.shape[0]
        t = self.time_mlp(time_emb)
        for i in range(batch_size):
            h[octree.batch_id(depth, True) == i] += t[i]
        h = self.silu(self.block2_norm(h, octree, depth))

        h = self.block2(h, octree, depth)

        return h + self.res_conv(x)


class Sparsity_UNetModel(nn.Module):
    def __init__(self,
                 base_channels: int = 64,
                 base_size: int = 32,
                 upfactor: int = 2,
                 dim_mults=(1, 4),
                 condition_classes: int = 1,
                 verbose: bool = False,
                 ):
        super().__init__()

        ref_dim_mults = (1, 2, 4, 8, 8, 16)
        assert base_size >= 8
        length = int(np.log2(upfactor)) + 4
        dim_mults = ref_dim_mults[:length]

        channels = [base_channels, *
                    map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(channels[:-1], channels[1:]))

        self.verbose = verbose
        emb_dim = base_channels * 4

        if condition_classes > 1:
            self.condition_input = True
            self.condition_emb = nn.Linear(condition_classes, emb_dim)
        else:
            self.condition_input = False

        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.input_emb = ocnn.nn.OctreeConv(
            1, base_channels, kernel_size=[3], nempty=True, use_bias=True)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out,
                            emb_dim=emb_dim, condition_input=self.condition_input),
                Downsample(dim_out) if not is_last else our_Identity()
            ]))

        mid_dim = channels[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, emb_dim=emb_dim, condition_input=self.condition_input)

        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, emb_dim=emb_dim, condition_input=self.condition_input)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in,
                            emb_dim=emb_dim, condition_input=self.condition_input),
                Upsample(
                    dim_in) if not is_last else our_Identity()
            ]))

        self.end_norm = normalization(base_channels)
        self.end = activation_function()
        self.out = ocnn.nn.OctreeConv(
            base_channels, 1, kernel_size=[3], nempty=True, use_bias=True)

    def forward(self, x, t, octree, condition=None):
        d = octree.depth
        x = self.input_emb(x, octree, d)
        t = self.time_emb(self.time_pos_emb(t))
        h = []

        for resnet, downsample in self.downs:
            x = resnet(x, t, octree, d, condition)
            if self.verbose:
                print(d)
                print(x.shape)
            h.append(x)
            x = downsample(x, octree, d)
            d -= 1
            if self.verbose:
                print(x.shape)

        d += 1
        x = self.mid_block1(x, t, octree, d, condition)
        if self.verbose:
            print(x.shape)
        x = self.mid_block2(x, t, octree, d, condition)
        if self.verbose:
            print(x.shape)

        for resnet, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            if self.verbose:
                print(x.shape)
            x = resnet(x, t, octree, d, condition)
            if self.verbose:
                print(x.shape)
            x = upsample(x, octree, d)
            d += 1
            if self.verbose:
                print(x.shape)

        x = self.end(self.end_norm(x, octree, d))

        if self.verbose:
            print(x.shape)
        return self.out(x, octree, d)
