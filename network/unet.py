import torch
import torch.nn as nn
from network.model_utils import *
from utils.utils import default, VIT_FEATURE_CHANNEL, CLIP_FEATURE_CHANNEL


class UNetModel(nn.Module):
    def __init__(self,
                 image_size: int = 64,
                 base_channels: int = 64,
                 dim_mults=(1, 2, 4, 8),
                 dropout: float = 0.1,
                 num_heads: int = 1,
                 world_dims: int = 3,
                 attention_resolutions=(4, 8),
                 with_attention: bool = False,
                 verbose: bool = False,
                 image_condition_dim: int = VIT_FEATURE_CHANNEL,
                 text_condition_dim: int = CLIP_FEATURE_CHANNEL,
                 kernel_size: float = 1.0,
                 use_sketch_condition: bool = True,
                 use_text_condition: bool = True,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 ):
        super().__init__()
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        channels = [base_channels, *
                    map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(channels[:-1], channels[1:]))

        self.verbose = verbose
        emb_dim = base_channels * 4

        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        )
        if self.use_text_condition:
            self.text_emb = nn.Sequential(
                nn.Linear(text_condition_dim, emb_dim),
                activation_function(),
                nn.Linear(emb_dim, emb_dim)
            )

        self.input_emb = conv_nd(world_dims, 2, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            use_cross = (res == 4 or res == 8)
            self.downs.append(nn.ModuleList([
                ResnetBlock(world_dims, dim_in, dim_out,
                            emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition),
                CrossAttention(feature_dim=dim_out, sketch_dim=image_condition_dim,
                               kernel_size=kernel_size, vit_local=vit_local, vit_global=vit_global,
                               num_heads=num_heads, image_size=res, world_dims=3,
                               drop_out=dropout) if use_cross and self.use_sketch_condition else our_Identity(),
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(
                        dim_out, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Downsample(
                    dim_out, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds *= 2

        mid_dim = channels[-1]
        res = image_size // ds
        self.mid_block1 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition)
        
        self.mid_cross_attn = CrossAttention(feature_dim=mid_dim, sketch_dim=image_condition_dim, vit_local=vit_local, vit_global=vit_global,
                                             kernel_size=kernel_size,
                                             num_heads=num_heads, image_size=res, world_dims=world_dims,
                                             drop_out=dropout) if self.use_sketch_condition else our_Identity()
        self.mid_self_attn = nn.Sequential(
            normalization(mid_dim),
            activation_function(),
            AttentionBlock(mid_dim, num_heads=num_heads)
        ) if ds in attention_resolutions and with_attention else our_Identity()
        self.mid_block2 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            use_cross = (res == 4 or res == 8)
            self.ups.append(nn.ModuleList([
                ResnetBlock(world_dims, dim_out * 2, dim_in,
                            emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition),
                CrossAttention(feature_dim=dim_in, sketch_dim=image_condition_dim,
                               kernel_size=kernel_size, vit_local=vit_local, vit_global=vit_global,
                               num_heads=num_heads, image_size=res, world_dims=3,
                               drop_out=dropout) if use_cross and self.use_sketch_condition else our_Identity(),
                nn.Sequential(
                    normalization(dim_in),
                    activation_function(),
                    AttentionBlock(
                        dim_in, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Upsample(
                    dim_in, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds //= 2

        self.end = nn.Sequential(
            normalization(base_channels),
            activation_function()
        )

        self.out = conv_nd(world_dims, base_channels, 1, 3, padding=1)

    def forward(self, x, t, img_condition, text_condition, projection_matrix, x_self_cond=None, kernel_size=None):

        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x, x_self_cond), dim=1)

        if self.verbose:
            print("input size:")
            print(x.shape)

        x = self.input_emb(x)
        t = self.time_emb(self.time_pos_emb(t))
        if self.use_text_condition:
            text_condition = self.text_emb(text_condition)
        h = []

        for resnet, cross_attn, self_attn, downsample in self.downs:
            x = resnet(x, t, text_condition)
            if self.verbose:
                print(x.shape)
                if type(cross_attn) == CrossAttention:
                    print("cross attention at resolution: ",
                          cross_attn.image_size)
            x = cross_attn(x, img_condition,  projection_matrix, kernel_size)
            x = self_attn(x)
            if self.verbose:
                print(x.shape)
            h.append(x)
            x = downsample(x)
            if self.verbose:
                print(x.shape)

        if self.verbose:
            print("enter bottle neck")
        x = self.mid_block1(x, t, text_condition)
        if self.verbose:
            print(x.shape)

        x = self.mid_cross_attn(
            x, img_condition, projection_matrix, kernel_size)
        x = self.mid_self_attn(x)
        if self.verbose:
            print("cross attention at resolution: ",
                  self.mid_cross_attn.image_size)
            print(x.shape)
        x = self.mid_block2(x, t, text_condition)
        if self.verbose:
            print(x.shape)
            print("finish bottle neck")

        for resnet, cross_attn, self_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            if self.verbose:
                print(x.shape)
            x = resnet(x, t, text_condition)
            if self.verbose:
                print(x.shape)
            x = cross_attn(x, img_condition, projection_matrix, kernel_size)
            x = self_attn(x)
            if self.verbose:
                if type(cross_attn) == CrossAttention:
                    print("cross attention at resolution: ",
                          cross_attn.image_size)
                print(x.shape)
            x = upsample(x)
            if self.verbose:
                print(x.shape)

        x = self.end(x)
        if self.verbose:
            print(x.shape)

        return self.out(x)
