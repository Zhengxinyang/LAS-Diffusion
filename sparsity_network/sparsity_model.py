import torch
from torch import nn
from tqdm import tqdm
from network.model_utils import *
from sparsity_network.unet import Sparsity_UNetModel
from random import random
from einops import  repeat
from torch.special import expm1
from ocnn.octree import Octree


class SDFDiffusion(nn.Module):
    def __init__(
            self,
            base_size: int = 32,
            upfactor: int = 2,
            base_channels: int = 128,
            verbose: bool = False,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
            sdf_clip_value: float = 0.05,
    ):
        super().__init__()
        self.base_size = base_size
        self.upfactor = upfactor
        self.eps = eps
        self.verbose = verbose
        self.sdf_clip_value = sdf_clip_value
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.denoise_fn = Sparsity_UNetModel(
            base_channels=base_channels, base_size=self.base_size, upfactor=self.upfactor,
            condition_classes=1, verbose=verbose)

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    def training_loss(self, x_start, octree: Octree, *args, **kwargs):
        batch_size = octree.batch_size
        times = torch.zeros(
            (batch_size,), device=self.device).float().uniform_(0, 1)

        noise_level = self.log_snr(times)
        x_t = []
        batch_id = octree.batch_id(octree.depth, True)
        for i in range(batch_size):
            feature = x_start[batch_id == i]
            alpha, sigma = log_snr_to_alpha_sigma(noise_level[i])
            x_t.append(feature * alpha + sigma * torch.randn_like(feature))

        x_t = torch.cat(x_t, dim=0)

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(x_t, noise_level, octree)
                self_cond.detach_()

        pred = self.denoise_fn(x_t, noise_level, octree, self_cond)
        sdf_loss = (pred - x_start) ** 2
        loss = torch.zeros((batch_size,), device=self.device)
        for i in range(batch_size):
            loss[i] = sdf_loss[batch_id == i].mean()

        return loss


    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, noise_feature, octree, steps, truncated_index: float = 0, verbose:bool = False):

        time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)
        x_start = None
        if verbose: 
            loops =  tqdm(time_pairs, desc='sampling loop time step')
        else:
            loops = time_pairs
        for time, time_next in loops:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            x_start = self.denoise_fn(
                noise_feature, log_snr, octree, x_start)

            x_start.clamp_(-1, 1)
            
            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (noise_feature * (1 - c) / alpha + c * x_start)

            variance = (sigma_next ** 2) * c
            if time_next > truncated_index:
                noise_feature = mean + \
                    torch.sqrt(variance) * torch.randn_like(noise_feature)
            else:
                noise_feature = mean

        return noise_feature

    @torch.no_grad()
    def ddim_sample(self, noise_feature, octree, steps, truncated_index: float = 0, verbose:bool = False):

        time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)
        x_start = None
        if verbose: 
            loops =  tqdm(time_pairs, desc='sampling loop time step')
        else:
            loops = time_pairs
        for time, time_next in loops:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            x_start  = self.denoise_fn(
                noise_feature, log_snr, octree, x_start)

            x_start.clamp_(-1, 1)
            pred_noise = (noise_feature - alpha * x_start) / \
                sigma.clamp(min=1e-8)

            noise_feature = x_start * alpha_next + pred_noise * sigma_next

        return noise_feature

    @torch.no_grad()
    def sample(self, data, octree, use_ddim: bool = False, steps: int = 1, truncated_index: float = 0.0, verbose:bool = False):
        noise_feature = torch.randn_like(data)
        sample_fn = self.ddpm_sample if not use_ddim else self.ddim_sample
        return sample_fn(noise_feature, octree, steps, truncated_index, verbose)
