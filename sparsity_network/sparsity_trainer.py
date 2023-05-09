import copy
from utils.utils import set_requires_grad
from utils.mesh_utils import process_sdf
from torch.utils.data import DataLoader
from network.model_utils import EMA
from sparsity_network.data_loader import get_shapenet_sparsity_dataset, get_shapenet_sparsity_dataset_for_forward
from pathlib import Path
from torch.optim import AdamW
from utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from sparsity_network.sparsity_model import SDFDiffusion
import torch.nn as nn
from ocnn.octree import Octree
import os
import torch
import numpy as np
import ocnn
from utils.sparsity_utils import voxel2octree, octree2sdfgrid


class Sparsity_DiffusionModel(LightningModule):
    def __init__(
        self,
        dataset_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        base_size: int = 32,
        upfactor: int = 2,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        ema_rate: float = 0.9999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        split_dataset: bool = False,
        data_augmentation: bool = False,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        sdf_clip_value: float = 0.015,
        noise_schedule="linear",
        noise_level: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.results_folder = Path(results_folder)
        self.noise_level = noise_level
        self.model = SDFDiffusion(base_size=base_size,
                                  upfactor=upfactor,
                                  base_channels=base_channels,
                                  verbose=verbose,
                                  noise_schedule=noise_schedule,
                                  sdf_clip_value=sdf_clip_value)

        self.octree_feature = ocnn.modules.InputFeature('F', True)
        self.batch_size = batch_size
        self.lr = lr
        self.base_size = base_size
        self.upfactor = upfactor
        self.sdf_clip_value = sdf_clip_value
        self.image_size = base_size * upfactor
        self.dataset_folder = dataset_folder
        self.data_class = data_class
        self.save_every_epoch = save_every_epoch

        self.split_dataset = split_dataset
        self.data_augmentation = data_augmentation

        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.denoise_fn.parameters(), lr=self.lr)
        return [optimizer]

    def train_dataloader(self):
        dataset, collate_fn = get_shapenet_sparsity_dataset(self.dataset_folder, self.data_class,
                                                            size=self.image_size, sdf_clip_value=self.sdf_clip_value,
                                                            noise_level=self.noise_level,
                                                            split_dataset=self.split_dataset,
                                                            data_augmentation=self.data_augmentation)

        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            num_workers=os.cpu_count()//2,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False)

        return dataloader

    def get_input_feature(self, octree):
        data = self.octree_feature(octree)
        return data

    def training_step(self, batch, batch_idx):

        octree = batch['octree']
        data = self.get_input_feature(octree)

        loss = self.model.training_loss(data, octree)

        loss = loss.mean()
        self.log("loss", loss.clone().detach().item(),
                 prog_bar=True, batch_size=self.batch_size)
        opt = self.optimizers()

        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()

        self.update_EMA()

    def on_train_epoch_end(self):
        self.log("current_epoch", self.current_epoch)

        return super().on_train_epoch_end()

    @torch.no_grad()
    def generate_results_from_single_voxel(self, low_res_voxel, ema=True, steps=1000,
                                           use_ddim: bool = False, truncated_index: float = 0., verbose: bool = False):

        generator = self.ema_model if ema else self.model
        assert low_res_voxel.shape[-1] == self.base_size
        octree = voxel2octree(low_res_voxel, upfactor=self.upfactor,
                              device=self.device)

        data = self.octree_feature(octree)

        data = generator.sample(data, octree, use_ddim=use_ddim, steps=steps,
                                truncated_index=truncated_index, verbose=verbose)

        return octree2sdfgrid(data, octree=octree, depth=octree.depth, scale=self.sdf_clip_value, nempty=True)

    @torch.no_grad()
    def generate_results_from_folder(self, folder, save_path, ema=True, batch_size=8,
                                     steps=1000, use_ddim: bool = False, truncated_index: float = 0., sort_npy: bool = True, level: float = 0.0,
                                     save_npy: bool = True, save_mesh: bool = True, start_index: int = 0, end_index: int = 10000, verbose: bool = False):

        generator = self.ema_model if ema else self.model

        dataset, collate_fn = get_shapenet_sparsity_dataset_for_forward(
            folder, size=self.image_size, base_size=self.base_size, sort_npy=sort_npy, start_index=start_index, end_index=end_index)

        assert len(dataset) > 0
        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            num_workers=os.cpu_count()//2,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        index = start_index
        for _, batch in enumerate(dataloader, 0):
            octree: Octree = batch['octree'].to(self.device)

            data = self.octree_feature(octree)
            data = generator.sample(
                data, octree, use_ddim=use_ddim, steps=steps, truncated_index=truncated_index, verbose=verbose)

            res = octree2sdfgrid(data, octree=octree, depth=octree.depth,
                                 scale=self.sdf_clip_value, nempty=True)
            for i in range(octree.batch_size):
                field = res[i]
                try:
                    if save_npy:
                        np.save(os.path.join(save_path, f"{index}.npy"), field)
                    if save_mesh:
                        mesh = process_sdf(field, level=level, normalize=True)
                        mesh.export(os.path.join(save_path, f"{index}.obj"))
                except Exception as e:
                    print(str(e))
                index += 1
