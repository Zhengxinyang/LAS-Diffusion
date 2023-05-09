import copy
from utils.utils import set_requires_grad
from torch.utils.data import DataLoader
from network.model_utils import EMA
from network.data_loader import occupancy_field_Dataset
from pathlib import Path
from torch.optim import AdamW,Adam
from utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from network.model import OccupancyDiffusion
import torch.nn as nn
import os
import random


class DiffusionModel(LightningModule):
    def __init__(
        self,
        sdf_folder: str = "",
        sketch_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        use_sketch_condition: bool = False,
        use_text_condition: bool = True,
        noise_schedule: str = "linear",
        debug: bool = False,
        image_feature_drop_out: float = 0.1,
        view_information_ratio: float = 0.5,
        data_augmentation: bool = False,
        kernel_size: float = 2.0,
        vit_global: bool = False,
        vit_local: bool = True,
        split_dataset: bool = False,
        elevation_zero: bool = False,
        detail_view: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        self.model = OccupancyDiffusion(image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        kernel_size=kernel_size,
                                        dropout=dropout,
                                        use_sketch_condition=use_sketch_condition,
                                        use_text_condition=use_text_condition,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        vit_global=vit_global,
                                        vit_local=vit_local,
                                        verbose=verbose)

        self.view_information_ratio = view_information_ratio
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.sdf_folder = sdf_folder
        self.sketch_folder = sketch_folder
        self.data_class = data_class
        self.data_augmentation = data_augmentation
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.image_feature_drop_out = image_feature_drop_out

        self.vit_global = vit_global
        self.vit_local = vit_local
        self.split_dataset = split_dataset
        self.elevation_zero = elevation_zero
        self.detail_view = detail_view
        self.optimizier = optimizier
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)
        if debug:
            self.num_workers = 1
        else:
            self.num_workers = os.cpu_count()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        return [optimizer]

    def train_dataloader(self):
        _dataset = occupancy_field_Dataset(sdf_folder=self.sdf_folder,
                                           sketch_folder=self.sketch_folder,
                                           data_class=self.data_class,
                                           size=self.image_size,
                                           data_augmentation=self.data_augmentation,
                                           feature_drop_out=self.image_feature_drop_out,
                                           vit_global=self.vit_global,
                                           vit_local=self.vit_local,
                                           split_dataset=self.split_dataset,
                                           elevation_zero=self.elevation_zero,
                                           detail_view=self.detail_view,
                                           use_sketch_condition=self.use_sketch_condition,
                                           use_text_condition=self.use_text_condition
                                           )
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]
        if self.use_sketch_condition:
            image_features = batch["image_feature"]
            if random.random() < self.view_information_ratio:
                projection_matrix = batch["projection_matrix"]
            else:
                projection_matrix = None
            kernel_size = self.kernel_size

        else:
            image_features = None
            projection_matrix = None
            kernel_size = None

        if self.use_text_condition:
            text_feature = batch["text_feature"]
        else:
            text_feature = None

        loss = self.model.training_loss(
            occupancy, image_features, text_feature, projection_matrix, kernel_size=kernel_size).mean()

        self.log("loss", loss.clone().detach().item(), prog_bar=True)

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
