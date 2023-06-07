import fire
import os
from sparsity_network.sparsity_trainer import Sparsity_DiffusionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from utils.shapenet_utils import SDF_CLIP_VALUE
from pytorch_lightning import loggers as pl_loggers
from utils.utils import ensure_directory, exists, run, get_tensorboard_dir, find_best_epoch
from utils.shapenet_utils import snc_category_to_synth_id_all

def train_from_folder(
    sdf_folder: str = "/home/D/dataset/",
    data_class="chair",
    results_folder: str = './results',
    name: str = "sparse_debug",
    base_size: int = 64,
    base_channels: int = 32,
    lr: float = 1e-4,
    batch_size: int = 4,
    ema_rate: float = 0.999,
    verbose: bool = False,
    noise_schedule: str = "linear",
    save_last: bool = True,
    training_epoch: int = 200,
    in_azure: bool = False,
    new: bool = True,
    continue_training: bool = False,
    debug: bool = False,
    seed: int = 777,
    save_every_epoch: int = 20,
    gradient_clip_val: float = 1.,
    noise_level: float = 0.3,
    split_dataset: bool = False,
    data_augmentation: bool = False,
):
    sdf_clip_value = SDF_CLIP_VALUE
    
    data_classes = list(snc_category_to_synth_id_all.keys())
    data_classes.extend(["debug", "class_5", "class_13", "all"])
    
    assert data_class in data_classes

    results_folder = results_folder + "/" + name
    ensure_directory(results_folder)
    if continue_training:
        pass
    elif new:
        run(f"rm -rf {results_folder}/*")

    model_args = dict(
        results_folder=results_folder,
        dataset_folder=sdf_folder,
        data_class=data_class,
        batch_size=batch_size,
        lr=lr,
        base_size=base_size,
        base_channels=base_channels,
        ema_rate=ema_rate,
        upfactor=2,
        verbose=verbose,
        training_epoch=training_epoch,
        gradient_clip_val=gradient_clip_val,
        sdf_clip_value=sdf_clip_value,
        noise_schedule=noise_schedule,
        noise_level=noise_level,
        split_dataset=split_dataset,
        data_augmentation=data_augmentation
    )
    seed_everything(seed)

    model = Sparsity_DiffusionModel(**model_args)

    if in_azure:
        try:
            log_dir = get_tensorboard_dir()
        except Exception as e:
            log_dir = results_folder
    else:
        log_dir = results_folder

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=None,
        name='logs',
        default_hp_metric=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="current_epoch",
        dirpath=results_folder,
        filename="{epoch:02d}",
        save_top_k=10,
        save_last=save_last,
        every_n_epochs=save_every_epoch,
        mode="max",
    )
    
    if save_last:
            last_ckpt = "last.ckpt"
    else:
        last_epoch = find_best_epoch(results_folder)
        if exists(last_epoch):
            last_ckpt = f"epoch={last_epoch:02d}.ckpt"
        else:
            last_ckpt = "last.ckpt"
    
    find_unused_parameters = False

    if in_azure:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPPlugin(
                              find_unused_parameters=find_unused_parameters),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=1,
                          callbacks=[checkpoint_callback])
    else:
        if debug:
            trainer = Trainer(devices=-1,
                              accelerator="gpu",
                              strategy=DDPPlugin(
                                  find_unused_parameters=find_unused_parameters),
                              logger=tb_logger,
                              max_epochs=training_epoch,
                              log_every_n_steps=1,
                              callbacks=[checkpoint_callback],
                              overfit_batches=0.01)

        else:
            trainer = Trainer(devices=-1,
                              accelerator="gpu",
                              strategy=DDPPlugin(
                                  find_unused_parameters=find_unused_parameters),
                              logger=tb_logger,
                              max_epochs=training_epoch,
                              log_every_n_steps=1,
                              callbacks=[checkpoint_callback])
          
    if continue_training and os.path.exists(results_folder + f'/{last_ckpt}'):
        trainer.fit(model, ckpt_path=results_folder + f'/{last_ckpt}')
    else:
        trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(train_from_folder)
