import fire
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import timm
from pathlib import Path
from PIL import Image
import copy
from utils.sketch_utils import _transform,get_sketch
from utils.utils import VIT_MODEL, set_requires_grad, SKETCH_PER_VIEW, ensure_directory
from utils.shapenet_utils import snc_synth_id_to_category_all, snc_synth_id_to_category_5, snc_category_to_synth_id_all
from utils.mesh_utils import process_sdf, augment_sdf
from utils.render.render import random_render_mesh

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import DDPPlugin



class our_Dataset(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str,  out_dir="", data_class: str = "all",
                 image_resolution: int = 224, start_index: int = 0, end_index: int = 10000):
        super().__init__()
        self.preprocess = _transform(image_resolution)
        self.out_dir = out_dir
        self.mixed_data = False
        if data_class != "mixed":
            self.sdf_folder = os.path.join(
                sdf_folder, snc_category_to_synth_id_all[data_class])
            self.sdf_paths = [p for p in Path(
                f'{self.sdf_folder}').glob('**/*.npy')]
            ensure_directory(os.path.join(
                out_dir, "edge", snc_category_to_synth_id_all[data_class]))
            ensure_directory(os.path.join(out_dir, "feature",
                             snc_category_to_synth_id_all[data_class]))
            ensure_directory(os.path.join(
                out_dir, "angles", snc_category_to_synth_id_all[data_class]))
            self.sdf_paths = self.sdf_paths[start_index:end_index]
        elif data_class == "mixed":
            self.mixed_data = True
            sdf_paths = []
            self.mixed_label = "00000002"
            for _label in list(snc_synth_id_to_category_all.keys()):
                _path = os.path.join(sdf_folder, _label)
                sdf_paths.extend(
                    [p for p in Path(f'{_path}').glob('**/*.npy')])

            self.all_sdf_paths = copy.deepcopy(sdf_paths)
            self.sdf_paths = []
            labels = list(snc_synth_id_to_category_all.keys())
            for _label in list(snc_synth_id_to_category_5.keys()):
                labels.remove(_label)
            for _label in labels:
                _path = os.path.join(sdf_folder, _label)
                self.sdf_paths.extend(
                    [p for p in Path(f'{_path}').glob('**/*.npy')])

            self.sdf_folder = sdf_folder
            self.sdf_paths = self.sdf_paths[start_index:end_index]

            ensure_directory(os.path.join(self.sdf_folder, self.mixed_label))
            ensure_directory(os.path.join(out_dir, "edge", self.mixed_label))
            ensure_directory(os.path.join(
                out_dir, "feature", self.mixed_label))
            ensure_directory(os.path.join(
                out_dir, "angles", self.mixed_label))
        else:
            raise NotImplementedError


    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):
        sdf_path = self.sdf_paths[index]

        model_name = str(sdf_path).split(".")[0]

        if self.mixed_data:
            model_name = os.path.join(
                self.mixed_label, model_name.split("/")[-2] + "_" + model_name.split("/")[-1])
        else:
            model_name = os.path.join(model_name.split(
                "/")[-2], model_name.split("/")[-1])

        rotation_angles = []
        elevation_angles = []
        images = []

        if self.mixed_data:
            sdf1 = np.load(sdf_path)
            idx = np.random.randint(0, len(self.all_sdf_paths))
            sdf2 = np.load(self.all_sdf_paths[idx])
            sdf_path2 = self.all_sdf_paths[idx]
            new_model_name = str(sdf_path2).split(".")[0]
            new_model_name = new_model_name.split(
                "/")[-2] + "_" + new_model_name.split("/")[-1]
            model_name = model_name + "_" + new_model_name

            sdf = np.minimum(augment_sdf(sdf1), augment_sdf(sdf2))
            np.save(os.path.join(self.sdf_folder, model_name + ".npy"), sdf)
        else:
            sdf = np.load(sdf_path)

        ensure_directory(os.path.join(self.out_dir, "edge", model_name))
        mesh = process_sdf(sdf, clean=False)

        for i in range(5):
            for j in range(SKETCH_PER_VIEW):
                rotation = i*45 + 22.5*(2*np.random.rand() - 1)
                elevation = 20 + 5*(2*np.random.rand() - 1)
                rotation_angles.append(rotation)
                elevation_angles.append(elevation)
                image = random_render_mesh(
                    mesh, return_pose=False, rotation=rotation, elevation=elevation)
                edges = get_sketch(image)
                _edges = Image.fromarray(edges, mode="L")
                _edges.save(os.path.join(self.out_dir, "edge",
                            model_name, f"edge_{i}_{j}.png"))
                images.append(self.preprocess(Image.fromarray(edges)))

        rotation_angles = np.stack(rotation_angles, 0)
        elevation_angles = np.stack(elevation_angles, 0)
        np.save(os.path.join(self.out_dir, "angles",
                model_name + "_rotation.npy"), rotation_angles)
        np.save(os.path.join(self.out_dir, "angles",
                model_name + "_elevation.npy"), elevation_angles)

        images = torch.stack(images, 0)

        return {'images': images,
                'model_name': model_name,
                'foo': np.random.rand(10).astype(np.float32)}


class GenerationDataModel(LightningModule):
    def __init__(
        self,
        sdf_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        start_index: int = 0,
        end_index: int = 100000
    ):
        super().__init__()
        self.num_workers = os.cpu_count()
        self.sdf_folder = sdf_folder
        self.data_class = data_class
        self.out_dir = results_folder
        self.feature_extractor = timm.create_model(VIT_MODEL, pretrained=True)
        self.model = nn.Linear(10, 1)
        set_requires_grad(self.feature_extractor, False)
        self.start_index = start_index
        self.end_index = end_index

    def train_dataloader(self):
        _dataset = our_Dataset(sdf_folder=self.sdf_folder,
                               out_dir=self.out_dir, data_class=self.data_class,
                               start_index=self.start_index, end_index=self.end_index)
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=1, shuffle=True, pin_memory=True, drop_last=False)
        return dataloader

    def training_step(self, batch, index):
        foo_data = batch['foo']
        images = batch['images']
        model_name = batch['model_name']
        loss = self.model(foo_data).mean()

        with torch.no_grad():
            image_features = self.feature_extractor.forward_features(
                images[0]).squeeze().cpu().numpy()

        out_root_dir = os.path.join(self.out_dir, "feature", model_name[0])

        ensure_directory(out_root_dir)
        for i in range(5*SKETCH_PER_VIEW):
            np.save(os.path.join(out_root_dir,
                    f"{i:02d}.npy"), image_features[i])
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-4)
        return None


def train_from_folder(
    sdf_folder: str = "/home/D/dataset/shapenet_sdf",
    results_folder: str = "/home/D/dataset/shapenet_edge_our_new_0508",
    data_class="chair",
    start_index: int = 0,
    end_index: int = 100000
):
    if data_class == "mixed":
        seed_everything(start_index)
    model_args = dict(
        results_folder=results_folder,
        sdf_folder=sdf_folder,
        data_class=data_class,
        start_index=start_index,
        end_index=end_index,
    )
    model = GenerationDataModel(**model_args)

    trainer = Trainer(devices=-1,
                      accelerator="gpu",
                      strategy=DDPPlugin(
                          find_unused_parameters=False),
                      max_epochs=1,
                      log_every_n_steps=1,)

    trainer.fit(model)


if __name__ == '__main__':

    fire.Fire(train_from_folder)
