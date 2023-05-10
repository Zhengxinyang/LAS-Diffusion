import torch
from pathlib import Path
import numpy as np
import os
from utils.shapenet_utils import snc_category_to_synth_id_all, snc_synth_id_to_category_all, TSDF_VALUE, snc_category_to_synth_id_5, snc_category_to_synth_id_13
import skimage.measure
from utils.sketch_utils import Projection_List, Projection_List_zero
from random import random
from utils.utils import SKETCH_PER_VIEW, SKETCH_NUMBER
from utils.sketch_utils import create_random_pose, get_P_from_transform_matrix


class occupancy_field_Dataset(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str, sketch_folder: str,
                 data_class: str = "all", data_augmentation=False,
                 size: int = 64,
                 feature_drop_out: float = 0.1,
                 split_dataset: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 elevation_zero: bool = False,
                 detail_view: bool = False,
                 use_text_condition: bool = False,
                 use_sketch_condition: bool = True
                 ):
        super().__init__()
        if data_class == "class_5":
            _data_classes = snc_category_to_synth_id_5.keys()
        elif data_class == "class_13":
            _data_classes = snc_category_to_synth_id_13.keys()
        elif data_class == "all":
            _data_classes = snc_category_to_synth_id_all.keys()
        else:
            _data_classes = [data_class]

        self.sdf_paths = []
        if split_dataset:
            assert data_class != "all"
            for _data_class in _data_classes:
                label = snc_category_to_synth_id_all[_data_class]
                filelist = os.path.join(sdf_folder, f"train_{label}.txt")
                with open(filelist) as fid:
                    lines = fid.readlines()
                for i in range(len(lines)):
                    lines[i] = os.path.join(
                        sdf_folder, label, lines[i].replace(".mat\n", ".npy"))
                self.sdf_paths.extend(lines)
        else:
            for _data_class in _data_classes:
                _label = snc_category_to_synth_id_all[_data_class]
                _path = os.path.join(sdf_folder, _label)
                self.sdf_paths.extend(
                    [p for p in Path(f'{_path}').glob('**/*.npy')])

        if data_augmentation:
            if detail_view:
                assert data_class == "all"
                mix_label = "00000001"
            else:
                mix_label = "00000002"
            _path = os.path.join(sdf_folder, mix_label)
            self.sdf_paths.extend(
                [p for p in Path(f'{_path}').glob('**/*.npy')])

        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        self.vit_global = vit_global
        self.vit_local = vit_local
        self.multiplier = 128 // size

        self.feature_drop_out = feature_drop_out
        self.sketch_folder = sketch_folder

        self.feature_folder = os.path.join(sketch_folder, "feature")

        if self.use_sketch_condition:

            assert self.vit_global or self.vit_local
            self.white_image_feature = np.load(os.path.join(
                self.feature_folder, "white_image_feature.npy"))
            self.detail_view = detail_view
            if not detail_view:
                if elevation_zero:
                    self.projection_list = Projection_List_zero
                else:
                    self.projection_list = Projection_List
            else:
                self.pm_folder = os.path.join(
                    sketch_folder, "projection_matrix")

        if self.use_text_condition:
            from utils.condition_data import text_features
            self.text_features = {}
            for _data_class in _data_classes:
                self.text_features[snc_category_to_synth_id_all[_data_class]
                                   ] = text_features[_data_class].astype(np.float32)
            self.an_object_feature = np.load(os.path.join(
                self.feature_folder, "an_object.npy")).astype(np.float32)

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):
        sdf_path = self.sdf_paths[index]
        sdf = np.load(sdf_path)

        model_name = str(sdf_path).split(".")[0]
        data_id = model_name.split("/")[-2]
        model_name = os.path.join(model_name.split(
            "/")[-2], model_name.split("/")[-1])

        res = {}
        if self.use_sketch_condition:
            if not self.detail_view:
                sketch_view_index = np.random.randint(0, 5 * SKETCH_PER_VIEW)
                try:
                    image_feature = np.load(os.path.join(
                        self.feature_folder, model_name, f"{sketch_view_index:02d}.npy"))
                except Exception as e:
                    print(str(e))
                    image_feature = self.white_image_feature
                pm = self.projection_list[sketch_view_index//SKETCH_PER_VIEW]
            else:
                sketch_view_index = np.random.randint(0, SKETCH_NUMBER)
                try:
                    image_feature = np.load(os.path.join(self.feature_folder, model_name))[
                        sketch_view_index]
                    pm = np.load(os.path.join(self.pm_folder, model_name))[
                        sketch_view_index]
                except Exception as e:
                    print(str(e))
                    image_feature = self.white_image_feature
                    pm = get_P_from_transform_matrix(create_random_pose())
                    projection_matrix = np.expand_dims(pm, 0)

            if random() < self.feature_drop_out:
                image_feature = self.white_image_feature

            if self.vit_global and not self.vit_local:
                image_feature = image_feature[0:1]
            elif self.vit_local and not self.vit_global:
                image_feature = image_feature[1:]

            res["image_feature"] = image_feature

            projection_matrix = np.expand_dims(pm, 0)
            res["projection_matrix"] = projection_matrix

        if self.use_text_condition:
            try:
                if data_id in snc_synth_id_to_category_all.keys():
                    text_feature = self.text_features[data_id]
                else:
                    text_feature = np.zeros_like(self.an_object_feature)
            except Exception as e:
                print(str(e))
                text_feature = self.an_object_feature
            if random() < self.feature_drop_out:
                text_feature = self.an_object_feature
            res["text_feature"] = text_feature
            
            if self.vit_global and not self.vit_local:
                # ablation study 1 in the paper
                sketch_view_index = np.random.randint(0, 5 * SKETCH_PER_VIEW)
                try:
                    image_feature = np.load(os.path.join(
                        self.feature_folder, model_name, f"{sketch_view_index:02d}.npy"))
                except Exception as e:
                    image_feature = self.white_image_feature
                    
                if random() < self.feature_drop_out:
                    image_feature = self.white_image_feature
                res["text_feature"] = image_feature[0]

        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (self.multiplier, self.multiplier, self.multiplier), np.max)
        res["occupancy"] = np.expand_dims(2 * occupancy_low - 1, axis=0)

        return res
