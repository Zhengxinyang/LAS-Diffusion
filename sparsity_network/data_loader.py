from pathlib import Path
from torch.utils import data
import torch
from ocnn.octree import Points, Octree
from ocnn.dataset import CollateBatch
import numpy as np
from utils.shapenet_utils import TSDF_VALUE, snc_category_to_synth_id_all, snc_category_to_synth_id_5, snc_category_to_synth_id_13
from utils.sparsity_utils import generate_template, voxel2indices
import os
import skimage.measure


class SparsityTransform:
    def __init__(self, depth: int, full_depth: int, size: int, sdf_clip_value: float,
                 noise_level: float):
        super().__init__()
        self.depth = depth
        self.full_depth = full_depth
        self.size = size
        self.sdf_clip_value = sdf_clip_value
        self.noise_level = noise_level

        self.template = generate_template(2)

    def __call__(self, sample, idx: int):
        points = self.preprocess(sample, idx)
        output = self.transform(points, idx)
        output['octree'] = self.points2octree(output['points'])
        return output

    def preprocess(self, sample, idx: int):
        thereshold = TSDF_VALUE * \
            (np.ones_like(sample) +
             (2 * np.random.rand(*sample.shape) - 1) * self.noise_level)

        occupancy_128 = np.where(abs(sample) < thereshold, np.ones_like(
            sample, dtype=np.float32), np.zeros_like(sample, dtype=np.float32))

        occupancy_64 = skimage.measure.block_reduce(
            occupancy_128, (2, 2, 2), np.max)
        indices_64 = voxel2indices(occupancy_64)

        indices_128 = np.repeat(indices_64, 2 ** 3, axis=0) * 2 + \
            np.repeat(self.template[None, :], len(
                indices_64), axis=0).reshape(-1, 3)

        features = np.clip(sample[indices_128[:, 0],
                                  indices_128[:, 1], indices_128[:, 2]], -self.sdf_clip_value, self.sdf_clip_value) / self.sdf_clip_value

        xyz = (2 * indices_128 + 1) / self.size - 1

        points = Points(torch.from_numpy(
            xyz), features=torch.from_numpy(features).to(torch.float32).unsqueeze(1))

        return points

    def transform(self, points: Points, idx: int):
        points.clip(min=-1, max=1)
        return {'points': points}

    def points2octree(self, points: Points):

        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree


class SDF_sparsity_Dataset(data.Dataset):
    def __init__(self, folder: str, data_class: str, size: int, sdf_clip_value: float,
                 noise_level: float, split_dataset: bool = False, data_augmentation=False):
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
                filelist = os.path.join(folder, f"train_{label}.txt")
                with open(filelist) as fid:
                    lines = fid.readlines()
                for i in range(len(lines)):
                    lines[i] = os.path.join(
                        folder, label, lines[i].replace(".mat\n", ".npy"))
                self.sdf_paths.extend(lines)
        else:
            for _data_class in _data_classes:
                _label = snc_category_to_synth_id_all[_data_class]
                _path = os.path.join(folder, _label)
                self.sdf_paths.extend(
                    [p for p in Path(f'{_path}').glob('**/*.npy')])
        if data_augmentation:
            assert data_class == "class_5"
            mix_label = "00000002"
            _path = os.path.join(folder, mix_label)
            self.sdf_paths.extend(
                [p for p in Path(f'{_path}').glob('**/*.npy')])

        depth = int(np.log2(size))
        full_depth = 1

        self.transform = SparsityTransform(depth=depth, full_depth=full_depth, size=size, sdf_clip_value=sdf_clip_value,
                                           noise_level=noise_level)

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):

        sample = np.load(self.sdf_paths[index])
        output = self.transform(sample, index)
        return output


def get_shapenet_sparsity_dataset(folder, data_class: str, size: int, sdf_clip_value: float,
                                  noise_level: float, split_dataset: bool = False, data_augmentation=False):
    collate_batch = CollateBatch(merge_points=True)

    dataset = SDF_sparsity_Dataset(
        folder=folder, data_class=data_class, size=size, sdf_clip_value=sdf_clip_value,
        noise_level=noise_level, split_dataset=split_dataset, data_augmentation=data_augmentation)

    return dataset, collate_batch


class SparsityTransform_for_forward:

    def __init__(self, depth: int, full_depth: int, size: int, base_size: int):
        super().__init__()
        self.depth = depth
        self.full_depth = full_depth
        self.size = size
        self.down_factor = size // base_size
        self.template = generate_template(self.down_factor)

    def __call__(self, sample, idx: int):
        points = self.preprocess(sample, idx)
        output = self.transform(points, idx)
        output['octree'] = self.points2octree(output['points'])
        return output

    def prepare(self, indices):
        high_indices = np.repeat(indices, self.down_factor ** 3, axis=0) * self.down_factor + \
            np.repeat(self.template[None, :], len(
                indices), axis=0).reshape(-1, 3)

        return high_indices

    def preprocess(self, sample, idx: int):

        sample[sample > 0] = 1
        sample[sample < 0] = 0
        indices = self.prepare(voxel2indices(sample))
        xyz = (2 * indices + 1) / self.size - 1
        features = np.ones((len(indices), 1))
        points = Points(torch.from_numpy(
            xyz), features=torch.from_numpy(features).to(torch.float32))

        return points

    def transform(self, points: Points, idx: int):
        points.clip(min=-1, max=1)
        return {'points': points}

    def points2octree(self, points: Points):

        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree


class SDF_sparsity_Dataset_for_forward(data.Dataset):
    def __init__(self, folder: str, size: int, base_size: int, sort_npy: bool = True, start_index: int = 0, end_index: int = 10000):
        super().__init__()
        # print(folder)
        sdf_paths = [p for p in Path(f'{folder}').glob('**/*.npy')]
        if sort_npy:
            sdf_paths.sort(key=self.sort_func)
        self.paths = sdf_paths[start_index:end_index]
        depth = int(np.log2(size))
        full_depth = 1
        self.transform = SparsityTransform_for_forward(
            depth=depth, full_depth=full_depth, size=size, base_size=base_size)

    def sort_func(self, item):
        return int(str(item).split("/")[-1].split(".")[0])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample = np.load(self.paths[index])
        output = self.transform(sample, index)
        return output


def get_shapenet_sparsity_dataset_for_forward(folder, size: int, base_size: int, sort_npy: bool = True, start_index: int = 0, end_index: int = 10000):
    collate_batch = CollateBatch(merge_points=True)
    dataset = SDF_sparsity_Dataset_for_forward(
        folder=folder, size=size, base_size=base_size, sort_npy=sort_npy, start_index=start_index, end_index=end_index)
    return dataset, collate_batch
