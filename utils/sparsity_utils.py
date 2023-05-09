from ocnn.octree import Points, Octree, key2xyz
import numpy as np
import torch
from ocnn.nn.octree_pad import octree_depad
from scipy.ndimage import binary_fill_holes


def generate_template(factor):
    template = np.zeros((factor ** 3, 3), dtype=np.int16)
    for i in range(factor):
        for j in range(factor):
            for k in range(factor):
                template[i*factor**2 + j*factor + k, 0] = i
                template[i*factor**2 + j*factor + k, 1] = j
                template[i*factor**2 + j*factor + k, 2] = k

    return template


def voxel2indices(voxel, depadding=False):
    assert type(voxel) == np.ndarray
    indices = np.stack(np.nonzero(voxel)).transpose()
    if depadding:
        template = generate_template(2)
        low_indices = np.unique(indices//2, axis=0)
        indices = np.repeat(low_indices, 8, axis=0) * 2 + \
            np.repeat(template[None, :], len(
                low_indices), axis=0).reshape(-1, 3)
    return indices


def voxel2octree(voxel, upfactor=2, device="cuda:0"):

    return indices2octree(indices=voxel2indices(voxel),
                          res=voxel.shape[-1],
                          upfactor=upfactor,
                          device=device)


def indices2octree(indices, res=32, upfactor=2, device="cuda:0"):
    template = generate_template(upfactor)

    indices = np.repeat(indices, upfactor ** 3, axis=0) * upfactor + \
        np.repeat(template[None, :], len(indices), axis=0).reshape(-1, 3)
    res *= upfactor

    xyz = (2 * indices + 1) / res - 1
    features = torch.ones((len(xyz), 1))

    points = Points(torch.from_numpy(xyz), features=features).to(device=device)
    points.clip(min=-1, max=1)

    depth = int(np.log2(res))
    octree = Octree(depth, 1, device=device)
    octree.build_octree(points)
    octree.construct_all_neigh()

    return octree


def octree2sdfgrid(data: torch.Tensor, octree: Octree, depth: int, scale: float,
                   nempty: bool = False):
    key = octree.keys[depth]
    if nempty:
        key = octree_depad(key, octree, depth)
    x, y, z, b = key2xyz(key, depth)
    num = 1 << depth
    batch_size = octree.batch_size
    size = (batch_size, num, num, num)
    vox = torch.ones(size, dtype=data.dtype, device=data.device) * scale
    mask = torch.zeros(size, dtype=torch.long, device=data.device)

    vox[b, x, y, z] = data[:, 0] * scale
    mask[b, x, y, z] = torch.ones_like(data[:, 0]).to(torch.long)
    vox = vox.cpu().numpy()
    mask = mask.cpu().numpy()

    for _batch in range(batch_size):
        _mask = binary_fill_holes(mask[_batch]).astype(np.int32) - mask[_batch]
        x, y, z = np.nonzero(_mask)
        vox[_batch, x, y, z] = -scale

    return vox


def load_input(npy_path):
    low_res_voxel = np.load(npy_path)
    low_res_voxel[low_res_voxel > 0] = 1
    low_res_voxel[low_res_voxel < 0] = 0
    return low_res_voxel
