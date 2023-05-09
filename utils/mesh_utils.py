import numpy as np
from skimage.measure import marching_cubes
import trimesh
from utils.utils import scale_to_unit_sphere_in_place


def augment_sdf(sdf):
    index = np.random.randint(0, 9)
    random_number = np.random.rand()
    if np.random.rand() < 0.5:
        if random_number < 1/3:
            _sdf = np.concatenate(
                [sdf[index:, :, :], np.ones_like(sdf[0:index, :, :])], 0)
        elif random_number < 2/3:
            _sdf = np.concatenate(
                [sdf[:, index:, :], np.ones_like(sdf[:, 0:index, :])], 1)
        else:
            _sdf = np.concatenate(
                [sdf[:, :, index:], np.ones_like(sdf[:, :, 0:index])], 2)
    else:
        if random_number < 1/3:
            _sdf = np.concatenate(
                [np.ones_like(sdf[0:index, :, :]), sdf[0:128-index, :, :]], 0)
        elif random_number < 2/3:
            _sdf = np.concatenate(
                [np.ones_like(sdf[:, 0:index, :]), sdf[:, 0:128-index, :]], 1)
        else:
            _sdf = np.concatenate(
                [np.ones_like(sdf[:, :, 0:index]), sdf[:, :, 0:128-index]], 2)
    return _sdf


def process_sdf(volume, level=0, padding=False, spacing=None, offset=-1, normalize=False, clean=True, smooth=False):
    try:
        if padding:
            volume = np.pad(volume, 1, mode='constant', constant_values=1)
        if spacing is None:
            spacing = 2/volume.shape[-1]
        vertices, faces, normals, _ = marching_cubes(
            volume, level=level, spacing=(spacing, spacing, spacing))
        if offset is not None:
            vertices += offset
        if normalize:
            _mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=normals)
            scale_to_unit_sphere_in_place(_mesh)
        else:
            _mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=normals)
        if clean:
            components = _mesh.split(only_watertight=False)
            bbox = []
            for c in components:
                bbmin = c.vertices.min(0)
                bbmax = c.vertices.max(0)
                bbox.append((bbmax - bbmin).max())
            max_component = np.argmax(bbox)
            _mesh = components[max_component]
        if smooth:
            _mesh = trimesh.smoothing.filter_laplacian(_mesh, lamb=0.05)
        return _mesh
    except Exception as e:
        print(str(e))
        return None


def voxel2mesh(voxel, threshold=0.4, use_vertex_normal: bool = False):
    verts, faces, vertex_normals = _voxel2mesh(voxel, threshold)
    if use_vertex_normal:
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=vertex_normals)
    else:
        return trimesh.Trimesh(vertices=verts, faces=faces)


def _voxel2mesh(voxels, threshold=0.5):

    top_verts = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    top_faces = [[0, 1, 3], [1, 2, 3]]
    top_normals = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

    bottom_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    bottom_faces = [[1, 0, 3], [2, 1, 3]]
    bottom_normals = [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]

    left_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    left_faces = [[0, 1, 3], [2, 0, 3]]
    left_normals = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

    right_verts = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    right_faces = [[1, 0, 3], [0, 2, 3]]
    right_normals = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    front_verts = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    front_faces = [[1, 0, 3], [0, 2, 3]]
    front_normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

    back_verts = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    back_faces = [[0, 1, 3], [2, 0, 3]]
    back_normals = [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]]

    top_verts = np.array(top_verts)
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts)
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts)
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts)
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts)
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts)
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim+2, dim+2, dim+2))
    new_voxels[1:dim+1, 1:dim+1, 1:dim+1] = voxels
    voxels = new_voxels

    scale = 2/dim
    verts = []
    faces = []
    vertex_normals = []
    curr_vert = 0
    a, b, c = np.where(voxels > threshold)

    for i, j, k in zip(a, b, c):
        if voxels[i, j, k+1] < threshold:
            verts.extend(scale * (top_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert)
            vertex_normals.extend(top_normals)
            curr_vert += len(top_verts)

        if voxels[i, j, k-1] < threshold:
            verts.extend(
                scale * (bottom_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert)
            vertex_normals.extend(bottom_normals)
            curr_vert += len(bottom_verts)

        if voxels[i-1, j, k] < threshold:
            verts.extend(scale * (left_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert)
            vertex_normals.extend(left_normals)
            curr_vert += len(left_verts)

        if voxels[i+1, j, k] < threshold:
            verts.extend(scale * (right_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert)
            vertex_normals.extend(right_normals)
            curr_vert += len(right_verts)

        if voxels[i, j+1, k] < threshold:
            verts.extend(scale * (front_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert)
            vertex_normals.extend(front_normals)
            curr_vert += len(front_verts)

        if voxels[i, j-1, k] < threshold:
            verts.extend(scale * (back_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert)
            vertex_normals.extend(back_normals)
            curr_vert += len(back_verts)

    return np.array(verts) - 1, np.array(faces), np.array(vertex_normals)
