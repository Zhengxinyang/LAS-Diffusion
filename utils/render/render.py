import numpy as np
from .render_utils import Render, create_pose, create_shapenet_chair_camera_pose
from ..sketch_utils import create_random_pose
import matplotlib
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
matplotlib.use("Agg")


FrontVector = (np.array([[0.52573, 0.38197, 0.85065],
                         [-0.20081, 0.61803, 0.85065],
                         [-0.64984, 0.00000, 0.85065],
                         [-0.20081, -0.61803,  0.85065],
                         [0.52573, -0.38197, 0.85065],
                         [0.85065, -0.61803, 0.20081],
                         [1.0515,  0.00000, -0.20081],
                         [0.85065, 0.61803, 0.20081],
                         [0.32492, 1.00000, -0.20081],
                         [-0.32492, 1.00000,  0.20081],
                         [-0.85065, 0.61803, -0.20081],
                         [-1.0515, 0.00000,  0.20081],
                         [-0.85065, -0.61803, -0.20081],
                         [-0.32492, -1.00000,  0.20081],
                         [0.32492, -1.00000, -0.20081],
                         [0.64984, 0.00000, -0.85065],
                         [0.20081, 0.61803, -0.85065],
                         [-0.52573, 0.38197, -0.85065],
                         [-0.52573, -0.38197, -0.85065],
                         [0.20081, -0.61803, -0.85065]]))


def render_mesh(mesh, resolution=1024, voxel_size=None, index=5, background=None, scale=2, no_fix_normal=True):
    if voxel_size is None:
        camera_pose = create_pose(FrontVector[index]*scale)
    else:
        camera_pose = create_shapenet_chair_camera_pose(voxel_size=voxel_size)

    render = Render(size=resolution, camera_pose=camera_pose,
                    background=background)

    triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(path=None,
                                                                                   clean=True,
                                                                                   mesh=mesh,
                                                                                   only_render_images=no_fix_normal)
    
    return rendered_image


def random_render_mesh(mesh, return_pose=True, rotation=None, elevation=None,return_angles=False, yfov=np.pi / 3.0, resolution:int = 224):
    if return_angles:
        camera_pose, rotation, elevation = create_random_pose(rotation=rotation, elevation=elevation, return_angles=True)
    else:
        camera_pose = create_random_pose(rotation=rotation, elevation=elevation)
    render = Render(size=resolution, camera_pose=camera_pose,
                    background=None,yfov=yfov)

    _, rendered_image, normal_map, depth_image, p_images = render.render(
        path=None, mesh=mesh, only_render_images=True)
    if return_pose:
        if return_angles:
            return rendered_image, camera_pose, rotation, elevation 
        else:
            return rendered_image, camera_pose
    else:
        return rendered_image

