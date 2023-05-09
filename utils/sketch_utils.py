import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import pyrr
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

try:
    import cv2
except ImportError:
    pass


def get_sketch(image):
    edges = cv2.Canny(image=image, threshold1=20, threshold2=180)
    edges = cv2.GaussianBlur(edges, (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.bitwise_not(edges)
    edges[edges < 255] = 0
    return edges


def create_random_pose(rotation=None, elevation=None, return_angles=False):
    if rotation is None:
        rotation = np.random.rand() * 360
    else:
        rotation = float(rotation)

    if elevation is None:
        elevation = np.random.rand() * 40
    else:
        elevation = float(elevation)
    eye = np.array([np.sin(rotation/180*np.pi)*np.cos(elevation/180*np.pi),
                    np.sin(elevation/180*np.pi),
                    np.cos(rotation/180*np.pi)*np.cos(elevation/180*np.pi)]) * 2.5

    target = np.zeros(3)
    camera_pose = np.array(pyrr.Matrix44.look_at(eye=eye,
                                                 target=target,
                                                 up=np.array([0.0, 1.0, 0])).T)
    if return_angles:
        return np.linalg.inv(camera_pose), rotation, elevation
    else:
        return np.linalg.inv(camera_pose)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_P_from_transform_matrix(matrix):

    location = matrix[0:3, 3]

    rotation = np.transpose(matrix[0:3, 0:3])
    t = np.tan(np.pi / 6.0)
    width = 224

    return np.array([[112 / t, 0, width/2], [0, 112 / t, width/2], [0, 0, 1]]) \
        @ np.concatenate([rotation, np.expand_dims(-1*rotation @ location, 1)], 1)


Projection_List = [
    get_P_from_transform_matrix(create_random_pose(rotation=0, elevation=20)),
    get_P_from_transform_matrix(create_random_pose(rotation=45, elevation=20)),
    get_P_from_transform_matrix(create_random_pose(rotation=90, elevation=20)),
    get_P_from_transform_matrix(
        create_random_pose(rotation=135, elevation=20)),
    get_P_from_transform_matrix(
        create_random_pose(rotation=180, elevation=20)),
]
Projection_List_zero = [
    get_P_from_transform_matrix(create_random_pose(rotation=0, elevation=0)),
    get_P_from_transform_matrix(create_random_pose(rotation=45, elevation=0)),
    get_P_from_transform_matrix(create_random_pose(rotation=90, elevation=0)),
    get_P_from_transform_matrix(create_random_pose(rotation=135, elevation=0)),
    get_P_from_transform_matrix(create_random_pose(rotation=180, elevation=0)),
]
