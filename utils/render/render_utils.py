from matplotlib import pyplot as plt
import pyrr
from pyrender import (
    DirectionalLight,
    SpotLight,
    PointLight,
)
import trimesh
import pyrender
import numpy as np
from PIL import Image
import time
import pyglet
import matplotlib
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
matplotlib.use("Agg")
pyglet.options['shadow_window'] = False

SIZE = None


class Render:
    def __init__(self, size, camera_pose, background=None, yfov=np.pi / 3.0):
        self.size = size
        global SIZE
        SIZE = size
        self.camera_pose = camera_pose
        self.background = background
        self.yfov = yfov

    def render(self, path, clean=True, intensity=3.0, mesh=None, only_render_images=False):
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = prepare_mesh(path, color=False, clean=clean)
        try:
            if mesh.visual.defined:
                mesh.visual.material.kwargs["Ns"] = 1.0
        except:
            print("Error loading material!")

        t1 = time.time()
        triangle_id, normal_map, depth_image, p_image = None, None, None, None
        if not only_render_images:

            triangle_id, normal_map, depth_image, p_image = correct_normals(
                mesh, self.camera_pose, correct=True)
        mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        rendered_image, _ = pyrender_rendering(
            mesh1, viz=False, light=True, camera_pose=self.camera_pose, intensity=intensity, bg_color=self.background, yfov=self.yfov,
        )

        return triangle_id, rendered_image, normal_map, depth_image, p_image

    def render_normal(self, path, clean=True, intensity=6.0, mesh=None):
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = prepare_mesh(path, color=False, clean=clean)
        try:
            if mesh.visual.defined:
                mesh.visual.material.kwargs["Ns"] = 1.0
        except:
            print("Error loading material!")

        triangle_id, normal_map, depth_image, p_image = correct_normals(
            mesh, self.camera_pose, correct=True)

        return normal_map, depth_image


def correct_normals(mesh, camera_pose, correct=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    a, b, index_tri, sign, p_image = trimesh_ray_tracing(
        mesh, camera_pose, resolution=SIZE*2, rayintersector=rayintersector
    )
    if correct:
        mesh.faces[index_tri[sign > 0]] = np.fliplr(
            mesh.faces[index_tri[sign > 0]])

    normalmap = render_normal_map(
        pyrender.Mesh.from_trimesh(mesh, smooth=False),
        camera_pose,
        SIZE,
        viz=False,
    )
    return b, normalmap, a, p_image


def normalize_mesh(mesh, mode="sphere"):
    if mode == "sphere":
        mesh.vertices = mesh.vertices - mesh.vertices.mean(0)
        scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
        mesh.vertices = mesh.vertices / scale
    elif mode == "com":
        box = mesh.bounding_box_oriented
        mesh.vertices = mesh.vertices - box.vertices.mean(0)
        scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
        mesh.vertices = mesh.vertices / scale


def prepare_mesh(model_name, color=False, clean=False):
    mesh = trimesh.load(model_name, force="mesh")

    if clean:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()

        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)

    normalize_mesh(mesh, "com")
    if color:
        mesh.visual.face_colors = generate_unique_colors(
            mesh.faces.shape[0]
        )
    return mesh


def generate_unique_colors(size):
    colors = np.arange(1, 254 * 254 * 254)
    z = np.random.choice(colors, (size), replace=False)
    colors = np.unravel_index(z, (255, 255, 255))
    colors = np.stack(colors, 1)
    return colors


def init_light(scene, camera_pose, intensity=6.0):
    direc_l = DirectionalLight(color=np.ones(3), intensity=intensity)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=intensity,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )
    point_l = PointLight(color=np.ones(3), intensity=2*intensity)

    direc_l_node = scene.add(direc_l, pose=camera_pose)
    point_l_node = scene.add(point_l, pose=camera_pose)
    spot_l_node = scene.add(spot_l, pose=camera_pose)
    return spot_l_node, direc_l_node, point_l_node


class CustomShaderCache:
    def __init__(self):
        self.program = None

    def get_program(
            self, vertex_shader, fragment_shader, geometry_shader=None, defines=None
    ):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(
                os.path.dirname(os.path.abspath(__file__)) +
                "/shades/mesh.vert",
                os.path.dirname(os.path.abspath(__file__)) + "/shades/mesh.frag", defines=defines
            )
        return self.program


def render_normal_map(mesh, camera_pose, size, viz=False):
    scene = pyrender.Scene(bg_color=(255, 255, 255))
    scene.add(mesh)
    # np.pi/3.0 <-> 60°
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    renderer._renderer._program_cache = CustomShaderCache()

    normals, depth = renderer.render(
        scene
    )

    world_space_normals = normals / 255 * 2 - 1

    if viz:
        image = Image.fromarray(normals, "RGB")
        image.show()

    return world_space_normals


def pyrender_rendering(mesh, camera_pose, viz=False, light=False, intensity=3.0, bg_color=None, yfov=np.pi / 3.0):

    r = pyrender.OffscreenRenderer(SIZE, SIZE)

    scene = pyrender.Scene(bg_color=bg_color)
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.)

    camera = scene.add(camera, pose=camera_pose)
    # light
    if light:
        lights = init_light(scene, camera_pose, intensity=intensity)

    scene.set_pose(camera, camera_pose)

    if light:
        color, depth = r.render(
            scene, flags=pyrender.constants.RenderFlags.ALL_SOLID)
    else:
        color, depth = r.render(
            scene, flags=pyrender.constants.RenderFlags.FLAT
        )

    if viz:
        plt.figure()
        plt.imshow(color)
    return color, depth


def create_shapenet_chair_camera_pose(voxel_size=32):
    frontvector = np.zeros(3)

    frontvector[0] = -voxel_size
    frontvector[1] = voxel_size
    frontvector[2] = -voxel_size

    camera_pose = np.array(pyrr.Matrix44.look_at(eye=frontvector,
                                                 target=np.ones(
                                                     3)*voxel_size/2,
                                                 up=np.array([0.0, 1.0, 0])).T)
    return np.linalg.inv(np.array(camera_pose))


def create_pose(eye):
    target = np.zeros(3)
    camera_pose = np.array(pyrr.Matrix44.look_at(eye=eye,
                                                 target=target,
                                                 up=np.array([0.0, 1.0, 0])).T)
    return np.linalg.inv(np.array(camera_pose))


def trimesh_ray_tracing(mesh, M, resolution=225, fov=60, rayintersector=None):

    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    scene.camera_transform = M @ extra
    scene.camera.resolution = [resolution, resolution]

    scene.camera.fov = fov, fov

    origins, vectors, pixels = scene.camera_rays()

    index_tri, index_ray, points = rayintersector.intersects_id(
        origins, vectors, multiple_hits=False, return_locations=True
    )
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    sign = trimesh.util.diagonal_dot(
        mesh.face_normals[index_tri], vectors[index_ray])

    pixel_ray = pixels[index_ray]

    a = np.zeros(scene.camera.resolution, dtype=np.uint8)
    b = np.ones(scene.camera.resolution, dtype=np.int32) * -1
    p_image = np.ones([scene.camera.resolution[0],
                      scene.camera.resolution[1], 3], dtype=np.float32) * -1

    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth
    b[pixel_ray[:, 0], pixel_ray[:, 1]] = index_tri
    p_image[pixel_ray[:, 0], pixel_ray[:, 1]] = points

    return a, b, index_tri, sign, p_image
