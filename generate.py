import torch
import numpy as np
from network.model_trainer import DiffusionModel
from utils.mesh_utils import voxel2mesh
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups
import argparse
import os
from tqdm import tqdm
from utils.utils import VIT_MODEL, png_fill_color
from PIL import Image
from utils.utils import png_fill_color
import timm
from utils.sketch_utils import _transform, create_random_pose, get_P_from_transform_matrix


def generate_unconditional(
    model_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 36,
    start_index: int = 0,
    steps: int = 50,
    truncated_time: float = 0.0,
):
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    postfix = f"{model_name}_{model_id}_{ema}_{steps}_{truncated_time}_unconditional"
    root_dir = os.path.join(output_path, postfix)

    ensure_directory(root_dir)
    batches = num_to_groups(num_generate, 50)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    index = start_index
    for batch in batches:
        res_tensor = generator.sample_unconditional(
            batch_size=batch, steps=steps, truncated_index=truncated_time)
        for i in tqdm(range(batch), desc=f'save results in one batch in {root_dir}'):
            voxel = res_tensor[i].squeeze().cpu().numpy()
            np.save(os.path.join(root_dir, str(index)), voxel)
            try:

                voxel[voxel > 0] = 1
                voxel[voxel < 0] = 0
                mesh = voxel2mesh(voxel)
                mesh.export(os.path.join(root_dir, str(index) + ".obj"))
            except Exception as e:
                print(str(e))
            index += 1


def generate_based_on_data_class(
    model_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 1,
    steps: int = 50,
    truncated_time: float = 0.0,
    w: float = 1.0,
    data_class: str = "chair",
):
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    assert discrete_diffusion.use_text_condition
    postfix = f"{model_name}_{model_id}_{ema}_{steps}_{truncated_time}_{w}_{data_class}"

    root_dir = os.path.join(output_path, postfix)
    ensure_directory(root_dir)

    from utils.shapenet_utils import snc_category_to_synth_id_all
    label = snc_category_to_synth_id_all[data_class]
    from utils.condition_data import text_features
    text_c = text_features[data_class]

    batches = num_to_groups(num_generate, 50)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    index = 0
    for batch in batches:
        res_tensor = generator.sample_with_text(text_c=text_c, batch_size=batch,
                                                steps=steps, truncated_index=truncated_time, text_w=w)

        for i in tqdm(range(batch), desc=f'save results in one batch in {root_dir}'):
            voxel = res_tensor[i].squeeze().cpu().numpy()
            np.save(os.path.join(root_dir, str(index)), voxel)
            try:
                voxel[voxel > 0] = 1
                voxel[voxel < 0] = 0
                mesh = voxel2mesh(voxel)
                mesh.export(os.path.join(root_dir, str(index) + ".obj"))
            except Exception as e:
                print(str(e))
            index += 1


def generate_based_on_sketch(
    model_path: str,
    sketch_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    num_generate: int = 1,
    steps: int = 50,
    truncated_time: float = 0.0,
    w: float = 1.0,
    view_information: int = 0,
    kernel_size: float = 2,
    detail_view: bool = False,
    rotation: float = 0.0,
    elevation: float = 0.0,
):

    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    image_name = sketch_path.split("/")[-2] + "_" + sketch_path.split("/")[-1].split(".")[0]

    postfix = f"{model_name}_{model_id}_{ema}_{image_name}_{w}_{view_information}"

    root_dir = os.path.join(output_path, postfix)
    ensure_directory(root_dir)
    preprocess = _transform(224)
    device = "cuda"

    feature_extractor = timm.create_model(
        VIT_MODEL, pretrained=True).to(device)
    with torch.no_grad():
        im = Image.open(sketch_path)
        im = png_fill_color(im).convert("RGB")
        im.save(os.path.join(root_dir, "input.png"))
        im = preprocess(im).unsqueeze(0).to(device)
        image_features = feature_extractor.forward_features(im)
        sketch_c = image_features.squeeze(0).cpu().numpy()

    from utils.sketch_utils import Projection_List, Projection_List_zero
    if detail_view:
        projection_matrix  = get_P_from_transform_matrix(
            create_random_pose(rotation=rotation, elevation=elevation))
    elif view_information == -1:
        projection_matrix = None
    else:
        if discrete_diffusion.elevation_zero:

            projection_matrix = Projection_List_zero[view_information]
        else:
            projection_matrix = Projection_List[view_information]
    batches = num_to_groups(num_generate, 32)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    index = 0
    for batch in batches:
        res_tensor = generator.sample_with_sketch(sketch_c=sketch_c, batch_size=batch,
                                                  projection_matrix=projection_matrix, kernel_size=kernel_size,
                                                  steps=steps, truncated_index=truncated_time, sketch_w=w)
        for i in tqdm(range(batch), desc=f'save results in one batch in {root_dir}'):
            voxel = res_tensor[i].squeeze().cpu().numpy()
            np.save(os.path.join(root_dir, str(index)), voxel)
            # print(voxel)
            try:
                voxel[voxel > 0] = 1
                voxel[voxel < 0] = 0
                mesh = voxel2mesh(voxel)
                mesh.export(os.path.join(root_dir, str(index) + ".obj"))
            except Exception as e:
                print(str(e))
            index += 1

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='generate_unconditional',
                        help="please choose :\n \
                            1. 'generate_unconditional' \n \
                            2. 'generate_based_on_class' \n \
                            3. 'generate_based_on_sketch' \n \ ")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--data_class", type=str, default="chair")
    parser.add_argument("--text_w", type=float, default=1.0)
    parser.add_argument("--image_path", type=str, default="test.png")
    parser.add_argument("--image_name", type=str2bool, default=False)
    parser.add_argument("--sketch_w", type=float, default=1.0)
    parser.add_argument("--view_information", type=int, default=0)
    parser.add_argument("--detail_view", type=str2bool, default=False)
    parser.add_argument("--rotation", type=float, default=0.)
    parser.add_argument("--elevation", type=float, default=0.)
    parser.add_argument("--kernel_size", type=float, default=4.)
    parser.add_argument("--verbose", type=str2bool, default=False)

    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)
    if method == "generate_unconditional":
        generate_unconditional(model_path=args.model_path, num_generate=args.num_generate,
                               output_path=args.output_path, ema=args.ema, start_index=args.start_index, steps=args.steps,
                               truncated_time=args.truncated_time)
    elif method == "generate_based_on_class":
        generate_based_on_data_class(model_path=args.model_path, num_generate=args.num_generate,
                                     output_path=args.output_path, ema=args.ema, steps=args.steps,
                                     truncated_time=args.truncated_time, w=args.text_w, data_class=args.data_class)
    elif method == "generate_based_on_sketch":
        generate_based_on_sketch(model_path=args.model_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                                 num_generate=args.num_generate, truncated_time=args.truncated_time,
                                 sketch_path=args.image_path, w=args.sketch_w,
                                 view_information=args.view_information, kernel_size=args.kernel_size,
                                 detail_view=args.detail_view,
                                 rotation=args.rotation, elevation=args.elevation
                                 )
    else:
        raise NotImplementedError
