import numpy as np
from sparsity_network.sparsity_trainer import Sparsity_DiffusionModel
from utils.utils import str2bool, ensure_directory
from utils.sparsity_utils import load_input
from utils.mesh_utils import process_sdf
import argparse
import os


def generate_one_mesh(
    model_path: str,
    npy_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 1000,
    truncated_index: float = 0,
    level: float = 0.0,
):
    npy_name = npy_path.split('/')[-1].split(".")[0]
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(
        model_path).cuda()

    res = discrete_diffusion.generate_results_from_single_voxel(
        low_res_voxel=load_input(npy_path), ema=ema, use_ddim=False, steps=steps, truncated_index=truncated_index)

    name = f"{model_name}_{model_id}_{ema}_ddpm_{steps}_{npy_name}_{truncated_index}"

    mesh = process_sdf(res[0], level=level, normalize=True)
    np.save(os.path.join(output_path, f"{name}_sdf.npy"), res[0])

    mesh.export(os.path.join(output_path, f"{name}.obj"))


def generate_meshes(
    model_path: str,
    npy_folder: str,
    batch_size: int,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 1000,
    based_gt: bool = False,
    truncated_index: float = 0,
    level: float = 0.0,
    save_npy: bool = True,
    save_mesh: bool = True,
    start_index: int = 0,
    end_index: int = 100000,
):
    npy_name = npy_folder.split('/')[-1]
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(
        model_path).cuda()

    postfix = f"{model_name}_{model_id}_{ema}_ddpm_{steps}_{npy_name}_{truncated_index}"
    save_path = os.path.join(output_path, postfix)
    if based_gt:
        save_path += "_gt"
    ensure_directory(save_path)
    discrete_diffusion.generate_results_from_folder(
        folder=npy_folder, ema=ema,
        save_path=save_path, batch_size=batch_size, use_ddim=False, steps=steps,
        truncated_index=truncated_index, sort_npy=not based_gt, level=level,
        save_npy=save_npy, save_mesh=save_mesh, start_index=start_index, end_index=end_index)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='generate_one_mesh',
                        help="please choose :\n \
                       1. 'generate_one_mesh'\n \
                       2. 'generate_meshes' \
                       ")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--npy_path", type=str, default="./test.npy")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--based_gt", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_npy", type=str2bool, default=True)
    parser.add_argument("--save_mesh", type=str2bool, default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--level", type=float, default=0.0)

    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)
    if method == "generate_one_mesh":
        generate_one_mesh(model_path=args.model_path, npy_path=args.npy_path, output_path=args.output_path, steps=args.steps, level=args.level,
                          ema=args.ema, truncated_index=args.truncated_time)
    elif method == "generate_meshes":
        generate_meshes(model_path=args.model_path, npy_folder=args.npy_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                        batch_size=args.batch_size, based_gt=args.based_gt, truncated_index=args.truncated_time, level=args.level, save_npy=args.save_npy, save_mesh=args.save_mesh,
                        start_index=args.start_index, end_index=args.end_index)
    else:
        raise NotImplementedError
