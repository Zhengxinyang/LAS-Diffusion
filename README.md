
# Locally Attentional SDF Diffusion for Controllable 3D Shape Generation (SIGGRAPH 2023)

This repository contains the core implementation of our paper:

**[Locally Attentional SDF Diffusion for Controllable 3D Shape Generation](https://zhengxinyang.github.io/projects/LAS-Diffusion.html)**
<br>
[Xin-Yang Zheng](https://zhengxinyang.github.io/),
[Hao Pan](https://haopan.github.io/),
[Peng-Shuai Wang](https://wang-ps.github.io/),
[Xin Tong](https://scholar.google.com/citations?user=P91a-UQAAAAJ),
[Yang Liu](https://xueyuhanlang.github.io/) and [Heung-Yeung Shum](https://www.microsoft.com/en-us/research/people/hshum/)
<br>
<br>

![teaser](assets/representative_full.jpg)


## Installation
Following is the suggested way to install the dependencies of our code:
```
conda create -n sketch_diffusion
conda activate sketch_diffusion

conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=10.2 -c pytorch -c nvidia

pip install tqdm fire einops pyrender pyrr trimesh ocnn timm scikit-image==0.18.2 scikit-learn==0.24.2 pytorch-lightning==1.6.1
```

## Data Preparation
### SDF data creation
Please ref to [SDF-StyleGAN](https://github.com/Zhengxinyang/SDF-StyleGAN) for generating the SDF field from ShapeNet data or your customized data.

### Sketch data creation
Please refer to  `prepare_sketch.py` for details.


## Pre-trained Models
We provide the pretrained models for the category-conditioned generation and sketch-conditioned generation. Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1mN6iZ-NHAkSyQ526bcoECiDrDSx4zL9B?usp=sharing) and put them in `checkpoints/`.

## Usage
Please refer to the scripts in `scripts/` for the usage of our code.
### Train from Scratch
```
bash scripts/train_sketch.sh
bash scripts/train_category.sh
```
### Category-conditioned generation
```
bash scripts/generate_category.sh
```
### Sketch-conditioned generation
```
bash scripts/generate_sketch.sh
```


## Citation
If you find our work useful in your research, please consider citing:
```
@article {zheng2023lasdiffusion,
  title      = {Locally Attentional SDF Diffusion for Controllable 3D Shape Generation},
  author     = {Zheng, Xin-Yang and Pan, Hao and Wang, Peng-Shuai and Tong, Xin and Liu, Yang and Shum, Heung-Yeung},
  journal    = {ACM Transactions on Graphics (SIGGRAPH)},
  volume     = {42},
  number     = {4},
  year       = {2023},
}
```


