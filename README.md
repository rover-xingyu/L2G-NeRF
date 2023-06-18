# L2G-NeRF: Local-to-Global Registration for Bundle-Adjusting Neural Radiance Fields
**[Project Page](https://rover-xingyu.github.io/L2G-NeRF/) |
[Paper](https://arxiv.org/pdf/2211.11505.pdf) |
[Video](https://www.youtube.com/watch?v=y8XP9Umt6Mw)**

[Yue Chen¹](https://fanegg.github.io/), 
[Xingyu Chen¹](https://rover-xingyu.github.io/), 
[Xuan Wang²](https://xuanwangvc.github.io/),
[Qi Zhang³](https://qzhang-cv.github.io/), 
[Yu Guo¹](https://yuguo-xjtu.github.io/), 
[Ying Shan³](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), 
[Fei Wang¹](http://www.aiar.xjtu.edu.cn/info/1046/1242.htm). 

[¹Xi'an Jiaotong University](http://en.xjtu.edu.cn/),
[²Ant Group](https://www.antgroup.com/en),
[³Tencent AI Lab](https://ai.tencent.com/ailab/en/index/). 

This repository is an official implementation of [L2G-NeRF](https://rover-xingyu.github.io/L2G-NeRF/) using [pytorch](https://pytorch.org/). 

# :computer: Installation

## Hardware

* We implement all experiments on a single NVIDIA GeForce RTX 2080 Ti GPU. 
* L2G-NeRF takes about 4.5 and 8 hours for training in synthetic objects and real-world scenes, respectively, while training BARF takes about 8 and 10.5 hours.

## Software

* Clone this repo by `git clone https://github.com/rover-xingyu/L2G-NeRF`
* This code is developed with Python3. PyTorch 1.9+ is required. It is recommended use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment, use `conda env create --file requirements.yaml python=3` to install the dependencies and activate it by `conda activate L2G-NeRF`
--------------------------------------

# :key: Training and Evaluation

## Data download

Both the Blender synthetic data and LLFF real-world data can be found in the [NeRF Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
For convenience, you can download them with the following script:
  ```bash
  # Blender
  gdown --id 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG # download nerf_synthetic.zip
  unzip nerf_synthetic.zip
  rm -f nerf_synthetic.zip
  mv nerf_synthetic data/blender
  # LLFF
  gdown --id 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g # download nerf_llff_data.zip
  unzip nerf_llff_data.zip
  rm -f nerf_llff_data.zip
  mv nerf_llff_data data/llff
  ```
--------------------------------------

## Running L2G-NeRF
To train and evaluate L2G-NeRF:
```bash
# <GROUP> and <NAME> can be set to your likes, while <SCENE> is specific to datasets

# NeRF (3D): Synthetic Objects
# Blender (<SCENE>={chair,drums,ficus,hotdog,lego,materials,mic,ship})
python3 train.py \
--model=l2g_nerf --yaml=l2g_nerf_blender \
--group=exp_synthetic --name=l2g_lego \
--data.scene=lego --gpu=3 \
--data.root=/the/data/path/of/nerf_synthetic/ \
--camera.noise_r=0.07 --camera.noise_t=0.5

python3 evaluate.py \
--model=l2g_nerf --yaml=l2g_nerf_blender \
--group=exp_synthetic --name=l2g_lego \
--data.scene=lego --gpu=3 \
--data.root=/the/data/path/of/nerf_synthetic/ \
--data.val_sub= --resume

# NeRF (3D): Real-World Scenes
# LLFF (<SCENE>={fern,flower,fortress,horns,leaves,orchids,room,trex})
python3 train.py \
--model=l2g_nerf --yaml=l2g_nerf_llff \
--group=exp_LLFF --name=l2g_fern \
--data.scene=fern --gpu=3 \
--data.root=/the/data/path/of/nerf_llff_data/ \
--loss_weight.global_alignment=2

python3 evaluate.py \
--model=l2g_nerf --yaml=l2g_nerf_llff \
--group=exp_LLFF --name=l2g_fern \
--data.scene=fern --gpu=3 \
--data.root=/the/data/path/of/nerf_llff_data/ \
--loss_weight.global_alignment=2 \
--resume

# Neural image Alignment (2D): Rigid
# use the image of “Girl With a Pearl Earring” renovation ©Koorosh Orooj (CC BY-SA 4.0) for rigid image alignment
python3 train.py \
--model=l2g_planar --yaml=l2g_planar \
--group=exp_planar --name=l2g_girl  \
--warp.type=rigid --warp.dof=3 \
--data.image_fname=data/girl.jpg \
--data.image_size=[595,512] \
--data.patch_crop=[260,260] \
--seed=1 --gpu=3

# Neural image Alignment (2D): Homography
# use the image of “cat” from ImageNet for homography image alignment
python3 train.py \
--model=l2g_planar --yaml=l2g_planar \
--group=exp_planar --name=l2g_cat \
--warp.type=homography --warp.dof=8 \
--data.image_fname=data/cat.jpg \
--data.image_size=[360,480] \
--data.patch_crop=[180,180] \
--gpu=0
```
--------------------------------------

## Running BARF
If you want to train and evaluate the BARF extension of the original NeRF model that jointly optimizes poses (coarse-to-fine positional encoding):
```bash
# <GROUP> and <NAME> can be set to your likes, while <SCENE> is specific to datasets

# NeRF (3D): Synthetic Objects
# Blender (<SCENE>={chair,drums,ficus,hotdog,lego,materials,mic,ship})
python3 train.py \
--model=barf --yaml=barf_blender \
--group=exp_synthetic --name=barf_lego \
--data.scene=lego --gpu=1 \
--data.root=/the/data/path/of/nerf_synthetic/ \
--camera.noise_r=0.07 --camera.noise_t=0.5

python3 evaluate.py \
--model=barf --yaml=barf_blender \
--group=exp_synthetic --name=barf_lego \
--data.scene=lego --gpu=1 \
--data.root=/the/data/path/of/nerf_synthetic/ \
--data.val_sub= --resume

# NeRF (3D): Real-World Scenes
# LLFF (<SCENE>={fern,flower,fortress,horns,leaves,orchids,room,trex})
python3 train.py \
--model=barf --yaml=barf_llff \
--group=exp_LLFF --name=barf_fern \
--data.scene=fern --gpu=1 \
--data.root=/the/data/path/of/nerf_llff_data/

python3 evaluate.py \
--model=barf --yaml=barf_llff \
--group=exp_LLFF --name=barf_fern \
--data.scene=fern --gpu=1 \
--data.root=/the/data/path/of/nerf_llff_data/ \
--resume

# Neural image Alignment (2D): Rigid
# use the image of “Girl With a Pearl Earring” renovation ©Koorosh Orooj (CC BY-SA 4.0) for rigid image alignment
python3 train.py \
--model=planar --yaml=planar \
--group=exp_planar --name=barf_girl \
--warp.type=rigid --warp.dof=3 \
--data.image_fname=data/girl.jpg \
--data.image_size=[595,512] \
--data.patch_crop=[260,260] \
--seed=1 --gpu=1

# Neural image Alignment (2D): Homography
# use the image of “cat” from ImageNet for homography image alignment
python3 train.py \
--model=planar --yaml=planar \
--group=exp_planar --name=barf_cat \
--warp.type=homography --warp.dof=8 \
--data.image_fname=data/cat.jpg \
--data.image_size=[360,480] \
--data.patch_crop=[180,180] \
--gpu=2
```
--------------------------------------

## Running Naive
If you want to train and evaluate the Naive extension of the original NeRF model that jointly optimizes poses (full positional encoding):

```bash
# <GROUP> and <NAME> can be set to your likes, while <SCENE> is specific to datasets

# NeRF (3D): Synthetic Objects
# Blender (<SCENE>={chair,drums,ficus,hotdog,lego,materials,mic,ship})
python3 train.py \
--model=barf --yaml=barf_blender \
--group=exp_synthetic --name=nerf_lego \
--data.scene=lego --gpu=2 \
--data.root=/home/cy/PNW/datasets/nerf_synthetic/ \
--barf_c2f=null \
--camera.noise_r=0.07 --camera.noise_t=0.5

python3 evaluate.py \
--model=barf --yaml=barf_blender \
--group=exp_synthetic --name=nerf_lego \
--data.scene=lego --gpu=2 \
--data.root=/home/cy/PNW/datasets/nerf_synthetic/ \
--barf_c2f=null \
--data.val_sub= --resume

# NeRF (3D): Real-World Scenes
# LLFF (<SCENE>={fern,flower,fortress,horns,leaves,orchids,room,trex})
python3 train.py \
--model=barf --yaml=barf_llff \
--group=exp_LLFF --name=nerf_fern \
--data.scene=fern --gpu=2 \
--data.root=/home/cy/PNW/datasets/nerf_llff_data/ \
--barf_c2f=null

python3 evaluate.py \
--model=barf --yaml=barf_llff \
--group=exp_LLFF --name=nerf_fern \
--data.scene=fern --gpu=2 \
--data.root=/home/cy/PNW/datasets/nerf_llff_data/ \
--barf_c2f=null --resume 

# Neural image Alignment (2D): Rigid
# use the image of “Girl With a Pearl Earring” renovation ©Koorosh Orooj (CC BY-SA 4.0) for rigid image alignment
python3 train.py \
--model=planar --yaml=planar \
--group=exp_planar --name=naive_girl \
--warp.type=rigid --warp.dof=3 \
--data.image_fname=data/girl.jpg \
--data.image_size=[595,512] \
--data.patch_crop=[260,260] \
--seed=1 --gpu=2 --barf_c2f=null

# Neural image Alignment (2D): Homography
# use the image of “cat” from ImageNet for homography image alignment
python3 train.py \
--model=planar --yaml=planar \
--group=exp_planar --name=naive_cat \
--warp.type=homography --warp.dof=8 \
--data.image_fname=data/cat.jpg \
--data.image_size=[360,480] \
--data.patch_crop=[180,180] \
--gpu=3 --barf_c2f=null
```
--------------------------------------

## Running reference NeRF
If you want to train and evaluate the reference NeRF models (assuming known camera poses):
```bash
# <GROUP> and <NAME> can be set to your likes, while <SCENE> is specific to datasets

# NeRF (3D): Synthetic Objects
# Blender (<SCENE>={chair,drums,ficus,hotdog,lego,materials,mic,ship})
python3 train.py \
--model=nerf --yaml=nerf_blender \
--group=exp_synthetic --name=ref_lego \
--data.scene=lego --gpu=0 \
--data.root=/home/cy/PNW/datasets/nerf_synthetic/ 

python3 evaluate.py \
--model=nerf --yaml=nerf_blender \
--group=exp_synthetic --name=ref_lego \
--data.scene=lego --gpu=0 \
--data.root=/home/cy/PNW/datasets/nerf_synthetic/ \
--data.val_sub= --resume

# NeRF (3D): Real-World Scenes
# LLFF (<SCENE>={fern,flower,fortress,horns,leaves,orchids,room,trex})
python3 train.py \
--model=nerf --yaml=nerf_llff \
--group=exp_LLFF --name=ref_fern \
--data.scene=fern --gpu=0 \
--data.root=/home/cy/PNW/datasets/nerf_llff_data/


python3 evaluate.py \
--model=nerf --yaml=nerf_llff \
--group=exp_LLFF --name=ref_fern \
--data.scene=fern --gpu=0 \
--data.root=/home/cy/PNW/datasets/nerf_llff_data/ \
--resume
```
--------------------------------------

# :laughing: Visualization

## Results and Videos
All the results will be stored in the directory `output/<GROUP>/<NAME>`.
You may want to organize your experiments by grouping different runs in the same group. Many videos will be created to visualize the pose optimization process and novel view synthesis.

## TensorBoard
The TensorBoard events include the following:
- **SCALARS**: the rendering losses and PSNR over the course of optimization. For L2G_NeRF/BARF/Naive, the rotational/translational errors with respect to the given poses are also computed.
- **IMAGES**: visualization of the RGB images and the RGB/depth rendering.

## Visdom
The visualization of 3D camera poses is provided in Visdom:
Run `visdom -port 8600` to start the Visdom server.  The Visdom host server is default to `localhost`; this can be overridden with `--visdom.server` (see `options/base.yaml` for details). If you want to disable Visdom visualization, add `--visdom!`.

## Mesh
The `extract_mesh.py` script provides a simple way to extract the underlying 3D geometry using marching cubes (supporte for the Blender dataset). Run as follows:
```bash
python3 extract_mesh.py \
--model=l2g_nerf --yaml=l2g_nerf_blender \
--group=exp_synthetic --name=l2g_lego \
--data.scene=lego --gpu=3 \
--data.root=/home/cy/PNW/datasets/nerf_synthetic/ \
--data.val_sub= --resume
```
--------------------------------------

# :mag_right: Codebase structure

The main engine and network architecture in `model/l2g_nerf.py` inherit those from `model/nerf.py`.

Some tips on using and understanding the codebase:
- The computation graph for forward/backprop is stored in `var` throughout the codebase.
- The losses are stored in `loss`. To add a new loss function, just implement it in `compute_loss()` and add its weight to `opt.loss_weight.<name>`. It will automatically be added to the overall loss and logged to Tensorboard.
- If you are using a multi-GPU machine, you can set `--gpu=<gpu_number>` to specify which GPU to use. Multi-GPU training/evaluation is currently not supported.
- To resume from a previous checkpoint, add `--resume=<ITER_NUMBER>`, or just `--resume` to resume from the latest checkpoint.
- To eliminate the global alignment objective, set `--loss_weight.global_alignment=null`, the ablation is equivalent to a local registration method.

--------------------------------------
# Citation

If you find this project useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{chen2023local,
  title={Local-to-global registration for bundle-adjusting neural radiance fields},
  author={Chen, Yue and Chen, Xingyu and Wang, Xuan and Zhang, Qi and Guo, Yu and Shan, Ying and Wang, Fei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8264--8273},
  year={2023}
}
```
--------------------------------------

# Acknowledge
Our code is based on the awesome pytorch implementation of Bundle-Adjusting Neural Radiance Fields ([BARF](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)). We appreciate all the contributors.
