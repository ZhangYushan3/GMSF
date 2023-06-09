# GMSF: Global Matching Scene Flow
### [Project Page (TODO)]() | [Paper](https://arxiv.org/abs/2305.17432)
<br/>

> GMSF: Global Matching Scene Flow  
> [Yushan Zhang](), [Johan Edstedt](https://scholar.google.com/citations?user=Ul-vMR0AAAAJ), [Bastian Wandt](https://scholar.google.com/citations?user=z4aXEBYAAAAJ), [Per-Erik Forssén](https://scholar.google.com/citations?user=SZ6jH-4AAAAJ), [Maria Magnusson](), [Michael Felsberg](https://scholar.google.com/citations?&user=lkWfR08AAAAJ)  
> Arxiv 2023

## Get started
# Installation:

Create a conda environment:
```bash
conda create -n GMSF python=3.8
conda activate GMSF
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install other dependencies:
```bash
pip install opencv-python open3d tensorboard imageio numba
```

# Dataset Preparation:

1. FlyingThings3D(HPLFlowNet without occlusion / CamLiFlow with occlusion):

Download [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).
flyingthings3d_disparity.tar.bz2, flyingthings3d_disparity_change.tar.bz2, FlyingThings3D_subset_disparity_occlusions.tar.bz2, FlyingThings3D_subset_flow.tar.bz2, FlyingThings3D_subset_flow_occlusions.tar.bz2 and FlyingThings3D_subset_image_clean.tar.bz2 are needed. Then extract the files in /path/to/flyingthings3d such that the directory looks like
```bash
/path/to/flyingthings3d
├── train/
│   ├── disparity/
│   ├── disparity_change/
│   ├── disparity_occlusions/
│   ├── flow/
│   ├── flow_occlusions/
│   ├── image_clean/
├── val/
│   ├── disparity/
│   ├── disparity_change/
│   ├── disparity_occlusions/
│   ├── flow/
│   ├── flow_occlusions/
│   ├── image_clean/
```
Preprocess dataset using the following command:
```bash
cd utils
python preprocess_flyingthings3d_subset.py --input_dir /mnt/data/flyingthings3d_subset --output_dir flyingthings3d_subset
python preprocess_flyingthings3d_subset.py --input_dir /mnt/data/flyingthings3d_subset --output_dir flyingthings3d_subset_non-occluded --remove_occluded_points
```

2. FlyingThings3D(FlowNet3D with occlusion):

The processed data is also provided [here](https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view?usp=sharing) for download (total size ~11GB)

3. KITTI(HPLFlowNet without occlusion):

First, download the following parts:
Main data: [data_scene_flow.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip)
Calibration files: [data_scene_flow_calib.zip](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip)
Unzip them and organize the directory as follows:
```bash
datasets/KITTI_stereo2015
├── testing
│   ├── calib_cam_to_cam
│   ├── calib_imu_to_velo
│   ├── calib_velo_to_cam
│   ├── image_2
│   ├── image_3
└── training
    ├── calib_cam_to_cam
    ├── calib_imu_to_velo
    ├── calib_velo_to_cam
    ├── disp_noc_0
    ├── disp_noc_1
    ├── disp_occ_0
    ├── disp_occ_1
    ├── flow_noc
    ├── flow_occ
    ├── image_2
    ├── image_3
    ├── obj_map
```
Preprocess dataset using the following command:
```bash
cd utils
python process_kitti.py datasets/KITTI_stereo2015/ SAVE_PATH/KITTI_processed_occ_final
```

4. KITTI(FlowNet3D with occlusion):

The processed data is also provided [here](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi) for download

# The datasets directory should be orginized as:
```bash
datasets
├── datasets_KITTI_flownet3d
│   ├── kitti_rm_ground
├── datasets_KITTI_hplflownet
│   ├── KITTI_processed_occ_final
├── FlyingThings3D_flownet3d
├── flyingthings3d_subset
│   ├── train
│   ├── val
├── flyingthings3d_subset_non-occluded
│   ├── train
│   ├── val
├── KITTI_stereo2015
│   ├── testing
│   ├── training
```

# Traning and Testing:
Training (HPLFlowNet / CamLiFlow with occlusion): 
```bash
cd gmsf
python main_gmsf.py \
    --checkpoint_dir checkpoints \
    --stage things_subset \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 8 \
    --feature_channels_point 128 \
    --lr 2e-4 --batch_size 8 --num_steps 600000
```
Training (HPLFlowNet without occlusion): 
```bash
cd gmsf
python main_gmsf.py \
    --checkpoint_dir checkpoints \
    --stage things_subset_non-occluded \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 8 \
    --feature_channels_point 128 \
    --lr 4e-4 --batch_size 8 --num_steps 600000
```
Training (FlowNet3D with occlusion): 
```bash
cd gmsf
python main_gmsf.py \
    --checkpoint_dir checkpoints \
    --stage things_flownet3d \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 8 \
    --feature_channels_point 128 \
    --lr 4e-4 --batch_size 8 --num_steps 600000
```
Testing (HPLFlowNet / CamLiFlow with occlusion): 
```bash
cd gmsf
python main_gmsf.py --resume checkpoints/step_600000.pth \
    --stage things_subset \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 8 \
    --feature_channels_point 128 \
    --eval
```
Testing (HPLFlowNet without occlusion):  
```bash
cd gmsf
python main_gmsf.py --resume checkpoints/step_600000.pth \
    --stage things_subset_non-occluded \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 8 \
    --feature_channels_point 128 \
    --eval
```
Testing (FlowNet3D with occlusion): 
```bash
cd gmsf
python main_gmsf.py --resume checkpoints/step_600000.pth \
    --stage things_flownet3d \
    --backbone DGCNN \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 8 \
    --feature_channels_point 128 \
    --eval
```

# Pretrained Checkpoints
Model trained on FTD_c: [MODEL_FTDc](https://drive.google.com/file/d/12ZSi6PwNcINSeXyVuJHZUMtmIMc2XJl9/view?usp=sharing)

Model trained on FTD_o: [MODEL_FTDo](https://drive.google.com/file/d/1eH8HAm0IaZhC2Sy-xCV_vxTMfpiyUMxH/view?usp=sharing)

Model trained on FTD_s: [MODEL_FTDs](https://drive.google.com/file/d/1YtAhkRSYzg42RZzGrqqlEOOhEPSByB1D/view?usp=sharing)

## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{zhang2023gmsf,
  title={GMSF: Global Matching Scene Flow},
  author={Zhang, Yushan and Edstedt, Johan and Wandt, Bastian and Forss{\'e}n, Per-Erik and Magnusson, Maria and Felsberg, Michael},
  journal={arXiv preprint arXiv:2305.17432},
  year={2023}
}
```