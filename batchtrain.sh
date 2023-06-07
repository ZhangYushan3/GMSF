#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-450 -p alvis
#SBATCH -t 3-00:00:00
#SBATCH --gpus-per-node=A100fat:4
#SBATCH --mail-user=yushan.zhang@liu.se --mail-type=end
#SBATCH --job-name=sceneflow

source /mimer/NOBACKUP/groups/alvis_cvl/yushanz/miniconda3/bin/activate sceneflow
cd /mimer/NOBACKUP/groups/alvis_cvl/yushanz/scene_flow/gmsf

python main_gmsf.py \
    --checkpoint_dir checkpoints/DGCNN_lr2e-4_t16 \
    --backbone DGCNN --stage things_subset \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 16 \
    --feature_channels_point 128 \
    --lr 2e-4 --batch_size 8 --num_steps 600000 \
    --resume checkpoints/DGCNN_lr2e-4_t16/checkpoint_latest.pth