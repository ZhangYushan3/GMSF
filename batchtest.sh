#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-450 -p alvis
#SBATCH -t 0-03:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --mail-user=yushan.zhang@liu.se --mail-type=end
#SBATCH --job-name=sceneflow

source /mimer/NOBACKUP/groups/alvis_cvl/yushanz/miniconda3/bin/activate sceneflow
cd /mimer/NOBACKUP/groups/alvis_cvl/yushanz/scene_flow/gmsf

python main_gmsf.py --resume checkpoints/DGCNN_lr4e-4_t8/step_200000.pth \
    --backbone DGCNN --stage things_subset \
    --num_transformer_pt_layers 1 \
    --num_transformer_layers 8 \
    --eval