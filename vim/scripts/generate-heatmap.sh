#!/bin/bash
#SBATCH --job-name=generate-heatmap
#SBATCH --output=generate-heatmap.out
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module load cuda/11.8
source ~/.bashrc
conda activate mamba

python generate_heatmaps.py \
    --root /storage/scratch/6403840/data/imagenet-tiny/train \
    --output /storage/scratch/6403840/data/imagenet-tiny/train_heat \
    --method bing \
    --bing_training_path /storage/scratch/6403840/data/BING_models