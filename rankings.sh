#!/bin/bash
#SBATCH --job-name=get-rankings
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

python patch_rankings.py \
    --data-path /storage/scratch/6403840/data/imagenet-tiny/val \
    --heatmap-path /storage/scratch/6403840/data/imagenet-tiny/val_heat \
    --output-csv /storage/scratch/6403840/data/imagenet-tiny/val_rankings.csv \
    --split val