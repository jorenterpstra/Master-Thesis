#!/bin/bash
#SBATCH --job-name=vim-extra-tiny
#SBATCH --time=10-00:00:00

# Get least used GPU before loading any CUDA modules but print how much memory is used for all GPUs
GPU_ID=$(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | \
         sort -n | \
         head -n1 | \
         cut -d, -f2)

# Export GPU selection BEFORE loading modules
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Setting CUDA_VISIBLE_DEVICES=$GPU_ID"
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "Setting MASTER_ADDR=$MASTER_ADDR"

# Load modules
module load cuda/11.8
source ~/.bashrc
conda activate mamba

torchrun \
    --nnodes=1 \
    main.py \
    --model vim_extra_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 16 \
    --data-path /storage/scratch/6403840/data/imagenet-tiny \
    --output_dir ./output/vim_extra_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --no_amp \
    --pin-mem \
    --mixup 0.0 \
    --cutmix 0.0 \
    # --debug 
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --batch-size 64 \
#     --drop-path 0.0 \
#     --epochs 100 \
#     --weight-decay 0.1 \
#     --num_workers 16\
#      --data-path /storage/scratch/6403840/data/imagenet \
#      --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#      --no_amp \
#      --pin-mem
