#!/bin/bash
#SBATCH --job-name=vim-extra-tiny
#SBATCH --time=10-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpu-freq=medium # Request medium priority GPU access

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

echo "Setting up distributed environment:"
echo "- MASTER_ADDR=$MASTER_ADDR"
echo "- MASTER_PORT=$MASTER_PORT"
echo "- SLURM environment: SLURM_PROCID=$SLURM_PROCID, SLURM_LOCALID=$SLURM_LOCALID"

# Load modules
module load cuda/11.8
source ~/.bashrc
conda activate mamba

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export PYTHONUNBUFFERED=1

# Select the GPUs with the least memory usage

srun --ntasks=2 --nodes=1 --gpus-per-task=1 --export=ALL\
    python main.py \
    --data-set IMNET \
    --model vim_extra_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 16 \
    --data-path /storage/scratch/6403840/data/imagenet-tiny \
    --output_dir ./output/vim_extra_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_no_spatial_transforms \
    --no_amp \
    --pin-mem \
    --mixup 0.0 \
    --cutmix 0.0 

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
