#!/bin/bash
#SBATCH --job-name=vim-extra-tiny
#SBATCH --time=10-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpu-freq=medium # Request medium priority GPU access

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export SLURM_PROCID=$SLURM_PROCID

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
# export NCCL_SOCKET_IFNAME=eno8303  # Use the network interface shown in previous logs
# export NCCL_IB_DISABLE=1          # Disable InfiniBand if not available
# export NCCL_P2P_DISABLE=0         # Enable P2P if available
# export NCCL_SHM_DISABLE=0         # Enable shared memory
# export NCCL_BLOCKING_WAIT=1       # Use blocking synchronization
# export NCCL_ASYNC_ERROR_HANDLING=1
# export PYTHONUNBUFFERED=1

# Select the GPUs with the least memory usage

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 main.py \
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
    --cutmix 0.0 \
    --debug

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
