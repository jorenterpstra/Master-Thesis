#!/bin/bash
#SBATCH --job-name=patch_scorer
#SBATCH --time=10:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Get GPU information
nvidia-smi

# Print available GPUs and their memory
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# Get least used GPU - fixed version
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | \
         awk -F, '{print $1 " " $2}' | \
         sort -k2 -n | \
         head -n1 | \
         cut -d' ' -f1)

echo "Selected GPU $GPU_ID with lowest memory usage"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Verify GPU selection
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
nvidia-smi -i $GPU_ID

# Load modules
module load cuda/11.8

# Run the training script with GPU info
python main.py \
    --data-root /scratch/6403840/data/imagenet \
    --save-root /scratch/6403840/Master-Thesis/patch/runs/imagenet_training \
    --batch-size 32 \
    --num-workers 8 \
    --epochs 100 \
    --lr 0.001 \
    --plot-every 5

# Print completion info
echo "End time: $(date)"
echo "Final GPU state:"
nvidia-smi -i $GPU_ID
