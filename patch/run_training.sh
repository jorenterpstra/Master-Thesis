#!/bin/bash
#SBATCH --job-name=patch_scorer
#SBATCH --time=10-00:00:00

# Get least used GPU before loading any CUDA modules but print how much memory is used for all GPUs
GPU_ID=$(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | \
         sort -n | \
         head -n1 | \
         cut -d, -f2)

# Export GPU selection BEFORE loading modules
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Setting CUDA_VISIBLE_DEVICES=$GPU_ID"

# Load modules
module load cuda/11.8
source ~/.bashrc
conda activate vim

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

echo "GPU memory usage for all GPUs:"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv

echo "GPU memory usage for selected GPU $GPU_ID:"
nvidia-smi -i $GPU_ID

# Run the training script with GPU info
python main.py \
    --data-root /scratch/6403840/data/imagenet \
    --save-root /scratch/6403840/Master-Thesis/patch/runs/imagenet_training \
    --batch-size 60 \
    --num-workers 4 \
    --epochs 100 \
    --lr 0.001 \
    --plot-every 5

# Print completion info
echo "End time: $(date)"
nvidia-smi -i $GPU_ID