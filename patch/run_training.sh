#!/bin/bash
#SBATCH --job-name=patch_scorer
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-gpu=6G

# Get least used GPU before loading any CUDA modules
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

# Run the training script with GPU info
python main.py \
    --data-root /scratch/6403840/data/imagenet \
    --save-root /scratch/6403840/Master-Thesis/patch/runs/imagenet_training \
    --batch-size 256 \
    --num-workers 8 \
    --epochs 100 \
    --lr 0.001 \
    --plot-every 5

# Print completion info
echo "End time: $(date)"
nvidia-smi -i $GPU_ID