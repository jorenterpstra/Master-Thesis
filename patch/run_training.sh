#!/bin/bash
#SBATCH --job-name=patch_scorer
#SBATCH --time=10:00:00

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load required modules (adjust according to your system)
module load cuda/11.8

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the training script
python main.py \
    --data-root /scratch/6403840/data/imagenet \
    --save-root /scratch/6403840/Master-Thesis/patch/runs/imagenet_training \
    --batch-size 32 \
    --num-workers 8 \
    --epochs 100 \
    --lr 0.001 \
    --plot-every 5

# Print completion time
echo "End time: $(date)"
