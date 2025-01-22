#!/bin/bash
#SBATCH --job-name=patch_scorer
#SBATCH --output=/scratch/6403840/Master-Thesis/patch/logs/%j_out.log    # %j is replaced by the job ID
#SBATCH --error=/scratch/6403840/Master-Thesis/patch/logs/%j_err.log     # Separate file for errors
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB

# Create logs directory if it doesn't exist
mkdir -p /scratch/6403840/Master-Thesis/patch/logs/

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load required modules (adjust according to your system)
module load cuda/11.8
module load python/3.10

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the training script
python main.py \
    --data-root /scratch/6403840/data/imagenet \
    --save-root /scratch/6403840/Master-Thesis/patch/runs/imagenet_training \
    --batch-size 32 \
    --num-workers 8 \
    --epochs 50 \
    --lr 0.001 \
    --plot-every 5

# Print completion time
echo "End time: $(date)"
