#!/bin/bash


# Load required modules (adjust according to your system)
module load cuda/11.7
module load python/3.9

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the training script
python main.py \
    --data-root /scratch/6403840/data/imagenet \
    --save-root runs/imagenet_training \
    --batch-size 32 \
    --num-workers 8 \
    --epochs 50 \
    --lr 0.001 \
    --plot-every 5
