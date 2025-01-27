#!/bin/bash

# Configuration
NUM_GPUS=4
BATCH_SIZE=32
NUM_WORKERS=8
EPOCHS=50
LEARNING_RATE=0.001
DATA_ROOT="/scratch/6403840/data/imagenet"
SAVE_ROOT="runs/imagenet_training"

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    main.py \
    --distributed \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --data-root $DATA_ROOT \
    --save-root $SAVE_ROOT
