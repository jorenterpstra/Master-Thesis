#!/bin/bash
#conda activate vim;
#cd Master_thesis/vim;

CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nnodes=1 \
    --nproc-per-node=4\
    --max-restarts=3 \
    main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 32 \
    --data-path /storage/scratch/6403840/data/imagenet_subset \
    --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2_subset \
    --no_amp \
    --pin-mem \
    --mixup 0.0 \
    --cutmix 0.0 \
    --debug \
    --rdzv_endpoint=localhost:29400

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
