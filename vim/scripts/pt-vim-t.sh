#!/bin/bash
#conda activate vim;
#cd Master_thesis/vim;

# CUDA_VISIBLE_DEVICES=0,1,2,3 python \
#     -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --use_env main.py \
#     --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --batch-size 128 \
#     --drop-path 0.0 \
#     --weight-decay 0.1 \
#     --num_workers 25 \
#     --data-path /storage/scratch/6403840/data/imagenet \
#     --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
#     --no_amp
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 25\
     --data-path /storage/scratch/6403840/data/imagenet \
     --output_dir ./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
     --no_amp
