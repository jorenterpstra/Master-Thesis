#!/bin/bash
#SBATCH --job-name=vim-extra-tiny
#SBATCH --time=10-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpu-freq=high # Request medium priority GPU access

#––– error‐trap boilerplate –––
set -o errexit       # exit on any error
set -o errtrace      # ensure ERR trap is inherited by functions, subshells

dump_gpus_and_procs() {
  echo
  echo "=================== GPU STATUS AT FAILURE ==================="
  date
  echo "--- nvidia-smi summary ---"
  nvidia-smi
  echo
  echo "--- compute-apps (pid, user, gpu_mem) ---"
  nvidia-smi --query-compute-apps=pid,username,used_memory --format=csv,noheader,nounits
  echo
  echo "--- all python / cuda processes ---"
  ps -eo pid,user,etime,cmd | grep -E 'python|cuda' || true
  echo "============================================================="
}

trap 'echo "!!! ERROR detected, dumping GPU & process info !!!"; dump_gpus_and_procs' ERR
#––––––––––––––––––––––––––––––––––

# Load modules
module load cuda/11.8
source ~/.bashrc
conda activate mamba

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=0
# export NCCL_BLOCKING_WAIT=1       # Use blocking synchronization
# export NCCL_ASYNC_ERROR_HANDLING=1
# export PYTHONUNBUFFERED=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501
# export CUDA_VISIBLE_DEVICES=2,3

# Minimum required memory in MB
REQUIRED_MEM=12480
NUM_GPUS=2

echo "Looking for $NUM_GPUS GPUs with at least $REQUIRED_MEM MB free..."

while true; do
    # Get free memory for all GPUs
    mapfile -t FREES < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    # Check how many GPUs meet the memory requirement
    GOOD_GPUS=()
    for i in "${!FREES[@]}"; do
        if [ "${FREES[$i]}" -ge "$REQUIRED_MEM" ]; then
            GOOD_GPUS+=("$i")
        fi
    done

    if [ "${#GOOD_GPUS[@]}" -ge "$NUM_GPUS" ]; then
        echo "Found ${#GOOD_GPUS[@]} suitable GPUs: ${GOOD_GPUS[*]}"
        # Set only those GPUs visible to your training script
        export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GOOD_GPUS[*]::NUM_GPUS}")
        echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
        break
    else
        echo "Only ${#GOOD_GPUS[@]} suitable GPUs found, retrying in 60s..."
        sleep 60
    fi
done

# torchrun will set RANK, LOCAL_RANK, WORLD_SIZE, etc.
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --data-set IMNET_HEAT \
    --model vim_extra_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --drop-path 0.0 \
    --weight-decay 0.1 \
    --num_workers 16 \
    --data-path /storage/scratch/6403840/data/imagenet-tiny \
    --heatmap-path /storage/scratch/6403840/data/imagenet-tiny \
    --return-rankings \
    --output_dir ./output/vim_extra_tiny_custom_transforms_random \
    --if_random_token_rank \
    --no_amp \
    --pin-mem \
    --mixup 0.0 \
    --cutmix 0.0 \

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
