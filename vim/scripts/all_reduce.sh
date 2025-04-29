#!/bin/bash
#SBATCH --job-name=vim-extra-tiny
#SBATCH --time=10-00:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --gpu-freq=medium # Request medium priority GPU access

# Load modules
module load cuda/11.8
source ~/.bashrc
conda activate mamba

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
python -m torch.distributed.run \
    --nproc_per_node=2 \
    scripts/all_reduce_test.py