#!/bin/bash

# Submit job with specific GPU type
sbatch --constraint="gpu_mem:6gb"--batch-size=128 run_training.sh

# Submit job with specific GPU type
sbatch --constraint="gpu_mem:8gb"--batch-size=256 run_training.sh