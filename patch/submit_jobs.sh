#!/bin/bash

# Submit job with specific GPU type
sbatch --batch-size=128 --time=10-00:00:00 --gres=gpu:1 --mem=16G --cpus-per-task=4 
