#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=80g
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0

conda activate dp

# python generate_truthfulqa_dataset.py
python unified_experiments.py
