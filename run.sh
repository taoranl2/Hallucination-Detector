#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=80g
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0

conda activate dp

# python generate_dataset.py -o large_dataset.json -n 3000

python unified_experiments.py \
    --data_path large_dataset.json \
    --black_box \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --num_samples 500 \
    --force_retrain \
    --no_save
