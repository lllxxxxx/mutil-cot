#!/bin/bash


# 获取 num_gpus
NUM_GPUS=$(python -c "from src.config import TrainConfig; from transformers import HfArgumentParser; parser = HfArgumentParser((TrainConfig,)); cfg, = parser.parse_args_into_dataclasses(); print(cfg.num_gpus)")

echo "Starting Training with $NUM_GPUS GPUs..."


accelerate launch --num_processes $NUM_GPUS --multi_gpu --mixed_precision bf16 main.py \
    --template qwen \


echo "Training finished."