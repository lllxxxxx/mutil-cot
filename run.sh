#!/bin/bash

# Kill any existing python processes using the port (cleanup from previous failed runs)
# Uncomment if needed: pkill -f "python.*main.py" 2>/dev/null || true

# 获取 num_gpus
NUM_GPUS=$(python -c "from src.config import TrainConfig; from transformers import HfArgumentParser; parser = HfArgumentParser((TrainConfig,)); cfg, = parser.parse_args_into_dataclasses(); print(cfg.num_gpus)")

echo "Starting Training with $NUM_GPUS GPUs..."

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable torch.compile to avoid recompilation on different sequence lengths
export TORCH_COMPILE_DISABLE=1

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    --mixed_precision bf16 \
    --main_process_port 0 \
    main.py \

echo "Training finished."