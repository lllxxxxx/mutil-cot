#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting Training..."

python main.py \
    --template qwen \


echo "Training finished."