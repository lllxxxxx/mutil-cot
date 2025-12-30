#!/bin/bash

cd "$(dirname "$0")"
echo "Current working directory: $(pwd)"
export PYTHONPATH=$PYTHONPATH:$(pwd)

python scripts/gen_data.py