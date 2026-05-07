#!/bin/bash

# qwen3.5-122B-A10B FP8 model with TP=2 on H20 GPU
# Prefill simulation

python3 main.py --config-path hf_configs/qwen3.5-122B-A10B_config.json \
  --device-type H20 \
  --world-size 2 \
  --tp-size 2 \
  --use-fp8-gemm \
  --max-prefill-tokens 4096 \
  --prefill-only
