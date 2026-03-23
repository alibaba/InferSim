#!/bin/bash

# qwen3.5-122B-A10B FP8 model with TP=2 on H20 GPU
# Decode simulation

python3 main.py --config-path hf_configs/qwen3.5-122B-A10B_config.json \
  --device-type H20 \
  --world-size 2 \
  --tp-size 2 \
  --decode-bs 168 \
  --use-fp8-gemm \
  --decode-only
