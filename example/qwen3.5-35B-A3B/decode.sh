#!/bin/bash

python3 main.py --config-path hf_configs/qwen3.5-35B-A3B_config.json  \
  --device-type H20 \
  --world-size 1 \
  --decode-bs 224 \
  --use-fp8-gemm \
  --decode-only
