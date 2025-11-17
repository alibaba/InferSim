#!/bin/bash

python3 main.py --config-path hf_configs/qwen3-next-80B-A3B_config.json  \
  --device-type H20 \
  --world-size 4 \
  --decode-bs 256 \
  --use-fp8-gemm \
  --decode-only
