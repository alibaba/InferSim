#!/bin/bash

python3 main.py --config-path hf_configs/qwen3.5-35B-A3B_config.json \
  --device-type H20 --world-size 1 \
  --use-fp8-gemm \
  --max-prefill-tokens 4096 \
  --prefill-only
