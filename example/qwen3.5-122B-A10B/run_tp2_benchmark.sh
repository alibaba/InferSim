#!/bin/bash

# Qwen3.5-122B-A10B FP8 model TP=2 benchmark script
# This script generates benchmark data for TP=2 tensor parallel configuration

set -e

CONFIG_PATH="hf_configs/qwen3.5-122B-A10B_config.json"
DEVICE_TYPE="H20"
TP_SIZE=2

# Create output directories
mkdir -p bench_data/mha/decode/${DEVICE_TYPE,,}
mkdir -p bench_data/mha/prefill/${DEVICE_TYPE,,}
mkdir -p bench_data/grouped_gemm/decode/${DEVICE_TYPE,,}
mkdir -p bench_data/grouped_gemm/prefill/${DEVICE_TYPE,,}

echo "=============================================="
echo "Running TP=${TP_SIZE} benchmarks for ${DEVICE_TYPE}"
echo "=============================================="

# 1. MHA Decode benchmark (generates 16-1-256.csv for TP=2)
echo ""
echo "1. Running MHA Decode benchmark..."
python3 kernel_benchmark/flashinfer_mha_decode.py \
    --config-path ${CONFIG_PATH} \
    --tp-size ${TP_SIZE} \
    --kv-cache-dtype bf16 \
    --fp16-tflops 148

# Move and rename the output file
if [ -f "attention_benchmark.csv" ]; then
    mv attention_benchmark.csv bench_data/mha/decode/${DEVICE_TYPE,,}/16-1-256.csv
    echo "   Saved to: bench_data/mha/decode/${DEVICE_TYPE,,}/16-1-256.csv"
fi

# 2. MHA Prefill benchmark (generates 16-1-256.csv for TP=2)
echo ""
echo "2. Running MHA Prefill benchmark..."
python3 kernel_benchmark/fa3_mha_prefill.py \
    --config-path ${CONFIG_PATH} \
    --tp-size ${TP_SIZE} \
    --fp16-tflops 148

# Move and rename the output file
if [ -f "attention_benchmark_tp${TP_SIZE}.csv" ]; then
    mv attention_benchmark_tp${TP_SIZE}.csv bench_data/mha/prefill/${DEVICE_TYPE,,}/16-1-256.csv
    echo "   Saved to: bench_data/mha/prefill/${DEVICE_TYPE,,}/16-1-256.csv"
fi

# 3. Grouped GEMM Contiguous benchmark (Prefill)
echo ""
echo "3. Running Grouped GEMM Contiguous benchmark (Prefill)..."
python3 kernel_benchmark/deepgemm_grouped_gemm_contiguous.py \
    --config-path ${CONFIG_PATH} \
    --tp-size ${TP_SIZE} \
    --gpu-tflops 296

# Append to existing data or create new
if [ -f "groupedgemm_contiguous.csv" ]; then
    # Add header if file doesn't exist
    if [ ! -f "bench_data/grouped_gemm/prefill/${DEVICE_TYPE,,}/data.csv" ]; then
        cp groupedgemm_contiguous.csv bench_data/grouped_gemm/prefill/${DEVICE_TYPE,,}/data.csv
    else
        # Append without header
        tail -n +2 groupedgemm_contiguous.csv >> bench_data/grouped_gemm/prefill/${DEVICE_TYPE,,}/data.csv
    fi
    rm groupedgemm_contiguous.csv
    echo "   Appended to: bench_data/grouped_gemm/prefill/${DEVICE_TYPE,,}/data.csv"
fi

# 4. Grouped GEMM Masked benchmark (Decode)
echo ""
echo "4. Running Grouped GEMM Masked benchmark (Decode)..."
python3 kernel_benchmark/deepgemm_grouped_gemm_masked.py \
    --config-path ${CONFIG_PATH} \
    --tp-size ${TP_SIZE} \
    --gpu-tflops 296

# Append to existing data or create new
if [ -f "groupedgemm_masked.csv" ]; then
    # Add header if file doesn't exist
    if [ ! -f "bench_data/grouped_gemm/decode/${DEVICE_TYPE,,}/data.csv" ]; then
        cp groupedgemm_masked.csv bench_data/grouped_gemm/decode/${DEVICE_TYPE,,}/data.csv
    else
        # Append without header
        tail -n +2 groupedgemm_masked.csv >> bench_data/grouped_gemm/decode/${DEVICE_TYPE,,}/data.csv
    fi
    rm groupedgemm_masked.csv
    echo "   Appended to: bench_data/grouped_gemm/decode/${DEVICE_TYPE,,}/data.csv"
fi

echo ""
echo "=============================================="
echo "TP=${TP_SIZE} benchmark completed!"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  - bench_data/mha/decode/${DEVICE_TYPE,,}/16-1-256.csv"
echo "  - bench_data/mha/prefill/${DEVICE_TYPE,,}/16-1-256.csv"
echo "  - bench_data/grouped_gemm/decode/${DEVICE_TYPE,,}/data.csv (updated)"
echo "  - bench_data/grouped_gemm/prefill/${DEVICE_TYPE,,}/data.csv (updated)"
echo ""
echo "You can now run the simulation scripts:"
echo "  bash example/qwen3.5-122B-A10B/decode.sh"
echo "  bash example/qwen3.5-122B-A10B/prefill.sh"
