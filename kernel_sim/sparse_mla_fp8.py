import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.model_config import ModelConfig
from hardware.gpu import gpu_map

DEQUANT_CYCLES = {
    90: {
        "fp8_to_half": 1 / 64,
        "half_to_fp32": 1 / 64,
        "fp32_to_bf16": 1 / 16,
        "bf16_mul_fp32": 1 / 256,
    }
}


def sparse_mla_fp8(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    topk: int,
    dim: int,
    gpu_type: str,
    dim_rope: int,
) -> tuple[float, float]:
    gpu = gpu_map[gpu_type]

    BLOCK_M = 64
    dim_nope = dim

    compute_volume_flop = (
        batch_size
        * num_heads
        * seq_len
        * sum([2 * (dim_nope + dim_rope) * topk, 2 * topk * dim_nope])
    )

    num_heads_per_block = BLOCK_M if BLOCK_M < num_heads else num_heads

    time_mma_per_token = (
        num_heads_per_block
        * (dim_nope + dim_nope + dim_rope)
        * 2
        / gpu.fp16_tflops
        / 1e12
        * gpu.num_sm
    )

    cycle_time = 1 / gpu.frequency / 1e6
    if gpu.sm_version not in DEQUANT_CYCLES:
        raise ValueError(f"Unsupported SM version: {gpu.sm_version}")
    cycles = DEQUANT_CYCLES[gpu.sm_version]
    fp8_to_half_cycles = cycles["fp8_to_half"]
    half_to_fp32_cycles = cycles["half_to_fp32"]
    fp32_to_bf16_cycles = cycles["fp32_to_bf16"]
    bf16_mul_fp32_cycles = cycles["bf16_mul_fp32"]

    total_dequant_cycles = (
        fp8_to_half_cycles
        + half_to_fp32_cycles
        + fp32_to_bf16_cycles
        + bf16_mul_fp32_cycles
    )

    time_dequant_per_token = total_dequant_cycles * dim_nope * cycle_time

    QUANT_TILE_SIZE = 128
    SIZEOF_BF16 = 2
    SIZEOF_FP8 = 1

    time_load_scales = dim_nope / QUANT_TILE_SIZE * 4 / gpu.mem_bw / 1e9 * gpu.num_sm
    time_load_nope = dim_nope * SIZEOF_FP8 / gpu.mem_bw / 1e9 * gpu.num_sm
    time_load_rope = SIZEOF_BF16 * dim_rope / gpu.mem_bw / 1e9 * gpu.num_sm

    time_load_kv_per_token = time_load_scales + time_load_nope + time_load_rope

    time_load_and_dequant_per_token = time_load_kv_per_token + time_dequant_per_token

    time_mma_per_block = BLOCK_M * time_mma_per_token
    time_load_and_dequant_per_block = (BLOCK_M / 2) * time_load_and_dequant_per_token

    time_per_block = max(
        seq_len * time_load_and_dequant_per_block, seq_len * time_mma_per_block
    )

    sum_block = topk / BLOCK_M * batch_size
    num_block_per_sm_parts = (sum_block + gpu.num_sm - 1) / gpu.num_sm

    time = num_block_per_sm_parts * time_per_block * 2

    time_ms = time * 1000
    theoretical_max_tflops = compute_volume_flop / time / 1e12

    return time_ms, theoretical_max_tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse MLA FP8 simulation")
    parser.add_argument(
        "--seq_len", type=int, required=True, help="Sequence length for query"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "hf_configs", "deepseek_v3.2_config.json"
        ),
        help="Path to model configuration file (default: deepseek_v3.2_config.json)",
    )
    parser.add_argument(
        "--gpu_type", type=str, required=True, help="GPU type (e.g., H800, H20)"
    )

    args = parser.parse_args()

    config = ModelConfig(args.config_path)

    dim = config.kv_lora_rank
    num_heads = config.num_attention_heads
    topk = config.index_topk
    dim_rope = config.qk_rope_head_dim
    batch_size = 128

    time_ms, tflops = sparse_mla_fp8(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=args.seq_len,
        topk=topk,
        dim=dim,
        gpu_type=args.gpu_type,
        dim_rope=dim_rope,
    )
    print(f"Time: {time_ms:.6f} ms")
    print(f"TFLOPS: {tflops:.6f}")
