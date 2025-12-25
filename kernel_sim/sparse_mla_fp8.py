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

config_path = os.path.join(
    os.path.dirname(__file__), "..", "hf_configs", "deepseek_v3.2_config.json"
)
config = ModelConfig(config_path)


def sparse_mla_fp8(batch_size, num_heads, s_q, topk, dim, gpu_type):
    gpu = gpu_map[gpu_type]

    block_M = 64
    dim_rope = config.qk_rope_head_dim
    dim_nope = dim

    compute_volume_flop = (
        batch_size
        * num_heads
        * s_q
        * sum([2 * (dim_nope + dim_rope) * topk, 2 * topk * dim_nope])
    )

    num_heads_per_block = block_M if block_M < num_heads else num_heads

    t_MMA_per_token = (
        num_heads_per_block
        * (dim_nope + dim_nope + dim_rope)
        * 2
        / gpu.fp16_tflops
        / 1e12
        * gpu.num_sm
    )

    cycle_time = 1 / gpu.frequency / 1e6

    cycles = DEQUANT_CYCLES[gpu.compute_capability]
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

    t_dequant_per_token = total_dequant_cycles * dim_nope * cycle_time

    quant_tile_size = 128
    sizeof_fp8 = 1
    sizeof_bf16 = 2

    t_load_scales = (dim_nope) / quant_tile_size * 4 / gpu.mem_bw / 1e9 * gpu.num_sm
    t_load_nope = dim_nope * sizeof_fp8 / gpu.mem_bw / 1e9 * gpu.num_sm
    t_load_rope = sizeof_bf16 * dim_rope / gpu.mem_bw / 1e9 * gpu.num_sm

    t_load_KV_per_token = t_load_scales + t_load_nope + t_load_rope

    t_LKAD_per_token = t_load_KV_per_token + t_dequant_per_token

    t_MMA_per_block = block_M * t_MMA_per_token
    t_LKAD_per_block = (block_M // 2) * t_LKAD_per_token

    t_per_block = (
        s_q * t_LKAD_per_block
        if t_LKAD_per_block > t_MMA_per_block
        else s_q * t_MMA_per_block
    )

    sum_block = topk / block_M * batch_size
    num_block_per_SM_parts = (sum_block + gpu.num_sm - 1) // gpu.num_sm

    time = num_block_per_SM_parts * t_per_block * 2

    time_usage = time * 1000
    theoretical_max_tflops = compute_volume_flop / time / 1e12

    return time_usage, theoretical_max_tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse MLA FP8 simulation")
    parser.add_argument(
        "--s_q", type=int, required=True, help="Sequence length for query"
    )
    # parser.add_argument("--topk", type=int, required=True, help="Top-k value")
    parser.add_argument(
        "--gpu_type", type=str, required=True, help="GPU type (e.g., H800, H20)"
    )

    args = parser.parse_args()

    dim = config.kv_lora_rank
    num_heads = config.num_attention_heads
    topk = config.index_topk
    batch_size = 128

    time_ms, tflops = sparse_mla_fp8(
        batch_size=batch_size,
        num_heads=num_heads,
        s_q=args.s_q,
        topk=topk,
        dim=dim,
        gpu_type=args.gpu_type,
    )
    print(f"Time: {time_ms:.6f} ms")
    print(f"TFLOPS: {tflops:.6f}")
