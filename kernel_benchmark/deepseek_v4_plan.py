"""Generate a GPU-independent kernel benchmark manifest for DeepSeek-V4.

The script intentionally only describes cases.  V4 MQA with compressed KV is
not MLA, so borrowing DeepSeek-V3 MFU data would produce misleading estimates.
"""

import argparse
import json
import os
import sys
from collections import Counter

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from config.model_config import ModelConfig  # noqa: E402


DECODE_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
PREFILL_TOKEN_COUNTS = [1024, 4096, 8192, 16384, 32768]
KV_LENGTHS = [1024, 4096, 8192, 16384, 32768, 65536, 131072]


def build_manifest(config: ModelConfig, world_sizes=(1, 2, 4, 8)):
    """Return every shape that needs a measured V4 kernel result."""
    if not config.is_deepseek_v4:
        raise ValueError("The benchmark planner only accepts model_type=deepseek_v4")

    ratio_counts = Counter(config.compress_ratios)
    output_input_dim = config.num_attention_heads * config.head_dim
    output_group_input_dim = output_input_dim // config.num_output_groups
    return {
        "model_type": config.model_type,
        "attention": {
            "kernel": "dsv4_compressed_mqa",
            "num_q_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "q_lora_rank": config.q_lora_rank,
            "o_lora_rank": config.o_lora_rank,
            "output_groups": config.num_output_groups,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "index_topk": config.index_topk,
            "sliding_window": config.sliding_window,
            "compression_layers": {
                str(ratio): ratio_counts[ratio] for ratio in sorted(ratio_counts)
            },
            "decode_batch_sizes": DECODE_BATCH_SIZES,
            "prefill_token_counts": PREFILL_TOKEN_COUNTS,
            "kv_lengths": KV_LENGTHS,
        },
        "dense_gemm": [
            {"name": "q_a", "k": config.hidden_size, "n": config.q_lora_rank},
            {"name": "q_b", "k": config.q_lora_rank, "n": output_input_dim},
            {"name": "wkv", "k": config.hidden_size, "n": config.head_dim},
            {
                "name": "o_a",
                "k": output_group_input_dim,
                "n": config.num_output_groups * config.o_lora_rank,
            },
            {
                "name": "o_b",
                "k": config.num_output_groups * config.o_lora_rank,
                "n": config.hidden_size,
            },
        ],
        "grouped_gemm": {
            "num_experts": config.num_routed_experts,
            "topk": config.num_experts_per_tok,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "world_sizes": [
                size
                for size in world_sizes
                if config.num_routed_experts % size == 0
            ],
            "decode_batch_sizes": DECODE_BATCH_SIZES,
            "prefill_token_counts": PREFILL_TOKEN_COUNTS,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, help="HF model config.json")
    parser.add_argument(
        "--world-sizes",
        default="1,2,4,8",
        help="Comma-separated EP world sizes to include in grouped-GEMM cases",
    )
    args = parser.parse_args()
    world_sizes = tuple(int(size) for size in args.world_sizes.split(",") if size)
    manifest = build_manifest(ModelConfig(args.config_path), world_sizes)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
