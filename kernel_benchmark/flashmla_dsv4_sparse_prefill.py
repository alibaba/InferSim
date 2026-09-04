"""Benchmark the sparse-prefill FlashMLA API used by DeepSeek-V4.

The benchmark does not reimplement attention.  It constructs a V4-shaped
workspace and calls the same ``flash_mla_sparse_fwd`` and index-combiner APIs
as SGLang's DeepSeek-V4 attention backend.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

CSV_FIELDS = ("dtype", "s_q", "s_kv", "latency_us", "mfu")
DSV4_ATTENTION_HEAD_DIM = 512


def load_shape(config_path: Path, tp_size: int) -> dict[str, int]:
    if config_path.is_dir():
        config_path = config_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    if num_heads % tp_size:
        raise ValueError("num_attention_heads must be divisible by tp_size")
    if head_dim != DSV4_ATTENTION_HEAD_DIM:
        raise ValueError(
            "This benchmark targets DeepSeek-V4 with a 512-wide attention "
            f"head, but config head_dim is {head_dim}"
        )
    return {
        "num_heads": num_heads // tp_size,
        # DeepSeek-V4 head_dim is the complete QK width: 448 NoPE + 64 RoPE.
        # FlashMLA sparse prefill receives this full width, not only RoPE.
        "head_dim": head_dim,
        "value_dim": head_dim,
        "sliding_window": int(config.get("sliding_window", 128)),
        "index_topk": int(config.get("index_topk", 1024)),
    }


def median_latency_us(run, warmup: int, repeats: int) -> float:
    import torch

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--ratio", type=int, choices=(4, 128), required=True)
    parser.add_argument("--q-len", type=int, default=4096)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--bf16-tflops", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if min(args.q_len, args.tp_size, args.repeats) <= 0 or args.warmup < 0:
        parser.error("q-len, tp-size and repeats must be positive")
    if args.bf16_tflops <= 0:
        parser.error("bf16-tflops must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    # These are the APIs called by DeepseekV4AttnBackend._forward_prefill_sparse.
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        combine_topk_swa_indices,
    )

    shape = load_shape(args.config_path, args.tp_size)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    q_len = args.q_len
    compressed_len = (q_len + args.ratio - 1) // args.ratio
    compressed_topk = (
        min(shape["index_topk"], compressed_len) if args.ratio == 4 else compressed_len
    )

    q = torch.randn(
        (q_len, shape["num_heads"], shape["head_dim"]),
        dtype=torch.bfloat16,
        device=device,
    )
    # SGLang dequantizes compressed KV followed by SWA KV into this workspace.
    kv = torch.randn(
        (compressed_len + q_len, 1, shape["head_dim"]),
        dtype=torch.bfloat16,
        device=device,
    )
    if q.shape[-1] != DSV4_ATTENTION_HEAD_DIM:
        raise AssertionError(f"unexpected Q head dimension: {q.shape[-1]}")
    if kv.shape[-1] != DSV4_ATTENTION_HEAD_DIM:
        raise AssertionError(f"unexpected KV head dimension: {kv.shape[-1]}")

    # Entries beyond each query's causal valid length are ignored by the
    # combiner.  The deterministic pattern keeps every consumed index valid.
    topk_indices = (
        torch.arange(compressed_topk, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(q_len, -1)
        .contiguous()
    )
    combined_indices, topk_length = combine_topk_swa_indices(
        topk_indices=topk_indices,
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([q_len], dtype=torch.int32, device=device),
        gather_lens=torch.tensor([q_len], dtype=torch.int32, device=device),
        compressed_base=torch.zeros(1, dtype=torch.int32, device=device),
        swa_base=torch.tensor([compressed_len], dtype=torch.int32, device=device),
        window_size=shape["sliding_window"],
        compress_ratio=args.ratio,
        topk=compressed_topk,
    )
    indices = combined_indices.unsqueeze(1)
    attn_sink = torch.zeros(shape["num_heads"], dtype=torch.float32, device=device)

    def run():
        return flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=shape["head_dim"] ** -0.5,
            d_v=shape["value_dim"],
            attn_sink=attn_sink,
            topk_length=topk_length,
        )

    # Trigger JIT compilation/autotuning before collecting samples.
    run()
    torch.cuda.synchronize()
    latency_us = median_latency_us(run, args.warmup, args.repeats)

    mean_keys = float(topk_length.float().mean().item())
    # QK and PV each contribute 2 FLOPs per multiply-accumulate:
    #   4 * queries * local_heads * compute_head_dim * effective_keys.
    # For attention TP2, local_heads = 128 / 2 = 64. For TP8 it is 16.
    flops = (
        4
        * q_len
        * shape["num_heads"]
        * shape["head_dim"]
        * mean_keys
    )
    achieved_tflops = flops / latency_us / 1e6
    # Follow the existing bench_data attention schema. s_kv is the physical
    # KV workspace length. The file name records local heads, the 512-element
    # compute dimension and the compression group, for example
    # attn-64-512-c4.csv for attention TP2.
    row = {
        "dtype": "bf16",
        "s_q": q_len,
        "s_kv": kv.shape[0],
        "latency_us": round(latency_us, 3),
        "mfu": round(achieved_tflops / args.bf16_tflops, 3),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(
        f"ratio={args.ratio}, mean_keys_per_query={mean_keys:.3f}, "
        f"achieved_tflops={achieved_tflops:.6f}"
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
