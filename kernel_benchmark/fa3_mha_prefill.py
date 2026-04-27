# Adapt from https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py
import argparse
import os
import sys

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from sgl_kernel.flash_attn import flash_attn_varlen_func

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from config.model_config import ModelConfig  # noqa E402
from flops.flops import get_mha_gflops  # noqa E402


def benchmark_forward(
    fn,
    *inputs,
    repeats=10,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    return t, m


def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean * 1e6


def prefill_attention_fa3():
    class Fa3Attention(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            num_q_heads,
            num_kv_heads,
            head_dim,
            warmup=10,
        ):
            seq_len = q.shape[0]
            scale = head_dim**-0.5
            # print(f"seq_len:{seq_len}, scale:{scale}")
            qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

            print(f"q:{q.shape}, k:{k.shape}, v:{v.shape} {q.dtype}")
            flash_attn_varlen_func(
                q=q.view(-1, num_q_heads, head_dim),
                k=k.view(-1, num_kv_heads, head_dim).to(q.dtype),
                v=v.view(-1, num_kv_heads, head_dim).to(q.dtype),
                cu_seqlens_q=qo_indptr,
                cu_seqlens_k=qo_indptr,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                softmax_scale=scale,
                causal=True,
                return_softmax_lse=True,
            )

            for _ in range(warmup):
                o = flash_attn_varlen_func(
                    q=q.view(-1, num_q_heads, head_dim),
                    k=k.view(-1, num_kv_heads, head_dim).to(q.dtype),
                    v=v.view(-1, num_kv_heads, head_dim).to(q.dtype),
                    cu_seqlens_q=qo_indptr,
                    cu_seqlens_k=qo_indptr,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                    softmax_scale=scale,
                    causal=True,
                    return_softmax_lse=True,
                )

            f = time_fwd(
                flash_attn_varlen_func,
                q=q.view(-1, num_q_heads, head_dim),
                k=k.view(-1, num_kv_heads, head_dim).to(q.dtype),
                v=v.view(-1, num_kv_heads, head_dim).to(q.dtype),
                cu_seqlens_q=qo_indptr,
                cu_seqlens_k=qo_indptr,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                softmax_scale=scale,
                causal=True,
                return_softmax_lse=True,
            )

            return f, o

    return Fa3Attention


def main(args):
    config = ModelConfig(args.config_path, tp_size=args.tp_size)
    fp16_tflops = args.fp16_tflops
    head_dim = config.head_dim
    # Use TP-aware head counts
    num_q_heads = config.num_attention_heads // args.tp_size
    num_kv_heads = config.num_key_value_heads // args.tp_size
    if num_q_heads == num_kv_heads:
        attn_type = "MHA"
    else:
        attn_type = "GQA"
    dtype = torch.bfloat16

    seq_lens = [1024, 4096, 8192, 16384, 32768]
    results = []

    attn_fa3 = prefill_attention_fa3().apply
    for seq_len in seq_lens:
        q = torch.randn(seq_len, num_q_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")

        attn_core_gflops, other_gflops = get_mha_gflops(config, 1, seq_len)
        # Adjust GFLOPs for TP
        attn_core_gflops = attn_core_gflops * seq_len / 2

        us_fa3, _ = attn_fa3(q, k, v, num_q_heads, num_kv_heads, head_dim)
        mfu = attn_core_gflops * 1e3 / (fp16_tflops * us_fa3)
        print(
            attn_type,
            "  ",
            num_q_heads,
            "  ",
            num_kv_heads,
            "  ",
            head_dim,
            "  ",
            seq_len,
            "  ",
            us_fa3,
            "  ",
            mfu,
        )

        results.append(
            {
                "dtype": "bf16",
                "num_attention_heads": num_q_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "tp_size": args.tp_size,
                "seq_len": seq_len,
                "latency_us": round(us_fa3, 3),
                "mfu": round(mfu, 3),
            }
        )

    df = pd.DataFrame(results)
    output_file = f"attention_benchmark_tp{args.tp_size}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # calculate_diff()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )
    # Tensor Parallel size
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--fp16-tflops", type=int, default=148, help="GPU FP16 TFLOPS"
    )

    args = parser.parse_args()
    main(args)
