# Adapt from https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py
import argparse
import os
import sys

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from flashinfer import BatchDecodeWithPagedKVCacheWrapper
from sglang.srt.layers.attention.flashinfer_backend import \
    should_use_tensor_core

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


def decode_attention_flashinfer(kv_cache_dtype, num_attention_heads, num_kv_heads):
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    use_tensor_cores = should_use_tensor_core(
        kv_cache_dtype=kv_cache_dtype,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
    )
    # use_tensor_cores = False
    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores
    )

    class FlashinferAttention(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q,
            kv_data,
            batch_size,
            kv_len,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            q_type,
            kv_type,
            sm_scale=None,
            logits_soft_cap=0.0,
            k_scale=None,
            v_scale=None,
            warmup=10,
        ):
            total_tokens = batch_size * kv_len
            kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
            kv_indices = torch.arange(0, total_tokens).to(0).int()
            kv_last_page_len = torch.full(
                (batch_size,), 1, dtype=torch.int32, device="cuda"
            )

            flashinfer_decode_wrapper.end_forward()
            flashinfer_decode_wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                1,
                pos_encoding_mode="NONE",
                q_data_type=q_type,
                kv_data_type=kv_type,
            )

            for _ in range(warmup):
                o = flashinfer_decode_wrapper.forward(
                    q.contiguous().view(-1, num_attention_heads, head_dim),
                    kv_data,
                    sm_scale=sm_scale,
                    logits_soft_cap=logits_soft_cap,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

            f = time_fwd(
                flashinfer_decode_wrapper.forward,
                q.contiguous().view(-1, num_attention_heads, head_dim),
                kv_data,
                sm_scale=sm_scale,
                logits_soft_cap=logits_soft_cap,
                k_scale=k_scale,
                v_scale=v_scale,
            )

            return f, o

    return FlashinferAttention


def main(args):
    config = ModelConfig(args.config_path)
    fp16_tflops = args.fp16_tflops
    head_dim = config.head_dim
    num_attention_heads = config.num_attention_heads // args.tp_size
    num_kv_heads = config.num_key_value_heads // args.tp_size
    if num_kv_heads == num_attention_heads:
        attn_type = "MHA"
    else:
        attn_type = "GQA"

    dtype = torch.bfloat16
    kv_cache_dtype = dtype
    if args.kv_cache_dtype == "fp8":
        kv_cache_dtype = torch.float8_e4m3fn

    batch_kv_mapping = {
        1: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        16: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        32: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        64: [1024, 4096, 8192, 16384, 32768, 65536, 131072],
        128: [1024, 4096, 8192, 16384, 32768, 65536],
        256: [1024, 4096, 8192, 16384],
        512: [1024, 4096, 8192],
    }

    configs = []
    results = []
    for batch_size, kv_len_range in batch_kv_mapping.items():
        configs.extend([(batch_size, kv_len) for kv_len in kv_len_range])

    attn_flashinfer = decode_attention_flashinfer(
        kv_cache_dtype, num_attention_heads, num_kv_heads
    ).apply
    for batch_size, kv_len in configs:
        q = torch.randn(
            batch_size, num_attention_heads, head_dim, dtype=dtype, device="cuda"
        )
        kv_data = (
            torch.randn(
                batch_size * kv_len,
                num_kv_heads,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            ).to(kv_cache_dtype),
            torch.randn(
                batch_size * kv_len,
                num_kv_heads,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            ).to(kv_cache_dtype),
        )
        attn_core_gflops, other_gflops = get_mha_gflops(config, 1, kv_len)
        attn_core_gflops = attn_core_gflops * batch_size / args.tp_size

        sm_scale = 1.0 / (head_dim ** 0.5)
        us_flashinfer, _ = attn_flashinfer(
            q,
            kv_data,
            batch_size,
            kv_len,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            dtype,
            kv_cache_dtype,
            sm_scale,
            0.0,
            None,
            None,
        )
        mfu = attn_core_gflops * 1e3 / (fp16_tflops * us_flashinfer)
        print(
            attn_type,
            "  ",
            num_attention_heads,
            "  ",
            num_kv_heads,
            "  ",
            head_dim,
            "  ",
            batch_size,
            "  ",
            kv_len,
            "  ",
            us_flashinfer,
            "  ",
            mfu,
        )

        results.append(
            {
                "dtype": "bf16",
                "kv_dtype": args.kv_cache_dtype,
                "batch_size": batch_size,
                "kv_len": kv_len,
                "latency_us": round(us_flashinfer, 3),
                "mfu": round(mfu, 3),
            }
        )

    df = pd.DataFrame(results)
    print("Writing result into attention_benchmark.csv...")
    df.to_csv("attention_benchmark.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )

    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bf16", "fp8"],
        default="bf16",
        help="dtype of KV Cache,choices: bf16, fp8",
        required=False,
    )
    parser.add_argument("--tp-size", type=int, default=1, help="tp size")
    parser.add_argument(
        "--fp16-tflops", type=int, default=148, help="GPU FP16 TFLOPS size"
    )

    args = parser.parse_args()
    main(args)
