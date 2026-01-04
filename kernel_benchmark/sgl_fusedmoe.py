import argparse
import os
import sys

import torch
import triton
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe as fused_moe_sglang,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from config.model_config import ModelConfig  # noqa E402


def fused_moe_sglang_api(
    x,
    w1,
    w2,
    input_gating,
    topk,
    use_fp8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    block_shape=None,
):
    topk_output = select_experts(
        hidden_states=x,
        router_logits=input_gating,
        topk_config=TopKConfig(top_k=topk, renormalize=False),
    )
    return fused_moe_sglang(
        x,
        w1,
        w2,
        topk_output,
        use_fp8_w8a8=use_fp8_w8a8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=list([128, 256, 512, 1024, 2048, 4096, 8192]),
        line_arg="provider",
        line_vals=[
            "sglang_fused_moe_triton",
        ],
        line_names=[
            "sglang_fused_moe_triton",
        ],
        styles=[
            ("blue", "-"),
            ("green", "-"),
        ],
        ylabel="Time (ms)",
        plot_name="fused-moe-performance",
        args={},
    )
)
def benchmark(
    batch_size,
    provider,
    model_config,
    tp_size,
    ep_size,
    use_fp8_w8a8=False,
    use_cuda_graph: bool = False,
):
    print(f"benchmark {provider} with batch_size={batch_size}")
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    num_tokens = batch_size
    num_experts = model_config.num_routed_experts // ep_size
    hidden_size = model_config.hidden_size
    shard_intermediate_size = model_config.intermediate_size // tp_size
    topk = model_config.num_experts_per_tok
    dtype = torch.bfloat16
    block_shape = [128, 128]

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    w1 = torch.randn(num_experts, shard_intermediate_size * 2, hidden_size, dtype=dtype)
    w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size, dtype=dtype)

    w1_tri = w1.clone()
    w2_tri = w2.clone()
    w1_tri = w1_tri.transpose(-2, -1).contiguous()
    w2_tri = w2_tri.transpose(-2, -1).contiguous()

    input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32)

    if provider == "sglang_fused_moe_triton":
        api_func = fused_moe_sglang_api
        api_kwargs = {
            "x": x,
            "w1": w1,
            "w2": w2,
            "input_gating": input_gating,
            "topk": topk,
            "use_fp8_w8a8": use_fp8_w8a8,
            "block_shape": block_shape,
        }

    # Warmup
    for _ in range(10):
        _ = api_func(**api_kwargs)
    torch.cuda.synchronize()

    if use_cuda_graph:
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            api_func(**api_kwargs)
        torch.cuda.synchronize()

        bench_lambda = lambda: graph.replay()  # noqa E731
    else:
        bench_lambda = lambda: api_func(**api_kwargs)  # noqa E731

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(bench_lambda, quantiles=quantiles)
    return ms, min_ms, max_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument("--ep-size", "--ep", type=int, default=1)
    parser.add_argument("--use-fp8-w8a8", action="store_true")
    parser.add_argument(
        "--use-cuda-graph", action="store_true", help="Enable CUDA Graph capture/replay"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/benchmark_ops/sglang_fused_moe/",
    )
    args = parser.parse_args()

    model_config = ModelConfig(args.config_path)
    benchmark.run(
        show_plots=True,
        print_data=True,
        save_path=args.save_path,
        model_config=model_config,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        use_fp8_w8a8=args.use_fp8_w8a8,
        use_cuda_graph=args.use_cuda_graph,
    )


if __name__ == "__main__":
    main()
