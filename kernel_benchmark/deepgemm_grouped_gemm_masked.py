# Adapt from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_fp8.py
import argparse
import enum
import os
import random
import sys
from typing import Generator

import deep_gemm
import pandas as pd
import torch
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import (ceil_div, per_block_cast_to_fp8,
                             per_token_cast_to_fp8)

parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(parent_dir))

from config.model_config import ModelConfig  # noqa E402


class KernelType(enum.Enum):
    # For SM100 GEMMs
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


def enumerate_m_grouped_masked() -> Generator:
    max_m = 4096
    for kernel_type in (KernelType.Kernel1D2D,):
        yield kernel_type, max_m


def generate_m_grouped_masked(
    num_groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    a = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.einsum("gmk,gnk->gmn", a, b)

    masked_m = torch.empty((num_groups,), device="cuda", dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    if use_bf16:
        return a, b, masked_m, d, ref_d

    a_fp8 = (
        torch.empty_like(a, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, max_m, ceil_div(k, 128)), device="cuda", dtype=torch.float
        ),
    )
    b_fp8 = (
        torch.empty_like(b, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, ceil_div(n, 128), ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = per_token_cast_to_fp8(a[i], use_ue8m0=use_ue8m0)
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)

    return a_fp8, b_fp8, masked_m, d, ref_d


def test_m_grouped_gemm_masked(num_groups, expected_m_per_group, k, n) -> None:
    print("Testing m-grouped masked GEMM:")

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, max_m in enumerate_m_grouped_masked():
        kernel_opt = "1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = False
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(
                num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
            )
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                a,
                b,
                d,
                masked_m,
                expected_m_per_group,
                disable_ue8m0_cast=disable_ue8m0_cast,
            )
            for j in range(num_groups):
                diff = calc_diff(
                    d[j, : masked_m[j].item()], ref_d[j, : masked_m[j].item()]
                )
                assert (
                    diff < 0.001
                ), f"{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}"

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked(
            num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0
        )

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                a,
                b,
                d,
                masked_m,
                expected_m_per_group,
                disable_ue8m0_cast=disable_ue8m0_cast,
            )

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        print(
            f" > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): "
            f"{t * 1e6:4.0f} us | "
            f"{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s"
        )
        tflops = 2 * valid_m * n * k / t / 1e12
        return t * 1e6, tflops


def main(args) -> None:
    print("Testing grouped masked GEMM:")
    config = ModelConfig(args.config_path)
    results = []

    for world_size in [4, 8]:
        num_local_experts = config.num_routed_experts // world_size
        num_groups = num_local_experts
        for bs in [8, 16, 32, 64, 128, 256, 512, 1024]:
            expected_m_per_group = round(bs * config.num_experts_per_tok / num_groups)
            print(f"expected_m_per_group: {expected_m_per_group}")
            if expected_m_per_group < 16:
                continue
            up_proj, up_tflops = test_m_grouped_gemm_masked(
                num_groups,
                expected_m_per_group,
                config.hidden_size,
                config.intermediate_size * 2,
            )
            down_proj, down_tflops = test_m_grouped_gemm_masked(
                num_groups,
                expected_m_per_group,
                config.intermediate_size,
                config.hidden_size,
            )
            results.append(
                {
                    "num_experts": config.num_routed_experts,
                    "num_gpus": world_size,
                    "num_local_experts": num_local_experts,
                    "topk": config.num_experts_per_tok,
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "batch_size_per_gpu": bs,
                    "tokens_per_expert": expected_m_per_group,
                    "up_proj_us": round(up_proj, 6),
                    "up_mfu": round(up_tflops / args.gpu_tflops, 6),
                    "down_proj_us": round(down_proj, 6),
                    "down_mfu": round(down_tflops / args.gpu_tflops, 6),
                }
            )
    print()
    df = pd.DataFrame(results)
    df.to_csv("groupedgemm_masked.csv", index=False)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print("Library path:")
    print(f" > {deep_gemm.__path__}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        help="The path of the hf model config.json",
        required=True,
    )
    parser.add_argument("--gpu-tflops", type=int, default=296, help="GPU FP8 TFLOPS")
    args = parser.parse_args()
    main(args)
