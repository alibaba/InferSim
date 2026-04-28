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
from deep_gemm.utils import (align, ceil_div,
                             get_mk_alignment_for_contiguous_layout,
                             per_block_cast_to_fp8, per_token_cast_to_fp8)

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


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1


def enumerate_m_grouped_contiguous() -> Generator:
    for kernel_type in (KernelType.Kernel1D2D,):
        for major_a, major_b in ((MajorTypeAB.KMajor, MajorTypeAB.KMajor),):
            yield kernel_type, major_a, major_b


def generate_m_grouped_contiguous(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    actual_ms = [
        int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)
    ]
    aligned_ms = [
        align(actual_m, get_mk_alignment_for_contiguous_layout())
        for actual_m in actual_ms
    ]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, m_indices, d, ref_d

    assert major_a.is_k_major()
    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (
        torch.empty_like(b, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, ceil_div(n, 128), ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].mT.contiguous().mT, b_fp8[1])
    return m, a_fp8, b_fp8, m_indices, d, ref_d


def test_m_grouped_gemm_contiguous(num_groups, expected_m_per_group, k, n) -> None:
    print("Testing m-grouped contiguous GEMM:")

    for kernel_type, major_a, major_b in enumerate_m_grouped_contiguous():
        major_opt = "N" if major_a.is_k_major() else "T"
        major_opt += "T" if major_b.is_k_major() else "N"
        kernel_opt = "1D1D" if kernel_type.is_1d1d() else "1D2D"
        use_ue8m0 = False
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
                num_groups,
                expected_m_per_group,
                n,
                k,
                major_a,
                major_b,
                use_ue8m0=use_ue8m0,
            )
            func_name = f"m_grouped_fp8_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(
                a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast
            )
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert (
                diff < 0.001
            ), f"{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}"
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(
            num_groups,
            expected_m_per_group,
            n,
            k,
            major_a,
            major_b,
            use_ue8m0=use_ue8m0,
        )

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast
            )

        t = bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
        print(
            f" > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}): "
            f"{t * 1e6:4.0f} us | "
            f"{2 * m * n * k / t / 1e12:4.0f} TFLOPS | "
            f"{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s"
        )
        tflops = 2 * m * n * k / t / 1e12
        return t * 1e6, tflops


def main(args) -> None:
    print("Testing grouped contiguous GEMM:")
    config = ModelConfig(args.config_path, tp_size=args.tp_size)
    results = []

    # Use original hidden_size, but TP-divided intermediate_size
    tp_hidden_size = config.hidden_size
    tp_intermediate_size = config.intermediate_size // args.tp_size

    for world_size in [1, 2, 4, 8]:
        if world_size % args.tp_size != 0:
            continue
        ep_size = world_size // args.tp_size
        num_local_experts = config.num_routed_experts // ep_size
        if num_local_experts < 4:
            continue
        if config.num_routed_experts % num_local_experts > 0:
            continue
        num_groups = num_local_experts
        for seq_len in [1024, 4096, 8192, 16384, 32768]:
            expected_m_per_group = round(
                seq_len * config.num_experts_per_tok / num_groups
            )
            if expected_m_per_group < 1:
                continue
            up_proj, up_tflops = test_m_grouped_gemm_contiguous(
                num_groups,
                expected_m_per_group,
                tp_hidden_size,
                tp_intermediate_size * 2,
            )
            down_proj, down_tflops = test_m_grouped_gemm_contiguous(
                num_groups,
                expected_m_per_group,
                tp_intermediate_size,
                tp_hidden_size,
            )
            results.append(
                {
                    "num_experts": config.num_routed_experts,
                    "num_gpus": world_size,
                    "num_local_experts": num_local_experts,
                    "topk": config.num_experts_per_tok,
                    "hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "tp_size": args.tp_size,
                    "seq_len_per_gpu": seq_len,
                    "tokens_per_expert": expected_m_per_group,
                    "up_proj_us": round(up_proj, 6),
                    "up_mfu": round(up_tflops / args.gpu_tflops, 6),
                    "down_proj_us": round(down_proj, 6),
                    "down_mfu": round(down_tflops / args.gpu_tflops, 6),
                }
            )
    print()
    df = pd.DataFrame(results)
    df.to_csv("groupedgemm_contiguous.csv", index=False)


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
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()
    main(args)
