import dataclasses
import random
import time

import torch
import torch.nn.functional as F
import triton
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)


@dataclasses.dataclass
class TestParam:
    bs: int = 256
    Hk: int = 16
    Hv: int = 32
    D: int = 128


@dataclasses.dataclass
class Testcase:
    t: TestParam
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    ssm_states: torch.Tensor
    cache_indices: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    cu_seqlens: torch.Tensor


def generate_testcase(t: TestParam) -> Testcase:
    seed = 1024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = "cuda"

    cu_seqlens = torch.arange(0, t.bs + 1, dtype=torch.int32, device=device)

    total_tokens = t.bs

    q = F.normalize(
        torch.randn((1, total_tokens, t.Hk, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    k = F.normalize(
        torch.randn((1, total_tokens, t.Hk, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    v = F.normalize(
        torch.randn((1, total_tokens, t.Hv, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )

    a = torch.randn((total_tokens, t.Hv), dtype=torch.bfloat16, device=device).sigmoid()
    b = torch.randn((total_tokens, t.Hv), dtype=torch.bfloat16, device=device).sigmoid()

    pool_size = t.bs
    initial_state = torch.randn(
        (pool_size, t.Hv, t.D, t.D), dtype=torch.float32, device=device
    )

    cache_indices = torch.arange(t.bs, dtype=torch.int32, device=device)
    
    A_log = torch.randn(t.Hv, dtype=torch.float32, device=device)
    dt_bias = torch.randn(t.Hv, dtype=torch.bfloat16, device=device)

    return Testcase(
        t=t,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        ssm_states=initial_state,
        cache_indices=cache_indices,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
    )


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = generate_testcase(p)
    torch.cuda.synchronize()

    def run_gdn_update():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=t.A_log,
            a=t.a,
            dt_bias=t.dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=t.q,
            k=t.k,
            v=t.v,
            b=t.b,
            initial_state_source=t.ssm_states,
            initial_state_indices=t.cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=t.cu_seqlens,
            is_kda=False,
        )

    core_attn_out = run_gdn_update()
    torch.cuda.synchronize()
    print(f"core_attn_out: {core_attn_out.shape}")

    ans_time: float = triton.testing.do_bench(run_gdn_update, warmup=10, rep=20) / 1000
    print(f"gdn_update: {ans_time * 1e6:4.0f} us")
    return True


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    performance_cases = [
        TestParam(bs=bs)
        for bs in [3, 6, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    ]

    for test in performance_cases:
        time.sleep(0.2)
        run_test(test)