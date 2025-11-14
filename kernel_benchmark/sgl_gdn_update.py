import dataclasses
import random
import time

import torch
import torch.nn.functional as F
import triton
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import \
    fused_sigmoid_gating_delta_rule_update


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


def generate_testcase(t: TestParam) -> Testcase:
    seed = 1024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    device = "cuda"
    q = F.normalize(
        torch.randn((1, t.bs, t.Hk, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    k = F.normalize(
        torch.randn((1, t.bs, t.Hk, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    v = F.normalize(
        torch.randn((1, t.bs, t.Hv, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    a = torch.randn((t.bs, t.Hv), dtype=torch.bfloat16, device=device).sigmoid()
    b = torch.randn((t.bs, t.Hv), dtype=torch.bfloat16, device=device).sigmoid()
    initial_state = torch.randn(
        (t.bs, t.Hv, t.D, t.D), dtype=torch.float32, device=device
    )
    cache_indices = torch.arange(t.bs, dtype=torch.int32, device=device)
    A_log = torch.randn(t.Hv, dtype=torch.float32, device=device).sigmoid()
    dt_bias = torch.randn(t.Hv, dtype=torch.bfloat16, device=device).sigmoid()

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
            dt_bias=t.dt_bias,
            q=t.q,
            k=t.k,
            v=t.v,
            a=t.a,
            b=t.b,
            initial_state_source=t.ssm_states,
            initial_state_indices=t.cache_indices,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    core_attn_out = run_gdn_update()
    torch.cuda.synchronize()
    print(f"core_attn_out:{core_attn_out.shape}")

    ans_time: float = triton.testing.do_bench(run_gdn_update, warmup=10, rep=20) / 1000  # type: ignore
    print(f"gdn_update:  {ans_time * 1e6:4.0f} us")
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

    testcases = performance_cases

    for test in testcases:
        time.sleep(0.2)
        is_correct = run_test(test)
