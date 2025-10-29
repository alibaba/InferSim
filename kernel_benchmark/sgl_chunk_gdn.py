import dataclasses
import random
import time

import torch
import torch.nn.functional as F
import triton
from sglang.srt.layers.attention.fla.chunk_delta_h import \
    chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import \
    chunk_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.solve_tril import solve_tril
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd


@dataclasses.dataclass
class TestParam:
    seq_len: int = 4096
    Hk: int = 16
    Hv: int = 32
    D: int = 128
    chunk_size: int = 64
    seed: int = 1024


@dataclasses.dataclass
class Testcase:
    t: TestParam
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    cu_seqlens: torch.Tensor
    initial_state: torch.Tensor


def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)

    device = "cuda"
    q = F.normalize(
        torch.randn((1, t.seq_len, t.Hk, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    k = F.normalize(
        torch.randn((1, t.seq_len, t.Hk, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    v = F.normalize(
        torch.randn((1, t.seq_len, t.Hv, t.D), dtype=torch.bfloat16, device=device),
        dim=-1,
    )
    beta = torch.randn(
        (1, t.seq_len, t.Hv), dtype=torch.bfloat16, device=device
    ).sigmoid()
    cu_seqlens = torch.tensor([0, t.seq_len], dtype=torch.int32, device=device)
    g = torch.randn((1, t.seq_len, t.Hv), dtype=torch.float32, device=device)
    g = chunk_local_cumsum(g, chunk_size=t.chunk_size, cu_seqlens=cu_seqlens)
    initial_state = torch.randn((1, t.Hv, t.D, t.D), dtype=torch.float32, device=device)

    return Testcase(
        t=t,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = generate_testcase(p)
    torch.cuda.synchronize()

    def run_kkt():
        return chunk_scaled_dot_kkt_fwd(
            k=t.k,
            beta=t.beta,
            g_cumsum=t.g,
            cu_seqlens=t.cu_seqlens,
            chunk_size=p.chunk_size,
        )

    A = run_kkt()
    torch.cuda.synchronize()
    print(f"A:{A.shape}")

    ans_time: float = triton.testing.do_bench(run_kkt, warmup=10, rep=20) / 1000  # type: ignore
    print(f"chunk_scaled_dot_kkt:  {ans_time * 1e6:4.0f} us")

    def run_tril():
        return solve_tril(A=A, cu_seqlens=t.cu_seqlens, output_dtype=t.k.dtype)

    ans_time: float = triton.testing.do_bench(run_tril, warmup=10, rep=20) / 1000  # type: ignore
    print(f"solve_tril:  {ans_time * 1e6:4.0f} us")

    A = run_tril()
    torch.cuda.synchronize()

    def run_w_u():
        return recompute_w_u_fwd(
            k=t.k,
            v=t.v,
            beta=t.beta,
            A=A,
            g_cumsum=t.g,
            cu_seqlens=t.cu_seqlens,
        )

    ans_time: float = triton.testing.do_bench(run_w_u, warmup=10, rep=20) / 1000  # type: ignore
    print(f"recompute_w_u_fwd:  {ans_time * 1e6:4.0f} us")

    w, u = run_w_u()
    torch.cuda.synchronize()

    def run_chunk_gdn_fwd_h():
        return chunk_gated_delta_rule_fwd_h(
            k=t.k,
            w=w,
            u=u,
            g=t.g,
            initial_state=t.initial_state,
            output_final_state=True,
            cu_seqlens=t.cu_seqlens,
        )

    ans_time: float = (
        triton.testing.do_bench(run_chunk_gdn_fwd_h, warmup=10, rep=20) / 1000
    )  # type: ignore
    print(f"chunk_gated_delta_rule_fwd_h:  {ans_time * 1e6:4.0f} us")

    h, v_new, final_state = run_chunk_gdn_fwd_h()
    torch.cuda.synchronize()

    def run_chunk_fwd_o():
        return chunk_fwd_o(
            q=t.q,
            k=t.k,
            v=v_new,
            h=h,
            g=t.g,
            scale=t.k.shape[-1] ** -0.5,
            cu_seqlens=t.cu_seqlens,
        )

    ans_time: float = triton.testing.do_bench(run_chunk_fwd_o, warmup=10, rep=20) / 1000  # type: ignore
    print(f"chunk_fwd_o:  {ans_time * 1e6:4.0f} us")

    return True


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    performance_cases = [TestParam(seq_len=seq_len) for seq_len in [4096]]

    testcases = performance_cases

    for test in testcases:
        time.sleep(0.2)
        is_correct = run_test(test)
