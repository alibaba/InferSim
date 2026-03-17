import dataclasses
import random
import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from sgl_kernel import causal_conv1d_fwd


@dataclasses.dataclass
class TestParam:
    seq_len: int = 131072
    seq_lens: list = dataclasses.field(default_factory=lambda: [2048, 1024, 1024])
    width: int = 4
    dim: int = 2048
    num_cache_slots: int = 440
    seed: int = 1024


@dataclasses.dataclass
class Testcase:
    t: TestParam
    x: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    initial_states: torch.Tensor
    has_initial_state_tensor: torch.Tensor


def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)

    x = torch.randn(
        1, t.dim, t.seq_len, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    weight = torch.randn(t.dim, t.width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(t.dim, device="cuda", dtype=torch.bfloat16)
    initial_states = torch.randn(
        1, t.dim, t.width - 1, device="cuda", dtype=torch.bfloat16
    )
    has_initial_state_tensor = torch.ones(1, dtype=torch.bool, device=x.device)

    return Testcase(
        t=t,
        x=x,
        weight=weight,
        bias=bias,
        initial_states=initial_states,
        has_initial_state_tensor=has_initial_state_tensor,
    )


def generate_testcase_varlen(t: TestParam):
    """
    生成 varlen 模式的测试数据，模拟 SGLang 实际的 continuous batching 场景
    x: (dim, cu_seqlen) - 多个序列在 seqlen 维度拼接
    """
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)

    batch = len(t.seq_lens)
    cu_seqlen = sum(t.seq_lens)

    # x: (dim, cu_seqlen) —— varlen 模式，序列在 seqlen 维拼接
    x = torch.randn(t.dim, cu_seqlen, device="cuda", dtype=torch.bfloat16).contiguous()
    weight = torch.randn(t.dim, t.width, device="cuda", dtype=torch.bfloat16)
    # SGLang 实际运行 bias=None
    bias = None

    # query_start_loc: (batch+1,)，记录每个序列的起止位置
    seqlens_tensor = torch.tensor([0] + t.seq_lens, dtype=torch.int32)
    query_start_loc = torch.cumsum(seqlens_tensor, dim=0).to(torch.int32).cuda()

    # cache_indices: (batch,)，每个序列映射到状态池中的哪个 slot
    cache_indices = torch.tensor([0, 5, 10], dtype=torch.int32, device="cuda")

    # conv_states: (num_cache_slots, dim, width-1) 状态池
    conv_states = torch.randn(
        t.num_cache_slots, t.dim, t.width - 1, device="cuda", dtype=torch.bfloat16
    )

    # has_initial_state: (batch,)，只有第1个序列有初始状态
    has_initial_state = torch.tensor(
        [True, False, False], dtype=torch.bool, device="cuda"
    )

    return x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation in ["silu", "swish"],
        pad_slot_id,
    )
    return x


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        print(f"initial_states:{initial_states.shape}  x:{x.shape}")
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    rtol, atol = 1e-2, 5e-2

    t = generate_testcase(p)
    torch.cuda.synchronize()

    x_ref = t.x.clone()
    weight_ref = t.weight.clone()
    bias_ref = t.bias.clone()
    initial_states_ref = t.initial_states.clone()

    out = causal_conv1d_fn(
        t.x,
        t.weight,
        t.bias,
        activation="silu",
        conv_states=t.initial_states,
        has_initial_state=t.has_initial_state_tensor,
    )

    out_ref, final_states_ref = causal_conv1d_ref(
        x_ref,
        weight_ref,
        bias_ref,
        initial_states=initial_states_ref,
        return_final_states=True,
        activation="silu",
    )

    assert t.initial_states is not None and final_states_ref is not None
    assert torch.allclose(t.initial_states, final_states_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    def run_ans():
        return causal_conv1d_fn(
            t.x,
            t.weight,
            t.bias,
            activation="silu",
            conv_states=t.initial_states,
            has_initial_state=t.has_initial_state_tensor,
        )

    run_ans()
    torch.cuda.synchronize()

    ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20) / 1000  # type: ignore
    print(f"causal_conv1d_fn:  {ans_time * 1e6:4.0f} us")

    return True


@torch.inference_mode()
def run_test_varlen(p: TestParam) -> bool:
    """
    Varlen 模式测试，模拟 SGLang 实际的 continuous batching 场景
    """
    print("================")
    print(f"Running varlen mode on {p}")
    torch.cuda.empty_cache()

    x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state = (
        generate_testcase_varlen(p)
    )
    torch.cuda.synchronize()

    def run_ans():
        return causal_conv1d_fn(
            x,
            weight,
            bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            conv_states=conv_states,
            activation="silu",
        )

    run_ans()
    torch.cuda.synchronize()

    ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20) / 1000  # type: ignore
    print(f"causal_conv1d_fn (varlen):  {ans_time * 1e6:4.0f} us")

    return True


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    # 原始 batch 模式测试
    print("=" * 50)
    print("Batch Mode Tests")
    print("=" * 50)
    performance_cases = [TestParam(seq_len=seq_len) for seq_len in [4096]]
    testcases = performance_cases
    for test in testcases:
        time.sleep(0.2)
        is_correct = run_test(test)

    # Varlen 模式测试（模拟 SGLang 实际运行场景）
    print("\n" + "=" * 50)
    print("Varlen Mode Tests (SGLang actual scenario)")
    print("=" * 50)
    varlen_cases = [
        TestParam(seq_lens=[2048, 1024, 1024], dim=8192, num_cache_slots=440),  # x: [8192, 4096]
        TestParam(seq_lens=[4096, 2048, 2048], dim=8192, num_cache_slots=440),  # x: [8192, 8192]
    ]
    for test in varlen_cases:
        time.sleep(0.2)
        is_correct = run_test_varlen(test)
