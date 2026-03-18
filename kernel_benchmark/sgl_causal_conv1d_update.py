import dataclasses
import random
import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
from sgl_kernel import causal_conv1d_update as causal_conv1d_update_kernel


@dataclasses.dataclass
class TestParam:
    batchsize: int = 1
    width: int = 4
    dim: int = 8192
    seed: int = 1024


@dataclasses.dataclass
class Testcase:
    t: TestParam
    x: torch.Tensor
    conv_state: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)

    x = torch.randn(
        t.batchsize, t.dim, 1, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    weight = torch.randn(t.dim, t.width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(t.dim, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.randn(
        t.batchsize, t.dim, t.width - 1, device="cuda", dtype=torch.bfloat16
    )

    return Testcase(
        t=t,
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}"
        )
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    causal_conv1d_update_kernel(
        x,
        conv_state,
        weight,
        bias,
        activation_val,
        cache_seqlens,
        conv_state_indices,
        pad_slot_id,
    )
    if unsqueeze:
        x = x.squeeze(-1)
    return x


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    rtol, atol = 1e-2, 5e-2

    t = generate_testcase(p)
    torch.cuda.synchronize()

    x_ref = t.x.clone()
    conv_state_ref = t.conv_state.clone()

    out = causal_conv1d_update(
        t.x,
        t.conv_state,
        t.weight,
        t.bias,
        activation="silu",
    )

    out_ref = causal_conv1d_update_ref(
        x_ref,
        conv_state_ref,
        t.weight,
        t.bias,
        activation="silu",
    )

    assert torch.equal(t.conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    def run_ans():
        return causal_conv1d_update(
            t.x,
            t.conv_state,
            t.weight,
            t.bias,
            activation="silu",
        )

    run_ans()
    torch.cuda.synchronize()

    ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20) / 1000  # type: ignore
    print(f"causal_conv1d_fn:  {ans_time * 1e6:4.0f} us")

    return True


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    performance_cases = [
        TestParam(batchsize=bs) for bs in [1, 16, 32, 64, 128, 256, 512, 1024]
    ]

    testcases = performance_cases

    for test in testcases:
        time.sleep(0.2)
        is_correct = run_test(test)
