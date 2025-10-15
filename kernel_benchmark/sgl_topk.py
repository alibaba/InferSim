import dataclasses
import random
import time

import torch
import triton
from sgl_kernel import fast_topk_v2


@dataclasses.dataclass
class TestParam:
    b: int
    s_kv: int
    max_seq_len: int = 131072
    seed: int = 1024


@dataclasses.dataclass
class Testcase:
    t: TestParam
    score: torch.Tensor
    lengths: torch.Tensor


def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)
    score = torch.randn((t.b, t.max_seq_len), dtype=torch.float32)
    lengths = torch.full((t.b,), t.s_kv, dtype=torch.int32, device="cuda")

    return Testcase(
        t=t,
        score=score,
        lengths=lengths,
    )


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = generate_testcase(p)
    torch.cuda.synchronize()

    def run_ans():
        return fast_topk_v2(t.score, t.lengths, 2048)

    run_ans()
    torch.cuda.synchronize()

    ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20) / 1000  # type: ignore
    print(f"TopK:  {ans_time * 1e6:4.0f} us")

    return True


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    performance_cases = [
        TestParam(64, s_kv) for s_kv in [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    ]

    testcases = performance_cases

    for test in testcases:
        time.sleep(0.2)
        is_correct = run_test(test)
