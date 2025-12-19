from dataclasses import dataclass


@dataclass
class GPU:
    fp16_tflops: float
    fp8_tflops: float
    mfu: float  # default mfu
    mem: float
    mem_bw: float  # GB/s
    nvlink_bw: float  # unidirectional GB/s
    rdma_bw: float  # unidirectional GB/s
    frequency: float = None # MHz
    num_sm: int = None
    compute_capability: float = None

h20 = GPU(
    fp16_tflops=148,
    fp8_tflops=296,
    mfu=0.6,
    mem=96,
    mem_bw=4096 * 0.8,
    nvlink_bw=900 * 0.8 / 2,
    rdma_bw=50 * 0.8,
    frequency=1980 * 0.9,
    num_sm=78,
    compute_capability=9.0,
)  # 25GB/s for 4 ibv devices, 50GB/s for 8 ibv devices

h800 = GPU(
    fp16_tflops=989,
    fp8_tflops=1979,
    mfu=0.35,
    mem=80,
    mem_bw=3430 * 0.8,
    nvlink_bw=400 * 0.8 / 2,
    rdma_bw=50 * 0.8,
    frequency=1980 * 0.9,
    num_sm=132,
    compute_capability=9.0,
)

h200 = GPU(
    fp16_tflops=989,
    fp8_tflops=1979,
    mfu=0.4,
    mem=141,
    mem_bw=4800 * 0.8,
    nvlink_bw=900 * 0.8 / 2,
    rdma_bw=50 * 0.8,
)

gb200 = GPU(
    fp16_tflops=2500,
    fp8_tflops=5000,
    mfu=0.5,
    mem=192,
    mem_bw=13400 * 0.8,
    nvlink_bw=1800 * 0.8 / 2,  # 1800GB/s, bi-directional
    rdma_bw=50 * 0.8,
)  # GB200 NVL72

gpu_map = {"H20": h20, "H800": h800, "H200": h200, "GB200": gb200}