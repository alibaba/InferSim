import math

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size
from layers.attn import create_attention
from layers.moe import MoE
from params.params import get_attn_params_size, get_expert_params_size


class Model:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]
        self.args = args
        self.config = config

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))
        attn_params_bytes = get_attn_params_size(self.config, self.args.use_fp8_gemm)
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One attn params size (MB):", attn_params_bytes / 1024 / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )
        params_per_gpu = attn_params_bytes + expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / self.args.world_size
        )
        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024
        params_per_gpu *= self.config.num_hidden_layers
        self.kvcache_mem = (
            self.gpu.mem - params_per_gpu - 15 - 5
        )  # 15GB for runtime, 5GB for encoder
        print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))
        print("{:<40} {:<10.2f}".format("KV cache space (GB):", self.kvcache_mem))
        context_len = self.args.target_isl + self.args.target_osl

        if self.args.decode_bs is None:
            target_bs = math.ceil(self.args.target_tgs * self.args.target_tpot / 1000)
        else:
            target_bs = self.args.decode_bs
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))
        print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))
        target_kvcache_bytes = (
            self.kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len
        )
        kvcache_bytes = get_kvcache_size(self.config, self.args.use_fp8_kv)
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-token KV cache size (KB):", kvcache_bytes / 1024
            )
        )
        if kvcache_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")
        self.kvcache_bytes = kvcache_bytes
        self.target_bs = target_bs

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Num hidden layers:", self.config.num_hidden_layers)
        )
        # per-token per-layer gflops
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config)
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer attn core (GFLOPs):", attn_core_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer others (GFLOPs):", other_gflops
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token attn core (GFLOPs):",
                attn_core_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token MoE (GFLOPs):", moe_gflops * self.config.num_hidden_layers
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token others (GFLOPs):",
                other_gflops * self.config.num_hidden_layers,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token total (GFLOPs):",
                (attn_core_gflops + moe_gflops + other_gflops)
                * self.config.num_hidden_layers,
            )
        )

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))
        print(
            "{:<40} {:<10}".format("Max prefill tokens:", self.args.max_prefill_tokens)
        )
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        attn_core_time = attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )
        attn_other_time = attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )
        attn_core_time *= math.ceil(self.args.max_prefill_tokens / self.args.target_isl)

        moe = MoE(self.config, self.args.use_fp8_gemm)
        moe_time = moe.prefill_moe(
            self.args.max_prefill_tokens, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.prefill_comm(self.args.max_prefill_tokens)
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

        num_tokens = self.args.max_prefill_tokens
        if self.args.enable_tbo:
            num_tokens *= 2
            ttft = max(
                (attn_core_time + attn_other_time) / self.args.sm_ratio, comm_time1
            )
            ttft += max(
                (attn_core_time + attn_other_time) / self.args.sm_ratio, comm_time2
            )
            ttft += max(moe_time / self.args.sm_ratio, comm_time1)
            ttft += max(moe_time / self.args.sm_ratio, comm_time2)
        else:
            ttft = attn_core_time
            ttft += moe_time
            ttft += attn_other_time
            ttft += comm_time1 + comm_time2
        ttft *= self.config.num_hidden_layers
        ttft *= 1000  # convert to ms
        ttft += 30  # for scheduler

        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):", num_tokens / (ttft / 1000)
            )
        )

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        attn_core_time = attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        attn_other_time = attn.decode_attn_others(self.target_bs, self.args.device_type)

        moe = MoE(self.config, self.args.use_fp8_gemm)
        moe_time = moe.decode_moe(
            self.target_bs, self.args.device_type, self.args.world_size
        )

        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.decode_comm(self.target_bs)
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))

        num_tokens = self.target_bs
        if self.args.enable_tbo:
            num_tokens *= 2
            tpot = max(
                attn_core_time + attn_other_time, moe_time + comm_time1 + comm_time2
            )
            tpot *= 2
        else:
            tpot = attn_core_time
            tpot += attn_other_time
            tpot += moe_time
            tpot += comm_time1 + comm_time2
        tpot *= self.config.num_hidden_layers
        tpot *= 1000  # convert to ms
        tpot += 5  # for scheduler

        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))
        print("{:<40} {:<10.0f}".format("Throughput (TGS):", num_tokens / tpot * 1000))
        if tpot > self.args.target_tpot:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")
