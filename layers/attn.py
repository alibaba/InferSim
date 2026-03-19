from flops.flops import gemm_flops
from hardware.gpu import gpu_map
from mfu.mfu import get_attn_decode_mfu, get_attn_prefill_mfu, get_gemm_mfu


def get_gemm_mfu_and_latency(m, k, n, device_type, use_fp8_gemm):
    gpu = gpu_map[device_type]
    gflops = gemm_flops(m, k, n) / 1e9
    mfu = get_gemm_mfu(device_type, m, k, n)
    latency = gflops / (gpu.fp16_tflops * 1024 * mfu)
    if use_fp8_gemm:
        latency = gflops / (gpu.fp8_tflops * 1024 * mfu)
    # print(f"Debug: gemm m:{m} k:{k} n:{n}")
    return latency


class MHA:
    def __init__(self, config, use_fp8_gemm, use_fp8_kv):
        self.use_fp8_gemm = use_fp8_gemm
        self.use_fp8_kv = use_fp8_kv
        self.config = config

    def get_attn_core_gflops(self, bs, kv_len):
        # Use TP-aware dimensions
        attn_core = (
            gemm_flops(
                bs, self.config.tp_num_attention_heads * self.config.head_dim, kv_len
            )
            * 2
        )
        return attn_core / 1e9

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops(1, kv_len)
        attn_core_mfu = get_attn_decode_mfu(
            self.config, bs, kv_len, device_type, self.use_fp8_kv
        )
        attn_core_time = (
            bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * kv_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)

    def decode_attn_others(self, bs, device_type):
        # Use TP-aware dimensions
        qkv_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.tp_hidden_size,
            n=(self.config.tp_num_attention_heads + self.config.tp_num_key_value_heads * 2)
            * self.config.head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("QKV_proj latency (us):", qkv_proj * 1e6))

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.tp_num_attention_heads * self.config.head_dim,
            n=self.config.tp_hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))
        return qkv_proj + o_proj

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops(1, seq_len)
        attn_core_mfu = get_attn_prefill_mfu(self.config, seq_len, device_type)
        attn_core_time = (
            seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)

    def prefill_attn_others(self, seq_len, device_type):
        return self.decode_attn_others(seq_len, device_type)


class MLA(MHA):
    def __init__(self, config, use_fp8_gemm, use_fp8_kv):
        self.use_fp8_gemm = use_fp8_gemm
        self.use_fp8_kv = use_fp8_kv
        self.config = config

    def get_attn_core_gflops_absorb(self, bs, kv_len):
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.kv_lora_rank + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.kv_lora_rank
        )
        return attn_core / 1e9

    def get_attn_core_gflops_noabsorb(self, bs, kv_len):
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.qk_nope_head_dim + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.v_head_dim
        )
        return attn_core / 1e9

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops_absorb(1, kv_len)
        attn_core_mfu = get_attn_decode_mfu(
            self.config, bs, kv_len, device_type, self.use_fp8_kv
        )
        attn_core_time = (
            bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * kv_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)

    def decode_attn_others(self, bs, device_type):
        # Use TP-aware dimensions
        q_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.tp_hidden_size,
            n=self.config.q_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("Q_down_proj latency (us):", q_down_proj * 1e6))

        q_up_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.q_lora_rank,
            n=self.config.tp_num_attention_heads * self.config.qk_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("Q_up_proj latency (us):", q_up_proj * 1e6))

        kv_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.tp_hidden_size,
            n=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print(
            "{:<40} {:<10.2f}".format("KV_down_proj latency (us):", kv_down_proj * 1e6)
        )

        bmm_q_wk = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.tp_num_attention_heads * self.config.qk_nope_head_dim,
            n=self.config.kv_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("bmm_q_wk latency (us):", bmm_q_wk * 1e6))

        bmm_o_wv = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.tp_num_attention_heads * self.config.kv_lora_rank,
            n=self.config.v_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("bmm_o_wv latency (us):", bmm_o_wv * 1e6))

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.tp_num_attention_heads * self.config.v_head_dim,
            n=self.config.tp_hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))
        return q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        attn_core_gflops = self.get_attn_core_gflops_noabsorb(1, seq_len)
        attn_core_mfu = get_attn_prefill_mfu(self.config, seq_len, device_type)
        attn_core_time = (
            seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))

        return max(attn_core_time, kv_load_time)


def create_attention(config, use_fp8_gemm, use_fp8_kv):
    if config.attn_type == "MHA/GQA":
        return MHA(config, use_fp8_gemm, use_fp8_kv)
    elif config.attn_type == "MLA":
        return MLA(config, use_fp8_gemm, use_fp8_kv)
