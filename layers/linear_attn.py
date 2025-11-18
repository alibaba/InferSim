from hardware.gpu import gpu_map
from layers.attn import get_gemm_mfu_and_latency
from mfu.mfu import (get_linear_attn_decode_latency,
                     get_linear_attn_prefill_latency)


class GDN:
    def __init__(self, config, use_fp8_gemm):
        self.use_fp8_gemm = use_fp8_gemm
        self.config = config

    def decode_attn_core(self, bs, states_bytes, device_type):
        gpu = gpu_map[device_type]
        t_attn_core = get_linear_attn_decode_latency(self.config, bs, device_type)
        t_states_load = (
            states_bytes
            * bs
            / self.config.num_linear_attn_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Linear attn core latency (us):", t_attn_core))
        print(
            "{:<40} {:<10.2f}".format(
                "States loading latency (us):", t_states_load * 1e6
            )
        )

        return max(t_attn_core / 1e6, t_states_load)

    def decode_attn_others(self, bs, device_type):
        key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim
        value_dim = (
            self.config.linear_num_value_heads * self.config.linear_value_head_dim
        )
        projection_size_qkvz = key_dim * 2 + value_dim * 2
        projection_size_ba = self.config.linear_num_value_heads * 2
        qkvz_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=projection_size_qkvz,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        qkvzba_proj = (
            qkvz_proj
            * (projection_size_qkvz + projection_size_ba)
            / projection_size_qkvz
        )
        print("{:<40} {:<10.2f}".format("qkvzba_proj latency (us):", qkvzba_proj * 1e6))

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.linear_num_value_heads * self.config.linear_value_head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))
        return qkvzba_proj + o_proj

    def prefill_attn_core(self, seq_len, states_bytes, device_type):
        gpu = gpu_map[device_type]
        t_attn_core = get_linear_attn_prefill_latency(self.config, seq_len, device_type)
        t_states_load = (
            states_bytes
            / self.config.num_linear_attn_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )

        print("{:<40} {:<10.2f}".format("Linear Attn core latency (us):", t_attn_core))
        print(
            "{:<40} {:<10.2f}".format(
                "States loading latency (us):", t_states_load * 1e6
            )
        )

        return max(t_attn_core / 1e6, t_states_load)

    def prefill_attn_others(self, seq_len, device_type):
        return self.decode_attn_others(seq_len, device_type)


def create_linear_attn(config, use_fp8_gemm):
    return GDN(config, use_fp8_gemm)
