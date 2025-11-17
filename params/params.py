from config.model_config import ModelConfig
from hardware.gpu import GPU


def get_mha_params_size(config: ModelConfig, use_fp8: bool):
    wq = config.hidden_size * config.num_attention_heads * config.head_dim
    wk = config.hidden_size * config.num_key_value_heads * config.head_dim
    wv = config.hidden_size * config.num_key_value_heads * config.head_dim
    wo = config.hidden_size * config.num_attention_heads * config.head_dim
    if use_fp8:
        return wq + wk + wv + wo
    return 2 * (wq + wk + wv + wo)


def get_mla_params_size(config: ModelConfig, use_fp8: bool):
    wq_down = config.hidden_size * config.q_lora_rank
    wq_up = config.q_lora_rank * config.num_attention_heads * config.qk_head_dim
    wkv_down = config.hidden_size * config.kv_lora_rank
    wkv_up = (
        config.kv_lora_rank
        * config.num_attention_heads
        * (config.qk_nope_head_dim + config.v_head_dim)
    )
    wo = config.hidden_size * config.num_attention_heads * config.v_head_dim
    if use_fp8:
        return wq_down + wq_up + wkv_down + wkv_up + wo
    return 2 * (wq_down + wq_up + wkv_down + wkv_up + wo)


def get_gdn_params_size(config: ModelConfig, use_fp8: bool):
    wq = config.hidden_size * config.linear_num_key_heads * config.linear_key_head_dim
    wk = wq
    wv = config.hidden_size * config.linear_num_value_heads * config.linear_value_head_dim
    wz = wv
    wa = config.hidden_size * config.linear_num_value_heads
    wb = wa
    s = wq + wk + wv + wz + wa + wb
    wconv = config.linear_num_key_heads * config.linear_key_head_dim * config.linear_conv_kernel_dim
    wconv += config.linear_num_key_heads * config.linear_key_head_dim * config.linear_conv_kernel_dim
    wconv += config.linear_num_value_heads * config.linear_value_head_dim * config.linear_conv_kernel_dim
    if use_fp8:
        return s + wconv
    return 2 * s + wconv


def get_attn_params_size(config: ModelConfig, use_fp8: bool):
    if config.attn_type == "MHA/GQA":
        return get_mha_params_size(config, use_fp8)
    elif config.attn_type == "MLA":
        return get_mla_params_size(config, use_fp8)


def get_linear_attn_params_size(config: ModelConfig, use_fp8: bool):
    return get_gdn_params_size(config, use_fp8)


def get_expert_params_size(config: ModelConfig, use_fp8: bool):
    w = 3 * config.hidden_size * config.intermediate_size
    if not use_fp8:
        w *= 2
    return w


def load_attn_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU):
    size = get_attn_params_size(config, use_fp8)
    return size / 1024 / 1024 / 1024 / gpu.mem_bw


def load_moe_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU, num_gpus):
    size = get_expert_params_size(config, use_fp8)
    size *= config.num_routed_experts / num_gpus
    return size / 1024 / 1024 / 1024 / gpu.mem_bw
