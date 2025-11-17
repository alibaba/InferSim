from config.model_config import ModelConfig


def get_mha_kvcache_size(config: ModelConfig, use_fp8):
    kvcache_size = (
        2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim
    )
    if not use_fp8:
        kvcache_size *= 2
    return kvcache_size


def get_mla_kvcache_size(config: ModelConfig, use_fp8):
    kvcache_size = config.num_hidden_layers * (
        config.kv_lora_rank + config.qk_rope_head_dim
    )
    if not use_fp8:
        kvcache_size *= 2
    return kvcache_size


def get_kvcache_size(config: ModelConfig, use_fp8):
    if config.attn_type == "MHA/GQA":
        return get_mha_kvcache_size(config, use_fp8)
    elif config.attn_type == "MLA":
        return get_mla_kvcache_size(config, use_fp8)


def get_states_size(config: ModelConfig):
    conv_state_size = (
        config.linear_num_value_heads * config.linear_value_head_dim
        + 2 * config.linear_num_key_heads * config.linear_key_head_dim
    ) * 2
    ssm_state_size = (
        config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4
    )
    states_size = config.num_linear_attn_layers * (conv_state_size + ssm_state_size)
    return states_size
