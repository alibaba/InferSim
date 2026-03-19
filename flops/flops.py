from config.model_config import ModelConfig


def gemm_flops(m, n, k):
    return 2.0 * m * n * k


def get_mha_gflops(config, bs, avg_context_len):
    # Use TP-aware dimensions
    q_proj = gemm_flops(
        bs, config.tp_hidden_size, config.tp_num_attention_heads * config.head_dim
    )
    k_proj = gemm_flops(
        bs, config.tp_hidden_size, config.tp_num_key_value_heads * config.head_dim
    )
    v_proj = gemm_flops(
        bs, config.tp_hidden_size, config.tp_num_key_value_heads * config.head_dim
    )
    o_proj = gemm_flops(
        bs, config.tp_num_attention_heads * config.head_dim, config.tp_hidden_size
    )
    attn_core = gemm_flops(
        bs, config.tp_num_attention_heads * config.head_dim, avg_context_len
    ) + gemm_flops(bs, avg_context_len, config.tp_num_attention_heads * config.head_dim)
    return attn_core / 1e9, (q_proj + k_proj + v_proj + o_proj) / 1e9


def get_mla_absorb_gflops(config, bs, avg_context_len):
    # Use TP-aware dimensions
    q_down_proj = gemm_flops(bs, config.tp_hidden_size, config.q_lora_rank)
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, config.tp_num_attention_heads * config.qk_head_dim
    )

    kv_down_proj = gemm_flops(
        bs, config.tp_hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )

    bmm_q_wk = config.tp_num_attention_heads * gemm_flops(
        bs, config.qk_nope_head_dim, config.kv_lora_rank
    )
    bmm_o_wv = config.tp_num_attention_heads * gemm_flops(
        bs, config.kv_lora_rank, config.v_head_dim
    )

    o_proj = gemm_flops(
        bs, config.tp_num_attention_heads * config.v_head_dim, config.tp_hidden_size
    )

    attn_core = gemm_flops(
        bs,
        config.tp_num_attention_heads * (config.kv_lora_rank + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs, avg_context_len, config.tp_num_attention_heads * config.kv_lora_rank
    )

    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj) / 1e9,
    )


def get_gqla_absorb_gflops(config, bs, avg_context_len):
    q_down_proj = gemm_flops(bs, config.hidden_size, config.q_lora_rank)
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, config.num_attention_heads * config.qk_head_dim
    )

    kv_down_proj = gemm_flops(
        bs, config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )

    bmm_q_wk = (
        2
        * config.num_attention_heads
        / 2
        * gemm_flops(bs, config.qk_nope_head_dim, config.kv_lora_rank / 2)
    )
    bmm_o_wv = (
        2
        * config.num_attention_heads
        / 2
        * gemm_flops(bs, config.kv_lora_rank / 2, config.v_head_dim)
    )

    o_proj = gemm_flops(
        bs, config.num_attention_heads * config.v_head_dim, config.hidden_size
    )

    attn_core = gemm_flops(
        bs,
        config.num_attention_heads
        * (config.kv_lora_rank / 2 + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs, avg_context_len, config.num_attention_heads * config.kv_lora_rank / 2
    )

    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj) / 1e9,
    )


def get_mla_noabsorb_gflops(config, bs, avg_context_len):
    # Use TP-aware dimensions
    q_down_proj = gemm_flops(bs, config.tp_hidden_size, config.q_lora_rank)
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, config.tp_num_attention_heads * config.qk_head_dim
    )

    kv_down_proj = gemm_flops(
        bs, config.tp_hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )
    kv_up_proj = gemm_flops(
        bs,
        config.kv_lora_rank,
        config.tp_num_attention_heads * (config.v_head_dim + config.qk_nope_head_dim),
    )

    o_proj = gemm_flops(
        bs, config.tp_num_attention_heads * config.v_head_dim, config.tp_hidden_size
    )

    attn_core = gemm_flops(
        bs,
        config.tp_num_attention_heads
        * (config.qk_nope_head_dim + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(bs, avg_context_len, config.tp_num_attention_heads * config.v_head_dim)

    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + kv_up_proj + o_proj) / 1e9,
    )


def get_attn_gflops(config: ModelConfig, avg_context_len: int, absorb=True):
    if config.attn_type == "MHA/GQA":
        return get_mha_gflops(config, bs=1, avg_context_len=avg_context_len)
    elif config.attn_type == "MLA":
        if absorb:
            return get_mla_absorb_gflops(config, bs=1, avg_context_len=avg_context_len)
        return get_mla_noabsorb_gflops(config, bs=1, avg_context_len=avg_context_len)


def get_moe_gflops(config: ModelConfig):
    # Use TP-aware dimensions
    act = config.num_shared_experts + config.num_experts_per_tok
    return act * 3.0 * gemm_flops(1, config.tp_hidden_size, config.tp_intermediate_size) / 1e9
