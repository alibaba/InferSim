from config.model_config import ModelConfig


def gemm_flops(m, n, k):
    return 2.0 * m * n * k


def get_mha_gflops(config, bs, avg_context_len, tp_size):
    # TP shards heads; hidden_size is NOT sharded
    tp_num_heads = config.num_attention_heads // tp_size
    tp_num_kv_heads = config.num_key_value_heads // tp_size
    q_proj = gemm_flops(
        bs, config.hidden_size, tp_num_heads * config.head_dim
    )
    k_proj = gemm_flops(
        bs, config.hidden_size, tp_num_kv_heads * config.head_dim
    )
    v_proj = gemm_flops(
        bs, config.hidden_size, tp_num_kv_heads * config.head_dim
    )
    o_proj = gemm_flops(
        bs, tp_num_heads * config.head_dim, config.hidden_size
    )
    attn_core = gemm_flops(
        bs, tp_num_heads * config.head_dim, avg_context_len
    ) + gemm_flops(bs, avg_context_len, tp_num_heads * config.head_dim)
    return attn_core / 1e9, (q_proj + k_proj + v_proj + o_proj) / 1e9


def get_mla_absorb_gflops(config, bs, avg_context_len, tp_size):
    tp_num_heads = config.num_attention_heads // tp_size
    q_down_proj = gemm_flops(bs, config.hidden_size, config.q_lora_rank)
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, tp_num_heads * config.qk_head_dim
    )

    kv_down_proj = gemm_flops(
        bs, config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )

    bmm_q_wk = tp_num_heads * gemm_flops(
        bs, config.qk_nope_head_dim, config.kv_lora_rank
    )
    bmm_o_wv = tp_num_heads * gemm_flops(
        bs, config.kv_lora_rank, config.v_head_dim
    )

    o_proj = gemm_flops(
        bs, tp_num_heads * config.v_head_dim, config.hidden_size
    )

    attn_core = gemm_flops(
        bs,
        tp_num_heads * (config.kv_lora_rank + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs, avg_context_len, tp_num_heads * config.kv_lora_rank
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


def get_mla_noabsorb_gflops(config, bs, avg_context_len, tp_size):
    tp_num_heads = config.num_attention_heads // tp_size
    q_down_proj = gemm_flops(bs, config.hidden_size, config.q_lora_rank)
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, tp_num_heads * config.qk_head_dim
    )

    kv_down_proj = gemm_flops(
        bs, config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )
    kv_up_proj = gemm_flops(
        bs,
        config.kv_lora_rank,
        tp_num_heads * (config.v_head_dim + config.qk_nope_head_dim),
    )

    o_proj = gemm_flops(
        bs, tp_num_heads * config.v_head_dim, config.hidden_size
    )

    attn_core = gemm_flops(
        bs,
        tp_num_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(bs, avg_context_len, tp_num_heads * config.v_head_dim)

    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + kv_up_proj + o_proj) / 1e9,
    )


def get_attn_gflops(
    config: ModelConfig, avg_context_len: int, tp_size: int, absorb=True
):
    if config.attn_type == "MHA/GQA":
        return get_mha_gflops(
            config, bs=1, avg_context_len=avg_context_len, tp_size=tp_size
        )
    elif config.attn_type == "MLA":
        if absorb:
            return get_mla_absorb_gflops(
                config, bs=1, avg_context_len=avg_context_len, tp_size=tp_size
            )
        return get_mla_noabsorb_gflops(
            config, bs=1, avg_context_len=avg_context_len, tp_size=tp_size
        )


def get_moe_gflops(config: ModelConfig, tp_size: int):
    # TP shards intermediate_size; hidden_size is NOT sharded
    tp_intermediate_size = config.intermediate_size // tp_size
    act = config.num_shared_experts + config.num_experts_per_tok
    return act * 3.0 * gemm_flops(1, config.hidden_size, tp_intermediate_size) / 1e9
