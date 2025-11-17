import json


class ModelConfig:
    def __init__(
        self,
        config_path,
    ):
        d = dict()
        with open(config_path, "r") as f:
            d = json.load(f)

        self.hidden_size = d["hidden_size"]
        self.num_hidden_layers = d["num_hidden_layers"]

        self.is_hybrid_linear = d.get("full_attention_interval") is not None
        if self.is_hybrid_linear:
            self.num_full_attn_layers = self.num_hidden_layers // d["full_attention_interval"]
            self.num_linear_attn_layers = self.num_hidden_layers - self.num_full_attn_layers
            self.linear_conv_kernel_dim = d["linear_conv_kernel_dim"]
            self.linear_key_head_dim = d["linear_key_head_dim"]
            self.linear_num_key_heads = d["linear_num_key_heads"]
            self.linear_value_head_dim = d["linear_value_head_dim"]
            self.linear_num_value_heads = d["linear_num_value_heads"]

        self.attn_type = "MHA/GQA"
        if "kv_lora_rank" in d:
            self.attn_type = "MLA"

        # attn
        if self.attn_type == "MHA/GQA":
            self.num_attention_heads = d["num_attention_heads"]
            self.num_key_value_heads = d["num_key_value_heads"]
            if "head_dim" in d:
                self.head_dim = d["head_dim"]
            else:
                self.head_dim = self.hidden_size // self.num_attention_heads
        elif self.attn_type == "MLA":
            self.q_lora_rank = d["q_lora_rank"]
            self.qk_nope_head_dim = d["qk_nope_head_dim"]
            self.qk_rope_head_dim = d["qk_rope_head_dim"]
            self.kv_lora_rank = d["kv_lora_rank"]
            self.num_attention_heads = d["num_attention_heads"]
            self.v_head_dim = d["v_head_dim"]
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # FFN/MoE
        self.is_moe = True
        if "num_routed_experts" in d:
            self.num_routed_experts = d["num_routed_experts"]
        elif "num_experts" in d:
            self.num_routed_experts = d["num_experts"]
        else:
            self.is_moe = False
            self.num_routed_experts = 1

        if self.is_moe:
            self.num_experts_per_tok = d["num_experts_per_tok"]
            self.intermediate_size = d["moe_intermediate_size"]
            self.num_shared_experts = d.get("num_shared_experts", 0)
        else:
            self.num_experts_per_tok = 1
            self.intermediate_size = d["intermediate_size"]
            self.num_shared_experts = 0
