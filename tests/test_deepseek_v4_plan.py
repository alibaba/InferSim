import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from config.model_config import ModelConfig
from kernel_benchmark.deepseek_v4_plan import build_manifest
from main import main


V4_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 7168,
    "num_hidden_layers": 3,
    "head_dim": 512,
    "num_attention_heads": 128,
    "num_key_value_heads": 1,
    "q_lora_rank": 1536,
    "o_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "o_groups": 16,
    "index_topk": 1024,
    "sliding_window": 128,
    "compress_ratios": [128, 4, 0, 128],
    "n_routed_experts": 384,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 3072,
}


class DeepSeekV4PlanTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(V4_CONFIG, self.temp)
        self.temp.close()

    def tearDown(self):
        os.unlink(self.temp.name)

    def test_config_normalizes_v4_fields(self):
        config = ModelConfig(self.temp.name)
        self.assertTrue(config.is_deepseek_v4)
        self.assertEqual(config.attn_type, "DSV4_MQA")
        self.assertEqual(config.num_routed_experts, 384)
        self.assertEqual(config.num_shared_experts, 1)
        self.assertEqual(config.nextn_compress_ratios, [128])

    def test_manifest_contains_v4_specific_cases(self):
        manifest = build_manifest(ModelConfig(self.temp.name), world_sizes=(1, 8, 7))
        self.assertEqual(manifest["attention"]["kernel"], "dsv4_compressed_mqa")
        self.assertEqual(manifest["attention"]["compression_layers"], {"0": 1, "4": 1, "128": 1})
        self.assertEqual(manifest["grouped_gemm"]["world_sizes"], [1, 8])
        self.assertEqual(manifest["grouped_gemm"]["intermediate_size"], 3072)

    def test_main_requires_v4_specific_measurements(self):
        with self.assertRaisesRegex(NotImplementedError, "dsv4_compressed_mqa"):
            main(SimpleNamespace(config_path=self.temp.name))


if __name__ == "__main__":
    unittest.main()
