import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from layers.attn import MLA


def mla_config():
    return SimpleNamespace(
        attn_type="MLA",
        num_attention_heads=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        num_hidden_layers=61,
    )


class MLADecodeAttentionTest(unittest.TestCase):
    def test_returns_existing_measured_latency_for_exact_benchmark_point(self):
        config = mla_config()
        attention = MLA(config, False, False, 1)

        with redirect_stdout(StringIO()):
            result = attention.decode_attn_core(
                bs=1,
                kv_len=1024,
                kvcache_bytes=61 * 1024**2,
                device_type="H800",
            )

        self.assertAlmostEqual(result, 20.910e-6, places=12)

    def test_preserves_analytical_fallback_without_benchmark_data(self):
        config = mla_config()
        attention = MLA(config, False, False, 1)

        with redirect_stdout(StringIO()):
            result = attention.decode_attn_core(
                bs=2,
                kv_len=1024,
                kvcache_bytes=0,
                device_type="H200",
            )

        self.assertAlmostEqual(result, 1.4419245298281092e-6, places=15)

    def test_preserves_kv_bound_fallback_without_benchmark_data(self):
        config = mla_config()
        attention = MLA(config, False, False, 1)

        with redirect_stdout(StringIO()):
            result = attention.decode_attn_core(
                bs=2,
                kv_len=1024,
                kvcache_bytes=61 * 1024**2,
                device_type="H200",
            )

        self.assertAlmostEqual(result, 0.0005208333333333333, places=15)


if __name__ == "__main__":
    unittest.main()
