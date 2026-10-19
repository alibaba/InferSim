import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

from layers.attn import MLA, create_attention


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


def mla_prefill_config():
    return SimpleNamespace(
        attn_type="MLA",
        num_attention_heads=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_hidden_layers=61,
    )


class MLAPrefillAttentionTest(unittest.TestCase):
    def test_prefill_uses_override_instead_of_bench_lookup(self):
        attention = MLA(mla_prefill_config(), False, False, 1, prefill_mfu=0.5)

        with mock.patch(
            "layers.attn.get_attn_prefill_mfu", return_value=0.9
        ) as lookup:
            with redirect_stdout(StringIO()) as buf:
                result = attention.prefill_attn_core(
                    seq_len=4096,
                    kvcache_bytes=0,
                    device_type="H800",
                )

        lookup.assert_not_called()
        self.assertIn("0.50", buf.getvalue())
        self.assertGreater(result, 0)

    def test_prefill_without_override_still_uses_bench_lookup(self):
        attention = MLA(mla_prefill_config(), False, False, 1)

        with mock.patch(
            "layers.attn.get_attn_prefill_mfu", return_value=0.25
        ) as lookup:
            with redirect_stdout(StringIO()) as buf:
                attention.prefill_attn_core(
                    seq_len=4096,
                    kvcache_bytes=0,
                    device_type="H800",
                )

        lookup.assert_called_once()
        self.assertIn("0.25", buf.getvalue())

    def test_create_attention_passes_prefill_mfu_to_mla(self):
        attention = create_attention(
            mla_prefill_config(), False, False, 1, prefill_mfu=0.5
        )

        self.assertIsInstance(attention, MLA)
        self.assertEqual(attention.prefill_mfu, 0.5)

        attention = create_attention(mla_prefill_config(), False, False, 1)
        self.assertIsNone(attention.prefill_mfu)
