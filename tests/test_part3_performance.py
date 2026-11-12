"""Tests for Part III: tiled attention, quantization, the benchmark harness, GQA and windows."""

import pytest
import torch

from mlle.bench.harness import markdown_table, peak_rss_bytes, time_fn
from mlle.foundations.attention import attention, causal_mask
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.inference.flash_sim import attention_tiled
from mlle.inference.kvcache import KVCache, kv_bytes_per_token
from mlle.inference.quantize import dequantize_tensor, quantize_model_, quantize_tensor


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("blocks", [(1, 1), (7, 5), (16, 64), (64, 16), (200, 200)])
def test_tiled_attention_equals_reference(causal, blocks):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 50, 16) for _ in range(3))
    ref, _ = attention(q, k, v, causal_mask(50) if causal else None)
    out, stats = attention_tiled(q, k, v, causal=causal, block_q=blocks[0], block_k=blocks[1])
    assert torch.allclose(out, ref, atol=1e-5)
    assert stats["peak_score_elements"] == blocks[0] * blocks[1]


def test_tiled_attention_is_stable_with_huge_scores():
    q = torch.full((1, 4, 8), 30.0)
    k = torch.full((1, 4, 8), 30.0)
    v = torch.randn(1, 4, 8)
    out, _ = attention_tiled(q, k, v, causal=True, block_q=2, block_k=2)  # exp(scores) alone would overflow fp32
    assert torch.isfinite(out).all()


def test_quantization_roundtrip_error_bounds():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    for per_channel in (False, True):
        q, scale = quantize_tensor(w, 8, per_channel)
        assert q.dtype == torch.int8 and q.abs().max() <= 127
        err = (w - dequantize_tensor(q, scale)).abs()
        assert torch.all(err <= scale / 2 + 1e-6)  # rounding error is at most half a step


def test_per_channel_beats_per_tensor_with_an_outlier_row():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    w[0] *= 50  # one row with large weights inflates a shared scale
    errs = {}
    for per_channel in (False, True):
        q, scale = quantize_tensor(w, 8, per_channel)
        errs[per_channel] = (w[1:] - dequantize_tensor(q, scale)[1:]).pow(2).mean().item()
    assert errs[True] < 0.01 * errs[False]


def test_quantize_model_reports_sizes_and_changes_little_at_8_bits():
    torch.manual_seed(0)
    model = MiniGPT(MiniGPTConfig.nanochat_style(50, block_size=16, n_layer=2, n_head=2, n_embd=32)).eval()
    for p in model.parameters():
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.2)
    idx = torch.randint(0, 50, (2, 16))
    before = model(idx)
    stats = quantize_model_(model, bits=8)
    after = model(idx)
    assert stats["bytes_quantized"] * 4 == stats["bytes_fp32"]
    assert stats["relative_error"] < 0.02
    assert (before - after).abs().max() < 0.25


def test_gqa_and_window_shrink_or_bound_the_cache():
    assert kv_bytes_per_token(12, 6, 128, 2) == 2 * kv_bytes_per_token(12, 3, 128, 2)
    full = MiniGPT(MiniGPTConfig.nanochat_style(50, block_size=64, n_layer=2, n_head=4, n_embd=32))
    gqa = MiniGPT(MiniGPTConfig.nanochat_style(50, block_size=64, n_layer=2, n_head=4, n_kv_head=1, n_embd=32))
    assert KVCache.for_model(gqa, 1).nbytes() * 4 == KVCache.for_model(full, 1).nbytes()
    assert gqa.num_params() < full.num_params()


def test_harness_basics():
    t = time_fn(lambda: sum(range(1000)), warmup=1, repeats=3)
    assert 0 < t["min_s"] <= t["median_s"] <= t["max_s"]
    assert peak_rss_bytes() > 1_000_000
    assert markdown_table([{"a": 1, "b": 2.5}], ["a", "b"]).count("\n") == 2
