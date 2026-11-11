"""MiniGPT: shapes, causality, switches, cached decoding, and that it actually learns."""

import math

import pytest
import torch

from mlle.data import pocket
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import CharTokenizer
from mlle.foundations.train import TrainConfig, get_batch, split_tokens, train, train_on_text
from mlle.inference.kvcache import KVCache, generate_cached, kv_bytes_per_token

STYLES = {
    "gpt2": lambda **kw: MiniGPTConfig.gpt2_style(50, **kw),
    "nanochat": lambda **kw: MiniGPTConfig.nanochat_style(50, **kw),
}


@pytest.mark.parametrize("style", STYLES)
def test_shapes_and_initial_loss(style):
    torch.manual_seed(0)
    model = MiniGPT(STYLES[style](block_size=16, n_layer=2, n_head=2, n_embd=32))
    idx = torch.randint(0, 50, (3, 16))
    logits = model(idx)
    assert logits.shape == (3, 16, 50)
    loss = model(idx, idx)
    # an untrained model should be close to uniform: loss ~ ln(vocab_size)
    assert abs(loss.item() - math.log(50)) < 0.5


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "rope", "none"])
def test_causality_future_tokens_do_not_change_past_logits(style, pos):
    torch.manual_seed(0)
    model = MiniGPT(STYLES[style](block_size=16, n_layer=2, n_head=2, n_embd=32, pos=pos)).eval()
    for p in model.parameters():  # make zero-initialised projections non-trivial
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.2)
    idx = torch.randint(0, 50, (1, 12))
    idx2 = idx.clone()
    idx2[0, 8:] = torch.randint(0, 50, (4,))
    a, b = model(idx), model(idx2)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-3)


def test_tied_weights_share_storage_and_param_counts():
    tied = MiniGPT(MiniGPTConfig.gpt2_style(50, n_embd=32, n_layer=1))
    untied = MiniGPT(MiniGPTConfig.gpt2_style(50, n_embd=32, n_layer=1, tie_weights=False))
    assert tied.lm_head.weight is tied.wte.weight
    assert untied.num_params() - tied.num_params() == 50 * 32


def test_parameter_count_formula_gpt2_style():
    V, T, L, C = 50, 16, 3, 32
    model = MiniGPT(MiniGPTConfig.gpt2_style(V, block_size=T, n_layer=L, n_head=2, n_embd=C))
    per_block = (4 * C * C + 4 * C) + (8 * C * C + 5 * C) + 2 * (2 * C)  # attention + MLP + two LayerNorms
    expected = V * C + T * C + L * per_block + 2 * C  # embeddings (tied head) + blocks + final LayerNorm
    assert model.num_params() == expected


def test_record_internals():
    model = MiniGPT(MiniGPTConfig.nanochat_style(50, block_size=16, n_layer=3, n_head=4, n_embd=32))
    model(torch.randint(0, 50, (2, 10)), record=True)
    att = model.internals["attention"]
    assert len(att) == 3 and att[0].shape == (2, 4, 10, 10)
    assert torch.allclose(att[0].sum(-1), torch.ones(2, 4, 10), atol=1e-5)
    assert len(model.internals["residual_stream"]) == 4


@pytest.mark.parametrize("kwargs", [
    dict(), dict(pos="learned"), dict(pos="sinusoidal"), dict(n_head=4, n_kv_head=2), dict(window=3),
])
def test_cached_logits_equal_full_forward(kwargs):
    torch.manual_seed(0)
    cfg = MiniGPTConfig.nanochat_style(50, **{**dict(block_size=32, n_layer=3, n_head=2, n_embd=32), **kwargs})
    model = MiniGPT(cfg).eval()
    for p in model.parameters():
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.2)
    idx = torch.randint(0, 50, (2, 20))
    full = model(idx)
    cache = KVCache.for_model(model, batch_size=2)
    parts = [model(idx[:, :7], cache=cache)]  # prefill 7 tokens ...
    for t in range(7, 20):  # ... then decode one at a time
        parts.append(model(idx[:, t : t + 1], cache=cache))
    assert cache.pos == 20
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-4)


def test_cached_generation_equals_naive_generation():
    torch.manual_seed(0)
    model = MiniGPT(MiniGPTConfig.nanochat_style(50, block_size=48, n_layer=2, n_head=2, n_embd=32)).eval()
    for p in model.parameters():
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.2)
    idx = torch.randint(0, 50, (2, 5))
    assert torch.equal(model.generate(idx, 20, temperature=0.0), generate_cached(model, idx, 20, temperature=0.0))
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(model.generate(idx, 20, top_k=10, generator=g1), generate_cached(model, idx, 20, top_k=10, generator=g2))


def test_kv_cache_memory_formula():
    model = MiniGPT(MiniGPTConfig.nanochat_style(50, block_size=64, n_layer=3, n_head=4, n_kv_head=2, n_embd=32))
    cache = KVCache.for_model(model, batch_size=5, max_len=64)
    assert cache.nbytes() == 5 * 64 * kv_bytes_per_token(3, 2, 8, 4)
    big = cache.expand_batch(1) if False else KVCache.for_model(model, 1).expand_batch(7)
    assert big.k.shape[1] == 7


def test_batches_are_shifted_by_one():
    tokens = torch.arange(1000)
    x, y = get_batch(tokens, 4, 16, torch.Generator().manual_seed(0))
    assert torch.equal(y, x + 1)
    tr, va = split_tokens(tokens, 0.1)
    assert len(va) == 100 and tr[-1] < va[0]


@pytest.mark.slow
def test_minigpt_learns_pocket_and_approaches_entropy_floor():
    text = pocket.generate_corpus(6000, seed=0)
    floor = pocket.entropy_rate(5000)["nats_per_char"]
    model, tok, result = train_on_text(
        text, dict(block_size=64, n_layer=2, n_head=2, n_embd=48),
        TrainConfig(steps=500, batch_size=32, lr=3e-3, eval_every=250), verbose=False,
    )
    assert result.history[0]["val_loss"] > 2.5
    assert floor < result.final_val_loss < floor + 0.15


def test_muon_training_path_runs():
    text = pocket.generate_corpus(500, seed=0)
    tok = CharTokenizer(text)
    tokens = torch.tensor(tok.encode(text))
    tr, va = split_tokens(tokens)
    model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=32))
    res = train(model, tr, va, TrainConfig(steps=30, batch_size=16, eval_every=30, optimizer="muon", lr=3e-3), verbose=False)
    assert res.history[-1]["val_loss"] < res.history[0]["val_loss"]
