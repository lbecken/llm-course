"""Tests for Part I: every claim the lessons make about the foundation code is checked here."""

import math
import random

import pytest
import torch
import torch.nn.functional as F

from mlle.data import pocket
from mlle.data.digits import NUM_CLASSES, NUM_PIXELS, make_digit_dataset
from mlle.data.tasks import AdditionTask, LookupSpec, lookup_batch
from mlle.foundations import attention as A
from mlle.foundations.bpe import BPETokenizer, SPECIAL_TOKENS, compression_ratio
from mlle.foundations.mlp import ManualMLP, cross_entropy, finite_difference_grad, train_digits
from mlle.foundations.ngram import CharTokenizer, NGramLM, perplexity
from mlle.foundations.optim import AdamW, Muon, lr_multiplier, newton_schulz
from mlle.foundations.positional import apply_rope, rope_cos_sin, sinusoidal_positions


# ----------------------------------------------------------------------------- Module 01
def test_manual_backprop_matches_autograd():
    x, y = make_digit_dataset(32, seed=1)
    model = ManualMLP(NUM_PIXELS, 16, NUM_CLASSES, seed=0)
    model.forward(x)
    model.backward(y)
    params = {k: v.clone().requires_grad_(True) for k, v in model.parameters().items()}
    h = torch.tanh(x @ params["W1"] + params["b1"])
    loss = F.cross_entropy(h @ params["W2"] + params["b2"], y)
    loss.backward()
    for name in params:
        assert torch.allclose(model.grads[name], params[name].grad, atol=1e-6), name


def test_manual_backprop_matches_finite_differences():
    x, y = make_digit_dataset(8, seed=2)
    x = x.double()
    model = ManualMLP(NUM_PIXELS, 8, NUM_CLASSES, seed=0)
    for name, p in model.parameters().items():
        setattr(model, name, p.double())
    model.forward(x)
    model.backward(y)
    for name, index in [("W1", (3, 2)), ("b1", (5,)), ("W2", (1, 7)), ("b2", (0,))]:
        numeric = finite_difference_grad(model, x, y, name, index, eps=1e-5)
        assert abs(numeric - model.grads[name][index].item()) < 1e-7


def test_cross_entropy_of_uniform_logits_is_log_c():
    logits = torch.zeros(4, 10)
    assert abs(cross_entropy(logits, torch.tensor([0, 3, 5, 9])).item() - math.log(10)) < 1e-6


def test_digit_mlp_learns():
    _, history = train_digits(steps=200, verbose=False)
    assert history[0]["test_acc"] < 0.3 and history[-1]["test_acc"] > 0.9


# ----------------------------------------------------------------------------- Module 02
def test_pocket_logprob_is_consistent_and_grammar_unambiguous():
    rng = random.Random(0)
    seen = {}
    for _ in range(3000):
        s = pocket.sample_sentence(rng)
        # a given string must always come with the same probability (one derivation per string)
        assert abs(seen.setdefault(s.text, s.logprob) - s.logprob) < 1e-9
        assert s.text.endswith(" .\n")


def test_pocket_entropy_rate_is_stable():
    a = pocket.entropy_rate(4000, seed=1)["nats_per_char"]
    b = pocket.entropy_rate(4000, seed=2)["nats_per_char"]
    assert abs(a - b) < 0.01 and 0.1 < a < 1.0


def test_ngram_models_improve_with_context_but_bigram_cannot_do_agreement():
    text = pocket.generate_corpus(3000, seed=0)
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    train, val = ids[: int(0.9 * len(ids))], ids[int(0.9 * len(ids)) :]
    ce = [NGramLM(order, tok.vocab_size).fit(train).cross_entropy(val[:5000]) for order in (0, 1, 3)]
    assert ce[0] > ce[1] > ce[2]
    assert ce[2] > pocket.entropy_rate(4000)["nats_per_char"]  # nobody beats the source entropy
    assert abs(perplexity(math.log(7)) - 7) < 1e-9


# ----------------------------------------------------------------------------- Module 03
def test_bpe_roundtrip_and_special_tokens():
    text = pocket.generate_corpus(300, seed=0) + "Unicode: naïve café 你好 🌍 12345"
    tok = BPETokenizer().train(text, vocab_size=256 + 64 + len(SPECIAL_TOKENS))
    for sample in [text[:500], "", "never seen: zebra 987654321 ß", "🌍🌍"]:
        assert tok.decode(tok.encode(sample)) == sample
    assert tok.vocab_size == 256 + 64 + len(SPECIAL_TOKENS)
    bos = tok.get_bos_token_id()
    assert tok.encode("hi", prepend="<|bos|>")[0] == bos
    # special-token text typed by a user must NOT become the special token
    assert bos not in tok.encode("<|bos|>")
    assert compression_ratio(tok, text) > 1.5


def test_bpe_save_load(tmp_path):
    text = pocket.generate_corpus(200, seed=1)
    tok = BPETokenizer().train(text, vocab_size=300)
    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    assert tok2.encode(text[:300]) == tok.encode(text[:300])


def test_bpe_first_merge_is_most_frequent_pair():
    tok = BPETokenizer().train("aaabdaaabac" * 10, vocab_size=257 + 0, special_tokens=[])
    assert tok.vocab[256] == b"aa"


# ----------------------------------------------------------------------------- Module 04
def test_sinusoidal_shape_and_range():
    pe = sinusoidal_positions(50, 16)
    assert pe.shape == (50, 16) and pe.abs().max() <= 1.0
    assert len({tuple(row.tolist()) for row in pe}) == 50  # every position has a distinct code


def test_rope_depends_only_on_relative_position():
    torch.manual_seed(0)
    D = 8
    q, k = torch.randn(D), torch.randn(D)
    cos, sin = rope_cos_sin(40, D)

    def score(m, n):
        return (apply_rope(q[None], cos[m : m + 1], sin[m : m + 1]) * apply_rope(k[None], cos[n : n + 1], sin[n : n + 1])).sum()

    assert torch.allclose(score(3, 1), score(13, 11), atol=1e-5)
    assert torch.allclose(score(20, 5), score(35, 20), atol=1e-5)
    assert not torch.allclose(score(3, 1), score(3, 2), atol=1e-3)
    # rotation preserves length
    x = torch.randn(2, 3, 40, D)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_offset_matches_slicing():
    cos, sin = rope_cos_sin(30, 8)
    cos2, sin2 = rope_cos_sin(10, 8, offset=20)
    assert torch.allclose(cos[20:], cos2) and torch.allclose(sin[20:], sin2)


# ----------------------------------------------------------------------------- Module 05
def test_attention_three_ways_agree():
    torch.manual_seed(0)
    T, D = 6, 8
    q, k, v = torch.randn(T, D), torch.randn(T, D), torch.randn(T, D)
    out_loop, w_loop = A.attention_loops(q, k, v)
    out_mat, w_mat = A.attention(q, k, v, A.causal_mask(T))
    out_ref = A.sdpa_reference(q[None, None], k[None, None], v[None, None])[0, 0]
    assert torch.allclose(out_loop, out_mat, atol=1e-6)
    assert torch.allclose(w_loop, w_mat, atol=1e-6)
    assert torch.allclose(out_mat, out_ref, atol=1e-6)
    assert torch.allclose(w_mat.sum(-1), torch.ones(T))
    assert torch.all(torch.triu(w_mat, diagonal=1) == 0)  # no weight on the future


def test_worked_example_from_lesson_05():
    """The numbers printed in modules/05-attention/lesson.md."""
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out, w = A.attention(q, k, v, A.causal_mask(3))
    expected_w = torch.tensor([[1.0, 0.0, 0.0], [0.3302, 0.6698, 0.0], [0.2483, 0.2483, 0.5035]])
    expected_out = torch.tensor([[1.0, 2.0], [2.3395, 3.3395], [3.5104, 4.5104]])
    assert torch.allclose(w, expected_w, atol=1e-4)
    assert torch.allclose(out, expected_out, atol=1e-4)


def test_causal_mask_variants():
    m = A.causal_mask(4)
    assert m.tolist() == [[True, False, False, False], [True, True, False, False], [True, True, True, False], [True, True, True, True]]
    assert A.causal_mask(1, 5).tolist() == [[True] * 5]  # one new query sees all cached keys
    w = A.causal_mask(5, window=2)
    assert w[4].tolist() == [False, False, True, True, True]


def test_attention_without_positions_is_permutation_equivariant():
    torch.manual_seed(0)
    mha = A.MultiHeadSelfAttention(16, 4)
    x = torch.randn(1, 5, 16)
    perm = torch.tensor([3, 0, 4, 1, 2])
    y = mha(x, causal=False)
    y_perm = mha(x[:, perm], causal=False)
    assert torch.allclose(y[:, perm], y_perm, atol=1e-5)


def test_mha_matches_torch_reference():
    torch.manual_seed(0)
    mha = A.MultiHeadSelfAttention(16, 4)
    x = torch.randn(2, 7, 16)
    y, w = mha(x, return_weights=True)
    assert w.shape == (2, 4, 7, 7)
    q, k, v = (mha.split_heads(p(x)) for p in (mha.q_proj, mha.k_proj, mha.v_proj))
    ref = A.sdpa_reference(q, k, v).transpose(1, 2).reshape(2, 7, 16)
    assert torch.allclose(y, mha.out_proj(ref), atol=1e-5)


def test_lookup_task_generator():
    spec = LookupSpec()
    tokens, labels, target_pos = lookup_batch(spec, 64, torch.Generator().manual_seed(0))
    assert tokens.shape == (64, spec.seq_len) and tokens.max() < spec.vocab_size
    pair = tokens[torch.arange(64), target_pos]
    assert torch.equal(pair % spec.num_values, labels)
    assert torch.equal(pair // spec.num_values, tokens[:, -1] - spec.num_keys * spec.num_values)


def test_addition_split_is_disjoint():
    task = AdditionTask(max_digits=2)
    train = {(p.a, p.b) for p in task.problems(2000, "train")}
    test = {(p.a, p.b) for p in task.problems(500, "test")}
    assert not train & test and len(test) > 20


# ----------------------------------------------------------------------------- Module 07
def test_adamw_matches_torch():
    torch.manual_seed(0)
    w1 = torch.randn(5, 3, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_(True)
    ours = AdamW([w1], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref = torch.optim.AdamW([w2], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    target = torch.randn(5, 3)
    for _ in range(20):
        for w, opt in ((w1, ours), (w2, ref)):
            loss = ((w - target) ** 2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
    assert torch.allclose(w1, w2, atol=1e-5)


@pytest.mark.parametrize("shape", [(16, 32), (32, 16), (24, 24)])
def test_newton_schulz_orthogonalizes(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape)
    X = newton_schulz(G, steps=5)
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.3  # singular values squeezed to about 1
    # same singular vectors as G: U^T X V should be (nearly) diagonal
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    core = U.T @ X @ Vh.T
    off = core - torch.diag(torch.diagonal(core))
    assert off.abs().max() < 1e-3


def test_muon_reduces_loss():
    torch.manual_seed(0)
    W = torch.zeros(8, 8, requires_grad=True)
    target = torch.randn(8, 8)
    opt = Muon([W], lr=0.05)
    first = None
    for _ in range(100):
        loss = ((W - target) ** 2).mean()
        first = first or loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.2 * first


def test_lr_schedule_shape():
    n = 1000
    m = [lr_multiplier(s, n) for s in range(n + 1)]
    assert abs(m[0] - 1 / 40) < 1e-9 and m[39] == 1.0 and m[200] == 1.0
    assert abs(m[n] - 0.05) < 1e-9 and m[700] < 1.0
    assert all(a >= b for a, b in zip(m[350:], m[351:]))  # monotone warmdown
