"""Tests for Part II helpers: packing, data hygiene, evaluation statistics, decoding."""

import math
import random

import torch

from mlle import evaluation as ev
from mlle.data import quality as dq
from mlle.data.packing import pack_rows, simulate_packing
from mlle.inference import sampling as S


# ----------------------------------------------------------------------------- Module 08
def _lengths(n=20000, seed=0):
    rng = random.Random(seed)
    return [max(2, int(rng.lognormvariate(5.0, 1.0))) for _ in range(n)]


def test_stream_packing_wastes_nothing_but_rarely_starts_at_a_document():
    st = simulate_packing(_lengths(), 257, "stream", num_rows=300)
    assert st.utilization == 1.0 and st.tokens_cropped == 0 and st.pad_positions == 0
    assert st.rows_starting_with_document < 0.2 * st.rows


def test_bestfit_crop_is_full_and_bos_aligned_but_discards_tokens():
    st = simulate_packing(_lengths(), 257, "bestfit_crop", num_rows=300)
    assert st.utilization == 1.0 and st.pad_positions == 0
    assert st.rows_starting_with_document == st.rows
    assert 0.0 < st.cropped_fraction < 0.9


def test_bestfit_pad_discards_nothing_but_wastes_positions():
    st = simulate_packing(_lengths(), 2049, "bestfit_pad", num_rows=100)
    assert st.tokens_cropped == 0 and st.pad_positions >= 0
    assert st.utilization > 0.95  # best-fit packing keeps padding small


def test_pack_rows_masks_padding_and_keeps_every_token():
    docs = [[1] * 5, [2] * 3, [3] * 7, [4] * 2]
    rows, masks = pack_rows(docs, row_capacity=8, pad_id=0, strategy="bestfit_pad")
    assert all(len(r) == 8 for r in rows)
    assert sum(sum(m) for m in masks) == 17  # no token lost
    for r, m in zip(rows, masks):
        assert all((tok != 0) == bool(flag) for tok, flag in zip(r, m))


def test_duplicate_detection_and_contamination():
    base = "the quick brown fox jumps over the lazy dog near the quiet river bank today"
    docs = [base, base.upper() + "  ", base.replace("today", "tonight"), "completely different text about transformers and attention layers in models"]
    assert dq.exact_duplicates(docs) == [(0, 1)]
    near = dq.near_duplicates(docs, threshold=0.6)
    assert (0, 2) in [(i, j) for i, j, _ in near] and all(3 not in (i, j) for i, j, _ in near)
    rep = dq.contamination_rate(train_docs=docs[:1], eval_docs=[docs[2], docs[3]], n=5)
    assert rep["flagged"] == [0] and rep["rate"] == 0.5


# ----------------------------------------------------------------------------- Module 10
def test_bits_per_byte_and_errors():
    assert abs(ev.bits_per_byte(math.log(2) * 100, 100) - 1.0) < 1e-12
    assert abs(ev.accuracy_stderr(0.5, 100) - 0.05) < 1e-12
    m, s = ev.mean_std([1.0, 2.0, 3.0])
    assert m == 2.0 and abs(s - 1.0) < 1e-12


def test_bootstrap_and_paired_bootstrap():
    rng = random.Random(0)
    difficulty = [rng.random() for _ in range(400)]
    a = [float(rng.random() < 0.9 * d + 0.08) for d in difficulty]  # A is slightly better on every item
    b = [float(rng.random() < 0.9 * d) for d in difficulty]
    mean, lo, hi = ev.bootstrap_ci(a)
    assert lo < mean < hi and hi - lo < 0.15
    res = ev.paired_bootstrap(a, b)
    assert res["mean_diff"] > 0 and res["p_a_better"] > 0.9


def test_top_k_and_top_p_filters():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    assert torch.isinf(S.top_k_filter(logits, 2)[0]).tolist() == [False, False, True, True]
    # p = 0.7: 0.5 alone is not enough, 0.5 + 0.3 reaches it -> keep two
    assert torch.isinf(S.top_p_filter(logits, 0.7)[0]).tolist() == [False, False, True, True]
    # p = 0.4: the first token already exceeds it -> keep one
    assert torch.isinf(S.top_p_filter(logits, 0.4)[0]).tolist() == [False, True, True, True]
    # order of the vocabulary must not matter
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.isinf(S.top_p_filter(logits[:, perm], 0.7)[0]).tolist() == [True, False, True, False]


def test_sampling_limits():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert S.sample_next(logits, temperature=0.0).item() == 1
    g = torch.Generator().manual_seed(0)
    draws = [S.sample_next(logits, temperature=1.0, top_k=1, generator=g).item() for _ in range(20)]
    assert set(draws) == {1}
    pen = S.repetition_penalty(logits, torch.tensor([[1]]), 2.0)
    assert pen[0, 1] == 1.5 and pen[0, 0] == 1.0
    assert S.distinct_n([1, 2, 3, 1, 2, 3, 1, 2, 3], 3) < 0.5 and S.distinct_n(list(range(20)), 3) == 1.0


def test_pocket_recognizer_matches_generator_and_rejects_errors():
    from mlle.data import pocket

    rng = random.Random(5)
    for _ in range(2000):
        s = pocket.sample_sentence(rng)
        lp = pocket.sentence_logprob(s.text)
        assert lp is not None and abs(lp - s.logprob) < 1e-9
    assert pocket.sentence_logprob("the cats that chase the dog sleep .") is not None
    for bad in ["the cats that chase the dog sleeps .", "the cat sleep .", "cat the sleeps .", "the cat sleeps",
                "the cat that chase the dog sleeps .", "the unicorn sleeps .", "the cat sees .", "the small small cat sleeps ."]:
        assert pocket.sentence_logprob(bad) is None, bad
