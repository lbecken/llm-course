"""MiniGPT must reproduce the logits of the real nanochat model (pinned revision).

Skipped automatically when the nanochat checkout or its dependencies are missing
(e.g. in the lightweight CI job). Run locally with:  make nanochat && make test-nanochat"""

import pytest
import torch

from mlle import nanochat_bridge as nb

pytestmark = pytest.mark.nanochat


def _nanochat_available() -> bool:
    try:
        nb.require_nanochat(force_fp32=True)
        import nanochat.gpt  # noqa: F401
        return True
    except Exception:
        return False


needs_nanochat = pytest.mark.skipif(not _nanochat_available(), reason="pinned nanochat checkout not available")


@needs_nanochat
def test_logits_match_when_extras_are_neutralised():
    nc = nb.build_nanochat_gpt()
    nb.randomize_for_test(nc)
    nb.neutralize_extras(nc)
    mini = nb.minigpt_from_nanochat(nc)
    idx = torch.randint(0, nc.config.vocab_size, (2, 24), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref, ours = nc(idx), mini(idx)
    assert ref.abs().max() > 1.0  # the comparison is not trivially about zeros
    assert torch.allclose(ref, ours, atol=1e-4)


@needs_nanochat
@pytest.mark.parametrize("extra", nb.EXTRAS)
def test_each_extra_changes_the_logits(extra):
    """Sanity check of the mapping: re-enabling any single extra must break parity."""
    idx = torch.randint(0, 96, (2, 24), generator=torch.Generator().manual_seed(1))
    nc = nb.build_nanochat_gpt()
    nb.randomize_for_test(nc)
    with torch.no_grad():
        nc.smear_lambda.fill_(0.5)
        nc.backout_lambda.fill_(0.2)
    baseline = nb.build_nanochat_gpt()
    nb.randomize_for_test(baseline)
    nb.neutralize_extras(baseline)
    nb.neutralize_extras(nc, keep=(extra,))
    with torch.no_grad():
        assert (nc(idx) - baseline(idx)).abs().max() > 1e-2


@needs_nanochat
def test_pinned_sha_matches_checkout():
    sha = nb.checkout_sha()
    assert sha is None or sha == nb.PINNED_SHA
