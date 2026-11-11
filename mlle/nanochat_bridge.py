"""
Bridge between the course code and the pinned nanochat checkout.

nanochat is kept OUTSIDE this repository (external/nanochat, git-ignored) at the commit
recorded in docs/upstream-version.md. Labs that need it call `require_nanochat()`, which
puts the checkout on sys.path and warns if its commit differs from the pin, because line
references and behaviour described in the lessons are only guaranteed for that commit.

The second half of this file supports the Module 06 parity lab: copy the weights of a
nanochat GPT into a MiniGPT and get the same logits.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

PINNED_SHA = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
PINNED_DATE = "2026-07-03"
UPSTREAM_URL = "https://github.com/karpathy/nanochat"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def nanochat_dir() -> Path:
    """Location of the checkout: $NANOCHAT_DIR if set, else <course repo>/external/nanochat."""
    return Path(os.environ.get("NANOCHAT_DIR", _REPO_ROOT / "external" / "nanochat"))


def checkout_sha(path: Path | None = None) -> str | None:
    path = path or nanochat_dir()
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def require_nanochat(force_fp32: bool = False) -> Path:
    """Make `import nanochat` work. Raises a helpful error if the checkout is missing.

    force_fp32: set NANOCHAT_DTYPE=float32 BEFORE nanochat is imported. nanochat picks its
    compute dtype at import time (bf16 on modern CUDA GPUs); numerical comparison labs need
    fp32 on every machine."""
    path = nanochat_dir()
    if not (path / "nanochat" / "gpt.py").exists():
        raise RuntimeError(
            f"nanochat checkout not found at {path}.\n"
            "Run `make nanochat` (or scripts/setup/fetch_nanochat.sh) from the course root, "
            "or set NANOCHAT_DIR to an existing checkout."
        )
    sha = checkout_sha(path)
    if sha is not None and sha != PINNED_SHA:
        warnings.warn(
            f"nanochat checkout is at {sha[:7]}, but the course is written against {PINNED_SHA[:7]} "
            f"({PINNED_DATE}). File/line references and behaviour may differ.",
            stacklevel=2,
        )
    if force_fp32:
        if "nanochat.common" in sys.modules and os.environ.get("NANOCHAT_DTYPE") != "float32":
            raise RuntimeError("nanochat was already imported with a different dtype; set NANOCHAT_DTYPE=float32 first")
        os.environ["NANOCHAT_DTYPE"] = "float32"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


# ----------------------------------------------------------------------------
# Module 06: nanochat <-> MiniGPT parity


def build_nanochat_gpt(vocab_size=96, sequence_len=32, n_layer=4, n_head=2, n_embd=32, window_pattern="L", seed=0):
    """A tiny nanochat GPT on CPU in fp32, initialised exactly as base_train.py does it."""
    import torch

    require_nanochat(force_fp32=True)
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(seed)
    config = GPTConfig(sequence_len=sequence_len, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                       n_kv_head=n_head, n_embd=n_embd, window_pattern=window_pattern)
    model = GPT(config)
    model.init_weights()
    return model.eval()


def randomize_for_test(model, seed: int = 0, std: float = 0.3) -> None:
    """nanochat initialises several matrices to zero (blocks start as the identity), which
    would make a parity test trivially pass. Fill every matrix with noise instead."""
    import torch

    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=g) * std)


EXTRAS = ("resid_lambdas", "x0_lambdas", "smear", "backout", "value_embeds")


def neutralize_extras(model, keep: tuple[str, ...] = ()) -> None:
    """Switch off nanochat's five additions to the plain Transformer, except those in `keep`.

        resid_lambdas -> 1    (no per-layer rescaling of the residual stream)
        x0_lambdas    -> 0    (no re-injection of the initial embedding)
        smear_lambda  -> 0    (no mixing-in of the previous token's embedding)
        backout_lambda-> 0    (no subtraction of the mid-depth residual before the head)
        value_embeds  -> 0    (no token-dependent vector added to attention values)
    """
    import torch

    with torch.no_grad():
        if "resid_lambdas" not in keep:
            model.resid_lambdas.fill_(1.0)
        if "x0_lambdas" not in keep:
            model.x0_lambdas.fill_(0.0)
        if "smear" not in keep:
            model.smear_lambda.fill_(0.0)
        if "backout" not in keep:
            model.backout_lambda.fill_(0.0)
        if "value_embeds" not in keep:
            for ve in model.value_embeds.values():
                ve.weight.zero_()


def minigpt_from_nanochat(nc_model):
    """Build a MiniGPT (nanochat style) and copy the weights of a nanochat GPT into it."""
    import torch

    from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig

    c = nc_model.config
    assert set(c.window_pattern.upper()) == {"L"}, "parity mapping assumes full-context attention in every layer"
    config = MiniGPTConfig.nanochat_style(
        vocab_size=c.vocab_size, block_size=c.sequence_len, n_layer=c.n_layer, n_head=c.n_head,
        n_kv_head=c.n_kv_head, n_embd=c.n_embd,
    )
    mini = MiniGPT(config)
    with torch.no_grad():
        # nanochat pads the vocabulary to a multiple of 64 for speed; the padding rows are never used
        mini.wte.weight.copy_(nc_model.transformer.wte.weight[: c.vocab_size].float())
        mini.lm_head.weight.copy_(nc_model.lm_head.weight[: c.vocab_size].float())
        for mine, theirs in zip(mini.h, nc_model.transformer.h):
            mine.attn.c_q.weight.copy_(theirs.attn.c_q.weight)
            mine.attn.c_k.weight.copy_(theirs.attn.c_k.weight)
            mine.attn.c_v.weight.copy_(theirs.attn.c_v.weight)
            mine.attn.c_proj.weight.copy_(theirs.attn.c_proj.weight)
            mine.mlp.c_fc.weight.copy_(theirs.mlp.c_fc.weight)
            mine.mlp.c_proj.weight.copy_(theirs.mlp.c_proj.weight)
    return mini.eval()
