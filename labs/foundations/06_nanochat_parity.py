"""
Lab 06b — Reproduce nanochat's logits with MiniGPT.   Tier A (seconds, CPU). Needs `make env-cpu`.

    python labs/foundations/06_nanochat_parity.py

Builds a tiny nanochat GPT (the real class from the pinned checkout), switches off its
five additions to the plain Transformer, copies the weights into MiniGPT and compares
logits. Then switches each addition back on, one at a time, and measures how much the
logits move.
Success criterion: parity error below 1e-4; every single extra changes the logits.
"""

import torch

from mlle import nanochat_bridge as nb

idx = torch.randint(0, 96, (2, 24), generator=torch.Generator().manual_seed(1))

nc = nb.build_nanochat_gpt(vocab_size=96, sequence_len=32, n_layer=4, n_head=2, n_embd=32)
nb.randomize_for_test(nc)  # nanochat zero-initialises its output projections; make the test non-trivial
nb.neutralize_extras(nc)
mini = nb.minigpt_from_nanochat(nc)
with torch.no_grad():
    ref, ours = nc(idx), mini(idx)
print(f"nanochat parameters: {sum(p.numel() for p in nc.parameters()):,}   MiniGPT parameters: {mini.num_params():,}")
print(f"largest |logit|: {ref.abs().max():.3f}   largest difference nanochat vs MiniGPT: {(ref - ours).abs().max():.2e}")
assert torch.allclose(ref, ours, atol=1e-4)
print("parity: ok  (with the five extras neutralised, nanochat IS the MiniGPT nanochat-style model)\n")

print("re-enable one extra at a time (smear_lambda=0.5 and backout_lambda=0.2 for visibility):")
for extra in nb.EXTRAS:
    m = nb.build_nanochat_gpt(vocab_size=96, sequence_len=32, n_layer=4, n_head=2, n_embd=32)
    nb.randomize_for_test(m)
    with torch.no_grad():
        m.smear_lambda.fill_(0.5)
        m.backout_lambda.fill_(0.2)
    nb.neutralize_extras(m, keep=(extra,))
    with torch.no_grad():
        delta = (m(idx) - ref).abs().max().item()
    print(f"  {extra:14s} -> max logit change {delta:.3f}")

counts = nc.num_scaling_params()
print("\nwhere nanochat's parameters are (tiny config):", counts)
