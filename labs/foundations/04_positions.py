"""
Lab 04 — Embeddings and three kinds of position information.   Tier A (about 2 minutes, CPU)

    python labs/foundations/04_positions.py            # demonstrations + the ablation
    python labs/foundations/04_positions.py --skip-training

1. An embedding lookup IS a matrix multiplication by a one-hot vector.
2. Sinusoidal codes: every position distinct, nearby positions similar.
3. RoPE: the query-key score depends only on the distance between two positions.
4. Ablation (predict first!): train the same MiniGPT with pos = none / learned /
   sinusoidal / rope on Pocket and compare validation loss and agreement accuracy.
Success criterion: parts 1-3 print "ok"; in part 4 every variant with position
information beats pos=none.
"""

import argparse

import torch

from mlle.data import pocket
from mlle.foundations.experiments import agreement_accuracy
from mlle.foundations.positional import apply_rope, rope_cos_sin, sinusoidal_positions
from mlle.foundations.train import TrainConfig, train_on_text
from mlle.utils import ascii_heatmap

ap = argparse.ArgumentParser()
ap.add_argument("--skip-training", action="store_true")
ap.add_argument("--steps", type=int, default=800)
args = ap.parse_args()

# %% 1. Embedding lookup == one-hot times matrix
torch.manual_seed(0)
E = torch.randn(10, 4)  # vocabulary of 10 tokens, 4-dimensional embeddings
ids = torch.tensor([3, 7, 3])
one_hot = torch.nn.functional.one_hot(ids, 10).float()  # (3, 10)
assert torch.equal(E[ids], one_hot @ E)
print("1. E[ids] == one_hot(ids) @ E : ok   (a lookup is a matmul that skips the zeros)")

# %% 2. Sinusoidal positions: similarity depends on distance
pe = sinusoidal_positions(32, 32)
sim = torch.nn.functional.cosine_similarity(pe[:, None, :], pe[None, :, :], dim=-1)  # (32, 32)
print("\n2. cosine similarity between sinusoidal position codes (rows/cols = positions 0..31, every 2nd shown)")
print(ascii_heatmap(sim[::2, ::2].clamp(0, 1), width=2))

# %% 3. RoPE: scores depend on relative position only
D = 8
q, k = torch.randn(D), torch.randn(D)
cos, sin = rope_cos_sin(64, D)


def score(m: int, n: int) -> float:
    return (apply_rope(q[None], cos[m : m + 1], sin[m : m + 1]) * apply_rope(k[None], cos[n : n + 1], sin[n : n + 1])).sum().item()


print("\n3. RoPE score <rot(q, m), rot(k, n)> for several (m, n):")
for m, n in [(5, 2), (25, 22), (60, 57), (5, 3), (5, 5)]:
    print(f"   m={m:2d} n={n:2d} distance={m - n}: {score(m, n):+.5f}")
assert abs(score(5, 2) - score(60, 57)) < 1e-4
print("   same distance -> same score: ok")

# %% 4. Ablation: what happens without position information?
if not args.skip_training:
    text = pocket.generate_corpus(10000, seed=0)
    floor = pocket.entropy_rate()["nats_per_char"]
    items = pocket.agreement_items(200, with_attractor=True)
    print(f"\n4. same model, same data, same seed; only `pos` changes (entropy floor {floor:.4f} nats/char)")
    print(f"{'pos':>12} {'val loss':>9} {'gap to floor':>13} {'agreement+attractor':>20}")
    for pos in ("none", "learned", "sinusoidal", "rope"):
        model, tok, res = train_on_text(
            text, dict(block_size=64, n_layer=2, n_head=2, n_embd=64, pos=pos),
            TrainConfig(steps=args.steps, batch_size=32, lr=3e-3, eval_every=args.steps), verbose=False,
        )
        print(f"{pos:>12} {res.final_val_loss:>9.4f} {res.final_val_loss - floor:>13.4f} {agreement_accuracy(model, tok, items):>20.2f}")
