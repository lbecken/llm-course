"""
Lab 05 — Attention from first principles.   Tier A (about 1 minute, CPU)

    python labs/foundations/05_attention.py

1. The worked example from the lesson, number by number.
2. Three implementations (loops, matrices, PyTorch's fused kernel) give the same answer.
3. Without position information, attention is permutation-equivariant.
4. One attention head learns to be a dictionary (the lookup task) and its attention
   weights show where it reads from.
5. Cost: time and memory of naive attention as the sequence grows.
Success criterion: all "ok" lines print; lookup accuracy reaches > 0.98 and the average
attention weight on the answer pair rises from 1/5 to > 0.9.
"""

import time

import torch

from mlle.data.tasks import LookupSpec, lookup_batch
from mlle.foundations import attention as A
from mlle.foundations.experiments import train_lookup
from mlle.utils import ascii_heatmap

torch.set_printoptions(precision=4, sci_mode=False)

# %% 1. Worked example: 3 tokens, 2 dimensions
q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
v = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
scores = q @ k.T / (2**0.5)
out, w = A.attention(q, k, v, A.causal_mask(3))
print("1. worked example\nscores = Q K^T / sqrt(2):\n", scores)
print("weights after causal mask + softmax:\n", w)
print("output = weights @ V:\n", out)

# %% 2. Three implementations agree
torch.manual_seed(0)
T, D = 6, 8
q, k, v = torch.randn(T, D), torch.randn(T, D), torch.randn(T, D)
out_loop, w_loop = A.attention_loops(q, k, v)
out_mat, w_mat = A.attention(q, k, v, A.causal_mask(T))
out_ref = A.sdpa_reference(q[None, None], k[None, None], v[None, None])[0, 0]
print(f"\n2. max |loops - matrices| = {(out_loop - out_mat).abs().max():.1e}, max |matrices - torch SDPA| = {(out_mat - out_ref).abs().max():.1e} : ok")

# %% 3. Permutation equivariance without positions
mha = A.MultiHeadSelfAttention(16, 4)
x = torch.randn(1, 5, 16)
perm = torch.tensor([3, 0, 4, 1, 2])
diff = (mha(x, causal=False)[:, perm] - mha(x[:, perm], causal=False)).abs().max()
print(f"3. shuffle inputs -> outputs shuffled identically (max difference {diff:.1e}) : ok")

# %% 4. One head as a dictionary
print("\n4. lookup task: pairs (key, value) in random order, then a query key; answer = the value")
spec = LookupSpec()
model, history = train_lookup(spec, steps=1500)
tokens, labels, target_pos = lookup_batch(spec, 6, torch.Generator().manual_seed(123))
logits, weights = model(tokens)
print("\nattention of the query over the 5 pairs, for 6 fresh examples ('*' marks the pair holding the answer):")
for i in range(6):
    cells = "  ".join(f"{weights[i, j].item():.2f}{'*' if j == target_pos[i] else ' '}" for j in range(spec.num_pairs))
    print(f"  {cells}   predicted {logits[i].argmax().item()} / true {labels[i].item()}")
print(ascii_heatmap(weights.detach(), row_labels=[f"ex{i}" for i in range(6)], col_labels=[f"p{j}" for j in range(spec.num_pairs)]))

# %% 5. Quadratic cost
print("\n5. naive attention, one head, D=64: time and size of the (T, T) weight matrix")
for T in (256, 512, 1024, 2048):
    q = torch.randn(1, 1, T, 64)
    t0 = time.perf_counter()
    for _ in range(5):
        A.attention(q, q, q, A.causal_mask(T))
    dt = (time.perf_counter() - t0) / 5
    print(f"  T={T:5d}: {dt * 1e3:7.2f} ms   weights matrix {T * T * 4 / 2**20:7.2f} MiB (fp32)")
