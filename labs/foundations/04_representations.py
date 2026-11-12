"""
Lab 04b — What do embeddings learn, and how do predictions form across layers?
Tier A (about 1 minute, CPU)

    python labs/foundations/04_representations.py

Trains two small word-level MiniGPTs on Pocket that differ ONLY in how the token
embedding is initialised (nanochat style: std 0.8;  small: std 0.02), then looks inside:
1. Nearest neighbours of INPUT embeddings (rows of wte) and OUTPUT embeddings (rows of lm_head).
2. Is grammatical NUMBER a direction?  cats - cat  vs  dogs - dog  vs  ...
3. "Logit lens": decode the residual stream after every layer with the final norm and
   output head, and measure how good the prediction already is.
Success criterion: both models reach the same loss; the small-init model shows a clear
number direction in its input embeddings (mean cosine > 0.5); per-layer loss decreases
with depth.
"""

import itertools

import torch
import torch.nn.functional as F

from mlle.data import pocket
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import WordTokenizer
from mlle.foundations.train import TrainConfig, split_tokens, train
from mlle.utils import seed_everything

text = pocket.generate_corpus(20000, seed=0)
tok = WordTokenizer(text)
tokens = torch.tensor(tok.encode(text))
train_tokens, val_tokens = split_tokens(tokens)
floor = pocket.entropy_rate()["nats_per_word"]
PROBES = ["cat", "cats", "sleeps", "sleep", "sees", "small"]


def neighbours(W: torch.Tensor, word: str, k: int = 4) -> str:
    E = F.normalize(W, dim=-1)  # unit-length rows
    sim = E @ E[tok.stoi[word]]  # cosine similarity of every token with `word`
    order = sim.argsort(descending=True)[1 : k + 1]  # skip the word itself
    return ", ".join(f"{tok.itos[i.item()]} ({sim[i]:.2f})" for i in order)


def number_direction(W: torch.Tensor) -> tuple[float, float]:
    """Mean and minimum cosine similarity between (plural - singular) vectors of different nouns."""
    diffs = [W[tok.stoi[pl]] - W[tok.stoi[sg]] for sg, pl in pocket.NOUNS]
    sims = [F.cosine_similarity(a, b, dim=0).item() for a, b in itertools.combinations(diffs, 2)]
    return sum(sims) / len(sims), min(sims)


models = {}
for name, init_std in [("nanochat init (wte std 0.8)", None), ("small init (wte std 0.02)", 0.02)]:
    seed_everything(0)
    model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=32, n_layer=3, n_head=2, n_embd=48))
    if init_std is not None:
        torch.nn.init.normal_(model.wte.weight, std=init_std)  # the ONLY difference between the two models
    result = train(model, train_tokens, val_tokens, TrainConfig(steps=600, batch_size=32, lr=3e-3, eval_every=600), verbose=False)
    models[name] = model.eval()
    print(f"=== {name}: val loss {result.final_val_loss:.4f} nats/word (entropy floor {floor:.4f})")
    for label, W in [("input  (wte)    ", model.wte.weight.detach()), ("output (lm_head)", model.lm_head.weight.detach())]:
        mean, low = number_direction(W)
        print(f"  {label} number direction: mean cosine {mean:+.3f}, min {low:+.3f}")
        for word in PROBES:
            print(f"      {word:7s} -> {neighbours(W, word)}")
    print()

# %% Logit lens: how early is the answer available?
model = models["nanochat init (wte std 0.8)"]
x = val_tokens[: 32 * 200].view(200, 32)
y = val_tokens[1 : 32 * 200 + 1].view(200, 32)
with torch.no_grad():
    model(x, record=True)
    print("loss when the residual stream is decoded after each stage (final norm + lm_head):")
    for i, h in enumerate(model.internals["residual_stream"]):
        logits = model.lm_head(model.norm_f(h))
        logits = model.config.logit_softcap * torch.tanh(logits / model.config.logit_softcap)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item()
        print(f"   after {'embedding' if i == 0 else f'block {i - 1}':10s}: {loss:.4f} nats/word")
print(f"   (entropy floor {floor:.4f}; uniform guess {torch.log(torch.tensor(float(tok.vocab_size))).item():.4f})")
