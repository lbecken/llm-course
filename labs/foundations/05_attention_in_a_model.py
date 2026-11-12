"""
Lab 05b — Attention patterns in a trained model, and why looking is not enough.
Tier A (about 30 s, CPU)

    python labs/foundations/05_attention_in_a_model.py

A word-level MiniGPT is trained on Pocket. For the prefix

    the cats that chase the dog  ___

the next word must be a PLURAL verb (agreeing with "cats", not with "dog").
1. Print the attention weights from the last position, per layer and head.
2. Form a hypothesis from the picture about where the model reads "plural" from.
3. Test it by INTERVENTION: forbid specific attention edges and measure how the
   probability of a plural verb changes.
Success criterion: you can state which interventions change the prediction and which do
not, and explain why the attention picture alone could not have told you.
"""

import torch

from mlle.data import pocket
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import WordTokenizer
from mlle.foundations.train import TrainConfig, split_tokens, train
from mlle.utils import seed_everything

text = pocket.generate_corpus(20000, seed=0)
tok = WordTokenizer(text)
tokens = torch.tensor(tok.encode(text))
train_tokens, val_tokens = split_tokens(tokens)
seed_everything(0)
model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=48))
result = train(model, train_tokens, val_tokens, TrainConfig(steps=800, batch_size=32, lr=3e-3, eval_every=800), verbose=False)
model.eval()
print(f"trained: val loss {result.final_val_loss:.4f} nats/word (floor {pocket.entropy_rate()['nats_per_word']:.4f})\n")

PLURAL = [tok.stoi[pl] for _, pl in pocket.VERBS_INTRANSITIVE + pocket.VERBS_TRANSITIVE]
SINGULAR = [tok.stoi[sg] for sg, _ in pocket.VERBS_INTRANSITIVE + pocket.VERBS_TRANSITIVE]


def p_plural(words: list[str], blocked: dict[int, list[tuple[int, int]]] | None = None) -> float:
    """P(next word is a plural verb | next word is a verb), optionally with attention edges
    removed. `blocked` maps layer index -> list of (query_pos, key_pos)."""
    for i, block in enumerate(model.h):
        block.attn.blocked_edges = (blocked or {}).get(i)
    idx = torch.tensor([[tok.stoi["\n"]] + [tok.stoi[w] for w in words]])  # sentences follow a newline
    with torch.no_grad():
        probs = torch.softmax(model(idx, record=True)[0, -1], dim=-1)
    for block in model.h:
        block.attn.blocked_edges = None
    return (probs[PLURAL].sum() / (probs[PLURAL].sum() + probs[SINGULAR].sum())).item()


words = "the cats that chase the dog".split()
labels = ["\\n"] + words
last = len(labels) - 1
print(f"prefix: {' '.join(words)} ___     P(plural verb) = {p_plural(words):.3f}   (correct: plural)\n")

# %% 1. The picture
print("1. attention from the LAST position (the one predicting the verb) to each earlier position")
print("   " + " " * 16 + "".join(f"{w:>8s}" for w in labels))
for layer, att in enumerate(model.internals["attention"]):
    for head in range(att.shape[1]):
        print(f"   layer {layer} head {head}: " + "".join(f"{att[0, head, last, j].item():8.2f}" for j in range(len(labels))))

# %% 2-3. Interventions
pos = {w: i for i, w in enumerate(labels)}  # position of each word ("the" -> its LAST occurrence)
subject, rc_verb, attractor = 2, 4, 6  # cats, chase, dog
all_layers = lambda edges: {i: edges for i in range(len(model.h))}
experiments = [
    ("nothing blocked (baseline)", None),
    ("last -> 'cats' (the subject), all layers", all_layers([(last, subject)])),
    ("last -> 'chase' (verb inside the relative clause), all layers", all_layers([(last, rc_verb)])),
    ("last -> 'cats' AND last -> 'chase', all layers", all_layers([(last, subject), (last, rc_verb)])),
    ("EVERY position -> 'cats' and 'chase' (no route for the number at all)",
     all_layers([(q, k) for q in range(len(labels)) for k in (subject, rc_verb) if q > k])),
    ("control: last -> first 'the', all layers", all_layers([(last, 1)])),
]
print("\n2. interventions: forbid attention edges, then re-measure")
for name, blocked in experiments:
    print(f"   {name:72s} P(plural verb) = {p_plural(words, blocked):.3f}")

# %% 4. The same test on a sentence where the two sources of number information disagree in position only
words2 = "the cat that chases the dogs".split()
print(f"\n3. mirror-image prefix: {' '.join(words2)} ___   P(plural verb) = {p_plural(words2):.3f}   (correct: singular)")
