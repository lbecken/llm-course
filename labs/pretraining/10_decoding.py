"""
Lab 10a — Decoding strategies, judged against ground truth.   Tier A (about 1 minute, CPU)

    python labs/pretraining/10_decoding.py

Trains a small MiniGPT on Pocket, then generates text with greedy decoding and a range
of temperatures, top-k and top-p settings. Because Pocket's grammar is known, every
generated sentence gets an exact verdict:

    valid       fraction of sentences that are grammatical (including number agreement)
    unique      fraction of DISTINCT sentences among the valid ones (diversity)
    nats/char   cross-entropy of the valid sentences under the TRUE grammar: how typical
                the generated text is (the language itself scores 0.314)

Success criterion: you can read the quality/diversity trade-off from the table and say
which setting you would choose for (a) a fact-like task and (b) creative variety.
"""

import torch

from mlle.data import pocket
from mlle.foundations.train import TrainConfig, train_on_text
from mlle.inference.sampling import generate

text = pocket.generate_corpus(20000, seed=0)
model, tok, result = train_on_text(text, dict(block_size=64, n_layer=2, n_head=2, n_embd=64),
                                   TrainConfig(steps=800, batch_size=32, lr=1e-2, eval_every=800), verbose=False)
model.eval()
print(f"model: val loss {result.final_val_loss:.4f} nats/char (floor {pocket.entropy_rate()['nats_per_char']:.4f})\n")

settings = [
    ("greedy (temperature 0)", dict(temperature=0.0)),
    ("temperature 0.5", dict(temperature=0.5)),
    ("temperature 0.8", dict(temperature=0.8)),
    ("temperature 1.0 (the model's own distribution)", dict(temperature=1.0)),
    ("temperature 1.5", dict(temperature=1.5)),
    ("temperature 2.5", dict(temperature=2.5)),
    ("temperature 1.5 + top-k 3", dict(temperature=1.5, top_k=3)),
    ("temperature 1.5 + top-p 0.9", dict(temperature=1.5, top_p=0.9)),
    ("temperature 1.0 + top-p 0.9", dict(temperature=1.0, top_p=0.9)),
]

N_PROMPTS, NEW = 64, 60
prompt = torch.tensor([tok.encode("\n")] * N_PROMPTS)  # start of a sentence
print(f"{'setting':48s} {'valid':>6} {'unique':>7} {'nats/char':>10}   example")
for name, kw in settings:
    g = torch.Generator().manual_seed(0)
    out = generate(model, prompt, NEW, generator=g, **kw)
    sentences = []
    for row in out:
        pieces = tok.decode(row[1:]).split("\n")
        sentences.extend(pieces[:-1])  # drop the last, possibly unfinished, sentence
    verdicts = [pocket.sentence_logprob(s) for s in sentences]
    valid = [(s, lp) for s, lp in zip(sentences, verdicts) if lp is not None]
    valid_rate = len(valid) / max(1, len(sentences))
    unique = len({s for s, _ in valid}) / max(1, len(valid))
    chars = sum(len(s) + 1 for s, _ in valid)
    nats = -sum(lp for _, lp in valid) / max(1, chars)
    example = next((s for s, lp in zip(sentences, verdicts) if lp is None), sentences[0] if sentences else "")
    tag = "INVALID: " if valid_rate < 1 and pocket.sentence_logprob(example) is None else ""
    print(f"{name:48s} {valid_rate:>6.2f} {unique:>7.2f} {nats:>10.3f}   {tag}{example[:60]}")
