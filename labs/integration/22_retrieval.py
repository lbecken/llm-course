"""
Lab 22 — Knowledge in the weights versus knowledge in the context (retrieval).
Tier A (about 2 minutes, CPU)

    python labs/integration/22_retrieval.py

A world of 26 facts "key=value" (one-letter keys, two-digit values). Three small models
of identical size are trained from scratch:

    closed-book           memorises the table:                          q.k=41
    open-book             is given ONE retrieved fact, then the question:  k=41.k=41
                          (values in its training data are random and never repeat: the
                           only way to answer is to READ the retrieved fact)
    open-book, checking   as above, but in 30 % of its training examples the retrieved fact
                          is about a DIFFERENT key, and the right reply is "--" (abstain)

Then the world changes: every value in the table is replaced. The retriever is ordinary
code: an exact key match over the current table. Measured: accuracy on the old and new
table, and what each open-book model does when the retriever returns the wrong fact or
a wrong value.
Predict before running: after the table changes, which model is right without
retraining, and what does the other one answer?
"""

import random

import torch

from mlle.finetune.data import documents_to_stream, make_tokenizer
from mlle.finetune.sft import complete
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.train import TrainConfig, split_tokens, train
from mlle.utils import REPO_ROOT, JsonlLogger

log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part6" / "22_retrieval.jsonl")
tok = make_tokenizer()
bos = tok.get_bos_token_id()
KEYS = list("abcdefghijklmnopqrstuvwxyz")
STEPS = 1500
new_val = lambda r: f"{r.randint(0, 99):02d}"  # noqa: E731
rng = random.Random(0)
old_table = {k: new_val(rng) for k in KEYS}
new_table = {k: f"{(int(old_table[k]) + rng.randint(1, 99)) % 100:02d}" for k in KEYS}  # every value changes


def train_model(docs):
    train_tokens, val_tokens = split_tokens(documents_to_stream(tok, docs))
    torch.manual_seed(0)
    model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=32, n_layer=2, n_head=4, n_embd=96))
    train(model, train_tokens, val_tokens, TrainConfig(steps=STEPS, batch_size=32, lr=3e-3, eval_every=STEPS), verbose=False)
    return model.eval()


def answer(model, prompt):
    return tok.decode(complete(model, [bos] + tok.encode(prompt), {bos}, max_new_tokens=2)[0])


def open_docs(r, n, mismatch):
    docs = []
    for _ in range(n):
        k, v = r.choice(KEYS), new_val(r)
        if r.random() < mismatch:  # the retriever brought a fact about another key: the honest reply is "--"
            other = r.choice([x for x in KEYS if x != k])
            docs.append(f"{other}={v}.{k}=--")
        else:
            docs.append(f"{k}={v}.{k}={v}")
    return docs


print(f"training three models (2 layers x 96, {STEPS} steps each) ...", flush=True)
r = random.Random(1)
models = {"closed-book": train_model([f"q.{k}={old_table[k]}" for k in KEYS for _ in range(400)]),
          "open-book": train_model(open_docs(r, 30000, 0.0)),
          "open-book, checking": train_model(open_docs(r, 30000, 0.3))}

r = random.Random(2)
questions = KEYS * 8
n = len(questions)
print(f"\n{'model':22s} {'condition':46s} {'right':>6} {'abstains':>9} {'repeats the context':>20}")


def report(name, condition, make_prompt, truth, context_value=None):
    right = abstain = repeat = 0
    for k in questions:
        prompt, shown = make_prompt(k)
        out = answer(models[name], prompt)
        right += out == truth[k]
        abstain += out == "--"
        repeat += shown is not None and out == shown
    print(f"{name:22s} {condition:46s} {right / n:>6.2f} {abstain / n:>9.2f} {repeat / n:>20.2f}", flush=True)
    log.log(model=name, condition=condition, right=right / n, abstains=abstain / n, repeats_context=repeat / n)


report("closed-book", "old table (as trained)", lambda k: (f"q.{k}=", None), old_table)
report("closed-book", "NEW table", lambda k: (f"q.{k}=", old_table[k]), new_table)  # "context" column: the stale value
for name in ("open-book", "open-book, checking"):
    report(name, "retriever correct, old table", lambda k: (f"{k}={old_table[k]}.{k}=", old_table[k]), old_table)
    report(name, "retriever correct, NEW table (no retraining)", lambda k: (f"{k}={new_table[k]}.{k}=", new_table[k]), new_table)

    def wrong_fact(k):
        other = r.choice([x for x in KEYS if x != k])
        return f"{other}={new_table[other]}.{k}=", new_table[other]

    report(name, "retriever returns a fact about ANOTHER key", wrong_fact, new_table)

    def wrong_value(k):
        lie = f"{(int(new_table[k]) + 37) % 100:02d}"
        return f"{k}={lie}.{k}=", lie

    report(name, "retriever returns a WRONG value for the key", wrong_value, new_table)

print("\n(closed-book 'repeats the context' = it still answers with the OLD value)")
print("Weights hold what was true at training time. Context holds whatever the application puts there, right or wrong.")
