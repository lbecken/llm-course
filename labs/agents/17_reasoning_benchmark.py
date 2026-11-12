"""
Lab 17 — Thinking as extra computation: scratchpads, voting, verification.
Tier A (about 5 minutes on a 4-core CPU; part B alone runs in seconds)

    python labs/agents/17_reasoning_benchmark.py
    python labs/agents/17_reasoning_benchmark.py --part b

Part A: REAL small models, trained here.
  A1  A sequential task: chains like 3*4+2-1 evaluated left to right modulo 5. Two models of
      identical size are trained from scratch on the same problems:
          direct      3*4+2-1=3                     (answer immediately)
          scratchpad  3*4+2-1=2+2-1=4-1=3           (one reduction at a time, then the answer)
      plus a control: the direct format trained on short chains only.
  A2  Majority voting over samples of the direct model: can voting create an answer the
      model cannot compute?
  A3  Voting on the under-trained chat model of Part IV (3-digit addition): accuracy of
      greedy, of one sample, of the majority of k samples; and agreement among samples as
      a confidence estimate.

Part B: a SIMULATED solver with stated error rates (mlle.agents.llm.NoisySolver), to study
orchestration at equal cost: direct, decomposition, voting, verify-and-retry with a
model check or an exact check, and "just use the tool". Its numbers are consequences of
assumptions, not measurements of any model.

Predict before running A1: the two models have the same weights budget and see the same
problems. Which is more accurate on 4-step chains, and by how much?
"""

import argparse
import random
from collections import Counter

import torch

from mlle.agents import reasoning as R
from mlle.agents.llm import NoisySolver
from mlle.utils import REPO_ROOT, JsonlLogger

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["a", "b", "both"], default="both")
ap.add_argument("--steps", type=int, default=2500)
args = ap.parse_args()
log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part5" / "17_reasoning.jsonl")


def part_a():
    from mlle.data.tasks import mod_chain, mod_chain_document
    from mlle.finetune.base import get_base_model, get_chat_model
    from mlle.finetune.data import conversations, documents_to_stream, extract_answer, make_tokenizer
    from mlle.finetune.sft import chat_accuracy, complete
    from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
    from mlle.foundations.train import TrainConfig, split_tokens, train

    tok = make_tokenizer()
    bos = tok.get_bos_token_id()

    def train_model(style, max_ops):
        rng = random.Random(0)
        stream = documents_to_stream(tok, [mod_chain_document(rng, style, 1, max_ops) for _ in range(40000)])
        train_tokens, val_tokens = split_tokens(stream)
        torch.manual_seed(0)
        model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=64, n_layer=2, n_head=4, n_embd=96))
        train(model, train_tokens, val_tokens, TrainConfig(steps=args.steps, batch_size=32, lr=3e-3, eval_every=args.steps), verbose=False)
        return model.eval()

    def evaluate(model, n_ops, n=200):
        rng = random.Random(100 + n_ops)
        right = tokens = 0
        for _ in range(n):
            expr, steps = mod_chain(rng, n_ops)
            out = complete(model, [bos] + tok.encode(expr + "="), {bos}, max_new_tokens=48)[0]
            tokens += len(out)
            right += tok.decode(out).split("=")[-1] == steps[-1]
        return right / n, tokens / n

    print(f"\nA1. chains modulo 5 (chance = 0.20); 2 layers x 96, {args.steps} steps each, 200 held-out chains per cell", flush=True)
    variants = [("direct, trained on 1-5 operations", "direct", 5), ("scratchpad, trained on 1-5 operations", "scratchpad", 5),
                ("direct, trained on 1-2 operations only", "direct", 2)]
    models = {}
    print(f"{'model':40s} " + " ".join(f"{str(k) + ' ops':>7}" for k in range(1, 7)) + f" {'tokens generated at 5 ops':>27}")
    for name, style, max_ops in variants:
        models[name] = train_model(style, max_ops)
        cells = [evaluate(models[name], k) for k in range(1, 7)]
        print(f"{name:40s} " + " ".join(f"{acc:>7.3f}" for acc, _ in cells) + f" {cells[4][1]:>27.1f}", flush=True)
        log.log(part="A1", model=name, **{f"acc_{k}ops": cells[k - 1][0] for k in range(1, 7)}, tokens_5ops=cells[4][1])
    print("   (6 operations is longer than anything in training)")
    rng = random.Random(7)
    expr, steps = mod_chain(rng, 4)
    for name in list(models)[:2]:
        out = complete(models[name], [bos] + tok.encode(expr + "="), {bos}, max_new_tokens=48)[0]
        print(f"   {name.split(',')[0]:11s} {expr}={tok.decode(out)}     (truth: {steps[-1]})")

    def vote_table(sample_fn, problems, ks, label):
        stats, agreement = Counter(), []
        for prompt, truth in problems:
            samples = sample_fn(prompt)
            for k in ks:
                stats[f"vote@{k}"] += Counter(samples[:k]).most_common(1)[0][0] == truth
                stats[f"any@{k}"] += truth in samples[:k]
            top, count = Counter(samples).most_common(1)[0]
            agreement.append((count / len(samples), top == truth))
        n = len(problems)
        print(f"   {label}")
        print("   " + " ".join(f"{'k=' + str(k):>8}" for k in ks))
        print("   " + " ".join(f"{stats[f'vote@{k}'] / n:>8.3f}" for k in ks) + "   majority vote is right")
        print("   " + " ".join(f"{stats[f'any@{k}'] / n:>8.3f}" for k in ks) + "   the right answer is AMONG the samples")
        log.log(part=label, **{k: v / n for k, v in stats.items()})
        return agreement

    print("\nA2. voting on the direct model (4 operations, temperature 1.0)")
    g = torch.Generator().manual_seed(0)
    direct = models["direct, trained on 1-5 operations"]
    rng = random.Random(104)
    problems = []
    for _ in range(200):
        expr, steps = mod_chain(rng, 4)
        problems.append(([bos] + tok.encode(expr + "="), steps[-1]))
    vote_table(lambda p: [tok.decode(s) for s in complete(direct, p, {bos}, 4, temperature=1.0, num_samples=15, generator=g)],
               problems, (1, 5, 15), "200 chains")

    print("\nA3. voting on the under-trained chat model of Part IV (3-digit addition, 300 held-out questions)")
    base, world = get_base_model(verbose=False)
    chat = get_chat_model(base, world, verbose=False)
    ctok = world.tok
    stop = {ctok.encode_special("<|assistant_end|>"), ctok.get_bos_token_id()}
    held_out = conversations(world.task, 300, "test", 1)
    print(f"   greedy decoding: {chat_accuracy(chat, ctok, held_out):.3f}")
    g = torch.Generator().manual_seed(0)
    agreement = vote_table(lambda p: [extract_answer(ctok.decode(s)) for s in complete(chat, p, stop, 8, temperature=1.0, num_samples=25, generator=g)],
                           [(ctok.render_for_completion(c), c["problem"].answer) for c in held_out], (1, 5, 15, 25), "samples at temperature 1.0")
    print("   agreement among the 25 samples as a confidence estimate:")
    for label, keep in (("at least 80 % agree", lambda a: a >= 0.8), ("50-80 % agree", lambda a: 0.5 <= a < 0.8), ("fewer than 50 % agree", lambda a: a < 0.5)):
        group = [ok for a, ok in agreement if keep(a)]
        print(f"      {label:22s} {len(group):>4} questions, majority answer right in {sum(group) / max(1, len(group)):.3f}")
        log.log(part="A3 agreement", bucket=label, n=len(group), accuracy=sum(group) / max(1, len(group)))


def part_b():
    print("\nB. SIMULATED solver (assumed error rates: 5 % per single step, 15 % per operation when answering a whole chain at once,")
    print("   10 % wrong checks). 400 problems per row. Numbers follow from these assumptions; they are not measurements of a model.")

    def row(name, strategy, problems, solver_kw=None, **kw):
        llm = NoisySolver(seed=0, **(solver_kw or {}))
        if strategy == "verify_model":
            strategy, kw = R.verify_and_retry, dict(checker=R.llm_checker(llm), check_cost=1)
        r = R.run_benchmark(strategy, llm, problems, **kw)
        log.log(part="B", strategy=name, **r, **(solver_kw or {}))
        return r

    strategies = [("direct", R.direct, {}), ("decompose (one call per step)", R.decompose, {}), ("vote, k=5", R.self_consistency, dict(k=5)),
                  ("vote, k=15", R.self_consistency, dict(k=15)), ("verify and retry, model checks", "verify_model", {}),
                  ("verify and retry, exact check", R.verify_and_retry, dict(checker=R.tool_checker)), ("tool only (no model)", R.tool_only, {})]
    print(f"\nB1. {'strategy':32s} {'4 ops: accuracy':>16} {'calls':>6} {'8 ops: accuracy':>16} {'calls':>6}")
    p4, p8 = R.make_chain_problems(400, 4, seed=1), R.make_chain_problems(400, 8, seed=1)
    for name, strategy, kw in strategies:
        a, b = row(name, strategy, p4, **kw), row(name, strategy, p8, **kw)
        print(f"    {name:32s} {a['accuracy']:>16.3f} {a['calls_per_problem']:>6.1f} {b['accuracy']:>16.3f} {b['calls_per_problem']:>6.1f}")

    print("\nB2. what if errors are SYSTEMATIC (the same wrong answer every time the question is asked)?  6 operations")
    print(f"    {'share of errors that are systematic':38s} {'direct':>7} {'vote k=15':>10} {'verify, model checks':>21} {'verify, exact check':>20}")
    p6 = R.make_chain_problems(400, 6, seed=2)
    for share in (0.0, 0.5, 1.0):
        cells = [row(n, s, p6, solver_kw=dict(systematic=share), **kw)["accuracy"] for n, s, kw in
                 (strategies[0], strategies[3], strategies[4], strategies[5])]
        print(f"    {share:<38.1f} {cells[0]:>7.3f} {cells[1]:>10.3f} {cells[2]:>21.3f} {cells[3]:>20.3f}")
    print("    Voting and re-sampling repair RANDOM errors. Against a consistent error they do nothing, at 15 times the cost.")


if args.part in ("a", "both"):
    part_a()
if args.part in ("b", "both"):
    part_b()
