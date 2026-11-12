"""
Lab 20 — An experiment loop that improves a training recipe, and what keeps it honest.
Tier A (about 3 minutes on a 4-core CPU: roughly 80 short training runs)

    python labs/agents/20_experiment_loop.py

The "agent" here proposes ONE change to a training configuration, the loop trains a small
MiniGPT on Pocket with it, and keeps the change only if the validation loss improves by
more than the measured seed-to-seed noise. Everything is appended to a ledger. The test
set is used exactly once, at the end.

  1  Measure the noise first.
  2  The bounded loop on real hyperparameters (learning rate, width, depth, warm-up).
  3  The same loop searching over a parameter that CANNOT matter (it only reseeds the
     run), with and without the noise threshold: how much "improvement" does unguarded
     search find in pure noise, and does it survive on the test set?
  4  A proposer that tries to step outside its search space, and what the loop does.

What is being improved is a configuration file. The loop's own code, search space,
budget and evaluation are fixed and outside its reach.
Predict before running part 3: 20 trials on a parameter with no effect, accepting any
improvement. How many will be accepted, and how large will the "gain" be compared with
the noise?
"""

import torch

from mlle.agents.experiment import estimate_noise, format_ledger, llm_proposer, random_proposer, run_experiment_loop
from mlle.agents.llm import ScriptedLLM
from mlle.data import pocket
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import CharTokenizer
from mlle.foundations.train import TrainConfig, estimate_loss, train
from mlle.utils import REPO_ROOT

text = pocket.generate_corpus(4000, seed=0)
tok = CharTokenizer(text)
tokens = torch.tensor(tok.encode(text))
n = len(tokens)
train_tokens, val_tokens, test_tokens = tokens[: int(0.8 * n)], tokens[int(0.8 * n) : int(0.9 * n)], tokens[int(0.9 * n) :]
STEPS = 150
runs = 0


def objective(config: dict, seed: int) -> dict:
    """Train for a fixed number of steps; score = minus the loss (higher is better)."""
    global runs
    runs += 1
    torch.manual_seed(seed + 1000 * config.get("reseed", 0))
    model = MiniGPT(MiniGPTConfig.nanochat_style(tok.vocab_size, block_size=32, n_layer=config["n_layer"], n_head=2, n_embd=config["n_embd"]))
    train(model, train_tokens, val_tokens, TrainConfig(steps=STEPS, batch_size=16, lr=config["lr"], warmup_steps=config["warmup"],
                                                         eval_every=10**9, seed=seed + 1000 * config.get("reseed", 0)), verbose=False)
    return {"val": -estimate_loss(model, val_tokens, 16, 10), "test": -estimate_loss(model, test_tokens, 16, 10)}


start = {"lr": 3e-4, "n_embd": 32, "n_layer": 1, "warmup": 0}
space = {"lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "n_embd": [32, 48, 64], "n_layer": [1, 2], "warmup": [0, 20]}
ledger = REPO_ROOT / "results" / "runs" / "part5" / "20_ledger.jsonl"
ledger.parent.mkdir(parents=True, exist_ok=True)
ledger.unlink(missing_ok=True)

# %% 1. Noise
noise = estimate_noise(objective, start, seeds=(101, 102, 103, 104, 105))
print(f"1. the starting configuration {start}, five seeds: validation scores differ by up to {noise:.4f} nats/char")
print("   A change must beat the incumbent by MORE than this to count as an improvement.")

# %% 2. The bounded loop
print(f"\n2. bounded loop: at most 14 trials, one change per trial, {STEPS} training steps each")
result = run_experiment_loop(objective, start, space, random_proposer(space), max_trials=14, max_seconds=240, min_improvement=noise,
                             ledger_path=ledger, seed=0)
print(format_ledger(result))
base = objective(start, 9999)
print(f"   test score of the starting configuration (same fresh seed): {base['test']:.4f};  of the chosen one: {result.test_score:.4f}")
print(f"   ledger with every trial: {ledger.relative_to(REPO_ROOT)}")

# %% 3. Searching in noise
print("\n3. searching over 'reseed', which changes nothing but the random seed (20 trials each)")
good = dict(result.best_config, reseed=0)
null_space = {"reseed": list(range(1, 40))}
print(f"   {'acceptance rule':44s} {'accepted':>9} {'validation gain':>16} {'test gain':>10}")
ref = objective(good, 0)
ref_test = objective(good, 9999)["test"]
for label, threshold in (("accept any improvement (threshold 0)", 0.0), (f"accept only gains above the noise ({noise:.4f})", noise)):
    r = run_experiment_loop(objective, good, null_space, random_proposer(null_space), max_trials=20, max_seconds=240, min_improvement=threshold, seed=0)
    accepted = sum(t.accepted for t in r.trials)
    print(f"   {label:44s} {accepted:>9d} {r.best_val - ref['val']:>+16.4f} {r.test_score - ref_test:>+10.4f}")
print("   Every accepted change in this part is, by construction, noise.")

# %% 4. A proposer that tries to leave the sandbox
print("\n4. a proposer (scripted stand-in for a model) that asks for things outside the search space")
script = ["CHANGE: steps=100000", "CHANGE: eval_set=train", "CHANGE: lr=0.5", "CHANGE: n_embd=48", "CHANGE: max_trials=1000"]
r = run_experiment_loop(objective, start, space, llm_proposer(ScriptedLLM(script), space), max_trials=5, max_seconds=120, min_improvement=noise, seed=0)
print(format_ledger(r))
print(f"\n{runs} training runs in total. The loop changed a configuration. It could not change its budget, its search space, its evaluation or itself.")
