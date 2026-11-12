"""
Lab 15a — Learning from a reward: REINFORCE, and what a flawed reward does.
Tier A (about 4 minutes on a 4-core CPU; uses the cached base and chat models)

    python labs/finetuning/15_reinforce.py
    python labs/finetuning/15_reinforce.py --lr 1e-3        # watch it destabilise

Starting point: a deliberately under-trained chat model (about half of its answers are
right). Training signal: for each prompt, sample 8 answers and score each with a
PROGRAM. No reference answer enters the loss. Three runs from the same checkpoint, on the
same 300 prompts:

    A  reward = 1 if the answer is exactly right            (the true objective)
    B  reward = 1 if the LAST DIGIT and the length are right (a lazy checker)
    C  supervised fine-tuning on the same 300 prompts with their reference answers

Reported at 5 points of training: accuracy on the training prompts, accuracy on held-out
prompts, the lazy checker's score, the mean sampled reward and the fraction of prompts
whose 8 samples did not all receive the same reward (only those produce a gradient).
Predict before running: in run B, the reward will rise. What will accuracy do?
"""

import argparse
import copy

from mlle.finetune.base import get_base_model, get_chat_model, pocket_loss
from mlle.finetune.data import conversations, extract_answer
from mlle.finetune.rl import reinforce_train, reward_exact, reward_last_digit
from mlle.finetune.sft import build_sft_rows, chat_accuracy, complete, sft_train
from mlle.utils import REPO_ROOT, JsonlLogger

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--n-prompts", type=int, default=300)
ap.add_argument("--prompts-per-step", type=int, default=16)
ap.add_argument("--num-samples", type=int, default=8)
args = ap.parse_args()

base, world = get_base_model()
tok, task = world.tok, world.task
start = get_chat_model(base, world)  # 300 SFT steps with replay: knows the format, unreliable at the task
train_convs = conversations(task, args.n_prompts, "train", 11)
held_out = conversations(task, 300, "test", 1)
stop = {tok.encode_special("<|assistant_end|>"), tok.get_bos_token_id()}
log = JsonlLogger(REPO_ROOT / "results" / "runs" / "part4" / "15_reinforce.jsonl")


def lazy_score(model, convs):
    hits = sum(reward_last_digit(c["problem"], extract_answer(tok.decode(complete(model, tok.render_for_completion(c), stop, 8)[0]))) for c in convs)
    return hits / len(convs)


def evaluate(model):
    return dict(acc_train=chat_accuracy(model, tok, train_convs), acc_new=chat_accuracy(model, tok, held_out), lazy=lazy_score(model, train_convs))


def show(history):
    print(f"{'step':>6} {'acc (train prompts)':>20} {'acc (new prompts)':>18} {'lazy checker':>13} {'sampled reward':>15} {'informative prompts':>20}")
    for e in history:
        extra = f"{e['train_reward']:>15.3f} {e['informative_prompts']:>20.2f}" if "train_reward" in e else f"{'':>15} {'':>20}"
        print(f"{e['step']:>6} {e['acc_train']:>20.3f} {e['acc_new']:>18.3f} {e['lazy']:>13.3f} {extra}")


print(f"\nstart: chat model with {start.num_params():,} parameters; {args.n_prompts} training prompts, "
      f"{args.prompts_per_step} prompts x {args.num_samples} samples per step, lr {args.lr}, {args.steps} steps")
for label, reward_fn in (("A  reward = exact answer", reward_exact), ("B  reward = last digit and length only (flawed)", reward_last_digit)):
    model = copy.deepcopy(start)
    history = reinforce_train(model, tok, train_convs, reward_fn, steps=args.steps, lr=args.lr, prompts_per_step=args.prompts_per_step,
                              num_samples=args.num_samples, eval_fn=evaluate, eval_every=max(1, args.steps // 4), verbose=False)
    print(f"\n{label}")
    show(history)
    print(f"       Pocket loss after training: {pocket_loss(model, world):.4f} (start: {pocket_loss(start, world):.4f})")
    for e in history:
        log.log(run=label[0], lr=args.lr, **e)

model = copy.deepcopy(start)
x, y = build_sft_rows(tok, train_convs, world.spec.block_size)
print("\nC  supervised fine-tuning on the same prompts WITH reference answers (same learning rate and number of steps)")
print(f"{'step':>6} {'acc (train prompts)':>20} {'acc (new prompts)':>18} {'lazy checker':>13}")
done = 0
for target in [0] + [args.steps * k // 4 for k in (1, 2, 3, 4)]:
    if target > done:
        sft_train(model, x, y, steps=target - done, lr=args.lr, seed=done)
        done = target
    e = evaluate(model)
    print(f"{target:>6} {e['acc_train']:>20.3f} {e['acc_new']:>18.3f} {e['lazy']:>13.3f}", flush=True)
    log.log(run="C", lr=args.lr, step=target, **e)
print("\nA reward says only HOW GOOD a sampled answer was; a label says WHAT the answer should have been.")
