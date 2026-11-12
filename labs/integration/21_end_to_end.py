"""
Lab 21 — The whole system in one run, with a manifest.
Tier A (about 3 minutes on a 4-core CPU; the base model is cached after its first use)

    python labs/integration/21_end_to_end.py

Every stage of the course, in miniature and in order, on the course's own code:

    1 data        Pocket sentences + arithmetic facts, generated from a seed
    2 tokenizer   characters + nanochat's nine special tokens
    3 pretrain    next-token prediction on the text stream          (Modules 07-09)
    4 evaluate    loss on held-out text, raw completion accuracy    (Module 10)
    5 fine-tune   chat format with calculator calls, masked loss    (Modules 14, 16)
    6 serve       KV-cached generation with the tool state machine  (Modules 11, 16)
    7 evaluate    exact answers with the tool on, off, and timing
    8 agent       the bounded loop of Module 18 around a deterministic environment

For each stage the script records WHAT ran, with which settings, how long it took and
what it measured, and writes one JSON manifest. The point of the lab is that file: a run
that can be explained, compared and repeated.
Success criterion: the manifest exists, every stage has a status and a metric, and two
runs with the same seed agree on every metric that is not a timing.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from mlle.agents.envs import NumberGame
from mlle.agents.loop import run_agent
from mlle.agents.policies import plan_then_execute
from mlle.agents.tool_engine import generate_with_tools, tool_chat_accuracy
from mlle.finetune.base import BaseSpec, get_base_model, pocket_loss, pretraining_batch_fn
from mlle.finetune.data import conversations
from mlle.finetune.sft import build_sft_rows, raw_completion_accuracy, sft_train
from mlle.inference.kvcache import kv_bytes_per_token
from mlle.utils import REPO_ROOT, environment_report

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sft-steps", type=int, default=600)
ap.add_argument("--out", default=str(REPO_ROOT / "results" / "runs" / "part6" / "manifest.json"))
args = ap.parse_args()

manifest = {"seed": args.seed, "environment": environment_report(), "stages": []}


def stage(name, fn, **settings):
    t0 = time.perf_counter()
    try:
        metrics, status = fn(), "ok"
    except Exception as e:  # noqa: BLE001  a failed stage is recorded, and the pipeline stops honestly
        metrics, status = {"error": f"{type(e).__name__}: {e}"}, "failed"
    entry = {"stage": name, "status": status, "seconds": round(time.perf_counter() - t0, 2), "settings": settings, "metrics": metrics}
    manifest["stages"].append(entry)
    shown = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    print(f"[{status:6s}] {name:10s} {entry['seconds']:>7.1f} s   {shown}", flush=True)
    if status != "ok":
        raise SystemExit(f"stage {name} failed; see manifest")
    return metrics


state = {}
spec = BaseSpec(seed=args.seed)


def s_data_and_pretrain():
    model, world = get_base_model(spec, verbose=False)
    state.update(base=model, world=world)
    digest = hashlib.sha256(world.train_tokens.numpy().tobytes()).hexdigest()[:16]
    return {"train_tokens": len(world.train_tokens), "val_tokens": len(world.val_tokens), "vocab_size": world.tok.vocab_size,
            "data_sha256_16": digest, "parameters": model.num_params()}


def s_eval_base():
    world, base = state["world"], state["base"]
    return {"pocket_loss_nats_per_char": pocket_loss(base, world),
            "raw_completion_accuracy": raw_completion_accuracy(base, world.tok, world.task.problems(200, "test", 1))}


def s_finetune():
    world = state["world"]
    model = copy.deepcopy(state["base"])
    convs = conversations(world.task, 3000, "train", 7, use_tool=True, ops=("+", "*"))
    x, y = build_sft_rows(world.tok, convs, world.spec.block_size)
    losses = sft_train(model, x, y, steps=args.sft_steps, lr=2e-3, seed=args.seed, extra_batch_fn=pretraining_batch_fn(world, seed=args.seed))
    state["assistant"] = model
    return {"conversations": len(convs), "rows": x.shape[0], "supervised_fraction": (y >= 0).float().mean().item(),
            "first_loss": losses[0], "last_loss": sum(losses[-20:]) / 20}


def s_serve_and_eval():
    world, model = state["world"], state["assistant"]
    held_out = conversations(world.task, 200, "test", 1, ops=("+", "*"))
    on = tool_chat_accuracy(model, world.tok, held_out, tools_enabled=True)
    off = tool_chat_accuracy(model, world.tok, held_out, tools_enabled=False)
    t0, tokens = time.perf_counter(), 0
    for conv in held_out[:50]:
        out, _ = generate_with_tools(model, world.tok, world.tok.render_for_completion(conv))
        tokens += len(out)
    dt = time.perf_counter() - t0
    c = model.config
    return {"accuracy_tool_on": on["accuracy"], "accuracy_tool_off": off["accuracy"], "tool_call_rate": on["tool_call_rate"],
            "fabricated_output_rate": on["fabricated_output_rate"], "pocket_loss_after_sft": pocket_loss(model, world),
            "decode_tokens_per_second": tokens / dt, "seconds_per_answer": dt / 50,
            "kv_cache_bytes_per_token": kv_bytes_per_token(c.n_layer, c.n_kv_head, c.n_embd // c.n_head, 4)}


def s_agent():
    outcomes, steps = {}, 0
    for start, target in [(3, 26), (5, 47), (2, 31), (7, 100), (4, 61)]:
        env = NumberGame(start, target)
        ep = run_agent(env, plan_then_execute(env), max_steps=15)
        outcomes[ep.outcome] = outcomes.get(ep.outcome, 0) + 1
        steps += len(ep.steps) - 1
    return {"episodes": 5, "outcomes": outcomes, "mean_steps": steps / 5}


print(f"end-to-end run, seed {args.seed}\n")
stage("pretrain", s_data_and_pretrain, **{k: getattr(spec, k) for k in ("n_layer", "n_embd", "steps", "max_digits", "n_sentences", "n_facts")})
stage("eval-base", s_eval_base, questions=200)
stage("fine-tune", s_finetune, steps=args.sft_steps, lr=2e-3, replay=True, tool_format=True)
stage("serve-eval", s_serve_and_eval, questions=200, decoding="greedy", kv_cache=True)
stage("agent", s_agent, policy="search", max_steps=15)

manifest["total_seconds"] = round(sum(s["seconds"] for s in manifest["stages"]), 1)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
slowest = max(manifest["stages"], key=lambda s: s["seconds"])
print(f"\ntotal {manifest['total_seconds']} s; slowest stage: {slowest['stage']} ({slowest['seconds']} s)")
print(f"manifest written to {out}")
