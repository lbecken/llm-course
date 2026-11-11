"""
How big is a nanochat run, before you start it?

Rebuilds the sizing arithmetic of nanochat's scripts/base_train.py at the pinned
revision WITHOUT allocating a model (it uses PyTorch's "meta" device: shapes only):

    depth -> width, heads -> parameter counts -> FLOPs per token
          -> training horizon in tokens (tokens : scaling-params ratio)
          -> automatic batch size -> number of optimizer steps

and, if you supply a measured throughput, converts tokens into wall-clock time.

    python scripts/experiments/estimate_run.py --depths 4 6 8 12 16 20 24
    python scripts/experiments/estimate_run.py --depths 12 --tok-per-sec 25000

The throughput must be MEASURED on your machine (see `make spark-probe`); this script
never guesses it. Parameter counts and FLOPs are exact properties of the architecture.
"""

from __future__ import annotations

import argparse
import math

import torch

from mlle.nanochat_bridge import require_nanochat


def build_meta(depth, aspect_ratio, head_dim, max_seq_len, vocab_size, window_pattern):
    from nanochat.gpt import GPT, GPTConfig

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim  # same rounding as base_train.build_model_meta
    num_heads = model_dim // head_dim
    config = GPTConfig(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=depth, n_head=num_heads,
                       n_kv_head=num_heads, n_embd=model_dim, window_pattern=window_pattern)
    with torch.device("meta"):
        return GPT(config)


def scaling_params(model) -> int:
    counts = model.num_scaling_params()
    return counts["transformer_matrices"] + counts["lm_head"]  # base_train.get_scaling_params


def estimate(depth, args) -> dict:
    model = build_meta(depth, args.aspect_ratio, args.head_dim, args.max_seq_len, args.vocab_size, args.window_pattern)
    ref = build_meta(12, args.aspect_ratio, args.head_dim, args.max_seq_len, args.vocab_size, args.window_pattern)
    n_scaling = scaling_params(model)
    target_tokens = int(args.target_param_data_ratio * n_scaling)
    d_ref = args.target_param_data_ratio * scaling_params(ref)
    b_ref = 2**19
    if args.total_batch_size > 0:
        total_batch = args.total_batch_size
    else:
        total_batch = 2 ** round(math.log2(b_ref * (target_tokens / d_ref) ** 0.383))
    steps = target_tokens // total_batch
    flops_per_token = model.estimate_flops()
    counts = model.num_scaling_params()
    out = {
        "depth": depth,
        "n_embd": model.config.n_embd,
        "n_head": model.config.n_head,
        "params_total": counts["total"],
        "params_scaling": n_scaling,
        "flops_per_token": flops_per_token,
        "train_tokens": steps * total_batch,
        "total_batch_tokens": total_batch,
        "steps": steps,
        "total_flops": flops_per_token * steps * total_batch,
        "kv_bytes_per_token_bf16": model.config.n_layer * 2 * model.config.n_kv_head * (model.config.n_embd // model.config.n_head) * 2,
    }
    if args.tok_per_sec:
        out["hours_at_measured_throughput"] = out["train_tokens"] / args.tok_per_sec / 3600
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 12, 16, 20, 24, 26])
    ap.add_argument("--aspect-ratio", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--window-pattern", type=str, default="SSSL")
    ap.add_argument("--target-param-data-ratio", type=float, default=12, help="base_train.py default at the pinned revision")
    ap.add_argument("--total-batch-size", type=int, default=-1, help="-1 = nanochat's automatic choice")
    ap.add_argument("--tok-per-sec", type=float, default=None, help="throughput you MEASURED for this depth on your machine")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = ap.parse_args()

    require_nanochat()
    rows = [estimate(d, args) for d in args.depths]
    cols = ["depth", "n_embd", "n_head", "params_total", "params_scaling", "flops_per_token", "total_batch_tokens", "steps", "train_tokens", "total_flops"]
    if args.tok_per_sec:
        cols.append("hours_at_measured_throughput")

    def fmt(key, v):
        if key in ("flops_per_token", "total_flops"):
            return f"{v:.2e}"
        if key == "hours_at_measured_throughput":
            return f"{v:.1f}"
        return f"{v:,}"

    if args.markdown:
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---:"] * len(cols)) + "|")
        for r in rows:
            print("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    else:
        for r in rows:
            print("  ".join(f"{c}={fmt(c, r[c])}" for c in cols))


if __name__ == "__main__":
    main()
