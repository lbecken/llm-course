"""
The course benchmark suite (Milestone 3).   Tier A: about 1 minute on a 4-core CPU.

    python scripts/benchmarks/run_all.py              # = make bench
    python scripts/benchmarks/run_all.py --quick      # smaller sizes, about 20 s (used by the tests)
    python scripts/benchmarks/run_all.py --device cuda

Every row compares a BASELINE with an OPTIMISED implementation of the same thing and
reports two kinds of column side by side:

    cost         time, memory or bytes, baseline versus optimised
    correctness  for exact optimisations: the largest difference between the two outputs
                 for approximate ones:    the change in a quality metric (validation loss)

A speed-up without its correctness column is not a result. Records are appended to
results/runs/benchmarks.jsonl (with the environment), and the table is written to
results/runs/benchmarks.md. Numbers are properties of the machine they were measured on.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mlle.bench.harness import current_rss_bytes, markdown_table, peak_rss_bytes, record, reset_peak_rss, time_fn
from mlle.data import pocket
from mlle.foundations.attention import attention, causal_mask
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.ngram import CharTokenizer
from mlle.foundations.train import TrainConfig, estimate_loss, split_tokens, train_on_text
from mlle.inference.flash_sim import attention_tiled
from mlle.inference.kvcache import KVCache, generate_cached
from mlle.inference.quantize import quantize_model_
from mlle.utils import REPO_ROOT, JsonlLogger, pick_device

TRAIN_CFG = dict(block_size=256, n_layer=8, n_head=4, n_embd=256)  # the model of the training-step rows


def randomised(model: MiniGPT) -> MiniGPT:
    """Untrained models have zero-initialised output projections, which would make every
    comparison trivially exact. Give all matrices non-trivial values."""
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                torch.nn.init.normal_(p, std=0.1)
    return model


# ----------------------------------------------------------------------------- worker (memory rows)
def training_step_worker(mode: str, rows: int) -> None:
    """One process per configuration: peak memory above idle and seconds per optimizer step,
    for an effective batch of 32 rows. Modes: plain | accum (4 micro-batches) | checkpoint."""
    torch.manual_seed(0)
    model = MiniGPT(MiniGPTConfig.nanochat_style(1024, **TRAIN_CFG))
    if mode == "checkpoint":
        for block in model.h:
            inner = block.forward
            block.forward = lambda x, cache=None, record=False, _f=inner: checkpoint(_f, x, use_reentrant=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 1024, (rows, 256))
    micro = 4 if mode == "accum" else 1
    opt.zero_grad(set_to_none=True)
    model(x[:1], x[:1]).backward()  # warm-up: allocates gradients and optimizer state
    opt.step()
    reset_peak_rss()
    base = current_rss_bytes()
    t0 = time.perf_counter()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        for i in range(micro):
            xb = x[i * rows // micro : (i + 1) * rows // micro]
            (model(xb, xb) / micro).backward()
        opt.step()
    print(f"{max(0, peak_rss_bytes() - base)} {(time.perf_counter() - t0) / 2}")


def run_worker(mode: str, rows: int) -> tuple[float, float]:
    out = subprocess.run([sys.executable, __file__, "--worker", mode, str(rows)], capture_output=True, text=True, check=True)
    mem, sec = out.stdout.split()
    return int(mem) / 2**20, float(sec)


# ----------------------------------------------------------------------------- benchmarks
def bench_generation(device, n_new: int) -> dict:
    torch.manual_seed(0)
    model = randomised(MiniGPT(MiniGPTConfig.nanochat_style(512, block_size=1024, n_layer=4, n_head=4, n_embd=128))).to(device).eval()
    prompt = torch.randint(0, 512, (1, 8), device=device)
    idx = torch.randint(0, 512, (1, 48), device=device)
    with torch.no_grad():
        cache = KVCache.for_model(model, batch_size=1, max_len=48)
        parts = [model(idx[:, :16], cache=cache)] + [model(idx[:, t : t + 1], cache=cache) for t in range(16, 48)]
        diff = (torch.cat(parts, dim=1) - model(idx)).abs().max().item()
        same = torch.equal(model.generate(prompt, n_new, temperature=0.0), generate_cached(model, prompt, n_new, temperature=0.0))
    naive = time_fn(lambda: model.generate(prompt, n_new, temperature=0.0), warmup=1, repeats=3, device=device)["median_s"]
    cached = time_fn(lambda: generate_cached(model, prompt, n_new, temperature=0.0), warmup=1, repeats=3, device=device)["median_s"]
    return dict(comparison=f"generate {n_new} tokens: full recomputation vs KV cache", kind="exact", unit="s",
                baseline=naive, optimised=cached, ratio=naive / cached,
                correctness=f"max logit diff {diff:.1e}; greedy tokens identical: {same}", ok=bool(same and diff < 1e-3))


def bench_attention(device, T: int) -> list[dict]:
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, T, 64, device=device) for _ in range(3))
    mask = causal_mask(T, device=device)
    ref, _ = attention(q, k, v, mask)
    fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    tiled, stats = attention_tiled(q, k, v, causal=True, block_q=128, block_k=128)
    t_naive = time_fn(lambda: attention(q, k, v, mask), warmup=1, repeats=3, device=device)["median_s"]
    t_fused = time_fn(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), warmup=1, repeats=3, device=device)["median_s"]
    d_fused, d_tiled = (ref - fused).abs().max().item(), (ref - tiled).abs().max().item()
    mib = 4 * 4 / 2**20  # heads x bytes per fp32 element
    return [
        dict(comparison=f"causal attention T={T}: materialised scores vs fused kernel (SDPA)", kind="exact", unit="s",
             baseline=t_naive, optimised=t_fused, ratio=t_naive / t_fused,
             correctness=f"max output diff {d_fused:.1e}", ok=d_fused < 1e-3),
        dict(comparison=f"causal attention T={T}: largest score intermediate, full matrix vs 128x128 tiles", kind="exact", unit="MiB",
             baseline=stats["naive_score_elements"] * mib, optimised=stats["peak_score_elements"] * mib,
             ratio=stats["naive_score_elements"] / stats["peak_score_elements"],
             correctness=f"max output diff {d_tiled:.1e}", ok=d_tiled < 1e-3),
    ]


def bench_precision(device) -> dict:
    torch.manual_seed(0)
    model = randomised(MiniGPT(MiniGPTConfig.nanochat_style(256, block_size=128, n_layer=4, n_head=4, n_embd=128))).to(device).eval()
    idx = torch.randint(0, 256, (8, 128), device=device)

    def fwd(bf16: bool):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            return model(idx, idx).item()

    l32, l16 = fwd(False), fwd(True)
    t32 = time_fn(lambda: fwd(False), warmup=1, repeats=5, device=device)["median_s"]
    t16 = time_fn(lambda: fwd(True), warmup=1, repeats=5, device=device)["median_s"]
    return dict(comparison="forward pass, 1 024 tokens: fp32 vs bf16 autocast", kind="approximate (rounding)", unit="s",
                baseline=t32, optimised=t16, ratio=t32 / t16,
                correctness=f"loss {l32:.5f} vs {l16:.5f} (diff {abs(l32 - l16):.1e})", ok=abs(l32 - l16) < 1e-2)


def bench_training_memory(rows: int) -> list[dict]:
    # gradients must agree before memory is worth comparing
    torch.manual_seed(0)
    small = MiniGPT(MiniGPTConfig.nanochat_style(1024, block_size=64, n_layer=2, n_head=4, n_embd=64))
    x = torch.randint(0, 1024, (8, 64))
    small.zero_grad()
    small(x, x).backward()
    ref = [p.grad.clone() for p in small.parameters()]
    small.zero_grad()
    for i in range(4):
        (small(x[i * 2 : i * 2 + 2], x[i * 2 : i * 2 + 2]) / 4).backward()
    d_accum = max((g - p.grad).abs().max().item() for g, p in zip(ref, small.parameters()))
    small.zero_grad()
    originals = [b.forward for b in small.h]
    for block in small.h:
        block.forward = lambda x, cache=None, record=False, _f=block.forward: checkpoint(_f, x, use_reentrant=False)
    small(x, x).backward()
    d_ckpt = max((g - p.grad).abs().max().item() for g, p in zip(ref, small.parameters()))
    for block, f in zip(small.h, originals):
        block.forward = f

    plain_mem, plain_s = run_worker("plain", rows)
    accum_mem, accum_s = run_worker("accum", rows)
    ckpt_mem, ckpt_s = run_worker("checkpoint", rows)
    name = f"training step, {rows} rows x 256 tokens, 6.8 M params"
    return [
        dict(comparison=f"{name}: one batch vs 4 accumulated micro-batches (peak memory)", kind="exact", unit="MiB",
             baseline=plain_mem, optimised=accum_mem, ratio=plain_mem / max(accum_mem, 1e-9),
             correctness=f"max gradient diff {d_accum:.1e}; step time {plain_s:.2f} s vs {accum_s:.2f} s", ok=d_accum < 1e-5),
        dict(comparison=f"{name}: plain vs activation checkpointing (peak memory)", kind="exact", unit="MiB",
             baseline=plain_mem, optimised=ckpt_mem, ratio=plain_mem / max(ckpt_mem, 1e-9),
             correctness=f"max gradient diff {d_ckpt:.1e}; step time {plain_s:.2f} s vs {ckpt_s:.2f} s", ok=d_ckpt < 1e-5),
    ]


def bench_quantization(steps: int) -> dict:
    text = pocket.generate_corpus(10000, seed=0)
    model, tok, _ = train_on_text(text, dict(block_size=64, n_layer=2, n_head=2, n_embd=64),
                                  TrainConfig(steps=steps, batch_size=32, lr=1e-2, eval_every=steps), verbose=False)
    _, val = split_tokens(torch.tensor(CharTokenizer(text).encode(text)))
    base = estimate_loss(model, val, 32, 20)
    q = copy.deepcopy(model)
    stats = quantize_model_(q, bits=8, per_channel=True)
    loss = estimate_loss(q, val, 32, 20)
    return dict(comparison="weight matrices of a trained MiniGPT: fp32 vs int8 per-channel (bytes)", kind="approximate", unit="bytes",
                baseline=float(stats["bytes_fp32"]), optimised=float(stats["bytes_quantized"]), ratio=stats["bytes_fp32"] / stats["bytes_quantized"],
                correctness=f"validation loss {base:.4f} -> {loss:.4f} ({loss - base:+.4f} nats/char)", ok=abs(loss - base) < 0.01)


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true", help="smaller problem sizes (about 20 s)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "runs"))
    ap.add_argument("--worker", nargs=2, metavar=("MODE", "ROWS"), help="internal: one training-step measurement")
    args = ap.parse_args()
    if args.worker:
        training_step_worker(args.worker[0], int(args.worker[1]))
        return 0

    device = pick_device(args.device)
    n_new, T, rows, steps = (64, 512, 8, 150) if args.quick else (256, 2048, 32, 600)
    rows_out = [bench_generation(device, n_new)]
    rows_out += bench_attention(device, T)
    rows_out.append(bench_precision(device))
    rows_out += bench_training_memory(rows)  # CPU process memory; on GPU use torch.cuda peak statistics instead
    rows_out.append(bench_quantization(steps))

    log = JsonlLogger(f"{args.out_dir}/benchmarks.jsonl")
    for r in rows_out:
        log.log(**record("course_suite", device=str(device), quick=args.quick, **r))
    table = markdown_table(rows_out, ["comparison", "kind", "unit", "baseline", "optimised", "ratio", "correctness"])
    header = f"Course benchmark suite on {device} ({'quick' if args.quick else 'full'} sizes). ratio = baseline / optimised.\n\n"
    with open(f"{args.out_dir}/benchmarks.md", "w") as f:
        f.write(header + table + "\n")
    print(header + table)
    failed = [r["comparison"] for r in rows_out if not r["ok"]]
    if failed:
        print("\nCORRECTNESS CHECK FAILED for: " + "; ".join(failed))
        return 1
    print(f"\nall correctness checks passed; records in {args.out_dir}/benchmarks.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
