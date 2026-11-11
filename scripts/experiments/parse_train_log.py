"""
Turn a nanochat base_train log into numbers: throughput, step time, peak memory, loss curve.

    python scripts/experiments/parse_train_log.py run.log
    python scripts/experiments/parse_train_log.py run.log --depth 12 --json probe.json

With --depth it also asks scripts/experiments/estimate_run.py how many tokens nanochat
would train that depth for, and converts the MEASURED tokens/second into hours.
The first `--skip` steps are ignored: they include compilation and warm-up.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

STEP_RE = re.compile(r"step (\d+)/(\d+) .*?loss: ([\d.]+) .*?dt: ([\d.]+)ms \| tok/sec: ([\d,]+)")
VAL_RE = re.compile(r"Step (\d+) \| Validation bpb: ([\d.]+)")
MEM_RE = re.compile(r"Peak memory usage: ([\d.]+)MiB")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse(text: str, skip: int = 10) -> dict:
    text = ANSI_RE.sub("", text)
    steps = [(int(s), float(loss), float(dt), int(tps.replace(",", ""))) for s, _, loss, dt, tps in STEP_RE.findall(text)]
    timed = [row for row in steps if row[0] >= skip]
    out = {
        "num_steps_logged": len(steps),
        "loss_curve": [(s, loss) for s, loss, _, _ in steps],
        "val_bpb": [(int(s), float(v)) for s, v in VAL_RE.findall(text)],
    }
    if timed:
        out["median_tok_per_sec"] = statistics.median(r[3] for r in timed)
        out["median_step_ms"] = statistics.median(r[2] for r in timed)
        out["min_tok_per_sec"] = min(r[3] for r in timed)
        out["max_tok_per_sec"] = max(r[3] for r in timed)
    mem = MEM_RE.search(text)
    if mem:
        out["peak_memory_mib"] = float(mem.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--skip", type=int, default=10, help="ignore steps before this one (compile + warm-up)")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--window-pattern", type=str, default="L")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    result = parse(Path(args.log).read_text(encoding="utf-8", errors="replace"), args.skip)
    if "median_tok_per_sec" not in result:
        sys.exit(f"no timed steps found after step {args.skip} in {args.log}")
    print(f"steps logged:        {result['num_steps_logged']}")
    print(f"median tokens/sec:   {result['median_tok_per_sec']:,.0f}  (min {result['min_tok_per_sec']:,}, max {result['max_tok_per_sec']:,})")
    print(f"median step time:    {result['median_step_ms']:.1f} ms")
    if "peak_memory_mib" in result:
        print(f"peak GPU memory:     {result['peak_memory_mib'] / 1024:.2f} GiB (0 on CPU)")
    if result["val_bpb"]:
        print("validation bpb:      " + ", ".join(f"{s}: {v:.4f}" for s, v in result["val_bpb"]))
    if args.depth is not None:
        here = Path(__file__).resolve().parent
        cmd = [sys.executable, str(here / "estimate_run.py"), "--depths", str(args.depth),
               "--window-pattern", args.window_pattern, "--tok-per-sec", str(result["median_tok_per_sec"])]
        est = subprocess.run(cmd, capture_output=True, text=True)
        line = [ln for ln in est.stdout.splitlines() if ln.startswith("depth=")]
        if line:
            print("full-run estimate:   " + line[-1])
            result["estimate"] = line[-1]
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
