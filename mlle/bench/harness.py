"""
A small, honest benchmarking harness (Part III).

Rules it enforces:
  * warm-up runs are discarded (allocators, caches, compilation, cuBLAS autotuning);
  * GPU work is synchronised before and after timing (CUDA is asynchronous);
  * several repetitions are taken and the MEDIAN is reported with min and max;
  * every record carries the environment it was measured in.
"""

from __future__ import annotations

import resource
import statistics
import sys
import time

import torch

from mlle.utils import environment_report, sync


def time_fn(fn, warmup: int = 2, repeats: int = 5, device="cpu") -> dict:
    """Run fn() `warmup` + `repeats` times; return median / min / max seconds of the timed runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(times), "min_s": min(times), "max_s": max(times), "repeats": repeats}


class CudaPeakMemory:
    """Context manager: peak GPU memory allocated by PyTorch inside the block (bytes). 0 on CPU."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.bytes = 0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.bytes = torch.cuda.max_memory_allocated(self.device)


def peak_rss_bytes() -> int:
    """Peak resident memory of THIS process so far. A high-water mark: to compare two
    configurations, run each in its own process (labs do this with subprocess)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # Linux reports kilobytes


def record(name: str, **fields) -> dict:
    return {"benchmark": name, **fields, "env": environment_report()}


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)
