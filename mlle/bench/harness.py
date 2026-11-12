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


def _proc_status_kib(field: str) -> int | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def peak_rss_bytes() -> int:
    """Peak resident memory of this process (since start, or since `reset_peak_rss`)."""
    hwm = _proc_status_kib("VmHWM")
    if hwm is not None:
        return hwm * 1024
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def current_rss_bytes() -> int:
    rss = _proc_status_kib("VmRSS")
    return rss * 1024 if rss is not None else peak_rss_bytes()


def reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS counter for this process (Linux). Returns False where unsupported;
    then `peak_rss_bytes` reports the peak since process start, so use one process per measurement."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


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
