"""Small utilities shared by labs, benchmarks and tests."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def seed_everything(seed: int = 0) -> None:
    """Seed Python and PyTorch RNGs. Labs call this so that results are repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str = "auto") -> torch.device:
    """Return the device to use. `prefer` is 'auto', 'cpu', 'cuda' or 'mps'."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device | str) -> None:
    """Wait for queued GPU work to finish. Without this, GPU timings are meaningless,
    because CUDA calls return before the kernel has actually run."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def timer(device: torch.device | str = "cpu"):
    """Context manager yielding a dict whose 'seconds' entry is filled on exit."""
    out = {}
    sync(device)
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        sync(device)
        out["seconds"] = time.perf_counter() - t0


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def environment_report() -> dict:
    """Facts about the machine that every recorded result should carry with it."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["cuda_version"] = torch.version.cuda
    return info


class JsonlLogger:
    """Append-only JSON-lines logger: one experiment record per line.

    Structured results are what make experiments comparable later. Every lab writes
    through this class into results/runs/ (git-ignored)."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record) -> dict:
        record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def ascii_heatmap(matrix, row_labels=None, col_labels=None, width: int = 3) -> str:
    """Render a 2-D matrix of values in [0, 1] as text, so that attention patterns can be
    inspected in a terminal (and in CI logs) without matplotlib."""
    shades = " .:-=+*#%@"
    rows = [[float(v) for v in row] for row in matrix]
    lines = []
    if col_labels is not None:
        lines.append(" " * 8 + "".join(f"{str(c)[:width]:>{width}}" for c in col_labels))
    for i, row in enumerate(rows):
        label = f"{str(row_labels[i])[:7]:>7} " if row_labels is not None else ""
        cells = "".join(
            f"{shades[min(len(shades) - 1, int(max(0.0, min(1.0, v)) * (len(shades) - 1) + 0.5))]:>{width}}"
            for v in row
        )
        lines.append(label + cells)
    return "\n".join(lines)
