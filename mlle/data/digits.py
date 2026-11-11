"""
A 1990s-style handwritten-digit problem, small enough to read in full.

Each digit 0-9 is a 5x7 bitmap (35 pixels). `make_digit_dataset` produces noisy copies:
pixels are flipped with probability `flip_prob` and Gaussian noise is added. The task is
ordinary supervised classification with 10 classes, which Module 01 uses as the bridge to
next-token prediction (classification with `vocab_size` classes).
"""

from __future__ import annotations

import torch

_FONT = {
    0: [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    1: ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    2: [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    3: [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."],
    4: ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    5: ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    6: ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    7: ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    8: [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    9: [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
}

HEIGHT, WIDTH = 7, 5
NUM_PIXELS = HEIGHT * WIDTH
NUM_CLASSES = 10


def digit_templates() -> torch.Tensor:
    """Clean bitmaps as a float tensor of shape (10, 35) with values in {0, 1}."""
    rows = []
    for d in range(10):
        rows.append([1.0 if ch == "#" else 0.0 for line in _FONT[d] for ch in line])
    return torch.tensor(rows)


def make_digit_dataset(n: int, flip_prob: float = 0.08, noise_std: float = 0.15, seed: int = 0):
    """Return (x, y): x has shape (n, 35), y has shape (n,) with integer labels 0..9."""
    g = torch.Generator().manual_seed(seed)
    templates = digit_templates()
    y = torch.randint(0, 10, (n,), generator=g)
    x = templates[y].clone()
    flips = torch.rand(x.shape, generator=g) < flip_prob
    x = torch.where(flips, 1.0 - x, x)
    x = x + noise_std * torch.randn(x.shape, generator=g)
    return x, y


def render(x: torch.Tensor) -> str:
    """ASCII picture of one 35-pixel example."""
    grid = x.view(HEIGHT, WIDTH)
    return "\n".join("".join("#" if v > 0.5 else "." for v in row) for row in grid)
