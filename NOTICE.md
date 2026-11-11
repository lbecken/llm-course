# Attribution

## nanochat

This course studies **nanochat** by Andrej Karpathy (<https://github.com/karpathy/nanochat>),
released under the MIT License, Copyright (c) 2025 Andrej Karpathy.

- nanochat's source code is **not redistributed** in this repository. It is fetched from
  the upstream repository at a pinned commit into `external/nanochat/` (git-ignored). See
  [docs/upstream-version.md](docs/upstream-version.md).
- Lessons quote short excerpts of nanochat's source for commentary, each with a link to
  the exact file and lines at the pinned commit. Those excerpts remain under nanochat's
  MIT License and copyright.
- A few course files deliberately mirror nanochat's names and conventions so that the two
  can be compared line by line (for example the special-token list in
  `mlle/foundations/bpe.py`, the RoPE convention in `mlle/foundations/positional.py`, the
  learning-rate schedule in `mlle/foundations/optim.py`). Each such place says so in a
  comment.
- The course is an independent educational project. It is not affiliated with or endorsed
  by Andrej Karpathy.

If you cite nanochat, use the citation from its README:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## Other sources

- The Muon optimizer and its Newton–Schulz coefficients (`mlle/foundations/optim.py`)
  follow Keller Jordan's public description (<https://kellerjordan.github.io/posts/muon/>).
- Papers, books and courses that informed each module are listed in that module's
  `references.md` and in [resources/](resources/).

## This repository

Course text and code are released under the MIT License (see [LICENSE](LICENSE)).
