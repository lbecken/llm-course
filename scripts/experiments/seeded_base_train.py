"""
Run nanochat's pretraining with a different random seed, WITHOUT modifying nanochat.

    cd external/nanochat
    python ../../scripts/experiments/seeded_base_train.py 1 --depth=4 ... (the usual base_train arguments)

nanochat fixes its global seed (`torch.manual_seed(42)` in `nanochat/common.py::compute_init`)
and `scripts/base_train.py` has no seed argument. To measure seed-to-seed noise (Modules 10
and 20) this wrapper shifts every call of `torch.manual_seed` by the given offset and then
runs `scripts.base_train` unchanged. Offset 0 reproduces the standard run.

The first argument is the offset; everything after it is passed to base_train.
"""

import os
import runpy
import sys

import torch

if len(sys.argv) < 2 or not sys.argv[1].lstrip("-").isdigit():
    sys.exit(__doc__)
offset = int(sys.argv.pop(1))
_manual_seed = torch.manual_seed
torch.manual_seed = lambda seed: _manual_seed(int(seed) + offset)
sys.path.insert(0, os.getcwd())  # so that `scripts` and `nanochat` resolve when started from the nanochat checkout
runpy.run_module("scripts.base_train", run_name="__main__")
