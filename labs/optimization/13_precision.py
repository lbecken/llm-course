"""
Lab 13a — Floating-point formats: range, resolution, and what breaks.
Tier A (seconds on CPU; adds GPU timings with --device cuda)

    python labs/optimization/13_precision.py
    python labs/optimization/13_precision.py --device cuda

1. Range and resolution of fp32, fp16, bf16 (and the two fp8 formats nanochat uses).
2. Small updates vanish: why master weights stay in fp32.
3. Overflow: why fp16 needs a gradient scaler and bf16 does not.
4. A model forward pass in bf16 versus fp32: how different are the logits and the loss?
5. (GPU only) training-step time and peak memory, fp32 versus bf16 autocast.
Success criterion: you can explain each printed difference from the number of exponent
and mantissa bits.
"""

import argparse

import torch

from mlle.bench.harness import CudaPeakMemory, time_fn
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.utils import pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
device = pick_device(args.device)

# %% 1. Range and resolution
print("1. formats")
print(f"{'dtype':>14} {'bits':>5} {'largest value':>15} {'spacing at 1.0 (eps)':>21} {'~decimal digits':>16}")
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt).replace('torch.', ''):>14} {fi.bits:>5} {fi.max:>15.3e} {fi.eps:>21.3e} {fi.precision:>16}")
for name in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, name):
        fi = torch.finfo(getattr(torch, name))
        print(f"{name:>14} {fi.bits:>5} {fi.max:>15.3e} {fi.eps:>21.3e} {'':>16}")

# %% 2. Small updates vanish in low precision
print("\n2. add 1e-3 to a weight of 1.0, one thousand times (exact answer 2.0)")
for dt in (torch.float32, torch.float16, torch.bfloat16):
    w = torch.tensor(1.0, dtype=dt)
    for _ in range(1000):
        w = w + torch.tensor(1e-3, dtype=dt)
    print(f"   {str(dt).replace('torch.', ''):>9}: {w.item():.6f}")
print("   bf16 has 8 bits of mantissa: near 1.0 its spacing is 0.0078, so +0.001 rounds back to 1.0 every time.")
print("   This is why nanochat keeps block weights in fp32 and only COMPUTES in bf16.")

# %% 3. Overflow
x = torch.tensor(300.0)
print("\n3. 300 * 300 = 90000")
for dt in (torch.float32, torch.float16, torch.bfloat16):
    print(f"   {str(dt).replace('torch.', ''):>9}: {(x.to(dt) * x.to(dt)).item()}")
print("   fp16 overflows above 65504; bf16 keeps fp32's exponent range and merely rounds.")

# %% 4. A forward pass in reduced precision
torch.manual_seed(0)
model = MiniGPT(MiniGPTConfig.nanochat_style(256, block_size=128, n_layer=4, n_head=4, n_embd=128)).to(device).eval()
with torch.no_grad():
    for p in model.parameters():
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.05)
idx = torch.randint(0, 256, (4, 128), device=device)
with torch.no_grad():
    ref_logits, ref_loss = model(idx), model(idx, idx).item()
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            bf_logits, bf_loss = model(idx).float(), model(idx, idx).item()
        print(f"\n4. same weights, same input: loss fp32 {ref_loss:.5f} vs bf16 {bf_loss:.5f} (difference {abs(ref_loss - bf_loss):.1e}); "
              f"largest logit difference {(ref_logits - bf_logits).abs().max().item():.3f} on logits of magnitude {ref_logits.abs().max().item():.1f}")
    except Exception as e:  # noqa: BLE001
        print(f"\n4. bf16 autocast not available on this device: {e}")

# %% 5. GPU timing
if device.type == "cuda":
    print("\n5. one training step (forward + backward), batch 16 x 512, model 8 layers x 512")
    big = MiniGPT(MiniGPTConfig.nanochat_style(4096, block_size=512, n_layer=8, n_head=8, n_embd=512)).to(device)
    xb = torch.randint(0, 4096, (16, 512), device=device)

    def step(dtype):
        big.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            loss = big(xb, xb)
        loss.backward()

    for name, dt in (("fp32", None), ("bf16", torch.bfloat16)):
        t = time_fn(lambda: step(dt), warmup=2, repeats=5, device=device)
        with CudaPeakMemory(device) as mem:
            step(dt)
        print(f"   {name}: {t['median_s'] * 1e3:8.1f} ms/step   {xb.numel() / t['median_s']:>10,.0f} tokens/s   peak memory {mem.bytes / 2**30:.2f} GiB")
else:
    print("\n5. (skipped: timings of fp32 vs bf16 are only meaningful on a GPU; run with --device cuda)")
