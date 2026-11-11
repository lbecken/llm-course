"""
Environment diagnostics: what can this machine actually run?

    python scripts/diagnostics/check_env.py            full report
    python scripts/diagnostics/check_env.py --quick    skip the slower checks (torch.compile)
    python scripts/diagnostics/check_env.py --json out.json

Every check is a small experiment with a PASS / FAIL / SKIP / INFO outcome. Nothing is
assumed from the name of the GPU. This script is the source of truth for the
hardware-dependent statements in docs/dgx-spark-setup.md: run it, keep the JSON.

Exit code is non-zero when --require-cuda is given and the GPU smoke test fails.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time

import torch
import torch.nn.functional as F

RESULTS: list[dict] = []


def record(name: str, status: str, detail: str = "", **extra):
    RESULTS.append({"check": name, "status": status, "detail": detail, **extra})
    print(f"[{status:4s}] {name}: {detail}")


def check_basics():
    record("python", "INFO", f"{platform.python_version()} on {platform.platform()} ({platform.machine()})")
    record("torch", "INFO", f"{torch.__version__} | built for CUDA {torch.version.cuda} | threads {torch.get_num_threads()}")
    try:
        import psutil

        vm = psutil.virtual_memory()
        record("system memory", "INFO", f"total {vm.total / 2**30:.1f} GiB, available {vm.available / 2**30:.1f} GiB",
               total_bytes=vm.total, available_bytes=vm.available)
    except ImportError:
        record("system memory", "SKIP", "psutil not installed")


def check_cuda() -> bool:
    if not torch.cuda.is_available():
        record("cuda", "SKIP", "torch.cuda.is_available() is False: CPU-only labs (Tier A) still work")
        return False
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    record("cuda device", "INFO", f"{name} | compute capability sm_{cap[0]}{cap[1]}", capability=list(cap))
    target = f"sm_{cap[0]}{cap[1]}"
    same_major = [a for a in archs if a.startswith(f"sm_{cap[0]}")]
    if target in archs:
        record("kernels for this GPU", "PASS", f"{target} is in the build's architecture list")
    elif same_major:
        record("kernels for this GPU", "INFO",
               f"{target} not in {archs}; the build has {same_major} (same generation). PyTorch may warn; "
               "the smoke tests below decide whether it works")
    else:
        record("kernels for this GPU", "FAIL", f"{target} not supported by this build: {archs}")
    free, total = torch.cuda.mem_get_info()
    record("gpu memory", "INFO",
           f"free {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB "
           "(on unified-memory systems such as DGX Spark this is shared with the CPU and the page cache)",
           free_bytes=free, total_bytes=total)
    try:
        x = torch.randn(2048, 2048, device="cuda")
        y = (x @ x).sum().item()
        assert y == y
        record("gpu matmul fp32", "PASS", "2048x2048 matmul ran")
    except Exception as e:  # noqa: BLE001
        record("gpu matmul fp32", "FAIL", repr(e)[:300])
        return False
    return True


def check_precision():
    dev = "cuda"
    for dtype in (torch.bfloat16, torch.float16):
        name = str(dtype).replace("torch.", "")
        try:
            a = torch.randn(1024, 1024, device=dev, dtype=dtype)
            b = (a @ a).float()
            ok = torch.isfinite(b).all().item()
            record(f"matmul {name}", "PASS" if ok else "FAIL", "finite result" if ok else "non-finite result")
        except Exception as e:  # noqa: BLE001
            record(f"matmul {name}", "FAIL", repr(e)[:200])
    # FP8 path used by nanochat's --fp8 (nanochat/fp8.py): torch._scaled_mm on float8 tensors
    try:
        a = torch.randn(256, 256, device=dev)
        w = torch.randn(256, 256, device=dev)
        one = torch.tensor(1.0, device=dev)
        out = torch._scaled_mm(a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn).t(), scale_a=one, scale_b=one,
                               out_dtype=torch.bfloat16)
        record("fp8 _scaled_mm", "PASS", f"ran, output {tuple(out.shape)} (nanochat --fp8 may be usable; measure before trusting)")
    except Exception as e:  # noqa: BLE001
        record("fp8 _scaled_mm", "FAIL", f"{type(e).__name__}: {str(e)[:160]} -> do not pass --fp8 to base_train")


def check_attention():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randn(2, 4, 128, 32, device=dev)
    try:
        y = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        record("SDPA causal attention", "PASS", f"output {tuple(y.shape)} on {dev}")
    except Exception as e:  # noqa: BLE001
        record("SDPA causal attention", "FAIL", repr(e)[:200])
    if dev == "cuda":
        flags = {k: getattr(torch.backends.cuda, k)() for k in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled")}
        record("SDPA backends enabled", "INFO", str(flags))


def check_nanochat():
    try:
        from mlle import nanochat_bridge as nb

        path = nb.require_nanochat()
    except Exception as e:  # noqa: BLE001
        record("nanochat checkout", "SKIP", str(e).splitlines()[0])
        return
    sha = nb.checkout_sha(path)
    record("nanochat checkout", "PASS" if sha == nb.PINNED_SHA else "INFO", f"{path} at {sha[:7] if sha else 'unknown'} (pin {nb.PINNED_SHA[:7]})")
    try:
        from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, get_peak_flops
        from nanochat.flash_attention import HAS_FA3, USE_FA3

        record("nanochat compute dtype", "INFO", f"{COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
        if USE_FA3:
            record("nanochat attention backend", "INFO", "Flash Attention 3 kernels loaded")
        else:
            record("nanochat attention backend", "INFO",
                   f"PyTorch SDPA fallback (HAS_FA3={HAS_FA3}). nanochat's own advice then applies: "
                   "train with --window-pattern L, because SDPA has no native sliding-window support")
        if torch.cuda.is_available():
            import logging

            logging.getLogger("nanochat.common").setLevel(logging.ERROR)
            peak = get_peak_flops(torch.cuda.get_device_name(0))
            if peak == float("inf"):
                record("nanochat MFU table", "INFO",
                       f"'{torch.cuda.get_device_name(0)}' is not in nanochat's peak-FLOPS table: the training log "
                       "will print bf16_mfu 0.00. Use tok/sec as the throughput metric instead")
            else:
                record("nanochat MFU table", "PASS", f"peak FLOPS {peak:.3e}")
    except Exception as e:  # noqa: BLE001
        record("nanochat import", "FAIL", repr(e)[:300])


def check_compile():
    """nanochat compiles the model AND both optimizer kernels (nanochat/optim.py uses
    @torch.compile unconditionally), so a broken compiler toolchain stops training even on CPU."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if shutil.which("gcc") is None and shutil.which("cc") is None and shutil.which("clang") is None:
        record("C compiler", "INFO", "no gcc/cc/clang found: torch.compile on CPU needs one")
    if dev == "cuda":
        record("TRITON_PTXAS_PATH", "INFO", os.environ.get("TRITON_PTXAS_PATH", "(not set)"))

    def f(x, y):
        return (torch.relu(x @ y).square() + 1.0).sum()

    try:
        t0 = time.perf_counter()
        g = torch.compile(f, dynamic=False)
        x = torch.randn(256, 256, device=dev)
        v1 = g(x, x).item()
        v2 = f(x, x).item()
        ok = abs(v1 - v2) <= 1e-3 * max(1.0, abs(v2))
        record("torch.compile", "PASS" if ok else "FAIL", f"compiled and ran in {time.perf_counter() - t0:.1f}s on {dev}")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ""
        if "ptxas" in msg or "sm_" in msg:
            hint = " | Triton's bundled ptxas does not know this GPU: export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas"
        record("torch.compile", "FAIL",
               f"{type(e).__name__}: {msg[:200]}{hint} | emergency switch: TORCHDYNAMO_DISABLE=1 runs everything uncompiled")


def check_host_to_device_copy():
    """A DGX Spark forum report describes very slow host-to-GPU copies from pageable memory
    for many small tensors. Measure instead of assuming."""
    n = 200
    pageable = [torch.randn(4096) for _ in range(n)]
    pinned = [t.pin_memory() for t in pageable]
    timings = {}
    for label, tensors in (("pageable", pageable), ("pinned", pinned)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in tensors:
            t.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        timings[label] = (time.perf_counter() - t0) / n * 1e6
    ratio = timings["pageable"] / max(timings["pinned"], 1e-9)
    record("host->GPU copy (4096 floats)", "INFO",
           f"pageable {timings['pageable']:.0f} us, pinned {timings['pinned']:.0f} us (ratio {ratio:.1f}x)", **timings)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip torch.compile and copy-timing checks")
    ap.add_argument("--require-cuda", action="store_true", help="exit non-zero unless the GPU smoke test passes")
    ap.add_argument("--json", type=str, default=None, help="write the results to this file")
    args = ap.parse_args()

    check_basics()
    gpu_ok = check_cuda()
    if gpu_ok:
        check_precision()
    check_attention()
    check_nanochat()
    if not args.quick:
        check_compile()
        if gpu_ok:
            check_host_to_device_copy()

    fails = [r for r in RESULTS if r["status"] == "FAIL"]
    print(f"\n{len(RESULTS)} checks, {len(fails)} failed.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"results": RESULTS, "argv": sys.argv}, f, indent=2)
        print(f"wrote {args.json}")
    if args.require_cuda and not gpu_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
