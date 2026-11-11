# DGX Spark setup guide

> **Status of this guide.** The course author had no DGX Spark (and no GPU at all) while
> writing it. Nothing below was executed on a GB10. Every statement carries one of three
> labels, and you should treat them differently:
>
> | Label | Meaning |
> |---|---|
> | **[source]** | Follows from nanochat's source code at the pinned commit. Verified by reading (and, where possible, running on CPU). |
> | **[reported]** | Reported by NVIDIA documentation or by named community sources, with link and date. Mostly from October–November 2025, i.e. against an *older* nanochat. |
> | **[to validate]** | A reasoned expectation that you must confirm on your machine. Section 8 is a checklist for doing so. |
>
> The arbiter is always `make doctor` ([`scripts/diagnostics/check_env.py`](../scripts/diagnostics/check_env.py)) and a short measured probe, never this text.

## 1. Why a Spark is not a small H100

nanochat's reference numbers come from a node with eight H100 GPUs. The Spark differs in
ways that change *which code paths run*, not only how fast.

| Property | 8×H100 reference node | DGX Spark | Consequence for this course |
|---|---|---|---|
| GPU | 8 × H100 (Hopper, SM 9.0) | 1 × GB10 (Blackwell-generation, reports compute capability 12.1) **[reported]** | Kernel selection is by architecture. Do not assume Hopper-specific kernels exist. |
| CPU | x86-64 | 20-core Arm (10 Cortex-X925 + 10 Cortex-A725) **[reported]** | Every wheel must exist for `aarch64`. |
| Memory | 8 × 80 GB HBM dedicated to the GPUs + separate host RAM | 128 GB LPDDR5x **shared** by CPU and GPU, 273 GB/s **[reported]** | The model, the data loader, the page cache and your browser compete for one pool. "GPU out of memory" can be caused by host-side usage. |
| Memory bandwidth | ~3.35 TB/s per H100 (nanochat's own table) **[source]** | 0.273 TB/s **[reported]** | Token-by-token decoding is bandwidth-bound (Module 11–12). Expect it to be the Spark's weakest regime relative to an H100, and batching to matter more. |
| Parallelism | 8 ranks via `torchrun` | 1 rank | nanochat falls back to gradient accumulation automatically **[source]**: same optimizer steps, ≥8× the wall-clock time per step before per-GPU speed differences. |
| OS / CUDA | any Linux, CUDA 12.x wheels | DGX OS (Ubuntu 24.04, arm64), CUDA 13 toolkit **[reported]** | nanochat's lockfile pins CUDA 12.8 wheels. See Section 3. |

Sources: [NVIDIA DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html);
`get_peak_bandwidth` in [`nanochat/common.py` L281–L328](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/common.py#L281-L328).

## 2. What nanochat's code will do on a GB10

Read from the pinned source; each item is something `make doctor` reports.

1. **Compute dtype: bf16.** `_detect_compute_dtype` returns `bfloat16` whenever
   `torch.cuda.get_device_capability() >= (8, 0)`
   ([`common.py` L17–L32](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/common.py#L17-L32)). **[source]**
   Override with `NANOCHAT_DTYPE=float32` for numerical-comparison labs.
2. **Attention kernel: most likely the PyTorch SDPA fallback, not Flash Attention 3.**
   `_load_flash_attention_3` uses the `varunneal/flash-attention-3` kernel only when the
   major capability is 9 (Hopper); for anything else it asks the `kernels` package
   whether `kernels-community/flash-attn3` has a build for this machine, and silently
   falls back to `F.scaled_dot_product_attention` if not. The source comment says
   Blackwell "needs SDPA fallback until FA3 is recompiled or FA4 is released"
   ([`flash_attention.py` L23–L46](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/flash_attention.py#L23-L46)). **[source]**
   Whether a matching kernel exists for `aarch64` + `sm_121` today: **[to validate]**.
3. **Therefore train with `--window-pattern L`.** Without FA3, sliding-window layers are
   implemented with an explicit boolean mask, and `base_train.py` itself warns "Your GPU
   utilization will be terrible" and recommends `--window-pattern L`
   ([`base_train.py` L102–L117](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L102-L117)). **[source]**
   This changes the architecture slightly (full attention in every layer), so a Spark run
   with `L` is not the same experiment as the upstream `SSSL` leaderboard runs. Record it.
4. **The log will print `bf16_mfu: 0.00`.** nanochat's peak-FLOPS table has no entry
   matching the device name "NVIDIA GB10"; unknown GPUs get `inf`, which makes model FLOPs
   utilization print as zero
   ([`common.py` L228–L279](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/common.py#L228-L279)). **[source]**
   Use `tok/sec` as the throughput metric. The same applies to MBU in `infer_bench.py`.
5. **`torch.compile` is not optional.** `base_train.py` compiles the model, and
   `nanochat/optim.py` decorates both optimizer kernels with `@torch.compile` at import
   time. On the Spark this needs a Triton whose `ptxas` recognises the GB10. **[source]**
   for the requirement; the failure and fix are **[reported]** (Section 4).
6. **FP8 (`--fp8`) is designed for H100-class GPUs.** nanochat's FP8 path calls
   `torch._scaled_mm` with tensor-wise scaling
   ([`fp8.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/fp8.py)). Whether the cuBLAS FP8 GEMM is available and
   beneficial on the GB10 is **[to validate]**; `make doctor` runs a one-line smoke test.
   Leave `--fp8` off unless that test passes *and* a probe shows a speedup with equal loss.
7. **Single process.** Run scripts with `python -m ...`, not `torchrun`. nanochat then
   skips all distributed code paths and uses gradient accumulation to reach the target
   batch size ([`base_train.py` L406–L413](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_train.py#L406-L413)). **[source]**

## 3. Installation

```bash
# 0. prerequisites: git, uv (https://docs.astral.sh/uv/), a C compiler (gcc), CUDA toolkit from DGX OS
git clone https://github.com/lbecken/llm-course.git && cd llm-course

# 1. pinned nanochat + its environment + this course's package, in one venv
make env-spark
source external/nanochat/.venv/bin/activate

# 2. the truth about your machine
make doctor
python scripts/diagnostics/check_env.py --json results/runs/doctor-spark.json
```

What `make env-spark` does ([`scripts/setup/setup_env.sh`](../scripts/setup/setup_env.sh)):

1. Clones nanochat at the pinned commit into `external/nanochat`.
2. Runs `uv sync --extra gpu --group dev`. nanochat's `uv.lock` contains
   `torch 2.9.1+cu128` wheels for `manylinux_2_28_aarch64` and `triton 3.5.1` for
   aarch64, so this step resolves on ARM64. **[source]** (read from `uv.lock`)
3. Runs the GPU smoke test. If the CUDA 12.8 build cannot drive the GB10, it reinstalls
   the same PyTorch version from the CUDA 13.0 wheel index
   (`https://download.pytorch.org/whl/cu130`), the route that community reports describe
   as working. **[reported]**, see below. From then on, do **not** use `uv run`/`uv sync`
   in that checkout (they would restore the locked build); activate the venv and call
   `python` directly.
4. Installs the course package in editable mode into the same environment.

What the community reports say (all against 2025 revisions of nanochat that pinned
torch 2.8; today's pin already requires 2.9.1):

- A working native setup used `torch>=2.9.0` from the `cu130` index and `triton>=3.5.0`,
  with the CUDA 13.0 toolkit installed
  ([nanochat Discussion #28](https://github.com/karpathy/nanochat/discussions/28), posts of 2025-10-23;
  [NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/anyone-got-nanochat-training-working-on-the-dgx-spark/348537);
  [objectgraph blog, 2025-11](https://objectgraph.com/blog/nanochat-dgx-spark/)). **[reported]**
- PyTorch prints a warning that the GB10 has CUDA capability 12.1 while the build lists
  support up to 12.0. Reporters found that training runs regardless. **[reported]**
  Whether your build prints it, and whether everything works, is exactly what
  `make doctor` checks.

## 4. The one environment variable you will probably need

The most consistently reported failure is at the first `torch.compile`:

```text
ptxas fatal : Value 'sm_121a' is not defined for option 'gpu-name'
```

Cause: Triton shipped a bundled `ptxas` from CUDA 12.8 that does not know the GB10.
Fix: point Triton at the CUDA 13 toolkit's assembler. **[reported]** (same sources as above)

```bash
export CUDA_HOME=/usr/local/cuda            # or /usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas
```

Put these in your shell profile, then re-run `make doctor`; the `torch.compile` check
must say PASS.

**Emergency switch.** `TORCHDYNAMO_DISABLE=1` turns every `torch.compile` into a no-op.
nanochat then trains uncompiled. This was verified on CPU with the pinned revision
(same loss curve; [results/reference/nanochat-local-cpu.md](../results/reference/nanochat-local-cpu.md)).
On a GPU it costs throughput (how much: **[to validate]**), so use it to get unblocked,
not as the normal mode.

## 5. Memory on a unified-memory machine

- The whole 128 GB is one pool. PyTorch's "GPU memory" and your host processes draw from
  it together. Before long runs, close other heavy processes and check `free -h`.
- The Linux page cache counts as "used" until it is reclaimed. After reading many data
  shards, free memory can look low without being a problem. If an allocation fails right
  after heavy file I/O, dropping the cache
  (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`) is a common remedy. **[reported]**
  as general DGX Spark guidance; confirm with `make doctor` before and after.
- Reported footprint for the 2025 d20 model (561 M parameters) at
  `--device-batch-size=32`, sequence length 2048: about 75–77 GB peak
  ([Discussion #28](https://github.com/karpathy/nanochat/discussions/28)). **[reported]**
  The current d20 has ~897 M parameters (value embeddings were added), so expect more.
  Measure with the probe; do not extrapolate.
- If you run out of memory: halve `--device-batch-size` (32 → 16 → 8). nanochat doubles
  gradient accumulation to compensate, so results stay comparable while each step takes
  about as long. **[source]**
- A forum report describes pathologically slow host-to-GPU copies from *pageable* memory
  for many small tensors on the Spark
  ([NVIDIA forum](https://forums.developer.nvidia.com/t/dgx-spark-arm64-cuda-13-pathological-slowdown-for-many-small-h2d-copies-from-pageable-cpu-memory-50x-vs-pinned-impacts-pytorch-model-load-patt/354506)). **[reported]**
  nanochat's training loader already uses one pinned buffer and a single copy per batch
  ([`dataloader.py` L111–L120](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/dataloader.py#L111-L120)), **[source]** so pretraining
  should be unaffected; checkpoint loading might be slow. `make doctor` times both kinds of copy.

## 6. Measure first: the probe

Never start a multi-hour run from a guess. The probe trains for 30 steps and reports
measured tokens/second, step time and peak memory, then converts nanochat's planned token
budget for that depth into hours.

```bash
cd external/nanochat
python -m nanochat.dataset -n 8      # 8 training shards + the validation shard, ~800 MB
python -m scripts.tok_train          # 32768-token tokenizer on ~2B characters
cd ../..

DEPTH=6  BATCH=32 scripts/experiments/spark_probe.sh
DEPTH=12 BATCH=32 scripts/experiments/spark_probe.sh
```

Keep the JSON files it writes to `results/runs/`; they are your hardware baseline for the
whole course. Details and the full cost model: [hardware-and-costs.md](hardware-and-costs.md).

## 7. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `ptxas fatal : Value 'sm_121a' is not defined` | Triton's bundled assembler is too old | Section 4 |
| `torch.cuda.is_available()` is False after `uv sync` | CPU wheel installed (forgot `--extra gpu`), or the CUDA 12.8 build cannot find CUDA 13 libraries | `make env-spark`; check `python -c "import torch; print(torch.__version__, torch.version.cuda)"` |
| Warning "GB10 ... cuda capability 12.1 ... supports 8.0 – 12.0" | Wheel was built before `sm_121` was listed | Informational if the `make doctor` smoke tests pass |
| Training runs but `bf16_mfu: 0.00` | GB10 missing from nanochat's FLOPS table | Expected (Section 2, item 4). Use `tok/sec`. |
| "WARNING: Flash Attention 3 not available" and low utilization | SDPA fallback with sliding windows | Add `--window-pattern L` |
| `uv sync` reinstalled the CUDA 12.8 build | `uv run`/`uv sync` enforce the lockfile | Re-run `make env-spark`; afterwards use the activated venv only |
| Out of memory at the first step | Batch too large for the shared pool | Halve `--device-batch-size`; free host memory; check `free -h` |
| Out of memory hours into a run, right after an evaluation or checkpoint | Evaluation allocates additional buffers; checkpoint writing uses host memory from the same pool | Lower `--eval-tokens`, keep `--save-every=-1`, leave ≥ 15 GB headroom in the probe |
| First step takes minutes | `torch.compile` warm-up | Normal; nanochat excludes the first 10 steps from its timing |
| `AssertionError: total_batch_size ... must be a multiple of ...` | `--total-batch-size` not divisible by `device_batch_size × max_seq_len` | Use powers of two for the batch size |
| Tokenizer or checkpoint "not found" | A different `NANOCHAT_BASE_DIR` than the one used to create them | `echo $NANOCHAT_BASE_DIR`; default is `~/.cache/nanochat` |
| Run dies when the SSH session drops | No terminal multiplexer | Use `tmux` or `screen` for anything in Tier C |

General (non-Spark) problems are in [troubleshooting.md](troubleshooting.md).

## 8. Validation checklist (fill this in on your machine)

Copy [results/templates/spark-validation.md](../results/templates/spark-validation.md) to
`results/runs/` and complete it. The items:

- [ ] `make doctor` output saved; all checks PASS or understood.
- [ ] Which PyTorch build is active (`torch.__version__`, `torch.version.cuda`), and which index it came from.
- [ ] Capability warning present or absent.
- [ ] `torch.compile` PASS; value of `TRITON_PTXAS_PATH`.
- [ ] nanochat attention backend reported by `make doctor` (FA3 or SDPA).
- [ ] FP8 `_scaled_mm` smoke test result.
- [ ] `make test-nanochat` passes on the GPU machine.
- [ ] Probe at depth 6 and depth 12: tokens/second, step time, peak memory.
- [ ] Same probe with `TORCHDYNAMO_DISABLE=1`: tokens/second (cost of not compiling).
- [ ] Same probe with `NANOCHAT_DTYPE=float32`: tokens/second and memory (benefit of bf16).
- [ ] Host-to-GPU copy timing, pageable vs pinned.

If your results contradict this guide, the guide is wrong. Please open an issue or a pull
request with your JSON files ([CONTRIBUTING.md](../CONTRIBUTING.md)).
