# Troubleshooting

DGX Spark specifics are in [dgx-spark-setup.md](dgx-spark-setup.md#7-troubleshooting).
First step for anything environment-related: `make doctor`.

## Installation

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'mlle'` | course package not installed in the active environment | `pip install -e .` (or `make env-cpu`, then activate `external/nanochat/.venv`) |
| `RuntimeError: nanochat checkout not found` | lab needs the pinned upstream checkout | `make nanochat` (code only) or `make env-cpu` (code + environment) |
| Warning "nanochat checkout is at … but the course is written against 92d63d4" | your checkout moved | `make nanochat` re-pins it; or accept the warning if you are studying a newer revision |
| `ModuleNotFoundError: rustbpe` / `tiktoken` / `pyarrow` | running a nanochat lab outside nanochat's environment | `source external/nanochat/.venv/bin/activate` |
| `pip install -e .` downloads a huge CUDA PyTorch on a CPU-only machine | default PyPI wheel on Linux includes CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cpu` first |
| `uv: command not found` | uv not installed | <https://docs.astral.sh/uv/getting-started/installation/> |

## Running nanochat

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: No dataset parquet files found` | `NANOCHAT_BASE_DIR` has no `base_data_climbmix/` | `make nanochat-local-data` (miniature) or `python -m nanochat.dataset -n 8` (real) |
| `FileNotFoundError: ... tokenizer.pkl` or "Token bytes not found" | tokenizer not trained in this base directory | `python -m scripts.tok_train` with the same `NANOCHAT_BASE_DIR` |
| `Tokenizer vocab size … does not match model config vocab size` | checkpoint and tokenizer come from different runs | keep one `NANOCHAT_BASE_DIR` per experiment family; retrain or point to the right directory |
| First training step takes 30–60 s | `torch.compile` of the model and both optimizer kernels | normal; for tiny throw-away runs use `TORCHDYNAMO_DISABLE=1` |
| `torch.compile` fails with a compiler error on CPU | no C compiler | install `gcc`/`clang`, or `TORCHDYNAMO_DISABLE=1` |
| `AssertionError: total_batch_size (...) must be a multiple of ...` | batch arithmetic | make `--total-batch-size` a multiple of `--device-batch-size × --max-seq-len` |
| `AssertionError: Training forward pass should have T > 1` | a sequence of length 1 was passed without a KV cache | nanochat's "smear" needs a previous token; use sequences of length ≥ 2 |
| Loss is `nan` | learning rate too high for your modification, or fp16 without the scaler | reduce `--matrix-lr`/`--embedding-lr`; use bf16 or fp32 |
| Evaluation (CORE) takes very long in a tiny run | CORE downloads a bundle and runs 22 tasks | `--core-metric-every=-1`, or `--core-metric-max-per-task=16` |
| `chat_sft` tries to download datasets | SFT uses SmolTalk, MMLU and GSM8K from the Hugging Face hub | needs network access once; data is cached under `$NANOCHAT_BASE_DIR/task_data` |
| wandb prompts for login | `--run` was set to something other than `dummy` | omit `--run` (default `dummy` disables wandb) or `wandb login` |

## Course labs

| Symptom | Cause | Fix |
|---|---|---|
| A lab is far slower than its stated tier | several CPU jobs at once: PyTorch oversubscribes threads | run one training job at a time, or set `OMP_NUM_THREADS` so the total does not exceed your core count |
| Results differ slightly from `results/reference/` | different PyTorch version or CPU; floating-point order of operations | compare trends and tolerances, not digits; the reference states its software versions |
| Results differ a lot between two of your own runs | different seed, or a real effect | run 3 seeds per condition before interpreting a difference (Module 10) |
| `make check-docs` reports a broken source link | nanochat pin changed, or a typo | see "Updating the pin" in [upstream-version.md](upstream-version.md) |
| GPU timing looks impossibly fast | CUDA calls are asynchronous | time with `mlle.utils.timer(device)`, which synchronizes before and after |

## Still stuck

1. Re-run with the smallest configuration that shows the problem.
2. Save `python scripts/diagnostics/check_env.py --json doctor.json`.
3. Open an issue with the command, the full error text and `doctor.json`.
