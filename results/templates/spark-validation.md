# DGX Spark validation record

Fill this in once, on your machine. It replaces the "[to validate]" statements in
`docs/dgx-spark-setup.md` with facts. Attach the JSON files.

| Item | Your result |
|---|---|
| Date | |
| DGX OS / kernel (`uname -a`) | |
| CUDA toolkit (`nvcc --version`) | |
| `torch.__version__`, `torch.version.cuda`, wheel index used | |
| GPU name and capability (`make doctor`) | |
| Capability warning printed? (exact text) | |
| `TRITON_PTXAS_PATH` | |
| `torch.compile` check | PASS / FAIL |
| nanochat attention backend | FA3 / SDPA |
| nanochat compute dtype | |
| FP8 `_scaled_mm` smoke test | PASS / FAIL |
| `make test` | |
| `make test-nanochat` | |
| Host-to-GPU copy, pageable vs pinned (µs) | |

## Probes (30 steps, first 10 discarded)

| depth | window | dtype | compile | device batch | tokens/s (median) | step time | peak memory | est. hours for full budget |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 6 | L | bf16 | on | 32 | | | | |
| 12 | L | bf16 | on | 32 | | | | |
| 12 | L | bf16 | off (`TORCHDYNAMO_DISABLE=1`) | 32 | | | | |
| 12 | L | fp32 (`NANOCHAT_DTYPE=float32`) | on | 32 | | | | |
| 12 | SSSL | bf16 | on | 32 | | | | |

## Notes, surprises, deviations from the guide
