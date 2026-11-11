#!/usr/bin/env bash
# Measure before you commit: a short pretraining probe that reports throughput and peak
# memory for one model size, then converts nanochat's planned token budget into hours.
#
# Prerequisites (once):  real data shards and a 32768-token tokenizer in $NANOCHAT_BASE_DIR
#   cd external/nanochat
#   python -m nanochat.dataset -n 8          # ~800 MB, 8 training shards + the validation shard
#   python -m scripts.tok_train              # 2B characters; reported to take about a minute on DGX Spark
#
# Usage:  DEPTH=12 BATCH=32 scripts/experiments/spark_probe.sh
#   DEPTH   model depth to probe (default 12)
#   BATCH   --device-batch-size (default 32; halve it if you run out of memory)
#   STEPS   optimizer steps to time (default 30; the first 10 are discarded as warm-up)
#   WINDOW  --window-pattern (default L, which is what nanochat itself recommends without Flash Attention 3)
#
# Output: results/runs/probe-d<DEPTH>-b<BATCH>.json  and a human-readable summary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="${NANOCHAT_DIR:-$ROOT/external/nanochat}"
DEPTH="${DEPTH:-12}"; BATCH="${BATCH:-32}"; STEPS="${STEPS:-30}"; WINDOW="${WINDOW:-L}"; SEQ="${SEQ:-2048}"
OUT_DIR="$ROOT/results/runs"; mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/probe-d${DEPTH}-b${BATCH}.log"

cd "$DIR"
# One micro-batch per optimizer step (no gradient accumulation) keeps the probe short;
# tokens/second is what we are after and it is nearly independent of the accumulation factor.
python -m scripts.base_train \
    --depth="$DEPTH" --window-pattern="$WINDOW" --max-seq-len="$SEQ" \
    --device-batch-size="$BATCH" --total-batch-size=$((BATCH * SEQ)) \
    --num-iterations="$STEPS" --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --model-tag="probe-d${DEPTH}" 2>&1 | tee "$LOG"

python "$ROOT/scripts/experiments/parse_train_log.py" "$LOG" --depth "$DEPTH" --window-pattern "$WINDOW" \
    --json "$OUT_DIR/probe-d${DEPTH}-b${BATCH}.json"
echo "note: the probe saved a checkpoint under \$NANOCHAT_BASE_DIR/base_checkpoints/probe-d${DEPTH}; delete it when done."
