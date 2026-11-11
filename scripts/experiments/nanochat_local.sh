#!/usr/bin/env bash
# Run the UNMODIFIED nanochat pretraining and evaluation scripts on the miniature local
# dataset created by scripts/experiments/make_local_dataset.py.  Tier A: about one to two
# minutes on a 4-core CPU, most of it torch.compile start-up.
#
#   export NANOCHAT_BASE_DIR=$PWD/results/runs/nanochat-local
#   python scripts/experiments/make_local_dataset.py
#   (cd external/nanochat && python -m scripts.tok_train --vocab-size=512 --max-chars=2000000)
#   scripts/experiments/nanochat_local.sh train
#   scripts/experiments/nanochat_local.sh eval
#
# Environment overrides: DEPTH (4), STEPS (300), SEQ (128), TAG (pocket-d$DEPTH), EXTRA (extra flags)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="${NANOCHAT_DIR:-$ROOT/external/nanochat}"
: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to a scratch directory (see the header of this script)}"
export NANOCHAT_BASE_DIR
DEPTH="${DEPTH:-4}"; STEPS="${STEPS:-300}"; SEQ="${SEQ:-128}"; TAG="${TAG:-pocket-d$DEPTH}"; EXTRA="${EXTRA:-}"

cd "$DIR"
case "${1:-train}" in
  train)
    # --aspect-ratio 16 and --head-dim 32 shrink the width (n_embd = depth*16 rounded up to 32)
    # --window-pattern L: full attention in every layer (the SDPA fallback has no fast sliding window)
    # --core-metric-every -1: the CORE benchmark needs real-world knowledge; meaningless for Pocket
    python -m scripts.base_train \
        --depth="$DEPTH" --aspect-ratio=16 --head-dim=32 --window-pattern=L \
        --max-seq-len="$SEQ" --device-batch-size=16 --total-batch-size=$((16 * SEQ)) \
        --num-iterations="$STEPS" --eval-every=100 --eval-tokens=$((160 * SEQ)) \
        --core-metric-every=-1 --sample-every=-1 --model-tag="$TAG" $EXTRA
    ;;
  eval)
    python -m scripts.base_eval --model-tag "$TAG" --eval bpb,sample \
        --device-batch-size 16 --split-tokens $((160 * SEQ))
    ;;
  *)
    echo "usage: $0 [train|eval]"; exit 1 ;;
esac
