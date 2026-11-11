#!/usr/bin/env bash
# Create nanochat's own virtual environment (from its lockfile) and install the course
# package into it, so that ONE environment runs both nanochat and the course labs.
#
#   scripts/setup/setup_env.sh cpu      CPU-only PyTorch (laptops, CI)                      [tested]
#   scripts/setup/setup_env.sh gpu      CUDA 12.8 wheels from nanochat's lockfile (x86_64)  [upstream default]
#   scripts/setup/setup_env.sh spark    DGX Spark (ARM64 + GB10): see docs/dgx-spark-setup.md [NOT yet validated on hardware]
#
# After it finishes:   source external/nanochat/.venv/bin/activate
set -euo pipefail

MODE="${1:-cpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="${NANOCHAT_DIR:-$ROOT/external/nanochat}"

command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
"$ROOT/scripts/setup/fetch_nanochat.sh"
cd "$DIR"

case "$MODE" in
  cpu)
    uv sync --extra cpu --group dev
    ;;
  gpu)
    uv sync --extra gpu --group dev
    ;;
  spark)
    # Step 1: nanochat's lockfile contains torch 2.9.1+cu128 wheels for aarch64, so the
    # stock install resolves on the Spark.
    uv sync --extra gpu --group dev
    # Step 2: let the diagnostics decide whether that build can drive the GB10 (sm_121).
    if ! .venv/bin/python "$ROOT/scripts/diagnostics/check_env.py" --quick --require-cuda; then
        echo
        echo "The cu128 build did not pass the GPU smoke test. Installing the CUDA 13.0 build of the"
        echo "same PyTorch version (the route reported to work on DGX Spark, see docs/dgx-spark-setup.md)."
        uv pip install --python .venv/bin/python --reinstall "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu130
        echo "NOTE: from now on activate the venv and call python directly; 'uv run' / 'uv sync' would"
        echo "      restore the locked cu128 build."
    fi
    ;;
  *)
    echo "usage: $0 [cpu|gpu|spark]"; exit 1
    ;;
esac

uv pip install --python .venv/bin/python -e "$ROOT[dev]"
echo
echo "Done. Next:"
echo "  source $DIR/.venv/bin/activate"
echo "  python $ROOT/scripts/diagnostics/check_env.py"
