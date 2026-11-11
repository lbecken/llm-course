#!/usr/bin/env bash
# Clone karpathy/nanochat into external/nanochat and check out the commit this course is
# written against. The upstream repository is never modified and never committed here.
set -euo pipefail

PIN="92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"   # 2026-07-03, see docs/upstream-version.md
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="${NANOCHAT_DIR:-$ROOT/external/nanochat}"

if [ ! -d "$DIR/.git" ]; then
    mkdir -p "$(dirname "$DIR")"
    git clone https://github.com/karpathy/nanochat.git "$DIR"
fi
if ! git -C "$DIR" cat-file -e "$PIN^{commit}" 2>/dev/null; then
    git -C "$DIR" fetch origin
fi
git -C "$DIR" -c advice.detachedHead=false checkout --quiet "$PIN"
echo "nanochat is at $(git -C "$DIR" rev-parse --short HEAD) in $DIR"
