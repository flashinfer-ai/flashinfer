#!/usr/bin/env bash
# Fixed P02/P03 Hopper FP8 token-sweep entry point.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" "$SCRIPT_DIR/run_token_sweep_benchmark.py" "$@"
