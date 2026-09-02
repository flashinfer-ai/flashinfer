#!/usr/bin/env bash
# Hopper FP8 token sweep: heuristic by default, exhaustive on request.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-python}"

exec "$PYTHON" "$SCRIPT_DIR/run_token_sweep_benchmark.py" "$@"
