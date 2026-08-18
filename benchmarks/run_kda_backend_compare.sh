#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
result_dir=${1:-"$repo_root/.codex_runs/kda_backend_compare"}
selected_backend=${2:-both}
case_set=${3:-all}
python_bin=${PYTHON_BIN:-python}

mkdir -p "$result_dir"
cd "$repo_root"

export PYTHONPATH="$repo_root"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_JIT_DIR=${FLASHINFER_JIT_DIR:-/tmp/flashinfer-kda-backend-jit}

if [[ "$selected_backend" == both || "$selected_backend" == cake ]]; then
  "$python_bin" benchmarks/bench_recurrent_kda_prefill.py \
    --backend cake \
    --case-set "$case_set" \
    --warmup-ms 20 \
    --bench-ms 100 \
    --json "$result_dir/cake.json"
fi

if [[ "$selected_backend" == both || "$selected_backend" == cute-dsl ]]; then
  "$python_bin" benchmarks/bench_recurrent_kda_prefill.py \
    --backend cute-dsl \
    --case-set "$case_set" \
    --warmup-ms 20 \
    --bench-ms 100 \
    --json "$result_dir/cute-dsl.json"
fi
