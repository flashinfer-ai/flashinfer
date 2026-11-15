#!/usr/bin/env bash
# Devbox: candidate bit-exact test groups (one pytest process per GPU) + cross-tree differential.
set -uo pipefail
R=/scratch/ziangli/flashinfer-w4a16-tp-low-concurrency
OUT=${OUT:-$R/results/tests/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
source "$R/venv/bin/activate"
export FLASHINFER_CUDA_ARCH_LIST=10.3a FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY=1
export FLASHINFER_DIST_AWARE_AUTOTUNE=0 FLASHINFER_STRICT_MOE_EP_TESTS=1
python -c "import torch; assert hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction_split_k')"
cand() {  # gpu name pytest-args...
  local gpu=$1 name=$2; shift 2
  (cd "$R/flashinfer" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PWD \
    FLASHINFER_WORKSPACE_BASE=$R/cache/candidate/ws CUTE_DSL_CACHE_DIR=$R/cache/candidate/cute-$gpu \
    timeout 7200 python -m pytest -q -rs --tb=short --durations=15 "$@" > "$OUT/$name.log" 2>&1;
    echo "exit=$?" >> "$OUT/$name.log") &
}
cand 2 w4a16-boundary tests/moe/test_cute_dsl_fused_moe.py -k "TestCuteDslMoeW4A16 and test_route_tile_boundary_accuracy"
cand 3 w4a16-class-rest tests/moe/test_cute_dsl_fused_moe.py -k "TestCuteDslMoeW4A16 and not test_route_tile_boundary_accuracy"
cand 4 w4a16-focused tests/moe/test_cute_dsl_fused_moe.py -k "w4a16 and (test_wrapper_cuda_graph or test_finalize_handles_cluster_padding or TestExpertParallelism or TestApiConsistency or (test_numerical_accuracy and 1024-2048))"
cand 5 w4a16-unified-cachekey tests/moe/test_unified_moe.py::test_cute_dsl_typed_activation_matches_flat_reference tests/moe/test_cute_dsl_moe_cache_key.py
(cd "$R/flashinfer" && CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD MEGA_NO_DIST=1 \
  FLASHINFER_WORKSPACE_BASE=$R/cache/candidate/ws CUTE_DSL_CACHE_DIR=$R/cache/candidate/cute-6 \
  timeout 7200 python -m pytest -q -rs --tb=short --confcutdir=tests/moe_ep -m arch_blackwell \
  tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -k w4a16 > "$OUT/megamoe-w4a16.log" 2>&1;
  echo "exit=$?" >> "$OUT/megamoe-w4a16.log") &
cand 7 w4a16-all tests/moe/test_cute_dsl_fused_moe.py -k w4a16
# Cross-tree differential at model dimensions (control vs candidate), then oracle samples.
(cd "$R/control/flashinfer" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
  FLASHINFER_WORKSPACE_BASE=$R/cache/control/ws CUTE_DSL_CACHE_DIR=$R/cache/control/cute-0 \
  timeout 7200 python $R/tools/diff_vs_control.py --save "$OUT/diff-control.pt" > "$OUT/diff-control.log" 2>&1;
  echo "exit=$?" >> "$OUT/diff-control.log") &
(cd "$R/flashinfer" && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
  FLASHINFER_WORKSPACE_BASE=$R/cache/candidate/ws CUTE_DSL_CACHE_DIR=$R/cache/candidate/cute-1 \
  timeout 7200 python $R/tools/diff_vs_control.py --save "$OUT/diff-candidate.pt" --oracle > "$OUT/diff-candidate.log" 2>&1;
  echo "exit=$?" >> "$OUT/diff-candidate.log") &
wait
(cd "$R/flashinfer" && PYTHONPATH=$PWD python $R/tools/diff_vs_control.py --compare "$OUT/diff-control.pt" "$OUT/diff-candidate.pt" > "$OUT/diff-compare.log" 2>&1; echo "exit=$?" >> "$OUT/diff-compare.log")
for f in "$OUT"/*.log; do echo "== $(basename $f)"; grep -E "passed|failed|error|skipped|exit=|mismatches|RESULT" "$f" | tail -4; done
echo "done $OUT"
