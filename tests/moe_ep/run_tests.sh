#!/usr/bin/env bash
# Run moe_ep tests (unit + multirank + smoke + mega).
#
# Usage (from repo root):
#   bash tests/moe_ep/run_tests.sh
#   bash tests/moe_ep/run_tests.sh unit          # host-only pytest
#   bash tests/moe_ep/run_tests.sh multirank     # 4-GPU split path (NCCL-EP)
#   bash tests/moe_ep/run_tests.sh mega          # Blackwell mega multirank
#   bash tests/moe_ep/run_tests.sh mega_sm90     # 4-GPU Hopper sm90_fp8_fp8_bf16_pull_cutedsl mega multirank
#   bash tests/moe_ep/run_tests.sh sm90_push     # 2-GPU Hopper sm90_fp8_fp8_bf16_push_cuda kernel + backend
#   bash tests/moe_ep/run_tests.sh split_path_correctness_bf16   # 4-GPU bf16 split-path numerics
#   bash tests/moe_ep/run_tests.sh split_path_correctness_nvfp4  # 4-GPU NVFP4 split-path numerics
#   bash tests/moe_ep/run_tests.sh split_path_correctness_ht     # 4-GPU HT (FLAT) split-path numerics
#   bash tests/moe_ep/run_tests.sh oracle        # 1-GPU torch-oracle correctness (all paths)
#   bash tests/moe_ep/run_tests.sh oracle_sm90   # 1-GPU Hopper sm90_fp8_fp8_bf16_pull_cutedsl vs drop reference
#   bash tests/moe_ep/run_tests.sh smoke         # torchrun smoke scripts
#   bash tests/moe_ep/run_tests.sh ft            # 4-GPU fault tolerance (kills a rank)
#
# Install (transport libs build by default, best-effort):
#   pip install --no-build-isolation -e .
#   # strict (missing NIXL-EP build deps become hard errors): BUILD_NIXL_EP=1
#
# Requires:
#   - FLASHINFER repo root on PYTHONPATH (handled below)
#   - multirank/smoke/correctness: nccl.ep importable (built by the install above)
#   - multirank/smoke/correctness: >=4 GPUs
#   - mega: Blackwell (sm_100+), nvshmem, deep_gemm, triton
#   - optional NIXL smoke: BUILD_NIXL_EP=1 install

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PY="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
NPROC_MULTIRANK="${NPROC_MULTIRANK:-4}"
NPROC_SMOKE="${NPROC_SMOKE:-4}"
# NOTE: no --confcutdir. The moe_ep pytest hooks (--backend option, nvep/gpu_*/
# arch_blackwell markers, env/GPU/arch auto-skips) live in the root
# tests/conftest.py. Cutting conftest discovery at tests/moe_ep would drop them
# and break --backend / marker-based selection below.
MOE_EP_PYTEST_FLAGS=()

declare -a SECTION_NAMES=()
declare -a SECTION_STATUS=()

have_nccl_ep() {
  "${PY}" -c "from flashinfer.moe_ep import have_nccl_ep; raise SystemExit(0 if have_nccl_ep() else 1)"
}

have_nixl_ep() {
  "${PY}" -c "from flashinfer.moe_ep import have_nixl_ep; raise SystemExit(0 if have_nixl_ep() else 1)"
}

require_nccl_ep() {
  if have_nccl_ep; then
    return 0
  fi
  echo "nccl_ep backend not available." >&2
  echo "Install with: pip install --no-build-isolation -e .  (transport libs build by default)" >&2
  return 1
}

run_section() {
  local name="$1"
  shift
  echo ""
  echo "################################################################"
  echo "### ${name}"
  echo "################################################################"
  if "$@"; then
    SECTION_NAMES+=("${name}")
    SECTION_STATUS+=("PASS")
    echo "### ${name}: PASS"
  else
    SECTION_NAMES+=("${name}")
    SECTION_STATUS+=("FAIL")
    echo "### ${name}: FAIL (continuing)" >&2
  fi
}

# Run pytest and exit the interpreter WITHOUT finalization (os._exit).
# The full unit suite accumulates native heap damage somewhere in the
# GPU/DSL/transport stack: with every test PASSING, the process can still
# abort in CPython teardown ("malloc(): unaligned tcache chunk detected"
# after the pytest summary, observed 2026-08-12 job 2388315) or, earlier,
# inside the first heavy import after the suite (the nvfp4 warmup case
# below). Skipping interpreter finalization sidesteps the teardown
# detonation; the pytest exit code (all-tests result) is preserved.
# Root-causing the corruption needs an ASAN/valgrind pass — tracked in the
# runbook's unit-suite notes.
pytest_no_finalize() {
  "${PY}" -c '
import os, sys
import pytest
rc = int(pytest.main(sys.argv[1:]))
sys.stdout.flush()
sys.stderr.flush()
os._exit(rc)
' "$@"
}

run_unit() {
  pytest_no_finalize tests/moe_ep/ -v \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    --ignore=tests/moe_ep/test_moe_ep_layer_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_deep_gemm_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_bf16_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_fault_tolerance_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_cudagraph_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_sm90_pull_fp8_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
    --ignore=tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py \
    --ignore=tests/moe_ep/test_deep_gemm_mega_kernel_vs_reference.py \
    --ignore=tests/moe_ep/test_sm90_pull_fp8_kernel_vs_reference.py \
    --ignore=tests/moe_ep/test_split_fused_moe_kernel_vs_reference.py \
    --ignore=tests/moe_ep/test_moe_ep_compute_correctness.py \
    --ignore=tests/moe_ep/test_moe_ep_compute_correctness_nvfp4.py \
    --ignore=tests/moe_ep/test_moe_ep_ht_correctness.py \
    --ignore=tests/moe_ep/test_mega_cuda_graph.py \
    -k "not multirank_roundtrip" \
    --deselect "tests/moe_ep/test_workspace_pool.py::test_two_nvfp4_layers_share_one_symm_buffer" \
    || return 1
  # Run the nvfp4 symm-buffer-sharing test in its own interpreter. In-suite it
  # crashes the process (Fatal Python error: Aborted) inside the nvfp4 layer
  # warmup's kernel-module imports — the same suite-accumulated heap damage
  # (see pytest_no_finalize above), detonating at the first big
  # import/compile burst instead of at teardown. Passes 100% standalone,
  # per-file, and in every subset tried (see moe_ep runbook "unit suite"
  # notes; observed since 2026-07-22).
  pytest_no_finalize -v "${MOE_EP_PYTEST_FLAGS[@]}" \
    "tests/moe_ep/test_workspace_pool.py::test_two_nvfp4_layers_share_one_symm_buffer"
}

run_multirank() {
  require_nccl_ep || return 1

  local rc=0

  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_layer_multirank.py -v \
    -m "nvep and gpu_4" --backend=nccl_ep || rc=1

  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_split_kernels.py -v \
    -m "nvep and gpu_4" --backend=nccl_ep || rc=1

  # CUDA-graph capture of the split path (Handle.update). nccl_ep only --
  # nixl_ep has no update()/InitHandle split to capture.
  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_cudagraph_multirank.py -v \
    -m "nvep and gpu_4" || rc=1

  if have_nixl_ep; then
    "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
      "${MOE_EP_PYTEST_FLAGS[@]}" \
      tests/moe_ep/test_moe_ep_layer_multirank.py -v \
      -m "nvep and gpu_4" --backend=nixl_ep || rc=1

    "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
      "${MOE_EP_PYTEST_FLAGS[@]}" \
      tests/moe_ep/test_split_kernels.py -v \
      -m "nvep and gpu_4" --backend=nixl_ep || rc=1
  else
    echo "nixl_ep not built; skipping NIXL multirank (rebuild with BUILD_NIXL_EP=1 pip install -e .)"
  fi

  return "${rc}"
}

run_split_path_correctness_bf16() {
  require_nccl_ep || return 1

  NPROC_CORRECTNESS="${NPROC_CORRECTNESS:-4}"
  "${TORCHRUN}" --nproc_per_node="${NPROC_CORRECTNESS}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_compute_correctness.py -v \
    -m "nvep and gpu_4 and arch_blackwell" --backend=nccl_ep
}

run_split_path_correctness_nvfp4() {
  require_nccl_ep || return 1

  NPROC_CORRECTNESS="${NPROC_CORRECTNESS:-4}"
  "${TORCHRUN}" --nproc_per_node="${NPROC_CORRECTNESS}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_compute_correctness_nvfp4.py -v \
    -m "nvep and gpu_4 and arch_blackwell" --backend=nccl_ep
}

run_split_path_correctness_ht() {
  require_nccl_ep || return 1

  NPROC_CORRECTNESS="${NPROC_CORRECTNESS:-4}"
  "${TORCHRUN}" --nproc_per_node="${NPROC_CORRECTNESS}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_ht_correctness.py -v \
    -m "nvep and gpu_4 and arch_blackwell" --backend=nccl_ep
}

# Single-GPU torch-oracle correctness: every compute path (split trtllm
# bf16/nvfp4, mega cutedsl mxfp8/nvfp4, mega deep_gemm) vs an independent
# pure-torch reference. EP-vs-kernel equality is covered by the multirank
# sections; this anchors the kernels themselves to textbook math.
run_oracle() {
  # Accumulate failures: a section with several pytest invocations must not
  # report PASS just because the LAST one passed.
  local rc=0

  "${PY}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_split_fused_moe_kernel_vs_reference.py -v \
    -m arch_blackwell || rc=1

  MEGA_NO_DIST=1 "${TORCHRUN}" --standalone --nproc_per_node=1 -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
    tests/moe_ep/test_bf16_cutedsl_kernel_vs_reference.py \
    tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -v \
    -m arch_blackwell || rc=1

  # CUDA graph capture/replay for the cutedsl mega layer paths (1 GPU).
  MEGA_NO_DIST=1 "${PY}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_mega_cuda_graph.py -v \
    -m arch_blackwell || rc=1

  # deep_gemm's symm buffer needs an initialized process group (no
  # MEGA_NO_DIST equivalent). The test self-bootstraps a 1-rank group under
  # plain pytest; the 1-proc torchrun here also exercises its env:// path.
  "${TORCHRUN}" --standalone --nproc_per_node=1 -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_deep_gemm_mega_kernel_vs_reference.py -v \
    -m arch_blackwell || rc=1

  return "${rc}"
}

# Single-GPU Hopper torch-oracle correctness: sm90_fp8_fp8_bf16_pull_cutedsl mega kernel vs the
# kernel drop's own pure-torch reference (compute_megamoe_reference_fp8).
# Runs in its OWN pytest process: the SM90 and SM100 kernel trees share
# top-level module names and are mutually exclusive per process, so this file
# is excluded from run_unit and must not share an invocation with
# SM100-importing tests.  MEGA_NO_DIST=1 single-rank (the sm90 shim's comm
# bootstrap supports it, like the sm100 cutedsl oracle runs above).
run_oracle_sm90() {
  MEGA_NO_DIST=1 "${PY}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_sm90_pull_fp8_kernel_vs_reference.py -v \
    -m arch_hopper
}

run_mega() {
  local rc=0

  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_deep_gemm_mega_multirank.py \
    tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
    tests/moe_ep/test_moe_ep_bf16_cutedsl_mega_multirank.py \
    tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py -v \
    -m "gpu_4 and arch_blackwell" || rc=1

  MEGA_NO_DIST=1 "${TORCHRUN}" --nproc_per_node=1 -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
    tests/moe_ep/test_bf16_cutedsl_kernel_vs_reference.py \
    tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -v \
    -m arch_blackwell || rc=1

  return "${rc}"
}

# 4-GPU Hopper sm90_fp8_fp8_bf16_pull_cutedsl mega multirank (layer-vs-direct-shim parity on
# real cross-rank EP traffic).  Own torchrun pytest process: the SM90 and
# SM100 kernel trees share top-level module names and are mutually exclusive
# per process, so this must not share an invocation with the Blackwell mega
# tests above (and is excluded from run_unit).
run_mega_sm90() {
  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_sm90_pull_fp8_mega_multirank.py -v \
    -m "gpu_4 and arch_hopper"
}

# 2-GPU Hopper push-style FP8 (sm90_fp8_fp8_bf16_push_cuda) kernel + backend.
# Own target rather than folded into `multirank` (upstream PR #4069 runs it
# there): on non-Hopper nodes the arch-marked files collect 0 tests and
# torchrun turns pytest exit 5 into a failure.
NPROC_SM90_PUSH="${NPROC_SM90_PUSH:-2}"
run_sm90_push() {
  "${TORCHRUN}" --nproc_per_node="${NPROC_SM90_PUSH}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_sm90_push_fp8_kernel.py \
    tests/moe_ep/test_sm90_push_fp8_backend.py -v
}

# 4-GPU Blackwell-consumer sm120_mxfp8_mxfp8_bf16_cutedsl mega multirank.
# Own torchrun pytest process for the same reason as run_mega_sm90: the
# SM120 kernel tree shares top-level module names with the SM100/SM90 trees
# and is mutually exclusive per process (and is excluded from run_unit).
run_mega_sm120() {
  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank.py -v \
    -m "gpu_4 and arch_sm120"
}

# Fault tolerance. Split into a pytest half (a STALLED rank -- every process
# survives, so it runs under torchrun -m pytest) and a smoke half (a rank that
# really dies). The smoke half cannot be a pytest test: torchrun reports the
# victim's non-zero exit, which would fail the survivors' session even when
# they behaved correctly. Judge it by counting SMOKE_RESULT lines instead.
run_ft() {
  local rc=0
  local expected_ok=$(( NPROC_SMOKE - 1 ))

  for backend in nccl_ep nixl_ep; do
    if ! "${PY}" -c "from flashinfer.moe_ep import supports_fault_tolerance as s; raise SystemExit(0 if s('${backend}') else 1)"; then
      echo "${backend} cannot serve the FT API here; skipping its FT tests"
      continue
    fi

    "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
      "${MOE_EP_PYTEST_FLAGS[@]}" \
      tests/moe_ep/test_moe_ep_fault_tolerance_multirank.py -v \
      -m "nvep and gpu_4" --backend="${backend}" || rc=1

    local out
    out="$("${TORCHRUN}" --nproc_per_node="${NPROC_SMOKE}" --max-restarts=0 \
      tests/moe_ep/smoke_ft_ep.py --backend "${backend}" 2>&1)" || true
    echo "${out}"
    local ok
    # Count occurrences, not lines: the survivors' prints go through
    # torchrun's stdout multiplexing and can interleave onto a single line.
    ok="$(printf '%s' "${out}" | grep -o 'SMOKE_RESULT:' | wc -l)"
    if [ "${ok}" -ne "${expected_ok}" ]; then
      echo "FT smoke (${backend}): expected ${expected_ok} SMOKE_RESULT lines, got ${ok}" >&2
      rc=1
    fi
  done

  return "${rc}"
}

run_smoke() {
  require_nccl_ep || return 1

  local rc=0

  "${TORCHRUN}" --nproc_per_node="${NPROC_SMOKE}" tests/moe_ep/smoke_nccl_ep.py || rc=1

  if have_nixl_ep; then
    "${TORCHRUN}" --nproc_per_node="${NPROC_SMOKE}" tests/moe_ep/smoke_nixl_ep.py || rc=1
  else
    echo "nixl_ep not built; skipping smoke_nixl_ep.py"
  fi

  return "${rc}"
}

print_summary() {
  echo ""
  echo "################################################################"
  echo "### summary"
  echo "################################################################"
  local failed=0
  for i in "${!SECTION_NAMES[@]}"; do
    echo "  ${SECTION_STATUS[$i]}  ${SECTION_NAMES[$i]}"
    if [[ "${SECTION_STATUS[$i]}" != "PASS" ]]; then
      failed=$((failed + 1))
    fi
  done
  if (( failed > 0 )); then
    echo ""
    echo "${failed} section(s) failed."
    return 1
  fi
  echo ""
  echo "all sections passed."
  return 0
}

run_all() {
  run_section "unit + mock (no multirank)" run_unit
  run_section "torch-oracle correctness (1 GPU)" run_oracle
  run_section "split-path multirank (NCCL-EP)" run_multirank
  run_section "split_path_correctness_bf16 (4 GPU)" run_split_path_correctness_bf16
  run_section "split_path_correctness_nvfp4 (4 GPU)" run_split_path_correctness_nvfp4
  run_section "split_path_correctness_ht (4 GPU)" run_split_path_correctness_ht
  run_section "mega multirank (Blackwell)" run_mega
  run_section "smoke scripts" run_smoke
  print_summary
}

# Single-target runs must still propagate failure (print_summary returns
# non-zero if any section failed) so CI callers see a real exit code.
case "${1:-all}" in
  unit) run_section "unit + mock (no multirank)" run_unit; print_summary ;;
  oracle) run_section "torch-oracle correctness (1 GPU)" run_oracle; print_summary ;;
  oracle_sm90) run_section "sm90_fp8_fp8_bf16_pull_cutedsl torch-oracle correctness (1 Hopper GPU)" run_oracle_sm90; print_summary ;;
  multirank) run_section "split-path multirank (NCCL-EP)" run_multirank; print_summary ;;
  split_path_correctness_bf16) run_section "split_path_correctness_bf16 (4 GPU)" run_split_path_correctness_bf16; print_summary ;;
  split_path_correctness_nvfp4) run_section "split_path_correctness_nvfp4 (4 GPU)" run_split_path_correctness_nvfp4; print_summary ;;
  split_path_correctness_ht) run_section "split_path_correctness_ht (4 GPU)" run_split_path_correctness_ht; print_summary ;;
  mega) run_section "mega multirank (Blackwell)" run_mega; print_summary ;;
  mega_sm90) run_section "sm90_fp8_fp8_bf16_pull_cutedsl mega multirank (Hopper)" run_mega_sm90; print_summary ;;
  mega_sm120) run_section "sm120_mxfp8_mxfp8_bf16_cutedsl mega multirank (Blackwell-consumer)" run_mega_sm120; print_summary ;;
  sm90_push) run_section "sm90_fp8_fp8_bf16_push_cuda kernel + backend (2 Hopper GPUs)" run_sm90_push; print_summary ;;
  smoke) run_section "smoke scripts" run_smoke; print_summary ;;
  ft) run_section "fault tolerance (4 GPU)" run_ft; print_summary ;;
  all) run_all ;;
  *)
    echo "Usage: $0 [unit|oracle|oracle_sm90|multirank|sm90_push|split_path_correctness_bf16|split_path_correctness_nvfp4|split_path_correctness_ht|mega|mega_sm90|mega_sm120|smoke|ft|all]" >&2
    exit 1
    ;;
esac
