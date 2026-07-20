#!/usr/bin/env bash
# Run moe_ep tests (unit + multirank + smoke + mega).
#
# Usage (from repo root):
#   bash tests/moe_ep/run_tests.sh
#   bash tests/moe_ep/run_tests.sh unit          # host-only pytest
#   bash tests/moe_ep/run_tests.sh multirank     # 4-GPU split path (NCCL-EP)
#   bash tests/moe_ep/run_tests.sh mega          # Blackwell mega multirank
#   bash tests/moe_ep/run_tests.sh split_path_correctness_bf16   # 4-GPU bf16 split-path numerics
#   bash tests/moe_ep/run_tests.sh split_path_correctness_nvfp4  # 4-GPU NVFP4 split-path numerics
#   bash tests/moe_ep/run_tests.sh split_path_correctness_ht     # 4-GPU HT (FLAT) split-path numerics
#   bash tests/moe_ep/run_tests.sh oracle        # 1-GPU torch-oracle correctness (all paths)
#   bash tests/moe_ep/run_tests.sh smoke         # torchrun smoke scripts
#
# Install (split NCCL-EP + mega runtime deps):
#   bash fast_install.sh
#   # equivalent: BUILD_NCCL_EP=1 pip install -e ".[nvep]" --no-build-isolation
#
# Requires:
#   - FLASHINFER repo root on PYTHONPATH (handled below)
#   - multirank/smoke/correctness: nccl.ep + staged libnccl_ep.so (fast_install.sh)
#   - multirank/smoke/correctness: >=4 GPUs
#   - mega: Blackwell (sm_100+), nvshmem, deep_gemm, triton
#   - optional NIXL smoke: FI_BUILD_NIXL_EP=1 / BUILD_NIXL_EP=1 install

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
  echo "Install with: bash fast_install.sh  (or BUILD_NCCL_EP=1 pip install -e \".[nvep]\")" >&2
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

run_unit() {
  "${PY}" -m pytest tests/moe_ep/ -v \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    --ignore=tests/moe_ep/test_moe_ep_layer_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_deep_gemm_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
    --ignore=tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py \
    --ignore=tests/moe_ep/test_deep_gemm_mega_kernel_vs_reference.py \
    --ignore=tests/moe_ep/test_split_fused_moe_kernel_vs_reference.py \
    --ignore=tests/moe_ep/test_moe_ep_compute_correctness.py \
    --ignore=tests/moe_ep/test_moe_ep_compute_correctness_nvfp4.py \
    --ignore=tests/moe_ep/test_moe_ep_ht_correctness.py \
    -k "not multirank_roundtrip"
}

run_multirank() {
  require_nccl_ep

  local rc=0

  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_layer_multirank.py -v \
    -m "nvep and gpu_4" --backend=nccl_ep || rc=1

  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_split_kernels.py -v \
    -m "nvep and gpu_4" --backend=nccl_ep || rc=1

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
    echo "nixl_ep not built; skipping NIXL multirank (set FI_BUILD_NIXL_EP=1 in fast_install.sh)"
  fi

  return "${rc}"
}

run_split_path_correctness_bf16() {
  require_nccl_ep

  NPROC_CORRECTNESS="${NPROC_CORRECTNESS:-4}"
  "${TORCHRUN}" --nproc_per_node="${NPROC_CORRECTNESS}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_compute_correctness.py -v \
    -m "nvep and gpu_4 and arch_blackwell" --backend=nccl_ep
}

run_split_path_correctness_nvfp4() {
  require_nccl_ep

  NPROC_CORRECTNESS="${NPROC_CORRECTNESS:-4}"
  "${TORCHRUN}" --nproc_per_node="${NPROC_CORRECTNESS}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_compute_correctness_nvfp4.py -v \
    -m "nvep and gpu_4 and arch_blackwell" --backend=nccl_ep
}

run_split_path_correctness_ht() {
  require_nccl_ep

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
    tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -v \
    -m arch_blackwell || rc=1

  # deep_gemm's symm buffer needs an initialized process group (no
  # MEGA_NO_DIST equivalent), hence the 1-proc torchrun.
  "${TORCHRUN}" --standalone --nproc_per_node=1 -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_deep_gemm_mega_kernel_vs_reference.py -v \
    -m arch_blackwell || rc=1

  return "${rc}"
}

run_mega() {
  local rc=0

  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_deep_gemm_mega_multirank.py \
    tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
    tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py -v \
    -m "gpu_4 and arch_blackwell" || rc=1

  MEGA_NO_DIST=1 "${TORCHRUN}" --nproc_per_node=1 -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
    tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -v \
    -m arch_blackwell || rc=1

  return "${rc}"
}

run_smoke() {
  require_nccl_ep

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
  multirank) run_section "split-path multirank (NCCL-EP)" run_multirank; print_summary ;;
  split_path_correctness_bf16) run_section "split_path_correctness_bf16 (4 GPU)" run_split_path_correctness_bf16; print_summary ;;
  split_path_correctness_nvfp4) run_section "split_path_correctness_nvfp4 (4 GPU)" run_split_path_correctness_nvfp4; print_summary ;;
  split_path_correctness_ht) run_section "split_path_correctness_ht (4 GPU)" run_split_path_correctness_ht; print_summary ;;
  mega) run_section "mega multirank (Blackwell)" run_mega; print_summary ;;
  smoke) run_section "smoke scripts" run_smoke; print_summary ;;
  all) run_all ;;
  *)
    echo "Usage: $0 [unit|oracle|multirank|split_path_correctness_bf16|split_path_correctness_nvfp4|split_path_correctness_ht|mega|smoke|all]" >&2
    exit 1
    ;;
esac
