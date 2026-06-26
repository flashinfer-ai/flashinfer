#!/usr/bin/env bash
# Run all moe_ep tests (unit + multirank + smoke).
#
# Usage (from anywhere):
#   bash tests/moe_ep/run_tests.sh
#   bash tests/moe_ep/run_tests.sh unit          # host-only pytest only
#   bash tests/moe_ep/run_tests.sh multirank     # 4-GPU split-path only
#   bash tests/moe_ep/run_tests.sh mega          # Blackwell mega only
#   bash tests/moe_ep/run_tests.sh correctness  # 4-GPU LL + HT + NVFP4 numerics
#
# Requires:
#   - FLASHINFER repo root on PYTHONPATH (handled below)
#   - multirank/smoke: BUILD_NVEP=1 install, >=4 GPUs
#   - mega: Blackwell (sm_100+), deep_gemm, triton

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
# Avoid loading tests/conftest.py (full flashinfer.jit); moe_ep hooks live locally.
MOE_EP_PYTEST_FLAGS=(--confcutdir=tests/moe_ep)

declare -a SECTION_NAMES=()
declare -a SECTION_STATUS=()

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
    --ignore=tests/moe_ep/test_moe_ep_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py \
    --ignore=tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
    --ignore=tests/moe_ep/test_moe_ep_compute_correctness.py \
    --ignore=tests/moe_ep/test_moe_ep_compute_correctness_nvfp4.py \
    --ignore=tests/moe_ep/test_moe_ep_ht_correctness.py \
    -k "not multirank_roundtrip"
}

run_multirank() {
  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_layer_multirank.py -v \
    -m "nvep and gpu_4" --backend=nccl_ep

  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_split_kernels.py -v \
    -m "nvep and gpu_4" --backend=nccl_ep

  # NIXL-EP (requires nixl_ep backend built)
  # "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
  #   tests/moe_ep/test_moe_ep_layer_multirank.py -v \
  #   -m "nvep and gpu_4" --backend=nixl_ep
  #
  # "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
  #   tests/moe_ep/test_split_kernels.py -v \
  #   -m "nvep and gpu_4" --backend=nixl_ep
}

run_correctness() {
  NPROC_CORRECTNESS="${NPROC_CORRECTNESS:-4}"
  "${TORCHRUN}" --nproc_per_node="${NPROC_CORRECTNESS}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_compute_correctness.py \
    tests/moe_ep/test_moe_ep_compute_correctness_nvfp4.py \
    tests/moe_ep/test_moe_ep_ht_correctness.py -v \
    -m "nvep and gpu_4 and arch_blackwell" --backend=nccl_ep
}

run_mega() {
  "${TORCHRUN}" --nproc_per_node="${NPROC_MULTIRANK}" -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_moe_ep_mega_multirank.py \
    tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
    tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py -v \
    -m "gpu_4 and arch_blackwell"

  MEGA_NO_DIST=1 "${TORCHRUN}" --nproc_per_node=1 -m pytest \
    "${MOE_EP_PYTEST_FLAGS[@]}" \
    tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py -v \
    -m arch_blackwell
}

run_smoke() {
  "${TORCHRUN}" --nproc_per_node="${NPROC_SMOKE}" tests/moe_ep/smoke_nccl_ep.py

  # NIXL-EP smoke (requires nixl_ep backend built)
  # "${TORCHRUN}" --nproc_per_node="${NPROC_SMOKE}" tests/moe_ep/smoke_nixl_ep.py
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
  run_section "split-path multirank (NCCL-EP)" run_multirank
  run_section "compute correctness (4 GPU)" run_correctness
  run_section "mega multirank (Blackwell)" run_mega
  run_section "smoke scripts" run_smoke
  print_summary
}

case "${1:-all}" in
  unit) run_section "unit + mock (no multirank)" run_unit ;;
  multirank) run_section "split-path multirank (NCCL-EP)" run_multirank ;;
  correctness) run_section "compute correctness (4 GPU)" run_correctness ;;
  mega) run_section "mega multirank (Blackwell)" run_mega ;;
  smoke) run_section "smoke scripts" run_smoke ;;
  all) run_all ;;
  *)
    echo "Usage: $0 [unit|multirank|correctness|mega|smoke|all]" >&2
    exit 1
    ;;
esac
