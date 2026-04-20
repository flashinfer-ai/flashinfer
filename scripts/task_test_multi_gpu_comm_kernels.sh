#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source test environment setup (handles package overrides like TVM-FFI)
source "${SCRIPT_DIR}/setup_test_env.sh"

# Disable sanity testing for multi-GPU tests (always run full suite)
# shellcheck disable=SC2034  # Used by common_test_functions.sh
DISABLE_SANITY_TEST=true

# Source common test functions
# shellcheck disable=SC1091  # File exists, checked separately
source "${SCRIPT_DIR}/test_utils.sh"

# =============================================================================
# Test file definitions
# =============================================================================

# MPI-based tests: require mpirun to launch multiple ranks
MPI_TEST_FILES="tests/comm/test_allreduce_unified_api.py tests/comm/test_allreduce_negative.py"

# Self-spawn tests: spawn their own subprocesses via multiprocessing, run with plain pytest
SPAWN_TEST_FILES="\
tests/comm/test_vllm_custom_allreduce.py \
tests/comm/test_mnnvl_custom_comm.py \
tests/comm/test_trtllm_mnnvl_allreduce_custom_comm.py \
tests/comm/test_trtllm_allreduce.py \
tests/comm/test_trtllm_allreduce_fusion.py \
tests/comm/test_trtllm_moe_allreduce_fusion.py \
tests/comm/test_trtllm_moe_allreduce_fusion_finalize.py \
tests/comm/test_allreduce_fusion_moe_unified_api.py"

# NVSHMEM tests: require nvshmem4py, run separately if available
NVSHMEM_TEST_FILES="tests/comm/test_nvshmem.py tests/comm/test_nvshmem_allreduce.py"

# =============================================================================
# Main execution
# =============================================================================

main() {
    EXIT_CODE=0
    # Parse command line arguments
    parse_args "$@"

    # Print test mode banner
    print_test_mode_banner

    # Install and verify (unless dry run)
    install_and_verify

    # When running inside a container launched by srun -N 1, SLURM and PMI
    # env vars leak in. This confuses mpirun's Hydra launcher (auto-detects
    # Slurm, tries to use srun which doesn't exist in the container) and test
    # helpers (misread rank/world_size). Unset ALL of them so mpirun and
    # torch.distributed behave identically to a bare node.
    if [ "${SLURM_NTASKS:-0}" -le 1 ]; then
        while IFS='=' read -r var _; do
            unset "$var"
        done < <(env | grep -E '^(SLURM_|PMI_)')
    fi

    # --- Phase 1: MPI-based tests (run with mpirun) ---
    : "${PYTEST_COMMAND_PREFIX:=mpirun -np 4}"
    echo "=== Phase 1: MPI-based multi-GPU tests (running with: ${PYTEST_COMMAND_PREFIX}) ==="
    for test_file in $MPI_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""

    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$MPI_TEST_FILES"
    else
        execute_tests "$MPI_TEST_FILES"
    fi

    # --- Phase 2: Self-spawn tests (run with plain pytest) ---
    PYTEST_COMMAND_PREFIX=""
    echo ""
    echo "=== Phase 2: Self-spawn multi-GPU tests (running with: pytest) ==="
    for test_file in $SPAWN_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""

    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$SPAWN_TEST_FILES"
    else
        execute_tests "$SPAWN_TEST_FILES"
    fi

    # --- Phase 3: NVSHMEM tests (optional, requires nvshmem4py) ---
    if python3 -c "import nvshmem4py" &>/dev/null || pip install -q nvshmem4py-cu12 2>/dev/null; then
        PYTEST_COMMAND_PREFIX=""
        echo ""
        echo "=== Phase 3: NVSHMEM multi-GPU tests ==="
        for test_file in $NVSHMEM_TEST_FILES; do
            echo "  $test_file"
        done
        echo ""

        if [ "$DRY_RUN" == "true" ]; then
            execute_dry_run "$NVSHMEM_TEST_FILES"
        else
            execute_tests "$NVSHMEM_TEST_FILES"
        fi
    else
        echo ""
        echo "=== Phase 3: Skipping NVSHMEM tests (nvshmem4py-cu12 not available) ==="
        echo ""
    fi

    exit "$EXIT_CODE"
}

main "$@"
