#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set MPI command prefix for multi-GPU tests
: "${PYTEST_COMMAND_PREFIX:=mpirun -np 4}"

# Disable sanity testing for multi-GPU tests (always run full suite)
# shellcheck disable=SC2034  # Used by common_test_functions.sh
DISABLE_SANITY_TEST=true

# Source common test functions
# shellcheck disable=SC1091  # File exists, checked separately
source "${SCRIPT_DIR}/test_utils.sh"

# Define the specific test files for multi-GPU comm tests (single-node)
# MPI_TEST_FILES="tests/comm/test_allreduce_unified_api.py tests/comm/test_allreduce_negative.py tests/comm/test_trtllm_allreduce_fusion.py"
# Add others back once they are fixed
MPI_TEST_FILES="tests/comm/test_allreduce_unified_api.py"

# These tests create their own distributed workers with torch.multiprocessing.spawn.
# Running them under mpirun would start multiple pytest parents and can collide on
# TCPStore ports.
SPAWN_MANAGED_TEST_FILES="tests/comm/test_quantized_allreduce.py"

# Tests that require torchrun instead of mpirun
TORCHRUN_TEST_FILES="tests/attention/test_parallel_attention.py tests/gemm/test_multi_gpu_cute_dsl_blockscaled_gemm_fusion.py"
: "${TORCHRUN_PREFIX:=torchrun --nproc_per_node=4}"

_run_pytest_group() {
    local files="$1"
    if [ -z "$files" ]; then
        echo "(none selected by TEST_PATH)"
        echo ""
        return 0
    fi
    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$files"
    else
        execute_tests "$files"
    fi
}

# Main execution
main() {
    parse_args "$@"
    print_test_mode_banner

    MPI_TEST_FILES="$(filter_files_by_test_path "$MPI_TEST_FILES")"
    SPAWN_MANAGED_TEST_FILES="$(filter_files_by_test_path "$SPAWN_MANAGED_TEST_FILES")"
    TORCHRUN_TEST_FILES="$(filter_files_by_test_path "$TORCHRUN_TEST_FILES")"
    if [ -n "${TEST_PATH:-}" ] && [ -z "${MPI_TEST_FILES}${SPAWN_MANAGED_TEST_FILES}${TORCHRUN_TEST_FILES}" ]; then
        echo "No multi-GPU files overlap TEST_PATH=${TEST_PATH}; skipping."
        exit 0
    fi

    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/setup_test_env.sh"
    install_and_verify

    echo "Multi-GPU comm kernel test files (running with: ${PYTEST_COMMAND_PREFIX}):"
    for test_file in $MPI_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""
    _run_pytest_group "$MPI_TEST_FILES"

    echo "Spawn-managed multi-GPU comm test files (running with plain pytest):"
    for test_file in $SPAWN_MANAGED_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""
    PYTEST_COMMAND_PREFIX= _run_pytest_group "$SPAWN_MANAGED_TEST_FILES"

    echo "Multi-GPU torchrun test files:"
    for test_file in $TORCHRUN_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""

    for test_file in $TORCHRUN_TEST_FILES; do
        echo "=========================================="
        echo "Running: ${TORCHRUN_PREFIX} -m pytest ${test_file} -v"
        echo "=========================================="
        if [ "$DRY_RUN" != "true" ]; then
            if ${TORCHRUN_PREFIX} -m pytest "${test_file}" -v; then
                echo "PASSED: $test_file"
            else
                echo "FAILED: $test_file"
                EXIT_CODE=1
            fi
        fi
        echo ""
    done

    exit "$EXIT_CODE"
}

main "$@"
