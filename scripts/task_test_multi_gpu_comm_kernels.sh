#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the pre-install guards and optional dependency overrides.
source "${SCRIPT_DIR}/setup_test_env.sh"

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

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    # Print test mode banner
    print_test_mode_banner

    # Install and verify (unless dry run)
    install_and_verify

    # Print test files
    echo "Multi-GPU comm kernel test files (running with: ${PYTEST_COMMAND_PREFIX}):"
    for test_file in $MPI_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""

    # Execute tests or dry run
    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$MPI_TEST_FILES"
    else
        execute_tests "$MPI_TEST_FILES"
    fi

    echo "Spawn-managed multi-GPU comm test files (running with plain pytest):"
    for test_file in $SPAWN_MANAGED_TEST_FILES; do
        echo "  $test_file"
    done
    echo ""

    if [ "$DRY_RUN" == "true" ]; then
        PYTEST_COMMAND_PREFIX= execute_dry_run "$SPAWN_MANAGED_TEST_FILES"
    else
        PYTEST_COMMAND_PREFIX= execute_tests "$SPAWN_MANAGED_TEST_FILES"
    fi

    # Execute torchrun tests (torchrun requires -m pytest, not direct pytest invocation)
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
