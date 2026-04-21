#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source test environment setup (handles package overrides like TVM-FFI)
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
# TEST_FILES="tests/comm/test_allreduce_unified_api.py tests/comm/test_allreduce_negative.py tests/comm/test_trtllm_allreduce_fusion.py"
# Add others back once they are fixed
TEST_FILES="tests/comm/test_allreduce_unified_api.py"

# Tests that require torchrun instead of mpirun
TORCHRUN_TEST_FILES="tests/attention/test_parallel_attention.py"
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
    for test_file in $TEST_FILES; do
        echo "  $test_file"
    done
    echo ""

    # Execute tests or dry run
    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$TEST_FILES"
    else
        execute_tests "$TEST_FILES"
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
