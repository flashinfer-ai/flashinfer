#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set MPI command prefix for multi-GPU tests
: ${PYTEST_COMMAND_PREFIX:="mpirun -np 4"}

# Source common test functions
source "${SCRIPT_DIR}/common_test_functions.sh"

# Define the specific test files for multi-GPU comm tests (single-node)
TEST_FILES="tests/comm/test_allreduce_unified_api.py tests/comm/test_allreduce_negative.py tests/comm/test_trtllm_allreduce_fusion.py"

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    # Clean Python cache
    clean_python_cache

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

    exit $EXIT_CODE
}

main "$@"
