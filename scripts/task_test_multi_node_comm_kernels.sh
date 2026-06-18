#!/bin/bash

set -eo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Source test environment setup (handles package overrides like TVM-FFI)
source "${SCRIPT_DIR}/setup_test_env.sh"

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
# echo "Cleaning Python bytecode cache..."
# find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
# find . -type f -name '*.pyc' -delete 2>/dev/null || true
# echo "Cache cleaned."
# echo ""

# Disable sanity testing for multi-node tests (always run full suite)
# shellcheck disable=SC2034  # Used by common_test_functions.sh
DISABLE_SANITY_TEST=true

# Source common test functions
# shellcheck disable=SC1091  # File exists, checked separately
source "${SCRIPT_DIR}/test_utils.sh"

# Define the specific test files for multi-node comm tests
TEST_FILES="tests/comm/test_mnnvl_memory.py tests/comm/test_trtllm_mnnvl_allreduce.py tests/comm/test_mnnvl_moe_alltoall.py"

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    # Print test mode banner
    print_test_mode_banner

    # Install and verify (unless dry run)
    install_and_verify

    # Print test files
    echo "Multi-node comm kernel test files:"
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

    exit "$EXIT_CODE"
}

main "$@"
