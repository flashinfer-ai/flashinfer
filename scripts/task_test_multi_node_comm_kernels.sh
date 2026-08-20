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
TEST_FILES="tests/comm/test_mnnvl_memory.py tests/comm/test_trtllm_mnnvl_allreduce.py tests/comm/test_mnnvl_moe_alltoall.py tests/comm/test_trtllm_allreduce_checkpoint.py"

# Main execution
main() {
    parse_args "$@"
    print_test_mode_banner

    TEST_FILES="$(filter_files_by_test_path "$TEST_FILES")"
    if [ -n "${TEST_PATH:-}" ] && [ -z "$TEST_FILES" ]; then
        echo "No multi-node files overlap TEST_PATH=${TEST_PATH}; skipping."
        exit 0
    fi

    install_and_verify

    echo "Multi-node comm kernel test files:"
    for test_file in $TEST_FILES; do
        echo "  $test_file"
    done
    echo ""

    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$TEST_FILES"
    else
        execute_tests "$TEST_FILES"
    fi

    exit "$EXIT_CODE"
}

main "$@"
