#!/bin/bash

set -eo pipefail

export PARALLEL_TESTS=true  # Enable parallel test execution for unit tests (auto-discovery mode)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common test functions
# shellcheck disable=SC1091  # File exists, checked separately
source "${SCRIPT_DIR}/test_utils.sh"

# TODO: Remove once CI container ships with nvshmem4py pre-installed.
pip install nvshmem4py-cu12

# Find and filter test files based on pytest.ini exclusions
find_test_files() {
    SEARCH_DIR="${TEST_PATH:-tests/}"

    if [ -n "$TEST_PATH" ]; then
        if [ ! -d "${SEARCH_DIR}" ]; then
            echo "ERROR: TEST_PATH '${SEARCH_DIR}' does not exist or is not a directory."
            echo "Available test directories:"
            find tests/ -maxdepth 1 -type d | sort | tail -n +2 | sed 's/^/  /'
            exit 1
        fi
        echo "🎯 TEST_PATH set: scoping test discovery to ${SEARCH_DIR}"
        echo ""
    fi

    echo "Reading pytest.ini for excluded directories..."
    EXCLUDED_DIRS=""
    if [ -f "./pytest.ini" ]; then
        # Extract norecursedirs from pytest.ini and convert to array
        NORECURSEDIRS=$(grep "^norecursedirs" ./pytest.ini | sed 's/norecursedirs\s*=\s*//' | sed 's/#.*//')
        if [ -n "$NORECURSEDIRS" ]; then
            EXCLUDED_DIRS=$(echo "$NORECURSEDIRS" | tr ',' ' ' | tr -s ' ')
            echo "⚠️  WARNING: Excluding directories from pytest.ini: $EXCLUDED_DIRS"
            echo ""
        fi
    fi

    echo "Finding all test_*.py files in ${SEARCH_DIR} directory..."

    # Find all test_*.py files
    ALL_TEST_FILES=$(find "${SEARCH_DIR}" -name "test_*.py" -type f | sort)

    # Filter out excluded files based on directory exclusions
    TEST_FILES=""
    for test_file in $ALL_TEST_FILES; do
        exclude_file=false
        test_dir=$(dirname "$test_file")

        for excluded_dir in $EXCLUDED_DIRS; do
            excluded_dir=$(echo "$excluded_dir" | xargs)  # trim whitespace
            if [ -n "$excluded_dir" ]; then
                # Check if this file's directory should be excluded
                if [[ "$test_dir" == *"/$excluded_dir" ]] || [[ "$test_dir" == "tests/$excluded_dir" ]] || [[ "$test_dir" == *"/$excluded_dir/"* ]]; then
                    exclude_file=true
                    break
                fi
            fi
        done

        if [ "$exclude_file" = false ]; then
            TEST_FILES="$TEST_FILES $test_file"
        fi
    done

    # Clean up whitespace
    TEST_FILES=$(echo "$TEST_FILES" | xargs)

    if [ -z "$TEST_FILES" ]; then
        echo "No test files found in ${SEARCH_DIR} directory (after exclusions)"
        exit 1
    fi

    echo "Found test files:"
    for test_file in $TEST_FILES; do
        echo "  $test_file"
    done
    echo ""
}

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    # Print test mode banner
    print_test_mode_banner

    # Install and verify (includes precompiled kernels)
    install_and_verify

    # apply dependency overrides after installation since pip may overwrite
    source "${SCRIPT_DIR}/setup_test_env.sh"

    # Find test files (unique to unit tests - auto-discovery)
    find_test_files

    # Execute tests or dry run
    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$TEST_FILES"
    else
        execute_tests "$TEST_FILES"
    fi

    exit "$EXIT_CODE"
}

main "$@"
