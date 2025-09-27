#!/bin/bash

set -eo pipefail
set -x

: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

# Check for dry-run mode
DRY_RUN=false
if [[ "$1" == "--dry-run" ]] || [[ "${DRY_RUN}" == "true" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No tests will be executed"
    echo ""
fi

if [ "$DRY_RUN" != "true" ]; then
    pip install -e . -v
fi

EXIT_CODE=0

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

echo "Finding all test files in tests/ directory..."

# Build find command with exclusions
FIND_CMD="find tests/ -name \"test_*.py\" -type f"
for excluded_dir in $EXCLUDED_DIRS; do
    excluded_dir=$(echo "$excluded_dir" | xargs)  # trim whitespace
    if [ -n "$excluded_dir" ]; then
        FIND_CMD="$FIND_CMD -not -path \"tests/$excluded_dir/*\" -not -path \"tests/*/$excluded_dir/*\""
    fi
done

# Execute the find command
TEST_FILES=$(eval $FIND_CMD | sort)

if [ -z "$TEST_FILES" ]; then
    echo "No test files found in tests/ directory (after exclusions)"
    exit 1
fi

echo "Found test files:"
echo "$TEST_FILES"
echo ""

FAILED_TESTS=""
TOTAL_TESTS=0
PASSED_TESTS=0

if [ "$DRY_RUN" == "true" ]; then
    echo "=========================================="
    echo "DRY RUN: Tests that would be executed"
    echo "=========================================="

    for test_file in $TEST_FILES; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        echo "$TOTAL_TESTS. pytest $test_file"
    done

    echo ""
    echo "=========================================="
    echo "DRY RUN SUMMARY"
    echo "=========================================="
    echo "Total test files that would be executed: $TOTAL_TESTS"
    echo ""
    echo "To actually run the tests, execute without --dry-run:"
    echo "  $0"
    echo "Or set DRY_RUN=false $0"
else
    for test_file in $TEST_FILES; do
        echo "=========================================="
        echo "Running: $test_file"
        echo "=========================================="

        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        if pytest "$test_file"; then
            echo "✅ PASSED: $test_file"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "❌ FAILED: $test_file"
            FAILED_TESTS="$FAILED_TESTS\n  - $test_file"
            EXIT_CODE=1
        fi

        echo ""
    done

    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo "Total tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"

    if [ -n "$FAILED_TESTS" ]; then
        echo ""
        echo "Failed test files:"
        echo -e "$FAILED_TESTS"
    fi
fi

exit $EXIT_CODE
