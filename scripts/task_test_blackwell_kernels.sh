#!/bin/bash

set -eo pipefail

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

echo "Finding test subdirectories in tests/ directory..."

# Find all subdirectories that contain test_*.py files
ALL_TEST_DIRS=$(find tests/ -name "test_*.py" -type f -exec dirname {} \; | sort -u)

# Filter out excluded directories
TEST_DIRS=""
for test_dir in $ALL_TEST_DIRS; do
    exclude_dir=false
    for excluded_dir in $EXCLUDED_DIRS; do
        excluded_dir=$(echo "$excluded_dir" | xargs)  # trim whitespace
        if [ -n "$excluded_dir" ]; then
            # Check if this directory should be excluded
            if [[ "$test_dir" == *"/$excluded_dir" ]] || [[ "$test_dir" == "tests/$excluded_dir" ]] || [[ "$test_dir" == *"/$excluded_dir/"* ]]; then
                exclude_dir=true
                break
            fi
        fi
    done

    if [ "$exclude_dir" = false ]; then
        TEST_DIRS="$TEST_DIRS $test_dir"
    fi
done

# Clean up whitespace
TEST_DIRS=$(echo "$TEST_DIRS" | xargs)

if [ -z "$TEST_DIRS" ]; then
    echo "No test directories found in tests/ directory (after exclusions)"
    exit 1
fi

echo "Found test directories:"
for test_dir in $TEST_DIRS; do
    test_count=$(find "$test_dir" -maxdepth 1 -name "test_*.py" -type f | wc -l)
    echo "  $test_dir ($test_count test files)"
done
echo ""

FAILED_TESTS=""
TOTAL_TESTS=0
PASSED_TESTS=0

if [ "$DRY_RUN" == "true" ]; then
    echo "=========================================="
    echo "DRY RUN: Tests that would be executed"
    echo "=========================================="

    for test_dir in $TEST_DIRS; do
        if [ "$test_dir" == "tests/utils" ] || [ "$test_dir" == "tests/comm" ]; then
            # Run utils and comm tests individually for debugging
            echo ""
            echo "📝 NOTE: $test_dir will be run individually for debugging"
            test_files=$(find "$test_dir" -maxdepth 1 -name "test_*.py" -type f | sort)
            for test_file in $test_files; do
                TOTAL_TESTS=$((TOTAL_TESTS + 1))
                echo "$TOTAL_TESTS. pytest $test_file"
            done
        else
            # Run other directories as groups
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            test_count=$(find "$test_dir" -maxdepth 1 -name "test_*.py" -type f | wc -l)
            echo "$TOTAL_TESTS. pytest $test_dir  (contains $test_count test files)"
        fi
    done

    echo ""
    echo "=========================================="
    echo "DRY RUN SUMMARY"
    echo "=========================================="
    echo "Total test commands that would be executed: $TOTAL_TESTS"
    echo ""
    echo "To actually run the tests, execute without --dry-run:"
    echo "  $0"
    echo "Or set DRY_RUN=false $0"
else
    for test_dir in $TEST_DIRS; do
        if [ "$test_dir" == "tests/utils" ] || [ "$test_dir" == "tests/comm" ]; then
            # Run utils and comm tests individually for debugging
            echo "=========================================="
            echo "Running $test_dir individually for debugging"
            echo "=========================================="

            test_files=$(find "$test_dir" -maxdepth 1 -name "test_*.py" -type f | sort)
            for test_file in $test_files; do
                echo "Running: pytest $test_file"
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
        else
            # Run other directories as groups
            echo "=========================================="
            echo "Running: pytest $test_dir"
            echo "=========================================="

            TOTAL_TESTS=$((TOTAL_TESTS + 1))

            if pytest "$test_dir"; then
                echo "✅ PASSED: $test_dir"
                PASSED_TESTS=$((PASSED_TESTS + 1))
            else
                echo "❌ FAILED: $test_dir"
                FAILED_TESTS="$FAILED_TESTS\n  - $test_dir"
                EXIT_CODE=1
            fi

            echo ""
        fi
    done

    echo "=========================================="
    echo "TEST SUMMARY"
    echo "=========================================="
    echo "Total test commands executed: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"

    if [ -n "$FAILED_TESTS" ]; then
        echo ""
        echo "Failed tests:"
        echo -e "$FAILED_TESTS"
    fi
fi

exit $EXIT_CODE
