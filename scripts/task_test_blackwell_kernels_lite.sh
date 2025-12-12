#!/bin/bash

set -eo pipefail

: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SAMPLE_RATE:=5}  # Run every Nth test (5 = ~20% coverage)

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."
echo ""

echo "🔬 SANITY TEST MODE - Running every ${SAMPLE_RATE}th test (~$((100 / SAMPLE_RATE))% coverage)"
echo ""

# Pytest configuration flags
PYTEST_FLAGS="--continue-on-collection-errors -s"

# Check for dry-run mode
DRY_RUN=false
if [[ "$1" == "--dry-run" ]] || [[ "${DRY_RUN}" == "true" ]]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No tests will be executed"
    echo ""
fi

if [ "$DRY_RUN" != "true" ]; then

    # TODO: add code to install precompiled kernels and jit-cache for the CUDA path versions we will add.
    # (Similar to lines 28 - 96 in task_test_blackwell_kernels.sh, but for different CUDA versions)

    # Install local python sources
    pip install -e . -v --no-deps
    echo ""

    # Verify installation
    echo "Verifying installation..."
    (cd /tmp && python -m flashinfer show-config)
    echo ""
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

echo "Finding all test_*.py files in tests/ directory..."

# Find all test_*.py files
ALL_TEST_FILES=$(find tests/ -name "test_*.py" -type f | sort)

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
    echo "No test files found in tests/ directory (after exclusions)"
    exit 1
fi

echo "Found test files:"
for test_file in $TEST_FILES; do
    echo "  $test_file"
done
echo ""

FAILED_TESTS=""
TOTAL_TESTS=0
PASSED_TESTS=0

if [ "$DRY_RUN" == "true" ]; then
    echo "=========================================="
    echo "DRY RUN: Tests that would be executed"
    echo "=========================================="

    FILE_COUNT=0
    TOTAL_TEST_CASES=0
    SAMPLED_TEST_CASES=0

    for test_file in $TEST_FILES; do
        FILE_COUNT=$((FILE_COUNT + 1))

        # Collect all test node IDs for this file
        echo ""
        echo "[$FILE_COUNT] Collecting tests from: $test_file"

        # Temporarily disable exit on error for collection
        set +e
        COLLECTION_OUTPUT=$(pytest --collect-only -q "$test_file" 2>&1)
        COLLECTION_EXIT_CODE=$?
        set -e

        ALL_NODE_IDS=$(echo "$COLLECTION_OUTPUT" | grep "::" || true)

        if [ -z "$ALL_NODE_IDS" ]; then
            if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
                echo "  ⚠️  Collection failed for $test_file (skipping)"
            else
                echo "  ⚠️  No tests found in $test_file"
            fi
            continue
        fi

        # Count total tests
        TOTAL_IN_FILE=$(echo "$ALL_NODE_IDS" | wc -l)
        TOTAL_TEST_CASES=$((TOTAL_TEST_CASES + TOTAL_IN_FILE))

        # Sample every Nth test
        SAMPLED_NODE_IDS=$(echo "$ALL_NODE_IDS" | awk "NR % $SAMPLE_RATE == 1")
        SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
        SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

        echo "  Total test cases: $TOTAL_IN_FILE"
        echo "  Sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test)"
        echo "  Sample of tests that would run:"
        echo "$SAMPLED_NODE_IDS" | head -5 | sed 's/^/    /' || true
        if [ "$SAMPLED_IN_FILE" -gt 5 ]; then
            echo "    ... and $((SAMPLED_IN_FILE - 5)) more"
        fi
    done

    echo ""
    echo "=========================================="
    echo "DRY RUN SUMMARY"
    echo "=========================================="
    echo "Total test files: $FILE_COUNT"
    echo "Total test cases (full suite): $TOTAL_TEST_CASES"
    echo "Sampled test cases (sanity): $SAMPLED_TEST_CASES"
    if [ "$TOTAL_TEST_CASES" -gt 0 ]; then
        echo "Coverage: ~$((SAMPLED_TEST_CASES * 100 / TOTAL_TEST_CASES))%"
    else
        echo "Coverage: N/A (no tests collected)"
    fi
    echo "Sample rate: every ${SAMPLE_RATE}th test"
    echo ""
    echo "To actually run the tests, execute without --dry-run:"
    echo "  $0"
    echo "Or set DRY_RUN=false $0"
else
    mkdir -p "${JUNIT_DIR}"

    FILE_COUNT=0
    TOTAL_TEST_CASES=0
    SAMPLED_TEST_CASES=0

    for test_file in $TEST_FILES; do
        FILE_COUNT=$((FILE_COUNT + 1))

        echo "=========================================="
        echo "[$FILE_COUNT] Processing: $test_file"
        echo "=========================================="

        # Collect all test node IDs for this file
        echo "Collecting test cases..."

        # Temporarily disable exit on error for collection
        set +e
        COLLECTION_OUTPUT=$(pytest --collect-only -q "$test_file" 2>&1)
        COLLECTION_EXIT_CODE=$?
        set -e

        ALL_NODE_IDS=$(echo "$COLLECTION_OUTPUT" | grep "::" || true)

        if [ -z "$ALL_NODE_IDS" ]; then
            if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
                echo "⚠️  Collection failed for $test_file (skipping)"
            else
                echo "⚠️  No tests found in $test_file"
            fi
            echo ""
            continue
        fi

        # Count total tests
        TOTAL_IN_FILE=$(echo "$ALL_NODE_IDS" | wc -l)
        TOTAL_TEST_CASES=$((TOTAL_TEST_CASES + TOTAL_IN_FILE))

        # Sample every Nth test
        SAMPLED_NODE_IDS=$(echo "$ALL_NODE_IDS" | awk "NR % $SAMPLE_RATE == 1")
        SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
        SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

        echo "Total test cases in file: $TOTAL_IN_FILE"
        echo "Running sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test)"

        if [ "$SAMPLED_IN_FILE" -eq 0 ]; then
            echo "⚠️  No tests sampled from $test_file, skipping"
            echo ""
            continue
        fi

        # Create a bash array with the node IDs
        mapfile -t SAMPLED_NODE_IDS_ARRAY <<< "$SAMPLED_NODE_IDS"

        JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${test_file}.xml"

        # Run pytest with the sampled node IDs
        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        if pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${SAMPLED_NODE_IDS_ARRAY[@]}"; then
            echo "✅ PASSED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo "❌ FAILED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
            FAILED_TESTS="$FAILED_TESTS\n  - $test_file"
            EXIT_CODE=1
        fi

        echo ""
    done

    echo "=========================================="
    echo "SANITY TEST SUMMARY"
    echo "=========================================="
    echo "Total test files executed: $TOTAL_TESTS"
    echo "Test files passed: $PASSED_TESTS"
    echo "Test files failed: $((TOTAL_TESTS - PASSED_TESTS))"
    echo ""
    echo "Total test cases (full suite): $TOTAL_TEST_CASES"
    echo "Sampled test cases (executed): $SAMPLED_TEST_CASES"
    if [ "$TOTAL_TEST_CASES" -gt 0 ]; then
        echo "Coverage: ~$((SAMPLED_TEST_CASES * 100 / TOTAL_TEST_CASES))%"
    else
        echo "Coverage: N/A (no tests collected)"
    fi
    echo "Sample rate: every ${SAMPLE_RATE}th test"

    if [ -n "$FAILED_TESTS" ]; then
        echo ""
        echo "Failed test files:"
        echo -e "$FAILED_TESTS"
    fi
fi

exit $EXIT_CODE
