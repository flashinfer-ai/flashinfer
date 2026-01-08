#!/bin/bash
# Common test functions for FlashInfer test scripts
# This file is meant to be sourced by test runner scripts

# Default environment variables
: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SAMPLE_RATE:=5}  # Run every Nth test in sanity mode (5 = ~20% coverage)

# Randomize starting offset (0 to SAMPLE_RATE-1) for sampling variety
if [ -z "${SAMPLE_OFFSET:-}" ]; then
    SAMPLE_OFFSET=$((RANDOM % SAMPLE_RATE))
fi

# Pytest configuration flags
PYTEST_FLAGS="--continue-on-collection-errors -s"

# Command prefix for pytest (e.g., "mpirun -np 4" for multi-GPU tests)
: ${PYTEST_COMMAND_PREFIX:=""}

# Global variables for test execution
FAILED_TESTS=""
TOTAL_TESTS=0
PASSED_TESTS=0
TOTAL_TEST_CASES=0
SAMPLED_TEST_CASES=0
EXIT_CODE=0

# Clean Python bytecode cache to avoid stale imports
clean_python_cache() {
    echo "Cleaning Python bytecode cache..."
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.pyc' -delete 2>/dev/null || true
    echo "Cache cleaned."
    echo ""
}

# Parse command line arguments
parse_args() {
    DRY_RUN=false
    SANITY_TEST=false
    for arg in "$@"; do
        case $arg in
            --dry-run)
                DRY_RUN=true
                ;;
            --sanity-test)
                SANITY_TEST=true
                ;;
        esac
    done
}

# Print test mode banner
print_test_mode_banner() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "🔍 DRY RUN MODE - No tests will be executed"
        echo ""
    fi

    if [ "$SANITY_TEST" = "true" ]; then
        echo "🔬 SANITY TEST MODE - Running every ${SAMPLE_RATE}th test (~$((100 / SAMPLE_RATE))% coverage)"
        echo "   Sampling pattern: offset=${SAMPLE_OFFSET} (tests #${SAMPLE_OFFSET}, #$((SAMPLE_OFFSET + SAMPLE_RATE)), #$((SAMPLE_OFFSET + SAMPLE_RATE * 2))...)"
        echo ""
    else
        echo "📋 FULL TEST MODE - Running all tests from each test file"
        echo ""
    fi
}

# Install and verify FlashInfer
install_and_verify() {
    if [ "$DRY_RUN" != "true" ]; then
        echo "Using CUDA version: ${CUDA_VERSION}"
        echo ""

        # Install local python sources
        pip install -e . -v --no-deps
        echo ""

        # Verify installation
        echo "Verifying installation..."
        (cd /tmp && python -m flashinfer show-config)
        echo ""
    fi
}

# Collect tests from a file
collect_tests() {
    local test_file=$1

    # Temporarily disable exit on error for collection
    set +e
    COLLECTION_OUTPUT=$(pytest --collect-only -q "$test_file" 2>&1)
    COLLECTION_EXIT_CODE=$?
    set -e

    ALL_NODE_IDS=$(echo "$COLLECTION_OUTPUT" | grep "::" || true)
}

# Sample tests based on SAMPLE_RATE and SAMPLE_OFFSET
sample_tests() {
    local all_node_ids=$1

    # Sample every Nth test with random offset
    SAMPLED_NODE_IDS=$(echo "$all_node_ids" | awk "NR % $SAMPLE_RATE == $SAMPLE_OFFSET")
    # Fallback: if no tests sampled (offset missed all tests), take the first test
    if [ -z "$SAMPLED_NODE_IDS" ] || [ $(echo "$SAMPLED_NODE_IDS" | wc -l) -eq 0 ]; then
        SAMPLED_NODE_IDS=$(echo "$all_node_ids" | head -1)
    fi
}

# Process a single test file for dry run (sanity mode)
dry_run_sanity_file() {
    local test_file=$1
    local file_count=$2

    echo ""
    echo "[$file_count] Collecting tests from: $test_file"

    collect_tests "$test_file"

    if [ -z "$ALL_NODE_IDS" ]; then
        if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
            echo "  ⚠️  Collection failed for $test_file (skipping)"
        else
            echo "  ⚠️  No tests found in $test_file"
        fi
        return
    fi

    # Count total tests
    TOTAL_IN_FILE=$(echo "$ALL_NODE_IDS" | wc -l)
    TOTAL_TEST_CASES=$((TOTAL_TEST_CASES + TOTAL_IN_FILE))

    sample_tests "$ALL_NODE_IDS"
    SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
    SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

    echo "  Total test cases: $TOTAL_IN_FILE"
    echo "  Sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET})"
    echo "  Sample of tests that would run:"
    echo "$SAMPLED_NODE_IDS" | head -5 | sed 's/^/    /' || true
    if [ "$SAMPLED_IN_FILE" -gt 5 ]; then
        echo "    ... and $((SAMPLED_IN_FILE - 5)) more"
    fi
}

# Process a single test file for dry run (full mode)
dry_run_full_file() {
    local test_file=$1

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    JUNIT_FILENAME="${test_file//\//_}.xml"
    JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${JUNIT_FILENAME}"
    echo "$TOTAL_TESTS. ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
}

# Print dry run summary
print_dry_run_summary() {
    if [ "$SANITY_TEST" == "true" ]; then
        echo ""
        echo "=========================================="
        echo "DRY RUN SUMMARY (SANITY MODE)"
        echo "=========================================="
        echo "Total test files: $FILE_COUNT"
        echo "Total test cases (full suite): $TOTAL_TEST_CASES"
        echo "Sampled test cases (sanity): $SAMPLED_TEST_CASES"
        if [ "$TOTAL_TEST_CASES" -gt 0 ]; then
            echo "Coverage: ~$((SAMPLED_TEST_CASES * 100 / TOTAL_TEST_CASES))%"
        else
            echo "Coverage: N/A (no tests collected)"
        fi
        echo "Sample rate: every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET}"
        echo ""
        echo "To reproduce this exact run:"
        echo "  SAMPLE_RATE=${SAMPLE_RATE} SAMPLE_OFFSET=${SAMPLE_OFFSET} $0 --sanity-test"
    else
        echo ""
        echo "=========================================="
        echo "DRY RUN SUMMARY"
        echo "=========================================="
        echo "Total test files that would be executed: $TOTAL_TESTS"
    fi

    echo ""
    echo "To actually run the tests, execute without --dry-run:"
    if [ "$SANITY_TEST" == "true" ]; then
        echo "  $0 --sanity-test"
        echo ""
        echo "To reproduce this exact sampling pattern:"
        echo "  SAMPLE_RATE=${SAMPLE_RATE} SAMPLE_OFFSET=${SAMPLE_OFFSET} $0 --sanity-test"
    else
        echo "  $0"
    fi
}

# Run a single test file in sanity mode
run_sanity_test_file() {
    local test_file=$1
    local file_count=$2

    echo "=========================================="
    echo "[$file_count] Processing: $test_file"
    echo "=========================================="

    echo "Collecting test cases..."

    collect_tests "$test_file"

    if [ -z "$ALL_NODE_IDS" ]; then
        if [ $COLLECTION_EXIT_CODE -ne 0 ]; then
            echo "⚠️  Collection failed for $test_file (skipping)"
        else
            echo "⚠️  No tests found in $test_file"
        fi
        echo ""
        return
    fi

    # Count total tests
    TOTAL_IN_FILE=$(echo "$ALL_NODE_IDS" | wc -l)
    TOTAL_TEST_CASES=$((TOTAL_TEST_CASES + TOTAL_IN_FILE))

    sample_tests "$ALL_NODE_IDS"
    SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
    SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

    echo "Total test cases in file: $TOTAL_IN_FILE"
    echo "Running sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET})"

    if [ "$SAMPLED_IN_FILE" -eq 0 ]; then
        echo "⚠️  No tests sampled from $test_file, skipping"
        echo ""
        return
    fi

    # Create a bash array with the node IDs
    mapfile -t SAMPLED_NODE_IDS_ARRAY <<< "$SAMPLED_NODE_IDS"

    JUNIT_FILENAME="${test_file//\//_}.xml"
    JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${JUNIT_FILENAME}"

    # Run pytest with the sampled node IDs
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${SAMPLED_NODE_IDS_ARRAY[@]}"; then
        echo "✅ PASSED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "❌ FAILED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
        FAILED_TESTS="$FAILED_TESTS\n  - $test_file"
        EXIT_CODE=1
    fi

    echo ""
}

# Run a single test file in full mode
run_full_test_file() {
    local test_file=$1

    echo "=========================================="
    JUNIT_FILENAME="${test_file//\//_}.xml"
    JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${JUNIT_FILENAME}"
    echo "Running: ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
    echo "=========================================="

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${test_file}"; then
        echo "✅ PASSED: $test_file"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "❌ FAILED: $test_file"
        FAILED_TESTS="$FAILED_TESTS\n  - $test_file"
        EXIT_CODE=1
    fi

    echo ""
}

# Print execution summary
print_execution_summary() {
    if [ "$SANITY_TEST" == "true" ]; then
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
        echo "Sample rate: every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET}"
        echo ""
        echo "To reproduce this exact run:"
        echo "  SAMPLE_RATE=${SAMPLE_RATE} SAMPLE_OFFSET=${SAMPLE_OFFSET} $0 --sanity-test"
    else
        echo "=========================================="
        echo "TEST SUMMARY"
        echo "=========================================="
        echo "Total test files executed: $TOTAL_TESTS"
        echo "Passed: $PASSED_TESTS"
        echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"
    fi

    if [ -n "$FAILED_TESTS" ]; then
        echo ""
        echo "Failed test files:"
        echo -e "$FAILED_TESTS"
    fi
}

# Main execution function for dry run mode
execute_dry_run() {
    local test_files=$1

    echo "=========================================="
    echo "DRY RUN: Tests that would be executed"
    echo "=========================================="

    if [ "$SANITY_TEST" == "true" ]; then
        FILE_COUNT=0
        for test_file in $test_files; do
            FILE_COUNT=$((FILE_COUNT + 1))
            dry_run_sanity_file "$test_file" "$FILE_COUNT"
        done
    else
        for test_file in $test_files; do
            dry_run_full_file "$test_file"
        done
    fi

    print_dry_run_summary
}

# Main execution function for actual test run
execute_tests() {
    local test_files=$1

    mkdir -p "${JUNIT_DIR}"

    if [ "$SANITY_TEST" == "true" ]; then
        FILE_COUNT=0
        for test_file in $test_files; do
            FILE_COUNT=$((FILE_COUNT + 1))
            run_sanity_test_file "$test_file" "$FILE_COUNT"
        done
    else
        for test_file in $test_files; do
            run_full_test_file "$test_file"
        done
    fi

    print_execution_summary
}
