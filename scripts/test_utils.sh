#!/bin/bash
# Common test functions for FlashInfer test scripts
# This file is meant to be sourced by test runner scripts

# Default environment variables
: "${JUNIT_DIR:=$(realpath ./junit)}"
: "${MAX_JOBS:=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SAMPLE_RATE:=5}"  # Run every Nth test in sanity mode (5 = ~20% coverage)

# Randomize starting offset (0 to SAMPLE_RATE-1) for sampling variety
if [ -z "${SAMPLE_OFFSET:-}" ]; then
    SAMPLE_OFFSET=$((RANDOM % SAMPLE_RATE))
fi

# Pytest configuration flags
PYTEST_FLAGS="--continue-on-collection-errors"

# Command prefix for pytest (e.g., "mpirun -np 4" for multi-GPU tests)
: "${PYTEST_COMMAND_PREFIX:=}"

# Global variables for test execution
FAILED_TESTS=""
TOTAL_TESTS=0
PASSED_TESTS=0
TOTAL_TEST_CASES=0
SAMPLED_TEST_CASES=0
# shellcheck disable=SC2034  # EXIT_CODE is used by calling scripts
EXIT_CODE=0

# Parse command line arguments
# Set DISABLE_SANITY_TEST=true before sourcing to disable sanity testing
: "${DISABLE_SANITY_TEST:=false}"

parse_args() {
    DRY_RUN=false
    SANITY_TEST=false
    for arg in "$@"; do
        case $arg in
            --dry-run)
                DRY_RUN=true
                ;;
            --sanity-test)
                if [ "$DISABLE_SANITY_TEST" = "true" ]; then
                    echo "⚠️  WARNING: Sanity testing is disabled for this test suite"
                    echo "    Running full tests instead"
                    echo ""
                else
                    SANITY_TEST=true
                fi
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

# Install precompiled kernels (CI build artifacts)
install_precompiled_kernels() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi

    JIT_ARCH_EFFECTIVE=""
    # Map CUDA_VERSION to CUDA_STREAM for artifact lookup
    if [[ "${CUDA_VERSION}" == cu* ]]; then
        CUDA_STREAM="${CUDA_VERSION}"
    elif [ "${CUDA_VERSION}" = "12.9.0" ]; then
        CUDA_STREAM="cu129"
    else
        CUDA_STREAM="cu130"
    fi
    echo "Using CUDA stream: ${CUDA_STREAM}"
    echo ""

    if [ -n "${JIT_ARCH}" ]; then
        # 12.0a for CUDA 12.9.0, 12.0f for CUDA 13.0.0
        if [ "${JIT_ARCH}" = "12.0" ]; then
            if [ "${CUDA_STREAM}" = "cu129" ]; then
                JIT_ARCH_EFFECTIVE="12.0a"
            else
                JIT_ARCH_EFFECTIVE="12.0f"
            fi
        else
            JIT_ARCH_EFFECTIVE="${JIT_ARCH}"
        fi

        echo "Using JIT_ARCH from environment: ${JIT_ARCH_EFFECTIVE}"
        DIST_CUBIN_DIR="../dist/${CUDA_STREAM}/${JIT_ARCH_EFFECTIVE}/cubin"
        DIST_JIT_CACHE_DIR="../dist/${CUDA_STREAM}/${JIT_ARCH_EFFECTIVE}/jit-cache"

        echo "==== Debug: listing artifact directories ===="
        echo "Tree under ../dist:"
        (cd .. && ls -al dist) || true
        echo ""
        echo "Tree under ../dist/${CUDA_STREAM}:"
        (cd .. && ls -al "dist/${CUDA_STREAM}") || true
        echo ""
        echo "Contents of ${DIST_CUBIN_DIR}:"
        ls -al "${DIST_CUBIN_DIR}" || true
        echo ""
        echo "Contents of ${DIST_JIT_CACHE_DIR}:"
        ls -al "${DIST_JIT_CACHE_DIR}" || true
        echo "============================================="

        if [ -d "${DIST_CUBIN_DIR}" ] && ls "${DIST_CUBIN_DIR}"/*.whl >/dev/null 2>&1; then
            echo "Installing flashinfer-cubin from ${DIST_CUBIN_DIR} ..."
            pip install -q "${DIST_CUBIN_DIR}"/*.whl
        else
            echo "ERROR: flashinfer-cubin wheel not found in ${DIST_CUBIN_DIR}. Ensure the CI build stage produced the artifact." >&2
        fi

        if [ -d "${DIST_JIT_CACHE_DIR}" ] && ls "${DIST_JIT_CACHE_DIR}"/*.whl >/dev/null 2>&1; then
            echo "Installing flashinfer-jit-cache from ${DIST_JIT_CACHE_DIR} ..."
            pip install -q "${DIST_JIT_CACHE_DIR}"/*.whl
        else
            echo "ERROR: flashinfer-jit-cache wheel not found in ${DIST_JIT_CACHE_DIR} for ${CUDA_VERSION}. Ensure the CI build stage produced the artifact." >&2
        fi
        echo ""
    fi
}

# Install and verify FlashInfer
install_and_verify() {
    if [ "$DRY_RUN" != "true" ]; then
        echo "Using CUDA version: ${CUDA_VERSION}"
        echo ""

        # Install precompiled kernels if enabled
        install_precompiled_kernels

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
    if [ -z "$SAMPLED_NODE_IDS" ] || [ "$(echo "$SAMPLED_NODE_IDS" | wc -l)" -eq 0 ]; then
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
    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX needs word splitting
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

    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX and PYTEST_FLAGS need word splitting
    if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${SAMPLED_NODE_IDS_ARRAY[@]}"; then
        echo "✅ PASSED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "❌ FAILED: $test_file ($SAMPLED_IN_FILE/$TOTAL_IN_FILE tests)"
        FAILED_TESTS="$FAILED_TESTS\n  - $test_file"
        # shellcheck disable=SC2034  # EXIT_CODE is used by calling scripts
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
    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX needs word splitting
    echo "Running: ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
    echo "=========================================="

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # shellcheck disable=SC2086  # PYTEST_COMMAND_PREFIX and PYTEST_FLAGS need word splitting
    if ${PYTEST_COMMAND_PREFIX} pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${test_file}"; then
        echo "✅ PASSED: $test_file"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "❌ FAILED: $test_file"
        FAILED_TESTS="$FAILED_TESTS\n  - $test_file"
        # shellcheck disable=SC2034  # EXIT_CODE is used by calling scripts
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
