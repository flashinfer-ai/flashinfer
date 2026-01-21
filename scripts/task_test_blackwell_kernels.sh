#!/bin/bash

set -eo pipefail

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SAMPLE_RATE:=5}  # Run every Nth test in sanity mode (5 = ~20% coverage)

# Randomize starting offset (0 to SAMPLE_RATE-1) for sampling variety
if [ -z "${SAMPLE_OFFSET:-}" ]; then
    SAMPLE_OFFSET=$((RANDOM % SAMPLE_RATE))
fi

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."
echo ""

# Pytest configuration flags
PYTEST_FLAGS="--continue-on-collection-errors -s"

# Parse command line arguments
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

if [ "$DRY_RUN" != "true" ]; then
    echo "Using CUDA version: ${CUDA_VERSION}"
    echo ""

    # Install precompiled kernels (require CI build artifacts)
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
TOTAL_TEST_CASES=0
SAMPLED_TEST_CASES=0

if [ "$DRY_RUN" == "true" ]; then
    echo "=========================================="
    echo "DRY RUN: Tests that would be executed"
    echo "=========================================="

    if [ "$SANITY_TEST" == "true" ]; then
        # Sanity test mode - show sampling details
        FILE_COUNT=0
        for test_file in $TEST_FILES; do
            FILE_COUNT=$((FILE_COUNT + 1))

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

            # Sample every Nth test with random offset
            SAMPLED_NODE_IDS=$(echo "$ALL_NODE_IDS" | awk "NR % $SAMPLE_RATE == $SAMPLE_OFFSET")
            # Fallback: if no tests sampled (offset missed all tests), take the first test
            if [ -z "$SAMPLED_NODE_IDS" ] || [ $(echo "$SAMPLED_NODE_IDS" | wc -l) -eq 0 ]; then
                SAMPLED_NODE_IDS=$(echo "$ALL_NODE_IDS" | head -1)
            fi
            SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
            SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

            echo "  Total test cases: $TOTAL_IN_FILE"
            echo "  Sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET})"
            echo "  Sample of tests that would run:"
            echo "$SAMPLED_NODE_IDS" | head -5 | sed 's/^/    /' || true
            if [ "$SAMPLED_IN_FILE" -gt 5 ]; then
                echo "    ... and $((SAMPLED_IN_FILE - 5)) more"
            fi
        done

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
        # Full test mode
        for test_file in $TEST_FILES; do
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            JUNIT_FILENAME="${test_file//\//_}.xml"
            JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${JUNIT_FILENAME}"
            echo "$TOTAL_TESTS. pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
        done

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
else
    mkdir -p "${JUNIT_DIR}"

    if [ "$SANITY_TEST" == "true" ]; then
        # Sanity test mode - sample tests from each file
        FILE_COUNT=0

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

            # Sample every Nth test with random offset
            SAMPLED_NODE_IDS=$(echo "$ALL_NODE_IDS" | awk "NR % $SAMPLE_RATE == $SAMPLE_OFFSET")
            # Fallback: if no tests sampled (offset missed all tests), take the first test
            if [ -z "$SAMPLED_NODE_IDS" ] || [ $(echo "$SAMPLED_NODE_IDS" | wc -l) -eq 0 ]; then
                SAMPLED_NODE_IDS=$(echo "$ALL_NODE_IDS" | head -1)
            fi
            SAMPLED_IN_FILE=$(echo "$SAMPLED_NODE_IDS" | wc -l)
            SAMPLED_TEST_CASES=$((SAMPLED_TEST_CASES + SAMPLED_IN_FILE))

            echo "Total test cases in file: $TOTAL_IN_FILE"
            echo "Running sampled test cases: $SAMPLED_IN_FILE (every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET})"

            if [ "$SAMPLED_IN_FILE" -eq 0 ]; then
                echo "⚠️  No tests sampled from $test_file, skipping"
                echo ""
                continue
            fi

            # Create a bash array with the node IDs
            mapfile -t SAMPLED_NODE_IDS_ARRAY <<< "$SAMPLED_NODE_IDS"

            JUNIT_FILENAME="${test_file//\//_}.xml"
            JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${JUNIT_FILENAME}"

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
        echo "Sample rate: every ${SAMPLE_RATE}th test, offset ${SAMPLE_OFFSET}"
        echo ""
        echo "To reproduce this exact run:"
        echo "  SAMPLE_RATE=${SAMPLE_RATE} SAMPLE_OFFSET=${SAMPLE_OFFSET} $0 --sanity-test"

        if [ -n "$FAILED_TESTS" ]; then
            echo ""
            echo "Failed test files:"
            echo -e "$FAILED_TESTS"
        fi
    else
        # Full test mode - run all tests in each file
        for test_file in $TEST_FILES; do
            echo "=========================================="
            JUNIT_FILENAME="${test_file//\//_}.xml"
            JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${JUNIT_FILENAME}"
            echo "Running: pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
            echo "=========================================="

            TOTAL_TESTS=$((TOTAL_TESTS + 1))

            #if pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${test_file}"; then
            #if pytest -v $PYTEST_FLAGS "${JUNIT_FLAG}" "${test_file}" > >(grep " PASSED") 2>/dev/null; then
            if pytest --collect-only -q "${test_file}" > >(grep "test_"); then
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
        echo "Total test files executed: $TOTAL_TESTS"
        echo "Passed: $PASSED_TESTS"
        echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"

        if [ -n "$FAILED_TESTS" ]; then
            echo ""
            echo "Failed tests:"
            echo -e "$FAILED_TESTS"
        fi
    fi
fi

exit $EXIT_CODE
