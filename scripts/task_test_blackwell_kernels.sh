#!/bin/bash

set -eo pipefail

: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

# Clean Python bytecode cache to avoid stale imports (e.g., after module refactoring)
echo "Cleaning Python bytecode cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true
echo "Cache cleaned."
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
    # Detect CUDA stream from CUDA_VERSION (e.g., 13.0.1 -> 13.0 -> cu130)
    if [ -z "${CUDA_STREAM}" ]; then
        CUDA_NUM=$(echo "${CUDA_VERSION}" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/')
        case "${CUDA_NUM}" in
            12.8) CUDA_STREAM=cu128 ;;
            12.9) CUDA_STREAM=cu129 ;;
            13.0) CUDA_STREAM=cu130 ;;
            *) CUDA_STREAM=cu129 ;;
        esac
    fi
    echo "Using CUDA stream: ${CUDA_STREAM}"
    echo ""

    # Install precompiled kernels
    echo "Installing flashinfer-cubin from PyPI/index..."
    pip install -q flashinfer-cubin
    echo "Installing flashinfer-jit-cache for ${CUDA_STREAM} from https://flashinfer.ai/whl/${CUDA_STREAM} ..."
    pip install -q --extra-index-url "https://flashinfer.ai/whl/${CUDA_STREAM}" flashinfer-jit-cache
    echo ""

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

    for test_file in $TEST_FILES; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${test_file}.xml"
        echo "$TOTAL_TESTS. pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
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
    mkdir -p "${JUNIT_DIR}"
    for test_file in $TEST_FILES; do
        echo "=========================================="
        JUNIT_FLAG="--junitxml=${JUNIT_DIR}/${test_file}.xml"
        echo "Running: pytest $PYTEST_FLAGS ${JUNIT_FLAG} \"${test_file}\""
        echo "=========================================="

        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        if pytest $PYTEST_FLAGS "${JUNIT_FLAG}" "${test_file}"; then
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

exit $EXIT_CODE
