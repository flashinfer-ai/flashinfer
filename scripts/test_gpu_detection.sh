#!/bin/bash
# Quick test script to verify GPU detection logic

# Source the test_utils to get the detect_gpus function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test_utils.sh"

# Test tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Helper function to assert equality
assert_equals() {
    local expected=$1
    local actual=$2
    local test_name=$3

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    if [ "$expected" = "$actual" ]; then
        echo "✅ PASS: $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo "❌ FAIL: $test_name"
        echo "   Expected: '$expected'"
        echo "   Got:      '$actual'"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

echo "Testing GPU detection..."
echo ""

# Test 1: With CUDA_VISIBLE_DEVICES set (comma-separated)
echo "Test 1: CUDA_VISIBLE_DEVICES='0,1,2,3'"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PARALLEL_TESTS=true
DETECTED=$(detect_gpus)
assert_equals "0 1 2 3" "$DETECTED" "Comma-separated GPU list"
echo ""

# Test 2: With CUDA_VISIBLE_DEVICES set (space-separated)
echo "Test 2: CUDA_VISIBLE_DEVICES='0 1 2 3'"
export CUDA_VISIBLE_DEVICES="0 1 2 3"
DETECTED=$(detect_gpus)
assert_equals "0 1 2 3" "$DETECTED" "Space-separated GPU list"
echo ""

# Test 3: Single GPU
echo "Test 3: CUDA_VISIBLE_DEVICES='0'"
export CUDA_VISIBLE_DEVICES="0"
DETECTED=$(detect_gpus)
assert_equals "0" "$DETECTED" "Single GPU"
echo ""

# Test 4: Parallel disabled
echo "Test 4: PARALLEL_TESTS=false"
export PARALLEL_TESTS=false
export CUDA_VISIBLE_DEVICES="0,1,2,3"
DETECTED=$(detect_gpus)
assert_equals "0" "$DETECTED" "Parallel disabled (should return single GPU)"
echo ""

# Test 5: Try nvidia-smi (if available)
echo "Test 5: Using nvidia-smi (CUDA_VISIBLE_DEVICES unset)"
unset CUDA_VISIBLE_DEVICES
export PARALLEL_TESTS=true
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi available - checking detection"
    DETECTED=$(detect_gpus)
    # Can't assert exact value as it depends on system, but should not be empty
    if [ -n "$DETECTED" ]; then
        echo "✅ PASS: nvidia-smi detection (detected: $DETECTED)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "❌ FAIL: nvidia-smi detection returned empty"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
else
    echo "nvidia-smi not available - testing fallback"
    DETECTED=$(detect_gpus)
    assert_equals "0" "$DETECTED" "Fallback to default GPU when nvidia-smi unavailable"
fi
echo ""

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Total:  $TESTS_TOTAL"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi
