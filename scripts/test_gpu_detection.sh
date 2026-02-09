#!/bin/bash
# Quick test script to verify GPU detection logic

# Source the test_utils to get the detect_gpus function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test_utils.sh"

echo "Testing GPU detection..."
echo ""

# Test 1: With CUDA_VISIBLE_DEVICES set (comma-separated)
echo "Test 1: CUDA_VISIBLE_DEVICES='0,1,2,3'"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PARALLEL_TESTS=true
DETECTED=$(detect_gpus)
echo "Detected GPUs: $DETECTED"
echo ""

# Test 2: With CUDA_VISIBLE_DEVICES set (space-separated)
echo "Test 2: CUDA_VISIBLE_DEVICES='0 1 2 3'"
export CUDA_VISIBLE_DEVICES="0 1 2 3"
DETECTED=$(detect_gpus)
echo "Detected GPUs: $DETECTED"
echo ""

# Test 3: Single GPU
echo "Test 3: CUDA_VISIBLE_DEVICES='0'"
export CUDA_VISIBLE_DEVICES="0"
DETECTED=$(detect_gpus)
echo "Detected GPUs: $DETECTED"
echo ""

# Test 4: Parallel disabled
echo "Test 4: PARALLEL_TESTS=false"
export PARALLEL_TESTS=false
export CUDA_VISIBLE_DEVICES="0,1,2,3"
DETECTED=$(detect_gpus)
echo "Detected GPUs: $DETECTED (should be just '0' when parallel disabled)"
echo ""

# Test 5: Try nvidia-smi (if available)
echo "Test 5: Using nvidia-smi (CUDA_VISIBLE_DEVICES unset)"
unset CUDA_VISIBLE_DEVICES
export PARALLEL_TESTS=true
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi available"
    DETECTED=$(detect_gpus)
    echo "Detected GPUs: $DETECTED"
else
    echo "nvidia-smi not available (expected on this machine)"
    DETECTED=$(detect_gpus)
    echo "Detected GPUs: $DETECTED (fallback to default)"
fi
echo ""

echo "GPU detection test complete!"
