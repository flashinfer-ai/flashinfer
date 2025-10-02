#!/bin/bash

set -eo pipefail
set -x

# MAX_JOBS = min(nproc, max(1, MemAvailable_GB/4))
MEM_AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')
NPROC=$(nproc)
MAX_JOBS=$(( MEM_AVAILABLE_GB / 4 ))
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
elif (( NPROC < MAX_JOBS )); then
  MAX_JOBS=$NPROC
fi

# Export MAX_JOBS for PyTorch's cpp_extension to use
export MAX_JOBS

: ${CUDA_VISIBLE_DEVICES:=""}
export FLASHINFER_CUDA_ARCH_LIST=$(python3 -c '
import torch
cuda_ver = torch.version.cuda
arches = ["7.5", "8.0", "8.9", "9.0a"]
if cuda_ver is not None:
    try:
        major, minor = map(int, cuda_ver.split(".")[:2])
        if (major, minor) >= (12, 8):
            arches.append("10.0a")
            arches.append("12.0a")
    except Exception:
        pass
print(" ".join(arches))
')

python -c "import torch; print(torch.__version__)"

# Detect CUDA version from the container
CUDA_VERSION=$(python3 -c 'import torch; print(torch.version.cuda)' | cut -d'.' -f1,2 | tr -d '.')
echo "Detected CUDA version: cu${CUDA_VERSION}"

cd flashinfer-jit-cache
python -m build --wheel

# Get the built wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Built wheel: $WHEEL_FILE"

# Test matrix: Python versions x PyTorch versions
PYTHON_VERSIONS=("3.10" "3.11" "3.12")
TORCH_VERSIONS=("2.7" "2.8")

# Function to test a specific Python + PyTorch combination
test_combination() {
    local python_ver=$1
    local torch_ver=$2
    local env_name="test-flashinfer-py${python_ver}-torch${torch_ver}"

    echo "========================================"
    echo "Testing Python ${python_ver} + PyTorch ${torch_ver}"
    echo "========================================"

    # Create conda environment
    conda create -y -n "$env_name" python="${python_ver}" || {
        echo "Failed to create conda environment for Python ${python_ver}"
        return 1
    }

    # Activate environment and run tests
    eval "$(conda shell.bash hook)"
    conda activate "$env_name" || {
        echo "Failed to activate conda environment $env_name"
        conda env remove -n "$env_name" -y
        return 1
    }

    # Install PyTorch with the detected CUDA version
    echo "Installing PyTorch ${torch_ver} with CUDA ${CUDA_VERSION}..."
    pip install "torch==${torch_ver}.0" --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}" || {
        echo "Failed to install PyTorch ${torch_ver}"
        conda deactivate
        conda env remove -n "$env_name" -y
        return 1
    }

    # Install flashinfer from source
    echo "Installing flashinfer from source..."
    cd ..
    pip install -e . || {
        echo "Failed to install flashinfer from source"
        conda deactivate
        conda env remove -n "$env_name" -y
        return 1
    }

    # Install flashinfer-jit-cache wheel
    echo "Installing flashinfer-jit-cache wheel..."
    pip install "flashinfer-jit-cache/$WHEEL_FILE" || {
        echo "Failed to install flashinfer-jit-cache wheel"
        conda deactivate
        conda env remove -n "$env_name" -y
        return 1
    }

    # Test with show-config
    echo "Running 'python -m flashinfer show-config'..."
    python -m flashinfer show-config || {
        echo "Failed to run 'python -m flashinfer show-config'"
        conda deactivate
        conda env remove -n "$env_name" -y
        return 1
    }

    # Verify all modules are compiled
    echo "Verifying all modules are compiled..."
    python ../scripts/verify_all_modules_compiled.py || {
        echo "Not all modules are compiled!"
        conda deactivate
        conda env remove -n "$env_name" -y
        return 1
    }

    # Clean up
    conda deactivate
    conda env remove -n "$env_name" -y

    echo "✓ Test passed for Python ${python_ver} + PyTorch ${torch_ver}"
    echo ""

    return 0
}

# Run tests for all combinations
FAILED_TESTS=()
for python_ver in "${PYTHON_VERSIONS[@]}"; do
    for torch_ver in "${TORCH_VERSIONS[@]}"; do
        if ! test_combination "$python_ver" "$torch_ver"; then
            FAILED_TESTS+=("Python ${python_ver} + PyTorch ${torch_ver}")
        fi
    done
done

# Report results
echo "========================================"
echo "Test Summary"
echo "========================================"
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    exit 1
fi
