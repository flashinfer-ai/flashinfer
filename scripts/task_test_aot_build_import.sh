#!/bin/bash

set -eo pipefail
set -x

# Set MAX_JOBS based on architecture and available memory
ARCH=$(uname -m)
MEM_AVAILABLE_GB=$(free -g | awk '/^Mem:/ {print $7}')
NPROC=$(nproc)

# Calculate base MAX_JOBS based on memory (4GB per job)
BASE_MAX_JOBS=$(( MEM_AVAILABLE_GB / 4 ))
if (( BASE_MAX_JOBS < 1 )); then
  BASE_MAX_JOBS=1
elif (( NPROC < BASE_MAX_JOBS )); then
  BASE_MAX_JOBS=$NPROC
fi

# Apply architecture-specific scaling
if [[ "$ARCH" == "aarch64" ]]; then
  # Use half the jobs on aarch64 compared to x86_64
  MAX_JOBS=$(( BASE_MAX_JOBS / 2 ))
  if (( MAX_JOBS < 1 )); then
    MAX_JOBS=1
  fi
else
  # x86_64, amd64, and other architectures use full capacity
  MAX_JOBS=$BASE_MAX_JOBS
fi

# Export MAX_JOBS for PyTorch's cpp_extension to use
export MAX_JOBS

: ${CUDA_VISIBLE_DEVICES:=""}
export FLASHINFER_CUDA_ARCH_LIST=$(python3 -c '
import torch
cuda_ver = torch.version.cuda
arches = ["7.5", "8.0", "8.9", "9.0"]
if cuda_ver is not None:
    try:
        major, minor = map(int, cuda_ver.split(".")[:2])
        if (major, minor) >= (12, 8):
            arches.append("10.0")
            arches.append("12.0")
    except Exception:
        pass
print(" ".join(arches))
')

python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
python -m flashinfer.aot --add-comm True --add-moe True
python -m build --wheel
pip install dist/*.whl

# test import
mkdir -p tmp
cd tmp
python -c "import flashinfer"
