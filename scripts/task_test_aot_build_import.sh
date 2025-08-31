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

: ${CUDA_VISIBLE_DEVICES:=""}
export TORCH_CUDA_ARCH_LIST=$(python3 -c '
import torch
cuda_ver = torch.version.cuda
arches = ["7.5", "8.0", "8.9", "9.0+PTX"]
if cuda_ver is not None:
    try:
        major, minor = map(int, cuda_ver.split(".")[:2])
        if (major, minor) >= (12, 8):
            arches.append("10.0+PTX")
            arches.append("12.0+PTX")
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
