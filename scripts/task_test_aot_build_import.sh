#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=""}
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0+PTX"
export FLASHINFER_ENABLE_AOT=1

python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
python -m build --wheel
pip install dist/*.whl

# test import
mkdir -p tmp
cd tmp
python -c "import flashinfer.flashinfer_kernels"
python -c "import flashinfer.flashinfer_kernels_sm90"
python -c "import flashinfer"
