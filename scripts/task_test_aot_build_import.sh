#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=""}
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0+PTX"
export FLASHINFER_ENABLE_AOT=1

python -m build --wheel
pip install dist/*.whl

# test import
python -c "import flashinfer"
