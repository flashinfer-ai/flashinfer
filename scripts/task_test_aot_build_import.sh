#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=""}

FLASHINFER_ENABLE_AOT=1 python -m build --wheel
pip install dist/*.whl

# test import
python -c "import flashinfer"
