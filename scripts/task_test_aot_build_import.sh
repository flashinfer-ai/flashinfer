#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=""}
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0+PTX 10.0 12.0"

python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
python -m flashinfer.aot --add-comm True --add-moe True
python -m build --wheel
pip install dist/*.whl

# test import
mkdir -p tmp
cd tmp
python -c "from flashinfer.page import gen_page_module; p = gen_page_module().aot_path; print(p); assert p.exists();"
