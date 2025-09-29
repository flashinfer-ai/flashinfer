#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

# pytest -s tests/GEMM/test_group_gemm.py
pytest -s tests/attention/test_logits_cap.py
pytest -s tests/attention/test_sliding_window.py
pytest -s tests/attention/test_tensor_cores_decode.py
pytest -s tests/attention/test_batch_decode_kernels.py
# pytest -s tests/attention/test_alibi.py
