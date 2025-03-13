#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

pytest -s tests/test_group_gemm.py
pytest -s tests/test_logits_cap.py
pytest -s tests/test_mla_decode_kernel.py
pytest -s tests/test_sliding_window.py
pytest -s tests/test_tensor_cores_decode.py
pytest -s tests/test_alibi.py
