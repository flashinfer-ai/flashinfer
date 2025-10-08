#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -e . -v
fi

# Run each test file separately to isolate CUDA memory issues
pytest -s tests/attention/test_logits_cap.py
pytest -s tests/attention/test_sliding_window.py
pytest -s tests/attention/test_tensor_cores_decode.py
pytest -s tests/attention/test_batch_decode_kernels.py
# pytest -s tests/gemm/test_group_gemm.py
# pytest -s tests/attention/test_alibi.py
