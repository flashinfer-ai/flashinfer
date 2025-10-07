#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -e . -v
fi

# Run all tests in a single pytest session for better coverage reporting
pytest -s \
  tests/utils/test_block_sparse.py \
  tests/utils/test_jit_example.py \
  tests/utils/test_jit_warmup.py \
  tests/utils/test_norm.py \
  tests/attention/test_rope.py \
  tests/attention/test_mla_page.py \
  tests/utils/test_quantization.py
