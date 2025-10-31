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
pytest -s tests/utils/test_block_sparse.py
pytest -s tests/utils/test_jit_example.py
pytest -s tests/utils/test_jit_warmup.py
pytest -s tests/utils/test_norm.py
pytest -s tests/attention/test_rope.py
pytest -s tests/attention/test_mla_page.py
pytest -s tests/utils/test_quantization.py
