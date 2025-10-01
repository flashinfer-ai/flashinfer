#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

pytest -s tests/utils/test_block_sparse.py
pytest -s tests/utils/test_jit_example.py
pytest -s tests/utils/test_jit_warmup.py
pytest -s tests/utils/test_norm.py
pytest -s tests/attention/test_rope.py
pytest -s tests/attention/test_mla_page.py
pytest -s tests/utils/test_quantization.py
