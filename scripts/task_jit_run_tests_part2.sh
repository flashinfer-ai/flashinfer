#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

pytest -s tests/test_block_sparse.py
pytest -s tests/test_jit_example.py
pytest -s tests/test_jit_warmup.py
pytest -s tests/test_norm.py
pytest -s tests/test_rope.py
pytest -s tests/test_quantization.py
pytest -s tests/test_sampling.py
