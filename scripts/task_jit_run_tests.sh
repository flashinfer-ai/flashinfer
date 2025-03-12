#!/bin/bash

set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

pytest -s tests/test_norm.py
pytest -s tests/test_sampling.py
pytest -s tests/test_jit_example.py
pytest -s tests/test_jit_warmup.py
