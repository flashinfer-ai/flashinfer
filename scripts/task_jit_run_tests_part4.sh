#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoid memory fragmentation
pytest -s tests/test_deepseek_mla.py
pytest -s tests/test_group_gemm.py
