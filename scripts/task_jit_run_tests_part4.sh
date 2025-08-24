#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoid memory fragmentation
pytest -s tests/test_deepseek_mla.py
pytest -s tests/test_group_gemm.py
# NOTE(Zihao): need to fix tile size on KV dimension for head_dim=256 on small shared memory architecture (sm89)
# pytest -s tests/test_batch_attention.py
