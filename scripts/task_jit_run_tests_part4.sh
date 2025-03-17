#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}  # avoid memory fragmentation

pip install -e . -v

pytest -s tests/test_deepseek_mla.py
