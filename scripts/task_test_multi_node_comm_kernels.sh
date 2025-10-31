#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

pytest -s tests/comm/test_mnnvl_memory.py
pytest -s tests/comm/test_trtllm_mnnvl_allreduce.py
