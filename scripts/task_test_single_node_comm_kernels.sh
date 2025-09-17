#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

# vllm ar
pytest -s tests/test_vllm_custom_allreduce.py
# trtllm ar + fusion
pytest -s tests/test_trtllm_allreduce.py
pytest -s tests/test_trtllm_allreduce_fusion.py
pytest -s tests/test_trtllm_cutlass_fused_moe.py
pytest -s tests/test_trtllm_moe_allreduce_fusion.py
pytest -s tests/test_trtllm_moe_allreduce_fusion_finalize.py
# nvshmem ar
pytest -s tests/test_nvshmem.py
pytest -s tests/test_nvshmem_allreduce.py
