#!/bin/bash

set -eo pipefail
set -x
: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v
pip install --upgrade nvidia-cudnn-cu12
pip install --upgrade cuda-python==12.*

EXIT_CODE=0

test_scripts=(
    "test_mnnvl_custom_comm.py"
    "test_mnnvl_memory.py"
    "test_nvshmem_allreduce.py"
    "test_trtllm_allreduce_fusion.py"
    "test_trtllm_allreduce.py"
    "test_trtllm_mnnvl_allreduce.py"
    "test_trtllm_moe_allreduce_fusion_finalize.py"
    "test_trtllm_moe_allreduce_fusion.py"
    "test_vllm_custom_allreduce.py"
    "test_trtllm_alltoall.py"
    "test_nvshmem.py"
)

for test_file in "${test_scripts[@]}"; do
  xml_name="${test_file%.py}.xml"
  pytest -s "tests/${test_file}" --junit-xml="${JUNIT_DIR}/$xml_name" || EXIT_CODE=1
done

exit $EXIT_CODE
