#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/jit_test_logging.sh"

if [ "$SKIP_INSTALL" = "0" ]; then
  install_flashinfer_editable
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Separate processes release CUDA memory between files.
pytest -s tests/attention/test_batch_prefill_kernels.py -k 'nvfp4 and large_head_bf16'
pytest -s tests/utils/test_logits_processor.py
pytest -s tests/attention/test_block_sparse.py
pytest -s tests/cli/test_cli_cmds.py
pytest -s tests/cli/test_cli_cmds_gpu.py
pytest -s tests/moe/test_bgmv_moe.py
pytest -s tests/moe/test_bgmv_moe_lora_delta.py
pytest -s tests/test_artifacts.py
# Cover the API gate without enrolling experimental backend tests.
pytest -s tests/experimental/test_experimental_api.py
