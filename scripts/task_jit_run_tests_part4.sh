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
pytest -s tests/attention/test_batch_prefill_kernels.py -k 'nvfp4 and not large_head_bf16'
pytest -s tests/attention/test_deepseek_mla.py
pytest -s tests/attention/test_mla_paged_attention_fuzz.py
pytest -s tests/attention/test_rope.py
pytest -s tests/attention/test_mla_page.py
pytest -s tests/utils/test_quantization.py
