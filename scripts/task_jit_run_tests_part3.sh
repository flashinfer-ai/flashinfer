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
pytest -s tests/utils/test_sampling.py
pytest -s tests/utils/test_topk.py
pytest -s tests/attention/test_batch_prefill_kernels.py -k 'not nvfp4'
