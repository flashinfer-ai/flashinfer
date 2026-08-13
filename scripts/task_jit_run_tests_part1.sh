#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

# Source the pre-install guards and optional dependency overrides.
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -e . -v
  # shellcheck disable=SC1091
  source "$(dirname "${BASH_SOURCE[0]}")/setup_ci_test_env.sh"
fi

# Run each test file separately to isolate CUDA memory issues
# moe_ep unit subset: host-only + single-GPU (multirank/mega auto-skip via
# markers; see tests/moe_ep/run_tests.sh and docs/design_docs/moe_ep_runbook.md)
bash tests/moe_ep/run_tests.sh unit
pytest -s tests/attention/test_logits_cap.py
pytest -s tests/attention/test_sliding_window.py
pytest -s tests/attention/test_tensor_cores_decode.py
pytest -s tests/attention/test_batch_decode_kernels.py
# pytest -s tests/gemm/test_group_gemm.py
# pytest -s tests/attention/test_alibi.py
