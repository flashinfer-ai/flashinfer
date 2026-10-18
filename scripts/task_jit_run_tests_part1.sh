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

# Print the resolved TVM-FFI version right before tests (traceability for the
# pre-release verification in PR #2599; the override itself is applied by
# setup_test_env.sh sourced above).
echo "Checking TVM-FFI version before tests..."
python -c "import tvm_ffi; print(f'TVM-FFI version: {tvm_ffi.__version__}')" || true
echo ""

# Separate processes release CUDA memory between files.
bash tests/moe_ep/run_tests.sh unit
pytest -s tests/attention/test_batch_decode_kernels.py
pytest -s tests/utils/test_jit_example.py
pytest -s tests/gemm/test_group_gemm.py
