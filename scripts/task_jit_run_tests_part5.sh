#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -e . -v
fi

# Run each test file separately to isolate CUDA memory issues
pytest -s tests/utils/test_logits_processor.py
pytest -s tests/cli/test_cli_cmds.py
pytest -s tests/cli/test_cli_cmds_gpu.py
