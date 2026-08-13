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
pytest -s tests/utils/test_logits_processor.py
pytest -s tests/cli/test_cli_cmds.py
pytest -s tests/cli/test_cli_cmds_gpu.py
pytest -s tests/moe/test_bgmv_moe.py
pytest -s tests/moe/test_bgmv_moe_lora_delta.py
