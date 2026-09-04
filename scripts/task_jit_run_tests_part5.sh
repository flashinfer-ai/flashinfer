#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

if [ "$SKIP_INSTALL" = "0" ]; then
  install_flashinfer_editable
fi

# Run each test file separately to isolate CUDA memory issues
pytest -s tests/utils/test_logits_processor.py
pytest -s tests/cli/test_cli_cmds.py
pytest -s tests/cli/test_cli_cmds_gpu.py
pytest -s tests/moe/test_bgmv_moe.py
pytest -s tests/moe/test_bgmv_moe_lora_delta.py

# tests/experimental/ is excluded from `pytest tests/` by norecursedirs, so it has
# to be named explicitly. Listed here, and not left to the targeted lane, because
# these cover the gating machinery itself -- @flashinfer_experimental_api and
# @experimental_backend live in flashinfer/api_logging.py and flashinfer/utils.py,
# which are stable core. A regression there breaks stable dispatch, so it must fail
# the stable lane rather than wait for someone to type a bot command. Tests of
# experimental *backends* stay with the targeted lane.
pytest -s tests/experimental/
