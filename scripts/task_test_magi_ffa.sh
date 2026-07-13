#!/bin/bash
# Real-path tests for the optional MagiAttention FFA adapter (flashinfer/magi_ffa).
# Runs inside the FlashInfer CI image via ci/bash.sh (see
# .github/workflows/magi-ffa-optin-test.yml).

set -eo pipefail
set -x
: "${MAX_JOBS:=$(nproc)}"
: "${CUDA_VISIBLE_DEVICES:=0}"

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

python -m pip install --no-build-isolation -e . -v

# MagiAttention is an OPTIONAL dependency; pinned to the version the adapter
# was validated against. Revalidate the nvidia-cutlass-dsl conflict below
# whenever this pin is bumped.
python -m pip install magi_attention==1.1.0.post10

# KNOWN DEPENDENCY CONFLICT: MagiAttention's install can downgrade
# nvidia-cutlass-dsl below FlashInfer's requirement (>=4.5.0,
# see requirements.txt), which breaks `import flashinfer`
# (cute.nvgpu.OperandMajorMode). Restore it AFTER installing MagiAttention.
# This is a validated override, not a resolver-clean optional extra.
python -m pip install "nvidia-cutlass-dsl>=4.5.0"

python - <<'PY'
from importlib import import_module
from importlib.metadata import version

print(f"magi_attention={version('magi_attention')}")
print(f"nvidia-cutlass-dsl={version('nvidia-cutlass-dsl')}")

import_module("flashinfer")
magi_api = import_module("magi_attention.api")
getattr(magi_api, "flex_flash_attn_func")

print("FlashInfer and MagiAttention imports OK")
PY

FLASHINFER_TEST_MAGI_FFA_EXTENDED=1 python -m pytest -s tests/ffa
