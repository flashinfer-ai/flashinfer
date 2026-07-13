#!/bin/bash
# Real-path tests for the optional MagiAttention FFA adapter (flashinfer/magi_ffa).
# Runs inside the FlashInfer CI image via ci/bash.sh (see
# .github/workflows/magi-ffa-optin-test.yml).

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

# Source test environment setup (handles package overrides like TVM-FFI)
source "$(dirname "${BASH_SOURCE[0]}")/setup_test_env.sh"

pip install -e . -v

# MagiAttention is an OPTIONAL dependency; pinned to the version the adapter
# was validated against. Revalidate the nvidia-cutlass-dsl conflict below
# whenever this pin is bumped.
pip install magi_attention==1.1.0.post10

# KNOWN DEPENDENCY CONFLICT: MagiAttention's install can downgrade
# nvidia-cutlass-dsl below FlashInfer's requirement (>=4.5.0,
# see requirements.txt), which breaks `import flashinfer`
# (cute.nvgpu.OperandMajorMode). Restore it AFTER installing MagiAttention.
# The adapter also enforces this at runtime
# (flashinfer/magi_ffa/_flex_flash_attn.py).
pip install "nvidia-cutlass-dsl>=4.5.0"

python -c "import flashinfer, magi_attention; from magi_attention.api import flex_flash_attn_func; print('imports OK')"

FLASHINFER_TEST_MAGI_FFA_EXTENDED=1 pytest -s tests/ffa
