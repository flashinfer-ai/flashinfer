#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the moe_ep smoke + multi-rank harness via torchrun.
#
# Env:
#   NPROC          GPUs on this node (default: 8; use 4 on a GB200 compute tray)
#   BACKEND        nccl_ep / nixl_ep / both (default: both)
#   PYTEST_EXTRA   extra args passed to pytest

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${NPROC:=8}"
: "${BACKEND:=both}"

cd "${REPO_ROOT}"

case "${BACKEND}" in
    both) BACKENDS=(nccl_ep nixl_ep) ;;
    nccl_ep|nixl_ep) BACKENDS=("${BACKEND}") ;;
    *) echo "Unknown BACKEND: ${BACKEND}" >&2; exit 1 ;;
esac

# Multi-rank correctness — torchrun-launched, one process group, one backend
# per invocation (pytest collects fixtures lazily; --backend selects).
# Uses the gpu_4 marker (a GB200 compute tray = 4 B200; relaxed from gpu_8).
RAN_BACKEND=0
for BE in "${BACKENDS[@]}"; do
    if ! python - "${BE}" <<'PY'
import sys
from flashinfer.moe_ep import available_backends

raise SystemExit(0 if sys.argv[1] in available_backends() else 1)
PY
    then
        if [[ "${BACKEND}" != "both" ]]; then
            echo "Requested backend ${BE} is not available" >&2
            exit 1
        fi
        echo "=== ${BE} not available; skipping ==="
        continue
    fi
    RAN_BACKEND=1

    echo "=== smoke_${BE} (nproc=${NPROC}) ==="
    torchrun --nproc_per_node="${NPROC}" "tests/moe_ep/smoke_${BE}.py"

    TEST_FILES=(
        tests/moe_ep/test_moe_ep_layer_multirank.py
        tests/moe_ep/test_split_layer_cudagraph_multirank.py
        # arch_blackwell gates the CuTe-DSL W4A16 kernel in pytest.
        tests/moe_ep/test_split_layer_w4a16_cudagraph_multirank.py
    )
    if [[ "${BE}" == "nccl_ep" ]]; then
        TEST_FILES+=(
            tests/moe_ep/test_moe_ep_cudagraph_multirank.py
            tests/moe_ep/test_nccl_ep_wrap_memo_multirank.py
        )
    fi
    echo "=== multi-rank ${BE} (nproc=${NPROC}) ==="
    torchrun --nproc_per_node="${NPROC}" \
        -m pytest "${TEST_FILES[@]}" \
        -v -m "nvep and gpu_4" --backend="${BE}" ${PYTEST_EXTRA:-}
done

if [[ "${RAN_BACKEND}" == "0" ]]; then
    echo "No MoE EP backend is available" >&2
    exit 1
fi

echo "=== all moe_ep smoke + multirank tests passed ==="
