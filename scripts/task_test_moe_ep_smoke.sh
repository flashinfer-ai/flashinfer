#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the moe_ep smoke + multi-rank harness via torchrun.
#
# Env:
#   NPROC          GPUs per node (default 8 on Lyris GB200 = 2 nodes × 4)
#   BACKEND        nccl_ep / nixl_ep / both (default: both)
#   PYTEST_EXTRA   extra args passed to pytest

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${NPROC:=8}"
: "${BACKEND:=both}"

cd "${REPO_ROOT}"

# Smoke each backend first (single-rank-per-node OK as a probe).
if [[ "${BACKEND}" == "nccl_ep" || "${BACKEND}" == "both" ]]; then
    echo "=== smoke_nccl_ep (nproc=${NPROC}) ==="
    torchrun --nproc_per_node="${NPROC}" \
        tests/moe_ep/smoke_nccl_ep.py
fi

if [[ "${BACKEND}" == "nixl_ep" || "${BACKEND}" == "both" ]]; then
    echo "=== smoke_nixl_ep (nproc=${NPROC}) ==="
    torchrun --nproc_per_node="${NPROC}" \
        tests/moe_ep/smoke_nixl_ep.py
fi

# Multi-rank correctness — torchrun-launched, one process group, one backend
# per invocation (pytest collects fixtures lazily; --backend selects).
# Uses the gpu_4 marker (a GB200 compute tray = 4 B200; relaxed from gpu_8).
for BE in $( [[ "${BACKEND}" == "both" ]] && echo "nccl_ep nixl_ep" || echo "${BACKEND}" ); do
    echo "=== multi-rank ${BE} (nproc=${NPROC}) ==="
    torchrun --nproc_per_node="${NPROC}" \
        -m pytest tests/moe_ep/test_moe_ep_layer_multirank.py \
        -v -m "nvep and gpu_4" --backend="${BE}" ${PYTEST_EXTRA:-}
done

echo "=== all moe_ep smoke + multirank tests passed ==="
