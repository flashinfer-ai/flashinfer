#!/bin/bash
# Install one layer's worth of Python packages into the manylinux AOT builder
# image. Each stage is its own `RUN` in docker/Dockerfile.aot-builder so the
# multi-GB CUDA wheels land in separate, individually cacheable layers.
#
# Usage: install_aot_builder_packages.sh <stage> <pytorch-index> [cudnn-version]
#   stage: tooling | torch-deps | cuda-libs | cudnn | torch | finalize
#
# Reads /install/.python-version, /install/aot-builder-requirements.txt.
set -euo pipefail

STAGE=${1:?stage is required}
PYTORCH_INDEX=${2:?PyTorch index is required}
CUDNN_VERSION=${3:-}

if [[ ! "${PYTORCH_INDEX}" =~ ^(nightly/)?cu[0-9]+$ ]]; then
  echo "ERROR: invalid PyTorch index path: ${PYTORCH_INDEX}" >&2
  exit 1
fi
CUDA_TAG="${PYTORCH_INDEX##*/}"
CUDA_MAJOR="${CUDA_TAG:2:2}"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_INDEX}"

PYTHON_VERSION="$(tr -d '[:space:]' < /install/.python-version)"
if [[ ! "${PYTHON_VERSION}" =~ ^3\.[0-9]+$ ]]; then
  echo "ERROR: invalid Python version: ${PYTHON_VERSION}" >&2
  exit 1
fi
PYTHON_ABI="cp${PYTHON_VERSION//./}"
PYTHON="/opt/python/${PYTHON_ABI}-${PYTHON_ABI}/bin/python"

export PIP_RETRIES=10 PIP_RESUME_RETRIES=10 PIP_DEFAULT_TIMEOUT=120
export PIP_NO_CACHE_DIR=1

PRE=()
if [[ "${PYTORCH_INDEX}" == nightly/* ]]; then
  PRE=(--pre)
fi

# Resolve torch (without installing it) and print "name==version" for every
# distribution it would install, torch itself included.
resolve_torch_closure() {
  "${PYTHON}" -m pip install --dry-run --ignore-installed --quiet \
    --report - "${PRE[@]}" torch --index-url "${PYTORCH_INDEX_URL}" |
    "${PYTHON}" -c '
import json, sys
report = json.load(sys.stdin)
for item in report["install"]:
    meta = item["metadata"]
    print(f"{meta[\"name\"]}=={meta[\"version\"]}")
'
}

closure_without() {
  # $1: grep -E pattern of distributions to drop; torch is always dropped.
  resolve_torch_closure | grep -Eiv "^(torch|$1)==" || true
}

case "${STAGE}" in
  tooling)
    "${PYTHON}" -m pip install --upgrade -r /install/aot-builder-requirements.txt
    ;;
  torch-deps)
    # Pure-Python / small dependencies of torch (sympy, networkx, jinja2, ...).
    mapfile -t PKGS < <(closure_without 'nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+|triton')
    "${PYTHON}" -m pip install --no-deps "${PKGS[@]}" --index-url "${PYTORCH_INDEX_URL}" "${PRE[@]}"
    ;;
  cuda-libs)
    # The large CUDA runtime wheels torch pins (cuBLAS, cuSOLVER, NCCL, NVSHMEM,
    # ...) except cuDNN, which has its own layer, plus triton.
    mapfile -t PKGS < <(resolve_torch_closure | grep -Ei '^(nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+|triton)==' | grep -Eiv '^nvidia-cudnn' || true)
    if (( ${#PKGS[@]} > 0 )); then
      "${PYTHON}" -m pip install --no-deps "${PKGS[@]}" --index-url "${PYTORCH_INDEX_URL}" "${PRE[@]}"
    fi
    ;;
  cudnn)
    : "${CUDNN_VERSION:?cuDNN version is required}"
    if [[ ! "${CUDNN_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "ERROR: invalid cuDNN package version: ${CUDNN_VERSION}" >&2
      exit 1
    fi
    # Same override docker/install/install_python_packages.sh applies to the CI
    # image: torch's cuDNN pin lags the version FlashInfer uses.
    if [[ "${CUDA_MAJOR}" == "13" ]]; then
      CUDNN_PACKAGE=nvidia-cudnn-cu13
    else
      CUDNN_PACKAGE=nvidia-cudnn-cu12
    fi
    "${PYTHON}" -m pip install --no-deps "${CUDNN_PACKAGE}==${CUDNN_VERSION}"
    ;;
  torch)
    TORCH_SPEC=$(resolve_torch_closure | grep -Ei '^torch==' | head -n1)
    "${PYTHON}" -m pip install --no-deps "${PRE[@]}" "${TORCH_SPEC}" \
      --index-url "${PYTORCH_INDEX_URL}"
    ;;
  finalize)
    mkdir -p /opt/flashinfer-aot-builder
    "${PYTHON}" -m pip freeze --all > /opt/flashinfer-aot-builder/freeze.txt
    "${PYTHON}" - "${PYTORCH_INDEX}" <<'PY'
import importlib.metadata as m
import json
import sys

import torch

manifest = {
    "pytorch_index": sys.argv[1],
    "torch_cuda": torch.version.cuda,
    "torch": m.version("torch"),
    "apache-tvm-ffi": m.version("apache-tvm-ffi"),
    "pip": m.version("pip"),
}
with open("/opt/flashinfer-aot-builder/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
    ;;
  *)
    echo "ERROR: unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
