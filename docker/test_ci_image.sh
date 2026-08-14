#!/usr/bin/env bash

set -euo pipefail

EXPECTED_CUDA_VERSION=${1:?expected CUDA version is required}
EXPECTED_DOCKER_ARCH=${2:?expected Docker architecture is required}

case "${EXPECTED_DOCKER_ARCH}" in
  amd64)
    EXPECTED_MACHINE=x86_64
    ;;
  arm64)
    EXPECTED_MACHINE=aarch64
    ;;
  *)
    echo "ERROR: unsupported Docker architecture: ${EXPECTED_DOCKER_ARCH}" >&2
    exit 1
    ;;
esac

ACTUAL_MACHINE=$(uname -m)
if [ "${ACTUAL_MACHINE}" != "${EXPECTED_MACHINE}" ]; then
  echo "ERROR: image architecture is ${ACTUAL_MACHINE}; expected ${EXPECTED_MACHINE}" >&2
  exit 1
fi

NVCC_CUDA_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)
if [ "${NVCC_CUDA_VERSION}" != "${EXPECTED_CUDA_VERSION}" ]; then
  echo "ERROR: nvcc reports CUDA ${NVCC_CUDA_VERSION:-unknown}; expected ${EXPECTED_CUDA_VERSION}" >&2
  exit 1
fi

command -v ptxas >/dev/null
command -v mpirun >/dev/null

python3 - "${EXPECTED_CUDA_VERSION}" "${EXPECTED_MACHINE}" <<'PY'
import importlib.metadata
import os
import platform
import subprocess
import sys

expected_cuda = sys.argv[1]
expected_machine = sys.argv[2]
expected_cudnn = os.environ["FLASHINFER_CUDNN_VERSION"]
cuda_major = expected_cuda.split(".")[0]
cudnn_package = f"nvidia-cudnn-cu{cuda_major}"

import cudnn  # noqa: F401
import cutlass  # noqa: F401
import torch
import tvm_ffi  # noqa: F401
from cuda.bindings import runtime as cuda_runtime  # noqa: F401

if platform.machine() != expected_machine:
    raise SystemExit(
        f"ERROR: Python reports architecture {platform.machine()}; "
        f"expected {expected_machine}"
    )
if torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"ERROR: PyTorch targets CUDA {torch.version.cuda}; expected {expected_cuda}"
    )
actual_cuda_python = importlib.metadata.version("cuda-python")
if actual_cuda_python.split(".")[:2] != expected_cuda.split(".")[:2]:
    raise SystemExit(
        f"ERROR: cuda-python targets CUDA {actual_cuda_python}; expected {expected_cuda}"
    )
actual_cudnn = importlib.metadata.version(cudnn_package)
if actual_cudnn != expected_cudnn:
    raise SystemExit(
        f"ERROR: {cudnn_package} is {actual_cudnn}; expected {expected_cudnn}"
    )
cudnn_parts = [int(part) for part in expected_cudnn.split(".")[:3]]
expected_backend = cudnn_parts[0] * 10000 + cudnn_parts[1] * 100 + cudnn_parts[2]
if cudnn.backend_version() != expected_backend:
    raise SystemExit(
        f"ERROR: cuDNN backend is {cudnn.backend_version()}; "
        f"expected {expected_backend}"
    )

distributions = (
    "apache-tvm-ffi",
    "cuda-python",
    cudnn_package,
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "torch",
)
for distribution in distributions:
    print(f"{distribution}=={importlib.metadata.version(distribution)}")
print(f"architecture={platform.machine()}")
print(f"cuda={torch.version.cuda}")

# FlashInfer intentionally needs a newer backend than the exact cuDNN version
# in the stable PyTorch wheel metadata. Reject every dependency error except
# that one checked-and-documented override.
check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if check.returncode == 0:
    print(check.stdout.strip())
else:
    unexpected = []
    allowed = []
    for line in filter(None, check.stdout.splitlines()):
        is_torch_cudnn_override = (
            line.startswith("torch ")
            and f"has requirement {cudnn_package}==" in line
            and f"but you have {cudnn_package} {expected_cudnn}" in line
        )
        (allowed if is_torch_cudnn_override else unexpected).append(line)
    if unexpected or not allowed:
        raise SystemExit(
            "ERROR: unexpected pip dependency errors:\n" + "\n".join(unexpected)
        )
    for line in allowed:
        print(f"Allowed FlashInfer cuDNN override: {line}")
PY

echo "Candidate CI image passed: CUDA ${EXPECTED_CUDA_VERSION}, ${EXPECTED_MACHINE}"
