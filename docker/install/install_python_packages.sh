#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u

pip3 install --upgrade "setuptools>=77" "pip>=24"

# Accept a PyTorch wheel index path (e.g., cu129 or nightly/cu134).
PYTORCH_INDEX=${1:-cu129}
if [[ ! "${PYTORCH_INDEX}" =~ ^(nightly/)?cu[0-9]+$ ]]; then
  echo "ERROR: invalid PyTorch index path: ${PYTORCH_INDEX}" >&2
  exit 1
fi
CUDNN_VERSION=${2:?cuDNN package version is required}
if [[ ! "${CUDNN_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: invalid cuDNN package version: ${CUDNN_VERSION}" >&2
  exit 1
fi

CUDA_TAG="${PYTORCH_INDEX##*/}"  # nightly/cu134 -> cu134
CUDA_MAJOR="${CUDA_TAG:2:2}"     # cu129 -> 12, cu134 -> 13
CUDA_MINOR="${CUDA_TAG:4}"       # cu129 -> 9, cu134 -> 4
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_INDEX}"
BUILD_DEPENDENCY_OUTPUT="$(
  PYTHONPATH=/install python3 -c \
    'import sys; from build_utils import get_build_dependency_requirements; print(*get_build_dependency_requirements(sys.argv[1]), sep="\n")' \
    "${CUDA_MAJOR}"
)"
BUILD_DEPENDENCIES=()
if [[ -n "${BUILD_DEPENDENCY_OUTPUT}" ]]; then
  mapfile -t BUILD_DEPENDENCIES <<< "${BUILD_DEPENDENCY_OUTPUT}"
fi

# Install torch with specific CUDA version first, followed by others in requirements.txt, and then others.
# This is to ensure that the torch version is compatible with the CUDA version.
TORCH_INSTALL_ARGS=(--force-reinstall torch --index-url "${PYTORCH_INDEX_URL}")
if [[ "${PYTORCH_INDEX}" == nightly/* ]]; then
  TORCH_INSTALL_ARGS=(--pre "${TORCH_INSTALL_ARGS[@]}")
fi
pip3 install "${TORCH_INSTALL_ARGS[@]}"

# Pin the +cuXXX torch: it is unpinned in requirements.txt, so a dependency
# conflict lets pip swap it for the PyPI build of a different CUDA major.
# Strip the local +cuXXX tag so the constraint remains resolvable from PyPI.
# mktemp, not /install: the .dev images run this as a non-root USER.
TORCH_CONSTRAINT="$(mktemp)"
python3 -c 'import importlib.metadata as m; print("torch==" + m.version("torch").split("+", 1)[0])' \
  > "$TORCH_CONSTRAINT"
export PIP_CONSTRAINT="$TORCH_CONSTRAINT"

# Resolve the remaining direct dependencies once. Match cuda-python to the
# image's CUDA major/minor; this also satisfies the matching nvshmem4py range.
CUDA_PYTHON="cuda-python==${CUDA_MAJOR}.${CUDA_MINOR}"
if [[ "${CUDA_TAG}" == cu13* ]]; then
  NVSHMEM4PY="nvshmem4py-cu13"
  CUDNN_PACKAGE="nvidia-cudnn-cu13"
else
  # nvshmem4py-cu12 declares <=12.9. PEP 440 treats 12.9.7 as newer than
  # 12.9, so the former ==12.* constraint did not actually satisfy it. The
  # exact ==12.9 constraint resolves to the compatible 12.9.0 release.
  NVSHMEM4PY="nvshmem4py-cu12"
  CUDNN_PACKAGE="nvidia-cudnn-cu12"
fi

# wheel is imported by flashinfer-jit-cache's build backend, which CI builds
# without isolation. Resolve the remaining image dependencies together once.
pip3 install \
  -r /install/requirements.txt \
  responses pytest scipy build wheel \
  "${CUDA_PYTHON}" \
  "${NVSHMEM4PY}" \
  "${BUILD_DEPENDENCIES[@]}"

# Torch 2.13's cu129/cu130 wheels exact-pin cuDNN 9.20, but current FlashInfer
# uses cuDNN 9.21-9.24 APIs and 9.20 has a known incomplete sublibrary set.
# Override only the backend package, last, so no later resolver pass undoes it.
# docker/test_ci_image.py verifies the installed package and loaded backend.
pip3 install --upgrade --no-deps "${CUDNN_PACKAGE}==${CUDNN_VERSION}"

# Fail the build if torch or cuda-python drifts off the requested CUDA release.
python3 -c "
import importlib.metadata as m, sys, torch
import cudnn
torch_cuda = torch.version.cuda or ''
if torch_cuda != '${CUDA_MAJOR}.${CUDA_MINOR}':
    sys.exit(f'ERROR: torch targets CUDA {torch_cuda}, expected ${CUDA_MAJOR}.${CUDA_MINOR}')
cuda_python = m.version('cuda-python')
if cuda_python.split('.')[:2] != ['${CUDA_MAJOR}', '${CUDA_MINOR}']:
    sys.exit(f'ERROR: cuda-python targets CUDA {cuda_python}, expected ${CUDA_MAJOR}.${CUDA_MINOR}')
cudnn_package_version = m.version('${CUDNN_PACKAGE}')
if cudnn_package_version != '${CUDNN_VERSION}':
    sys.exit(f'ERROR: ${CUDNN_PACKAGE} is {cudnn_package_version}, expected ${CUDNN_VERSION}')
cudnn_parts = [int(part) for part in cudnn_package_version.split('.')[:3]]
expected_backend = cudnn_parts[0] * 10000 + cudnn_parts[1] * 100 + cudnn_parts[2]
if cudnn.backend_version() != expected_backend:
    sys.exit(f'ERROR: cuDNN backend is {cudnn.backend_version()}, expected {expected_backend}')
print('CUDA ${CUDA_MAJOR}.${CUDA_MINOR} check passed:', torch.__version__)
print('${CUDNN_PACKAGE} check passed:', cudnn_package_version)
"
