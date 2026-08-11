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

CUDA_TAG="${PYTORCH_INDEX##*/}"  # nightly/cu134 -> cu134
CUDA_MAJOR="${CUDA_TAG:2:2}"     # cu129 -> 12, cu134 -> 13
CUDA_MINOR="${CUDA_TAG:4}"       # cu129 -> 9, cu134 -> 4
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_INDEX}"

# Install torch with specific CUDA version first, followed by others in requirements.txt, and then others.
# This is to ensure that the torch version is compatible with the CUDA version.
TORCH_INSTALL_ARGS=(--force-reinstall torch --index-url "${PYTORCH_INDEX_URL}")
if [[ "${PYTORCH_INDEX}" == nightly/* ]]; then
  TORCH_INSTALL_ARGS=(--pre "${TORCH_INSTALL_ARGS[@]}")
fi
pip3 install "${TORCH_INSTALL_ARGS[@]}"

# Pin the +cuXXX torch: it is unpinned in requirements.txt, so a dependency
# conflict lets pip swap it for the PyPI build of a different CUDA major.
# Pin the dist version: the PyPI wheel installs as "2.13.0", not "2.13.0+cu130".
# mktemp, not /install: the .dev images run this as a non-root USER.
TORCH_CONSTRAINT="$(mktemp)"
python3 -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' \
  > "$TORCH_CONSTRAINT"
export PIP_CONSTRAINT="$TORCH_CONSTRAINT"

# Pick the CUDA-major-matched packages: nvshmem4py-cuXX pins cuda-python to its
# own major, so mixing them leaves a broken dependency graph. cuda-python is
# installed before requirements.txt (floored >=12.0, else pip takes CUDA-13 and
# drags torch along) and again after, since nvshmem4py can pull it back down.
if [[ "${CUDA_TAG}" == cu13* ]]; then
  CUDA_PYTHON="cuda-python==13.0"
  NVSHMEM4PY="nvshmem4py-cu13"
else
  CUDA_PYTHON="cuda-python==12.*"
  NVSHMEM4PY="nvshmem4py-cu12"
fi

pip3 install --upgrade "$CUDA_PYTHON"
pip3 install -r /install/requirements.txt
pip3 install responses pytest scipy build "$NVSHMEM4PY"
pip3 install --upgrade "$CUDA_PYTHON"

# Install cudnn package based on CUDA version
if [[ "${CUDA_TAG}" == cu13* ]]; then
  pip3 install --upgrade nvidia-cudnn-cu13
  pip3 install --upgrade "nvidia-cutlass-dsl[cu13]==4.7.0"
else
  pip3 install --upgrade nvidia-cudnn-cu12
fi

# Fail the build if torch drifts off the requested CUDA release or cuda-python
# drifts off its CUDA major.
python3 -c "
import importlib.metadata as m, sys, torch
torch_cuda = torch.version.cuda or ''
if torch_cuda != '${CUDA_MAJOR}.${CUDA_MINOR}':
    sys.exit(f'ERROR: torch targets CUDA {torch_cuda}, expected ${CUDA_MAJOR}.${CUDA_MINOR}')
cuda_python = m.version('cuda-python')
if cuda_python.split('.')[0] != '${CUDA_MAJOR}':
    sys.exit(f'ERROR: cuda-python targets CUDA {cuda_python}, expected CUDA ${CUDA_MAJOR}')
print('CUDA ${CUDA_MAJOR}.${CUDA_MINOR} check passed:', torch.__version__)
"
