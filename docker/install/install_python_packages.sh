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

# Accept CUDA version as parameter (e.g., cu126, cu128, cu129)
CUDA_VERSION=${1:-cu128}
CUDA_TAG="${CUDA_VERSION##*/}"    # cu129 -> cu129, nightly/cu132 -> cu132
CUDA_MAJOR="${CUDA_TAG:2:2}"      # cu129 -> 12, cu132 -> 13

# Install torch with specific CUDA version first, followed by others in requirements.txt, and then others.
# This is to ensure that the torch version is compatible with the CUDA version.
pip3 install --force-reinstall torch --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

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
if [[ "$CUDA_VERSION" == *"cu13"* ]]; then
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
if [[ "$CUDA_VERSION" == *"cu13"* ]]; then
  pip3 install --upgrade nvidia-cudnn-cu13
  pip3 install --upgrade "nvidia-cutlass-dsl[cu13]>=4.5.0"
else
  pip3 install --upgrade nvidia-cudnn-cu12
fi

# Fail the build if torch or cuda-python drifted off this image's CUDA major.
python3 -c "
import importlib.metadata as m, sys, torch
for name, ver in (('torch', torch.version.cuda or ''), ('cuda-python', m.version('cuda-python'))):
    if ver.split('.')[0] != '${CUDA_MAJOR}':
        sys.exit(f'ERROR: {name} targets CUDA {ver}, but this image is CUDA ${CUDA_MAJOR}')
print('CUDA ${CUDA_MAJOR} check passed:', torch.__version__)
"
