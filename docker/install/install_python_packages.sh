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

# Accept CUDA version as parameter (e.g., cu126, cu130, cu132).
CUDA_VERSION=${1:-cu126}

is_cuda13() {
  local cuda_version="${1#nightly/}"
  [[ "${cuda_version}" == cu13* || "${cuda_version}" == 13.* ]]
}

torch_index_cuda_version() {
  local cuda_version="${1}"
  local cuda_version_no_stream="${cuda_version#nightly/}"
  if [[ "${cuda_version_no_stream}" =~ ^([0-9]+)\.([0-9]+) ]]; then
    cuda_version_no_stream="cu${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    if [[ "${cuda_version}" == nightly/* ]]; then
      echo "nightly/${cuda_version_no_stream}"
    else
      echo "${cuda_version_no_stream}"
    fi
    return
  fi
  echo "${cuda_version}"
}

# Install torch with specific CUDA version first, followed by others in requirements.txt, and then others.
# This is to ensure that the torch version is compatible with the CUDA version.
TORCH_CUDA_VERSION=$(torch_index_cuda_version "${CUDA_VERSION}")
pip3 install --force-reinstall torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}
if is_cuda13 "${CUDA_VERSION}"; then
  pip3 install -r /install/requirements-cu13.txt
else
  pip3 install -r /install/requirements.txt
fi
pip3 install responses pytest scipy build cuda-python nvshmem4py-cu12

# Install cudnn package based on CUDA version
if is_cuda13 "${CUDA_VERSION}"; then
  pip3 install --upgrade cuda-python==13.0
  pip3 install --upgrade nvidia-cudnn-cu13
else
  pip3 install --upgrade cuda-python==12.*
  pip3 install --upgrade nvidia-cudnn-cu12
fi
