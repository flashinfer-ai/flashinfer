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

# Accept CUDA version as parameter (e.g., cu126, cu128, cu129)
CUDA_VERSION=${1:-cu128}

# Install torch with specific CUDA version first, followed by others in requirements.txt, and then others.
# This is to ensure that the torch version is compatible with the CUDA version.
pip3 install --force-reinstall torch --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip3 install -r /install/requirements.txt
pip3 install responses pytest scipy build cuda-python nvidia-nvshmem-cu12

# Install cudnn package based on CUDA version
if [[ "$CUDA_VERSION" == *"cu13"* ]]; then
  pip3 install --upgrade cuda-python==13.0
  pip3 install "nvidia-cudnn-cu13>=9.14.0.64"
else
  pip3 install --upgrade cuda-python==12.*
  pip3 install "nvidia-cudnn-cu12>=9.14.0.64"
fi
