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

pip3 install torch --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip3 install requests ninja pytest numpy scipy build nvidia-ml-py cuda-python einops nvidia-nvshmem-cu12
pip3 install 'apache-tvm-ffi>=0.1.0b6'
pip3 install nvidia-cutlass-dsl
pip3 install 'nvidia-cudnn-frontend>=1.13.0'

# Install cudnn package based on CUDA version
if [[ "$CUDA_VERSION" == *"cu13"* ]]; then
    CUDNN_PACKAGE="nvidia-cudnn-cu13>=9.12.0.46"
else
    CUDNN_PACKAGE="nvidia-cudnn-cu12>=9.11.0.98"
fi

if [[ -n "$CUDNN_PACKAGE" ]]; then
    pip3 install $CUDNN_PACKAGE
fi
