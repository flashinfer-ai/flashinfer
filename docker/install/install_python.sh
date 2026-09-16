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
set -o pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <conda-root> <environment-name> <python-version-file>" >&2
    exit 2
fi

CONDA_ROOT="$1"
CONDA_ENV_NAME="$2"
PYTHON_VERSION_FILE="$3"
PYTHON_VERSION="$(tr -d '[:space:]' < "$PYTHON_VERSION_FILE")"

if [[ ! "$PYTHON_VERSION" =~ ^3\.[0-9]+$ ]]; then
    echo "Invalid Python version in $PYTHON_VERSION_FILE: $PYTHON_VERSION" >&2
    exit 2
fi

# Install python and pip. Don't modify this to add Python package dependencies,
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$CONDA_ROOT"

"$CONDA_ROOT/bin/conda" create -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"

# Give Dockerfiles a stable path for Python-installed NVIDIA libraries so a
# Python minor-version bump only requires changing .python-version.
CONDA_ENV_PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME"
PYTHON_SITE_PACKAGES="$(
    "$CONDA_ENV_PATH/bin/python" -c \
        'import sysconfig; print(sysconfig.get_path("purelib"))'
)"
ln -s "$PYTHON_SITE_PACKAGES" "$CONDA_ENV_PATH/python-site-packages"
