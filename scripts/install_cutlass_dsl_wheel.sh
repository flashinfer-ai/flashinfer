#!/usr/bin/env bash
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Temporary, reproducible Attention-TS compiler package. The wheel and its
# full build provenance are published in this FlashInfer fork's GitLab generic
# package registry rather than checked into Git history.
readonly VERSION="4.7.0+flashinfer.5"
readonly WHEEL="nvidia_cutlass_dsl-4.7.0+flashinfer.5-cp312-cp312-linux_x86_64.whl"
readonly SHA256="f933a7ddb0b2d118dc4f5ab4cb68c906a633be41ddaa1e30f9334eeb40eb44e2"
readonly PACKAGE_ROOT="https://gitlab-master.nvidia.com/api/v4/projects/193706/packages/generic/nvidia-cutlass-dsl/4.7.0%2Bflashinfer.5"
readonly WHEEL_URL="${PACKAGE_ROOT}/nvidia_cutlass_dsl-4.7.0%2Bflashinfer.5-cp312-cp312-linux_x86_64.whl"
readonly PROVENANCE_URL="${PACKAGE_ROOT}/PROVENANCE.md"

PYTHON_BIN=${PYTHON_BIN:-python3}
CACHE_DIR=${CUTLASS_DSL_WHEEL_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/flashinfer/wheels}
WHEEL_PATH="${CACHE_DIR}/${WHEEL}"

command -v curl >/dev/null || {
  echo "curl is required to download the CUTLASS DSL wheel" >&2
  exit 69
}
command -v sha256sum >/dev/null || {
  echo "sha256sum is required to verify the CUTLASS DSL wheel" >&2
  exit 69
}
command -v "${PYTHON_BIN}" >/dev/null || {
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 69
}

mkdir -p "${CACHE_DIR}"
if ! printf '%s  %s\n' "${SHA256}" "${WHEEL_PATH}" | sha256sum --check --status; then
  tmp_path="${WHEEL_PATH}.tmp.$$"
  trap 'rm -f "${tmp_path:-}"' EXIT
  curl_args=(
    --fail
    --location
    --retry 3
    --output "${tmp_path}"
  )
  if [[ -n ${GITLAB_ACCESS_TOKEN:-} ]]; then
    curl_args+=(--header "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}")
  elif [[ -n ${CI_JOB_TOKEN:-} ]]; then
    curl_args+=(--header "JOB-TOKEN: ${CI_JOB_TOKEN}")
  fi
  curl "${curl_args[@]}" "${WHEEL_URL}"
  printf '%s  %s\n' "${SHA256}" "${tmp_path}" | sha256sum --check
  mv "${tmp_path}" "${WHEEL_PATH}"
  trap - EXIT
fi

"${PYTHON_BIN}" -m pip install --upgrade "${WHEEL_PATH}"
"${PYTHON_BIN}" - "${VERSION}" <<'PY'
import importlib.metadata
import pathlib
import sys

import cutlass

expected = sys.argv[1]
actual = importlib.metadata.version("nvidia-cutlass-dsl")
if actual != expected:
    raise SystemExit(f"expected nvidia-cutlass-dsl {expected}, found {actual}")
print(f"installed nvidia-cutlass-dsl {actual} from {pathlib.Path(cutlass.__file__).resolve()}")
PY

echo "build provenance: ${PROVENANCE_URL}"
