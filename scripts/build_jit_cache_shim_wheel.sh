#!/bin/bash
# Build the flashinfer-jit-cache shim wheel directly on the runner.
#
# The shim is metadata only: flashinfer-jit-cache/build_backend.py writes
# _build_meta.py and the exact provider requirements, then hands off to
# setuptools, and the backend fixes the platform tag to
# manylinux_2_28_<machine>. It needs no torch, CUDA toolkit or manylinux
# image. Provider builds still run in the builder container
# (build_flashinfer_jit_cache_artifact.sh); running the shim there too cost
# each shim job a ~5 min image pull and a 3-4 GB torch install for a 5 KB
# wheel.
#
# Run it on a Linux runner whose CPU architecture matches the wheel. The build
# requirements go into a throwaway virtual environment, so the runner's Python
# is left as it was.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${FLASHINFER_JIT_CACHE_PROVIDER_ARCHS:?FLASHINFER_JIT_CACHE_PROVIDER_ARCHS must be set}"
: "${FLASHINFER_LOCAL_VERSION:?FLASHINFER_LOCAL_VERSION must be set}"
PYTHON=${PYTHON:-python3}

expected_version=$(tr -d '[:space:]' < "${REPO_ROOT}/version.txt")
if [ -n "${FLASHINFER_DEV_RELEASE_SUFFIX:-}" ]; then
  expected_version="${expected_version}.dev${FLASHINFER_DEV_RELEASE_SUFFIX}"
fi
expected_version="${expected_version}+${FLASHINFER_LOCAL_VERSION}"

echo "Building flashinfer-jit-cache shim ${expected_version}"
echo "Provider architectures: ${FLASHINFER_JIT_CACHE_PROVIDER_ARCHS}"

venv_root=$(mktemp -d)
trap 'rm -rf -- "${venv_root}"' EXIT
"${PYTHON}" -m venv "${venv_root}/venv"
venv_python="${venv_root}/venv/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1
"${venv_python}" -m pip install --quiet "setuptools>=77" wheel build

cd "${REPO_ROOT}/flashinfer-jit-cache"
rm -rf -- dist build ./*.egg-info flashinfer_jit_cache/jit_cache
rm -f -- \
  flashinfer_jit_cache/_build_meta.py \
  flashinfer_jit_cache/_provider_requirements.txt

# --no-isolation: the virtual environment above already is the isolated
# environment, and reusing it saves a second setuptools download.
"${venv_python}" -m build --wheel --no-isolation --outdir dist .

shopt -s nullglob
wheels=(dist/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "Expected exactly one shim wheel in dist/, found ${#wheels[@]}" >&2
  exit 1
fi
if [ -n "${ARCH:-}" ] && [[ "${wheels[0]}" != *"-manylinux_2_28_${ARCH}.whl" ]]; then
  echo "Shim wheel ${wheels[0]} is not tagged manylinux_2_28_${ARCH}" >&2
  exit 1
fi
ls -l dist/

echo "Verifying built package metadata..."
"${venv_python}" -m pip install --quiet --force-reinstall --no-deps "${wheels[0]}"
# Import from outside the source tree so the installed wheel is what is checked.
(cd "${venv_root}" && "${venv_python}" - "${expected_version}" <<'PY'
import sys

import flashinfer_jit_cache

print(f"Package version: {flashinfer_jit_cache.__version__}")
print(f"Git version: {flashinfer_jit_cache.__git_version__}")
if flashinfer_jit_cache.__version__ != sys.argv[1]:
    raise SystemExit(f"Expected shim version {sys.argv[1]}")
PY
)
