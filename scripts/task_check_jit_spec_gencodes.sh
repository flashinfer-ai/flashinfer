#!/bin/bash
# Lint the provider spec CUDA targets for this image's CUDA version, without
# compiling anything: the quick PR check behind the AOT lanes' hours-long builds.
# Runs the same lint the AOT lane starts with, for every provider architecture
# either CPU architecture builds for this CUDA version.

set -eo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# shellcheck source=scripts/setup_test_env.sh
source "${SCRIPT_DIR}/setup_test_env.sh"
# shellcheck source=scripts/jit_cache_build_common.sh
source "${SCRIPT_DIR}/jit_cache_build_common.sh"

CUDA_VERSION=$(python3 -c 'import torch; print(torch.version.cuda)' | cut -d'.' -f1,2 | tr -d '.')
CUDA_LABEL="cu${CUDA_VERSION}"

# Spec generation does not depend on the CPU architecture, but the two can
# build different provider sets, so lint the union.
PROVIDER_ARCHITECTURES=$(python3 - "${CUDA_LABEL}" <<'PY'
import json
import sys
from pathlib import Path

entry = next(
    item
    for item in json.loads(Path("ci/cuda-versions.json").read_text())["jit_cache"]
    if item["label"] == sys.argv[1]
)
architectures = dict.fromkeys(
    entry["x86_64_provider_architectures"] + entry["aarch64_provider_architectures"]
)
print(" ".join(architectures))
PY
)
read -ra PROVIDER_ARCHITECTURE_LIST <<< "${PROVIDER_ARCHITECTURES}"
echo "CUDA: ${CUDA_LABEL}; provider architectures: ${PROVIDER_ARCHITECTURES}"

# Spec generation imports flashinfer, whose build writes only package metadata here.
FLASHINFER_EDITABLE_SPEC="."
if [[ "${CUDA_VERSION}" == 13* ]]; then
    FLASHINFER_EDITABLE_SPEC=".[cu13]"
fi
env FLASHINFER_BUILD_NO_PIP=1 \
    pip install --no-build-isolation --no-deps -e "${FLASHINFER_EDITABLE_SPEC}" -v

: "${SPEC_LINT_CONCURRENCY:=3}"
run_provider_spec_lint "${SPEC_LINT_CONCURRENCY}" "${PROVIDER_ARCHITECTURE_LIST[@]}"
