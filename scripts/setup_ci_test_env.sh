#!/bin/bash
# Set up dependency versions for source-based CI tests.
#
# This script must be sourced after the project dependencies are installed.
# Published and nightly FlashInfer packages retain the version declared in
# requirements.txt; only source-based CI validation opts into CUTLASS DSL 4.7.

: "${CUTLASS_DSL_CI_VERSION:=${CUTLASS_DSL_VERSION:-4.7.0}}"

CUDA_MAJOR=$(python -c \
  'import torch; version = torch.version.cuda; print(version.split(".")[0] if version else "12")' \
  2>/dev/null || echo 12)
if [ "$CUDA_MAJOR" = "13" ]; then
  CUTLASS_DSL_PACKAGE="nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_CI_VERSION}"
  CUTLASS_DSL_PACKAGES=(
    nvidia-cutlass-dsl
    nvidia-cutlass-dsl-libs-core
    nvidia-cutlass-dsl-libs-base
    nvidia-cutlass-dsl-libs-cu12
    nvidia-cutlass-dsl-libs-cu13
  )
else
  CUTLASS_DSL_PACKAGE="nvidia-cutlass-dsl==${CUTLASS_DSL_CI_VERSION}"
  CUTLASS_DSL_PACKAGES=(
    nvidia-cutlass-dsl
    nvidia-cutlass-dsl-libs-core
    nvidia-cutlass-dsl-libs-base
    nvidia-cutlass-dsl-libs-cu12
  )
fi

pip uninstall -y \
  nvidia-cutlass-dsl \
  nvidia-cutlass-dsl-libs-core \
  nvidia-cutlass-dsl-libs-base \
  nvidia-cutlass-dsl-libs-cu12 \
  nvidia-cutlass-dsl-libs-cu13 2>/dev/null || true
pip install "${CUTLASS_DSL_PACKAGE}"

python - "${CUTLASS_DSL_CI_VERSION}" "${CUTLASS_DSL_PACKAGES[@]}" <<'PY'
import importlib.metadata as metadata
import sys

expected, *packages = sys.argv[1:]
for name in packages:
    version = metadata.version(name)
    if version != expected:
        raise SystemExit(f"ERROR: {name} is {version}, expected {expected}")
print(f"CUTLASS DSL CI check passed: {expected}")
PY
