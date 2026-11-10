#!/bin/bash
# Resolve the DSL's matching libraries even when other CI dependencies use --no-deps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
requirements=$(python - "${REPO_ROOT}" <<'PYTHON'
import sys
from importlib.metadata import PackageNotFoundError, version

import torch

sys.path.insert(0, sys.argv[1])
from build_utils import get_cuda_extra_dependency_requirements

cuda_major = torch.version.cuda.split(".")[0]
# The DSL loader can select an installed cu13 extension even with CUDA 12
# PyTorch. Preserve that extra so pip also upgrades its matching library.
if cuda_major == "12":
    try:
        version("nvidia-cutlass-dsl-libs-cu13")
    except PackageNotFoundError:
        pass
    else:
        cuda_major = "13"
for requirement in get_cuda_extra_dependency_requirements(cuda_major):
    if requirement.startswith("nvidia-cutlass-dsl"):
        print(requirement)
PYTHON
)
readarray -t dsl_requirements <<< "${requirements}"
# Do not use --no-deps: updating just the metadata wheel leaves libs-base,
# libs-core and the CUDA libraries at the old image version. Without --upgrade,
# pip retains an already supported newer DSL and repairs its matching libraries.
python -m pip install "${dsl_requirements[@]}"
# A successful pip resolve alone does not prove the Python/native stack agrees.
python -c 'import cutlass; print("CuTe DSL import verified:", cutlass.__version__)'
