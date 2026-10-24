#!/bin/bash
# Resolve the DSL's matching libraries even when other CI dependencies use --no-deps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
requirements=$(python - "${REPO_ROOT}" <<'PYTHON'
import sys

import torch

sys.path.insert(0, sys.argv[1])
from build_utils import get_cuda_extra_dependency_requirements

cuda_major = torch.version.cuda.split(".")[0]
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
