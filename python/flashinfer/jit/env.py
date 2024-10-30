"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pathlib
import re

from torch.utils.cpp_extension import _get_cuda_arch_flags


def _get_workspace_dir_name() -> pathlib.Path:
    flags = _get_cuda_arch_flags()
    arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
    # e.g.: $HOME/.cache/flashinfer/75_80_89_90/
    return pathlib.Path.home() / ".cache" / "flashinfer" / arch


# use pathlib
FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"
_package_root = pathlib.Path(__file__).resolve().parents[1]
FLASHINFER_INCLUDE_DIR = _package_root / "data" / "include"
FLASHINFER_CSRC_DIR = _package_root / "data" / "csrc"
CUTLASS_INCLUDE_DIRS = [
    _package_root / "data" / "cutlass" / "include",
    _package_root / "data" / "cutlass" / "tools" / "util" / "include",
]
