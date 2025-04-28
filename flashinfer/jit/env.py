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

import os
import pathlib
import re
import warnings

from torch.utils.cpp_extension import _get_cuda_arch_flags

from ..get_include_paths import get_csrc_dir, get_include, get_tvm_binding_dir


def _get_workspace_dir_name() -> pathlib.Path:
    try:
        with warnings.catch_warnings():
            # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
            warnings.filterwarnings(
                "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
            )
            flags = _get_cuda_arch_flags()
        arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
    except Exception:
        arch = "noarch"
    flashinfer_base = os.getenv(
        "FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix()
    )
    # e.g.: $HOME/.cache/flashinfer/75_80_89_90/
    return pathlib.Path(flashinfer_base) / ".cache" / "flashinfer" / arch


# use pathlib
FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"
FLASHINFER_INCLUDE_DIR = pathlib.Path(get_include())
FLASHINFER_CSRC_DIR = pathlib.Path(get_csrc_dir())
FLASHINFER_TVM_BINDING_DIR = pathlib.Path(get_tvm_binding_dir())
