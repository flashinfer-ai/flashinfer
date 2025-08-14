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

# NOTE(lequn): Do not "from .jit.env import xxx".
# Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
# This helps AOT script to override envs.

import os
import pathlib
import re
import warnings

from torch.utils.cpp_extension import _get_cuda_arch_flags

FLASHINFER_BASE_DIR = pathlib.Path(
    os.getenv("FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

FLASHINFER_CACHE_DIR = FLASHINFER_BASE_DIR / ".cache" / "flashinfer"
FLASHINFER_CUBIN_DIR = pathlib.Path(
    os.getenv("FLASHINFER_CUBIN_DIR", (FLASHINFER_CACHE_DIR / "cubins").as_posix())
)


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
    # e.g.: $HOME/.cache/flashinfer/75_80_89_90/
    return FLASHINFER_CACHE_DIR / arch


# use pathlib
FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"
_package_root = pathlib.Path(__file__).resolve().parents[1]
FLASHINFER_DATA = _package_root / "data"
FLASHINFER_INCLUDE_DIR = _package_root / "data" / "include"
FLASHINFER_CSRC_DIR = _package_root / "data" / "csrc"
# FLASHINFER_SRC_DIR = _package_root / "data" / "src"
FLASHINFER_TVM_BINDING_DIR = _package_root / "data" / "tvm_binding"
FLASHINFER_AOT_DIR = _package_root / "data" / "aot"
CUTLASS_INCLUDE_DIRS = [
    _package_root / "data" / "cutlass" / "include",
    _package_root / "data" / "cutlass" / "tools" / "util" / "include",
]
SPDLOG_INCLUDE_DIR = _package_root / "data" / "spdlog" / "include"


def get_nvshmem_include_dirs():
    paths = os.environ.get("NVSHMEM_INCLUDE_PATH")
    if paths is not None:
        return [pathlib.Path(p) for p in paths.split(os.pathsep) if p]

    import nvidia.nvshmem

    path = pathlib.Path(nvidia.nvshmem.__path__[0]) / "include"
    return [path]


def get_nvshmem_lib_dirs():
    paths = os.environ.get("NVSHMEM_LIBRARY_PATH")
    if paths is not None:
        return [pathlib.Path(p) for p in paths.split(os.pathsep) if p]

    import nvidia.nvshmem

    path = pathlib.Path(nvidia.nvshmem.__path__[0]) / "lib"
    return [path]
