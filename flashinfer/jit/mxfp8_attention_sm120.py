"""
Copyright (c) 2026 by FlashInfer team.

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

import functools
from pathlib import Path

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm120a_nvcc_flags


_MXFP8_ATTENTION_SM120_MODULE_NAME = "mxfp8_attention_sm120"

_MXFP8_ATTENTION_SM120_SOURCE_FILES = (
    "mxfp8_attention_sm120/mxfp8_attention_sm120_binding.cu",
)


_MXFP8_ATTENTION_SM120_CUDA_FLAGS = [
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
]


def _mxfp8_attention_sm120_source_path(source_file: str) -> Path:
    package_data_path = jit_env.FLASHINFER_CSRC_DIR / source_file
    if package_data_path.exists():
        return package_data_path
    return _repo_root() / "csrc" / source_file


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mxfp8_attention_sm120_include_paths() -> list[Path]:
    root = _repo_root()
    candidates = [
        root / "include",
        root / "csrc",
        root / "3rdparty" / "cutlass" / "include",
        root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
        root / "3rdparty" / "spdlog" / "include",
    ]
    return [path for path in candidates if path.exists()]


@functools.cache
def gen_mxfp8_attention_sm120_module() -> JitSpec:
    source_paths = [
        _mxfp8_attention_sm120_source_path(source_file)
        for source_file in _MXFP8_ATTENTION_SM120_SOURCE_FILES
    ]
    include_paths: list[str | Path] = []
    include_paths.extend(_mxfp8_attention_sm120_include_paths())
    return gen_jit_spec(
        _MXFP8_ATTENTION_SM120_MODULE_NAME,
        source_paths,
        extra_cuda_cflags=sm120a_nvcc_flags + _MXFP8_ATTENTION_SM120_CUDA_FLAGS,
        extra_include_paths=include_paths,
    )
