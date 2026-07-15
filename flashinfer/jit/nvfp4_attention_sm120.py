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
from .core import JitSpec, current_compilation_context, gen_jit_spec, sm120a_nvcc_flags


_NVFP4_ATTENTION_SM120_MODULE_NAME = "nvfp4_attention_sm120"

_NVFP4_ATTENTION_SM120_SOURCE_FILES = (
    "nvfp4_attention_sm120/nvfp4_attention_sm120_binding.cu",
    "nvfp4_attention_sm120/nvfp4_attention_sm120_quantize.cu",
)


_NVFP4_ATTENTION_SM120_CUDA_FLAGS = [
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "-U__CUDA_NO_NVFP4_OPERATORS__",
    "-U__CUDA_NO_NVFP4_CONVERSIONS__",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-DQBLKSIZE=128",
    "-DKBLKSIZE=128",
    "-DCTA256",
    "-DDQINRMEM",
    "-DPINGPONG_MATH_ORDER",
    "-DPINGPONG_EARLY_RELEASE_K",
    "-DCAUSAL_DISABLE_QK_ORDER",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
]


def _nvfp4_attention_sm120_source_path(source_file: str) -> Path:
    package_data_path = jit_env.FLASHINFER_CSRC_DIR / source_file
    if package_data_path.exists():
        return package_data_path
    return _repo_root() / "csrc" / source_file


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _nvfp4_attention_sm120_include_paths() -> list[Path]:
    root = _repo_root()
    candidates = [
        root / "include",
        root / "csrc",
        root / "3rdparty" / "cccl" / "cub",
        root / "3rdparty" / "cccl" / "libcudacxx" / "include",
        root / "3rdparty" / "cccl" / "thrust",
        root / "3rdparty" / "cutlass" / "include",
        root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
        root / "3rdparty" / "spdlog" / "include",
    ]
    return [path for path in candidates if path.exists()]


@functools.cache
def gen_nvfp4_attention_sm120_module() -> JitSpec:
    source_paths = [
        _nvfp4_attention_sm120_source_path(source_file)
        for source_file in _NVFP4_ATTENTION_SM120_SOURCE_FILES
    ]
    include_paths: list[str | Path] = []
    include_paths.extend(_nvfp4_attention_sm120_include_paths())
    # CUDA < 12.9 can't target the 12.0f/12.1a split (SM121 needs >= 12.9), so fall back to sm_120a.
    try:
        nvcc_flags = current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[12]
        )
    except RuntimeError:
        nvcc_flags = sm120a_nvcc_flags
    return gen_jit_spec(
        _NVFP4_ATTENTION_SM120_MODULE_NAME,
        source_paths,
        extra_cuda_cflags=nvcc_flags + _NVFP4_ATTENTION_SM120_CUDA_FLAGS,
        extra_include_paths=include_paths,
    )
