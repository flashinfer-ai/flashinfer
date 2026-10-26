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

from pathlib import Path

from . import env as jit_env
from .core import (
    JitSpec,
    current_compilation_context,
    gen_jit_spec,
    refresh_current_compilation_context,
)

_CUDA_SOURCE_NAME = "cake_minimax_h3_sm120_nvfp4_varlen_attention_sm120a.cu"
# GB202 only: the device code uses mma.sync kind::mxf4nvf4 (block-scaled QK^T and PV) + TMA (no tcgen05),
# and the register schedule is tuned for the SM120 tensor-pipe / SMEM budget.
_SUPPORTED_MAJOR_VERSIONS = [12]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cuda_source() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / _CUDA_SOURCE_NAME
    if packaged.is_file():
        return packaged
    source_tree = _repo_root() / "csrc" / _CUDA_SOURCE_NAME
    if source_tree.is_file():
        return source_tree
    raise FileNotFoundError(
        "MiniMax-H3 SM120 NVFP4 varlen attention CUDA source not found. Checked:\n"
        f"  - {packaged}\n  - {source_tree}"
    )


def gen_minimax_h3_sm120_nvfp4_varlen_attention_module() -> JitSpec:
    """JIT spec for the SM120 (GB202) NVFP4 (SageAttention3-recipe) MiniMax-H3 packed-varlen attention kernels."""

    compilation_context = current_compilation_context
    if not compilation_context.TARGET_CUDA_ARCHS:
        compilation_context = refresh_current_compilation_context()
    nvcc_flags = compilation_context.get_nvcc_flags_list(
        supported_major_versions=_SUPPORTED_MAJOR_VERSIONS
    )
    return gen_jit_spec(
        "minimax_h3_sm120_nvfp4_varlen_attention_sm120a_v1",
        [_cuda_source()],
        extra_cuda_cflags=nvcc_flags,
    )


__all__ = ["gen_minimax_h3_sm120_nvfp4_varlen_attention_module"]
