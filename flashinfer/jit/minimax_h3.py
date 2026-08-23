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
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags
from .cpp_ext import is_cuda_version_at_least


_CUDA_SOURCE_NAME = "minimax_h3_bf16_pre_attention_sm103a.cu"
_PRECISE_MATH_FLAGS = [
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _minimax_h3_cuda_source() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / _CUDA_SOURCE_NAME
    if packaged.is_file():
        return packaged

    source_tree = _repo_root() / "csrc" / _CUDA_SOURCE_NAME
    if source_tree.is_file():
        return source_tree

    raise FileNotFoundError(
        f"MiniMax-H3 CUDA source not found. Checked:\n  - {packaged}\n  - {source_tree}"
    )


def _minimax_h3_include_dir() -> Path:
    packaged = jit_env.FLASHINFER_INCLUDE_DIR
    if packaged.is_dir():
        return packaged

    source_tree = _repo_root() / "include"
    if source_tree.is_dir():
        return source_tree

    raise FileNotFoundError(
        f"FlashInfer headers not found. Checked:\n  - {packaged}\n  - {source_tree}"
    )


def gen_minimax_h3_bf16_pre_attention_module() -> JitSpec:
    """Return the SM103a-only JIT spec for fused BF16 pre-attention."""

    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError("SM103a compilation requires CUDA 12.9 or newer")
    source = _minimax_h3_cuda_source()
    return gen_jit_spec(
        "minimax_h3_bf16_pre_attention_sm103a_v1",
        [source],
        extra_cuda_cflags=sm103a_nvcc_flags + _PRECISE_MATH_FLAGS,
        extra_include_paths=[_minimax_h3_include_dir()],
    )


__all__ = ["gen_minimax_h3_bf16_pre_attention_module"]
