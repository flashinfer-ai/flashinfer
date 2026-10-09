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

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCES = ("nvfp4_sparse_mla_decode.cu", "nvfp4_sparse_mla_decode_jit_binding.cu")
_HEADERS = ("nvfp4_sparse_mla_decode.cuh",)
# Architecture-specific targets: the kernel uses e2m1 conversions that the family-portable targets lack.
_ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}


def _header_dirs() -> list[Path]:
    """Directories holding FlashInfer's binding headers, installed or in a source checkout."""
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file():
        return source
    raise FileNotFoundError(
        "FlashInfer binding headers (tvm_ffi_utils.h) were not found"
    )


def _source_digest() -> str:
    digest = hashlib.sha256()
    for name in (*_SOURCES, *_HEADERS):
        digest.update((_CSRC / name).read_bytes())
    return digest.hexdigest()[:16]


@functools.cache
def gen_nvfp4_sparse_mla_decode_module(arch: str) -> JitSpec:
    """JIT spec of the NVFP4 sparse-MLA decode kernel for ``arch`` (``sm_100a`` or ``sm_103a``)."""
    if arch not in _ARCH_NVCC_FLAGS:
        raise ValueError(f"NVFP4 sparse MLA decode is not built for {arch}")
    return gen_jit_spec(
        name=f"nvfp4_sparse_mla_decode_{arch.replace('_', '')}_{_source_digest()}",
        sources=[_CSRC / name for name in _SOURCES],
        extra_cuda_cflags=list(_ARCH_NVCC_FLAGS[arch]),
        extra_include_paths=[_CSRC, *_header_dirs()],
        # The kernel was validated without fast math; keep exp2f and the f16 conversions exact.
        use_fast_math=False,
    )
