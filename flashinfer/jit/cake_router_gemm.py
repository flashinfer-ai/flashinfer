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

# Cake Router GEMM for datacenter Blackwell: the 32 source-built programs of
# ``csrc/cake_router_gemm`` (16 token counts x 2 hidden sizes) behind one TVM-FFI
# binding, built through the standard JIT path.

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any, Optional

from ..compilation_context import CompilationContext
from . import env as jit_env
from .core import JitSpec, gen_jit_spec

# The kernels use plain vector loads, FP32 FMA, warp shuffles and shared memory, so the
# sources compile for every 10.x build target of the process (``FLASHINFER_CUDA_ARCH_LIST``
# on AOT hosts, the visible devices otherwise).
SUPPORTED_MAJOR_VERSIONS = (10,)
# Capabilities the kernels were validated and measured on; other 10.x devices keep the
# reference router GEMM.
VALIDATED_CAPABILITIES = ((10, 0), (10, 3))
NUM_TOKENS = tuple(range(1, 17))
HIDDEN_DIMS = (6144, 7168)
# One device source per (num_tokens, hidden_dim), declared and launched by
# ``cake_router_gemm.cu``.
DEVICE_SOURCES = tuple(
    f"cake_router_gemm_m{num_tokens}_k{hidden_dim}_device.cu"
    for hidden_dim in HIDDEN_DIMS
    for num_tokens in NUM_TOKENS
)


def _csrc_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "cake_router_gemm"
    if packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_router_gemm"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake Router GEMM sources were not found. Checked:\n"
        f"  - {packaged}\n"
        f"  - {checkout}"
    )


def _capability_of(major: Any, minor: Any) -> tuple[int, int]:
    """``(major, minor)`` of a ``CompilationContext.TARGET_CUDA_ARCHS`` entry.

    ``(10, "3a")`` -> ``(10, 3)``.
    """
    return int(major), int(re.sub(r"[a-z]+$", "", str(minor)))


@functools.cache
def target_capabilities() -> tuple[tuple[int, int], ...]:
    """Compute capabilities the module is built for, read once per process."""
    context = CompilationContext()
    return tuple(
        sorted(
            {
                _capability_of(major, minor)
                for major, minor in context.TARGET_CUDA_ARCHS
                if int(major) in SUPPORTED_MAJOR_VERSIONS
            }
        )
    )


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """``(major, minor)`` when the Cake kernels serve this device, else ``None``.

    A device outside :data:`VALIDATED_CAPABILITIES`, or one left out of
    ``FLASHINFER_CUDA_ARCH_LIST``, keeps the reference router GEMM.
    """
    key = (int(capability[0]), int(capability[1]))
    if key in VALIDATED_CAPABILITIES and key in target_capabilities():
        return key
    return None


def nvcc_flags() -> list[str]:
    """``-gencode`` flags for every 10.x build target plus FlashInfer's common flags."""
    return CompilationContext().get_nvcc_flags_list(
        supported_major_versions=list(SUPPORTED_MAJOR_VERSIONS)
    )


def gen_cake_router_gemm_module() -> JitSpec:
    """JIT spec of the Cake router GEMM binding and its 32 programs.

    One fatbin for every 10.x build target.
    """
    csrc = _csrc_dir()
    sources = [csrc / "cake_router_gemm.cu"] + [csrc / name for name in DEVICE_SOURCES]
    missing = [source.name for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cake Router GEMM source package under {csrc} is incomplete: missing {missing}"
        )
    return gen_jit_spec(
        "cake_router_gemm",
        sources,
        extra_cuda_cflags=nvcc_flags(),
        extra_include_paths=[csrc.parent],
    )


__all__ = [
    "DEVICE_SOURCES",
    "HIDDEN_DIMS",
    "NUM_TOKENS",
    "SUPPORTED_MAJOR_VERSIONS",
    "VALIDATED_CAPABILITIES",
    "gen_cake_router_gemm_module",
    "nvcc_flags",
    "supported_capability",
    "target_capabilities",
]
