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

JIT loader of the Cake fused tensor-parallel GEMM + MNNVL Lamport all-reduce (SM100 / SM103).

One generated device translation unit (``csrc/cake_gemm_allreduce/cake_gemm_allreduce_kernels.cu``: twelve bf16
kernels, world sizes 2 / 4 / 8 x token tiles 16 one-shot and 32 / 64 / 128 two-shot) compiled together with its tvm-ffi
launcher into one module that carries a cubin for every supported build target (``FLASHINFER_CUDA_ARCH_LIST`` when set,
otherwise the visible devices).  ``ROUTES`` and ``COMPILE_FLAGS`` are filled by the exporter from the device bundle's
launch contract (``manifest.json`` next to the sources).
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Any, Optional

import torch

from ..compilation_context import CompilationContext
from . import env as jit_env
from .core import JitSpec, gen_jit_spec

SOURCE_PACKAGE = "cake_gemm_allreduce"
MODULE_NAME = "cake_gemm_allreduce"
SOURCES = ("csrc/cake_gemm_allreduce/cake_gemm_allreduce_binding.cu",)
KERNEL_SOURCE = "csrc/cake_gemm_allreduce/cake_gemm_allreduce_kernels.cu"
COMPILE_FLAGS: list[str] = ["--use_fast_math"]
SUPPORTED_MAJOR_VERSIONS: tuple[int, ...] = (10,)
SUPPORTED_CAPABILITIES: tuple[tuple[int, int], ...] = ((10, 0), (10, 3))
WORLD_SIZES: tuple[int, ...] = (2, 4, 8)
TOKEN_TILES: tuple[int, ...] = (16, 32, 64, 128)
ONESHOT_MAX_TOKENS = 16
BLOCK_THREADS = 192
COLUMN_TILE = 64
K_MULTIPLE = 128
ELEMENT_BYTES = 2
LAMPORT_BUFFERS = 3
# (world_size, token_tile, twoshot) -> kernel symbol and dynamic shared memory; the launch grid is
# (N / 64, ceil(M / token_tile), 1) with 192 threads, PDL on request.
ROUTES: dict[tuple[int, int, bool], dict[str, Any]] = {
    (2, 16, False): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_oneshot_n16_bf16_ws2",
        "dynamic_smem_bytes": 126080,
    },
    (2, 32, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n32_bf16_ws2",
        "dynamic_smem_bytes": 152704,
    },
    (2, 64, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n64_bf16_ws2",
        "dynamic_smem_bytes": 206080,
    },
    (2, 128, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n128_bf16_ws2",
        "dynamic_smem_bytes": 214528,
    },
    (4, 16, False): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_oneshot_n16_bf16_ws4",
        "dynamic_smem_bytes": 126080,
    },
    (4, 32, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n32_bf16_ws4",
        "dynamic_smem_bytes": 152704,
    },
    (4, 64, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n64_bf16_ws4",
        "dynamic_smem_bytes": 206080,
    },
    (4, 128, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n128_bf16_ws4",
        "dynamic_smem_bytes": 214528,
    },
    (8, 16, False): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_oneshot_n16_bf16_ws8",
        "dynamic_smem_bytes": 126080,
    },
    (8, 32, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n32_bf16_ws8",
        "dynamic_smem_bytes": 152704,
    },
    (8, 64, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n64_bf16_ws8",
        "dynamic_smem_bytes": 206080,
    },
    (8, 128, True): {
        "symbol": "kernel_cake_blackwell_gemm_allreduce_twoshot_n128_bf16_ws8",
        "dynamic_smem_bytes": 214528,
    },
}


def select_route(num_tokens: int) -> tuple[int, bool]:
    """``(token_tile, twoshot)`` for ``num_tokens`` rows: the smallest tile covering M (128 above), one-shot to 16."""
    if (
        isinstance(num_tokens, bool)
        or not isinstance(num_tokens, int)
        or num_tokens <= 0
    ):
        raise ValueError("num_tokens must be a positive integer")
    if num_tokens <= ONESHOT_MAX_TOKENS:
        return ONESHOT_MAX_TOKENS, False
    for tile in TOKEN_TILES:
        if tile > ONESHOT_MAX_TOKENS and num_tokens <= tile:
            return tile, True
    return TOKEN_TILES[-1], True


def required_workspace_bytes(num_tokens: int, hidden_dim: int, world_size: int) -> int:
    """Bytes needed in each of the three Lamport buffers for ``num_tokens`` rows of ``hidden_dim``."""
    if int(world_size) not in WORLD_SIZES:
        raise ValueError(f"world_size must be one of {WORLD_SIZES}")
    if (
        isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim <= 0
    ):
        raise ValueError("hidden_dim must be a positive integer")
    _tile, twoshot = select_route(num_tokens)
    if twoshot:
        return (
            2
            * math.ceil(num_tokens / world_size)
            * world_size
            * hidden_dim
            * ELEMENT_BYTES
        )
    return num_tokens * hidden_dim * world_size * ELEMENT_BYTES


def _capability_of(major: Any, minor: Any) -> tuple[int, int]:
    """``(major, minor)`` of a ``CompilationContext.TARGET_CUDA_ARCHS`` entry (``(10, "3a")`` -> ``(10, 3)``)."""
    return int(major), int(re.sub(r"[a-z]+$", "", str(minor)))


@functools.cache
def target_capabilities() -> tuple[tuple[int, int], ...]:
    """Supported capabilities the module is built for in this process, ascending."""
    context = CompilationContext()
    targets = {
        _capability_of(major, minor) for major, minor in context.TARGET_CUDA_ARCHS
    }
    return tuple(sorted(targets & set(SUPPORTED_CAPABILITIES)))


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` when the kernels are built for it, else ``None``."""
    key = (int(capability[0]), int(capability[1]))
    return key if key in target_capabilities() else None


def nvcc_flags() -> list[str]:
    """``-gencode`` flags for every supported build target plus FlashInfer's common flags."""
    return CompilationContext().get_nvcc_flags_list(
        supported_major_versions=list(SUPPORTED_MAJOR_VERSIONS)
    )


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake fused GEMM + all-reduce CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the {SOURCE_PACKAGE} package: {relative!r}"
        )
    return _source_dir() / parts[2]


@functools.cache
def spec() -> JitSpec:
    flags = [*nvcc_flags(), *COMPILE_FLAGS]
    source_dir = _source_dir()
    return gen_jit_spec(
        name=MODULE_NAME,
        sources=[_source_path(relative) for relative in SOURCES],
        extra_cuda_cflags=flags,
        extra_include_paths=[source_dir.parent, source_dir],
        use_fast_math="--use_fast_math" in COMPILE_FLAGS,
    )


@functools.cache
def load():
    return spec().build_and_load()


def run_cake_gemm_allreduce(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    *,
    mc_ptr: int,
    uc_ptr_local: int,
    uc_ptrs_dev: int,
    buffer_flags: torch.Tensor,
    rank: int,
    world_size: int,
    token_tile: int,
    twoshot: bool,
    launch_with_pdl: bool,
) -> None:
    """Launch the route ``(world_size, token_tile, twoshot)`` on the current stream of ``x``'s device."""
    if (int(world_size), int(token_tile), bool(twoshot)) not in ROUTES:
        raise ValueError(
            f"no fused GEMM + all-reduce kernel for world_size={world_size} token_tile={token_tile} twoshot={twoshot}"
        )
    capability = torch.cuda.get_device_capability(x.device)
    if supported_capability(capability) is None:
        raise ValueError(
            "the Cake fused GEMM + all-reduce is not built for compute capability "
            f"{capability[0]}.{capability[1]} (build targets: "
            f"{[f'{major}.{minor}' for major, minor in target_capabilities()]}; "
            "set FLASHINFER_CUDA_ARCH_LIST to include it)"
        )
    load().run(
        x,
        w,
        out,
        int(mc_ptr),
        int(uc_ptr_local),
        int(uc_ptrs_dev),
        buffer_flags,
        int(rank),
        int(world_size),
        int(token_tile),
        int(bool(twoshot)),
        int(bool(launch_with_pdl)),
    )
