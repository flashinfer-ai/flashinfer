# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration of the generated Blackwell contiguous grouped FP8 GEMM programs.

One record per exported physical program (one Weave-IR kernel route on one
architecture).  Each record names its generated translation units under
``csrc/cake_grouped_fp8_gemm``, the exact compile flags of the source build,
the tvm-ffi entry, the positional argument plan and the caller-owned TMA
descriptor storage size.  ``MODULES`` and ``ROUTE_GEOMETRY`` are populated
verbatim by the generated-program export; do not edit them by hand.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags

MODULES: dict[str, dict[str, Any]] = {}

# Tile geometry of every host route (including routes without an exported
# program), resolved from the source dispatcher: {route: {tile_m, tile_n,
# cluster_ctas}}.
ROUTE_GEOMETRY: dict[str, dict[str, int]] = {}

ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags}
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a"}


def _source_dirs() -> tuple[Path, Path]:
    checkout = Path(__file__).resolve().parents[3]
    source_root = checkout / "csrc"
    include_root = checkout / "include"
    if not (source_root / "tvm_ffi_utils.h").is_file():
        source_root = jit_env.FLASHINFER_CSRC_DIR
        include_root = jit_env.FLASHINFER_INCLUDE_DIR
    if not (source_root / "tvm_ffi_utils.h").is_file():
        raise FileNotFoundError("FlashInfer binding headers were not found")
    return source_root, include_root


def generated_program_available(device) -> bool:
    """True when this checkout registers generated programs for ``device``."""
    import torch

    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    return arch is not None and any(r["arch"] == arch for r in MODULES.values())


def select_module(arch: str, route: str) -> str:
    """Return the registered module name for one ``(arch, route)`` pair."""
    for name, record in MODULES.items():
        if record["arch"] == arch and record["route"] == route:
            return name
    raise NotImplementedError(
        f"no generated contiguous grouped FP8 GEMM program is registered for "
        f"route {route!r} on {arch}"
    )


@functools.cache
def gen_cake_grouped_fp8_gemm_module(name: str) -> JitSpec:
    record = MODULES[name]
    source_root, include_root = _source_dirs()
    sources = [source_root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=record["cache_name"],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            source_root,
            *sorted({p.parent for p in sources}),
            include_root,
        ],
        use_fast_math=False,
    )


@functools.cache
def load_cake_grouped_fp8_gemm_module(name: str):
    return gen_cake_grouped_fp8_gemm_module(name).build_and_load()


__all__ = [
    "MODULES",
    "ROUTE_GEOMETRY",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gen_cake_grouped_fp8_gemm_module",
    "generated_program_available",
    "load_cake_grouped_fp8_gemm_module",
    "select_module",
]
