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
from pathlib import Path
from typing import Any, Optional

from ...jit import env as jit_env
from ...jit.core import (
    current_compilation_context,
    gen_jit_spec,
    refresh_current_compilation_context,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Explicit target-owned registration of the generated ragged BF16 grouped GEMM
# programs.  One record per (operation, output dtype, weight-gradient k tile,
# architecture), named ``cake_moe_grouped_gemm_<op>[_<bf16|f32>_k<tile>]__<arch>``.
# Each record carries its architecture, operation, output dtype, k tile,
# public helper, stage list and host-plan summary, plus one physical entry per
# stage (``main``; weight gradient also ``tail_reduce``) with the translation
# units, compile flags, FFI entry, argument plan, closure identity, launch
# geometry, caller-owned TMA descriptor workspace size and, for pointer-ABI
# stages, the descriptor preparation entry the host calls once per prepared
# launch.  Populated verbatim by the generated-program export; do not edit by
# hand.  Empty until the export is delivered.
MODULES: dict[str, dict[str, Any]] = {}

# Module constants of the production host planner (tile geometry, cluster
# shape, descriptor slot layout, k-tile selection thresholds and the split-K
# tail cost model), filled by the same export.  ``cake_backend`` reproduces the
# production launch plans from these values alone.
HOST_PLAN_CONSTANTS: dict[str, Any] = {}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
OPS = ("fwd", "dgrad", "wgrad")
# Exact-architecture payloads (tcgen05 / TMEM): one module per exact target,
# built with FlashInfer's exact flag sets, never a family target.
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def record_name(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Registry key of the program serving ``op`` on ``arch``.

    Weight-gradient programs are further keyed by output dtype (``"bfloat16"``
    / ``"float32"``) and k tile (256 / 512).
    """
    if op not in OPS:
        raise ValueError(
            f"unknown grouped GEMM operation {op!r}; expected one of {OPS}"
        )
    if arch not in ARCHES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHES}")
    base = f"cake_moe_grouped_gemm_{op}"
    if op == "wgrad":
        if out_dtype not in ("bfloat16", "float32") or tile_k not in (256, 512):
            raise ValueError(
                "weight-gradient programs are keyed by out_dtype ('bfloat16' / "
                f"'float32') and tile_k (256 / 512); got {out_dtype!r}, {tile_k!r}"
            )
        base += f"_{'f32' if out_dtype == 'float32' else 'bf16'}_k{tile_k}"
    return f"{base}__{arch}"


def registered_arches() -> tuple[str, ...]:
    """Architectures with at least one registered program, in ``ARCHES`` order."""
    present = {record["arch"] for record in MODULES.values()}
    return tuple(arch for arch in ARCHES if arch in present)


def program_registered(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> bool:
    """True when this checkout registers the program for ``op`` on ``arch``.

    For the weight gradient with ``tile_k=None`` any k tile of ``out_dtype``
    counts; ``out_dtype=None`` accepts either output dtype.
    """
    if op != "wgrad":
        return record_name(op, arch) in MODULES
    dtypes = (out_dtype,) if out_dtype else ("bfloat16", "float32")
    tiles = (tile_k,) if tile_k else (256, 512)
    return any(
        record_name(op, arch, out_dtype=d, tile_k=t) in MODULES
        for d in dtypes
        for t in tiles
    )


def select_module(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Return the registered record name for ``op`` on ``arch`` or raise."""
    name = record_name(op, arch, out_dtype=out_dtype, tile_k=tile_k)
    record = MODULES.get(name)
    if record is None:
        raise NotImplementedError(
            f"The generated ragged grouped GEMM program {name!r} is not registered "
            "in this checkout yet (the module registry of "
            "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit is filled by the "
            "generated-program export)"
        )
    if record["arch"] != arch or record["op"] != op:
        raise RuntimeError(
            f"registered module {name!r} is a {record['op']} program for "
            f"{record['arch']}, bound to {op} on {arch}"
        )
    return name


def build_target_arches() -> frozenset[str]:
    """Exact architectures FlashInfer builds for, restricted to ``ARCHES``.

    Follows ``FLASHINFER_CUDA_ARCH_LIST`` when set and the visible devices
    otherwise (``flashinfer.compilation_context.CompilationContext``); the
    loader never probes ``nvcc`` or the device itself.
    """
    context = current_compilation_context
    if not context.TARGET_CUDA_ARCHS:
        context = refresh_current_compilation_context()
    targets = {f"sm_{major}{minor}" for major, minor in context.TARGET_CUDA_ARCHS}
    return frozenset(targets) & frozenset(ARCHES)


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


def _physical(name: str, stage: str) -> dict[str, Any]:
    record = MODULES[name]
    if stage not in record["stages"]:
        raise KeyError(
            f"generated program {name!r} has stages {list(record['stages'])}, not {stage!r}"
        )
    return record[stage]


@functools.cache
def gen_module(name: str, stage: str):
    """JIT spec of one physical stage of a registered program.

    The cache name carries the sealed closure identity of the stage so a
    changed generated closure never reuses a stale extension; the JIT
    workspace directory already encodes the target set.
    """
    record = MODULES[name]
    physical = _physical(name, stage)
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_module(name: str, stage: str):
    """Build (once) and load one physical stage of a registered program."""
    arch = MODULES[name]["arch"]
    targets = build_target_arches()
    if arch not in targets:
        raise RuntimeError(
            f"generated program {name!r} targets {arch}, which is not a FlashInfer "
            f"build target (targets: {sorted(targets) or 'none'}); set "
            "FLASHINFER_CUDA_ARCH_LIST to include it"
        )
    return gen_module(name, stage).build_and_load()
