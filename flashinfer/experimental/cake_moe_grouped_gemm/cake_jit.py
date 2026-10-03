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

# Registry of the generated ragged grouped GEMM programs (filled verbatim by the
# Cake generated-program export; do not edit the literals by hand).
#
# ``PROGRAMS``: one record per physical program -- one kernel source plus its
# host binding, shared by every architecture in ``arches`` (genuine
# per-architecture lowering differences live inside the source under exact
# ``__CUDA_ARCH__`` regions) -- with its compile flags, FFI entry, argument
# plan, launch shape and per-architecture closure identity.  A program is
# JIT-compiled once per (architecture, compile-line specialization).
#
# ``ROUTES``: one record per public route -- ``fwd``, ``dgrad`` and
# ``wgrad_{bf16,f32}_k{256,512}`` -- naming its architectures, host plan and
# ordered stages.  Each stage binds a program and the compile-line
# specializations the loader defines for it (the weight-gradient tail-reduce
# program takes the k width of the partials it sums, ``WGRAD_TILE``, from the
# route's tile instead of carrying one source per tile).
PROGRAMS: dict[str, dict[str, Any]] = {}
ROUTES: dict[str, dict[str, Any]] = {}
HOST_PLAN_CONSTANTS: dict[str, Any] = {}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
OPS = ("fwd", "dgrad", "wgrad")
# Exact-architecture payloads (tcgen05 / TMEM): one build per exact target with
# FlashInfer's exact flag sets, never a family target.
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def record_name(
    op: str, *, out_dtype: Optional[str] = None, tile_k: Optional[int] = None
) -> str:
    """Route key of ``op``; weight-gradient routes are further keyed by output dtype and k tile."""
    if op not in OPS:
        raise ValueError(
            f"unknown grouped GEMM operation {op!r}; expected one of {OPS}"
        )
    base = f"cake_moe_grouped_gemm_{op}"
    if op == "wgrad":
        if out_dtype not in ("bfloat16", "float32") or tile_k not in (256, 512):
            raise ValueError(
                "weight-gradient routes are keyed by out_dtype ('bfloat16' / "
                f"'float32') and tile_k (256 / 512); got {out_dtype!r}, {tile_k!r}"
            )
        base += f"_{'f32' if out_dtype == 'float32' else 'bf16'}_k{tile_k}"
    return base


def registered_arches() -> tuple[str, ...]:
    """Architectures with at least one registered route, in ``ARCHES`` order."""
    present = {arch for route in ROUTES.values() for arch in route["arches"]}
    return tuple(arch for arch in ARCHES if arch in present)


def program_registered(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> bool:
    """True when this checkout registers a route of ``op`` for ``arch``.

    For the weight gradient with ``tile_k=None`` any k tile of ``out_dtype``
    counts; ``out_dtype=None`` accepts either output dtype.
    """
    if arch not in ARCHES:
        return False
    if op != "wgrad":
        route = ROUTES.get(record_name(op))
        return route is not None and arch in route["arches"]
    dtypes = (out_dtype,) if out_dtype else ("bfloat16", "float32")
    tiles = (tile_k,) if tile_k else (256, 512)
    for dtype in dtypes:
        for tile in tiles:
            route = ROUTES.get(record_name(op, out_dtype=dtype, tile_k=tile))
            if route is not None and arch in route["arches"]:
                return True
    return False


def select_module(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Return the registered route name serving ``op`` on ``arch`` or raise."""
    if arch not in ARCHES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHES}")
    name = record_name(op, out_dtype=out_dtype, tile_k=tile_k)
    route = ROUTES.get(name)
    if route is None or arch not in route["arches"]:
        raise NotImplementedError(
            f"The generated ragged grouped GEMM route {name!r} is not registered for "
            f"{arch} in this checkout (the registry of "
            "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit is filled by the "
            "generated-program export)"
        )
    if route["op"] != op:
        raise RuntimeError(
            f"registered route {name!r} is a {route['op']} route, bound to {op}"
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


def stage_binding(name: str, stage: str) -> dict[str, Any]:
    """The ``{"name", "program", "specializations"}`` record of ``stage`` of route ``name``."""
    route = ROUTES[name]
    for item in route["stages"]:
        if item["name"] == stage:
            return item
    raise KeyError(
        f"route {name!r} has stages {[s['name'] for s in route['stages']]}, not {stage!r}"
    )


def _specialization_items(
    specializations: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(specializations.items()))


@functools.cache
def _gen_module(program: str, arch: str, specializations: tuple[tuple[str, Any], ...]):
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not delivered for {arch}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    tag = "".join(f"_{key.lower()}{value}" for key, value in specializations)
    return gen_jit_spec(
        # The spec name carries the exact target, the compile-line specialization
        # values and the sealed closure identity, so a changed closure or value
        # never reuses a stale extension.
        name=f"{program}_{arch}{tag}_" + record["closure_sha256"][arch][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
            *[f"-D{key}={value}" for key, value in specializations],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


def gen_module(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
):
    """JIT spec of one physical program for ``arch`` under ``specializations``."""
    return _gen_module(program, arch, _specialization_items(specializations or {}))


@functools.cache
def _load_module(program: str, arch: str, specializations: tuple[tuple[str, Any], ...]):
    targets = build_target_arches()
    if arch not in targets:
        raise RuntimeError(
            f"generated program {program!r} is requested for {arch}, which is not a "
            f"FlashInfer build target (targets: {sorted(targets) or 'none'}); set "
            "FLASHINFER_CUDA_ARCH_LIST to include it"
        )
    return _gen_module(program, arch, specializations).build_and_load()


def load_module(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
):
    """Build (once per target and specialization) and load one physical program."""
    return _load_module(program, arch, _specialization_items(specializations or {}))
