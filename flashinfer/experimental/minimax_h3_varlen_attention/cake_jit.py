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
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated programs.
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): translation units, compile flags, FFI entry, argument
# plan and closure identity.  ``ROUTES`` maps ``"<variant>__<arch>"`` to the
# ordered stage -> module assignment of that program family:
#
# * ``bf16``          stages ``("attention", "combine")``
# * ``nvfp4_fp4pv``   stages ``("quantize", "attention", "attention_split", "combine")``
# * ``nvfp4_fp8pv``   stages ``("quantize", "attention", "attention_split", "combine")``
#
# ``quantize`` is one fused single-launch quantizer per PV mode
# (``minimax_h3_varlen_nvfp4_quantize_qkv`` / ``..._quantize_qk_fp8v``); the
# NVFP4 routes carry two attention programs, the dense ``attention`` (no
# K/V-split code; bound for plans without split units) and
# ``attention_split`` (reads the unit's K/V block range and writes partial
# rows; bound when the plan has split units) -- the runner binds exactly one
# of them per plan -- and both bindings carry the programmatic dependent
# launch attribute; ``combine`` is the shared K/V-split merge kernel
# (``minimax_h3_varlen_split_combine``) that finishes the units the host
# planner split over their K/V range (skipped by the runner when a plan has no
# split units).  Every module is an exact-arch program (the
# sm_100a NVFP4 attention uses the hybrid exp2 recipe, sm_103a
# ``tcgen05.ld.red``).  Both literals are populated verbatim by the
# generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {}
ROUTES: dict[str, dict[str, Any]] = {}

VARIANTS = ("bf16", "nvfp4_fp4pv", "nvfp4_fp8pv")
STAGES = {
    "bf16": ("attention", "combine"),
    "nvfp4_fp4pv": ("quantize", "attention", "attention_split", "combine"),
    "nvfp4_fp8pv": ("quantize", "attention", "attention_split", "combine"),
}
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def route_name(variant: str, arch: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown MiniMax-H3 varlen attention variant {variant!r}")
    return f"{variant}__{arch}"


def select_route(variant: str, arch: str) -> dict[str, Any]:
    """Return the registered route record for ``variant`` on ``arch``."""
    record = ROUTES.get(route_name(variant, arch))
    if record is None:
        raise NotImplementedError(
            f"The generated MiniMax-H3 packed-varlen {variant} attention program "
            f"for {arch} is not registered in this checkout yet "
            "(see flashinfer-ai/flashinfer#4532)"
        )
    if tuple(record["stages"]) != STAGES[variant]:
        raise RuntimeError(
            f"registered route {route_name(variant, arch)!r} has stages "
            f"{tuple(record['stages'])!r}, expected {STAGES[variant]!r}"
        )
    return record


def route_available(variant: str, arch: str) -> bool:
    return route_name(variant, arch) in ROUTES


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


@functools.cache
def gen_cake_minimax_h3_varlen_attention_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_minimax_h3_varlen_attention_module(name: str):
    return gen_cake_minimax_h3_varlen_attention_module(name).build_and_load()
