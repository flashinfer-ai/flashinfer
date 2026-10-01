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

# Explicit target-owned registration of the generated programs of the per-token
# NVFP4 path (SM100 / SM103).
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): translation units, compile flags, FFI entry, argument
# plan and closure identity.  ``KERNELS`` maps ``"<arch>"`` to the logical
# kernel key -> module assignment the host dispatcher resolves at preparation:
#
# * ``quant:k<K>_t<threads>_mb<min_blocks>_<bf16|f16>[_fold]``
#   the per-token quantizer (one CTA per token row, register-resident row)
#   for one activation width, CTA width and launch-bounds occupancy,
#   input dtype and folded output scale (``cake_backend.quant_plan``);
# * ``gemm:<orientation><tile>[_k512][_2cta][_sk<split>]_<bf16|f16>[_a<hint>]
#   [_b<hint>][_l2<promo>][_g<group>][_clc][_amc<width>]``
#   the persistent block-scaled tcgen05 GEMM with the per-token alpha epilogue
#   for one tactic (``cake_backend.default_tactic`` / ``gemm_kernel_key``):
#   ``m<tile_n>`` is the 128-token (``_2cta``: 256-token cluster pair) tile
#   with ``tile_n`` weight columns, ``n<tile_n>`` the swapped orientation of
#   ``M <= 32`` with ``tile_n`` tokens per tile; ``_sk`` is the cluster split-K
#   variant of either orientation, ``_amc`` the 1-CTA program whose cluster
#   of ``width`` weight tiles shares the activation stages by multicast.
#
# Every module is an exact-architecture program (the GEMM uses tcgen05 / TMEM
# and ``cta_group::2``).  Both literals are populated verbatim by the
# generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {}
KERNELS: dict[str, dict[str, str]] = {}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when ``arch`` is registered and carries every key in ``required_keys``."""
    table = KERNELS.get(arch)
    return table is not None and all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered physical module for ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if table is None:
        raise NotImplementedError(
            f"The generated per-token NVFP4 programs for {arch} are not "
            "registered in this checkout yet"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated per-token NVFP4 kernel {key!r} for {arch} is not "
            "registered in this checkout (the shape is outside the validated "
            "matrix of the cake backend)"
        )
    record = MODULES[name]
    if record["arch"] != arch:
        raise RuntimeError(
            f"registered module {name!r} is an {record['arch']} program bound to {arch}"
        )
    return name


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
def gen_cake_nvfp4_per_token_module(name: str):
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
def load_cake_nvfp4_per_token_module(name: str):
    return gen_cake_nvfp4_per_token_module(name).build_and_load()
