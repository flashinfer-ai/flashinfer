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

# Explicit target-owned registration of the generated programs of the
# Kimi-K3 TP12 fused LatentMoE communication tail (SM100 / SM103, twelve
# ranks in one multi-node NVLink domain).
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): translation units, compile flags, FFI entry, argument
# plan and closure identity.  ``KERNELS`` maps ``"<arch>"`` to the logical
# kernel key -> module assignment the host runtime resolves at preparation:
#
# * ``k1_oneshot:r<rank>``  the one-shot Lamport all-reduce of the routed
#   partial fused with KimiRMSNorm for ``M <= 16`` tokens; the rank is a
#   trace-time specialisation (reduction order and local packet position),
#   so one module per rank;
# * ``k1_twoshot:<grouped|pinned>``  the token-sliced two-shot form (owner
#   reduce + norm, multicast broadcast) for ``M > 16``; ``grouped`` issues all
#   twelve remote loads of the owner retry body before the first test
#   (``M <= 256``), ``pinned`` keeps the per-rank pinned body (``M > 256``);
# * ``k3:<grouped|pinned>``  the column reduce-scatter of the shared partial,
#   fused add of this rank's up-projection slice, one BF16 rounding and the
#   multicast all-gather of the output row, with the same poll schedule
#   selection by ``M``.
#
# Every module is an exact-architecture program launched with programmatic
# dependent launch.  Both literals are populated verbatim by the
# generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {}

KERNELS: dict[str, dict[str, str]] = {}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
WORLD_SIZE = 12
POLL_SCHEDULES = ("grouped", "pinned")


def required_kernel_keys() -> tuple[str, ...]:
    """Every logical kernel the runtime can select on one architecture."""
    return (
        *(f"k1_oneshot:r{rank}" for rank in range(WORLD_SIZE)),
        *(f"k1_twoshot:{schedule}" for schedule in POLL_SCHEDULES),
        *(f"k3:{schedule}" for schedule in POLL_SCHEDULES),
    )


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when ``arch`` is registered and carries every key in ``required_keys``."""
    table = KERNELS.get(arch)
    return table is not None and all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered physical module for ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if table is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 TP12 tail programs for {arch} are not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#4542)"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 TP12 tail kernel {key!r} for {arch} is not "
            "registered in this checkout (see flashinfer-ai/flashinfer#4542)"
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
def gen_cake_kimi_k3_tp12_tail_module(name: str):
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
def load_cake_kimi_k3_tp12_tail_module(name: str):
    return gen_cake_kimi_k3_tp12_tail_module(name).build_and_load()
