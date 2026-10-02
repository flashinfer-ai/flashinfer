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
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Explicit target-owned registration of the generated indexer programs.  One
# record per architecture (``sm_100a``, ``sm_103a``, ``sm_107a``).  A record
# carries ``arch``, the host binding profile ``abi`` (the keyword set its
# kernels expect, see ``cake_backend.CONTRACT_TENSORS`` / ``CONTRACT_SCALARS``),
# the list of kernel ``stages`` it registers, the host-evaluated candidate-gate
# policy ``gate_policy`` (see ``cake_backend.GatePolicy``), the documented
# numerics of the program (``numerics``: the zero-sign policy of the head
# reduction), and one physical entry per stage (translation units, compile
# flags, FFI entry, argument plan, grid rule, launch geometry and closure
# identity).
#
# PLACEHOLDER: the registry is empty until the generated-program export lands.
# ``select_module`` raises ``NotImplementedError`` for every architecture,
# ``cake_backend.generated_program_available`` returns ``False`` and the GPU
# tests skip.  Populated verbatim by the export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {}

# Kernel stages of one indexer call, in launch order.  ``scan`` is the
# persistent fused kernel (scoring, exact candidate gate, per-row selection; it
# writes the unordered selected (id, score) pairs and the padding straight into
# the outputs); ``finalize`` / ``finalize_small`` sort every row by ascending
# key id in place (one CTA per row).  A record registers ``scan`` and at least
# one finalize stage; ``finalize_small`` serves ``top_k <=
# gate_policy["finalize_small_max_top_k"]`` and ``finalize`` every ``top_k``.
STAGES = ("scan", "finalize", "finalize_small")
FINALIZE_STAGES = ("finalize", "finalize_small")
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 / SM107.)"""
    return arch in ARCH_NVCC_FLAGS


def registered_archs() -> tuple[str, ...]:
    """Architectures with a registered program, in ``ARCHES`` order."""
    present = {record["arch"] for record in MODULES.values()}
    return tuple(arch for arch in ARCHES if arch in present)


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch``."""
    names = [name for name, record in MODULES.items() if record["arch"] == arch]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one DSA indexer program: {names}"
        )
    if not names:
        raise NotImplementedError(
            f"The generated DSA indexer top-k program for {arch} is not registered "
            "in this checkout yet (see flashinfer-ai/flashinfer#5676)"
        )
    return names[0]


def registered_stages(name: str) -> tuple[str, ...]:
    """Stages a record registers, in launch order."""
    present = tuple(stage for stage in STAGES if stage in MODULES[name])
    declared = tuple(MODULES[name].get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record {name!r} declares stages {declared} but carries {present}"
        )
    return present


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
def gen_cake_dsa_indexer_module(name: str, stage: str):
    record = MODULES[name]
    if not toolchain_supports(record["arch"]):
        raise RuntimeError(
            f"generated DSA indexer program {name!r} targets {record['arch']}, "
            "which this checkout cannot compile"
        )
    physical = record[stage]
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
def load_cake_dsa_indexer_module(name: str, stage: str):
    return gen_cake_dsa_indexer_module(name, stage).build_and_load()
