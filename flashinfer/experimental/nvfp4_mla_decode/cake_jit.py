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

# Registry of the generated Cake programs (written by the generated-program
# export; do not edit by hand).
#
# Every program is one device translation unit plus its host binding, shared by
# every architecture in ``ARCHES``; the loader compiles it with the exact flag
# set of the device it runs on.  ``ARG_PLANS`` holds the one argument order per
# stage role (``main``: the persistent decode kernel; ``reduce``: the split-KV
# combine kernel), ``COMPILE_FLAGS`` the extra nvcc flags per role, ``PROGRAMS``
# every program once with its role, sources and architectures, and
# ``STAGE_PROGRAMS`` the program of each stage.
ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
STAGES = ("main", "reduce")
COMPILE_FLAGS: dict[str, list[str]] = {"main": ["--use_fast_math"], "reduce": []}
FFI_ENTRY: str = "run"
ARG_PLANS: dict[str, list[list[str]]] = {
    "main": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "QS"],
        ["tma_buffer", "KV"],
        ["tma_buffer", "KVS"],
        ["tma_buffer", "OT"],
        ["tma_buffer", "PO"],
        ["buffer", "O"],
        ["buffer", "LSE"],
        ["buffer", "partial_o"],
        ["buffer", "partial_lse"],
        ["buffer", "work_table"],
        ["buffer", "unit_first"],
        ["buffer", "page_table"],
        ["buffer", "seq_lens"],
        ["buffer", "q_indptr"],
        ["buffer", "sinks"],
        ["parameter", "rows_total"],
        ["parameter", "num_heads"],
        ["parameter", "q_len"],
        ["parameter", "max_pages"],
        ["parameter", "max_splits"],
        ["parameter", "scale_log2"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "reduce": [
        ["buffer", "o"],
        ["buffer", "lse"],
        ["buffer", "partial_o"],
        ["buffer", "partial_lse"],
        ["buffer", "row_splits"],
        ["parameter", "total_q"],
        ["parameter", "num_heads"],
        ["parameter", "max_splits"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_nvfp4_mla_decode_8bec898f3f4f8f29be51": {
        "role": "reduce",
        "sources": [
            "cake_nvfp4_mla_decode/cake_nvfp4_mla_decode_8bec898f3f4f8f29be51_kernel.cu",
            "cake_nvfp4_mla_decode/cake_nvfp4_mla_decode_8bec898f3f4f8f29be51_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_nvfp4_mla_decode_b777c32b356f30dc993a": {
        "role": "main",
        "sources": [
            "cake_nvfp4_mla_decode/cake_nvfp4_mla_decode_b777c32b356f30dc993a_kernel.cu",
            "cake_nvfp4_mla_decode/cake_nvfp4_mla_decode_b777c32b356f30dc993a_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
}
STAGE_PROGRAMS: dict[str, str] = {
    "main": "cake_nvfp4_mla_decode_b777c32b356f30dc993a",
    "reduce": "cake_nvfp4_mla_decode_8bec898f3f4f8f29be51",
}
TRACKING_ISSUE = "flashinfer-ai/flashinfer#5780"


def program_available(arch: str) -> bool:
    """True when every stage program is registered for ``arch``."""
    return all(
        arch in PROGRAMS[program]["arches"] for program in STAGE_PROGRAMS.values()
    )


def select_program(stage: str, arch: str) -> str:
    """Return the registered program of ``stage`` for ``arch`` (``sm_100a`` / ``sm_103a``)."""
    program = STAGE_PROGRAMS[stage]
    if arch not in PROGRAMS[program]["arches"]:
        raise NotImplementedError(
            f"The generated NVFP4 MLA decode {stage} program is not registered for "
            f"{arch} in this checkout (registered: {PROGRAMS[program]['arches']}; "
            f"see {TRACKING_ISSUE})"
        )
    return program


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
def gen_program(program: str, arch: str):
    """JIT spec of ``program`` compiled for ``arch`` (one cached library per pair)."""
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not built for {arch!r}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{program}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[arch], *COMPILE_FLAGS[record["role"]]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *{p.parent for p in sources}, *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_program(program: str, arch: str):
    return gen_program(program, arch).build_and_load()
