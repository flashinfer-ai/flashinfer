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
from ...jit.cpp_ext import get_cuda_version

# Registry of the generated Cake programs (written by the generated-program
# export; do not edit by hand).
#
# Every program is one device translation unit plus its host binding, shared by
# every architecture in ``ARCHES``; the loader compiles it with the exact flag
# set of the device it runs on.  ``ARG_PLANS`` holds the one argument order per
# stage role (``main``: the swapped-AB attention kernel; ``reduce``: the
# split-KV merge; ``quantize``: the BF16 -> NVFP4 query quantizer),
# ``COMPILE_FLAGS`` the extra nvcc flags per role, ``PROGRAMS`` every program
# once with its role, sources and architectures, and ``KERNELS`` the logical
# kernel key (``main_rt16`` / ``main_rt32`` / ``main_rt48``, ``reduce_w4`` /
# ``reduce_w2`` / ``reduce_w1`` / ``reduce_cta``, ``quantize``) -> program.  The
# attention programs spell the Blackwell QMUL4 in PTX ISA 9.4 and carry
# ``min_cuda_version``: the loader refuses an older toolkit.
ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
STAGES = ("quantize", "main", "reduce")
COMPILE_FLAGS: dict[str, list[str]] = {
    "quantize": ["--use_fast_math"],
    "main": ["--use_fast_math"],
    "reduce": ["--use_fast_math"],
}
FFI_ENTRY: str = "run"
ARG_PLANS: dict[str, list[list[str]]] = {
    "quantize": [
        ["buffer", "q_bf16"],
        ["buffer", "q_nope"],
        ["buffer", "q_sf"],
        ["buffer", "q_rope"],
        ["buffer", "q_scale"],
        ["parameter", "rows"],
        ["parameter", "c_nope"],
        ["parameter", "c_rope"],
        ["parameter", "kpe_scale"],
        ["parameter", "ckv_scale"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "main": [
        ["tma_buffer", "tmap_qn"],
        ["tma_buffer", "tmap_qs"],
        ["tma_buffer", "tmap_qr"],
        ["tma_buffer", "tmap_k"],
        ["tma_buffer", "tmap_ks"],
        ["tma_buffer", "tmap_kr"],
        ["buffer", "q_scale"],
        ["buffer", "partial_O"],
        ["buffer", "partial_max"],
        ["buffer", "partial_sum"],
        ["buffer", "lse"],
        ["buffer", "seq_lens"],
        ["buffer", "kv_len_global"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "page_table"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "num_split"],
        ["parameter", "max_pages_per_seq"],
        ["parameter", "page_shift"],
        ["parameter", "cp_world"],
        ["parameter", "cp_rank"],
        ["parameter", "has_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "reduce": [
        ["buffer", "partial_O"],
        ["buffer", "partial_max"],
        ["buffer", "partial_sum"],
        ["buffer", "O"],
        ["buffer", "lse"],
        ["buffer", "cum_seq_lens_q"],
        ["parameter", "batch"],
        ["parameter", "num_heads"],
        ["parameter", "num_split"],
        ["parameter", "bmm2_scale"],
        ["parameter", "lse_bias"],
        ["parameter", "has_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_mla_nvfp4_paged_decode_19c7a30a6bd596fcdaf0": {
        "role": "main",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_19c7a30a6bd596fcdaf0_cu134_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_19c7a30a6bd596fcdaf0_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "min_cuda_version": "13.4",
    },
    "cake_mla_nvfp4_paged_decode_461fdb80f34f6610cafd": {
        "role": "main",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_461fdb80f34f6610cafd_cu134_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_461fdb80f34f6610cafd_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "min_cuda_version": "13.4",
    },
    "cake_mla_nvfp4_paged_decode_6f1a02023e9c749c14e7": {
        "role": "reduce",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_6f1a02023e9c749c14e7_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_6f1a02023e9c749c14e7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_nvfp4_paged_decode_98d7b856095fc8779a66": {
        "role": "reduce",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_98d7b856095fc8779a66_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_98d7b856095fc8779a66_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_nvfp4_paged_decode_a97a6e8c4568efb5162e": {
        "role": "main",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_a97a6e8c4568efb5162e_cu134_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_a97a6e8c4568efb5162e_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "min_cuda_version": "13.4",
    },
    "cake_mla_nvfp4_paged_decode_a9bf6a63f937e7cacaba": {
        "role": "quantize",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_a9bf6a63f937e7cacaba_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_a9bf6a63f937e7cacaba_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_nvfp4_paged_decode_b2e4c5513e1d7e5618c0": {
        "role": "reduce",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_b2e4c5513e1d7e5618c0_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_b2e4c5513e1d7e5618c0_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_nvfp4_paged_decode_d0b9705bb5f3deafe9b7": {
        "role": "reduce",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_d0b9705bb5f3deafe9b7_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_d0b9705bb5f3deafe9b7_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_mla_nvfp4_paged_decode_d2829e4e3bf495b79161": {
        "role": "main",
        "sources": [
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_d2829e4e3bf495b79161_cu134_kernel.cu",
            "cake_mla_nvfp4_paged_decode/cake_mla_nvfp4_paged_decode_d2829e4e3bf495b79161_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "min_cuda_version": "13.4",
    },
}
KERNELS: dict[str, str] = {
    "main_rt16": "cake_mla_nvfp4_paged_decode_d2829e4e3bf495b79161",
    "main_rt32": "cake_mla_nvfp4_paged_decode_19c7a30a6bd596fcdaf0",
    "main_rt48": "cake_mla_nvfp4_paged_decode_a97a6e8c4568efb5162e",
    "main_wide": "cake_mla_nvfp4_paged_decode_461fdb80f34f6610cafd",
    "quantize": "cake_mla_nvfp4_paged_decode_a9bf6a63f937e7cacaba",
    "reduce_cta": "cake_mla_nvfp4_paged_decode_d0b9705bb5f3deafe9b7",
    "reduce_w1": "cake_mla_nvfp4_paged_decode_b2e4c5513e1d7e5618c0",
    "reduce_w2": "cake_mla_nvfp4_paged_decode_98d7b856095fc8779a66",
    "reduce_w4": "cake_mla_nvfp4_paged_decode_6f1a02023e9c749c14e7",
}
TRACKING_ISSUE = "flashinfer-ai/flashinfer#4644"


def program_available(arch: str) -> bool:
    """True when every kernel key is registered for ``arch``."""
    return all(arch in PROGRAMS[program]["arches"] for program in KERNELS.values())


def toolkit_supports(program: str) -> bool:
    """True when the nvcc of this environment meets the program's ``min_cuda_version``."""
    minimum = PROGRAMS[program].get("min_cuda_version")
    if minimum is None:
        return True
    version = get_cuda_version()
    major, minor = (int(part) for part in minimum.split("."))
    return (version.major, version.minor) >= (major, minor)


def select_program(kind: str, arch: str) -> str:
    """Return the registered program of kernel ``kind`` for ``arch`` (``sm_100a`` / ``sm_103a``)."""
    program = KERNELS.get(kind)
    if program is None:
        raise ValueError(f"Cake NVFP4 MLA decode has no generated kernel {kind!r}")
    if arch not in PROGRAMS[program]["arches"]:
        raise NotImplementedError(
            f"The generated Cake NVFP4 MLA decode kernel {kind!r} is not registered for "
            f"{arch} in this checkout (registered: {PROGRAMS[program]['arches']}; "
            f"see {TRACKING_ISSUE})"
        )
    if not toolkit_supports(program):
        raise NotImplementedError(
            f"The generated Cake NVFP4 MLA decode kernel {kind!r} spells the Blackwell "
            f"QMUL4 in PTX ISA 9.4 and needs CUDA {PROGRAMS[program]['min_cuda_version']} "
            f"or newer; this environment's nvcc is {get_cuda_version()} (see {TRACKING_ISSUE})"
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
