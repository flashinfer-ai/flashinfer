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

# Generated programs of the NVFP4 paged-KV MSA decode, one record per delivered
# source pair.  A program lists the architectures its source compiles for
# (``arches``); the loader compiles it with the exact flag set of the device it
# runs on, so two architectures never share one cached library.  ``route`` is
# ``persistent`` (split-KV decode; ``splits`` CTAs per work item, ``tail`` for
# the last-round-split variant whose ``cluster_capacity`` maps a part's SM
# count to the CTAs it co-schedules as ``splits``-CTA clusters) or ``short``
# (one ``cluster``-CTA cluster per work item of at most ``max_pages`` pages, at
# most ``max_clusters`` clusters per launch).  ``ARG_PLANS`` lists the host
# binding's argument order once per route.  Both literals are written by the
# generated-program export; do not edit by hand.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_msa_nvfp4_decode_229d54fc8dddea35a745": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "persistent",
        "splits": 8,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_229d54fc8dddea35a745_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_229d54fc8dddea35a745_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_2342d72176334739de30": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "persistent",
        "splits": 4,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_2342d72176334739de30_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_2342d72176334739de30_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_2fc0d04d28bfa0354349": {
        "arches": ["sm_107a"],
        "route": "persistent",
        "splits": 1,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_2fc0d04d28bfa0354349_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_2fc0d04d28bfa0354349_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_964bc168196f68c6d253": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "persistent",
        "splits": 1,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_964bc168196f68c6d253_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_964bc168196f68c6d253_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_97ddd7fd98fe8dfce00c": {
        "arches": ["sm_107a"],
        "route": "persistent",
        "splits": 2,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_97ddd7fd98fe8dfce00c_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_97ddd7fd98fe8dfce00c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_ac1f2f3299fa7f8141f1": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "route": "short",
        "splits": 1,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_ac1f2f3299fa7f8141f1_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_ac1f2f3299fa7f8141f1_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
    },
    "cake_msa_nvfp4_decode_b263f585211d72cdf304": {
        "arches": ["sm_107a"],
        "route": "persistent",
        "splits": 2,
        "tail": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_b263f585211d72cdf304_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_b263f585211d72cdf304_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
        "cluster_capacity": {"212": 212},
    },
    "cake_msa_nvfp4_decode_b86d9fd7247676d1c491": {
        "arches": ["sm_107a"],
        "route": "persistent",
        "splits": 4,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_b86d9fd7247676d1c491_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_b86d9fd7247676d1c491_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_ba3a78ef20ec7066c0c0": {
        "arches": ["sm_107a"],
        "route": "persistent",
        "splits": 8,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_ba3a78ef20ec7066c0c0_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_ba3a78ef20ec7066c0c0_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_df2a726f1b0c11c802eb": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "persistent",
        "splits": 2,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_df2a726f1b0c11c802eb_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_df2a726f1b0c11c802eb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
}
ARG_PLANS: dict[str, list[list[str]]] = {
    "persistent": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "K_scale"],
        ["tma_buffer", "V"],
        ["tma_buffer", "V_scale"],
        ["buffer", "O"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "task_kind"],
        ["buffer", "task_kv_head"],
        ["parameter", "total_q"],
        ["parameter", "seqlen_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "output_scale"],
        ["parameter", "msa_max_pages"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "short": [
        ["buffer", "Q"],
        ["buffer", "K"],
        ["buffer", "K_scale"],
        ["buffer", "V"],
        ["buffer", "V_scale"],
        ["buffer", "O"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "task_kind"],
        ["buffer", "task_kv_head"],
        ["parameter", "total_q"],
        ["parameter", "seqlen_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "output_scale"],
        ["parameter", "msa_max_pages"],
        ["parameter", "k_page_stride"],
        ["parameter", "k_head_stride"],
        ["parameter", "ks_page_stride"],
        ["parameter", "ks_head_stride"],
        ["parameter", "v_page_stride"],
        ["parameter", "v_head_stride"],
        ["parameter", "vs_page_stride"],
        ["parameter", "vs_head_stride"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


@functools.cache
def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?

    ``compute_107a`` (Rubin) needs a CUDA toolkit that lists it; a toolkit
    without it must decline the sm_107a builds up front instead of failing
    inside the JIT build, so a checkout that registers sm_107a programs stays
    importable and testable on a toolkit that only knows 10.0 / 10.3.
    """
    if arch not in ARCH_NVCC_FLAGS:
        return False
    if arch == "sm_107a":
        from flashinfer.compilation_context import _nvcc_supports_sm107

        return bool(_nvcc_supports_sm107())
    return True


def programs_for(arch: str) -> dict[str, dict[str, Any]]:
    """Registered programs whose source compiles for ``arch``."""
    return {
        name: record for name, record in PROGRAMS.items() if arch in record["arches"]
    }


def registered_split_factors(arch: str) -> tuple[int, ...]:
    """Split-KV factors with a persistent program for ``arch`` (plain launches)."""
    return tuple(
        sorted(
            int(record["splits"])
            for record in programs_for(arch).values()
            if record["route"] == "persistent" and not record["tail"]
        )
    )


def tail_programs(arch: str) -> list[tuple[str, dict[str, Any]]]:
    """Last-round-split programs of ``arch``, ascending split factor."""
    return sorted(
        (
            (name, record)
            for name, record in programs_for(arch).items()
            if record["route"] == "persistent" and record["tail"]
        ),
        key=lambda item: int(item[1]["splits"]),
    )


def select_program(arch: str, *, splits: int, tail: bool = False) -> str:
    """Name of the persistent program of ``arch`` for split factor ``splits``."""
    for name, record in programs_for(arch).items():
        if (
            record["route"] == "persistent"
            and int(record["splits"]) == int(splits)
            and bool(record["tail"]) == bool(tail)
        ):
            return name
    raise NotImplementedError(
        "The generated NVFP4 MSA decode program for "
        f"{arch} with split factor {splits}{' (last-round split)' if tail else ''} "
        "is not registered in this checkout"
    )


def short_program(arch: str) -> str | None:
    """Name of the short-item cluster program of ``arch``, or ``None``."""
    names = [
        name
        for name, record in programs_for(arch).items()
        if record["route"] == "short"
    ]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one short-item NVFP4 MSA decode program: {names}"
        )
    return names[0] if names else None


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
def gen_program_spec(name: str, arch: str):
    """JIT spec of program ``name`` compiled for ``arch`` (one library per pair)."""
    record = PROGRAMS[name]
    if arch not in record["arches"]:
        raise ValueError(f"program {name!r} is not registered for {arch}")
    if not toolchain_supports(arch):
        raise RuntimeError(
            f"generated NVFP4 MSA decode program {name!r} cannot be built for {arch}: "
            "the CUDA toolkit of this checkout cannot compile it (nvcc does not "
            "list compute_107a; a Rubin-capable toolkit is required)"
        )
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[arch], *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *{p.parent for p in sources}, *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_program(name: str, arch: str):
    return gen_program_spec(name, arch).build_and_load()
