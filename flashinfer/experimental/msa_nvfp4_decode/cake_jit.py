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
# count to the CTAs it co-schedules as ``splits``-CTA clusters), ``short``
# (one ``cluster``-CTA cluster per work item of at most ``max_pages`` pages, at
# most ``max_clusters`` clusters per launch) or ``union`` (the request-packed
# program of ``use_cache_hit=True``: ``union_q`` query tokens per request
# traced into one CTA, ``splits`` CTAs per (request, KV head) unit, ``cluster``
# true when ``splits > 1`` and the launch is one ``splits``-CTA cluster per
# unit).  ``ARG_PLANS`` lists the host binding's argument order once per
# route.  Both literals are written by the generated-program export; do not
# edit by hand.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_msa_nvfp4_decode_101dbbbac3c7c2b0929f": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 2,
        "splits": 1,
        "cluster": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_101dbbbac3c7c2b0929f_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_101dbbbac3c7c2b0929f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_21e04eaff519ea6cd53a": {
        "arches": ["sm_107a"],
        "route": "persistent",
        "splits": 2,
        "tail": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_21e04eaff519ea6cd53a_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_21e04eaff519ea6cd53a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
        "cluster_capacity": {"212": 212},
    },
    "cake_msa_nvfp4_decode_44bc4ca2b8b4088f3deb": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 4,
        "splits": 8,
        "cluster": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_44bc4ca2b8b4088f3deb_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_44bc4ca2b8b4088f3deb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_6618d950c63c1f5e1ebe": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "route": "persistent",
        "splits": 4,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_6618d950c63c1f5e1ebe_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_6618d950c63c1f5e1ebe_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_6c186c5235ad08b4ec9f": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "route": "short",
        "splits": 1,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_6c186c5235ad08b4ec9f_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_6c186c5235ad08b4ec9f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
    },
    "cake_msa_nvfp4_decode_a779981b54a3d1f15057": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 4,
        "splits": 2,
        "cluster": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_a779981b54a3d1f15057_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_a779981b54a3d1f15057_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_aa3e5180abff0afd4101": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "route": "persistent",
        "splits": 8,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_aa3e5180abff0afd4101_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_aa3e5180abff0afd4101_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_af2e689ccab0037b5cda": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 2,
        "splits": 4,
        "cluster": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_af2e689ccab0037b5cda_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_af2e689ccab0037b5cda_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_b1fa7590f47de655145a": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 2,
        "splits": 8,
        "cluster": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_b1fa7590f47de655145a_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_b1fa7590f47de655145a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_cb6eb3a9a43c7f43bab8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "route": "persistent",
        "splits": 1,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_cb6eb3a9a43c7f43bab8_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_cb6eb3a9a43c7f43bab8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
    "cake_msa_nvfp4_decode_e52248a9dd736f72904e": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 2,
        "splits": 2,
        "cluster": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_e52248a9dd736f72904e_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_e52248a9dd736f72904e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_e55acc6adbda3b342ca9": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 4,
        "splits": 4,
        "cluster": True,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_e55acc6adbda3b342ca9_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_e55acc6adbda3b342ca9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_e8432419536e3287998f": {
        "arches": ["sm_100a", "sm_103a"],
        "route": "union",
        "union_q": 4,
        "splits": 1,
        "cluster": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_e8432419536e3287998f_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_e8432419536e3287998f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
    },
    "cake_msa_nvfp4_decode_f789cce307344c958485": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "route": "persistent",
        "splits": 2,
        "tail": False,
        "sources": [
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_f789cce307344c958485_kernel.cu",
            "cake_msa_nvfp4_decode/cake_msa_nvfp4_decode_f789cce307344c958485_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ctas_per_sm": 1,
    },
}
ARG_PLANS: dict[str, list[list[str]]] = {
    "union": [
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
        ["parameter", "q2k_token_stride"],
        ["parameter", "q2k_head_stride"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
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


def union_programs(arch: str) -> dict[str, dict[str, Any]]:
    """Request-packed union programs of ``arch`` (``use_cache_hit=True``), by name."""
    return {
        name: record
        for name, record in programs_for(arch).items()
        if record["route"] == "union"
    }


def registered_union_pairs(arch: str) -> tuple[tuple[int, int], ...]:
    """``(union_q, splits)`` pairs with a union program for ``arch``, ascending."""
    return tuple(
        sorted(
            (int(record["union_q"]), int(record["splits"]))
            for record in union_programs(arch).values()
        )
    )


def select_union_program(arch: str, *, union_q: int, splits: int) -> str:
    """Name of the union program of ``arch`` for ``union_q`` query tokens and ``splits`` CTAs per unit."""
    for name, record in union_programs(arch).items():
        if int(record["union_q"]) == int(union_q) and int(record["splits"]) == int(
            splits
        ):
            return name
    raise NotImplementedError(
        "The generated request-packed NVFP4 MSA decode program for "
        f"{arch} with {union_q} query tokens per request and {splits} CTAs per unit "
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


_LOADED: dict[tuple[str, str], Any] = {}


def load_program(name: str, arch: str):
    """Build (once per process) and load program ``name`` for ``arch``."""
    key = (name, arch)
    module = _LOADED.get(key)
    if module is None:
        module = gen_program_spec(name, arch).build_and_load()
        _LOADED[key] = module
    return module


def program_loaded(name: str, arch: str) -> bool:
    """True once :func:`load_program` has built and loaded ``(name, arch)`` in this process.

    A first-time JIT build cannot happen inside a CUDA-graph capture; the
    backend's ``use_cache_hit=True`` arm checks this before binding under capture.
    """
    return (name, arch) in _LOADED
