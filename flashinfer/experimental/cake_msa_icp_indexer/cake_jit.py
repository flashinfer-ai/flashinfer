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

# Generated programs of the MiniMax-M3 ICP decode indexer chain, one record per
# delivered source pair.  A record lists the architectures its source compiles
# for (``arches``); the loader compiles it with the exact flag set of the device
# it runs on, so two architectures never share one cached library.  ``role`` is
# ``scorer`` (the fused FP8 index scorer with the in-epilogue top-16 of one
# rank's page fragments, keyed by ``query_len`` / ``rows_per_rank`` /
# ``page_size`` and carrying its tile configuration ``bpt`` / ``slots`` /
# ``stages`` and the resident CTAs per SM ``ctas_per_sm``) or ``merge`` (the
# per-row merge whose tail exchanges the top-16 lists over symmetric memory,
# keyed by ``world_size`` / ``warps`` / ``entries_per_thread``).  ``ARG_PLANS``
# lists the host binding's argument order once per role.  ``POLICY`` carries
# the host-side planning constants the backend mirrors (merge warp policy,
# pad encodings, production switches).  All three literals are written by the
# generated-program export; do not edit by hand.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_msa_icp_indexer_0526764aadf29021f835": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_0526764aadf29021f835_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_0526764aadf29021f835_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 8,
        "rows_per_rank": 32,
        "page_size": 128,
        "bpt": 2,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_1e761920da002ac94538": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_1e761920da002ac94538_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_1e761920da002ac94538_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 2,
        "warps": 4,
        "entries_per_thread": 6,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": True,
    },
    "cake_msa_icp_indexer_1efb99d6a16097468ed4": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_1efb99d6a16097468ed4_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_1efb99d6a16097468ed4_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 2,
        "rows_per_rank": 32,
        "page_size": 128,
        "bpt": 2,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_2ec5a89c6565dfcd7514": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_2ec5a89c6565dfcd7514_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_2ec5a89c6565dfcd7514_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 2,
        "warps": 8,
        "entries_per_thread": 6,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": True,
    },
    "cake_msa_icp_indexer_39d4ed41c00ea86fdd77": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_39d4ed41c00ea86fdd77_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_39d4ed41c00ea86fdd77_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 4,
        "warps": 2,
        "entries_per_thread": 4,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": False,
    },
    "cake_msa_icp_indexer_4a17d4a2448067c8f248": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_4a17d4a2448067c8f248_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_4a17d4a2448067c8f248_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 4,
        "warps": 8,
        "entries_per_thread": 10,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": False,
    },
    "cake_msa_icp_indexer_4f32c350f45f5f393a0b": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_4f32c350f45f5f393a0b_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_4f32c350f45f5f393a0b_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 2,
        "warps": 1,
        "entries_per_thread": 8,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": True,
    },
    "cake_msa_icp_indexer_56918d06c4ae67812752": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_56918d06c4ae67812752_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_56918d06c4ae67812752_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 3,
        "rows_per_rank": 32,
        "page_size": 128,
        "bpt": 2,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_5aaad7f513fe5e4e7fce": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_5aaad7f513fe5e4e7fce_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_5aaad7f513fe5e4e7fce_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 2,
        "warps": 8,
        "entries_per_thread": 10,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": True,
    },
    "cake_msa_icp_indexer_6c9de96daa7d9ac678d6": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_6c9de96daa7d9ac678d6_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_6c9de96daa7d9ac678d6_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 2,
        "warps": 1,
        "entries_per_thread": 2,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": True,
    },
    "cake_msa_icp_indexer_71ba5a628ebea112847c": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_71ba5a628ebea112847c_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_71ba5a628ebea112847c_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 2,
        "rows_per_rank": 64,
        "page_size": 128,
        "bpt": 1,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_7a747c58ca2af7bb6786": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_7a747c58ca2af7bb6786_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_7a747c58ca2af7bb6786_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 2,
        "warps": 1,
        "entries_per_thread": 4,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": True,
    },
    "cake_msa_icp_indexer_7e593093c70057f71290": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_7e593093c70057f71290_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_7e593093c70057f71290_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 4,
        "rows_per_rank": 32,
        "page_size": 128,
        "bpt": 2,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_a1d2b2946134a90f832e": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_a1d2b2946134a90f832e_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_a1d2b2946134a90f832e_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 1,
        "rows_per_rank": 128,
        "page_size": 256,
        "bpt": 1,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 1,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_a25ac9f6d1806c937555": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_a25ac9f6d1806c937555_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_a25ac9f6d1806c937555_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 4,
        "rows_per_rank": 64,
        "page_size": 128,
        "bpt": 1,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_a8101e4dbbf1639b3bbd": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_a8101e4dbbf1639b3bbd_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_a8101e4dbbf1639b3bbd_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 1,
        "rows_per_rank": 32,
        "page_size": 128,
        "bpt": 2,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_ac84ae929e6b0b76f4a4": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_ac84ae929e6b0b76f4a4_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_ac84ae929e6b0b76f4a4_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 4,
        "rows_per_rank": 128,
        "page_size": 256,
        "bpt": 1,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 1,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_cd167b915dd26c85797e": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_cd167b915dd26c85797e_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_cd167b915dd26c85797e_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "world_size": 4,
        "warps": 2,
        "entries_per_thread": 2,
        "late_wait": True,
        "small_fast": True,
        "reg_tail": False,
    },
    "cake_msa_icp_indexer_cd26681ee2375382ee29": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_cd26681ee2375382ee29_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_cd26681ee2375382ee29_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 1,
        "rows_per_rank": 64,
        "page_size": 128,
        "bpt": 1,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_d1e79de3128a4be7527e": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_d1e79de3128a4be7527e_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_d1e79de3128a4be7527e_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 3,
        "rows_per_rank": 64,
        "page_size": 128,
        "bpt": 1,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
    "cake_msa_icp_indexer_dc24f0e9db942be02542": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [
            "cake_msa_icp_indexer/cake_msa_icp_indexer_dc24f0e9db942be02542_kernel.cu",
            "cake_msa_icp_indexer/cake_msa_icp_indexer_dc24f0e9db942be02542_binding.cu",
        ],
        "compile_flags": [],
        "use_pdl": True,
        "query_len": 8,
        "rows_per_rank": 64,
        "page_size": 128,
        "bpt": 1,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": 2,
        "pdl_trigger": "epi",
    },
}
ARG_PLANS: dict[str, list[list[str]]] = {
    "scorer": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["buffer", "block_table"],
        ["buffer", "score"],
        ["buffer", "valid"],
        ["buffer", "seq_lens"],
        ["buffer", "query_start_loc"],
        ["buffer", "positions"],
        ["buffer", "active_rows"],
        ["parameter", "total_tokens"],
        ["parameter", "token_begin"],
        ["parameter", "token_count"],
        ["parameter", "request_begin"],
        ["parameter", "score_stride"],
        ["parameter", "block_table_stride"],
        ["parameter", "split_count"],
        ["parameter", "rank"],
        ["buffer", "forced_column"],
        ["buffer", "pool"],
        ["parameter", "pool_stride"],
        ["buffer", "row_meta"],
        ["buffer", "status"],
        ["buffer", "gen_ctr"],
        ["parameter", "x_region_stride"],
        ["parameter", "x_par_stride"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "merge": [
        ["buffer", "pool"],
        ["buffer", "row_meta"],
        ["buffer", "query_start_loc"],
        ["buffer", "candidates"],
        ["buffer", "status"],
        ["parameter", "pool_stride"],
        ["parameter", "request_count"],
        ["buffer", "forced_block"],
        ["buffer", "n_ordinary"],
        ["buffer", "out"],
        ["buffer", "gen_ctr"],
        ["buffer", "stamps"],
        ["parameter", "x_par_stride"],
        ["parameter", "rank"],
        ["parameter", "token_begin"],
        ["parameter", "timeout_ns"],
        ["parameter", "x_phase"],
        ["parameter", "pg_world"],
        ["parameter", "pg_rank"],
        ["parameter", "pg_flags"],
        ["buffer", "xpool"],
        ["parameter", "xpool_peers"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
POLICY: dict[str, Any] = {
    "topk": 16,
    "num_heads": 4,
    "head_dim": 128,
    "pad_key": 8388607,
    "pad_id": 2147483647,
    "merge_w_policy": [[256, 1], [512, 2], [1024, 4]],
    "merge_w_default": 8,
    "merge_epl_max": 10,
    "merge_w_max": 16,
    "merge_production": {
        "late_wait": True,
        "scorer_pdl_trigger": "epi",
        "small_fast": True,
        "reg_tail": True,
    },
    "status": {
        "nan_candidate": 1,
        "layout_overflow": 2,
        "no_progress": 4,
        "timeout": 8,
        "contract": 16,
    },
    "phases": {"push_poll": 0, "push_only": 1, "poll_only": 2},
}

SOURCE_PACKAGE = "cake_msa_icp_indexer"

ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def programs_for(arch: str) -> dict[str, dict[str, Any]]:
    """Registered programs whose source compiles for ``arch``."""
    return {
        name: record for name, record in PROGRAMS.items() if arch in record["arches"]
    }


def scorer_programs(arch: str) -> dict[str, dict[str, Any]]:
    return {
        name: record
        for name, record in programs_for(arch).items()
        if record["role"] == "scorer"
    }


def merge_programs(arch: str) -> dict[str, dict[str, Any]]:
    return {
        name: record
        for name, record in programs_for(arch).items()
        if record["role"] == "merge"
    }


def select_scorer_program(
    arch: str, *, query_len: int, rows_per_rank: int, page_size: int
) -> str:
    """Name of the fused scorer program of ``arch`` for one index geometry."""
    names = [
        name
        for name, record in scorer_programs(arch).items()
        if int(record["query_len"]) == int(query_len)
        and int(record["rows_per_rank"]) == int(rows_per_rank)
        and int(record["page_size"]) == int(page_size)
    ]
    if len(names) != 1:
        raise NotImplementedError(
            "The generated ICP indexer scorer program for "
            f"{arch} with query_len={query_len}, rows_per_rank={rows_per_rank}, "
            f"page_size={page_size} is not registered in this checkout"
            + (f" ({len(names)} candidates)" if names else "")
        )
    return names[0]


def select_merge_program(
    arch: str, *, world_size: int, warps: int, entries_per_thread: int
) -> str:
    """Name of the exchange-merge program of ``arch`` for one merge geometry."""
    names = [
        name
        for name, record in merge_programs(arch).items()
        if int(record["world_size"]) == int(world_size)
        and int(record["warps"]) == int(warps)
        and int(record["entries_per_thread"]) == int(entries_per_thread)
    ]
    if len(names) != 1:
        raise NotImplementedError(
            "The generated ICP indexer exchange-merge program for "
            f"{arch} with world_size={world_size}, warps={warps}, "
            f"entries_per_thread={entries_per_thread} is not registered in this checkout"
            + (f" ({len(names)} candidates)" if names else "")
        )
    return names[0]


def _csrc_root() -> Path:
    """The ``csrc`` root holding ``cake_msa_icp_indexer/`` (installed data or the checkout)."""
    installed = Path(jit_env.FLASHINFER_CSRC_DIR)
    if (installed / SOURCE_PACKAGE).is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[3] / "csrc"
    if (checkout / SOURCE_PACKAGE).is_dir():
        return checkout
    raise FileNotFoundError(
        f"generated ICP indexer sources ({SOURCE_PACKAGE}/) were not found under "
        f"{installed} or {checkout}"
    )


def _header_dirs() -> list[Path]:
    installed = [
        Path(jit_env.FLASHINFER_CSRC_DIR),
        Path(jit_env.FLASHINFER_INCLUDE_DIR),
    ]
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
    if arch not in ARCH_NVCC_FLAGS:
        raise ValueError(f"no nvcc flag set for {arch}")
    root = _csrc_root()
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
