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
# delivered program.  A record lists the architectures its source compiles
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
# pad encodings, production switches).  ``PROGRAMS`` is expanded from one table
# row per record (the record hash, its host-shim namespace and its key) plus
# the per-family constants; ``sources`` and ``compile_flags`` follow from the
# role and the key: the scorer records of one ``rows_per_rank`` family share
# one kernel text (the family's ``query_len`` 1 export), every record builds
# its role's host binding, and the flags select the record's kernel symbol,
# host-shim namespace and launch constants.  The tables, ``ARG_PLANS`` and
# ``POLICY`` are written by the generated-program export; do not edit by hand.
# The shared sources and the tabular form are a hand fold of the exported
# literals pending their regeneration by the export.

# Scorer rows: (record hash, host-shim namespace suffix, query_len,
# rows_per_rank).
_SCORER_ROWS: tuple[tuple[str, str, int, int], ...] = (
    ("0526764aadf29021f835", "46b468ca2078abe4", 8, 32),
    ("1efb99d6a16097468ed4", "61a3d4f8952519f6", 2, 32),
    ("56918d06c4ae67812752", "deb8d69a66fbf462", 3, 32),
    ("71ba5a628ebea112847c", "72403d0537dcc693", 2, 64),
    ("7e593093c70057f71290", "5f794fce523fb1cc", 4, 32),
    ("a1d2b2946134a90f832e", "2b60f8cc4e7f722f", 1, 128),
    ("a25ac9f6d1806c937555", "a2cc60e7cc2ed3a6", 4, 64),
    ("a8101e4dbbf1639b3bbd", "cd1c9ab33b97201e", 1, 32),
    ("ac84ae929e6b0b76f4a4", "befc48ed7922eb6a", 4, 128),
    ("cd26681ee2375382ee29", "ecad07f07917fb3b", 1, 64),
    ("d1e79de3128a4be7527e", "f1fda020885d0261", 3, 64),
    ("dc24f0e9db942be02542", "d748cc06c6edb174", 8, 64),
)
# rows_per_rank -> (shared kernel text: the family's query_len 1 export,
# page_size, bpt, ctas_per_sm, dynamic shared memory bytes of the launch).
_SCORER_FAMILIES: dict[int, tuple[str, int, int, int, int]] = {
    32: ("a8101e4dbbf1639b3bbd", 128, 2, 2, 66560),
    64: ("cd26681ee2375382ee29", 128, 1, 2, 66560),
    128: ("a1d2b2946134a90f832e", 256, 1, 1, 115712),
}
# Merge rows: (record hash, host-shim namespace suffix, world_size, warps,
# entries_per_thread); each merge record builds its own kernel text.
_MERGE_ROWS: tuple[tuple[str, str, int, int, int], ...] = (
    ("133176f530b7352070c3", "2d7cf17183503496", 2, 2, 8),
    ("177f6799f01a652ddd21", "58e050de77950599", 2, 4, 8),
    ("17f1f4efee096a75db68", "83a03baeada0253c", 2, 1, 6),
    ("1e761920da002ac94538", "7d7d86ce82514c6a", 2, 4, 6),
    ("2ec5a89c6565dfcd7514", "94d730f261c73513", 2, 8, 6),
    ("39d4ed41c00ea86fdd77", "5c72164dfad5d8d4", 4, 2, 4),
    ("4a17d4a2448067c8f248", "ea1d2723a86d61cf", 4, 8, 10),
    ("4f32c350f45f5f393a0b", "653f3a7eafcaaf38", 2, 1, 8),
    ("5aaad7f513fe5e4e7fce", "bee55e82844155f1", 2, 8, 10),
    ("6c9de96daa7d9ac678d6", "dcc1237482a5bc01", 2, 1, 2),
    ("7a747c58ca2af7bb6786", "5fe18fd647da4532", 2, 1, 4),
    ("9c8c1dc86f1b2c21e293", "67f0b3173822d417", 2, 2, 6),
    ("cd167b915dd26c85797e", "921aa87847d33a7f", 4, 2, 2),
    ("f76a34922ef374f481e6", "d1cf9c1e048be3f7", 2, 8, 8),
)
# warps -> dynamic shared memory bytes of the merge launch (block threads are
# ``32 * warps``).
_MERGE_SMEM: dict[int, int] = {1: 768, 2: 1280, 4: 2304, 8: 4352}
# The host binding every record of a role builds.
_SCORER_BINDING = "a8101e4dbbf1639b3bbd"
_MERGE_BINDING = "6c9de96daa7d9ac678d6"


def _source(digest: str, kind: str) -> str:
    return f"cake_msa_icp_indexer/cake_msa_icp_indexer_{digest}_{kind}.cu"


def _scorer_record(
    digest: str, shim: str, query_len: int, rows_per_rank: int
) -> tuple[str, dict[str, Any]]:
    kernel, page_size, bpt, ctas_per_sm, smem = _SCORER_FAMILIES[rows_per_rank]
    name = f"cake_msa_icp_indexer_{digest}"
    record: dict[str, Any] = {
        "arches": ["sm_100a", "sm_103a"],
        "role": "scorer",
        "sources": [_source(kernel, "kernel"), _source(_SCORER_BINDING, "binding")],
        "compile_flags": [
            f"-DBLOCK_Q={4 * query_len}",
            f"-DCAKE_PROGRAM_KERNEL=kernel_{name}",
            f"-DCAKE_PROGRAM_SHIM=cake_host_shim_{shim}",
            f"-DCAKE_BIND_KBOX_ROWS={rows_per_rank}",
            f"-DCAKE_BIND_SMEM={smem}",
        ],
        "query_len": query_len,
        "rows_per_rank": rows_per_rank,
        "page_size": page_size,
        "bpt": bpt,
        "slots": 4,
        "stages": 6,
        "ctas_per_sm": ctas_per_sm,
        "pdl_trigger": "epi",
    }
    return name, record


def _merge_record(
    digest: str, shim: str, world_size: int, warps: int, entries_per_thread: int
) -> tuple[str, dict[str, Any]]:
    name = f"cake_msa_icp_indexer_{digest}"
    record: dict[str, Any] = {
        "arches": ["sm_100a", "sm_103a"],
        "role": "merge",
        "sources": [_source(digest, "kernel"), _source(_MERGE_BINDING, "binding")],
        "compile_flags": [
            f"-DCAKE_PROGRAM_KERNEL=kernel_{name}",
            f"-DCAKE_PROGRAM_SHIM=cake_host_shim_{shim}",
            f"-DCAKE_BIND_THREADS={32 * warps}",
            f"-DCAKE_BIND_SMEM={_MERGE_SMEM[warps]}",
            f"-DCAKE_BIND_WORLD={world_size}",
        ],
        "world_size": world_size,
        "warps": warps,
        "entries_per_thread": entries_per_thread,
        "late_wait": True,
        "small_fast": True,
    }
    return name, record


PROGRAMS: dict[str, dict[str, Any]] = dict(
    sorted(
        [_scorer_record(*row) for row in _SCORER_ROWS]
        + [_merge_record(*row) for row in _MERGE_ROWS],
        key=lambda item: item[0],
    )
)
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
    },
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
