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

Registry and loader glue of the Cake QSA sparse paged GQA decode over an
NVFP4 KV cache for SM80, SM90, SM100, SM103, SM107 and SM120: two split-KV decode
programs partitioning the architectures -- the ``mma.sync`` decode of ``sm_80``,
``sm_90a`` and ``sm_120a`` (its plain K-first build, its whole-tile build delivered for
``sm_80`` and ``sm_120a``, and its cooperative build, which merges the splits in the
same launch) and the tcgen05 decode of ``sm_100a``, ``sm_103a`` and ``sm_107a``
(its plain and its cooperative build; tensor memory, one CTA per SM; and its
persistent-walk build delivered for ``sm_103a`` and ``sm_107a``) -- and the separate
split-KV merge with the sigmoid output gate, delivered for every architecture.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled for each architecture
of the record's ``arches`` (one source per program; ``sm_80``, ``sm_90a`` and
``sm_120a`` for the ``mma.sync`` decodes, ``sm_100a``, ``sm_103a`` and ``sm_107a`` for the
tcgen05 decodes, all six for the merge, ``sm_80`` and ``sm_120a`` for the
whole-tile decode, ``sm_103a`` and ``sm_107a`` for the persistent walk).  A record
carries the host plan only: the program's ``library`` stem (its ``build.json``
entry is ``<library>_<arch>``), the architectures it is delivered for, the
``stage`` of the launch order it fills and its ``variant``, the
``select_rule`` that picks the records of one call from host-known scalars,
the ``grid_rule`` that shapes its launch, the ``workspace_rule`` naming the
scratch it writes, the kernel-parameter -> logical-argument map (the
identity: a logical name is the kernel parameter's name) and the argument
plan of its ``run`` entry.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library from a prebuilt ``aot/`` file when a
builder compiled it and otherwise through FlashInfer's JIT from the same
sources and flags.  Every ``build.json`` entry records ``--use_fast_math``
(the flag the kernels were validated and measured with).

The rules are data, evaluated here (:func:`select_records`,
:func:`launch_grid`) from host-known scalars: the call's *form* (``"split"``:
the plain decode, then the merge when the call has more than one split;
``"fused"``: the cooperative decode alone, its CTAs merging the splits
through a counter per (row, KV head)), the device's exported architecture
``arch`` and whether the call is *wide* at the architecture's grid bound (the
whole-tile record's ``grid`` map: on ``sm_80`` a multi-wave, long-stream grid
-- above one resident wave, ``DECODE_CTAS_PER_SM`` CTAs per SM times the SM
count, with every CTA walking at least ``DECODE_LONG_STREAM_TILES`` tiles; on
``sm_120a`` a long stream alone, at least ``DECODE_STREAM_TILES`` tiles per
CTA whatever the grid), the CTA count ``walk`` of the *persistent walk* when
the host serves the call with it (``0`` otherwise: a two-wave split-form grid
on ``sm_103a`` whose longest walked CTA is shorter than the plain program's,
the host's :func:`.cake_backend.select_walk`), the row count ``rows``, the KV
head count ``kv_heads``, the query head count ``qo_heads`` and the split count
``splits`` (the walk's partial slots per (row, KV head), :func:`walk_splits`,
where the walk serves the call).  Only the records delivered for ``arch``
(``record["arches"]``) are candidates: each decode program has a ``form``
record per form.  A selection rule of kind ``form`` admits its form on every
architecture its record ships for; one of kind ``form_grid`` admits its form
on the architectures its ``grid`` map names, each at the grid bound the map
gives it, and wins over a ``form`` rule when both admit (the whole-tile
decode: the split form on ``sm_80`` / ``sm_120a`` on a wide call; the
persistent walk: the split form on ``sm_103a`` / ``sm_107a`` on a walked
grid).  No rule
reads a tensor.  Which form a call takes, its split count, whether the call
is wide and whether the walk serves it are the host's decisions
(:mod:`.cake_backend`), not rules.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
import operator
from typing import Any, Callable

import tvm_ffi

# The five verified builds, each compiled with its exact ``-gencode`` flag set
# (``build.json`` ``arch_flags``): ``sm_80`` serves every compute capability
# 8.x device, ``sm_90a`` the H100 (9.0), ``sm_100a`` the B200 / GB200 (10.0),
# ``sm_103a`` the B300 / GB300 (10.3), ``sm_107a`` the R200 (10.7) and ``sm_120a``
# the RTX PRO 6000 Blackwell (12.0: no tcgen05 instructions and no tensor memory,
# so the ``mma.sync`` decode serves it, the SM80 program built for sm_120a).  No
# other capability is a target: the table is capability-exact (12.1 is not in
# it).  A record lists the subset it is delivered for (``record["arches"]``).
ARCHES = ("sm_80", "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (8, 0): "sm_80",
    (8, 6): "sm_80",
    (8, 7): "sm_80",
    (8, 9): "sm_80",
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
    (12, 0): "sm_120a",
}
PREFIX = "cake_qsa_nvfp4"

# The launch order of a call: the decode (one of the architecture's builds) and,
# in the split form with more than one split, the merge behind it.
STAGES = ("decode", "merge")
# The two forms a call takes (``select_rule`` value of kinds ``form`` and
# ``form_grid``).
FORMS = ("split", "fused")
# The selection rule kinds, least specific first: ``form`` admits its ``value``
# form on every architecture its record ships for; ``form_grid`` admits its
# ``value`` form on the architectures its ``grid`` map names alone, each only at
# the grid bound the map gives it (``GRID_BOUNDS``: ``"multi_wave_long"`` is a
# decode grid above one resident wave whose CTAs each walk at least
# ``DECODE_LONG_STREAM_TILES`` tiles, the host's ``wide``; ``"long_stream"`` is
# CTAs each walking at least ``DECODE_STREAM_TILES`` tiles whatever the grid,
# the host's ``wide`` too; ``"two_wave_walk"`` is a split-form grid the
# persistent walk serves, the host's ``walk > 0``).  When both admit, the more
# specific wins.
SELECT_RULES = ("form", "form_grid")
GRID_BOUNDS = ("multi_wave_long", "long_stream", "two_wave_walk")
# The merge's split bound: a (row, head) has at most this many partials (one
# per lane of the merge's four warps).  The host's split rule never exceeds it.
MAX_SPLITS = 128
# The merge's warps per CTA: above this many splits one CTA merges one head
# with every warp owning a chunk of splits; at or below, one warp per head and
# this many heads per CTA (``grid_rule`` kind ``rows_head_groups``).
MERGE_WARPS = 4
# The decode's tile: the selection is split into tiles of this many tokens.
TILE_TOKENS = 64
# Per architecture: the block of its decode (128 threads for the ``mma.sync``
# decode, 512 for the tcgen05 one), the resident CTAs per SM the split rule
# counts on (the plain decode's launch bound: two for the ``mma.sync`` decode,
# one for the tcgen05 decode, whose tensor memory and shared-memory pool hold
# the SM) and the fused form's capacity per SM, the co-resident build's
# launch bound: the form rule takes the fused form only below
# ``DECODE_FUSED_CTAS_PER_SM[arch]`` x the SM count CTAs (``sm_80`` is absent:
# its calls take the split form alone).  Within the capacity sm_90a always
# takes the fused form and sm_120a does except on a wide call (the whole-tile
# program's long-stream window, which ships in the split form alone); sm_100a /
# sm_103a take it under the host's eager rule (``cake_backend.select_form``: a
# host of ``FUSED_HOSTS``, rows <= 4, no capture -- and that list is empty, so
# they keep the split form on every host), sm_107a never (the split form on
# every host).  The
# exporter holds the launch-bound tables to the kernel files and the fused
# table to its host policy.
DECODE_THREADS = {
    "sm_80": 128,
    "sm_90a": 128,
    "sm_100a": 512,
    "sm_103a": 512,
    "sm_107a": 512,
    "sm_120a": 128,
}
DECODE_CTAS_PER_SM = {
    "sm_80": 2,
    "sm_90a": 2,
    "sm_100a": 1,
    "sm_103a": 1,
    "sm_107a": 1,
    "sm_120a": 2,
}
DECODE_FUSED_CTAS_PER_SM = {
    "sm_90a": 2,
    "sm_100a": 1,
    "sm_103a": 1,
    "sm_107a": 1,
    "sm_120a": 2,
}
# The long-stream bound of the ``multi_wave_long`` grid class: the per-CTA tile
# stream, ``ceil(tiles / splits)``, from which the whole-tile program measured
# faster than the K-first one above one resident wave; 16 tiles = 1024 selected
# tokens (the 128-row shapes at 17 tiles per CTA favoured it, the 64-row shapes
# at 9 did not).
DECODE_LONG_STREAM_TILES = 16
# The stream bound of the ``long_stream`` grid class (``sm_120a``): the per-CTA
# tile stream from which the whole-tile program with the L2 evict-first policy
# measured faster than the K-first one whatever the grid (an RTX PRO 6000 holds
# every owner grid within one wave of its 376 resident CTAs); 11 tiles = 704
# selected tokens (rows 32 / 64 / 128 at 11 / 17 / 33 tiles per CTA favoured it
# under both L2 flushes, rows 16 at 7 tiles did not).
DECODE_STREAM_TILES = 11
# The persistent walk's wave bound per architecture (the kernel file's
# ``WALK_MAX_WAVES`` table, held by the exporter): a split-form call whose plain
# decode grid runs more than one resident wave and at most this many, and whose
# longest walked CTA is shorter than the plain program's longest
# (:func:`walk_applies`), launches the persistent walk (``select_rule`` kind
# ``form_grid``, grid ``two_wave_walk``) on :func:`walk_ctas` CTAs with
# :func:`walk_splits` partial slots per (row, KV head) instead of the plain split
# count.  The classes the paired measurements cover: two waves on ``sm_103a``
# (rows 128 of the owner's shapes on 148 / 152 SMs), three on ``sm_107a`` (rows
# 128 and 256 on an R200's 212 SMs).
WALK_MAX_WAVES = {"sm_103a": 2, "sm_107a": 3}

# Logical argument names the host backend binds: the kernels' own parameter
# names (``record["logical"]`` is the identity map).  Decode: ``q`` BF16
# ``[rows, qo_heads, 256]`` (strided), ``k_data`` / ``v_data`` uint8
# ``[blocks, page, kv_heads, 128]`` packed E2M1 (any strides with a contiguous
# last dim), ``k_sf`` / ``v_sf`` their E4M3 block scales as uint8 views
# ``[blocks, page, kv_heads, 16]``, ``indices`` int32 ``[rows, topk + 1]``,
# ``block_table`` int32 ``[requests, table_width]``, ``token_to_req`` int32
# ``[rows]``, ``gate`` BF16 ``[rows, qo_heads, 256]`` (dense), ``out`` BF16
# ``[rows, qo_heads, 256]`` (strided), ``partial_out`` FP32 ``[splits, rows,
# qo_heads, 256]`` and ``partial_lse`` FP32 ``[splits, rows, qo_heads]``
# (``None`` at one split), ``merge_counters`` uint32 ``[rows * kv_heads]``
# (the fused build's group counters; ``None`` for the plain build).  Merge:
# ``partial_out`` / ``partial_lse`` / ``gate`` / ``out`` as above.
LOGICAL_TENSORS = (
    "q",
    "k_data",
    "v_data",
    "k_sf",
    "v_sf",
    "indices",
    "block_table",
    "token_to_req",
    "gate",
    "out",
    "partial_out",
    "partial_lse",
    "merge_counters",
)
LOGICAL_SCALARS = (
    "q_stride_row",
    "q_stride_head",
    "data_stride_block",
    "data_stride_token",
    "data_stride_head",
    "sf_stride_block",
    "sf_stride_token",
    "sf_stride_head",
    "indices_stride_row",
    "table_stride_req",
    "out_stride_row",
    "out_stride_head",
    "score_scale",
    "output_scale",
    "num_rows",
    "num_requests",
    "num_blocks",
    "page_div",
    "table_width",
    "topk",
    "group_size",
    "num_qo_heads",
    "num_tiles",
    "num_splits",
)
# Position of each logical argument in the flat sequence the host's fast path
# binds, ``LOGICAL_TENSORS`` then ``LOGICAL_SCALARS`` (:func:`resolved_launches`).
# Derived, not a literal: the exporter and the manifest checker skip it.
LOGICAL_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(LOGICAL_TENSORS + LOGICAL_SCALARS)
}

# The kernel parameter names of each program, in parameter order: the identity
# ``logical`` map of its record and the order its ``arg_plan`` follows.  The
# decode's plain and cooperative builds take the same parameters.
DECODE_PARAMETERS = (
    "q",
    "k_data",
    "v_data",
    "k_sf",
    "v_sf",
    "indices",
    "block_table",
    "token_to_req",
    "gate",
    "out",
    "partial_out",
    "partial_lse",
    "merge_counters",
    "q_stride_row",
    "q_stride_head",
    "data_stride_block",
    "data_stride_token",
    "data_stride_head",
    "sf_stride_block",
    "sf_stride_token",
    "sf_stride_head",
    "indices_stride_row",
    "table_stride_req",
    "out_stride_row",
    "out_stride_head",
    "score_scale",
    "output_scale",
    "num_rows",
    "num_requests",
    "num_blocks",
    "page_div",
    "table_width",
    "topk",
    "group_size",
    "num_qo_heads",
    "num_tiles",
    "num_splits",
)
MERGE_PARAMETERS = (
    "partial_out",
    "partial_lse",
    "gate",
    "out",
    "out_stride_row",
    "out_stride_head",
    "num_splits",
    "num_qo_heads",
)

# Filled mechanically by the exporter (the derived leaves ``kernel_symbol``,
# ``launch``, ``arg_plan``, ``logical``); the host plan (``stage``, ``variant``,
# ``select_rule``, ``grid_rule``, ``workspace_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal:
# FlashInfer's manifest checker and the exporter read this file with
# ``ast.literal_eval``.  Record names are the stages, suffixed with the variant
# where a stage has several programs.  The ``mma.sync`` decode's K-first builds
# (``decode``, ``decode_coop``) and its whole-tile build (``decode_whole_tile``)
# differ in the program text alone: same parameters, launch facts, grid and
# workspace rules.  The tcgen05 decode's builds (``decode_tcgen05``,
# ``decode_tcgen05_coop``) take the same parameters, grid and workspace rules
# with their own block (512 threads) and shared-memory pool.
MODULES: dict[str, dict[str, Any]] = {
    "decode": {
        "arches": ["sm_80", "sm_90a", "sm_120a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_kv_heads_splits"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_f0ae7070812a3e062d6f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 49792,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_f0ae7070812a3e062d6f",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {"kind": "form", "value": "split"},
        "stage": "decode",
        "variant": "plain",
        "workspace_rule": {"kind": "split_partials"},
    },
    "decode_whole_tile": {
        "arches": ["sm_80", "sm_120a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_kv_heads_splits"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_ecb46ae2b040daa1c35d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 49792,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_ecb46ae2b040daa1c35d",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {
            "grid": {"sm_80": "multi_wave_long", "sm_120a": "long_stream"},
            "kind": "form_grid",
            "value": "split",
        },
        "stage": "decode",
        "variant": "whole_tile",
        "workspace_rule": {"kind": "split_partials"},
    },
    "decode_coop": {
        "arches": ["sm_80", "sm_90a", "sm_120a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_kv_heads_splits"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_4f13e4710a51ac27ff6f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 49792,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_4f13e4710a51ac27ff6f",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {"kind": "form", "value": "fused"},
        "stage": "decode",
        "variant": "coop",
        "workspace_rule": {"kind": "fused_partials_counters"},
    },
    "decode_tcgen05": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_kv_heads_splits"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_6692f8122124ee26fba0",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 159872,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_6692f8122124ee26fba0",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {"kind": "form", "value": "split"},
        "stage": "decode",
        "variant": "tcgen05",
        "workspace_rule": {"kind": "split_partials"},
    },
    "decode_tcgen05_coop": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_kv_heads_splits"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_b56eb690df0312fb5df3",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 159872,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_b56eb690df0312fb5df3",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {"kind": "form", "value": "fused"},
        "stage": "decode",
        "variant": "tcgen05_coop",
        "workspace_rule": {"kind": "fused_partials_counters"},
    },
    "decode_tcgen05_persistent": {
        "arches": ["sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "walk_ctas"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_dfd8600d03040c9f808d",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 192896,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_dfd8600d03040c9f808d",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {
            "grid": {"sm_103a": "two_wave_walk", "sm_107a": "two_wave_walk"},
            "kind": "form_grid",
            "value": "split",
        },
        "stage": "decode",
        "variant": "tcgen05_persistent",
        "workspace_rule": {"kind": "split_partials"},
    },
    "merge": {
        "arches": ["sm_80", "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a"],
        "arg_plan": [
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "num_splits"],
            ["parameter", "num_qo_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_head_groups", "warps": 4},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_aa2349948b0a711974f9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_aa2349948b0a711974f9",
        "logical": {
            "gate": "gate",
            "num_qo_heads": "num_qo_heads",
            "num_splits": "num_splits",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
        },
        "select_rule": {"kind": "form", "value": "split"},
        "stage": "merge",
        "variant": None,
        "workspace_rule": None,
    },
}


def records_of_stage(stage: str) -> tuple[str, ...]:
    """The record names of ``stage`` in registry order."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {STAGES}")
    return tuple(name for name, rec in MODULES.items() if rec["stage"] == stage)


def _admits(
    rule: dict[str, Any], *, form: str, arch: str, wide: bool, walk: int = 0
) -> bool:
    """Whether a record's selection rule admits the call; the caller has
    already kept the records delivered for ``arch`` alone.  A ``form_grid``
    rule reads the grid bound its ``grid`` map gives ``arch`` (none: the rule
    does not admit): ``multi_wave_long`` and ``long_stream`` are the host's
    ``wide``, ``two_wave_walk`` the host's ``walk > 0``."""
    kind = rule["kind"]
    if kind == "form":
        return rule["value"] == form
    if kind == "form_grid":
        # ``grid`` maps each architecture the rule refines to its grid bound: the
        # host's ``wide`` is the verdict of ``multi_wave_long`` / ``long_stream``
        # for the device's architecture, ``walk > 0`` that of ``two_wave_walk``.
        if rule["value"] != form:
            return False
        bound = rule["grid"].get(arch)
        if bound is None:
            return False
        if bound in ("multi_wave_long", "long_stream"):
            return bool(wide)
        if bound == "two_wave_walk":
            return int(walk) > 0
        raise ValueError(
            f"unknown Cake QSA NVFP4 decode grid class {bound!r}; the classes are {GRID_BOUNDS}"
        )
    raise ValueError(f"unknown Cake QSA NVFP4 decode selection rule: {kind!r}")


def select_record(
    stage: str, *, form: str, arch: str, wide: bool, walk: int = 0
) -> str:
    """The one record of ``stage`` the call selects.

    ``form`` is ``"split"`` (the plain decode and, above one split, the merge)
    or ``"fused"`` (the cooperative decode, which merges its splits itself);
    ``arch`` is the device's exported architecture, ``wide`` whether the
    call is wide at the architecture's grid bound -- on ``sm_80`` above one
    resident wave with at least ``DECODE_LONG_STREAM_TILES`` tiles per CTA, on
    ``sm_120a`` at least ``DECODE_STREAM_TILES`` tiles per CTA -- and ``walk``
    the CTA count of the persistent walk when the host serves the call with
    it, ``0`` otherwise (all three the host's: :mod:`.cake_backend`).
    Only the records delivered for ``arch`` are candidates (the ``mma.sync``
    decode's on ``sm_80`` / ``sm_90a`` / ``sm_120a``, the tcgen05 decode's on
    ``sm_100a`` / ``sm_103a`` / ``sm_107a``); of those whose rule admits the call, the most specific rule
    kind (``SELECT_RULES`` order) wins: on ``sm_80`` and ``sm_120a`` a wide
    split-form call selects the whole-tile decode over the K-first one, on
    ``sm_103a`` / ``sm_107a`` a walked split-form call the persistent walk over
    the plain tcgen05 decode.  A grid class no record of the architecture reads falls
    through to its ``form`` record (``wide`` on ``sm_90a``, ``walk`` on
    ``sm_100a``; :func:`resolved_launches` refuses a walk the selected decode
    cannot launch).  The ``merge`` stage has no
    record in the fused form (:func:`select_records` leaves it out); asking for
    one is an error.  Exactly one record remains, or the registry is
    inconsistent.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; the forms are {FORMS}")
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; the architectures are {ARCHES}")
    if int(walk) < 0:
        raise ValueError(
            f"walk must be 0 (no walk) or the walk's CTA count, got {walk}"
        )
    admitted = [
        name
        for name in records_of_stage(stage)
        if arch in MODULES[name]["arches"]
        and _admits(
            MODULES[name]["select_rule"], form=form, arch=arch, wide=wide, walk=walk
        )
    ]
    if admitted:
        top = max(
            SELECT_RULES.index(MODULES[n]["select_rule"]["kind"]) for n in admitted
        )
        admitted = [
            n
            for n in admitted
            if SELECT_RULES.index(MODULES[n]["select_rule"]["kind"]) == top
        ]
    if len(admitted) != 1:
        raise ValueError(
            f"stage {stage!r}: records {admitted} admit form={form!r}, arch={arch!r}, "
            f"wide={bool(wide)}, walk={int(walk)}; exactly one must"
        )
    return admitted[0]


def select_records(
    *, form: str, splits: int, arch: str, wide: bool, walk: int = 0
) -> tuple[str, ...]:
    """The records one call launches, in launch order: the decode of ``form``
    on ``arch`` at the grid classes ``wide`` / ``walk`` and, in the split form
    with ``splits > 1``, the merge (``splits`` is the call's: the walk's slots
    where the walk serves it)."""
    if int(splits) < 1:
        raise ValueError(f"splits must be at least 1, got {splits}")
    names = [select_record("decode", form=form, arch=arch, wide=wide, walk=walk)]
    if form == "split" and int(splits) > 1:
        names.append(select_record("merge", form=form, arch=arch, wide=wide, walk=walk))
    return tuple(names)


def merge_heads_per_cta(splits: int) -> int:
    """The heads one merge CTA serves: one above ``MERGE_WARPS`` splits (a warp
    per chunk of splits), ``MERGE_WARPS`` at or below (a warp per head)."""
    return 1 if int(splits) > MERGE_WARPS else MERGE_WARPS


def launch_grid(
    name: str,
    *,
    rows: int,
    kv_heads: int,
    qo_heads: int,
    splits: int,
    walk: int = 0,
) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its recorded grid rule.

    ``rows_kv_heads_splits`` launches ``(rows, kv_heads, splits)`` (the decode:
    one CTA per (row, KV head, split)); ``rows_head_groups`` launches ``(rows,
    ceil(qo_heads / merge_heads_per_cta(splits)), 1)`` (the merge);
    ``walk_ctas`` launches ``(walk, 1, 1)`` (the persistent walk: the host's
    CTA count, at least one).
    """
    rule = MODULES[name]["grid_rule"]
    kind = rule["kind"]
    if int(rows) < 1 or int(kv_heads) < 1 or int(qo_heads) < 1 or int(splits) < 1:
        raise ValueError(
            f"record {name}: rows={rows}, kv_heads={kv_heads}, qo_heads={qo_heads}, "
            f"splits={splits} must all be at least one"
        )
    if kind == "rows_kv_heads_splits":
        return (int(rows), int(kv_heads), int(splits))
    if kind == "walk_ctas":
        if int(walk) < 1:
            raise ValueError(
                f"record {name}: the persistent walk needs its CTA count (walk={walk})"
            )
        return (int(walk), 1, 1)
    if kind == "rows_head_groups":
        if int(rule["warps"]) != MERGE_WARPS:
            raise ValueError(
                f"record {name}: grid rule warps {rule['warps']} is not the merge's {MERGE_WARPS}"
            )
        per = merge_heads_per_cta(splits)
        return (int(rows), -(-int(qo_heads) // per), 1)
    raise ValueError(f"unknown Cake QSA NVFP4 decode grid rule: {kind!r}")


# ---- The persistent walk's arithmetic: pure integers, mirrors of the ``sm_103a`` kernel file's ``walk_ctas`` /
# ``walk_splits`` / ``walk_applies`` (the kernel's slot math: slot ``s = pair x tiles + tile`` over ``pairs = rows x
# kv_heads``; CTA ``c`` of ``G`` walks the slots ``[c S // G, (c + 1) S // G)`` with ``S = pairs x tiles``; the CTA
# holding slot ``s`` is ``((s + 1) G - 1) // S``).  The host policy that calls them is :mod:`.cake_backend`.


def walk_ctas(sm_count: int, pairs: int, tiles: int) -> int:
    """``G = min(sm_count, pairs x tiles)``: one CTA per SM, never more CTAs
    than tile slots."""
    if int(sm_count) < 1 or int(pairs) < 1 or int(tiles) < 1:
        raise ValueError(
            f"sm_count={sm_count}, pairs={pairs}, tiles={tiles} must all be at least 1"
        )
    return min(int(sm_count), int(pairs) * int(tiles))


def _slot_owner(slot: int, ctas: int, slots: int) -> int:
    return ((slot + 1) * ctas - 1) // slots


def walk_splits(ctas: int, pairs: int, tiles: int) -> int:
    """The partial slots per (row, KV head) the persistent walk needs as its
    ``num_splits``: the most segments any pair is cut into by ``ctas`` CTAs
    walking the ``pairs x tiles`` slots in contiguous ranges (exact over the
    pairs; ``1`` when every pair sits inside one CTA, and the kernel then
    writes the output itself)."""
    slots = int(pairs) * int(tiles)
    if not 1 <= int(ctas) <= slots:
        raise ValueError(f"ctas={ctas} is not between 1 and the {slots} tile slots")
    return max(
        _slot_owner(p * int(tiles) + int(tiles) - 1, int(ctas), slots)
        - _slot_owner(p * int(tiles), int(ctas), slots)
        + 1
        for p in range(int(pairs))
    )


def walk_applies(
    rows: int, kv_heads: int, tiles: int, splits: int, sm_count: int, arch: str
) -> bool:
    """Whether the walk shortens a split-form call on an ``arch`` device (the
    kernel file's rule): the plain program's grid ``rows x kv_heads x splits``
    at the plain split count ``splits`` runs more than one resident wave of
    ``sm_count`` CTAs (``waves = ceil(rows x kv_heads x splits / sm_count) >
    1``), at most ``WALK_MAX_WAVES[arch]`` (the architecture's measured wave
    range), and the walk's longest CTA walks fewer tiles than the plain
    program's longest, ``ceil(pairs x tiles / G) < waves x ceil(tiles /
    splits)``.  Pure in its integers; whether the architecture ships the walk
    is the host's (:func:`.cake_backend.select_walk`); an architecture outside
    ``WALK_MAX_WAVES`` is refused."""
    if arch not in WALK_MAX_WAVES:
        raise ValueError(
            f"the persistent walk has no wave bound for {arch!r}; it ships for "
            f"{tuple(WALK_MAX_WAVES)}"
        )
    if (
        int(rows) < 1
        or int(kv_heads) < 1
        or int(tiles) < 1
        or int(splits) < 1
        or int(sm_count) < 1
    ):
        raise ValueError(
            f"rows={rows}, kv_heads={kv_heads}, tiles={tiles}, splits={splits}, "
            f"sm_count={sm_count} must all be at least 1"
        )
    pairs = int(rows) * int(kv_heads)
    ctas = walk_ctas(sm_count, pairs, tiles)
    longest_walk = -(-(pairs * int(tiles)) // ctas)
    waves = -(-(pairs * int(splits)) // int(sm_count))
    if waves <= 1 or waves > WALK_MAX_WAVES[arch]:
        return False
    return longest_walk < waves * -(-int(tiles) // int(splits))


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_capability(device_index: int) -> tuple[int, int]:
    """Compute capability of one CUDA device (immutable per process; queried once
    per device index, never per launch)."""
    import torch

    major, minor = torch.cuda.get_device_capability(device_index)
    return (int(major), int(minor))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target; the error is not cached, the architecture is)."""
    capability = device_capability(device_index)
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake QSA NVFP4 decode target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake QSA NVFP4 decode record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    from . import (
        cake_build,
    )  # written by the exporter beside build.json; absent before the first render

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    from . import (
        cake_build,
    )  # written by the exporter beside build.json; absent before the first render

    return cake_build.jit_specs(arch)


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter (buffers
    and scalars alike) and ``(None, axis)`` per grid argument, in ``run``
    order; computed once."""
    record = MODULES[name]
    logical = record["logical"]
    if not record["arg_plan"]:
        raise RuntimeError(
            f"record {name} has no argument plan: the package was not exported "
            "(the exporter writes the derived leaves of cake_jit.MODULES)"
        )
    plan: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append((None, "xyz".index(key[-1])))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], -1))
    return tuple(plan)


@functools.cache
def _raw_stream_getter() -> Callable[[int], int]:
    """``torch._C._cuda_getCurrentRawStream`` -- the handle of a device's
    current torch stream as an integer, without constructing a ``Stream``
    object -- or the ``torch.cuda`` form on a torch build without it.  Bound
    on first use: this module imports torch lazily."""
    import torch

    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if getter is not None:
        return getter

    def fallback(device_index: int) -> int:
        return torch.cuda.current_stream(device_index).cuda_stream

    return fallback


def current_raw_stream(device_index: int) -> int:
    """The ``cudaStream_t`` handle (an integer) of torch's current stream on
    CUDA device ``device_index``; the key of the host's per-stream caches."""
    return _raw_stream_getter()(device_index)


_FFI_DEVICES: dict[int, Any] = {}


def ffi_stream_context(device_index: int, stream: int) -> Any:
    """The tvm-ffi environment-stream context that runs the launches inside it
    on ``stream`` (a raw ``cudaStream_t`` handle, :func:`current_raw_stream`)
    of CUDA device ``device_index``: the bindings read that stream.  The FFI
    device handle is kept per index; the context itself is cheap, unlike
    ``tvm_ffi.use_torch_stream()``, which builds a ``torch.cuda.Stream`` and
    parses a device string on every entry."""
    device = _FFI_DEVICES.get(device_index)
    if device is None:
        device = _FFI_DEVICES[device_index] = tvm_ffi.device(f"cuda:{device_index}")
    return tvm_ffi.use_raw_stream(device, stream)


@functools.lru_cache(maxsize=4096)
def resolved_launches(
    *,
    form: str,
    splits: int,
    arch: str,
    wide: bool,
    rows: int,
    kv_heads: int,
    qo_heads: int,
    walk: int = 0,
) -> tuple[tuple[Any, Callable[[Any], tuple[Any, ...]], tuple[int, int, int]], ...]:
    """The launches of one call, resolved once per key of host scalars: per
    record of :func:`select_records`, in launch order, its loaded FFI entry,
    a getter that picks the record's kernel arguments out of the flat
    argument sequence ``LOGICAL_TENSORS + LOGICAL_SCALARS`` (``LOGICAL_INDEX``)
    and its launch grid (:func:`launch_grid`).  A launch is then
    ``entry(*getter(arguments), *grid)``: every argument plan ends with the
    three grid arguments, and the host interprets no plan per call.  A walk
    (``walk > 0``) whose selected decode record keeps its own grid rule is
    refused: the walk's split count is the walk's slots, not the plain cut.
    """
    names = select_records(form=form, splits=splits, arch=arch, wide=wide, walk=walk)
    if int(walk) > 0 and MODULES[names[0]]["grid_rule"]["kind"] != "walk_ctas":
        raise ValueError(
            f"walk={int(walk)} CTAs on {arch}, but the selected decode record {names[0]!r} "
            "is not the persistent walk (its grid rule is "
            f"{MODULES[names[0]]['grid_rule']['kind']!r})"
        )
    launches = []
    for name in names:
        plan = _argument_plan(name)
        grid_axes = [axis for logical, axis in plan if logical is None]
        if grid_axes != [0, 1, 2] or any(logical is None for logical, _ in plan[:-3]):
            raise RuntimeError(
                f"record {name}: the grid arguments are not the suffix of its argument plan"
            )
        logicals = [logical for logical, _ in plan[:-3]]
        for logical in logicals:
            if logical not in LOGICAL_INDEX:
                raise ValueError(f"record {name} needs logical argument {logical!r}")
        getter = operator.itemgetter(*(LOGICAL_INDEX[logical] for logical in logicals))
        entry = getattr(load(name, arch), MODULES[name]["ffi_entry"])
        grid = launch_grid(
            name,
            rows=rows,
            kv_heads=kv_heads,
            qo_heads=qo_heads,
            splits=splits,
            walk=walk,
        )
        launches.append((entry, getter, grid))
    return tuple(launches)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to
    tensors (``None`` for an absent optional tensor) and Python scalars; the
    record maps its kernel parameters onto that vocabulary and ``grid`` is the
    call's launch grid (:func:`launch_grid`).  The kernel runs on the current
    torch stream of the device.  The host's entry resolves its launches once
    per shape instead (:func:`resolved_launches`); this form interprets the
    plan per call and serves callers that bind by logical name.
    """
    record = MODULES[name]
    args = []
    for logical_name, axis in _argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    module = load(name, device_arch(device_index))
    with ffi_stream_context(device_index, current_raw_stream(device_index)):
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "DECODE_CTAS_PER_SM",
    "DECODE_FUSED_CTAS_PER_SM",
    "DECODE_LONG_STREAM_TILES",
    "DECODE_PARAMETERS",
    "DECODE_STREAM_TILES",
    "DECODE_THREADS",
    "FORMS",
    "GRID_BOUNDS",
    "LOGICAL_INDEX",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_SPLITS",
    "MERGE_PARAMETERS",
    "MERGE_WARPS",
    "MODULES",
    "PREFIX",
    "SELECT_RULES",
    "STAGES",
    "TILE_TOKENS",
    "WALK_MAX_WAVES",
    "arch_for_capability",
    "current_raw_stream",
    "device_arch",
    "device_capability",
    "ffi_stream_context",
    "launch_grid",
    "library_name",
    "load",
    "merge_heads_per_cta",
    "records_of_stage",
    "resolved_launches",
    "run_module",
    "select_record",
    "select_records",
    "sm_count",
    "specs_for_arch",
    "walk_applies",
    "walk_ctas",
    "walk_splits",
]
