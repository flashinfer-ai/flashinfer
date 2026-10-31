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

Registry and loader glue of the Cake MiniMax-M3 index-context-parallel (ICP)
indexer kernels: the prefill device plan and chunk scorer, the decode scorer,
the prefill Top16 selector (carrier rows), the decode Top16 selector and
publisher (peer receive rows) and the owner-side candidate merge.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled for each exact
architecture in its ``arches`` (``sm_100a``, ``sm_103a``, ``sm_107a``; no
family target).  A record carries the host plan: the program's ``library``
stem (its ``build.json`` entry is ``<library>_<arch>``), its ``role`` among
:data:`ROLES`, the local K rows per page ``rows`` it is built for (``None``
when the program serves every page layout), the ``page`` size it is built for
(``None`` when its rows serve every page size) and the ``grid_rule`` that
shapes its launch.  The tensor-core roles (``score``, ``decode_plane``) have one
program per ``rows`` value and architecture group: ``sm_107a`` runs its own
schedule, never the ``sm_100a`` / ``sm_103a`` one.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan.  What to compile -- sources, digests, nvcc
flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves a library from a prebuilt ``aot/`` file when a
builder compiled it and otherwise through FlashInfer's JIT from the same
sources and flags.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
PREFIX = "cake_msa_icp_indexer"

# The kernels of one layer, in launch order.  Prefill / mixed: ``plan`` (once
# per query window, reused by every layer), ``score`` and ``select_prefill``
# per bounded query chunk, the caller's exchange, then ``merge``.  Decode:
# ``decode_plane`` -> ``select_publish`` -> merge (the modular chain), or
# ``decode_topk`` -> ``topk16_publish`` -> merge (the fused chain).  A prefill
# merge is ``merge_prefill``; a decode merge is ``merge_legacy`` (one token per
# CTA) for legacy rows of at least 8 tokens, else ``merge`` (4 tokens per CTA),
# from launch metadata only.
ROLES = (
    "plan",
    "score",
    "decode_plane",
    "decode_topk",
    "select_prefill",
    "select_publish",
    "topk16_publish",
    "merge",
    "merge_legacy",
    "merge_prefill",
)
# Grid rules of the records (evaluated by :func:`launch_grid`): one CTA, one
# CTA per streaming multiprocessor (the persistent prefill scorer, whose plan
# holds one work range per SM), one CTA per (request, split) of the decode
# scorer, a fixed number of (token, head) rows per CTA, or a fixed number of
# tokens per CTA along x with one head per CTA along y.
GRID_RULES = ("single", "per_sm", "request_splits", "tokens_per_cta_by_head")
# The most splits a decode scorer request is cut into (one list per split in
# the fused path's pool).
MAX_DECODE_SPLITS = 128
# The most splits of the fused path's list scorer per architecture (the
# publisher merges every list of a row; past this count the merge costs more
# than the extra scorer parallelism saves).
TOPK16_SPLIT_CAP: dict[str, int] = {"sm_100a": 48, "sm_103a": 48, "sm_107a": 48}
# The ``launch`` argument of a record whose ``launch.use_pdl`` is
# ``"runtime"``: ``run`` takes it right before the grid, 1 adds the
# programmatic-dependent-launch attribute to the call, 0 launches plain.
LAUNCH_ARGUMENT = "enable_pdl"

# Logical argument names the host backend binds: the kernels' own parameter
# names (``record["logical"]`` is the identity map).  Tensors: ``q`` FP8 index
# queries ``[T, H, 128]``, ``k`` the rank's FP8 index-K shard ``[pages, R,
# 128]``, ``block_table`` int32 ``[B, columns]``, ``seq_lens`` int32 ``[B]``,
# ``num_valid_pages`` int32 ``[T]`` (global B128 count per token), the plan's
# segment metadata ``qo_lens`` / ``kv_lens`` / ``qo_offset`` int32 ``[B]`` and
# ``qo_indptr`` / ``kv_indptr`` int32 ``[B + 1]``, ``work_range`` /
# ``work_info`` int64 plan storage, ``max_score`` / ``scores`` FP32
# ``[T, H, K]`` score rows, ``pool`` the fused path's sorted candidate lists,
# ``carrier`` int32 ``[H, T, 33]`` prefill rows, ``recv_bases`` int64
# ``[world]`` peer receive-slot addresses, ``errors`` int32 ``[1]`` timeout
# counter, ``generations`` int32 ``[3, H, capacity]`` publish counters,
# ``recv`` the owner's receive rows of one slot set, ``expected`` int32
# ``[hpr, capacity]`` merge counters, ``output`` int32 ``[T, hpr, 16]``.
LOGICAL_TENSORS = (
    "q",
    "k",
    "block_table",
    "seq_lens",
    "num_valid_pages",
    "qo_lens",
    "kv_lens",
    "qo_indptr",
    "kv_indptr",
    "qo_offset",
    "work_range",
    "work_info",
    "max_score",
    "scores",
    "pool",
    "carrier",
    "recv_bases",
    "errors",
    "generations",
    "recv",
    "expected",
    "output",
)
# Scalars: sizes (``batch``, ``num_tokens``, ``num_heads``, ``heads_per_rank``,
# ``total_q``, ``num_ctas``, ``columns`` the B128 column bound of a prefill
# plan, ``score_cols`` the K extent of a score row, ``capacity``, ``splits``),
# the layout (``rank``, ``world``, ``page_size``, ``compact``,
# ``index_scale``, ``index_offset``, ``query_len``), element strides
# (``bt_stride``, ``score_stride_t``, ``score_stride_h``,
# ``carrier_stride_h``, ``carrier_stride_t``, ``pool_row_stride`` (uint64
# words between the (token, head) rows of the decode pool), ``recv_stride_src``,
# ``recv_stride_h``, ``recv_stride_t``, ``out_stride_t``, ``out_stride_h``)
# and the selection / protocol scalars (``force_begin``, ``force_end``,
# ``init_blocks``, ``local_blocks``, ``slot``, ``spin_limit``, ``tagged``,
# ``prefill``).
LOGICAL_SCALARS = (
    "batch",
    "num_tokens",
    "num_heads",
    "heads_per_rank",
    "total_q",
    "num_ctas",
    "columns",
    "score_cols",
    "capacity",
    "splits",
    "rank",
    "world",
    "page_size",
    "compact",
    "index_scale",
    "index_offset",
    "query_len",
    "bt_stride",
    "score_stride_t",
    "score_stride_h",
    "carrier_stride_h",
    "carrier_stride_t",
    "pool_row_stride",
    "recv_stride_src",
    "recv_stride_h",
    "recv_stride_t",
    "out_stride_t",
    "out_stride_h",
    "force_begin",
    "force_end",
    "init_blocks",
    "local_blocks",
    "slot",
    "spin_limit",
    "tagged",
    "prefill",
)

# Filled by the exporter: the host plan of every record comes from the
# family's table, the derived leaves from the rendered binding.  Every value is
# a literal (FlashInfer's manifest checker and the exporter read this file with
# ``ast.literal_eval``).
MODULES: dict[str, dict[str, Any]] = {
    "select_prefill": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "scores"],
            ["buffer", "num_valid_pages"],
            ["buffer", "carrier"],
            ["parameter", "total_q"],
            ["parameter", "score_cols"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "carrier_stride_h"],
            ["parameter", "carrier_stride_t"],
            ["parameter", "force_begin"],
            ["parameter", "force_end"],
            ["parameter", "index_scale"],
            ["parameter", "index_offset"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "tokens_per_cta_by_head", "rows": 4},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_2e98af744f6a0568d51d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_2e98af744f6a0568d51d",
        "logical": {
            "carrier": "carrier",
            "carrier_stride_h": "carrier_stride_h",
            "carrier_stride_t": "carrier_stride_t",
            "force_begin": "force_begin",
            "force_end": "force_end",
            "index_offset": "index_offset",
            "index_scale": "index_scale",
            "num_valid_pages": "num_valid_pages",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "total_q": "total_q",
        },
        "page": None,
        "role": "select_prefill",
        "rows": None,
    },
    "select_publish": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "scores"],
            ["buffer", "num_valid_pages"],
            ["buffer", "recv_bases"],
            ["buffer", "errors"],
            ["buffer", "generations"],
            ["parameter", "total_q"],
            ["parameter", "score_cols"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "slot"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "heads_per_rank"],
            ["parameter", "capacity"],
            ["parameter", "force_begin"],
            ["parameter", "force_end"],
            ["parameter", "spin_limit"],
            ["parameter", "tagged"],
            ["parameter", "index_scale"],
            ["parameter", "index_offset"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "tokens_per_cta_by_head", "rows": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_7bb0a4a2f772f7ac3c9b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2560,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_7bb0a4a2f772f7ac3c9b",
        "logical": {
            "capacity": "capacity",
            "errors": "errors",
            "force_begin": "force_begin",
            "force_end": "force_end",
            "generations": "generations",
            "heads_per_rank": "heads_per_rank",
            "index_offset": "index_offset",
            "index_scale": "index_scale",
            "num_valid_pages": "num_valid_pages",
            "rank": "rank",
            "recv_bases": "recv_bases",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "slot": "slot",
            "spin_limit": "spin_limit",
            "tagged": "tagged",
            "total_q": "total_q",
            "world": "world",
        },
        "page": None,
        "role": "select_publish",
        "rows": None,
    },
    "topk16_publish": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "pool"],
            ["buffer", "recv_bases"],
            ["buffer", "errors"],
            ["buffer", "generations"],
            ["parameter", "total_q"],
            ["parameter", "num_heads"],
            ["parameter", "splits"],
            ["parameter", "pool_row_stride"],
            ["parameter", "slot"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "heads_per_rank"],
            ["parameter", "capacity"],
            ["parameter", "spin_limit"],
            ["parameter", "tagged"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "tokens_per_cta_by_head", "rows": 4},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_1916ac93b213cdcffd4c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_1916ac93b213cdcffd4c",
        "logical": {
            "capacity": "capacity",
            "errors": "errors",
            "generations": "generations",
            "heads_per_rank": "heads_per_rank",
            "num_heads": "num_heads",
            "pool": "pool",
            "pool_row_stride": "pool_row_stride",
            "rank": "rank",
            "recv_bases": "recv_bases",
            "slot": "slot",
            "spin_limit": "spin_limit",
            "splits": "splits",
            "tagged": "tagged",
            "total_q": "total_q",
            "world": "world",
        },
        "page": None,
        "role": "topk16_publish",
        "rows": None,
    },
    "plan": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "batch"],
            ["parameter", "num_ctas"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "single"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_eb804215a3d74733ee47",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "library": "cake_msa_icp_indexer_eb804215a3d74733ee47",
        "logical": {
            "batch": "batch",
            "kv_lens": "kv_lens",
            "num_ctas": "num_ctas",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": None,
        "role": "plan",
        "rows": None,
    },
    "score_p128_r32": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_3f58c1277a9aa67c62ea",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_3f58c1277a9aa67c62ea",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 128,
        "role": "score",
        "rows": 32,
    },
    "score_p256_r128": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_e771f2e0e8a3d0486617",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_e771f2e0e8a3d0486617",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 256,
        "role": "score",
        "rows": 128,
    },
    "decode_plane_r64": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "scores"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 2},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_e8edcf7dc7352d860f85",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 78848,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_e8edcf7dc7352d860f85",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_plane",
        "rows": 64,
    },
    "decode_plane_r32": {
        "arches": ["sm_100a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "scores"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 2},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_773a535c12217a4e6d61",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 78848,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_773a535c12217a4e6d61",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_plane",
        "rows": 32,
    },
    "decode_plane_r128": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "scores"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_0bfa7f89b16ae9496bbc",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 117760,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_0bfa7f89b16ae9496bbc",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_plane",
        "rows": 128,
    },
    "decode_plane_r64_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "scores"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_eb67d6f02150430ee88f",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 214016,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_eb67d6f02150430ee88f",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_plane",
        "rows": 64,
    },
    "decode_plane_r32_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "scores"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_d0a9cf9610e2e4cee0a3",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 214016,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_d0a9cf9610e2e4cee0a3",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_plane",
        "rows": 32,
    },
    "decode_plane_r128_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "scores"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_1a581eb3e07af939f48e",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 214016,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_1a581eb3e07af939f48e",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_plane",
        "rows": 128,
    },
    "decode_topk16_r64": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "pool"],
            ["parameter", "bt_stride"],
            ["parameter", "pool_row_stride"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 2},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_821db17698295956517d",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 78848,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_821db17698295956517d",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "pool": "pool",
            "pool_row_stride": "pool_row_stride",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_topk",
        "rows": 64,
    },
    "decode_topk16_r32": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "pool"],
            ["parameter", "bt_stride"],
            ["parameter", "pool_row_stride"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 2},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_fb8e62bdd96d246d094f",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 78848,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_fb8e62bdd96d246d094f",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "pool": "pool",
            "pool_row_stride": "pool_row_stride",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_topk",
        "rows": 32,
    },
    "decode_topk16_r128": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "pool"],
            ["parameter", "bt_stride"],
            ["parameter", "pool_row_stride"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_c5c00bd12fefb9c0022b",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 117760,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_c5c00bd12fefb9c0022b",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "pool": "pool",
            "pool_row_stride": "pool_row_stride",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_topk",
        "rows": 128,
    },
    "decode_topk16_r64_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "pool"],
            ["parameter", "bt_stride"],
            ["parameter", "pool_row_stride"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_c09fe410560446efaf41",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 214016,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_c09fe410560446efaf41",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "pool": "pool",
            "pool_row_stride": "pool_row_stride",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_topk",
        "rows": 64,
    },
    "decode_topk16_r32_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "pool"],
            ["parameter", "bt_stride"],
            ["parameter", "pool_row_stride"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_93248b5daab257e7289b",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 214016,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_93248b5daab257e7289b",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "pool": "pool",
            "pool_row_stride": "pool_row_stride",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_topk",
        "rows": 32,
    },
    "decode_topk16_r128_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "pool"],
            ["parameter", "bt_stride"],
            ["parameter", "pool_row_stride"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_11eab789a82a86e6106d",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 214016,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_11eab789a82a86e6106d",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "pool": "pool",
            "pool_row_stride": "pool_row_stride",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_topk",
        "rows": 128,
    },
    "merge": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "recv"],
            ["buffer", "num_valid_pages"],
            ["buffer", "output"],
            ["buffer", "errors"],
            ["buffer", "expected"],
            ["parameter", "recv_stride_src"],
            ["parameter", "recv_stride_h"],
            ["parameter", "recv_stride_t"],
            ["parameter", "out_stride_t"],
            ["parameter", "out_stride_h"],
            ["parameter", "capacity"],
            ["parameter", "num_tokens"],
            ["parameter", "world"],
            ["parameter", "tagged"],
            ["parameter", "prefill"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["parameter", "spin_limit"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "tokens_per_cta_by_head", "rows": 4},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_1ec85900940aaadd4f09",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_1ec85900940aaadd4f09",
        "logical": {
            "capacity": "capacity",
            "errors": "errors",
            "expected": "expected",
            "init_blocks": "init_blocks",
            "local_blocks": "local_blocks",
            "num_tokens": "num_tokens",
            "num_valid_pages": "num_valid_pages",
            "out_stride_h": "out_stride_h",
            "out_stride_t": "out_stride_t",
            "output": "output",
            "prefill": "prefill",
            "recv": "recv",
            "recv_stride_h": "recv_stride_h",
            "recv_stride_src": "recv_stride_src",
            "recv_stride_t": "recv_stride_t",
            "spin_limit": "spin_limit",
            "tagged": "tagged",
            "world": "world",
        },
        "page": None,
        "role": "merge",
        "rows": None,
    },
    "score_p128_r64_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_cef6a25ea6bc74e445b7",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_cef6a25ea6bc74e445b7",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 128,
        "role": "score",
        "rows": 64,
    },
    "score_p128_r32_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_6c21eaf8bf4bce12f366",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_6c21eaf8bf4bce12f366",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 128,
        "role": "score",
        "rows": 32,
    },
    "score_p256_r128_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_77c23fce3c147c3ffe86",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_77c23fce3c147c3ffe86",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 256,
        "role": "score",
        "rows": 128,
    },
    "score_p256_r64_sm107": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_2b5be5f74c1cbc31330d",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_2b5be5f74c1cbc31330d",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 256,
        "role": "score",
        "rows": 64,
    },
    "merge_legacy": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "recv"],
            ["buffer", "num_valid_pages"],
            ["buffer", "output"],
            ["buffer", "errors"],
            ["buffer", "expected"],
            ["parameter", "recv_stride_src"],
            ["parameter", "recv_stride_h"],
            ["parameter", "recv_stride_t"],
            ["parameter", "out_stride_t"],
            ["parameter", "out_stride_h"],
            ["parameter", "capacity"],
            ["parameter", "num_tokens"],
            ["parameter", "world"],
            ["parameter", "tagged"],
            ["parameter", "prefill"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["parameter", "spin_limit"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "tokens_per_cta_by_head", "rows": 1},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_f2b78240f58013243eae",
        "launch": {
            "block": (32, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_f2b78240f58013243eae",
        "logical": {
            "capacity": "capacity",
            "errors": "errors",
            "expected": "expected",
            "init_blocks": "init_blocks",
            "local_blocks": "local_blocks",
            "num_tokens": "num_tokens",
            "num_valid_pages": "num_valid_pages",
            "out_stride_h": "out_stride_h",
            "out_stride_t": "out_stride_t",
            "output": "output",
            "prefill": "prefill",
            "recv": "recv",
            "recv_stride_h": "recv_stride_h",
            "recv_stride_src": "recv_stride_src",
            "recv_stride_t": "recv_stride_t",
            "spin_limit": "spin_limit",
            "tagged": "tagged",
            "world": "world",
        },
        "page": None,
        "role": "merge_legacy",
        "rows": None,
    },
    "decode_plane_r32_s2k4": {
        "arches": ["sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "seq_lens"],
            ["buffer", "num_valid_pages"],
            ["buffer", "scores"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "num_heads"],
            ["parameter", "query_len"],
            ["parameter", "columns"],
            ["parameter", "splits"],
            ["parameter", "rank"],
            ["parameter", "world"],
            ["parameter", "page_size"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "request_splits", "ctas_per_sm": 4},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_30c38a77cc5a9f85484e",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_30c38a77cc5a9f85484e",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "columns": "columns",
            "init_blocks": "init_blocks",
            "k": "k",
            "local_blocks": "local_blocks",
            "num_heads": "num_heads",
            "num_valid_pages": "num_valid_pages",
            "page_size": "page_size",
            "q": "q",
            "query_len": "query_len",
            "rank": "rank",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "scores": "scores",
            "seq_lens": "seq_lens",
            "splits": "splits",
            "world": "world",
        },
        "page": None,
        "role": "decode_plane",
        "rows": 32,
    },
    "merge_prefill": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "recv"],
            ["buffer", "num_valid_pages"],
            ["buffer", "output"],
            ["buffer", "errors"],
            ["buffer", "expected"],
            ["parameter", "recv_stride_src"],
            ["parameter", "recv_stride_h"],
            ["parameter", "recv_stride_t"],
            ["parameter", "out_stride_t"],
            ["parameter", "out_stride_h"],
            ["parameter", "capacity"],
            ["parameter", "num_tokens"],
            ["parameter", "world"],
            ["parameter", "tagged"],
            ["parameter", "prefill"],
            ["parameter", "init_blocks"],
            ["parameter", "local_blocks"],
            ["parameter", "spin_limit"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "tokens_per_cta_by_head", "rows": 4},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_529912a06e17ebe18c55",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_529912a06e17ebe18c55",
        "logical": {
            "capacity": "capacity",
            "errors": "errors",
            "expected": "expected",
            "init_blocks": "init_blocks",
            "local_blocks": "local_blocks",
            "num_tokens": "num_tokens",
            "num_valid_pages": "num_valid_pages",
            "out_stride_h": "out_stride_h",
            "out_stride_t": "out_stride_t",
            "output": "output",
            "prefill": "prefill",
            "recv": "recv",
            "recv_stride_h": "recv_stride_h",
            "recv_stride_src": "recv_stride_src",
            "recv_stride_t": "recv_stride_t",
            "spin_limit": "spin_limit",
            "tagged": "tagged",
            "world": "world",
        },
        "page": None,
        "role": "merge_prefill",
        "rows": None,
    },
    "score_p128_r64_sm100": {
        "arches": ["sm_100a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_4d1eadb61143514d9c8d",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_4d1eadb61143514d9c8d",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 128,
        "role": "score",
        "rows": 64,
    },
    "score_p128_r64_sm103": {
        "arches": ["sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_51c095cb92a3ad587332",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_51c095cb92a3ad587332",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 128,
        "role": "score",
        "rows": 64,
    },
    "score_p256_r64_sm100": {
        "arches": ["sm_100a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_cdcc9a2c2f7c3df64d6e",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_cdcc9a2c2f7c3df64d6e",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 256,
        "role": "score",
        "rows": 64,
    },
    "score_p256_r64_sm103": {
        "arches": ["sm_103a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "block_table"],
            ["buffer", "max_score"],
            ["buffer", "qo_lens"],
            ["buffer", "kv_lens"],
            ["buffer", "qo_indptr"],
            ["buffer", "qo_offset"],
            ["buffer", "work_range"],
            ["buffer", "work_info"],
            ["parameter", "rank"],
            ["parameter", "compact"],
            ["parameter", "bt_stride"],
            ["parameter", "score_stride_t"],
            ["parameter", "score_stride_h"],
            ["parameter", "score_cols"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "per_sm"},
        "kernel_symbol": "kernel_cake_msa_icp_indexer_456b0292da8f135336c6",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": "runtime",
        },
        "library": "cake_msa_icp_indexer_456b0292da8f135336c6",
        "logical": {
            "block_table": "block_table",
            "bt_stride": "bt_stride",
            "compact": "compact",
            "k": "k",
            "kv_lens": "kv_lens",
            "max_score": "max_score",
            "q": "q",
            "qo_indptr": "qo_indptr",
            "qo_lens": "qo_lens",
            "qo_offset": "qo_offset",
            "rank": "rank",
            "score_cols": "score_cols",
            "score_stride_h": "score_stride_h",
            "score_stride_t": "score_stride_t",
            "work_info": "work_info",
            "work_range": "work_range",
        },
        "page": 256,
        "role": "score",
        "rows": 64,
    },
}


@functools.cache
def _routes() -> dict[tuple[str, int | None, int | None, str], str]:
    """``(role, rows, page, arch) -> record name``: exactly one record per
    key (``page`` ``None``: the record serves every page size)."""
    routes: dict[tuple[str, int | None, int | None, str], str] = {}
    for name, record in MODULES.items():
        for arch in record["arches"]:
            key = (record["role"], record["rows"], record["page"], arch)
            if key in routes:
                raise RuntimeError(f"records {routes[key]} and {name} both serve {key}")
            routes[key] = name
    return routes


def route(role: str, rows: int | None, arch: str, page: int | None = None) -> str:
    """The record of ``role`` (built for ``rows`` local K rows per page, or
    ``None`` for a layout-independent role) on ``arch``: the one built for
    ``page`` when there is one, else the page-independent one.

    ``NotImplementedError`` names a missing (role, rows, arch) program.
    """
    routes = _routes()
    name = routes.get((role, rows, page, arch)) or routes.get((role, rows, None, arch))
    if name is None:
        raise NotImplementedError(
            f"no Cake MSA ICP indexer program for role {role!r}"
            + (f" at {rows} K rows per page" if rows is not None else "")
            + f" on {arch}"
        )
    return name


def has_role(role: str, arch: str) -> bool:
    """Whether any record of ``role`` is delivered for ``arch``."""
    return any(key[0] == role and key[3] == arch for key in _routes())


def launch_grid(
    name: str,
    *,
    batch: int = 0,
    splits: int = 0,
    sm_count: int = 0,
    tokens: int = 0,
    heads: int = 0,
) -> tuple[int, int, int]:
    """Grid of record ``name`` from host-known sizes (its ``grid_rule``)."""
    rule = MODULES[name]["grid_rule"]
    kind = rule["kind"]
    if kind == "single":
        return (1, 1, 1)
    if kind == "per_sm":
        return (sm_count, 1, 1)
    if kind == "request_splits":
        return (batch, splits, 1)
    if kind == "tokens_per_cta_by_head":
        per = int(rule["rows"])
        return ((tokens + per - 1) // per, heads, 1)
    raise RuntimeError(f"record {name}: grid rule {kind!r} is not one of {GRID_RULES}")


def ctas_per_sm(name: str) -> int:
    """Resident CTAs per SM of a ``request_splits`` record (its split rule)."""
    return int(MODULES[name]["grid_rule"]["ctas_per_sm"])


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    return ARCH_BY_CAPABILITY.get((int(capability[0]), int(capability[1])))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``ValueError`` otherwise)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise ValueError(
            f"device {device_index} has compute capability {capability}; the Cake "
            "MSA ICP indexer supports 10.0, 10.3 and 10.7"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``."""
    return f"{MODULES[name]['library']}_{arch}"


# DLPack's CUDA device type and tvm-ffi's environment stream setter (the one
# ``tvm_ffi.use_torch_stream`` calls); without it ``run_module`` falls back to
# ``use_torch_stream``.
_DL_CUDA = 2
_set_ffi_stream = getattr(
    getattr(tvm_ffi, "core", None), "_env_set_current_stream", None
)

_LOADED: dict[tuple[str, str], Any] = {}
# (record, device) pairs whose kernel is loaded in that device's context.
_RESIDENT: set[tuple[str, int]] = set()
# (record, arch, device) -> (module, bound FFI entry, argument plan).
_LAUNCHERS: dict[
    tuple[str, str, int], tuple[Any, Any, tuple[tuple[str, Any], ...]]
] = {}


@functools.cache
def _runtime(path: str) -> Any:
    """The CUDA runtime calls of the runtime the library at ``path`` binds to.

    Each symbol is resolved through the library's own handle (the library and
    its dependencies), never by loading ``libcudart.so`` by name: a process can
    map a second runtime copy (Nsight Systems' OS-runtime tracing makes a
    by-name load find the CUDA toolkit's libcudart next to the one PyTorch
    loaded), and that copy knows none of the library's kernels.  Driver calls
    come from that runtime's driver entry points, so they act on the driver
    instance the runtime uses; a driver without ``cuFuncLoad`` (CUDA < 12.4)
    leaves them ``None``."""
    import ctypes
    import types

    library = ctypes.CDLL(path)

    def bind(name: str, restype: Any, *argtypes: Any) -> Any:
        address = ctypes.cast(getattr(library, name), ctypes.c_void_p).value
        return ctypes.CFUNCTYPE(restype, *argtypes)(address)

    calls = types.SimpleNamespace(
        library=library,
        cudaGetFuncBySymbol=bind(
            "cudaGetFuncBySymbol", ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        ),
        cudaFuncGetAttributes=bind(
            "cudaFuncGetAttributes", ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
        ),
        cudaGetErrorString=bind("cudaGetErrorString", ctypes.c_char_p, ctypes.c_int),
        cudaGetLastError=bind("cudaGetLastError", ctypes.c_int),
        cuFuncLoad=None,
        cuFuncIsLoaded=None,
        cuGetErrorString=None,
    )
    try:
        entry_point = bind(
            "cudaGetDriverEntryPointByVersion",
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_ulonglong,
            ctypes.c_void_p,
        )
    except AttributeError:
        return calls
    driver = {}
    for name, restype, argtypes in (
        ("cuFuncLoad", ctypes.c_int, (ctypes.c_void_p,)),
        ("cuFuncIsLoaded", ctypes.c_int, (ctypes.c_void_p, ctypes.c_void_p)),
        ("cuGetErrorString", ctypes.c_int, (ctypes.c_int, ctypes.c_void_p)),
    ):
        function, found = ctypes.c_void_p(), ctypes.c_int(-1)
        status = entry_point(
            name.encode(), ctypes.byref(function), 12040, 0, ctypes.byref(found)
        )
        if (
            status != 0 or found.value != 0 or not function.value
        ):  # 0: cudaDriverEntryPointSuccess
            calls.cudaGetLastError()
            return calls
        driver[name] = ctypes.CFUNCTYPE(restype, *argtypes)(function.value)
    calls.__dict__.update(driver)
    return calls


def _symbol_values(path: str, names: tuple[str, ...]) -> dict[str, int]:
    """``st_value`` of ``names`` in the ELF64 symbol tables of ``path``.

    The kernel's host stub is a hidden (local) symbol of the library, so it is
    read from ``.symtab`` rather than resolved with ``dlsym``."""
    import struct

    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise RuntimeError(f"{path} is not a little-endian ELF64 library")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    wanted = {name.encode(): name for name in names}
    found: dict[str, int] = {}
    for _, kind, _, _, offset, size, link, _, _, entsize in sections:
        if kind not in (2, 11) or not entsize:  # SHT_SYMTAB, SHT_DYNSYM
            continue
        strtab = sections[link][4]
        for at in range(offset, offset + size, entsize):
            st_name, _, _, _, st_value, _ = struct.unpack_from("<IBBHQQ", data, at)
            end = data.index(b"\0", strtab + st_name)
            name = wanted.get(data[strtab + st_name : end])
            if name is not None and st_value:
                found[name] = st_value
    return found


def _make_resident(name: str, arch: str, device: int) -> None:
    """Load the kernel of record ``name`` into ``device``'s context now.

    Under CUDA lazy module loading a kernel is loaded at its first launch.  A
    publisher loaded while a peer's merge is already spinning on its rows can
    stall until the spin limit, so the kernel is loaded ahead of any launch.
    The kernel's host stub is the library's load base (from the exported
    ``__tvm_ffi_run``) plus its symbol value.  The runtime the library binds to
    (:func:`_runtime`) turns the stub into the context's function handle
    (``cudaGetFuncBySymbol``) and the driver loads it (``cuFuncLoad``, checked
    with ``cuFuncIsLoaded``); ``cudaFuncGetAttributes`` on the stub is the path
    for a driver without ``cuFuncLoad``.  A kernel that neither path loads
    raises."""
    import ctypes

    from . import cake_build

    path = str(cake_build.library_path(library_name(name, arch)))
    kernel = MODULES[name]["kernel_symbol"]
    values = _symbol_values(path, (kernel, "__tvm_ffi_run"))
    if len(values) != 2:
        raise RuntimeError(
            f"Cake MSA ICP indexer program {name} for {arch}: {path} lacks the symbol "
            f"{kernel} (a stripped library cannot be made resident)"
        )
    try:
        calls = _runtime(path)
    except AttributeError as error:
        raise RuntimeError(
            f"Cake MSA ICP indexer program {name} for {arch}: {path} does not resolve the CUDA "
            f"runtime it links ({error})"
        ) from error
    entry = ctypes.cast(calls.library.__tvm_ffi_run, ctypes.c_void_p).value
    stub = ctypes.c_void_p(entry - values["__tvm_ffi_run"] + values[kernel])

    def runtime_error(status: int) -> str:
        calls.cudaGetLastError()
        return f"{status} ({calls.cudaGetErrorString(status).decode()})"

    def driver_error(status: int) -> str:
        text = ctypes.c_char_p()
        calls.cuGetErrorString(status, ctypes.byref(text))
        return f"{status} ({text.value.decode() if text.value else 'unknown'})"

    failures = []
    if calls.cuFuncLoad is not None:
        function = ctypes.c_void_p()
        status = calls.cudaGetFuncBySymbol(ctypes.byref(function), stub)
        if status != 0 or not function.value:
            failures.append(f"cudaGetFuncBySymbol: {runtime_error(status)}")
        else:
            status = calls.cuFuncLoad(function)
            if status != 0:
                failures.append(f"cuFuncLoad: {driver_error(status)}")
            else:
                state = ctypes.c_int(0)
                status = calls.cuFuncIsLoaded(ctypes.byref(state), function)
                if status == 0 and state.value == 1:  # CU_FUNCTION_LOADING_STATE_LOADED
                    return
                failures.append(
                    f"cuFuncIsLoaded: {driver_error(status) if status else f'state {state.value}'}"
                )
    attributes = ctypes.create_string_buffer(1024)
    status = calls.cudaFuncGetAttributes(attributes, stub)
    if status == 0:
        return
    failures.append(f"cudaFuncGetAttributes: {runtime_error(status)}")
    raise RuntimeError(
        f"Cake MSA ICP indexer program {name} for {arch}: the kernel could not be made resident "
        f"on device {device} ({'; '.join(failures)})"
    )


def load(name: str, arch: str, device_index: int | None = None) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags, with its kernel resident in the
    context of device ``device_index`` (default: the current device;
    :func:`_make_resident`).  Under CUDA Graph capture on that device an
    unloaded program is refused before any build (call ``warmup`` on every
    rank first)."""
    import torch

    current = torch.cuda.current_device()
    device = current if device_index is None else device_index
    module = _LOADED.get((name, arch))
    if module is not None and (name, device) in _RESIDENT:
        return module
    if device != current:
        # The capture check and the residency calls act on the current
        # device's stream and context: run them on ``device_index``.
        with torch.cuda.device(device):
            return load(name, arch, device)
    if module is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Cake MSA ICP indexer program {name} for {arch} is not loaded; "
                "warm it up before CUDA Graph capture"
            )
        from . import cake_build

        module = cake_build.load_library(library_name(name, arch))
        _LOADED[(name, arch)] = module
    _make_resident(name, arch, device)
    _RESIDENT.add((name, device))
    return module


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file."""
    from . import cake_build

    return cake_build.jit_specs(arch)


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str, Any], ...]:
    """The record's ``run`` arguments: ``("value", logical name)`` per kernel
    parameter, ``("launch", None)`` for the per-call PDL switch and
    ``("grid", axis)`` per grid argument."""
    record = MODULES[name]
    if not record["arg_plan"]:
        raise RuntimeError(
            f"record {name} has no argument plan (the package was not exported)"
        )
    plan: list[tuple[str, Any]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append(("grid", "xyz".index(key[-1])))
        elif kind == "launch":
            plan.append(("launch", None))
        else:
            plan.append(("value", record["logical"][key]))
    return tuple(plan)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    arch: str,
    enable_pdl: bool = False,
    device_index: int | None = None,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid`` on
    torch's current stream of device ``device_index``, the device of the
    arguments (default: the current device).  ``enable_pdl`` is the per-call
    programmatic dependent launch switch of a ``launch.use_pdl = "runtime"``
    record; a record without it launches as built and takes no switch."""
    import torch

    device = torch.cuda.current_device() if device_index is None else device_index
    # The bound entry and argument plan of a loaded, resident record (load()
    # runs once per (record, arch, device); the cache follows _LOADED and
    # _RESIDENT, so dropping either reloads).
    launcher = _LAUNCHERS.get((name, arch, device))
    if (
        launcher is None
        or launcher[0] is not _LOADED.get((name, arch))
        or (name, device) not in _RESIDENT
    ):
        module = load(name, arch, device)
        launcher = (
            module,
            getattr(module, MODULES[name]["ffi_entry"]),
            _argument_plan(name),
        )
        _LAUNCHERS[(name, arch, device)] = launcher
    _, entry, plan = launcher
    pdl = 1 if enable_pdl else 0
    args = [
        grid[key] if kind == "grid" else pdl if kind == "launch" else values[key]
        for kind, key in plan
    ]
    if _set_ffi_stream is None:
        with torch.cuda.device(device), tvm_ffi.use_torch_stream():
            entry(*args)
        return
    # tvm_ffi.use_torch_stream() without its per-call torch Stream object and
    # device-string parsing: the binding launches on the FFI environment
    # stream of the device, set here to torch's current (or capturing) stream
    # of that device and restored afterwards.
    previous = _set_ffi_stream(
        _DL_CUDA, device, torch._C._cuda_getCurrentRawStream(device)
    )
    try:
        entry(*args)
    finally:
        _set_ffi_stream(_DL_CUDA, device, previous)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GRID_RULES",
    "LAUNCH_ARGUMENT",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_DECODE_SPLITS",
    "TOPK16_SPLIT_CAP",
    "MODULES",
    "PREFIX",
    "ROLES",
    "arch_for_capability",
    "ctas_per_sm",
    "device_arch",
    "has_role",
    "launch_grid",
    "library_name",
    "load",
    "route",
    "run_module",
    "sm_count",
    "specs_for_arch",
]
