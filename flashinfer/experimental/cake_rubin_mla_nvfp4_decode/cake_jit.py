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

Registry and loader glue of the Cake dense NVFP4 MLA decode on Rubin (SM107):
the query quantizer, the mixed-cluster decode (its three builds) and the
split-KV merge (its six routed forms).

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled for ``sm_107a``.  A
record carries the host plan only: the program's ``library`` stem (its
``build.json`` entry is ``<library>_<arch>``), the architectures it is
delivered for, the ``stage`` of the launch order it fills and its ``variant``,
the ``select_rule`` that picks one variant of a stage from the host-known
plan scalars, the ``grid_rule`` that shapes its launch, the kernel-parameter
-> logical-argument map and the argument plan of its ``run`` entry.  The
decode and merge records launch under ``launch.use_pdl = "runtime"``: their
plans carry the ``launch`` argument ``enable_pdl`` (0 or 1) right before the
grid, and the host sets it per plan (``cake_backend.pdl_for``: on when a split
merge exists and the decode is one hardware wave of clusters and pairs, off
otherwise; the
quantizer launches without the attribute and triggers its dependents).  What to
compile -- sources, digests, nvcc flags -- is recorded once in ``build.json``
next to this file; :mod:`.cake_build` serves the library from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's
JIT from the same sources and flags.

The rules are data, evaluated here (:func:`select_record`, :func:`launch_grid`)
from four host-known plan scalars: the packed (token, head) row count
``rows``, the work-item count ``items``, the count ``n_full`` of full-tile
items among them and the partial slots per row ``splits``.  No rule reads a
tensor.  The ``decode`` stage has three builds of one kernel: ``decode``
(``mixed``) runs clusters of four on the full-tile items and the pair body on
the double items of half tiles; ``decode_full`` (``full``) is the same kernel
built for a plan without half tiles (``n_full == items``), whose four-CTA body
skips the double-item test at each item's end; ``decode_full_pairpipe``
(``full_pairpipe``) is the full build with the pair body's converter pipeline,
taking the plans whose scalars lie in its ``plan_within`` boxes (inclusive
``[rows_lo, rows_hi, n_full_lo, n_full_hi, items_lo, items_hi, splits_lo,
splits_hi]``, a ``hi`` of 0 unbounded), which the ``all_full`` rule of
``decode_full`` excepts.  The ``merge`` stage has six forms of one kernel, chosen
by the row count and the partials per row (the ``rows_splits_within`` boxes
of the records: one warp per row above 2048 rows or four up to 2048 rows at
17-32 partials, the two- and eight-warp forms and the four-warp column form
that issue their loads ahead of the merged-row probe elsewhere, the eight-warp
form's 48-slot program for rows of up to 384 at 33-48 partials); the host
launches it only when the plan has more than one split.  The ``quantize`` stage has one program.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

# One verified build: R200 (``sm_107a``), compiled with its exact ``-gencode``
# flag set (``build.json`` ``arch_flags``); no family fallback.
ARCHES = ("sm_107a",)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 7): "sm_107a",
}
PREFIX = "cake_rubin_mla_nvfp4_decode"

# The launch order of the complete call: the BF16 -> NVFP4 query quantizer,
# the decode (one of its three builds) and, when the plan has more than one
# split, the split-KV merge behind it.
STAGES = ("quantize", "decode", "merge")
# The merge's split-slot bound: a row has at most this many partials (two per
# lane of the merge's statistics pass).
MAX_SPLITS = 64
# The ``launch`` argument of a record whose ``launch.use_pdl`` is ``"runtime"``
# (the decode and the merge): ``run`` takes it right before the grid, 1 adds
# the programmatic-dependent-launch attribute to the call, 0 launches plain.
LAUNCH_ARGUMENT = "enable_pdl"

# Logical argument names the host backend binds: the kernels' own parameter
# names (``record["logical"]`` is the identity map).  Decode: ``QN`` / ``QS`` /
# ``QR`` the query's packed E2M1 bytes, E4M3 block scales and E4M3 rope as 2-D
# ``[rows, 256 | 32 | 64]`` uint8 views (TMA operands), ``CKV`` / ``CKS`` /
# ``KPE`` the cache's as ``[pages * 64, 256 | 32 | 64]`` uint8 views (strided
# rows allowed), ``ckv_words`` / ``cks_words`` the latent and block scales as
# int32 words from the same storage, ``O`` BF16 ``[rows, 512]``, ``LSE`` FP32
# ``[rows]`` (natural log), ``partial_o`` BF16 ``[rows, max_splits, 512]``,
# ``partial_lse`` FP32 ``[rows, max_splits]`` (log2 domain), ``q_scale`` FP32
# ``[rows]``, ``item_table`` int32 ``[items, 6]``, ``page_table`` int32
# ``[B, max_pages]``, ``seq_lens`` int32 ``[B]`` (this rank's local lengths),
# ``q_indptr`` int32 ``[B + 1]``, ``row_limit`` int32 ``[rows]``, ``O_T`` /
# ``O_T16`` the output again as the epilogue's 32- and 16-row store maps,
# ``PO_T`` / ``PO_T16`` the partials again.  Merge: ``partial_o`` /
# ``partial_lse`` / ``O`` as above and ``lse`` FP32 ``[rows]``.  Quantizer:
# ``q`` BF16 ``[rows, 576]``, ``q_nope`` uint32 ``[rows, 64]``, ``q_sf`` uint8
# ``[rows, 32]``, ``q_rope`` uint32 ``[rows, 16]``, ``q_scale`` FP32 ``[rows]``.
LOGICAL_TENSORS = (
    "QN",
    "QS",
    "QR",
    "CKV",
    "CKS",
    "KPE",
    "ckv_words",
    "cks_words",
    "O",
    "LSE",
    "partial_o",
    "partial_lse",
    "q_scale",
    "item_table",
    "page_table",
    "seq_lens",
    "q_indptr",
    "row_limit",
    "O_T",
    "PO_T",
    "O_T16",
    "PO_T16",
    "lse",
    "q",
    "q_nope",
    "q_sf",
    "q_rope",
)
LOGICAL_SCALARS = (
    "num_heads",
    "max_pages",
    "items",
    "max_splits",
    "ckv_stride_words",
    "cks_stride_words",
    "logit_scale",
    "o_factor",
    "n_full",
    "rows",
    "num_split",
    "has_lse",
    "c_nope",
    "c_rope",
    "kpe_scale",
    "ckv_scale",
)

# The kernel parameter names of each program, in parameter order: the identity
# ``logical`` map of its record and the order its ``arg_plan`` follows.
DECODE_PARAMETERS = (
    "QN",
    "QS",
    "QR",
    "CKV",
    "CKS",
    "KPE",
    "ckv_words",
    "cks_words",
    "O",
    "LSE",
    "partial_o",
    "partial_lse",
    "q_scale",
    "item_table",
    "page_table",
    "seq_lens",
    "q_indptr",
    "row_limit",
    "num_heads",
    "max_pages",
    "items",
    "max_splits",
    "ckv_stride_words",
    "cks_stride_words",
    "logit_scale",
    "o_factor",
    "O_T",
    "PO_T",
    "O_T16",
    "PO_T16",
    "n_full",
)
MERGE_PARAMETERS = (
    "partial_o",
    "partial_lse",
    "O",
    "lse",
    "rows",
    "num_split",
    "has_lse",
)
QUANTIZE_PARAMETERS = (
    "q",
    "q_nope",
    "q_sf",
    "q_rope",
    "q_scale",
    "rows",
    "c_nope",
    "c_rope",
    "kpe_scale",
    "ckv_scale",
)

# Filled mechanically by the exporter (the derived leaves ``kernel_symbol``, ``launch``, ``arg_plan``,
# ``logical``); the host plan (``stage``, ``variant``, ``select_rule``, ``grid_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal: FlashInfer's manifest checker and the exporter read this
# file with ``ast.literal_eval``.  Record names are the stages, suffixed with the variant where a stage has
# several programs.
MODULES: dict[str, dict[str, Any]] = {
    "decode": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "QN"],
            ["buffer", "QS"],
            ["buffer", "QR"],
            ["buffer", "CKV"],
            ["buffer", "CKS"],
            ["buffer", "KPE"],
            ["buffer", "ckv_words"],
            ["buffer", "cks_words"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "q_scale"],
            ["buffer", "item_table"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "row_limit"],
            ["parameter", "num_heads"],
            ["parameter", "max_pages"],
            ["parameter", "items"],
            ["parameter", "max_splits"],
            ["parameter", "ckv_stride_words"],
            ["parameter", "cks_stride_words"],
            ["parameter", "logit_scale"],
            ["parameter", "o_factor"],
            ["buffer", "O_T"],
            ["buffer", "PO_T"],
            ["buffer", "O_T16"],
            ["buffer", "PO_T16"],
            ["parameter", "n_full"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "ctas_per_item", "ctas": 4},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_a91f9079af0188f9b47c",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 332800,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_a91f9079af0188f9b47c",
        "logical": {
            "CKS": "CKS",
            "CKV": "CKV",
            "KPE": "KPE",
            "LSE": "LSE",
            "O": "O",
            "O_T": "O_T",
            "O_T16": "O_T16",
            "PO_T": "PO_T",
            "PO_T16": "PO_T16",
            "QN": "QN",
            "QR": "QR",
            "QS": "QS",
            "cks_stride_words": "cks_stride_words",
            "cks_words": "cks_words",
            "ckv_stride_words": "ckv_stride_words",
            "ckv_words": "ckv_words",
            "item_table": "item_table",
            "items": "items",
            "logit_scale": "logit_scale",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "n_full": "n_full",
            "num_heads": "num_heads",
            "o_factor": "o_factor",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_scale": "q_scale",
            "row_limit": "row_limit",
            "seq_lens": "seq_lens",
        },
        "select_rule": {"kind": "all_full", "value": False},
        "stage": "decode",
        "variant": "mixed",
    },
    "decode_full": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "QN"],
            ["buffer", "QS"],
            ["buffer", "QR"],
            ["buffer", "CKV"],
            ["buffer", "CKS"],
            ["buffer", "KPE"],
            ["buffer", "ckv_words"],
            ["buffer", "cks_words"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "q_scale"],
            ["buffer", "item_table"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "row_limit"],
            ["parameter", "num_heads"],
            ["parameter", "max_pages"],
            ["parameter", "items"],
            ["parameter", "max_splits"],
            ["parameter", "ckv_stride_words"],
            ["parameter", "cks_stride_words"],
            ["parameter", "logit_scale"],
            ["parameter", "o_factor"],
            ["buffer", "O_T"],
            ["buffer", "PO_T"],
            ["buffer", "O_T16"],
            ["buffer", "PO_T16"],
            ["parameter", "n_full"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "ctas_per_item", "ctas": 4},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_06e6c38c07d47bffc22e",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 332800,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_06e6c38c07d47bffc22e",
        "logical": {
            "CKS": "CKS",
            "CKV": "CKV",
            "KPE": "KPE",
            "LSE": "LSE",
            "O": "O",
            "O_T": "O_T",
            "O_T16": "O_T16",
            "PO_T": "PO_T",
            "PO_T16": "PO_T16",
            "QN": "QN",
            "QR": "QR",
            "QS": "QS",
            "cks_stride_words": "cks_stride_words",
            "cks_words": "cks_words",
            "ckv_stride_words": "ckv_stride_words",
            "ckv_words": "ckv_words",
            "item_table": "item_table",
            "items": "items",
            "logit_scale": "logit_scale",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "n_full": "n_full",
            "num_heads": "num_heads",
            "o_factor": "o_factor",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_scale": "q_scale",
            "row_limit": "row_limit",
            "seq_lens": "seq_lens",
        },
        "select_rule": {
            "kind": "all_full",
            "value": True,
            "except": [[49152, 49152, 618, 618, 618, 618, 4, 4]],
        },
        "stage": "decode",
        "variant": "full",
    },
    "decode_full_pairpipe": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "QN"],
            ["buffer", "QS"],
            ["buffer", "QR"],
            ["buffer", "CKV"],
            ["buffer", "CKS"],
            ["buffer", "KPE"],
            ["buffer", "ckv_words"],
            ["buffer", "cks_words"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "q_scale"],
            ["buffer", "item_table"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "q_indptr"],
            ["buffer", "row_limit"],
            ["parameter", "num_heads"],
            ["parameter", "max_pages"],
            ["parameter", "items"],
            ["parameter", "max_splits"],
            ["parameter", "ckv_stride_words"],
            ["parameter", "cks_stride_words"],
            ["parameter", "logit_scale"],
            ["parameter", "o_factor"],
            ["buffer", "O_T"],
            ["buffer", "PO_T"],
            ["buffer", "O_T16"],
            ["buffer", "PO_T16"],
            ["parameter", "n_full"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "ctas_per_item", "ctas": 4},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_e8b2e957ca01007cc461",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 332800,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_e8b2e957ca01007cc461",
        "logical": {
            "CKS": "CKS",
            "CKV": "CKV",
            "KPE": "KPE",
            "LSE": "LSE",
            "O": "O",
            "O_T": "O_T",
            "O_T16": "O_T16",
            "PO_T": "PO_T",
            "PO_T16": "PO_T16",
            "QN": "QN",
            "QR": "QR",
            "QS": "QS",
            "cks_stride_words": "cks_stride_words",
            "cks_words": "cks_words",
            "ckv_stride_words": "ckv_stride_words",
            "ckv_words": "ckv_words",
            "item_table": "item_table",
            "items": "items",
            "logit_scale": "logit_scale",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "n_full": "n_full",
            "num_heads": "num_heads",
            "o_factor": "o_factor",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_scale": "q_scale",
            "row_limit": "row_limit",
            "seq_lens": "seq_lens",
        },
        "select_rule": {
            "kind": "plan_within",
            "boxes": [[49152, 49152, 618, 618, 618, 618, 4, 4]],
        },
        "stage": "decode",
        "variant": "full_pairpipe",
    },
    "merge_w1": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "O"],
            ["buffer", "lse"],
            ["parameter", "rows"],
            ["parameter", "num_split"],
            ["parameter", "has_lse"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 8},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_3a007c4be08b7b22f587",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_3a007c4be08b7b22f587",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {
            "kind": "rows_splits_within",
            "boxes": [[1665, 2048, 1, 8], [2049, 0, 1, 64]],
        },
        "stage": "merge",
        "variant": "w1",
    },
    "merge_w4": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "O"],
            ["buffer", "lse"],
            ["parameter", "rows"],
            ["parameter", "num_split"],
            ["parameter", "has_lse"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 2},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_66cbdc7d0d294087c5da",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_66cbdc7d0d294087c5da",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {
            "kind": "rows_splits_within",
            "boxes": [[449, 2048, 17, 32]],
        },
        "stage": "merge",
        "variant": "w4",
    },
    "merge_w2_pal": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "O"],
            ["buffer", "lse"],
            ["parameter", "rows"],
            ["parameter", "num_split"],
            ["parameter", "has_lse"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 4},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_24df5b75726c0c32ca0e",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_24df5b75726c0c32ca0e",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {
            "kind": "rows_splits_within",
            "boxes": [[385, 1664, 1, 16], [1665, 2048, 9, 16]],
        },
        "stage": "merge",
        "variant": "w2_pal",
    },
    "merge_w8_pal": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "O"],
            ["buffer", "lse"],
            ["parameter", "rows"],
            ["parameter", "num_split"],
            ["parameter", "has_lse"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 1},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_b0f5ae1afdea3c25afbf",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_b0f5ae1afdea3c25afbf",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {
            "kind": "rows_splits_within",
            "boxes": [[1, 384, 1, 32], [1, 384, 49, 64], [385, 2048, 33, 64]],
        },
        "stage": "merge",
        "variant": "w8_pal",
    },
    "merge_w8_b48_pal": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "O"],
            ["buffer", "lse"],
            ["parameter", "rows"],
            ["parameter", "num_split"],
            ["parameter", "has_lse"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 1},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_c250bbe9796073f50725",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_c250bbe9796073f50725",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {
            "kind": "rows_splits_within",
            "boxes": [[1, 384, 33, 48]],
        },
        "stage": "merge",
        "variant": "w8_b48_pal",
    },
    "merge_w4_c4_col_pal": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "partial_o"],
            ["buffer", "partial_lse"],
            ["buffer", "O"],
            ["buffer", "lse"],
            ["parameter", "rows"],
            ["parameter", "num_split"],
            ["parameter", "has_lse"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 1},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_2fcc8b55756c2d3bb4ed",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 1024,
            "use_pdl": "runtime",
        },
        "library": "cake_rubin_mla_nvfp4_decode_2fcc8b55756c2d3bb4ed",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {
            "kind": "rows_splits_within",
            "boxes": [[385, 448, 17, 32]],
        },
        "stage": "merge",
        "variant": "w4_c4_col_pal",
    },
    "quantize": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "q"],
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
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 4},
        "kernel_symbol": "kernel_cake_rubin_mla_nvfp4_decode_302684a62be285e34842",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rubin_mla_nvfp4_decode_302684a62be285e34842",
        "logical": {
            "c_nope": "c_nope",
            "c_rope": "c_rope",
            "ckv_scale": "ckv_scale",
            "kpe_scale": "kpe_scale",
            "q": "q",
            "q_nope": "q_nope",
            "q_rope": "q_rope",
            "q_scale": "q_scale",
            "q_sf": "q_sf",
            "rows": "rows",
        },
        "select_rule": {"kind": "always"},
        "stage": "quantize",
        "variant": None,
    },
}


def records_of_stage(stage: str) -> tuple[str, ...]:
    """The record names of ``stage`` in registry order."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {STAGES}")
    return tuple(name for name, rec in MODULES.items() if rec["stage"] == stage)


def _plan_within(
    box: list[Any], rows: int, n_full: int, items: int, splits: int
) -> bool:
    """Whether the plan scalars lie in the inclusive box ``[rows_lo, rows_hi,
    n_full_lo, n_full_hi, items_lo, items_hi, splits_lo, splits_hi]`` (a ``hi``
    of 0 is unbounded)."""
    bounds = [int(v) for v in box]
    return all(
        bounds[2 * axis] <= value <= (bounds[2 * axis + 1] or value)
        for axis, value in enumerate((rows, n_full, items, splits))
    )


def _admits(
    rule: dict[str, Any], *, rows: int, n_full: int, items: int, splits: int = 0
) -> bool:
    kind = rule["kind"]
    if kind == "rows_splits_within":
        # Inclusive boxes [rows_lo, rows_hi, splits_lo, splits_hi]; rows_hi 0 is unbounded.
        return any(
            int(lo) <= rows <= (int(hi) or rows) and int(slo) <= splits <= int(shi)
            for lo, hi, slo, shi in rule["boxes"]
        )
    if kind == "always":
        return True
    if kind == "all_full":
        # The plan's fullness is the rule's value and its scalars lie in none
        # of the boxes the rule excepts.
        if (n_full == items) != bool(rule["value"]):
            return False
        return not any(
            _plan_within(box, rows, n_full, items, splits)
            for box in rule.get("except", ())
        )
    if kind == "plan_within":
        return any(
            _plan_within(box, rows, n_full, items, splits) for box in rule["boxes"]
        )
    if kind == "rows_above":
        return rows > int(rule["value"])
    if kind == "rows_between":
        return int(rule["lo"]) <= rows <= int(rule["hi"])
    if kind == "rows_at_most":
        return rows <= int(rule["value"])
    raise ValueError(f"unknown Cake Rubin NVFP4 MLA decode selection rule: {kind!r}")


def select_record(
    stage: str, *, rows: int, n_full: int, items: int, splits: int = 0
) -> str:
    """The one record of ``stage`` the plan scalars select.

    ``rows`` is the packed (token, head) row count of the call, ``items`` the
    plan's work-item count, ``n_full`` how many of them are full-tile items
    (the plan lists those first) and ``splits`` the partial slots per row.
    The ``decode`` stage switches on ``all_full`` and ``plan_within``:
    ``decode_full_pairpipe`` when the four scalars lie in one of its boxes
    (``plan_within``: inclusive boxes of rows, full-tile items, items and
    splits, which ``decode_full``'s rule excepts), else ``decode_full`` when
    ``n_full == items`` (no half tile), else ``decode``.  The ``merge`` stage switches on the row count and the
    partials per row ``splits`` (``rows_splits_within``: inclusive boxes of rows and partials; eight warps per row up to 384 rows, its 48-slot program at 33-48 partials, or, up to 2048 rows, over 32 partials; otherwise up to 2048 rows the fewest
    warps whose one load round holds the partials, one up to 8, two up to 16, four up to 32, two staying up to
    1664 rows; one warp above 2048 rows); whether the
    merge runs at all (``max_splits > 1``) is the host's decision, not a rule.
    ``always`` is the rule of a single-program stage.  Exactly one record
    admits every problem, or the registry is inconsistent.
    """
    if stage in ("merge", "quantize") and int(rows) < 1:
        raise ValueError(f"stage {stage!r}: rows must be at least 1, got {rows}")
    admitted = [
        name
        for name in records_of_stage(stage)
        if _admits(
            MODULES[name]["select_rule"],
            rows=int(rows),
            n_full=int(n_full),
            items=int(items),
            splits=int(splits),
        )
    ]
    if len(admitted) != 1:
        raise ValueError(
            f"stage {stage!r}: records {admitted} admit rows={rows}, n_full={n_full}, "
            f"items={items}; exactly one must"
        )
    return admitted[0]


def launch_grid(name: str, *, items: int, rows: int) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its recorded grid rule.

    ``ctas_per_item`` launches ``(ctas * items, 1, 1)`` (the decode: four CTAs
    per work item, the clusters of four and the pairs decoded from the CTA
    id); ``rows_per_cta`` launches ``(ceil(rows / rows_per_cta), 1, 1)`` (the
    merge: ``8 / warps_per_row`` rows per CTA; the quantizer: four rows per
    CTA).
    """
    rule = MODULES[name]["grid_rule"]
    kind = rule["kind"]
    if kind == "ctas_per_item":
        if int(items) < 1:
            raise ValueError(f"record {name}: items={items} must be at least one")
        return (int(rule["ctas"]) * int(items), 1, 1)
    if kind == "rows_per_cta":
        if int(rows) < 1:
            raise ValueError(f"record {name}: rows={rows} must be at least one")
        per = int(rule["rows"])
        return (-(-int(rows) // per), 1, 1)
    raise ValueError(f"unknown Cake Rubin NVFP4 MLA decode grid rule: {kind!r}")


def merge_warps_per_row(rows: int, splits: int) -> int:
    """Warps per row of the merge record ``select_record("merge", ...)`` picks for ``rows`` rows of ``splits``
    partials each: eight warps per row up to 384 rows or, up to 2048 rows, over 32 partials; otherwise up to 2048 rows the fewest
    warps whose one load round holds the partials, one up to 8, two up to 16, four up to 32, two staying up to
    1664 rows; one warp above 2048 rows (fewer, longer CTAs measured faster under the decode's dependent launch
    while their grid shrinks within one wave)."""
    if rows > 2048:
        return 1
    if rows <= 384 or splits > 32:
        return 8
    if splits > 16:
        return 4
    if splits > 8 or rows <= 1664:
        return 2
    return 1


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake Rubin NVFP4 MLA decode target; the exported architectures are {ARCHES}"
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
            f"Cake Rubin NVFP4 MLA decode record {name} is not delivered for {arch}; "
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
def _argument_plan(name: str) -> tuple[tuple[str, Any], ...]:
    """The record's plan in ``run`` order, computed once: ``("value", logical
    name)`` per kernel parameter (buffers, TMA operands and scalars alike),
    ``("launch", None)`` for the per-call PDL switch of a record under
    ``use_pdl = "runtime"`` and ``("grid", axis)`` per grid argument."""
    record = MODULES[name]
    logical = record["logical"]
    if not record["arg_plan"]:
        raise RuntimeError(
            f"record {name} has no argument plan: the package was not exported "
            "(the exporter writes the derived leaves of cake_jit.MODULES)"
        )
    plan: list[tuple[str, Any]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append(("grid", "xyz".index(key[-1])))
        elif kind == "launch":
            if key != LAUNCH_ARGUMENT:
                raise ValueError(
                    f"record {name}: launch argument {key!r} is not {LAUNCH_ARGUMENT!r}"
                )
            plan.append(("launch", None))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append(("value", logical[key]))
    return tuple(plan)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
    enable_pdl: int = 0,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to
    tensors and Python scalars; the record maps its kernel parameters onto that
    vocabulary and ``grid`` is the plan's launch grid (:func:`launch_grid`).
    ``enable_pdl`` (0 or 1) is the per-call programmatic-dependent-launch
    switch of a record under ``launch.use_pdl = "runtime"`` (the decode and the
    merge; the host sets it from the plan, ``cake_backend.pdl_for``); a record
    launched with a fixed attribute (the quantizer) takes none, so a truthy
    ``enable_pdl`` for it is refused.  The kernel runs on the current torch
    stream of the device.
    """
    record = MODULES[name]
    plan = _argument_plan(name)
    if enable_pdl and ("launch", None) not in plan:
        raise ValueError(
            f"record {name} takes no per-call launch switch (launch.use_pdl is "
            f"{record['launch']['use_pdl']!r}); enable_pdl must be 0"
        )
    args = []
    for kind, key in plan:
        if kind == "grid":
            args.append(int(grid[key]))
        elif kind == "launch":
            args.append(int(bool(enable_pdl)))
        else:
            try:
                args.append(values[key])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {key!r}"
                ) from None
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "DECODE_PARAMETERS",
    "LAUNCH_ARGUMENT",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_SPLITS",
    "MERGE_PARAMETERS",
    "MODULES",
    "PREFIX",
    "QUANTIZE_PARAMETERS",
    "STAGES",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "merge_warps_per_row",
    "records_of_stage",
    "run_module",
    "select_record",
    "sm_count",
    "specs_for_arch",
]
