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
from typing import Any

from . import cake_build

# Explicit target-owned registration of the generated chunked LM-head + loss
# programs, one record per architecture (the kernels take the call geometry
# H / V as launch scalars; one record serves every admissible geometry).  The
# generated-program export writes the compact tables between the two
# ``generated registry`` markers; ``MODULES`` is their expansion.  A record
# carries ``arch``, the host binding profile ``abi`` (the keyword set its
# kernels expect, see ``cake_backend``), the list of kernel ``stages`` it
# registers (launch order), the ``geometry`` the kernels were built for (see
# ``cake_backend.Geometry``: the vocabulary columns per row-statistics
# partial, the GEMM row tile and CTA pair, the K block of the weight-gradient
# GEMM, the element vector of the cast, the divisibility ``V`` / ``H`` / the
# row stride of ``X`` must satisfy, the label element type and, per GEMM, the
# cluster width, the work item's columns and the default raster height) and
# one physical entry per stage: ``module`` (the generated program),
# ``library`` (the architecture-free stem of the stage's library; ``build.json``
# names the per-architecture build ``<library>_<arch>``, see ``cake_build``),
# ``ffi_entry``, ``arg_plan``, ``tma_workspace_bytes``, ``workspace_bytes``,
# ``grid`` and ``launch``.  What to compile -- the translation units, the nvcc
# flags (including the toolchain-conditional ``nvcc_flags_if``) and the
# closure digest -- is recorded in ``build.json`` only, never here.
#
# Compact form: ``_ARG_PLANS`` and ``_LAUNCHES`` hold the distinct argument
# plans and launch geometries by name; a stage row is
# ``[unit, arg_plan, launch, grid_x, grid_y, grid_z]`` plus an optional
# trailing mapping of the fields that differ from their defaults
# (``ffi_entry`` ``"run"``, ``tma_workspace_bytes`` and ``workspace_bytes``
# ``0``).  The unit names the stage's program and library stem
# ``cake_lm_head_loss_<unit>`` (one architecture-neutral translation unit per
# kernel, shared by the records of every architecture; the per-architecture
# lowering sits under ``__CUDA_ARCH__`` guards) whose ``build.json`` entry
# ``cake_lm_head_loss_<unit>_<arch>`` lists the kernel translation unit that
# defines the stage's symbol -- the unit's own ``csrc/cake_lm_head_loss/
# cake_lm_head_loss_<unit>_kernel.cu``, or a sibling form's file with this
# unit's kernel symbol named through a compile definition -- and the launch
# stub ``..._launch.cu`` that binds the shared launcher under
# ``csrc/cake_lm_head_loss/shim/`` to the kernel.  Do not edit by hand.
# --- generated registry (written by the Cake export; do not edit) ---
_ARG_PLANS: dict[str, list[list[str]]] = {
    "gather_rows_bf16": [
        ["buffer", "x"],
        ["buffer", "idx_lo"],
        ["buffer", "count"],
        ["buffer", "out"],
        ["parameter", "ld_x_words"],
        ["parameter", "row_vecs"],
        ["parameter", "num_rows"],
        ["parameter", "T"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_logits": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "C"],
        ["buffer", "STATS_OUT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "first_chunk"],
        ["buffer", "WS"],
        ["parameter", "ws_slab"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["parameter", "group_n"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_logits_dz": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "C"],
        ["buffer", "STATS_OUT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "first_chunk"],
        ["buffer", "WS"],
        ["parameter", "ws_slab"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["parameter", "group_n"],
        ["buffer", "labels"],
        ["buffer", "lse"],
        ["buffer", "dlogp"],
        ["parameter", "row0"],
        ["parameter", "d_off"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_logits_dz_tma": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C"],
        ["buffer", "STATS_OUT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "first_chunk"],
        ["buffer", "WS"],
        ["parameter", "ws_slab"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["parameter", "group_n"],
        ["buffer", "labels"],
        ["buffer", "lse"],
        ["buffer", "dlogp"],
        ["parameter", "row0"],
        ["parameter", "d_off"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_logits_lbl": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "C"],
        ["buffer", "STATS_OUT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "first_chunk"],
        ["buffer", "WS"],
        ["parameter", "ws_slab"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["parameter", "group_n"],
        ["buffer", "labels"],
        ["parameter", "row0"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "row_finalize": [
        ["buffer", "stats"],
        ["buffer", "z"],
        ["buffer", "labels"],
        ["buffer", "infer_logp"],
        ["buffer", "loss_weights"],
        ["buffer", "d_in"],
        ["buffer", "lse"],
        ["buffer", "logp"],
        ["buffer", "d"],
        ["buffer", "term"],
        ["parameter", "rows_c"],
        ["parameter", "row0"],
        ["parameter", "V"],
        ["parameter", "num_tiles"],
        ["parameter", "mode"],
        ["parameter", "loss_div"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "loss_reduce": [
        ["buffer", "term"],
        ["buffer", "loss_acc"],
        ["buffer", "loss_out"],
        ["parameter", "rows_c"],
        ["parameter", "first_chunk"],
        ["parameter", "last_chunk"],
        ["parameter", "mode"],
        ["parameter", "loss_div"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "row_grad": [
        ["buffer", "z"],
        ["buffer", "labels"],
        ["buffer", "lse"],
        ["buffer", "d"],
        ["parameter", "row0"],
        ["parameter", "d_off"],
        ["parameter", "V"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_dx": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C"],
        ["buffer", "STATS_OUT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "first_chunk"],
        ["buffer", "WS"],
        ["parameter", "ws_slab"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["parameter", "group_n"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_dx_2": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C"],
        ["buffer", "STATS_OUT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "first_chunk"],
        ["tma_buffer", "WS"],
        ["parameter", "ws_slab"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["parameter", "group_n"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_dx_ptr": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "C"],
        ["buffer", "STATS_OUT"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "first_chunk"],
        ["buffer", "WS"],
        ["parameter", "ws_slab"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["parameter", "group_n"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_dx_red": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C"],
        ["parameter", "M"],
        ["parameter", "m_tiles"],
        ["parameter", "k_iters"],
        ["parameter", "ldc"],
        ["parameter", "k_slices"],
        ["parameter", "k_slice_iters"],
        ["parameter", "group_m"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "scale_cast_bf16": [
        ["buffer", "acc"],
        ["buffer", "g"],
        ["buffer", "out"],
        ["parameter", "num_vecs"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "scale_cast_scatter_bf16": [
        ["buffer", "acc"],
        ["buffer", "g"],
        ["buffer", "scan"],
        ["buffer", "idx_lo"],
        ["buffer", "out"],
        ["parameter", "row_vecs"],
        ["parameter", "num_rows"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_check": [
        ["buffer", "ids"],
        ["buffer", "offsets"],
        ["buffer", "labels"],
        ["buffer", "word"],
        ["buffer", "lpos"],
        ["buffer", "offsets32"],
        ["buffer", "ids32"],
        ["parameter", "T"],
        ["parameter", "M"],
        ["parameter", "V"],
        ["parameter", "k_cap"],
        ["parameter", "has_labels"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_row_finalize": [
        ["buffer", "z"],
        ["buffer", "tau"],
        ["buffer", "offsets"],
        ["buffer", "lpos"],
        ["buffer", "teacher"],
        ["buffer", "lse"],
        ["buffer", "packed"],
        ["buffer", "selected"],
        ["buffer", "kl"],
        ["parameter", "T"],
        ["parameter", "M"],
        ["parameter", "outputs"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_row_grad": [
        ["buffer", "z"],
        ["buffer", "tau"],
        ["buffer", "lse"],
        ["buffer", "offsets"],
        ["buffer", "lpos"],
        ["buffer", "teacher"],
        ["buffer", "g"],
        ["buffer", "d"],
        ["buffer", "e"],
        ["buffer", "dz"],
        ["parameter", "T"],
        ["parameter", "M"],
        ["parameter", "upstreams"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_gather_bf16": [
        ["buffer", "zblk"],
        ["buffer", "ids"],
        ["buffer", "offsets"],
        ["buffer", "z"],
        ["parameter", "t0"],
        ["parameter", "rows"],
        ["parameter", "v0"],
        ["parameter", "vb"],
        ["parameter", "ldc"],
        ["parameter", "M"],
        ["parameter", "V"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_scatter_dz_bf16": [
        ["buffer", "dz"],
        ["buffer", "ids"],
        ["buffer", "offsets"],
        ["buffer", "dzblk"],
        ["parameter", "t0"],
        ["parameter", "rows"],
        ["parameter", "v0"],
        ["parameter", "vb"],
        ["parameter", "ldc"],
        ["parameter", "M"],
        ["parameter", "V"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_logits_gather_gemm": [
        ["tma_buffer", "W"],
        ["tma_buffer", "X"],
        ["buffer", "z"],
        ["buffer", "ids"],
        ["buffer", "offsets"],
        ["parameter", "M"],
        ["parameter", "T"],
        ["parameter", "V"],
        ["parameter", "H"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_sga_dx": [
        ["buffer", "rows"],
        ["buffer", "scale"],
        ["buffer", "ids"],
        ["buffer", "offsets"],
        ["buffer", "out"],
        ["parameter", "ld_rows"],
        ["parameter", "H"],
        ["parameter", "V"],
        ["parameter", "M"],
        ["parameter", "T"],
        ["parameter", "min_pairs"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_sga_dw": [
        ["buffer", "rows"],
        ["buffer", "scale"],
        ["buffer", "keys"],
        ["buffer", "offsets"],
        ["buffer", "out"],
        ["buffer", "partial"],
        ["parameter", "ld_rows"],
        ["parameter", "H"],
        ["parameter", "V"],
        ["parameter", "M"],
        ["parameter", "T"],
        ["parameter", "n_keys"],
        ["parameter", "split"],
        ["parameter", "span"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "support_sga_combine": [
        ["buffer", "keys"],
        ["buffer", "partial"],
        ["buffer", "out"],
        ["parameter", "H"],
        ["parameter", "V"],
        ["parameter", "n_keys"],
        ["parameter", "split"],
        ["parameter", "span"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}
_LAUNCHES: dict[str, dict[str, list[int]]] = {
    "b256_c1": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    "b224_c2": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
    "b128_c1": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
    "b320_c1": {"block": [320, 1, 1], "cluster": [1, 1, 1]},
}
_REGISTRY: dict[str, dict[str, Any]] = {
    "cake_lm_head_loss_sm_100a": {
        "arch": "sm_100a",
        "abi": "lm_head_loss_v2",
        "geometry": {
            "stats_tile": 256,
            "row_tile": 128,
            "cta_group": 2,
            "k_block": 64,
            "cast_vec": 8,
            "vocab_multiple": 256,
            "hidden_multiple": 256,
            "ld_multiple": 8,
            "labels_dtype": "int64",
            "hidden": None,
            "vocab": None,
            "logits_cluster_ctas": 2,
            "dx_cluster_ctas": 2,
            "dw_cluster_ctas": 2,
            "logits_item_cols": 256,
            "dx_item_cols": 512,
            "dw_item_cols": 512,
            "logits_group_m": 32,
            "dx_group_m": 16,
            "dw_group_m": 8,
        },
        "stages": {
            "gather_rows_bf16": [
                "faa8f4d9322bc102406b",
                "gather_rows_bf16",
                "b256_c1",
                "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))",
                1,
                1,
            ],
            "gemm_logits": [
                "00e8c73054c99d86507a",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_mcnt": [
                "cee1ab9bd1930b12da6e",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_nostats": [
                "ad12a468f7a8ba9cc2d1",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_nostats_mcnt": [
                "0ef6f04083534aaba9af",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_dz": [
                "74e676b5299ea491d208",
                "gemm_logits_dz",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_lbl": [
                "baa5c79e0c574e090c55",
                "gemm_logits_lbl",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "row_finalize": [
                "8af5f6b171c7907d9a42",
                "row_finalize",
                "b256_c1",
                "rows_c/8",
                1,
                1,
            ],
            "loss_reduce": ["48134805ead9ce0f7416", "loss_reduce", "b256_c1", 1, 1, 1],
            "row_grad": [
                "23560b287fedb0fe4daa",
                "row_grad",
                "b128_c1",
                "max(1, V//8/1024)",
                "rows_c",
                1,
            ],
            "gemm_dx": [
                "47bb4b3e5db1ee732230",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*2",
                1,
                1,
            ],
            "gemm_dx_red": [
                "d349713eae71a22b63ca",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices)*2",
                1,
                1,
            ],
            "gemm_dx_tn256": [
                "3da632b3f70182303342",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dx_red_tn256": [
                "8fac37f18451d7ab7f8c",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices)*4",
                1,
                1,
            ],
            "gemm_dx_st3": [
                "47bb4b3e5db1ee732230",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*2",
                1,
                1,
            ],
            "gemm_dx_red_st3": [
                "d349713eae71a22b63ca",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices)*2",
                1,
                1,
            ],
            "gemm_dx_split": [
                "ddf6e091b0d2ae4d1d2d",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)+dx_split)*2",
                1,
                1,
            ],
            "gemm_dx_red_split": [
                "855c7771cd80451ba0bc",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices+dx_split)*2",
                1,
                1,
            ],
            "gemm_dx_st3_split": [
                "ddf6e091b0d2ae4d1d2d",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)+dx_split)*2",
                1,
                1,
            ],
            "gemm_dx_red_st3_split": [
                "855c7771cd80451ba0bc",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices+dx_split)*2",
                1,
                1,
            ],
            "gemm_dw_acc": [
                "909c70b0efa84b67f0fb",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_acc_gn": [
                "a3564016593b7189015e",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_bf16": [
                "fee47365dff8c41b26a6",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_bf16_tma": [
                "09ef1e4bd18f200203e8",
                "gemm_dx_2",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_f32": [
                "482c931ebbf9454ad2a9",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_f32_gn": [
                "efd33e45c3d57ceab618",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "scale_cast_bf16": [
                "bf98c810b48ec46ca531",
                "scale_cast_bf16",
                "b256_c1",
                "max(1, min(num_vecs/2048, 65535))",
                1,
                1,
            ],
            "scale_cast_f32": [
                "bdb11a3e3dd2edc86bce",
                "scale_cast_bf16",
                "b256_c1",
                "max(1, min(num_vecs/2048, 65535))",
                1,
                1,
            ],
            "scale_cast_scatter_bf16": [
                "c4947e5ca18c5b70acf2",
                "scale_cast_scatter_bf16",
                "b256_c1",
                "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))",
                1,
                1,
            ],
            "support_check_ids_i32_off_i32": [
                "785d7d1c8b05732fdcac",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_check_ids_i32_off_i64": [
                "be490d8e4e1229e23209",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_check_ids_i64_off_i32": [
                "28b60085275b93cf26f1",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_check_ids_i64_off_i64": [
                "24c31ef3b0a8d94b7c61",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_row_finalize": [
                "9560888ff69b2cb782a4",
                "support_row_finalize",
                "b256_c1",
                "min(T/8, sms*8)",
                1,
                1,
            ],
            "support_row_grad": [
                "1dda66b97cd91f34c383",
                "support_row_grad",
                "b256_c1",
                "min(T/8, sms*8)",
                1,
                1,
            ],
            "support_gather_bf16": [
                "2fd364e7bc74cecdeb06",
                "support_gather_bf16",
                "b256_c1",
                "min(rows/8, sms*8)",
                1,
                1,
            ],
            "support_scatter_dz_bf16": [
                "ca5145901164bcebaf88",
                "support_scatter_dz_bf16",
                "b256_c1",
                "min(rows, sms*8)",
                1,
                1,
            ],
            "support_logits_gather_gemm": [
                "f926c93e1045aea7d54d",
                "support_logits_gather_gemm",
                "b320_c1",
                "min(M/128, sms)",
                1,
                1,
            ],
            "support_sga_dx": [
                "cdb02947ec690f88abee",
                "support_sga_dx",
                "b128_c1",
                "T",
                "H/1024",
                1,
            ],
            "support_sga_dw": [
                "b881fe55c1861f94eb8a",
                "support_sga_dw",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
            "support_sga_dw_f32_add": [
                "7899a85c3ed1d235b1c0",
                "support_sga_dw",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
            "support_sga_combine": [
                "de69c0fe6529fe04c0aa",
                "support_sga_combine",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
            "support_sga_combine_f32_add": [
                "596408d23bf05fda10a3",
                "support_sga_combine",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
        },
    },
    "cake_lm_head_loss_sm_103a": {
        "arch": "sm_103a",
        "abi": "lm_head_loss_v2",
        "geometry": {
            "stats_tile": 256,
            "row_tile": 128,
            "cta_group": 2,
            "k_block": 64,
            "cast_vec": 8,
            "vocab_multiple": 256,
            "hidden_multiple": 256,
            "ld_multiple": 8,
            "labels_dtype": "int64",
            "hidden": None,
            "vocab": None,
            "logits_cluster_ctas": 2,
            "dx_cluster_ctas": 2,
            "dw_cluster_ctas": 2,
            "logits_item_cols": 256,
            "dx_item_cols": 512,
            "dw_item_cols": 512,
            "logits_group_m": 32,
            "dx_group_m": 16,
            "dw_group_m": 2,
        },
        "stages": {
            "gather_rows_bf16": [
                "faa8f4d9322bc102406b",
                "gather_rows_bf16",
                "b256_c1",
                "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))",
                1,
                1,
            ],
            "gemm_logits": [
                "f7c6a51ed55107c50805",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_mcnt": [
                "3883b3f0cac99ce5da5b",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_nostats": [
                "b11d843d00ca35085b0d",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_nostats_mcnt": [
                "3c126d44ffeb10c86d19",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_dz": [
                "65f522900277104d4974",
                "gemm_logits_dz_tma",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "gemm_logits_lbl": [
                "65094a9661592f6c8c82",
                "gemm_logits_lbl",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//256))*2",
                1,
                1,
            ],
            "row_finalize": [
                "8af5f6b171c7907d9a42",
                "row_finalize",
                "b256_c1",
                "rows_c/8",
                1,
                1,
            ],
            "loss_reduce": ["48134805ead9ce0f7416", "loss_reduce", "b256_c1", 1, 1, 1],
            "row_grad": [
                "23560b287fedb0fe4daa",
                "row_grad",
                "b128_c1",
                "max(1, V//8/1024)",
                "rows_c",
                1,
            ],
            "gemm_dx": [
                "47eb707694e21f2e15bc",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*2",
                1,
                1,
            ],
            "gemm_dx_red": [
                "6f0834bda13f3e7f3c4b",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices)*2",
                1,
                1,
            ],
            "gemm_dx_tn256": [
                "3da632b3f70182303342",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dx_red_tn256": [
                "8fac37f18451d7ab7f8c",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices)*4",
                1,
                1,
            ],
            "gemm_dx_st3": [
                "9a63c0f173aec8077e02",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*2",
                1,
                1,
            ],
            "gemm_dx_red_st3": [
                "b6cc83ec28675234575f",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices)*2",
                1,
                1,
            ],
            "gemm_dx_split": [
                "ffbb5726e9d16ea2787e",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)+dx_split)*2",
                1,
                1,
            ],
            "gemm_dx_red_split": [
                "254fee4132e11fc54959",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices+dx_split)*2",
                1,
                1,
            ],
            "gemm_dx_st3_split": [
                "473c4161fa8332f993b1",
                "gemm_dx",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)+dx_split)*2",
                1,
                1,
            ],
            "gemm_dx_red_st3_split": [
                "2528a3b18f768419f7ab",
                "gemm_dx_red",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512)*k_slices+dx_split)*2",
                1,
                1,
            ],
            "gemm_dx_swab": [
                "67ab493714f0d3328dfe",
                "gemm_dx_ptr",
                "b224_c2",
                "max(1, ldc//256*(M/512))*2",
                1,
                1,
            ],
            "gemm_dx_red_swab": [
                "896fe483cd77d7865548",
                "gemm_dx_red",
                "b224_c2",
                "max(1, ldc//256*(M/512)*k_slices)*2",
                1,
                1,
            ],
            "gemm_dw_acc": [
                "909c70b0efa84b67f0fb",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_acc_gn": [
                "a3564016593b7189015e",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_bf16": [
                "fee47365dff8c41b26a6",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_bf16_tma": [
                "09ef1e4bd18f200203e8",
                "gemm_dx_2",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_f32": [
                "482c931ebbf9454ad2a9",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "gemm_dw_cast_f32_gn": [
                "efd33e45c3d57ceab618",
                "gemm_logits",
                "b224_c2",
                "max(1, m_tiles//2*(ldc//512))*4",
                1,
                1,
            ],
            "scale_cast_bf16": [
                "bf98c810b48ec46ca531",
                "scale_cast_bf16",
                "b256_c1",
                "max(1, min(num_vecs/2048, 65535))",
                1,
                1,
            ],
            "scale_cast_f32": [
                "bdb11a3e3dd2edc86bce",
                "scale_cast_bf16",
                "b256_c1",
                "max(1, min(num_vecs/2048, 65535))",
                1,
                1,
            ],
            "scale_cast_scatter_bf16": [
                "c4947e5ca18c5b70acf2",
                "scale_cast_scatter_bf16",
                "b256_c1",
                "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))",
                1,
                1,
            ],
            "support_check_ids_i32_off_i32": [
                "785d7d1c8b05732fdcac",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_check_ids_i32_off_i64": [
                "be490d8e4e1229e23209",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_check_ids_i64_off_i32": [
                "28b60085275b93cf26f1",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_check_ids_i64_off_i64": [
                "24c31ef3b0a8d94b7c61",
                "support_check",
                "b256_c1",
                "min(T/8, sms*4)",
                1,
                1,
            ],
            "support_row_finalize": [
                "9560888ff69b2cb782a4",
                "support_row_finalize",
                "b256_c1",
                "min(T/8, sms*8)",
                1,
                1,
            ],
            "support_row_grad": [
                "1dda66b97cd91f34c383",
                "support_row_grad",
                "b256_c1",
                "min(T/8, sms*8)",
                1,
                1,
            ],
            "support_gather_bf16": [
                "2fd364e7bc74cecdeb06",
                "support_gather_bf16",
                "b256_c1",
                "min(rows/8, sms*8)",
                1,
                1,
            ],
            "support_scatter_dz_bf16": [
                "ca5145901164bcebaf88",
                "support_scatter_dz_bf16",
                "b256_c1",
                "min(rows, sms*8)",
                1,
                1,
            ],
            "support_logits_gather_gemm": [
                "f926c93e1045aea7d54d",
                "support_logits_gather_gemm",
                "b320_c1",
                "min(M/128, sms)",
                1,
                1,
            ],
            "support_sga_dx": [
                "cdb02947ec690f88abee",
                "support_sga_dx",
                "b128_c1",
                "T",
                "H/1024",
                1,
            ],
            "support_sga_dw": [
                "b881fe55c1861f94eb8a",
                "support_sga_dw",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
            "support_sga_dw_f32_add": [
                "7899a85c3ed1d235b1c0",
                "support_sga_dw",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
            "support_sga_combine": [
                "de69c0fe6529fe04c0aa",
                "support_sga_combine",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
            "support_sga_combine_f32_add": [
                "596408d23bf05fda10a3",
                "support_sga_combine",
                "b128_c1",
                "n_keys/span",
                "H/1024",
                1,
            ],
        },
    },
}
# --- end generated registry ---

STAGE_ROW_FIELDS = (
    "unit",
    "arg_plan",
    "launch",
    "grid_x",
    "grid_y",
    "grid_z",
)


def library_stem(unit: str) -> str:
    """Architecture-free library stem of the generated program of ``unit``; ``build.json`` names the
    per-architecture build ``<stem>_<arch>`` (:func:`library_name`)."""
    return f"cake_lm_head_loss_{unit}"


def _expand_registry(
    registry: dict[str, dict[str, Any]],
    arg_plans: dict[str, list[list[str]]],
    launches: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, Any]]:
    """The per-stage physical records of every program from the compact tables.

    Self-contained by contract with the Cake exporter: ``exports/lm_head_loss/export.py``
    ``_scaffold_expansion`` executes only ``STAGE_ROW_FIELDS`` and this function (extracted from the
    scaffold at ``TARGET_REVISION``) to check that the compact tables it renders expand to the physical
    records, so this body may use its parameters, ``STAGE_ROW_FIELDS`` and builtins only -- no other
    module-level name (``library_stem`` is spelled out as the f-string below for that reason).
    """
    modules: dict[str, dict[str, Any]] = {}
    for name, compact in registry.items():
        arch = compact["arch"]
        record: dict[str, Any] = {
            "arch": arch,
            "abi": compact["abi"],
            "stages": list(compact["stages"]),
            "geometry": dict(compact["geometry"]),
        }
        for stage, row in compact["stages"].items():
            if len(row) not in (len(STAGE_ROW_FIELDS), len(STAGE_ROW_FIELDS) + 1):
                raise ValueError(f"{name}/{stage}: malformed registry row {row!r}")
            unit, plan, launch, grid_x, grid_y, grid_z = row[: len(STAGE_ROW_FIELDS)]
            extra = (
                dict(row[len(STAGE_ROW_FIELDS)])
                if len(row) > len(STAGE_ROW_FIELDS)
                else {}
            )
            physical: dict[str, Any] = {
                "module": f"cake_lm_head_loss_{unit}",
                "library": f"cake_lm_head_loss_{unit}",
                "ffi_entry": extra.pop("ffi_entry", "run"),
                "arg_plan": [list(item) for item in arg_plans[plan]],
                "tma_workspace_bytes": int(extra.pop("tma_workspace_bytes", 0)),
                "workspace_bytes": int(extra.pop("workspace_bytes", 0)),
                "grid": [grid_x, grid_y, grid_z],
                "launch": {key: list(value) for key, value in launches[launch].items()},
            }
            physical.update(extra)
            record[stage] = physical
        modules[name] = record
    return modules


MODULES: dict[str, dict[str, Any]] = _expand_registry(_REGISTRY, _ARG_PLANS, _LAUNCHES)

# Kernel stages of one token chunk of the training step, in launch order.
#
# ``gemm_logits``          z_c = bf16(X_c @ W^T) for the chunk's rows plus the
#                          per-(row, vocabulary tile) online (max, sum-exp)
#                          partials of the bf16-rounded logits.
# ``gemm_logits_nostats``  the same GEMM without the statistics (the
#                          recompute of the log-probability backward).
# ``gemm_logits_dz``       the log-probability backward's recompute with the row
#                          gradient in its epilogue: dz_c = bf16(dlogp_t *
#                          (1[v = y_t] - exp(z - lse_t))) in place of z_c
#                          (bitwise ``gemm_logits_nostats`` + ``row_grad``).
# ``gemm_logits_lbl``      the log-probability forward's logits GEMM: the
#                          statistics plus z_c at the label column alone (the
#                          one element ``row_finalize`` reads).
# ``row_finalize``         merges the partials into ``lse`` (every row),
#                          gathers the selected logit, writes ``logp`` (0 on
#                          ignored rows) and, per objective, the per-row
#                          logit-gradient scale ``d`` and loss term.
# ``loss_reduce``          fixed-order sum of the chunk's loss terms into the
#                          loss accumulator (chunk order); the last chunk
#                          writes the finished loss.
# ``row_grad``             dz_c = d_t * (1[v = y_t] - exp(z - lse_t)) in bf16,
#                          in place over ``z_c``; ignored rows become zero.
# ``gemm_dx``              dX_acc[rows] = fp32(dz_c @ W) (one K slice, plain
#                          store: the deterministic package's program).
# ``gemm_dx_red``          the same GEMM as ``k_slices`` (a launch scalar)
#                          K-slice work items per output tile, every slice's
#                          FP32 tile added into the chunk's rows of dX_acc by a
#                          TMA / bulk ``.add.f32`` reduction in the order the
#                          items finish (no slab workspace, no slab sum); the
#                          host fills those rows with -0.0 first and picks the
#                          slice count per chunk (a chunk of one slice runs
#                          ``gemm_dx``).
# ``gemm_dw_acc``          dW_acc (=|+=) fp32(dz_c^T @ X_c): store on the first
#                          chunk, accumulate afterwards (chunk order = the
#                          reduction order, no atomics).
# ``gemm_dw_cast_bf16``    the LAST chunk's dz_c^T @ X_c with the upstream scale
# ``gemm_dw_cast_f32``     and the output cast fused into the epilogue:
#                          dW = cast(g * (dW_acc + tile)) -- or cast(g * tile)
#                          for a one-chunk plan -- run in the backward where
#                          g is known (fuse_dw_cast; bitwise gemm_dw_acc +
#                          scale_cast, one pass over dW fewer).
# ``scale_cast_bf16``      out = bf16(g * acc) over a flat fp32 accumulator
# ``scale_cast_f32``       out = g * acc (fp32) -- the single output cast of
#                          ``dW`` (either) and ``dX`` (bf16) in the backward.
# ``gather_rows_bf16``     chunk 0's row gather of the hidden valid-row count
#                          path: out[r] = X[idx[r]] for r < min(count, num_rows)
#                          with the count read from device memory, exact zeros
#                          after (launched before chunk 0's logits GEMM).
# ``gemm_logits_mcnt``     the logits GEMMs (with / without the statistics)
# ``gemm_logits_nostats_mcnt``  whose valid-row bound is read from device
#                          memory: chunk 0's GEMM of the hidden valid-row count
#                          path, its stores bounded by min(count, M) while M is
#                          the buffer extent min(chunk, T).
#
# A record registers the subset its program uses; the host refuses an entry
# point whose stages are missing.
STAGES = (
    "gather_rows_bf16",
    "gemm_logits",
    "gemm_logits_mcnt",
    "gemm_logits_nostats",
    "gemm_logits_nostats_mcnt",
    "gemm_logits_dz",
    "gemm_logits_lbl",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "gemm_dx",
    "gemm_dx_tn256",
    "gemm_dx_st3",
    "gemm_dx_split",
    "gemm_dx_st3_split",
    "gemm_dx_swab",
    "gemm_dx_red",
    "gemm_dx_red_tn256",
    "gemm_dx_red_st3",
    "gemm_dx_red_split",
    "gemm_dx_red_st3_split",
    "gemm_dx_red_swab",
    "gemm_dw_acc",
    "gemm_dw_acc_gn",
    "gemm_dw_cast_bf16",
    "gemm_dw_cast_bf16_tma",
    "gemm_dw_cast_f32",
    "gemm_dw_cast_f32_gn",
    "scale_cast_bf16",
    "scale_cast_f32",
    "scale_cast_scatter_bf16",
)
# The base stages (one per kernel role of the chunk loop).  Every other name in
# ``STAGES`` is a structural FORM of a base GEMM stage -- a distinct kernel body,
# named by the outputs of the host's per-chunk instance rules that select one
# (``cake_backend.stage_variant``): ``_mcnt`` the device-count form of the
# logits GEMMs (the hidden valid-row count path's chunk 0), ``_tn256`` the narrow
# tile and ``_st3`` the 3-deep operand ring of the dX GEMMs (``gemm_dx`` and its
# K-sliced reduction form ``gemm_dx_red``, a base stage of its own), ``_split`` the wide
# dX tile whose last wave's trailing items run as 256-wide half items (the grid
# carries the split count), ``_swab`` the wide dX tile with swapped operands
# (dX^T = W^T dz^T: the chunk's tokens on the MMA's N; ``gemm_dx_swab`` stores
# C per thread, ``gemm_dx_red_swab`` adds 32 x 16 boxes by TMA reduce),
# ``_tma`` the TMA-epilogue form of a GEMM's stores (the bf16 fused weight-
# gradient cast of a multi-chunk call's last chunk reads the old FP32 sum and
# stores the bf16 tile as TMA sub-tiles: the same per-element arithmetic),
# ``_gn`` the 2-D blocked raster of the weight-gradient accumulate / fp32 fused cast.  The
# call geometry (``H`` / ``V``) and the rules' numeric outputs -- the slice
# count, the raster height ``group_m``, the block width ``group_n`` -- are
# launch scalars of these kernels (``cake_backend.gemm_scalars``), so one
# record per architecture serves every admissible geometry.  A record
# registers the base stages plus the forms its architecture's rules can
# reach.  ``scale_cast_scatter_bf16`` (the one-pass scale, cast and scatter of
# a compacted ``dX``) is the fused dX finalize's kernel
# (``cake_backend.DX_FINALIZE_STAGES``); a row kernel, geometry-generic.
# ``gather_rows_bf16`` (chunk 0's row gather of the hidden valid-row count path,
# ``cake_backend.HIDDEN_COUNT_STAGE``) is a row kernel too.
BASE_STAGES = (
    "gather_rows_bf16",
    "gemm_logits",
    "gemm_logits_nostats",
    "gemm_logits_dz",
    "gemm_logits_lbl",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "gemm_dx",
    "gemm_dx_red",
    "gemm_dw_acc",
    "gemm_dw_cast_bf16",
    "gemm_dw_cast_f32",
    "scale_cast_bf16",
    "scale_cast_f32",
    "scale_cast_scatter_bf16",
)
GEMM_STAGES = tuple(stage for stage in STAGES if stage.startswith("gemm_"))
ROW_STAGES = (
    "gather_rows_bf16",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "scale_cast_bf16",
    "scale_cast_f32",
    "scale_cast_scatter_bf16",
)


@functools.cache
def toolchain_runs_hidden_count(arch: str) -> bool:
    """Does the hidden valid-row count (``cake_backend.hidden_count_eligible``) run on ``arch`` with the CUDA
    toolkit that built (or will build) this package's libraries?

    On CUDA 13.0 the sm_103a programs are built at ptxas -O0 (``build.json`` carries ``["-Xptxas", "-O0"]``
    under ``nvcc_flags_if`` for ``cuda == 13.0``: that toolchain's ptxas mis-schedules the sm_103a 2-CTA TMA
    producer loops of these programs, see the package README).  The -O1 code completed the focused dW
    accumulate calls and the test file's shipped call schedule, but with the hidden count's schedule -- the
    compaction index, chunk 0's row gather and device-count logits GEMM queued before the count is read back --
    the test file still reaches a ``cudaErrorIllegalInstruction`` from the cluster launch of the dW accumulate
    GEMM at its shortest K schedule (the one-row tail chunk of a two-chunk call), in every run; with the host
    count it passes, as does the -O0 code with the hidden count.  So on CUDA 13.0 the sm_103a calls take the
    host-count path (the same kernels per row, so bitwise the same outputs; the count and the index are formed
    on the host before the first launch).  Every other architecture / toolchain pair runs the hidden count.  A
    mitigation of the ptxas 13.0 fault as observed, not a root-cause fix.

    The toolkit version is :func:`cake_build.record_cuda_version`: the version that built the prebuilt
    libraries (``aot/build_info.json``) when the package ships them, else the local toolkit of the JIT path.
    Neither changes while the package is loaded, so the answer is computed once per architecture and process
    (the check reads the library manifest and stats every library file: milliseconds per call on a network
    file system, paid by every call of three chunks or more before its first launch).
    """
    if arch != "sm_103a":
        return True
    return tuple(cake_build.record_cuda_version(arch)) != (13, 0)


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch``: the architecture's sole record (the kernels take ``H`` / ``V``
    as launch scalars, so one record serves every admissible geometry)."""
    names = [name for name, record in MODULES.items() if record["arch"] == arch]
    if not names:
        raise NotImplementedError(
            f"The generated chunked LM-head + loss program for {arch} is not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#5680)"
        )
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one chunked LM-head program: {names}"
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


def library_name(name: str, stage: str) -> str:
    """``build.json`` entry of ``stage`` of record ``name``: the stage's library stem plus the record's
    architecture (``cake_lm_head_loss_<unit>_<arch>``)."""
    record = MODULES[name]
    return f"{record[stage]['library']}_{record['arch']}"


@functools.cache
def load_cake_lm_head_loss_module(name: str, stage: str):
    """Load (once per record and stage) the generated program of ``stage``: the prebuilt library under
    ``aot/`` when the package ships it, else FlashInfer's JIT build of its ``build.json`` entry."""
    return cake_build.load_library(library_name(name, stage))


def specs_for_arch(arch: str):
    """JIT specs of every ``build.json`` entry of ``arch`` without a prebuilt library (the
    ``flashinfer.jit.build_jit_specs`` warm-up)."""
    return cake_build.jit_specs(arch)
