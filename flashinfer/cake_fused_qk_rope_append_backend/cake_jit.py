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

Host plan and loader glue of the Cake fused QK RMSNorm + NeoX RoPE + BF16 Q
output + paged KV append kernel (SM90 / SM100 / SM103).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` carries the host plan only: the kernel's
``library`` stem, the architectures it is delivered for, its launch record,
its stage name and knobs, the grid rule and the kernel-parameter ->
logical-argument map.  What to compile -- sources, digests, nvcc flags -- is
recorded once in ``build.json`` next to this file; :mod:`.cake_build` serves
the library ``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder
compiled it and otherwise through FlashInfer's JIT from the same sources and
flags, verifying every source digest before the first build.

A record is one *stage*: a build of the kernel with one set of knobs (the
head configuration, warps per row, the V group's warp, the request lookup,
the clearing granularity, the launch bound and this round's levers: preload
rounds of the lookup, one weight vector per warp, the page-table window of
the clearing warps, bf16x2 carriers, the pre-rotation sum of squares and
256-bit loads).  Every stage writes bit-identical outputs; the host picks one
per call from the architecture and the problem shape with the Cake stage
policy (``STAGE_POLICY``, mirrored by :func:`policy_knobs` /
:func:`stage_for`).  The host plan is :func:`launch_plan` (the row and
clearing CTAs and the grid) and the policy; everything else derives from
them.

Every stage shipped on SM100 / SM103 has a *pdl twin* ``<stage>_pdl``: the
same build with ``pdl`` set, whose kernel waits for the preceding kernel on
the stream (``griddepcontrol.wait``) before it reads ``qkv`` and before any
store, launched with the programmatic-stream-serialization attribute.  A call
that asks for programmatic dependent launch (``stage_for(...,
enable_pdl=True)``) takes the twin when :func:`pdl_enabled` admits the problem
(``PDL_POLICY``); every other call, and every call on SM90, launches the plain
stage.

The program name ``cake_fused_qk_rope_append_<20 hex>`` is the first twenty
hex digits of the SHA-256 of the compact JSON ``{"build": <build keywords,
sorted>, "module": <dotted kernel module>}`` of the kernel the program was
rendered from; the knobs of a record are those build keywords resolved with
the kernel file's defaults.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "hq8_hkv1_vsub3_d_lookup_rounds2_packed_xs1_pre_rot_ss1": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 2,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": True,
            "wide_f32_loads": False,
            "pre_rot_ss": True,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_63105e1a61ed4766441d",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_63105e1a61ed4766441d",
        # Hopper (8, 1) under 128 requests, decode and prefill: the request
        # guess, V on warp 3, two preload rounds of the lookup, bf16x2 carriers
        # and the pre-rotation sum of squares.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_63105e1a61ed4766441d",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_lookup_rounds2_packed_xs1_pre_rot_ss1",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1_pt_window1_min_blocks5": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 5,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 1,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_3108e9daa2452ea5fd88",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_3108e9daa2452ea5fd88",
        # Hopper (8, 1) decode from 128 requests: the register diet (no preload
        # rounds, one weight vector per warp, five CTAs per SM) and the page-table
        # tail window of the clearing warps.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_3108e9daa2452ea5fd88",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1_pt_window1_min_blocks5",
        "workspace_rule": None,
    },
    "hq64_hkv8_w20_vsub18_d_lookup_rounds0_single_weight1_pt_window1": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 20,
            "v_sub_base": 18,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 5,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 1,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_f164763681df41a715d6",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_f164763681df41a715d6",
        # Hopper twenty-warp (64, 8) decode: the request guess, no preload rounds,
        # one weight vector per warp and the page-table window.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_f164763681df41a715d6",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_w20_vsub18_d_lookup_rounds0_single_weight1_pt_window1",
        "workspace_rule": None,
    },
    "hq64_hkv8_w20_vsub18_d_lookup_rounds0_packed_xs1": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 20,
            "v_sub_base": 18,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 5,
            "lookup_rounds": 0,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": True,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_3d07b839a08ac7a12e12",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_3d07b839a08ac7a12e12",
        # Hopper twenty-warp (64, 8) prefill: the request guess, no preload rounds
        # and bf16x2 carriers.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_3d07b839a08ac7a12e12",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_w20_vsub18_d_lookup_rounds0_packed_xs1",
        "workspace_rule": None,
    },
    "hq64_hkv8_d2_lookup_rounds2": {
        "arches": [
            "sm_90a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 2,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 2,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_308c30c6332bdb059f8d",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_308c30c6332bdb059f8d",
        # Hopper eight-warp (64, 8) rows the twenty-warp fit refuses: the
        # proportional first guess and two preload rounds.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_308c30c6332bdb059f8d",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_d2_lookup_rounds2",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d": {
        "arches": [
            "sm_90a",
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_366635b453cef4e37f3c",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_366635b453cef4e37f3c",
        # The previous export's (8, 1) direct stage: Hopper (8, 1) prefill from 128
        # requests; Blackwell (8, 1) decode under 256 requests and (8, 1) prefill
        # from 257 requests.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_366635b453cef4e37f3c",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_pre_rot_ss1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": True,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_a3d5543b1bfaa33b5388",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_a3d5543b1bfaa33b5388",
        # Blackwell (8, 1) direct rows at norm policy 1 (decode under 256 requests,
        # prefill from 257): the pre-rotation sum of squares.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_a3d5543b1bfaa33b5388",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_pre_rot_ss1",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_e0919cff43f10890a9ab",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_e0919cff43f10890a9ab",
        # Blackwell (8, 1) decode from 256 requests: the register diet.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_e0919cff43f10890a9ab",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_u2_lookup_rounds0_single_weight1_pt_window1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 1,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_51f187bea89da5d1cc16",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_51f187bea89da5d1cc16",
        # Blackwell (8, 1) decode from 1024 clearing CTAs: the diet, the page-table
        # window and two clearing units per warp.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_51f187bea89da5d1cc16",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_u2_lookup_rounds0_single_weight1_pt_window1",
        "workspace_rule": None,
    },
    "hq8_hkv1_lookup_rounds2_wide_f32_loads1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 2,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": True,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_2c93543d41c4a3f0f200",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_2c93543d41c4a3f0f200",
        # Blackwell (8, 1) prefill under 257 requests: two preload rounds and
        # 256-bit weight and cos / sin loads.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_2c93543d41c4a3f0f200",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_lookup_rounds2_wide_f32_loads1",
        "workspace_rule": None,
    },
    "hq8_hkv1_lookup_rounds2_wide_f32_loads1_pre_rot_ss1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 2,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": True,
            "pre_rot_ss": True,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_d52123e18fcc1071f99c",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_d52123e18fcc1071f99c",
        # Blackwell (8, 1) prefill under 257 requests at norm policy 1: the same
        # with the pre-rotation sum of squares.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_d52123e18fcc1071f99c",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_lookup_rounds2_wide_f32_loads1_pre_rot_ss1",
        "workspace_rule": None,
    },
    "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 20,
            "v_sub_base": 18,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 5,
            "lookup_rounds": 0,
            "single_weight": False,
            "pt_window": 1,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_24c52ba5cfa5262067f5",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_24c52ba5cfa5262067f5",
        # Blackwell twenty-warp (64, 8) decode: the request guess, no preload
        # rounds, the page-table window and two clearing units per warp.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_24c52ba5cfa5262067f5",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1",
        "workspace_rule": None,
    },
    "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 20,
            "v_sub_base": 18,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 5,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": True,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_76cecca35fef4e498d9c",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_76cecca35fef4e498d9c",
        # Blackwell twenty-warp (64, 8) prefill: no preload rounds, one weight
        # vector per warp and 256-bit loads.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_76cecca35fef4e498d9c",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1",
        "workspace_rule": None,
    },
    "hq64_hkv8": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_2c73520879a07637c5d3",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_2c73520879a07637c5d3",
        # Blackwell eight-warp (64, 8) prefill under 257 requests and under 2048
        # clearing CTAs (unchanged).
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_2c73520879a07637c5d3",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8",
        "workspace_rule": None,
    },
    "hq64_hkv8_d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_f1bbb59ef9eac339a36a",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_f1bbb59ef9eac339a36a",
        # Blackwell eight-warp (64, 8) decode, and prefill from 257 requests, under
        # 2048 clearing CTAs: the request guess (unchanged).
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_f1bbb59ef9eac339a36a",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_d",
        "workspace_rule": None,
    },
    "hq64_hkv8_u2": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_7ed8c87694ee23da3dd7",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_7ed8c87694ee23da3dd7",
        # Blackwell eight-warp (64, 8) prefill under 257 requests from 2048 clearing
        # CTAs: two clearing units per warp (unchanged).
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_7ed8c87694ee23da3dd7",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_u2",
        "workspace_rule": None,
    },
    "hq64_hkv8_d_u2": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_499c198e31935d7fdcdb",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_499c198e31935d7fdcdb",
        # Blackwell eight-warp (64, 8) prefill from 257 requests and from 2048
        # clearing CTAs: the request guess and two clearing units per warp
        # (unchanged; the decode rows of this class take the round-two stage).
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_499c198e31935d7fdcdb",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_d_u2",
        "workspace_rule": None,
    },
    "hq64_hkv8_d_u2_lookup_rounds0_pt_window1_packed_xs1": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 0,
            "single_weight": False,
            "pt_window": 1,
            "packed_xs": True,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": False,
        },
        "kernel": "cake_fused_qk_rope_append_d50efbacb8d02b520661",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_d50efbacb8d02b520661",
        # Blackwell eight-warp (64, 8) decode from 2048 clearing CTAs: the request
        # guess and two clearing units per warp with no preload rounds, the
        # page-table window and bf16x2 carriers (round two).
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fused_qk_rope_append_d50efbacb8d02b520661",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_d_u2_lookup_rounds0_pt_window1_packed_xs1",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_5bc2fa0cac8e05b05678",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_5bc2fa0cac8e05b05678",
        # The pdl twin of ``hq8_hkv1_vsub3_d``: it waits for the preceding kernel before
        # it reads ``qkv`` and before any store; launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_5bc2fa0cac8e05b05678",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_pdl",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_pre_rot_ss1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": True,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_2a7e2d4f4e15b66ca2b6",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_2a7e2d4f4e15b66ca2b6",
        # The pdl twin of ``hq8_hkv1_vsub3_d_pre_rot_ss1``: it waits for the preceding
        # kernel before it reads ``qkv`` and before any store; launched with the PDL
        # attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_2a7e2d4f4e15b66ca2b6",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_pre_rot_ss1_pdl",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_2ddea755da583a931846",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_2ddea755da583a931846",
        # The pdl twin of ``hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1``: it waits
        # for the preceding kernel before it reads ``qkv`` and before any store;
        # launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_2ddea755da583a931846",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1_pdl",
        "workspace_rule": None,
    },
    "hq8_hkv1_vsub3_d_u2_lookup_rounds0_single_weight1_pt_window1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 3,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 1,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_da5bbed8637be2718ecd",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_da5bbed8637be2718ecd",
        # The pdl twin of
        # ``hq8_hkv1_vsub3_d_u2_lookup_rounds0_single_weight1_pt_window1``: it waits for
        # the preceding kernel before it reads ``qkv`` and before any store; launched
        # with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_da5bbed8637be2718ecd",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_vsub3_d_u2_lookup_rounds0_single_weight1_pt_window1_pdl",
        "workspace_rule": None,
    },
    "hq8_hkv1_lookup_rounds2_wide_f32_loads1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 2,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": True,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_ce5588fccf1feb7db99a",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_ce5588fccf1feb7db99a",
        # The pdl twin of ``hq8_hkv1_lookup_rounds2_wide_f32_loads1``: it waits for the
        # preceding kernel before it reads ``qkv`` and before any store; launched with
        # the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_ce5588fccf1feb7db99a",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_lookup_rounds2_wide_f32_loads1_pdl",
        "workspace_rule": None,
    },
    "hq8_hkv1_lookup_rounds2_wide_f32_loads1_pre_rot_ss1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "warps_per_row": 4,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 2,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": True,
            "pre_rot_ss": True,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_86f0f050578247e2a896",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_86f0f050578247e2a896",
        # The pdl twin of ``hq8_hkv1_lookup_rounds2_wide_f32_loads1_pre_rot_ss1``: it
        # waits for the preceding kernel before it reads ``qkv`` and before any store;
        # launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_86f0f050578247e2a896",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq8_hkv1_lookup_rounds2_wide_f32_loads1_pre_rot_ss1_pdl",
        "workspace_rule": None,
    },
    "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 20,
            "v_sub_base": 18,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 5,
            "lookup_rounds": 0,
            "single_weight": False,
            "pt_window": 1,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_a0a0909d59d9c7fafc63",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_a0a0909d59d9c7fafc63",
        # The pdl twin of ``hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1``: it
        # waits for the preceding kernel before it reads ``qkv`` and before any store;
        # launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_a0a0909d59d9c7fafc63",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1_pdl",
        "workspace_rule": None,
    },
    "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 20,
            "v_sub_base": 18,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 5,
            "lookup_rounds": 0,
            "single_weight": True,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": True,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_56e8bfeeacdee0554cd0",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_56e8bfeeacdee0554cd0",
        # The pdl twin of
        # ``hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1``: it
        # waits for the preceding kernel before it reads ``qkv`` and before any store;
        # launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_56e8bfeeacdee0554cd0",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1_pdl",
        "workspace_rule": None,
    },
    "hq64_hkv8_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_6c16c55982e0830fb59a",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_6c16c55982e0830fb59a",
        # The pdl twin of ``hq64_hkv8``: it waits for the preceding kernel before it
        # reads ``qkv`` and before any store; launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_6c16c55982e0830fb59a",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_pdl",
        "workspace_rule": None,
    },
    "hq64_hkv8_d_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 1,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_b046014aea3abcee7bc4",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_b046014aea3abcee7bc4",
        # The pdl twin of ``hq64_hkv8_d``: it waits for the preceding kernel before it
        # reads ``qkv`` and before any store; launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_b046014aea3abcee7bc4",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_d_pdl",
        "workspace_rule": None,
    },
    "hq64_hkv8_u2_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": 0,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_c9bcc54237ad68962a31",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_c9bcc54237ad68962a31",
        # The pdl twin of ``hq64_hkv8_u2``: it waits for the preceding kernel before it
        # reads ``qkv`` and before any store; launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_c9bcc54237ad68962a31",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_u2_pdl",
        "workspace_rule": None,
    },
    "hq64_hkv8_d_u2_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 8,
            "single_weight": False,
            "pt_window": 0,
            "packed_xs": False,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_600cd2aae873e0bfe0a7",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_600cd2aae873e0bfe0a7",
        # The pdl twin of ``hq64_hkv8_d_u2``: it waits for the preceding kernel before
        # it reads ``qkv`` and before any store; launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_600cd2aae873e0bfe0a7",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_d_u2_pdl",
        "workspace_rule": None,
    },
    "hq64_hkv8_d_u2_lookup_rounds0_pt_window1_packed_xs1_pdl": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "qkv"],
            ["buffer", "cos_sin"],
            ["buffer", "q_indptr"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "k_cache"],
            ["buffer", "v_cache"],
            ["buffer", "out_q"],
            ["buffer", "out_k"],
            ["buffer", "out_v"],
            ["buffer", "q_norm_weight"],
            ["buffer", "k_norm_weight"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "max_pages"],
            ["parameter", "page_size"],
            ["parameter", "k_page_stride"],
            ["parameter", "v_page_stride"],
            ["parameter", "num_row_ctas"],
            ["parameter", "clear_units_per_request"],
            ["parameter", "qk_norm_policy"],
            ["parameter", "eps"],
            ["parameter", "use_out_k"],
            ["parameter", "use_out_v"],
            ["parameter", "clear_last_page"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_ctas_plus_clear_ctas",
        },
        "knobs": {
            "num_q_heads": 64,
            "num_kv_heads": 8,
            "warps_per_row": 8,
            "v_sub_base": 0,
            "clear_unit_chunks": 256,
            "direct_lookup": True,
            "clear_units_per_warp": 2,
            "min_blocks": 0,
            "lookup_rounds": 0,
            "single_weight": False,
            "pt_window": 1,
            "packed_xs": True,
            "wide_f32_loads": False,
            "pre_rot_ss": False,
            "pdl": True,
        },
        "kernel": "cake_fused_qk_rope_append_c62b8a406fda98f675db",
        "kernel_symbol": "kernel_cake_fused_qk_rope_append_c62b8a406fda98f675db",
        # The pdl twin of ``hq64_hkv8_d_u2_lookup_rounds0_pt_window1_packed_xs1``: it
        # waits for the preceding kernel before it reads ``qkv`` and before any store;
        # launched with the PDL attribute.
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_fused_qk_rope_append_c62b8a406fda98f675db",
        "logical": {
            "clear_last_page": "clear_last_page",
            "clear_units_per_request": "clear_units_per_request",
            "cos_sin": "cos_sin",
            "eps": "eps",
            "k_cache": "k_cache",
            "k_norm_weight": "k_norm_weight",
            "k_page_stride": "k_page_stride",
            "max_pages": "max_pages",
            "num_requests": "num_requests",
            "num_row_ctas": "num_row_ctas",
            "num_rows": "num_rows",
            "out_k": "out_k",
            "out_q": "out_q",
            "out_v": "out_v",
            "page_size": "page_size",
            "page_table": "page_table",
            "q_indptr": "q_indptr",
            "q_norm_weight": "q_norm_weight",
            "qk_norm_policy": "qk_norm_policy",
            "qkv": "qkv",
            "seq_lens": "seq_lens",
            "use_out_k": "use_out_k",
            "use_out_v": "use_out_v",
            "v_cache": "v_cache",
            "v_page_stride": "v_page_stride",
        },
        "stage": "hq64_hkv8_d_u2_lookup_rounds0_pt_window1_packed_xs1_pdl",
        "workspace_rule": None,
    },
}

# One verified build per architecture: H100 / H200 (``sm_90a``), B200
# (``sm_100a``) and B300 / GB300 (``sm_103a``).  Every architecture is compiled
# with its exact ``-gencode`` flag set (``build.json`` ``arch_flags``); there is
# no family fallback.
ARCHES = ("sm_90a", "sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
HEAD_DIM = 128
# BF16 elements per 16-byte vector; a clearing unit is ``clear_unit_chunks``
# such vectors.
VEC = 8
THREADS = 128
WARPS_PER_CTA = 4
SUPPORTED_HEADS = ((8, 1), (64, 8))
# The build keywords a record's ``knobs`` carry: what the policy, the launch
# plan and the kernel's own symbol are derived from.  The six after the first
# eight are this round's levers; ``stage_name`` spells them in this order.  The
# last, ``pdl``, is the programmatic-dependent-launch build of a pdl twin.
KNOB_KEYS = (
    "num_q_heads",
    "num_kv_heads",
    "warps_per_row",
    "v_sub_base",
    "clear_unit_chunks",
    "direct_lookup",
    "clear_units_per_warp",
    "min_blocks",
    "lookup_rounds",
    "single_weight",
    "pt_window",
    "packed_xs",
    "wide_f32_loads",
    "pre_rot_ss",
    "pdl",
)
ROUND_KNOBS = KNOB_KEYS[8:14]
# The Cake stage policy: the constants and the rule of this round (``version``
# is the policy table's).  ``policy_knobs`` is the executable form.
STAGE_POLICY: dict[str, Any] = {
    "constants": {
        "clear_ctas_per_free_sm": 16,
        "direct_min_requests": 257,
        "sm100_decode_8_1_min_requests": 1,
        "sm100_u2_min_clear_ctas": 2048,
        "v_sub_base_8_1": 3,
        "w20": 20,
        "w20_v_sub_base": 18,
        "clear_unit_chunks": 256,
        "sm100_8_1_diet_min_requests": 256,
        "sm100_8_1_u2_min_clear_ctas": 1024,
        "sm90_8_1_diet_min_requests": 128,
        "sm90_8_1_lookup_rounds": 2,
        "sm90_8_1_min_blocks": 5,
        "w20_min_blocks": 5,
        "sm100_8_1_prefill_lookup_rounds": 2,
    },
    "rule": (
        "decode = (num_rows == num_requests); hopper = arch sm_90*; "
        "clear_ctas = ceil(num_requests * clear_units_per_request(page_size, "
        "Hkv, clear_unit_chunks) / 4); w20 = (Hq,Hkv)==(64,8) and num_rows <= "
        "num_sms and clear_ctas <= (num_sms - num_rows) * clear_ctas_per_free_sm "
        "-> warps_per_row=w20, v_sub_base=w20_v_sub_base, min_blocks=w20_min_blocks; "
        "direct = num_requests >= direct_min_requests or (hopper and "
        "((Hq,Hkv)==(8,1) or w20)) or (not hopper and ((Hq,Hkv)==(64,8) and "
        "decode or (Hq,Hkv)==(8,1) and decode and num_requests >= "
        "sm100_decode_8_1_min_requests)) -> direct_lookup=1; (Hq,Hkv)==(8,1) and "
        "(hopper or direct) -> v_sub_base=v_sub_base_8_1; not hopper and "
        "(Hq,Hkv)==(64,8) and not w20 and clear_ctas >= sm100_u2_min_clear_ctas "
        "-> clear_units_per_warp=2. Round one: (8,1) hopper: num_requests >= "
        "sm90_8_1_diet_min_requests -> (decode only) lookup_rounds=0, "
        "single_weight, min_blocks=sm90_8_1_min_blocks, pt_window=1, else "
        "lookup_rounds=sm90_8_1_lookup_rounds, packed_xs, pre_rot_ss; (8,1) "
        "blackwell: direct and decode and num_requests >= "
        "sm100_8_1_diet_min_requests -> lookup_rounds=0, single_weight (+ "
        "pt_window=1, clear_units_per_warp=2 from sm100_8_1_u2_min_clear_ctas "
        "clearing CTAs); not decode and not direct -> lookup_rounds="
        "sm100_8_1_prefill_lookup_rounds, wide_f32_loads (+ pre_rot_ss at "
        "qk_norm_policy 1); direct otherwise at qk_norm_policy 1 -> pre_rot_ss; "
        "w20 decode: hopper -> lookup_rounds=0, single_weight, pt_window=1, "
        "blackwell -> lookup_rounds=0, pt_window=1, clear_units_per_warp=2; w20 "
        "prefill: blackwell -> no direct_lookup, lookup_rounds=0, single_weight, "
        "wide_f32_loads, "
        "hopper -> lookup_rounds=0, packed_xs; (64,8) eight-warp hopper -> "
        "direct_lookup=2, lookup_rounds=2. Round two: (64,8) eight-warp blackwell "
        "decode at clear_units_per_warp=2 -> lookup_rounds=0, pt_window=1, "
        "packed_xs"
    ),
    "version": "round2-2026-10-20T15:40Z",
}
# The suffix of a pdl twin's record name (``stage_name`` of knobs with ``pdl``).
PDL_SUFFIX = "_pdl"
# Which calls that ask for programmatic dependent launch take the selected
# stage's pdl twin; ``pdl_enabled`` is the executable form.  The twin overlaps
# the kernel's metadata prologue (and, behind a kernel that triggers its
# dependents early, its launch) with the preceding kernel; the problems the
# rule keeps plain are those where the twin did not run faster.
PDL_POLICY: dict[str, Any] = {
    "arches": ("sm_100a", "sm_103a"),
    "constants": {
        "hq8_decode_off_requests": {"sm_100a": (512, 547), "sm_103a": (512, 547)},
        "hq64_off_min_rows": 600,
    },
    "rule": (
        "enable_pdl and arch in arches and not off -> the pdl twin, else the "
        "plain stage; decode = (num_rows == num_requests); off = (Hq, Hkv) == (64, 8) "
        "and num_rows >= hq64_off_min_rows, or (Hq, Hkv) == (8, 1) and decode and "
        "lo <= num_requests <= hi for (lo, hi) = hq8_decode_off_requests[arch], or "
        "any other head configuration"
    ),
    "version": "pdl1-2026-10-31",
}
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "qkv",
    "cos_sin",
    "q_indptr",
    "seq_lens",
    "page_table",
    "k_cache",
    "v_cache",
    "out_q",
    "out_k",
    "out_v",
    "q_norm_weight",
    "k_norm_weight",
)
LOGICAL_SCALARS = (
    "num_rows",
    "num_requests",
    "max_pages",
    "page_size",
    "k_page_stride",
    "v_page_stride",
    "num_row_ctas",
    "clear_units_per_request",
    "qk_norm_policy",
    "eps",
    "use_out_k",
    "use_out_v",
    "clear_last_page",
)


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def stages_for(arch: str) -> tuple[str, ...]:
    """Stage names (module records) shipped for ``arch``, in record order."""

    return tuple(name for name, record in MODULES.items() if arch in record["arches"])


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def clear_units_per_request(
    page_size: int, num_kv_heads: int, clear_unit_chunks: int = 256
) -> int:
    """4 KiB clearing units of one request's last page (K and V), as the kernel
    counts them."""

    chunks = int(page_size) * int(num_kv_heads) * HEAD_DIM // VEC
    return 2 * _ceil_div(chunks, clear_unit_chunks)


def launch_plan(
    record: dict[str, Any], *, num_rows: int, num_requests: int, page_size: int
) -> dict[str, Any]:
    """The host-derived scalars and the grid of one stage for one problem (the
    kernel's ``launch_plan``).

    ``rows_per_cta = max(1, 4 // warps_per_row)`` rows share a CTA when a row
    takes fewer than four warps, ``ceil(warps_per_row / 4)`` CTAs serve a row
    that takes more; the clearing CTAs follow, four warps each, every warp
    zeroing ``clear_units_per_warp`` units of one request's last page.  This
    round's levers do not change the plan.
    """

    knobs = record["knobs"]
    warps_per_row = int(knobs["warps_per_row"])
    units_per_warp = int(knobs["clear_units_per_warp"])
    rows_per_cta = max(1, WARPS_PER_CTA // warps_per_row)
    ctas_per_row = _ceil_div(warps_per_row, WARPS_PER_CTA)
    num_row_ctas = _ceil_div(num_rows, rows_per_cta) * ctas_per_row
    units = clear_units_per_request(
        page_size, int(knobs["num_kv_heads"]), int(knobs["clear_unit_chunks"])
    )
    warps_per_request = _ceil_div(units, units_per_warp)
    num_clear_ctas = _ceil_div(int(num_requests) * warps_per_request, WARPS_PER_CTA)
    return {
        "num_row_ctas": num_row_ctas,
        "clear_units_per_request": units,
        "grid": (num_row_ctas + num_clear_ctas, 1, 1),
    }


def launch_grid(
    record: dict[str, Any], *, num_rows: int, num_requests: int, page_size: int
) -> tuple[int, int, int]:
    """Launch grid of one stage for one problem under its grid rule
    (``row_ctas_plus_clear_ctas``)."""

    rule = record["grid_rule"]
    if rule["kind"] != "row_ctas_plus_clear_ctas":
        raise ValueError(
            f"unknown Cake fused QK RoPE append grid rule: {rule['kind']!r}"
        )
    return launch_plan(
        record, num_rows=num_rows, num_requests=num_requests, page_size=page_size
    )["grid"]


def policy_knobs(
    num_q_heads: int,
    num_kv_heads: int,
    *,
    arch: str,
    num_rows: int,
    num_requests: int,
    page_size: int,
    num_sms: int,
    is_prefill: bool = False,
    qk_norm_policy: int | None = None,
) -> dict[str, int | bool]:
    """The knobs the Cake stage policy (``STAGE_POLICY``) selects beyond the head
    configuration's base build.

    Keys are build keywords (``warps_per_row``, ``v_sub_base``,
    ``direct_lookup``, ``clear_units_per_warp``, ``min_blocks`` and this round's
    levers); an empty result is the base build.  ``is_prefill`` is accepted for
    the shared policy signature; like the policy table, the rule derives decode
    from ``num_rows == num_requests``.  ``qk_norm_policy`` (0 / 1 / 2, ``None``
    when unknown) selects the pre-rotation sum of squares on Blackwell for norm
    policy 1 only.
    """

    del is_prefill
    constants = STAGE_POLICY["constants"]
    hopper = arch.startswith("sm_90")
    heads = (int(num_q_heads), int(num_kv_heads))
    if heads not in SUPPORTED_HEADS:
        raise ValueError(
            f"(num_q_heads, num_kv_heads) = {heads} is not supported; "
            f"expected one of {SUPPORTED_HEADS}"
        )
    units = clear_units_per_request(
        page_size, num_kv_heads, int(constants["clear_unit_chunks"])
    )
    clear_ctas = _ceil_div(int(num_requests) * units, WARPS_PER_CTA)
    decode = int(num_rows) == int(num_requests)
    requests = int(num_requests)
    policy_one = qk_norm_policy is not None and int(qk_norm_policy) == 1
    out: dict[str, int | bool] = {}
    w20 = False
    if heads == (64, 8):
        free_sms = int(num_sms) - int(num_rows)
        w20 = (
            free_sms >= 0
            and clear_ctas <= free_sms * constants["clear_ctas_per_free_sm"]
        )
        if w20:
            out["warps_per_row"] = constants["w20"]
            out["v_sub_base"] = constants["w20_v_sub_base"]
            out["min_blocks"] = constants["w20_min_blocks"]
    direct = requests >= constants["direct_min_requests"]
    if hopper:
        direct = direct or heads == (8, 1) or w20
    else:
        direct = (
            direct
            or (heads == (64, 8) and decode)
            or (
                heads == (8, 1)
                and decode
                and requests >= constants["sm100_decode_8_1_min_requests"]
            )
        )
    if direct:
        out["direct_lookup"] = True
    if heads == (8, 1) and (hopper or direct):
        out["v_sub_base"] = constants["v_sub_base_8_1"]
    # No shipped build combines twenty warps per row with two clearing units per
    # warp on the eight-warp rule: when both conditions hold, w20 wins.
    if (
        not hopper
        and heads == (64, 8)
        and not w20
        and clear_ctas >= constants["sm100_u2_min_clear_ctas"]
    ):
        out["clear_units_per_warp"] = 2
    # This round's levers, per row class.
    if heads == (8, 1):
        if hopper:
            if requests >= constants["sm90_8_1_diet_min_requests"]:
                # A prefill row from 128 requests keeps the bare direct stage.
                if decode:
                    out["lookup_rounds"] = 0
                    out["single_weight"] = True
                    out["min_blocks"] = constants["sm90_8_1_min_blocks"]
                    out["pt_window"] = 1
            else:
                out["lookup_rounds"] = constants["sm90_8_1_lookup_rounds"]
                out["packed_xs"] = True
                out["pre_rot_ss"] = True
        elif direct and decode and requests >= constants["sm100_8_1_diet_min_requests"]:
            out["lookup_rounds"] = 0
            out["single_weight"] = True
            if clear_ctas >= constants["sm100_8_1_u2_min_clear_ctas"]:
                out["pt_window"] = 1
                out["clear_units_per_warp"] = 2
        elif not decode and not direct:
            # Prefill under 257 requests; from 257 the bare direct stage.
            out["lookup_rounds"] = constants["sm100_8_1_prefill_lookup_rounds"]
            out["wide_f32_loads"] = True
            if policy_one:
                out["pre_rot_ss"] = True
        elif direct and policy_one:
            # Decode under the diet threshold (and direct prefill) at norm
            # policy 1.
            out["pre_rot_ss"] = True
    elif w20:
        if decode:
            out["lookup_rounds"] = 0
            if hopper:
                out["single_weight"] = True
            else:
                out["clear_units_per_warp"] = 2
            out["pt_window"] = 1
        elif not hopper:
            # Blackwell W20 prefill never takes the direct lookup (the W20 rule
            # needs num_rows <= num_sms, so a shape-valid call never reaches
            # the request-count direct rule here).
            out.pop("direct_lookup", None)
            out["lookup_rounds"] = 0
            out["single_weight"] = True
            out["wide_f32_loads"] = True
        else:
            out["lookup_rounds"] = 0
            out["packed_xs"] = True
    elif not hopper and decode and out.get("clear_units_per_warp") == 2:
        # Round two: the Blackwell eight-warp (64, 8) decode rows with two
        # clearing units per warp take no preload rounds, the page-table window
        # and bf16x2 carriers on the direct u2 build.
        out["lookup_rounds"] = 0
        out["pt_window"] = 1
        out["packed_xs"] = True
    elif hopper:
        # The eight-warp (64, 8) rows: the proportional first guess and two
        # preload rounds.
        out["direct_lookup"] = 2
        out["lookup_rounds"] = 2
    return out


def pdl_enabled(
    num_q_heads: int,
    num_kv_heads: int,
    *,
    arch: str,
    num_rows: int,
    num_requests: int,
    is_prefill: bool = False,
) -> bool:
    """Whether a call that asks for programmatic dependent launch takes the
    selected stage's pdl twin (``PDL_POLICY``).

    ``is_prefill`` is accepted for the shared signature; like the stage policy,
    the rule derives decode from ``num_rows == num_requests``.
    """

    del is_prefill
    if arch not in PDL_POLICY["arches"]:
        return False
    constants = PDL_POLICY["constants"]
    heads = (int(num_q_heads), int(num_kv_heads))
    if heads == (64, 8):
        return int(num_rows) < int(constants["hq64_off_min_rows"])
    if heads == (8, 1):
        low, high = constants["hq8_decode_off_requests"][arch]
        decode = int(num_rows) == int(num_requests)
        return not (decode and int(low) <= int(num_requests) <= int(high))
    return False


def stage_name(
    num_q_heads: int, num_kv_heads: int, knobs: dict[str, int | bool]
) -> str:
    """Stage name of one policy outcome: ``hq8_hkv1_vsub3_d``,
    ``hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1`` ...

    ``_w20``, ``_vsub<N>``, ``_d`` / ``_d2`` (the ``request == row`` guess / the
    proportional guess), ``_u<N>``, then ``_<knob><int>`` per lever set in
    ``ROUND_KNOBS`` order, then ``_min_blocks<N>`` unless the bound is the
    twenty-warp builds' implied one, then ``_pdl`` for a pdl twin.
    """

    parts = [f"hq{int(num_q_heads)}_hkv{int(num_kv_heads)}"]
    warps = int(knobs.get("warps_per_row", 4))
    if warps == 20:
        parts.append("w20")
    if "v_sub_base" in knobs and int(knobs["v_sub_base"]) != 0:
        parts.append(f"vsub{int(knobs['v_sub_base'])}")
    direct = int(knobs.get("direct_lookup", 0))
    if direct == 1:
        parts.append("d")
    elif direct:
        parts.append(f"d{direct}")
    if int(knobs.get("clear_units_per_warp", 1)) > 1:
        parts.append(f"u{int(knobs['clear_units_per_warp'])}")
    for key in ROUND_KNOBS:
        if key in knobs:
            parts.append(f"{key}{int(knobs[key])}")
    if warps != 20 and int(knobs.get("min_blocks", 0)):
        parts.append(f"min_blocks{int(knobs['min_blocks'])}")
    return "_".join(parts) + (PDL_SUFFIX if knobs.get("pdl", False) else "")


def stage_for(
    num_q_heads: int,
    num_kv_heads: int,
    *,
    arch: str,
    num_rows: int,
    num_requests: int,
    page_size: int,
    num_sms: int,
    is_prefill: bool = False,
    qk_norm_policy: int | None = None,
    enable_pdl: bool = False,
) -> str:
    """The module record the policy selects for one problem on ``arch``; a stage
    the package does not ship for the architecture is an error, never a silent
    fallback.  ``enable_pdl`` asks for the stage's pdl twin, which the call
    takes when :func:`pdl_enabled` admits the problem."""

    knobs = policy_knobs(
        num_q_heads,
        num_kv_heads,
        arch=arch,
        num_rows=num_rows,
        num_requests=num_requests,
        page_size=page_size,
        num_sms=num_sms,
        is_prefill=is_prefill,
        qk_norm_policy=qk_norm_policy,
    )
    if enable_pdl and pdl_enabled(
        num_q_heads,
        num_kv_heads,
        arch=arch,
        num_rows=num_rows,
        num_requests=num_requests,
        is_prefill=is_prefill,
    ):
        knobs = {**knobs, "pdl": True}
    name = stage_name(num_q_heads, num_kv_heads, knobs)
    shipped = stages_for(arch)
    if name not in shipped:
        raise RuntimeError(
            f"the Cake stage policy selected {name!r} for {arch}, which the package "
            f"does not ship ({shipped})"
        )
    return name


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an
    exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            "Cake fused QK RoPE append target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``:
    ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake fused QK RoPE append module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch`` (prebuilt
    ``aot/`` library, else the JIT build of the same sources and flags through
    ``cake_build.load_library``)."""

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file
    (ahead-of-time warm-up)."""

    return cake_build.jit_specs(arch)


def run_module(name: str, values: dict[str, Any], *, device_index: int) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``
    to tensors and Python scalars (``num_row_ctas`` and ``clear_units_per_request``
    are :func:`launch_plan`'s); the grid is :func:`launch_grid` of the record for
    the problem the scalars describe, and the kernel runs on the current torch
    stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(
        record,
        num_rows=int(values["num_rows"]),
        num_requests=int(values["num_requests"]),
        page_size=int(values["page_size"]),
    )
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HEAD_DIM",
    "KNOB_KEYS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PDL_POLICY",
    "PDL_SUFFIX",
    "ROUND_KNOBS",
    "STAGE_POLICY",
    "SUPPORTED_HEADS",
    "THREADS",
    "VEC",
    "WARPS_PER_CTA",
    "arch_for_capability",
    "clear_units_per_request",
    "device_arch",
    "launch_grid",
    "launch_plan",
    "library_name",
    "load",
    "pdl_enabled",
    "policy_knobs",
    "run_module",
    "sm_count",
    "specs_for_arch",
    "stage_for",
    "stage_name",
    "stages_for",
]
