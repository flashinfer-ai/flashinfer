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

# Explicit target-owned registration of the generated programs.
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): translation units, compile flags, FFI entry, argument
# plan and closure identity.  ``ROUTES`` maps ``"<variant>__<arch>"`` to the
# ordered stage -> module assignment of that program family:
#
# * ``bf16``          stages ``("attention", "combine")``
# * ``nvfp4_fp4pv``   stages ``("quantize", "attention", "attention_split", "combine")``
# * ``nvfp4_fp8pv``   stages ``("amax", "quantize", "attention", "attention_split", "combine")``
#
# ``quantize`` is one fused single-launch quantizer per PV mode
# (``minimax_h3_varlen_nvfp4_quantize_qkv`` / ``..._quantize_qk_fp8v``); the
# fp8 route precedes it with ``amax`` (``minimax_h3_varlen_v_amax_partial``,
# one partial ``max|V|`` per CTA, folded by the quantizer, which is that
# kernel's programmatic dependent launch); the
# NVFP4 routes carry two attention programs, the dense ``attention`` (no
# K/V-split code; bound for plans without split units) and
# ``attention_split`` (reads the unit's K/V block range and writes partial
# rows; bound when the plan has split units) -- the runner binds exactly one
# of them per plan -- and both bindings carry the programmatic dependent
# launch attribute; ``combine`` is the shared K/V-split merge kernel
# (``minimax_h3_varlen_split_combine``) that finishes the units the host
# planner split over their K/V range (skipped by the runner when a plan has no
# split units).  Every module is an exact-arch program (the
# sm_100a NVFP4 attention uses the hybrid exp2 recipe, sm_103a
# ``tcgen05.ld.red``).  Both literals are populated verbatim by the
# generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "v"],
            ["buffer", "v_amax_partial"],
            ["parameter", "num_vectors"],
            ["parameter", "num_chunks"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "4862ba65aa4c28ec98c60edb1412f741c51f037881316426b42e0c65f453e68b",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_0e4830f8fd365c118d4f": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_0e4830f8fd365c118d4f_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_0e4830f8fd365c118d4f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "55a3ad8e2476d649ec249502c8c97d129ef5db8bc03649ff03264280009265de",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_167e2767ad20213a59b1": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_167e2767ad20213a59b1_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_167e2767ad20213a59b1_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "d25951d96025810f7031475e470b4419a83369a7dc683e9513dca3c81850b965",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_27aad524c1ac040acff4": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_27aad524c1ac040acff4_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_27aad524c1ac040acff4_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["buffer", "Q_raw"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["buffer", "O"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "unit_table"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "total_tiles"],
            ["parameter", "num_heads"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "b29fe1b52cdda36a45f5ae8087dcbd0c44fa7be1147e2d828f286d56b7163299",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_330521b5d6cd0a93df49": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_330521b5d6cd0a93df49_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_330521b5d6cd0a93df49_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "ba485be97a568ac61ddbdfa5cb520d882b41225d042fd1eab07a3860e9c7e116",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_530411d4fd6f12553109": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_530411d4fd6f12553109_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_530411d4fd6f12553109_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "v"],
            ["buffer", "v_amax_partial"],
            ["parameter", "num_vectors"],
            ["parameter", "num_chunks"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "162f956cdb1167b9198fc7056bfdea0df5d0e8e4f48515fe7b3353ec14e6f8ef",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_557c0fe57677e7688034": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_557c0fe57677e7688034_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_557c0fe57677e7688034_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp8"],
            ["buffer", "v_amax"],
            ["buffer", "v_amax_partial"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "b18a222cb1c5a90bcfefbf15cc0d9b00a00edbbf5c80e0bdc3b85e4c02e64568",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_6ac01dbc67b31d533887": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_6ac01dbc67b31d533887_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_6ac01dbc67b31d533887_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["buffer", "Q_raw"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["buffer", "O"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "unit_table"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "total_tiles"],
            ["parameter", "num_heads"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "b5034be139710382448fc275864993118528bb3129255a795f093861a41dbe04",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_78e74a16416a7d27c9f8": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_78e74a16416a7d27c9f8_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_78e74a16416a7d27c9f8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "fe47213d5e6b1c52e5acde6cfb1be771b995a6c7f85893374fefc3195bd1833d",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_79cc755a24fa6f9758c8": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_79cc755a24fa6f9758c8_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_79cc755a24fa6f9758c8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["buffer", "combine_table"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "O"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "f704e2e9b8c8d9a37927496935df9d0f2a7a8e1864e51c2f631c6a878f53ca3b",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_7dd4710e5471034647d6": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_7dd4710e5471034647d6_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_7dd4710e5471034647d6_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "776d8d99939d4f8ef34f7f92616752ce86161a46e3c13d53de8d57d6cd9067e6",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "da314f71be42553d5f9b961030bdfe2ff95ea25157dfae2159de369195a54c50",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_b117806ae1ee4e7876de": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_b117806ae1ee4e7876de_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_b117806ae1ee4e7876de_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "b15c062da7291c3cec8d1880324b10b8000cd0c7ac5f6a26ef81d0705887e3a8",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_b5202869834cd7822fb1": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_b5202869834cd7822fb1_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_b5202869834cd7822fb1_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["buffer", "combine_table"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "O"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "4d4e7cb2d794176103f5a6688187bf931f134ddd0d22cb066cd1b2e94c4c7a1f",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_bb24687b0d61491f8d5d": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_bb24687b0d61491f8d5d_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_bb24687b0d61491f8d5d_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "Vt"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["tma_buffer", "SFVtLo"],
            ["tma_buffer", "SFVtHi"],
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "99e320ece176ac74c73143f4fc833db3328b8d64f7a47592413ffce4f7878912",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_c175a6b3704c50592480": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_c175a6b3704c50592480_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_c175a6b3704c50592480_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp8"],
            ["buffer", "v_amax"],
            ["buffer", "v_amax_partial"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "f9990b56bd2835a2899989bb58930feb41e06b58570ee9690bc41112c84916eb",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_c91aa0cff6efdec01179": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_c91aa0cff6efdec01179_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_c91aa0cff6efdec01179_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp4_t"],
            ["buffer", "v_scale_lo"],
            ["buffer", "v_scale_hi"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "083d2fda59c1be45619ccddf1d1b79d2162197466da06e8489d5e0e05930c099",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_e70c217406e9eafae840": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_e70c217406e9eafae840_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_e70c217406e9eafae840_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp4_t"],
            ["buffer", "v_scale_lo"],
            ["buffer", "v_scale_hi"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "3bb08ae9bdc287e537bf567f7ec1e99a42a9fc015eef2e080bbe776157165668",
        "tma_workspace_bytes": 0,
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "bf16__sm_100a": {
        "arch": "sm_100a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_27aad524c1ac040acff4",
            "combine": "cake_minimax_h3_varlen_attention_b5202869834cd7822fb1",
        },
    },
    "bf16__sm_103a": {
        "arch": "sm_103a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_6ac01dbc67b31d533887",
            "combine": "cake_minimax_h3_varlen_attention_79cc755a24fa6f9758c8",
        },
    },
    "nvfp4_fp4pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_e70c217406e9eafae840",
            "attention": "cake_minimax_h3_varlen_attention_bb24687b0d61491f8d5d",
            "attention_split": "cake_minimax_h3_varlen_attention_78e74a16416a7d27c9f8",
            "combine": "cake_minimax_h3_varlen_attention_b5202869834cd7822fb1",
        },
    },
    "nvfp4_fp4pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_c91aa0cff6efdec01179",
            "attention": "cake_minimax_h3_varlen_attention_b117806ae1ee4e7876de",
            "attention_split": "cake_minimax_h3_varlen_attention_a9de8bf510ed0bfbb02a",
            "combine": "cake_minimax_h3_varlen_attention_79cc755a24fa6f9758c8",
        },
    },
    "nvfp4_fp8pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_530411d4fd6f12553109",
            "quantize": "cake_minimax_h3_varlen_attention_c175a6b3704c50592480",
            "attention": "cake_minimax_h3_varlen_attention_167e2767ad20213a59b1",
            "attention_split": "cake_minimax_h3_varlen_attention_7dd4710e5471034647d6",
            "combine": "cake_minimax_h3_varlen_attention_b5202869834cd7822fb1",
        },
    },
    "nvfp4_fp8pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_09d2c9ea5209af5c2d70",
            "quantize": "cake_minimax_h3_varlen_attention_557c0fe57677e7688034",
            "attention": "cake_minimax_h3_varlen_attention_0e4830f8fd365c118d4f",
            "attention_split": "cake_minimax_h3_varlen_attention_330521b5d6cd0a93df49",
            "combine": "cake_minimax_h3_varlen_attention_79cc755a24fa6f9758c8",
        },
    },
}

VARIANTS = ("bf16", "nvfp4_fp4pv", "nvfp4_fp8pv")
STAGES = {
    "bf16": ("attention", "combine"),
    "nvfp4_fp4pv": ("quantize", "attention", "attention_split", "combine"),
    "nvfp4_fp8pv": ("amax", "quantize", "attention", "attention_split", "combine"),
}
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def route_name(variant: str, arch: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown MiniMax-H3 varlen attention variant {variant!r}")
    return f"{variant}__{arch}"


def select_route(variant: str, arch: str) -> dict[str, Any]:
    """Return the registered route record for ``variant`` on ``arch``."""
    record = ROUTES.get(route_name(variant, arch))
    if record is None:
        raise NotImplementedError(
            f"The generated MiniMax-H3 packed-varlen {variant} attention program "
            f"for {arch} is not registered in this checkout yet "
            "(see flashinfer-ai/flashinfer#4532)"
        )
    if tuple(record["stages"]) != STAGES[variant]:
        raise RuntimeError(
            f"registered route {route_name(variant, arch)!r} has stages "
            f"{tuple(record['stages'])!r}, expected {STAGES[variant]!r}"
        )
    return record


def route_available(variant: str, arch: str) -> bool:
    return route_name(variant, arch) in ROUTES


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
def gen_cake_minimax_h3_varlen_attention_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_minimax_h3_varlen_attention_module(name: str):
    return gen_cake_minimax_h3_varlen_attention_module(name).build_and_load()
