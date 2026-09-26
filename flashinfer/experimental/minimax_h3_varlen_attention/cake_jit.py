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
    "cake_minimax_h3_varlen_attention_0a226e2039639577757b": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_0a226e2039639577757b_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_0a226e2039639577757b_binding.cu",
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
        "closure_sha256": "8fed934ee43f2adf74e880145c8d6b29633c60c81efdf8bb190fc03c79ba2535",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_24284288ccef5e627ea2": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_24284288ccef5e627ea2_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_24284288ccef5e627ea2_binding.cu",
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
        "closure_sha256": "ca6b0394c3b93a444974c2dc727569b3efeb6e69adfa6ca11a07dc5ebcdecb67",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_323b4084b42159087576": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_323b4084b42159087576_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_323b4084b42159087576_binding.cu",
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
        "closure_sha256": "3aea705e25ab741868aaa584da14ec162bbca444f08c410226cab1a919e43865",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_39a37bc532bd45f66386": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_39a37bc532bd45f66386_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_39a37bc532bd45f66386_binding.cu",
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
        "closure_sha256": "6bf30f425e5514a25fb0ccb02865ce6318f5abce4342bef95ba1599adb926c55",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_4b58552420c22471767c": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_4b58552420c22471767c_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_4b58552420c22471767c_binding.cu",
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
        "closure_sha256": "d874bbcbd361684ed6d77a221403adf3a65ad47b3e2ff6dd8f3bf42f12dcad22",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_7de2105cc1efe0aa7c9c": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_7de2105cc1efe0aa7c9c_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_7de2105cc1efe0aa7c9c_binding.cu",
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
        "closure_sha256": "2ecf4fad062af344013a00933735ed3012dd6d0787b848979b61267af9efbaf4",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_8e8640c99222bd36536a": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_8e8640c99222bd36536a_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_8e8640c99222bd36536a_binding.cu",
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
        "closure_sha256": "6daa0f6788baf911c60b6182ef199dee9a4232e4706ce12cf962bc8797725122",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_9d6c35e4b873f1959551": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_9d6c35e4b873f1959551_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_9d6c35e4b873f1959551_binding.cu",
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
        "closure_sha256": "fea7a4c86216c93e75f40329318b2350eff98d36e7a3fe8d49c8e3aaede5e745",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_a8874976405d8fd0a3dc": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_a8874976405d8fd0a3dc_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_a8874976405d8fd0a3dc_binding.cu",
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
        "closure_sha256": "9c8738cd52b2b8d8cbbe3409e4deb75c8d782e20b0e296f96e1acf4c1b75e9aa",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_aefd08ded934cf3d3a73": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_aefd08ded934cf3d3a73_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_aefd08ded934cf3d3a73_binding.cu",
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
        "closure_sha256": "f216aa3b08c2d93f97d337233e05ef1e01ab70111c4e3f41c950ab9eb748afea",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_b178bc1be1a901d7b4ea": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_b178bc1be1a901d7b4ea_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_b178bc1be1a901d7b4ea_binding.cu",
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
        "closure_sha256": "c1af59e4a048d44b07c66820127db0a18c5a8748cbd4353a1f92ef4c34fcf756",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_ba9ebb1a4ea326f0bb9a": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_ba9ebb1a4ea326f0bb9a_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_ba9ebb1a4ea326f0bb9a_binding.cu",
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
        "closure_sha256": "109dcc55fc89cd2f27bdc2289a97b92bdfb0beb89b8aac31f52de457b74076c8",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_bebe383c025a8dabc7f8": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_bebe383c025a8dabc7f8_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_bebe383c025a8dabc7f8_binding.cu",
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
        "closure_sha256": "b2ab20bd8a48a19a6993a6d9293fb0935d74c216fb3ef9c1f27aff576d0be253",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_ccb677c5f7ec9549b929": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_ccb677c5f7ec9549b929_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_ccb677c5f7ec9549b929_binding.cu",
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
        "closure_sha256": "3d47e59f54c841b3d2396b0fc5ea023c730cd1fe5df4d666e6acecf981734a23",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_d38039c24e563ad5c8cf": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_d38039c24e563ad5c8cf_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_d38039c24e563ad5c8cf_binding.cu",
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
        "closure_sha256": "283318512f501fd9476624f38d6de556f8d829e2502baf467a14e4b5a9c64893",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_e32130f3f01903fa4d27": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_e32130f3f01903fa4d27_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_e32130f3f01903fa4d27_binding.cu",
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
        "closure_sha256": "52c73a20df8b0e3b657f339e2f3c858971a5a713bb8048dd1c6480768203c19a",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_f6b0a347ba82c9a16d05": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_f6b0a347ba82c9a16d05_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_f6b0a347ba82c9a16d05_binding.cu",
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
        "closure_sha256": "4dc93c9baa2c878384f6edb6b7fe297655701ac1293a0f8d0de3871465de07eb",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_fb5d4ed2aeabea63cc94": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_fb5d4ed2aeabea63cc94_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_fb5d4ed2aeabea63cc94_binding.cu",
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
        "closure_sha256": "3f3dbe35111d206c63568ed1eead5813416091d2bdfc2dd31d02ef3b2e0507d6",
        "tma_workspace_bytes": 0,
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "bf16__sm_100a": {
        "arch": "sm_100a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_4b58552420c22471767c",
            "combine": "cake_minimax_h3_varlen_attention_7de2105cc1efe0aa7c9c",
        },
    },
    "bf16__sm_103a": {
        "arch": "sm_103a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_b178bc1be1a901d7b4ea",
            "combine": "cake_minimax_h3_varlen_attention_ba9ebb1a4ea326f0bb9a",
        },
    },
    "nvfp4_fp4pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_323b4084b42159087576",
            "attention": "cake_minimax_h3_varlen_attention_24284288ccef5e627ea2",
            "attention_split": "cake_minimax_h3_varlen_attention_8e8640c99222bd36536a",
            "combine": "cake_minimax_h3_varlen_attention_7de2105cc1efe0aa7c9c",
        },
    },
    "nvfp4_fp4pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_fb5d4ed2aeabea63cc94",
            "attention": "cake_minimax_h3_varlen_attention_aefd08ded934cf3d3a73",
            "attention_split": "cake_minimax_h3_varlen_attention_39a37bc532bd45f66386",
            "combine": "cake_minimax_h3_varlen_attention_ba9ebb1a4ea326f0bb9a",
        },
    },
    "nvfp4_fp8pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_ccb677c5f7ec9549b929",
            "quantize": "cake_minimax_h3_varlen_attention_a8874976405d8fd0a3dc",
            "attention": "cake_minimax_h3_varlen_attention_e32130f3f01903fa4d27",
            "attention_split": "cake_minimax_h3_varlen_attention_9d6c35e4b873f1959551",
            "combine": "cake_minimax_h3_varlen_attention_7de2105cc1efe0aa7c9c",
        },
    },
    "nvfp4_fp8pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_f6b0a347ba82c9a16d05",
            "quantize": "cake_minimax_h3_varlen_attention_0a226e2039639577757b",
            "attention": "cake_minimax_h3_varlen_attention_bebe383c025a8dabc7f8",
            "attention_split": "cake_minimax_h3_varlen_attention_d38039c24e563ad5c8cf",
            "combine": "cake_minimax_h3_varlen_attention_ba9ebb1a4ea326f0bb9a",
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
