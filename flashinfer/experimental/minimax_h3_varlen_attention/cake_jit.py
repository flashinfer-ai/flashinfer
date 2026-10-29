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
# ``MODULES`` holds one record per physical generated program (one kernel
# source plus its host binding, shared by every architecture in ``arches``):
# translation units, compile flags, FFI entry, argument plan and closure
# identity.  Architecture-specific lowering (the SM103 ``tcgen05.ld.red``
# score drain against the SM100 packed-polynomial exp2 emulation, the BF16
# per-architecture schedule) lives inside the one source under exact
# ``__CUDA_ARCH__`` regions, so a program is compiled once per exact target
# (``sm100a_nvcc_flags`` / ``sm103a_nvcc_flags``) and the JIT caches one
# library per (program, arch).  ``ROUTES`` maps ``"<variant>__<arch>"`` to the
# ordered stage -> program assignment of that program family:
#
# * ``bf16``          stages ``("attention", "attention_dyn", "attention_sread",
#                     "attention_tiny", "combine")`` on sm_100a and ``("attention",
#                     "attention_dyn", "attention_tiny", "combine")`` on sm_103a
# * ``nvfp4_fp4pv``   stages ``("quantize", "quantize_tp", "attention",
#                     "attention_split", "combine")`` on sm_100a and ``("quantize",
#                     "attention", "attention_split", "combine")`` on sm_103a
# * ``nvfp4_fp8pv``   stages ``("amax", "quantize", "quantize_early_qk",
#                     "attention", "attention_qs", "attention_split", "combine")``
#                     on sm_100a and ``("amax", "quantize", "quantize_early_qk",
#                     "attention", "attention_split", "combine")`` on sm_103a
#
# ``quantize`` is one fused single-launch quantizer per PV mode
# (``minimax_h3_varlen_nvfp4_quantize_qkv`` / ``..._quantize_qk_fp8v``); the
# fp8 route also carries a second build of its quantizer,
# ``quantize_early_qk`` (``..._quantize_qk_fp8v_early``, bitwise-identical
# outputs; it quantizes and stores Q and K before its grid-dependency wait),
# and the fp4 route on ``NVFP4_QUANTIZE_TP_ARCHES`` a second build of its
# quantizer, ``quantize_tp`` (its ``pv_mode="fp4_tp"`` build, bitwise-identical
# outputs; it takes ``block_token`` / ``block_valid`` by value, as
# ``cake_backend.QUANTIZE_TP_MAX_PB``-entry launch parameters) -- that
# route's ``modules`` also name ``quantize_tp_mb8``, the
# ``pv_mode="fp4_tp_mb8"`` build of ``quantize_tp`` (bitwise-identical
# outputs; eight resident CTAs per SM), which is no stage: the host binds it
# in the ``quantize_tp`` slot when the quantize grid lies in
# ``cake_backend.QUANTIZE_TP_MB8_GRID`` -- and the host binds one of them per
# plan (``cake_backend.nvfp4_quantize_stage``); the fp8 route precedes the
# quantizer with ``amax`` (``minimax_h3_varlen_v_amax_partial``,
# one partial ``max|V|`` per CTA, folded by the quantizer, which is that
# kernel's programmatic dependent launch); the NVFP4 routes carry two
# attention programs, the dense ``attention`` (no K/V-split code; bound for
# plans without split units) and ``attention_split`` (reads the unit's K/V
# block range and writes partial rows; bound when the plan has split units),
# and the fp8 route on ``NVFP4_QS_DENSE_ARCHES`` a third, ``attention_qs`` (the
# single-stage dense program, bound for plans with tail-round Q-split units and
# no K/V-split units; ``attention`` is the two-stage dense program there,
# ``cake_backend.NVFP4_TWO_STAGE_DENSE``) -- the runner binds exactly one of
# them per plan -- and every attention binding carries the programmatic
# dependent launch attribute; the BF16 route carries four
# bitwise-identical attention programs selected per plan by the host planner
# (``cake_backend.build_bf16_segment_plan``): ``attention_tiny`` (on
# ``BF16_TINY_ARCHES``; one CTA per head and segment, for plans whose every
# segment has at most ``cake_backend.BF16_TINY_SEGMENT_TOKENS`` tokens and at
# most ``cake_backend.BF16_TINY_MAX_SEGMENTS`` non-empty segments),
# ``attention`` (the static schedule), ``attention_dyn`` (the dynamic
# whole-unit ticket scheduler of the long multi-wave plans) and, on
# ``BF16_SREAD_ARCHES`` only, ``attention_sread`` (the publish-tail cut of the
# remaining plans); ``combine`` is the shared
# K/V-split merge kernel (``minimax_h3_varlen_split_combine``) that finishes
# the units the host planner split over their K/V range (skipped by the
# runner when a plan has no split units).  The generated-program export
# writes the ``MODULES`` records and the ``ROUTES`` module leaves of the
# programs it renders; the ``ROUTES`` stage lists, and the leaf of a new
# stage (naming the program the export then supersedes there, or, as for
# ``quantize_tp_mb8``, the program it renders there), are hand-written.
MODULES: dict[str, dict[str, Any]] = {
    "cake_minimax_h3_varlen_attention_028f5068a74b38fa522e": {
        "arches": ["sm_100a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_028f5068a74b38fa522e_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_028f5068a74b38fa522e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "O"],
            ["buffer", "O_raw"],
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
        "closure_sha256": {
            "sm_100a": "38643c9b3514b4e506d9aaa0f66e540e50e62f72ca3e679d03c27a2ab9bc2be1",
        },
    },
    "cake_minimax_h3_varlen_attention_0f1da56372d7a27d3ae7": {
        "arches": ["sm_100a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_0f1da56372d7a27d3ae7_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_0f1da56372d7a27d3ae7_binding.cu",
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
            ["param_array", "block_token"],
            ["param_array", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "d519cd9a6dce85477dbb2e1edbc2cd62d131a2f267969f6132733938d6bf4144",
        },
    },
    "cake_minimax_h3_varlen_attention_1617b51b89eb9de58a16": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_1617b51b89eb9de58a16_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_1617b51b89eb9de58a16_binding.cu",
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
            ["parameter", "num_tiles"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "9a0a913aa33d8e6b0002a205628c4f35b71cf19ff859ff3f7e4906ded24a7be2",
            "sm_103a": "2fc6be3e45ef754fb62361527c721eb6c1306675bb07021ea1350b48b59567a3",
        },
    },
    "cake_minimax_h3_varlen_attention_1f0285434cb51b016f42": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_1f0285434cb51b016f42_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_1f0285434cb51b016f42_binding.cu",
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
            ["parameter", "num_tiles"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "05045e0d630ac7f1709ec94b303498dd89167aad54420b255b5fcbe206b6a606",
            "sm_103a": "8902346e089b6ee2d811e05862970f74b9370181f603d6475cd0ed9b83880d73",
        },
    },
    "cake_minimax_h3_varlen_attention_3f7f340abbf71c6a17d1": {
        "arches": ["sm_100a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_3f7f340abbf71c6a17d1_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_3f7f340abbf71c6a17d1_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "fb95f9c2c7942157daee2f331ff215f734ec4ac5e6515704d0968bc74734f95b",
        },
    },
    "cake_minimax_h3_varlen_attention_5d47bc81a6048253de1c": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_5d47bc81a6048253de1c_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_5d47bc81a6048253de1c_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "33db922ad9156f0bc277bae365abe80af5dc5540e6eebdab11eb7e017282dd64",
            "sm_103a": "6370cadadc1fe376ae4ace77878e14dedd6c46b3b79ac0e4dcf2d77c6fb9c10d",
        },
    },
    "cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "13baf152bd7b7925fffd3e582d5b3ac256ddd79adcb547aae258fa194ad6f6da",
            "sm_103a": "7b1e45c8a582c43d76427aaec4536e92d5c3e5f092c70002f3763254ac38d9c1",
        },
    },
    "cake_minimax_h3_varlen_attention_86b8d0417fc79d859642": {
        "arches": ["sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_86b8d0417fc79d859642_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_86b8d0417fc79d859642_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "O"],
            ["buffer", "O_raw"],
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
        "closure_sha256": {
            "sm_103a": "bd1a323a9ed27f687466e740582e6dc2d377d34a666333420c685f5dc1d395ce",
        },
    },
    "cake_minimax_h3_varlen_attention_89f32e4786d93ee3b7bb": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_89f32e4786d93ee3b7bb_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_89f32e4786d93ee3b7bb_binding.cu",
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
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "a2a7f9f98be75b10b415ccd7e7da1260aa281298e675bc0065be62edf328f69b",
            "sm_103a": "8ce9a55b5acd313a4d9049b3c15d1966608567d6551f730593eb3ed1c575ade1",
        },
    },
    "cake_minimax_h3_varlen_attention_90b49eb7762c49d34e34": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_90b49eb7762c49d34e34_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_90b49eb7762c49d34e34_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "c50ef448f3cd1f0ccf34e1776c12cff3e701a6958a1b93fb9028ee55f351f332",
            "sm_103a": "52ea0e8c9ca399aa8b16810a776dbd5c827a3656f4cdd8cb224e29796c5d7558",
        },
    },
    "cake_minimax_h3_varlen_attention_90ffa4217c59f26945b8": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_90ffa4217c59f26945b8_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_90ffa4217c59f26945b8_binding.cu",
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
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "749faaaab491edde43170fa34a7e487cdc0da2de2ac1fd003143d44d17fff7ed",
            "sm_103a": "ee0092a06b2bbc99b230fff71c428aa522a478cc7511daf5ac5ae3025647351b",
        },
    },
    "cake_minimax_h3_varlen_attention_bf109517d2bd582e5452": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_bf109517d2bd582e5452_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_bf109517d2bd582e5452_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "cc15aec95e62143cc09da3909ae98eba6a9985fc9e3aa612d2407dd0ffc053d0",
            "sm_103a": "394d55adb54aee77c7294cebb23749b711f247c64294bdf90bb83e1463672159",
        },
    },
    "cake_minimax_h3_varlen_attention_cb6b99d150feff074838": {
        "arches": ["sm_100a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_cb6b99d150feff074838_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_cb6b99d150feff074838_binding.cu",
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
            ["param_array", "block_token"],
            ["param_array", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "86d40c8f8d41b0e071d43969328e29844fa5b00a00334467e68e9ba02d71e3d3",
        },
    },
    "cake_minimax_h3_varlen_attention_d11c1c74c7173bf8207e": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d11c1c74c7173bf8207e_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d11c1c74c7173bf8207e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["buffer", "O"],
            ["param_array", "cu_seqlens"],
            ["parameter", "num_heads"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "d1c95e04815a65ddc379ff4139cf3fcc93c9591d2f363875c0fdbe131c513b49",
            "sm_103a": "2866e0c60c5283d53eeac68d640988472aebdc34b22f2c420215b6b23f793447",
        },
    },
    "cake_minimax_h3_varlen_attention_d1a5a13036fdbd9c9202": {
        "arches": ["sm_100a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d1a5a13036fdbd9c9202_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d1a5a13036fdbd9c9202_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "O"],
            ["buffer", "O_raw"],
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
        "closure_sha256": {
            "sm_100a": "db6ca825098937f75d114566c93561f2e75eb015dd21f2940d002010f63d6e63",
        },
    },
    "cake_minimax_h3_varlen_attention_f4b5b5cf11a77bbe1bb9": {
        "arches": ["sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_f4b5b5cf11a77bbe1bb9_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_f4b5b5cf11a77bbe1bb9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "O"],
            ["buffer", "O_raw"],
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
        "closure_sha256": {
            "sm_103a": "f2c1717ba3e090c0568e6295a43aa1a73a79a7096a8ce2e23cfe85ca29d3c6b8",
        },
    },
    "cake_minimax_h3_varlen_attention_f79569004c670f16a764": {
        "arches": ["sm_100a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_f79569004c670f16a764_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_f79569004c670f16a764_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "O"],
            ["buffer", "O_raw"],
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
        "closure_sha256": {
            "sm_100a": "407b18ff3e772dccb76abf341c3759df7d3c9013c7cc0976543c573d40b7dcf6",
        },
    },
    "cake_minimax_h3_varlen_attention_fe186ded0289cd7dfbe4": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_fe186ded0289cd7dfbe4_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_fe186ded0289cd7dfbe4_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "c2b633689289cd6c1b7d1d82372b1e29c8ed1cf7d8eebfe3b799a50788fb1b00",
            "sm_103a": "46b28531061783d1645eb5ad428f9a00c0b6465279a627f4e8b07bb55b84ca1e",
        },
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "bf16__sm_100a": {
        "arch": "sm_100a",
        "variant": "bf16",
        "stages": [
            "attention",
            "attention_dyn",
            "attention_sread",
            "attention_tiny",
            "combine",
        ],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_028f5068a74b38fa522e",
            "attention_dyn": "cake_minimax_h3_varlen_attention_d1a5a13036fdbd9c9202",
            "attention_sread": "cake_minimax_h3_varlen_attention_f79569004c670f16a764",
            "attention_tiny": "cake_minimax_h3_varlen_attention_d11c1c74c7173bf8207e",
            "combine": "cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0",
        },
    },
    "bf16__sm_103a": {
        "arch": "sm_103a",
        "variant": "bf16",
        "stages": ["attention", "attention_dyn", "attention_tiny", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_86b8d0417fc79d859642",
            "attention_dyn": "cake_minimax_h3_varlen_attention_f4b5b5cf11a77bbe1bb9",
            "attention_tiny": "cake_minimax_h3_varlen_attention_d11c1c74c7173bf8207e",
            "combine": "cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0",
        },
    },
    "nvfp4_fp4pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp4pv",
        "stages": [
            "quantize",
            "quantize_tp",
            "attention",
            "attention_split",
            "combine",
        ],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_5d47bc81a6048253de1c",
            "quantize_tp": "cake_minimax_h3_varlen_attention_0f1da56372d7a27d3ae7",
            "quantize_tp_mb8": "cake_minimax_h3_varlen_attention_cb6b99d150feff074838",
            "attention": "cake_minimax_h3_varlen_attention_1f0285434cb51b016f42",
            "attention_split": "cake_minimax_h3_varlen_attention_90ffa4217c59f26945b8",
            "combine": "cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0",
        },
    },
    "nvfp4_fp4pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_5d47bc81a6048253de1c",
            "attention": "cake_minimax_h3_varlen_attention_1f0285434cb51b016f42",
            "attention_split": "cake_minimax_h3_varlen_attention_90ffa4217c59f26945b8",
            "combine": "cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0",
        },
    },
    "nvfp4_fp8pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp8pv",
        "stages": [
            "amax",
            "quantize",
            "quantize_early_qk",
            "attention",
            "attention_qs",
            "attention_split",
            "combine",
        ],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_fe186ded0289cd7dfbe4",
            "quantize": "cake_minimax_h3_varlen_attention_90b49eb7762c49d34e34",
            "quantize_early_qk": "cake_minimax_h3_varlen_attention_bf109517d2bd582e5452",
            "attention": "cake_minimax_h3_varlen_attention_3f7f340abbf71c6a17d1",
            "attention_qs": "cake_minimax_h3_varlen_attention_1617b51b89eb9de58a16",
            "attention_split": "cake_minimax_h3_varlen_attention_89f32e4786d93ee3b7bb",
            "combine": "cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0",
        },
    },
    "nvfp4_fp8pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp8pv",
        "stages": [
            "amax",
            "quantize",
            "quantize_early_qk",
            "attention",
            "attention_split",
            "combine",
        ],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_fe186ded0289cd7dfbe4",
            "quantize": "cake_minimax_h3_varlen_attention_90b49eb7762c49d34e34",
            "quantize_early_qk": "cake_minimax_h3_varlen_attention_bf109517d2bd582e5452",
            "attention": "cake_minimax_h3_varlen_attention_1617b51b89eb9de58a16",
            "attention_split": "cake_minimax_h3_varlen_attention_89f32e4786d93ee3b7bb",
            "combine": "cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0",
        },
    },
}

VARIANTS = ("bf16", "nvfp4_fp4pv", "nvfp4_fp8pv", "fp8")
STAGES = {
    "bf16": (
        "attention",
        "attention_dyn",
        "attention_sread",
        "attention_tiny",
        "combine",
    ),
    "nvfp4_fp4pv": (
        "quantize",
        "quantize_tp",
        "attention",
        "attention_split",
        "combine",
    ),
    "nvfp4_fp8pv": (
        "amax",
        "quantize",
        "quantize_early_qk",
        "attention",
        "attention_qs",
        "attention_split",
        "combine",
    ),
    "fp8": ("attention", "attention_tiny", "combine"),
}

# The all-FP8 route (``variant="fp8"``, flashinfer-ai/flashinfer#4532
# candidate 3B): E4M3 ``query`` / ``key`` / ``value`` with per-tensor descales,
# two static attention programs over the BF16 planner's unit table (the
# general one, and its ``tail_skip=32`` build for plans whose every segment
# has at most ``FP8_TINY_SEGMENT_TOKENS`` tokens) and the BF16 route's
# ``combine`` program for the K/V-split partials.  The attention programs are
# in-tree generated programs (``flashinfer/jit/
# cake_minimax_h3_varlen_fp8_attention.py``, ``CakeProgram`` constants named
# by ``programs``), not ``MODULES`` records; the combine stage stays a
# ``MODULES`` record, the one the BF16 route names (so a delivery that
# supersedes the combine program re-points this route too).  Hand-written:
# the in-tree export does not edit this registry.
FP8_TINY_SEGMENT_TOKENS = 32
FP8_ROUTES: dict[str, dict[str, Any]] = {
    f"fp8__{_arch}": {
        "arch": _arch,
        "variant": "fp8",
        "stages": ["attention", "attention_tiny", "combine"],
        "modules": {"combine": ROUTES[f"bf16__{_arch}"]["modules"]["combine"]},
        "programs": {"attention": "ATTENTION", "attention_tiny": "ATTENTION_TINY"},
    }
    for _arch in ("sm_100a", "sm_103a")
}
ROUTES.update(FP8_ROUTES)

#: The generated loader of the in-tree FP8 attention program.
FP8_LOADER = "cake_minimax_h3_varlen_fp8_attention"


@functools.cache
def in_tree_program(constant: str) -> Any:
    """The ``CakeProgram`` constant ``constant`` of the FP8 loader, or ``None`` when this checkout does not ship
    the generated loader yet (the route then reports itself unavailable)."""
    try:
        from ...jit import cake_minimax_h3_varlen_fp8_attention as loader
    except ImportError:
        return None
    return getattr(loader, constant, None)


def route_program(record: dict[str, Any], stage: str) -> Any:
    """The in-tree ``CakeProgram`` of ``stage`` in the route ``record``."""
    program = in_tree_program(record["programs"][stage])
    if program is None:
        raise NotImplementedError(
            f"The generated MiniMax-H3 packed-varlen {record['variant']} attention "
            f"program for {record['arch']} is not registered in this checkout yet "
            "(see flashinfer-ai/flashinfer#4532)"
        )
    return program


ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
# Architectures whose BF16 route carries the ``attention_sread`` program (the
# Cake planner's ``SREAD_UNIFIED_DISPATCH_ARCHS``); the other routes bind the
# static program for the plans that are not dynamically scheduled.
BF16_SREAD_ARCHES = ("sm_100a",)
# Architectures whose BF16 route carries the ``attention_tiny`` program (the
# architectures its program is built for); a tiny plan on any other route
# binds the general programs.
BF16_TINY_ARCHES = ("sm_100a", "sm_103a")
# Architectures whose fp4-PV route carries the ``quantize_tp`` program (the
# by-value-table build of the fp4 quantizer, built for sm_100a only); a plan on
# any other route binds ``quantize``.
NVFP4_QUANTIZE_TP_ARCHES = ("sm_100a",)
# Architectures whose fp8-PV route carries the ``attention_qs`` program (the
# single-stage dense program beside the two-stage ``attention`` program of
# ``cake_backend.NVFP4_TWO_STAGE_DENSE``); a plan on any other route binds
# ``attention`` for every plan without K/V-split units.
NVFP4_QS_DENSE_ARCHES = ("sm_100a",)


def stages_for(variant: str, arch: str) -> tuple[str, ...]:
    """The ordered stages of ``variant`` on ``arch``."""
    stages = STAGES[variant]
    if variant == "bf16" and arch not in BF16_SREAD_ARCHES:
        stages = tuple(s for s in stages if s != "attention_sread")
    if variant == "bf16" and arch not in BF16_TINY_ARCHES:
        stages = tuple(s for s in stages if s != "attention_tiny")
    if variant == "nvfp4_fp4pv" and arch not in NVFP4_QUANTIZE_TP_ARCHES:
        stages = tuple(s for s in stages if s != "quantize_tp")
    if variant == "nvfp4_fp8pv" and arch not in NVFP4_QS_DENSE_ARCHES:
        stages = tuple(s for s in stages if s != "attention_qs")
    return stages


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
    if tuple(record["stages"]) != stages_for(variant, arch):
        raise RuntimeError(
            f"registered route {route_name(variant, arch)!r} has stages "
            f"{tuple(record['stages'])!r}, expected {stages_for(variant, arch)!r}"
        )
    for stage, module in record["modules"].items():
        if arch not in MODULES[module]["arches"]:
            raise RuntimeError(
                f"route {route_name(variant, arch)!r} stage {stage!r} names program "
                f"{module!r}, which is not built for {arch}"
            )
    for stage in record.get("programs", {}):
        program = route_program(record, stage)
        if arch not in program.targets:
            raise RuntimeError(
                f"route {route_name(variant, arch)!r} stage {stage!r} names generated "
                f"program {program.name!r}, which is not built for {arch}: "
                f"{list(program.targets)}"
            )
    return record


def route_available(variant: str, arch: str) -> bool:
    record = ROUTES.get(route_name(variant, arch))
    if record is None:
        return False
    return all(
        in_tree_program(constant) is not None
        for constant in record.get("programs", {}).values()
    )


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
def gen_cake_minimax_h3_varlen_attention_module(name: str, arch: str):
    """JIT spec of program ``name`` compiled for the exact target ``arch``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(
            f"program {name!r} is not built for {arch!r}: {record['arches']}"
        )
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    # ``closure_sha256[arch]`` is the export's module receipt identity for this target.
    return gen_jit_spec(
        name=f"{name}_{arch}_" + record["closure_sha256"][arch][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_minimax_h3_varlen_attention_module(name: str, arch: str):
    return gen_cake_minimax_h3_varlen_attention_module(name, arch).build_and_load()
