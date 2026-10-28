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
# ``cake_backend.QUANTIZE_TP_MAX_PB``-entry launch parameters), and the host
# binds one of the two per plan
# (``cake_backend.nvfp4_quantize_stage``); the fp8 route precedes the
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
# stage (naming the program the export then supersedes there), are
# hand-written.
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
            "sm_100a": "cdb07c53047e2471fc725e86cbcb441befd497c1422ab3582284b97eb7f56866",
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
            "sm_100a": "749c26154b6934302d6464f4b7bb582f5b9eccf1d1051b2ef8d575bd2e3d8bcf",
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
            "sm_100a": "3c3802902e7d8606d61803358476c114739cb54c72133b475d676fb35706f1cb",
            "sm_103a": "023d480ad29eeae4423732e9d8a3d943c42db865594a9c7a2f849dcb0980ce51",
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
            "sm_100a": "d3488aab563beec201cce094d69e565df6a658add852df98421010f0849703a4",
            "sm_103a": "0ff7c73cdb55adf429eede69e68988426eda73e5cc1a7c00abf2c5c757887385",
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
            "sm_100a": "edb087ce95e504bd245050cf227392f281da26a74191431ad323eb59595122f6",
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
            "sm_100a": "c91b73c51704ad94f99367b6f7a744f79cde3846b9a66784374deb2ad69c0a4a",
            "sm_103a": "a2f321eadec172be99673e940ffc9fac274167cee8e8308f9a53c848760d54b0",
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
            "sm_103a": "e97617db9997a977cbb941de4e522d7d7bf7082d94b3d0d44f63c18fec88456e",
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
            "sm_100a": "967a70e2878490ff0b7b0639053556ff68ebe9aba17ca41a40d9459817e71d68",
            "sm_103a": "2384549d0c76b081eed3d13c097d623b88f35a5962ad55abaabf71dfd975c089",
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
            "sm_100a": "84d0908e22ae259c49fa27a41dbda989e495e1d92773ac7e109b363cadcdf0b8",
            "sm_103a": "ee2f5e87f88da2ff5cec386cadc50f6fc0d6e0bb3f16f777c9637567204c23c9",
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
            "sm_100a": "9eb2f1b20aae639e6acb320742175a4c186f76ac5ee947244ae4fcafa5c627c4",
            "sm_103a": "9e40de944fc17cf2ca5387c26c533f393f9436b1ba9b078d05e3eeea151b7fc5",
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
            "sm_100a": "cff6eb2f690f7258d404b81320a1588f8fc6efc3b40af040f2fd36b0d6048bfb",
            "sm_103a": "ddac1d9411a56c3110a8c4c7ee44890b93ba5d256465f7a1d2e87df149c3bbd8",
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
            "sm_100a": "48d2699eb56d68eb70b462b9d21482aa5607d6e48bf39b179a734f834175ba70",
            "sm_103a": "3648803b9f68011ce8055f1790136a2a3da0c50b3cdb27e68bbf2ba49589f7fc",
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
            "sm_100a": "6782ef8f73e17c6f0b0136ae312df346f94183c070cc5fe93441640efe7e649b",
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
            "sm_103a": "08918927d4694a64e3e256d4bf111bad89dc23b6a6a0d1343c472a5335ae9007",
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
            "sm_100a": "3054545e8e14964e591bec43e539909ead29ac4683d6568552c92e9de5f2f012",
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
