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

JIT registration of the Cake-generated Blackwell MSA programs.  One device /
binding source pair per physical schedule, compiled with the exact flag set of
the device it runs on; the architecture is part of the JIT spec name so two
architectures never share one cached library.  ``ARG_PLANS`` lists the host
binding's argument order once per route family; each ``MODULES`` record names
its plan.
"""

import functools
from pathlib import Path
from typing import Any, Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

BlackwellMSATarget = Literal["sm100a", "sm103a"]
BlackwellMSAVariant = str
_NVCC_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_TARGET_ARCH: dict[BlackwellMSATarget, str] = {"sm100a": "sm_100a", "sm103a": "sm_103a"}

MODULES: dict[str, dict[str, Any]] = {
    "cake_blackwell_msa_0bb6210f55740fc901ee": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_0bb6210f55740fc901ee_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_0bb6210f55740fc901ee_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_0c83769fc6cdcfadb2d0": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_0c83769fc6cdcfadb2d0_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_0c83769fc6cdcfadb2d0_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_122d046ce6496a110567": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_122d046ce6496a110567_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_122d046ce6496a110567_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_1a54cd2531ea0b47c832": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_1a54cd2531ea0b47c832_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_1a54cd2531ea0b47c832_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_1d0a9120638c2d29c6bf": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_1d0a9120638c2d29c6bf_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_1d0a9120638c2d29c6bf_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "kmaj_p_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_20354ee19ac6500cd3e7": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_20354ee19ac6500cd3e7_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_20354ee19ac6500cd3e7_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_20585affcadecd41283e": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_20585affcadecd41283e_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_20585affcadecd41283e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_reduce",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_20abed2035425df5fd26": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_20abed2035425df5fd26_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_20abed2035425df5fd26_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_257bad493205f6bf5e16": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_257bad493205f6bf5e16_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_257bad493205f6bf5e16_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_fp8_q1_flat_xform2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_27f68603780b991a0f8a": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_27f68603780b991a0f8a_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_27f68603780b991a0f8a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_uniform_fp8",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_2be65623fe731f383035": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_2be65623fe731f383035_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_2be65623fe731f383035_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
        "arches": ["sm_100a"],
    },
    "cake_blackwell_msa_2ffd8cc9735419ada74a": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_2ffd8cc9735419ada74a_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_2ffd8cc9735419ada74a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_33a4eea7ce8a780334bd": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_33a4eea7ce8a780334bd_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_33a4eea7ce8a780334bd_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_55d535f060e0f1832077": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_55d535f060e0f1832077_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_55d535f060e0f1832077_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_63e75266177bb32039c8": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_63e75266177bb32039c8_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_63e75266177bb32039c8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_63f39782ab0c25d9fbeb": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_63f39782ab0c25d9fbeb_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_63f39782ab0c25d9fbeb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "kmaj_p_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_651272858f5dbbf73ed7": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_651272858f5dbbf73ed7_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_651272858f5dbbf73ed7_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "kmaj_p_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_68493caedaae47897f78": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_68493caedaae47897f78_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_68493caedaae47897f78_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_6d3a9fa0d331c4f4f6c5": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_6d3a9fa0d331c4f4f6c5_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_6d3a9fa0d331c4f4f6c5_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_6e1adc85275f4235e8b4": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_6e1adc85275f4235e8b4_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_6e1adc85275f4235e8b4_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_7abe03b6bbeb05295938": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_7abe03b6bbeb05295938_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_7abe03b6bbeb05295938_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_reduce",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_81fdf422f4015ca16a10": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_81fdf422f4015ca16a10_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_81fdf422f4015ca16a10_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_8a61e37bccdaf60470ef": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_8a61e37bccdaf60470ef_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_8a61e37bccdaf60470ef_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_94e0b9ebcc109376d181": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_94e0b9ebcc109376d181_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_94e0b9ebcc109376d181_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_94e24f46ab7b87bb0dfb": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_94e24f46ab7b87bb0dfb_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_94e24f46ab7b87bb0dfb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_9769cb736d4783aed640": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_9769cb736d4783aed640_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_9769cb736d4783aed640_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_a070ff8bfefdf7e4c6e9": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_a070ff8bfefdf7e4c6e9_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_a070ff8bfefdf7e4c6e9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_bc07249d0e47672e8e26": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_bc07249d0e47672e8e26_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_bc07249d0e47672e8e26_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select_streaming",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_c3cc6426f72489566e5b": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_c3cc6426f72489566e5b_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_c3cc6426f72489566e5b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_fp8_q1_paged_xform2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_cb2d66040aa8dd689059": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_cb2d66040aa8dd689059_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_cb2d66040aa8dd689059_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_e25d0af330369bae8525": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_e25d0af330369bae8525_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_e25d0af330369bae8525_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "kmaj_p_reduce",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_f4c4135904984b79e0d9": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_f4c4135904984b79e0d9_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_f4c4135904984b79e0d9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_f6244dc9e9c70766baff": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_f6244dc9e9c70766baff_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_f6244dc9e9c70766baff_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_f8b0f3032c3d2d438e19": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_f8b0f3032c3d2d438e19_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_f8b0f3032c3d2d438e19_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
        "arches": ["sm_103a"],
    },
    "cake_blackwell_msa_fa9d1e77e08ed596123b": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_fa9d1e77e08ed596123b_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_fa9d1e77e08ed596123b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
}

ARG_PLANS: dict[str, list[list[str]]] = {
    "decode_fp8_q1_flat_xform2": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "V"],
        ["buffer", "O"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "kv_indptr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["parameter", "num_requests"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "msa_max_pages"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "decode_fp8_q1_paged_xform2": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "V"],
        ["buffer", "O"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "kv_indptr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["parameter", "num_requests"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "msa_max_pages"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "decode_m16": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "Q_prefill"],
        ["buffer", "Q_prefill_raw"],
        ["tma_buffer", "K"],
        ["tma_buffer", "K_scale"],
        ["tma_buffer", "K_prefill_pair"],
        ["tma_buffer", "V"],
        ["tma_buffer", "V_scale"],
        ["tma_buffer", "V_prefill_pair"],
        ["tma_buffer", "KV"],
        ["buffer", "O"],
        ["buffer", "partial_O"],
        ["buffer", "partial_M"],
        ["buffer", "partial_D"],
        ["buffer", "split_completion"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "qo_indptr"],
        ["buffer", "kv_indptr"],
        ["buffer", "kv_len_arr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["buffer", "task_q_tile"],
        ["buffer", "task_split"],
        ["buffer", "task_kv_tile_begin"],
        ["buffer", "task_kv_tile_end"],
        ["buffer", "task_qo_begin"],
        ["buffer", "task_qo_end"],
        ["buffer", "task_page_begin"],
        ["buffer", "task_page_end"],
        ["buffer", "status"],
        ["parameter", "num_requests"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "max_kv_tiles"],
        ["parameter", "max_splits"],
        ["parameter", "max_task_claims"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "k_global_scale"],
        ["parameter", "v_global_scale"],
        ["parameter", "attention_mode"],
        ["parameter", "is_causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "record_tasks"],
        ["parameter", "msa_max_pages"],
        ["parameter", "msa_split_policy"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "decode_uniform_fp8": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "K_scale"],
        ["tma_buffer", "V"],
        ["tma_buffer", "V_scale"],
        ["buffer", "O"],
        ["buffer", "partial_O"],
        ["buffer", "partial_M"],
        ["buffer", "partial_D"],
        ["buffer", "split_completion"],
        ["buffer", "msa_lse"],
        ["buffer", "kv_indices"],
        ["buffer", "kv_indptr"],
        ["buffer", "task_kind"],
        ["buffer", "task_request"],
        ["buffer", "task_kv_head"],
        ["parameter", "total_q"],
        ["parameter", "seqlen_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "output_scale"],
        ["parameter", "msa_max_pages"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "kmaj_p_main": [
        ["tma_buffer", "q"],
        ["tma_buffer", "k"],
        ["tma_buffer", "v"],
        ["buffer", "scheduler_metadata"],
        ["buffer", "k2q_row_ptr"],
        ["buffer", "k2q_qsplit_indices"],
        ["buffer", "partial_o"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["buffer", "q_offsets"],
        ["buffer", "kv_lens"],
        ["buffer", "page_table"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "total_rows"],
        ["parameter", "nnz_per_head"],
        ["parameter", "num_work_items"],
        ["parameter", "topk"],
        ["parameter", "max_pages"],
        ["parameter", "causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "lse_temperature_scale"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "kmaj_p_reduce": [
        ["buffer", "partial_o"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "split_counts"],
        ["buffer", "out"],
        ["buffer", "lse"],
        ["buffer", "temperature_lse"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "qhead_per_kv"],
        ["parameter", "topk"],
        ["parameter", "return_softmax_lse"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "long_bf16_reverse_main": [
        ["tma_buffer", "q"],
        ["tma_buffer", "k"],
        ["tma_buffer", "v"],
        ["buffer", "scheduler_metadata"],
        ["buffer", "k2q_row_ptr"],
        ["buffer", "k2q_qsplit_indices"],
        ["buffer", "partial_o"],
        ["buffer", "partial_scale"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "out"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["buffer", "q_offsets"],
        ["buffer", "kv_lens"],
        ["buffer", "page_table"],
        ["parameter", "q_group_segment_end_128"],
        ["parameter", "q_group_segment_end_64"],
        ["parameter", "q_group_segment_end_32"],
        ["parameter", "q_group_segment_end_16"],
        ["parameter", "q_group_segment_end_8"],
        ["parameter", "q_group_segment_end_4"],
        ["parameter", "q_group_segment_end_2"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "total_rows"],
        ["parameter", "nnz_per_head"],
        ["parameter", "work_capacity"],
        ["parameter", "num_work_items"],
        ["parameter", "topk"],
        ["parameter", "max_pages"],
        ["parameter", "causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "lse_temperature_scale"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "long_bf16_reverse_reduce": [
        ["buffer", "partial_o"],
        ["buffer", "partial_scale"],
        ["buffer", "partial_lse"],
        ["buffer", "partial_temperature_lse"],
        ["buffer", "split_counts"],
        ["buffer", "out"],
        ["buffer", "lse"],
        ["buffer", "temperature_lse"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "qhead_per_kv"],
        ["parameter", "topk"],
        ["parameter", "return_softmax_lse"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "prefill_union": [
        ["tma_buffer", "q"],
        ["tma_buffer", "k"],
        ["tma_buffer", "k_scale"],
        ["tma_buffer", "v"],
        ["tma_buffer", "v_scale"],
        ["buffer", "out"],
        ["buffer", "lse"],
        ["buffer", "temperature_lse"],
        ["buffer", "q2k_indices"],
        ["buffer", "cu_seqlens_q"],
        ["buffer", "cu_seqlens_k"],
        ["buffer", "q_offsets"],
        ["buffer", "kv_lens"],
        ["buffer", "page_table"],
        ["parameter", "total_q"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_kv_heads"],
        ["parameter", "topk"],
        ["parameter", "batch_size"],
        ["parameter", "uniform_q_len"],
        ["parameter", "max_pages"],
        ["parameter", "causal"],
        ["parameter", "derive_q_offset"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "k_global_scale"],
        ["parameter", "v_global_scale"],
        ["parameter", "lse_temperature_scale"],
        ["parameter", "return_softmax_lse"],
        ["parameter", "return_temperature_lse"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "topk_select": [
        ["buffer", "max_score"],
        ["buffer", "output"],
        ["buffer", "partials"],
        ["buffer", "counters"],
        ["buffer", "block_max"],
        ["parameter", "num_heads"],
        ["parameter", "max_k_tiles"],
        ["parameter", "total_q"],
        ["parameter", "num_valid_pages"],
        ["parameter", "force_begin_blocks"],
        ["parameter", "force_end_blocks"],
        ["parameter", "num_qtiles"],
        ["parameter", "num_slices"],
        ["parameter", "slice_len"],
        ["parameter", "index_bits"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "topk_select_streaming": [
        ["buffer", "max_score"],
        ["buffer", "output"],
        ["parameter", "num_heads"],
        ["parameter", "max_k_tiles"],
        ["parameter", "total_q"],
        ["parameter", "num_valid_pages"],
        ["parameter", "force_begin_blocks"],
        ["parameter", "force_end_blocks"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

ROUTES: dict[str, dict[str, str]] = {
    "sm100a": {
        "decode_fp8_q1:q1_flat_xform2:main": "cake_blackwell_msa_257bad493205f6bf5e16",
        "decode_fp8_q1:q1_paged_xform2:main": "cake_blackwell_msa_c3cc6426f72489566e5b",
        "decode_m16:bfloat16:bfloat16:flat:main": "cake_blackwell_msa_20abed2035425df5fd26",
        "decode_m16:bfloat16:bfloat16:paged:main": "cake_blackwell_msa_68493caedaae47897f78",
        "decode_m16:bfloat16:float8_e4m3fn:flat:main": "cake_blackwell_msa_6e1adc85275f4235e8b4",
        "decode_m16:bfloat16:float8_e4m3fn:paged:main": "cake_blackwell_msa_f4c4135904984b79e0d9",
        "decode_m16:float16:float16:flat:main": "cake_blackwell_msa_fa9d1e77e08ed596123b",
        "decode_m16:float16:float16:paged:main": "cake_blackwell_msa_1a54cd2531ea0b47c832",
        "decode_uniform_fp8:paged:main": "cake_blackwell_msa_27f68603780b991a0f8a",
        "kmaj_p:flat:gqa16:main": "cake_blackwell_msa_651272858f5dbbf73ed7",
        "kmaj_p:flat:gqa16:reduce": "cake_blackwell_msa_e25d0af330369bae8525",
        "kmaj_p:paged:gqa16:main": "cake_blackwell_msa_63f39782ab0c25d9fbeb",
        "kmaj_p:paged:gqa16:reduce": "cake_blackwell_msa_e25d0af330369bae8525",
        "kmaj_p:paged:gqa8:main": "cake_blackwell_msa_1d0a9120638c2d29c6bf",
        "kmaj_p:paged:gqa8:reduce": "cake_blackwell_msa_e25d0af330369bae8525",
        "long_bf16_reverse:flat:gqa16:main": "cake_blackwell_msa_20354ee19ac6500cd3e7",
        "long_bf16_reverse:flat:gqa16:reduce": "cake_blackwell_msa_20585affcadecd41283e",
        "long_bf16_reverse:paged:gqa16:main": "cake_blackwell_msa_f6244dc9e9c70766baff",
        "long_bf16_reverse:paged:gqa16:reduce": "cake_blackwell_msa_20585affcadecd41283e",
        "long_bf16_reverse:paged:gqa8:main": "cake_blackwell_msa_6d3a9fa0d331c4f4f6c5",
        "long_bf16_reverse:paged:gqa8:reduce": "cake_blackwell_msa_7abe03b6bbeb05295938",
        "prefill_union:bfloat16:bfloat16:flat:gqa0:any:main": "cake_blackwell_msa_0c83769fc6cdcfadb2d0",
        "prefill_union:bfloat16:bfloat16:flat:gqa16:any:main": "cake_blackwell_msa_a070ff8bfefdf7e4c6e9",
        "prefill_union:bfloat16:bfloat16:flat:gqa8:any:main": "cake_blackwell_msa_0bb6210f55740fc901ee",
        "prefill_union:bfloat16:bfloat16:paged:gqa0:any:main": "cake_blackwell_msa_122d046ce6496a110567",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:any:main": "cake_blackwell_msa_cb2d66040aa8dd689059",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_large:main": "cake_blackwell_msa_94e24f46ab7b87bb0dfb",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_mask64:main": "cake_blackwell_msa_33a4eea7ce8a780334bd",
        "prefill_union:bfloat16:bfloat16:paged:gqa8:any:main": "cake_blackwell_msa_94e0b9ebcc109376d181",
        "prefill_union:bfloat16:float8_e4m3fn:flat:gqa0:any:main": "cake_blackwell_msa_63e75266177bb32039c8",
        "prefill_union:bfloat16:float8_e4m3fn:paged:gqa0:any:main": "cake_blackwell_msa_8a61e37bccdaf60470ef",
        "prefill_union:float16:float16:flat:gqa0:any:main": "cake_blackwell_msa_2ffd8cc9735419ada74a",
        "prefill_union:float16:float16:paged:gqa0:any:main": "cake_blackwell_msa_9769cb736d4783aed640",
        "topk_select:main": "cake_blackwell_msa_2be65623fe731f383035",
        "topk_select_keq:main": "cake_blackwell_msa_81fdf422f4015ca16a10",
        "topk_select_q32:main": "cake_blackwell_msa_55d535f060e0f1832077",
        "topk_select_streaming:main": "cake_blackwell_msa_bc07249d0e47672e8e26",
    },
    "sm103a": {
        "decode_fp8_q1:q1_flat_xform2:main": "cake_blackwell_msa_257bad493205f6bf5e16",
        "decode_fp8_q1:q1_paged_xform2:main": "cake_blackwell_msa_c3cc6426f72489566e5b",
        "decode_m16:bfloat16:bfloat16:flat:main": "cake_blackwell_msa_20abed2035425df5fd26",
        "decode_m16:bfloat16:bfloat16:paged:main": "cake_blackwell_msa_68493caedaae47897f78",
        "decode_m16:bfloat16:float8_e4m3fn:flat:main": "cake_blackwell_msa_6e1adc85275f4235e8b4",
        "decode_m16:bfloat16:float8_e4m3fn:paged:main": "cake_blackwell_msa_f4c4135904984b79e0d9",
        "decode_m16:float16:float16:flat:main": "cake_blackwell_msa_fa9d1e77e08ed596123b",
        "decode_m16:float16:float16:paged:main": "cake_blackwell_msa_1a54cd2531ea0b47c832",
        "decode_uniform_fp8:paged:main": "cake_blackwell_msa_27f68603780b991a0f8a",
        "kmaj_p:flat:gqa16:main": "cake_blackwell_msa_651272858f5dbbf73ed7",
        "kmaj_p:flat:gqa16:reduce": "cake_blackwell_msa_e25d0af330369bae8525",
        "kmaj_p:paged:gqa16:main": "cake_blackwell_msa_63f39782ab0c25d9fbeb",
        "kmaj_p:paged:gqa16:reduce": "cake_blackwell_msa_e25d0af330369bae8525",
        "kmaj_p:paged:gqa8:main": "cake_blackwell_msa_1d0a9120638c2d29c6bf",
        "kmaj_p:paged:gqa8:reduce": "cake_blackwell_msa_e25d0af330369bae8525",
        "long_bf16_reverse:flat:gqa16:main": "cake_blackwell_msa_20354ee19ac6500cd3e7",
        "long_bf16_reverse:flat:gqa16:reduce": "cake_blackwell_msa_20585affcadecd41283e",
        "long_bf16_reverse:paged:gqa16:main": "cake_blackwell_msa_f6244dc9e9c70766baff",
        "long_bf16_reverse:paged:gqa16:reduce": "cake_blackwell_msa_20585affcadecd41283e",
        "long_bf16_reverse:paged:gqa8:main": "cake_blackwell_msa_6d3a9fa0d331c4f4f6c5",
        "long_bf16_reverse:paged:gqa8:reduce": "cake_blackwell_msa_7abe03b6bbeb05295938",
        "prefill_union:bfloat16:bfloat16:flat:gqa0:any:main": "cake_blackwell_msa_0c83769fc6cdcfadb2d0",
        "prefill_union:bfloat16:bfloat16:flat:gqa16:any:main": "cake_blackwell_msa_a070ff8bfefdf7e4c6e9",
        "prefill_union:bfloat16:bfloat16:flat:gqa8:any:main": "cake_blackwell_msa_0bb6210f55740fc901ee",
        "prefill_union:bfloat16:bfloat16:paged:gqa0:any:main": "cake_blackwell_msa_122d046ce6496a110567",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:any:main": "cake_blackwell_msa_cb2d66040aa8dd689059",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_large:main": "cake_blackwell_msa_94e24f46ab7b87bb0dfb",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_mask64:main": "cake_blackwell_msa_33a4eea7ce8a780334bd",
        "prefill_union:bfloat16:bfloat16:paged:gqa8:any:main": "cake_blackwell_msa_94e0b9ebcc109376d181",
        "prefill_union:bfloat16:float8_e4m3fn:flat:gqa0:any:main": "cake_blackwell_msa_63e75266177bb32039c8",
        "prefill_union:bfloat16:float8_e4m3fn:paged:gqa0:any:main": "cake_blackwell_msa_8a61e37bccdaf60470ef",
        "prefill_union:float16:float16:flat:gqa0:any:main": "cake_blackwell_msa_2ffd8cc9735419ada74a",
        "prefill_union:float16:float16:paged:gqa0:any:main": "cake_blackwell_msa_9769cb736d4783aed640",
        "topk_select:main": "cake_blackwell_msa_f8b0f3032c3d2d438e19",
        "topk_select_keq:main": "cake_blackwell_msa_81fdf422f4015ca16a10",
        "topk_select_q32:main": "cake_blackwell_msa_55d535f060e0f1832077",
        "topk_select_streaming:main": "cake_blackwell_msa_bc07249d0e47672e8e26",
    },
}

# Programs built for each target (ahead-of-time build list).
BLACKWELL_MSA_VARIANTS_BY_TARGET: dict[
    BlackwellMSATarget, tuple[BlackwellMSAVariant, ...]
] = {
    target: tuple(name for name, record in MODULES.items() if arch in record["arches"])
    for target, arch in _TARGET_ARCH.items()
}
BLACKWELL_MSA_VARIANTS: tuple[BlackwellMSAVariant, ...] = tuple(MODULES)


def _csrc_dir() -> Path:
    for root in (
        jit_env.FLASHINFER_CSRC_DIR,
        Path(__file__).resolve().parents[2] / "csrc",
    ):
        if (root / "blackwell_msa").exists():
            return root
    raise FileNotFoundError(
        "Blackwell MSA CUDA sources were not found under csrc/blackwell_msa"
    )


def _include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    return Path(__file__).resolve().parents[2] / "include"


def route_program(key: str, target: BlackwellMSATarget) -> str:
    """Program name of one logical route (``<route key>:<stage>``) on ``target``."""
    try:
        return ROUTES[target][key]
    except KeyError:
        raise KeyError(
            f"no Blackwell MSA program for route {key!r} on {target}"
        ) from None


def get_blackwell_msa_uri(name: BlackwellMSAVariant, target: BlackwellMSATarget) -> str:
    """Target-specific JIT / AOT key of one program (the two targets never share a library)."""
    if target not in _TARGET_ARCH:
        raise ValueError(f"unsupported Blackwell MSA target: {target}")
    record = MODULES.get(name)
    if record is None or _TARGET_ARCH[target] not in record["arches"]:
        raise ValueError(f"unsupported Blackwell MSA program/target: {name}/{target}")
    return f"blackwell_msa_{name}_{target}"


@functools.cache
def gen_blackwell_msa_module(
    name: BlackwellMSAVariant, target: BlackwellMSATarget
) -> JitSpec:
    uri = get_blackwell_msa_uri(name, target)
    record = MODULES[name]
    csrc = _csrc_dir()
    # The record's ``compile_flags`` carry the numerics the program was built and
    # validated with; the JIT default (``use_fast_math=True``) would compile the
    # programs whose record has no ``--use_fast_math`` with a different one.
    return gen_jit_spec(
        name=uri,
        sources=[csrc / path for path in record["sources"]],
        extra_cuda_cflags=[*_NVCC_FLAGS[target], *record["compile_flags"]],
        extra_include_paths=[csrc / "blackwell_msa", csrc, _include_dir()],
        use_fast_math=False,
    )


@functools.cache
def load_blackwell_msa_module(name: BlackwellMSAVariant, target: BlackwellMSATarget):
    return gen_blackwell_msa_module(name, target).build_and_load()


def get_blackwell_msa_module(name: BlackwellMSAVariant, target: BlackwellMSATarget):
    """The loaded module of one program (the entry point ``flashinfer.jit`` re-exports)."""
    return load_blackwell_msa_module(name, target)


__all__ = [
    "ARG_PLANS",
    "BLACKWELL_MSA_VARIANTS",
    "BLACKWELL_MSA_VARIANTS_BY_TARGET",
    "MODULES",
    "ROUTES",
    "BlackwellMSATarget",
    "BlackwellMSAVariant",
    "gen_blackwell_msa_module",
    "get_blackwell_msa_module",
    "get_blackwell_msa_uri",
    "load_blackwell_msa_module",
    "route_program",
]
