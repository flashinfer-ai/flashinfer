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
    "cake_blackwell_msa_02eac1ffa510c1f3d961": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_02eac1ffa510c1f3d961_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_02eac1ffa510c1f3d961_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_044ebdf2b23464ba3049": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_044ebdf2b23464ba3049_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_044ebdf2b23464ba3049_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_04f607bad91a7de6910b": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_04f607bad91a7de6910b_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_04f607bad91a7de6910b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_077434d0c6f2e2ac8a55": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_077434d0c6f2e2ac8a55_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_077434d0c6f2e2ac8a55_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_0d391c271a9e1dd7de1b": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_0d391c271a9e1dd7de1b_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_0d391c271a9e1dd7de1b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_fp8_q1_paged_xform2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_0e42bddc8e39e77775dc": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_0e42bddc8e39e77775dc_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_0e42bddc8e39e77775dc_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_1befeb7dca7d89f3cfe0": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_1befeb7dca7d89f3cfe0_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_1befeb7dca7d89f3cfe0_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_2448a5362a514c775e1c": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_2448a5362a514c775e1c_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_2448a5362a514c775e1c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_249de81529722ce8f3bb": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_249de81529722ce8f3bb_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_249de81529722ce8f3bb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_270296b5b4d459acf326": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_270296b5b4d459acf326_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_270296b5b4d459acf326_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_2722338b9f9f22a3f32c": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_2722338b9f9f22a3f32c_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_2722338b9f9f22a3f32c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_reduce",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_2949bc72ee78a52859c9": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_2949bc72ee78a52859c9_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_2949bc72ee78a52859c9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_fp8_q1_flat_xform2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_39657d215e82b0d3aa39": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_39657d215e82b0d3aa39_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_39657d215e82b0d3aa39_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_4fca3487da07d952e083": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_4fca3487da07d952e083_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_4fca3487da07d952e083_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_uniform_fp8",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_5848221d6bba8237cc0c": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_5848221d6bba8237cc0c_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_5848221d6bba8237cc0c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_67fbde298dbcafee2cc0": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_67fbde298dbcafee2cc0_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_67fbde298dbcafee2cc0_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_711d5eeebecfcde61503": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_711d5eeebecfcde61503_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_711d5eeebecfcde61503_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_reduce",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_74572ca8a8bc44c5f461": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_74572ca8a8bc44c5f461_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_74572ca8a8bc44c5f461_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_8828a5c33a15158853e5": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_8828a5c33a15158853e5_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_8828a5c33a15158853e5_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_89539a2a0a479c9002e5": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_89539a2a0a479c9002e5_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_89539a2a0a479c9002e5_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_9c2eacf905b06a86fca3": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_9c2eacf905b06a86fca3_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_9c2eacf905b06a86fca3_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_c14b5fe5d3aa9f47ab07": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_c14b5fe5d3aa9f47ab07_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_c14b5fe5d3aa9f47ab07_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
        "arches": ["sm_100a"],
    },
    "cake_blackwell_msa_c3da0a7883053ffdf6fe": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_c3da0a7883053ffdf6fe_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_c3da0a7883053ffdf6fe_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_c6ed8357874024ae9789": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_c6ed8357874024ae9789_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_c6ed8357874024ae9789_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_cc96ea7befd47c2a0b22": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_cc96ea7befd47c2a0b22_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_cc96ea7befd47c2a0b22_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_d03b9d08c1e9314fbc4a": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_d03b9d08c1e9314fbc4a_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_d03b9d08c1e9314fbc4a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select_streaming",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_d24af147efd25f98362a": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_d24af147efd25f98362a_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_d24af147efd25f98362a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "long_bf16_reverse_main",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_dcb5380b7118cf9060dc": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_dcb5380b7118cf9060dc_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_dcb5380b7118cf9060dc_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "decode_m16",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_e9c87303e5be2b03a12c": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_e9c87303e5be2b03a12c_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_e9c87303e5be2b03a12c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": "prefill_union",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_blackwell_msa_f35ca4f807d8df7077ed": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_f35ca4f807d8df7077ed_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_f35ca4f807d8df7077ed_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
        "arches": ["sm_103a"],
    },
    "cake_blackwell_msa_f756fe7c24c2f8829427": {
        "sources": [
            "blackwell_msa/cake_blackwell_msa_f756fe7c24c2f8829427_kernel.cu",
            "blackwell_msa/cake_blackwell_msa_f756fe7c24c2f8829427_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": "topk_select",
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
        "decode_fp8_q1:q1_flat_xform2:main": "cake_blackwell_msa_2949bc72ee78a52859c9",
        "decode_fp8_q1:q1_paged_xform2:main": "cake_blackwell_msa_0d391c271a9e1dd7de1b",
        "decode_m16:bfloat16:bfloat16:flat:main": "cake_blackwell_msa_2448a5362a514c775e1c",
        "decode_m16:bfloat16:bfloat16:paged:main": "cake_blackwell_msa_5848221d6bba8237cc0c",
        "decode_m16:bfloat16:float8_e4m3fn:flat:main": "cake_blackwell_msa_077434d0c6f2e2ac8a55",
        "decode_m16:bfloat16:float8_e4m3fn:paged:main": "cake_blackwell_msa_270296b5b4d459acf326",
        "decode_m16:float16:float16:flat:main": "cake_blackwell_msa_67fbde298dbcafee2cc0",
        "decode_m16:float16:float16:paged:main": "cake_blackwell_msa_dcb5380b7118cf9060dc",
        "decode_uniform_fp8:paged:main": "cake_blackwell_msa_4fca3487da07d952e083",
        "long_bf16_reverse:flat:gqa16:main": "cake_blackwell_msa_249de81529722ce8f3bb",
        "long_bf16_reverse:flat:gqa16:reduce": "cake_blackwell_msa_2722338b9f9f22a3f32c",
        "long_bf16_reverse:paged:gqa16:main": "cake_blackwell_msa_d24af147efd25f98362a",
        "long_bf16_reverse:paged:gqa16:reduce": "cake_blackwell_msa_2722338b9f9f22a3f32c",
        "long_bf16_reverse:paged:gqa8:main": "cake_blackwell_msa_c3da0a7883053ffdf6fe",
        "long_bf16_reverse:paged:gqa8:reduce": "cake_blackwell_msa_711d5eeebecfcde61503",
        "prefill_union:bfloat16:bfloat16:flat:gqa0:any:main": "cake_blackwell_msa_1befeb7dca7d89f3cfe0",
        "prefill_union:bfloat16:bfloat16:flat:gqa16:any:main": "cake_blackwell_msa_044ebdf2b23464ba3049",
        "prefill_union:bfloat16:bfloat16:flat:gqa8:any:main": "cake_blackwell_msa_0e42bddc8e39e77775dc",
        "prefill_union:bfloat16:bfloat16:paged:gqa0:any:main": "cake_blackwell_msa_8828a5c33a15158853e5",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:any:main": "cake_blackwell_msa_e9c87303e5be2b03a12c",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_large:main": "cake_blackwell_msa_39657d215e82b0d3aa39",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_mask64:main": "cake_blackwell_msa_9c2eacf905b06a86fca3",
        "prefill_union:bfloat16:bfloat16:paged:gqa8:any:main": "cake_blackwell_msa_04f607bad91a7de6910b",
        "prefill_union:bfloat16:float8_e4m3fn:flat:gqa0:any:main": "cake_blackwell_msa_cc96ea7befd47c2a0b22",
        "prefill_union:bfloat16:float8_e4m3fn:paged:gqa0:any:main": "cake_blackwell_msa_02eac1ffa510c1f3d961",
        "prefill_union:float16:float16:flat:gqa0:any:main": "cake_blackwell_msa_74572ca8a8bc44c5f461",
        "prefill_union:float16:float16:paged:gqa0:any:main": "cake_blackwell_msa_c6ed8357874024ae9789",
        "topk_select:main": "cake_blackwell_msa_c14b5fe5d3aa9f47ab07",
        "topk_select_keq:main": "cake_blackwell_msa_89539a2a0a479c9002e5",
        "topk_select_q32:main": "cake_blackwell_msa_f756fe7c24c2f8829427",
        "topk_select_streaming:main": "cake_blackwell_msa_d03b9d08c1e9314fbc4a",
    },
    "sm103a": {
        "decode_fp8_q1:q1_flat_xform2:main": "cake_blackwell_msa_2949bc72ee78a52859c9",
        "decode_fp8_q1:q1_paged_xform2:main": "cake_blackwell_msa_0d391c271a9e1dd7de1b",
        "decode_m16:bfloat16:bfloat16:flat:main": "cake_blackwell_msa_2448a5362a514c775e1c",
        "decode_m16:bfloat16:bfloat16:paged:main": "cake_blackwell_msa_5848221d6bba8237cc0c",
        "decode_m16:bfloat16:float8_e4m3fn:flat:main": "cake_blackwell_msa_077434d0c6f2e2ac8a55",
        "decode_m16:bfloat16:float8_e4m3fn:paged:main": "cake_blackwell_msa_270296b5b4d459acf326",
        "decode_m16:float16:float16:flat:main": "cake_blackwell_msa_67fbde298dbcafee2cc0",
        "decode_m16:float16:float16:paged:main": "cake_blackwell_msa_dcb5380b7118cf9060dc",
        "decode_uniform_fp8:paged:main": "cake_blackwell_msa_4fca3487da07d952e083",
        "long_bf16_reverse:flat:gqa16:main": "cake_blackwell_msa_249de81529722ce8f3bb",
        "long_bf16_reverse:flat:gqa16:reduce": "cake_blackwell_msa_2722338b9f9f22a3f32c",
        "long_bf16_reverse:paged:gqa16:main": "cake_blackwell_msa_d24af147efd25f98362a",
        "long_bf16_reverse:paged:gqa16:reduce": "cake_blackwell_msa_2722338b9f9f22a3f32c",
        "long_bf16_reverse:paged:gqa8:main": "cake_blackwell_msa_c3da0a7883053ffdf6fe",
        "long_bf16_reverse:paged:gqa8:reduce": "cake_blackwell_msa_711d5eeebecfcde61503",
        "prefill_union:bfloat16:bfloat16:flat:gqa0:any:main": "cake_blackwell_msa_1befeb7dca7d89f3cfe0",
        "prefill_union:bfloat16:bfloat16:flat:gqa16:any:main": "cake_blackwell_msa_044ebdf2b23464ba3049",
        "prefill_union:bfloat16:bfloat16:flat:gqa8:any:main": "cake_blackwell_msa_0e42bddc8e39e77775dc",
        "prefill_union:bfloat16:bfloat16:paged:gqa0:any:main": "cake_blackwell_msa_8828a5c33a15158853e5",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:any:main": "cake_blackwell_msa_e9c87303e5be2b03a12c",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_large:main": "cake_blackwell_msa_39657d215e82b0d3aa39",
        "prefill_union:bfloat16:bfloat16:paged:gqa16:causal_mask64:main": "cake_blackwell_msa_9c2eacf905b06a86fca3",
        "prefill_union:bfloat16:bfloat16:paged:gqa8:any:main": "cake_blackwell_msa_04f607bad91a7de6910b",
        "prefill_union:bfloat16:float8_e4m3fn:flat:gqa0:any:main": "cake_blackwell_msa_cc96ea7befd47c2a0b22",
        "prefill_union:bfloat16:float8_e4m3fn:paged:gqa0:any:main": "cake_blackwell_msa_02eac1ffa510c1f3d961",
        "prefill_union:float16:float16:flat:gqa0:any:main": "cake_blackwell_msa_74572ca8a8bc44c5f461",
        "prefill_union:float16:float16:paged:gqa0:any:main": "cake_blackwell_msa_c6ed8357874024ae9789",
        "topk_select:main": "cake_blackwell_msa_f35ca4f807d8df7077ed",
        "topk_select_keq:main": "cake_blackwell_msa_89539a2a0a479c9002e5",
        "topk_select_q32:main": "cake_blackwell_msa_f756fe7c24c2f8829427",
        "topk_select_streaming:main": "cake_blackwell_msa_d03b9d08c1e9314fbc4a",
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
    return gen_jit_spec(
        name=uri,
        sources=[csrc / path for path in record["sources"]],
        extra_cuda_cflags=[*_NVCC_FLAGS[target], *record["compile_flags"]],
        extra_include_paths=[csrc / "blackwell_msa", csrc, _include_dir()],
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
