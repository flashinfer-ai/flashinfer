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
from ...jit.core import (
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Explicit target-owned registration of the generated programs.  One record per
# (architecture, route, split-KV factor): the ``swap_tsk`` records (one per
# split factor) carry the persistent decode kernel with its translation units,
# compile flags, FFI entry and argument plan, plus the resident-CTA count per SM
# the host uses to size the persistent grid; the optional ``short`` record of an
# architecture carries the eight-CTA-cluster register-MMA program that serves
# work items of at most ``max_pages`` selected pages (``cluster`` CTAs per item,
# at most ``max_clusters`` clusters per launch).  Populated verbatim by the
# generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_msa_nvfp4_decode_sm_100a_short": {
        "arch": "sm_100a",
        "route": "short",
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
        "main": {
            "module": "cake_msa_nvfp4_decode_3315c75c98f2051881c7",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_3315c75c98f2051881c7_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_3315c75c98f2051881c7_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Q"],
                ["buffer", "K"],
                ["buffer", "K_scale"],
                ["buffer", "V"],
                ["buffer", "V_scale"],
                ["buffer", "O"],
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
                ["parameter", "k_page_stride"],
                ["parameter", "k_head_stride"],
                ["parameter", "ks_page_stride"],
                ["parameter", "ks_head_stride"],
                ["parameter", "v_page_stride"],
                ["parameter", "v_head_stride"],
                ["parameter", "vs_page_stride"],
                ["parameter", "vs_head_stride"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "6a609efdbb2e27efa3773e34f569952d6d86299d6e0f5c823e4c3762ce5c2c36",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "6a609efdbb2e27efa3773e34f569952d6d86299d6e0f5c823e4c3762ce5c2c36",
    },
    "cake_msa_nvfp4_decode_sm_100a_split1": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_123f8bc7650339f2c1a0",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_123f8bc7650339f2c1a0_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_123f8bc7650339f2c1a0_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "a4f435c944046afa9c56ff9d26350a5b23e6eab577d6565362348a6d1c8f5769",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "a4f435c944046afa9c56ff9d26350a5b23e6eab577d6565362348a6d1c8f5769",
    },
    "cake_msa_nvfp4_decode_sm_100a_split2": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_815e077b27d49bfe8b02",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_815e077b27d49bfe8b02_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_815e077b27d49bfe8b02_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "a5c37c05ae279dc588e8104e49d6d627e6211e7d3f3eda9a9a64cdbb30943bc8",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "a5c37c05ae279dc588e8104e49d6d627e6211e7d3f3eda9a9a64cdbb30943bc8",
    },
    "cake_msa_nvfp4_decode_sm_100a_split4": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_55399337fc30e0c4cf3f",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_55399337fc30e0c4cf3f_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_55399337fc30e0c4cf3f_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "4c55213d54ad4bc10f4057d0a54711382adff7fdb2fb6cfc7066f0ec345b3493",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "4c55213d54ad4bc10f4057d0a54711382adff7fdb2fb6cfc7066f0ec345b3493",
    },
    "cake_msa_nvfp4_decode_sm_100a_split8": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_5564f84b1ecdb6abe723",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_5564f84b1ecdb6abe723_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_5564f84b1ecdb6abe723_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "7b17495ab4baf21e55a368ee503704061878080550f3970fbd8c314967498b86",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "7b17495ab4baf21e55a368ee503704061878080550f3970fbd8c314967498b86",
    },
    "cake_msa_nvfp4_decode_sm_103a_short": {
        "arch": "sm_103a",
        "route": "short",
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
        "main": {
            "module": "cake_msa_nvfp4_decode_a986570485bc48cb7dfb",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_a986570485bc48cb7dfb_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_a986570485bc48cb7dfb_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Q"],
                ["buffer", "K"],
                ["buffer", "K_scale"],
                ["buffer", "V"],
                ["buffer", "V_scale"],
                ["buffer", "O"],
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
                ["parameter", "k_page_stride"],
                ["parameter", "k_head_stride"],
                ["parameter", "ks_page_stride"],
                ["parameter", "ks_head_stride"],
                ["parameter", "v_page_stride"],
                ["parameter", "v_head_stride"],
                ["parameter", "vs_page_stride"],
                ["parameter", "vs_head_stride"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "e6a16604a5d78197f115dffc39aa7d8d705112427fb30416c9a2e4e45665c6e7",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "e6a16604a5d78197f115dffc39aa7d8d705112427fb30416c9a2e4e45665c6e7",
    },
    "cake_msa_nvfp4_decode_sm_103a_split1": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_df10120763f6eabf9bfc",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_df10120763f6eabf9bfc_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_df10120763f6eabf9bfc_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "2910fb5c4f4fb7b6c0c72b7193af35ba525b50891d770140bc4440a73d07b859",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "2910fb5c4f4fb7b6c0c72b7193af35ba525b50891d770140bc4440a73d07b859",
    },
    "cake_msa_nvfp4_decode_sm_103a_split2": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_300bb58da7076d94526a",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_300bb58da7076d94526a_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_300bb58da7076d94526a_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "7ac3187ceb73c50c3ae4840690b0c39fcfb26e7cade641356c9e8043a3c4f698",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "7ac3187ceb73c50c3ae4840690b0c39fcfb26e7cade641356c9e8043a3c4f698",
    },
    "cake_msa_nvfp4_decode_sm_103a_split4": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_d226a8e29c2d55209e73",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_d226a8e29c2d55209e73_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_d226a8e29c2d55209e73_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "ea397df957a596dc90d757d003bcf9eabec7793f4afa68434f0e1e421576c72a",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "ea397df957a596dc90d757d003bcf9eabec7793f4afa68434f0e1e421576c72a",
    },
    "cake_msa_nvfp4_decode_sm_103a_split8": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_5e5a1f178ce88407c390",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_5e5a1f178ce88407c390_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_5e5a1f178ce88407c390_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "c77f61b041ff81119c8f82660449343a5f841ae0491305d6e52fc207c1d6d3e9",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "c77f61b041ff81119c8f82660449343a5f841ae0491305d6e52fc207c1d6d3e9",
    },
    "cake_msa_nvfp4_decode_sm_107a_short": {
        "arch": "sm_107a",
        "route": "short",
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
        "main": {
            "module": "cake_msa_nvfp4_decode_6a8694ec487b295ace4f",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_6a8694ec487b295ace4f_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_6a8694ec487b295ace4f_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "Q"],
                ["buffer", "K"],
                ["buffer", "K_scale"],
                ["buffer", "V"],
                ["buffer", "V_scale"],
                ["buffer", "O"],
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
                ["parameter", "k_page_stride"],
                ["parameter", "k_head_stride"],
                ["parameter", "ks_page_stride"],
                ["parameter", "ks_head_stride"],
                ["parameter", "v_page_stride"],
                ["parameter", "v_head_stride"],
                ["parameter", "vs_page_stride"],
                ["parameter", "vs_head_stride"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "e522be77288fef5222515662569d6fe8241dd4c2d3560e4017b38754798d141c",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "e522be77288fef5222515662569d6fe8241dd4c2d3560e4017b38754798d141c",
    },
    "cake_msa_nvfp4_decode_sm_107a_split1": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_5e871ee52ee85a414ae1",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_5e871ee52ee85a414ae1_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_5e871ee52ee85a414ae1_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "4912a641f6902389154081346bf4bf39e17dbc9e6f60f7ceb2b8770447519b03",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "4912a641f6902389154081346bf4bf39e17dbc9e6f60f7ceb2b8770447519b03",
    },
    "cake_msa_nvfp4_decode_sm_107a_split2": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_182f9d52bf1c6572d4d0",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_182f9d52bf1c6572d4d0_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_182f9d52bf1c6572d4d0_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "4ef4ef45ad9ef5ef04bdb956ac96f8755392257e47da00cab18d016c7fd1712a",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "4ef4ef45ad9ef5ef04bdb956ac96f8755392257e47da00cab18d016c7fd1712a",
    },
    "cake_msa_nvfp4_decode_sm_107a_split2_tail": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "tail": True,
        "cluster_capacity": {"212": 212},
        "main": {
            "module": "cake_msa_nvfp4_decode_0641ef0627df9beabd69",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_0641ef0627df9beabd69_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_0641ef0627df9beabd69_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "432124bc42f3ac6265b78b0900dc63aa882b0995c4b6050618913b236603ccd7",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "432124bc42f3ac6265b78b0900dc63aa882b0995c4b6050618913b236603ccd7",
    },
    "cake_msa_nvfp4_decode_sm_107a_split4": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_db06f6a504f85d470e05",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_db06f6a504f85d470e05_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_db06f6a504f85d470e05_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "8b045c58f2a9b6943bf7eea47724802c6c575b659e68929169b6744640ed55f8",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "8b045c58f2a9b6943bf7eea47724802c6c575b659e68929169b6744640ed55f8",
    },
    "cake_msa_nvfp4_decode_sm_107a_split8": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_b12c6b9e2f99940d3ba7",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_b12c6b9e2f99940d3ba7_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_b12c6b9e2f99940d3ba7_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
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
            "closure_sha256": "2e53f88bbec0edfa4fc05bcd08f6b4913e51284005f8ccb47a8138dff478c83d",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "2e53f88bbec0edfa4fc05bcd08f6b4913e51284005f8ccb47a8138dff478c83d",
    },
}

STAGES = ("main",)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}


@functools.cache
def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?

    ``compute_107a`` (Rubin) needs a CUDA toolkit that lists it; a toolkit
    without it must decline the sm_107a programs up front instead of failing
    inside the JIT build, so a checkout that registers sm_107a programs stays
    importable and testable on a toolkit that only knows 10.0 / 10.3.
    """
    if arch not in ARCH_NVCC_FLAGS:
        return False
    if arch == "sm_107a":
        from flashinfer.compilation_context import _nvcc_supports_sm107

        return bool(_nvcc_supports_sm107())
    return True


def route_of(record: dict[str, Any]) -> str:
    """Route family of a registry record (``swap_tsk`` persistent or ``short`` cluster program)."""
    return str(record.get("route", "swap_tsk"))


def registered_split_factors(arch: str) -> tuple[int, ...]:
    """Split-KV factors with a registered persistent program for ``arch``."""
    return tuple(
        sorted(
            int(record["splits"])
            for record in MODULES.values()
            if record["arch"] == arch
            and route_of(record) == "swap_tsk"
            and not record.get("tail", False)
        )
    )


def tail_records(arch: str) -> list[dict[str, Any]]:
    """Registered last-round-split programs of ``arch``, ascending split factor.

    A tail program runs every full persistent round unsplit and only the
    remainder items of the last round as ``splits``-CTA cluster units; its
    ``cluster_capacity`` maps a part's SM count to the CTAs the driver
    co-schedules as such clusters (clusters must fit inside one GPC).
    """
    return sorted(
        (
            record
            for record in MODULES.values()
            if record["arch"] == arch
            and route_of(record) == "swap_tsk"
            and record.get("tail", False)
        ),
        key=lambda record: int(record["splits"]),
    )


def select_module(arch: str, splits: int, tail: bool = False) -> str:
    """Return the registered persistent module name for ``arch`` and split factor ``splits``
    (``tail`` selects the last-round-split program of that factor)."""
    for name, record in MODULES.items():
        if (
            record["arch"] == arch
            and route_of(record) == "swap_tsk"
            and int(record["splits"]) == int(splits)
            and bool(record.get("tail", False)) == bool(tail)
        ):
            return name
    raise NotImplementedError(
        "The generated NVFP4 MSA decode program for "
        f"{arch} with split factor {splits}{' (last-round split)' if tail else ''} "
        "is not registered in this checkout"
    )


def select_short_module(arch: str) -> str | None:
    """Return the short-item cluster program registered for ``arch``, or ``None``.

    The short program serves work items whose requests span at most
    ``MODULES[name]["max_pages"]`` pages; an architecture without it serves
    those items with the persistent program.
    """
    names = [
        name
        for name, record in MODULES.items()
        if record["arch"] == arch and route_of(record) == "short"
    ]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one short-item NVFP4 MSA decode program: {names}"
        )
    return names[0] if names else None


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
def gen_cake_msa_nvfp4_decode_module(name, stage):
    record = MODULES[name]
    if not toolchain_supports(record["arch"]):
        raise RuntimeError(
            f"generated NVFP4 MSA decode program {name!r} targets {record['arch']}, "
            "which the CUDA toolkit of this checkout cannot compile (nvcc does not "
            "list compute_107a; a Rubin-capable toolkit is required)"
        )
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_msa_nvfp4_decode_module(name, stage):
    return gen_cake_msa_nvfp4_decode_module(name, stage).build_and_load()
