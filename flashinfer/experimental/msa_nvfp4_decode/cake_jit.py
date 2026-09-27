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
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags, sm107a_nvcc_flags

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
            "module": "cake_msa_nvfp4_decode_f62b8cc4971210ce5df8",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_f62b8cc4971210ce5df8_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_f62b8cc4971210ce5df8_binding.cu",
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
            "closure_sha256": "ad1613401c3f1abdffce42e0e31f3dd790d726a003cb820273ea812f86e2c253",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "ad1613401c3f1abdffce42e0e31f3dd790d726a003cb820273ea812f86e2c253",
    },
    "cake_msa_nvfp4_decode_sm_100a_split1": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_74caf47eb3f9bbc0c26f",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_74caf47eb3f9bbc0c26f_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_74caf47eb3f9bbc0c26f_binding.cu",
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
            "closure_sha256": "40a19d1002c1bbb4cc25b6fd234b523f7f81d9cf9fd1cbbd5d9edeea109c4197",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "40a19d1002c1bbb4cc25b6fd234b523f7f81d9cf9fd1cbbd5d9edeea109c4197",
    },
    "cake_msa_nvfp4_decode_sm_100a_split2": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_2071c802132b03213e1b",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_2071c802132b03213e1b_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_2071c802132b03213e1b_binding.cu",
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
            "closure_sha256": "f1a144cb1b5c2b59ea52c411f92ae41e6a195e881c2b4837f30c02523b34e1c1",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "f1a144cb1b5c2b59ea52c411f92ae41e6a195e881c2b4837f30c02523b34e1c1",
    },
    "cake_msa_nvfp4_decode_sm_100a_split4": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_0ded7f5f18b1e269ed7f",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_0ded7f5f18b1e269ed7f_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_0ded7f5f18b1e269ed7f_binding.cu",
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
            "closure_sha256": "15b4f64596418e733eb17468b4e681fe7ccae2dd77ad2594de1735e56b30563d",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "15b4f64596418e733eb17468b4e681fe7ccae2dd77ad2594de1735e56b30563d",
    },
    "cake_msa_nvfp4_decode_sm_100a_split8": {
        "arch": "sm_100a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_055676901ed5624ae569",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_055676901ed5624ae569_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_055676901ed5624ae569_binding.cu",
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
            "closure_sha256": "b387f822276773d09a44327411f5df9e341b4df25907dbcfe6d3277faf3d5520",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "b387f822276773d09a44327411f5df9e341b4df25907dbcfe6d3277faf3d5520",
    },
    "cake_msa_nvfp4_decode_sm_103a_short": {
        "arch": "sm_103a",
        "route": "short",
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
        "main": {
            "module": "cake_msa_nvfp4_decode_f750028fa8e0583e1812",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_f750028fa8e0583e1812_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_f750028fa8e0583e1812_binding.cu",
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
            "closure_sha256": "e72f6e6dcb508a79efb1e5bd97316ef96819e384283ffe912bb39cd0386edaed",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "e72f6e6dcb508a79efb1e5bd97316ef96819e384283ffe912bb39cd0386edaed",
    },
    "cake_msa_nvfp4_decode_sm_103a_split1": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_dc1e1dae0ac8b407b225",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_dc1e1dae0ac8b407b225_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_dc1e1dae0ac8b407b225_binding.cu",
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
            "closure_sha256": "7d357847242f3373b10ca8c1a1030dc45dd8f7b0f4f512d5659ebef74af5a91a",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "7d357847242f3373b10ca8c1a1030dc45dd8f7b0f4f512d5659ebef74af5a91a",
    },
    "cake_msa_nvfp4_decode_sm_103a_split2": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_8622acc83797aaf2f6c4",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_8622acc83797aaf2f6c4_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_8622acc83797aaf2f6c4_binding.cu",
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
            "closure_sha256": "8d6ce1913b5e89b1b1a9f25bcb9fa492a3e326d906a2f094ae7e8deabf6f7ac9",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "8d6ce1913b5e89b1b1a9f25bcb9fa492a3e326d906a2f094ae7e8deabf6f7ac9",
    },
    "cake_msa_nvfp4_decode_sm_103a_split4": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_ba79d275e912eecb1198",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_ba79d275e912eecb1198_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_ba79d275e912eecb1198_binding.cu",
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
            "closure_sha256": "0b2aa453df4b8112abe65f9d2784c2175cce8cc52766f3329bf021710c2459a8",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "0b2aa453df4b8112abe65f9d2784c2175cce8cc52766f3329bf021710c2459a8",
    },
    "cake_msa_nvfp4_decode_sm_103a_split8": {
        "arch": "sm_103a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_7da0c10a58d37547c262",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_7da0c10a58d37547c262_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_7da0c10a58d37547c262_binding.cu",
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
            "closure_sha256": "20492b6c3183e9515ec4ad93f29ceb13ef71d7e835dae0d4f05fbe9c675aede0",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "20492b6c3183e9515ec4ad93f29ceb13ef71d7e835dae0d4f05fbe9c675aede0",
    },
    "cake_msa_nvfp4_decode_sm_107a_short": {
        "arch": "sm_107a",
        "route": "short",
        "max_pages": 4,
        "cluster": 8,
        "max_clusters": 16,
        "main": {
            "module": "cake_msa_nvfp4_decode_0b0bb3212c5d3c2b372c",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_0b0bb3212c5d3c2b372c_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_0b0bb3212c5d3c2b372c_binding.cu",
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
            "closure_sha256": "e6ad63bc63047c2786a7185b480b90333853435eb42360686f220dd93da7b529",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "e6ad63bc63047c2786a7185b480b90333853435eb42360686f220dd93da7b529",
    },
    "cake_msa_nvfp4_decode_sm_107a_split1": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_780d72feabb01882fabc",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_780d72feabb01882fabc_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_780d72feabb01882fabc_binding.cu",
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
            "closure_sha256": "dd805d3d2cb81ee535d76be70a00354e55c4c08a4cf7963458b33ff450371e5e",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "dd805d3d2cb81ee535d76be70a00354e55c4c08a4cf7963458b33ff450371e5e",
    },
    "cake_msa_nvfp4_decode_sm_107a_split2": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_24c3dac42f9ae886e492",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_24c3dac42f9ae886e492_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_24c3dac42f9ae886e492_binding.cu",
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
            "closure_sha256": "3fc49ad3b42a62e5894c4cc08dc5111ccd507fd5a533fa0cb275def31d7c24cd",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "3fc49ad3b42a62e5894c4cc08dc5111ccd507fd5a533fa0cb275def31d7c24cd",
    },
    "cake_msa_nvfp4_decode_sm_107a_split2_tail": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 2,
        "ctas_per_sm": 1,
        "tail": True,
        "cluster_capacity": {"212": 212},
        "main": {
            "module": "cake_msa_nvfp4_decode_0829f13190fd65e47445",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_0829f13190fd65e47445_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_0829f13190fd65e47445_binding.cu",
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
            "closure_sha256": "17faf5a69813405cf2315d291c77d6f640fd74bf7fb3e446442b8d74e9ea2700",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "17faf5a69813405cf2315d291c77d6f640fd74bf7fb3e446442b8d74e9ea2700",
    },
    "cake_msa_nvfp4_decode_sm_107a_split4": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_22cdb6a3cdd3ffc233f9",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_22cdb6a3cdd3ffc233f9_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_22cdb6a3cdd3ffc233f9_binding.cu",
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
            "closure_sha256": "c2a681349d85526cca365f40731cb4f62833e177b14ad0305e1fd0c9d7a29974",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "c2a681349d85526cca365f40731cb4f62833e177b14ad0305e1fd0c9d7a29974",
    },
    "cake_msa_nvfp4_decode_sm_107a_split8": {
        "arch": "sm_107a",
        "route": "swap_tsk",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_9c00c66c9b8126c7f8f1",
            "sources": [
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_9c00c66c9b8126c7f8f1_kernel.cu",
                "cake_msa_nvfp4_decode/sm_107a/cake_msa_nvfp4_decode_9c00c66c9b8126c7f8f1_binding.cu",
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
            "closure_sha256": "4d7b84e6d6f66e7c23b110cd6ae85f88b2e92b391bdb06cd8d790254186a3579",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "4d7b84e6d6f66e7c23b110cd6ae85f88b2e92b391bdb06cd8d790254186a3579",
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
