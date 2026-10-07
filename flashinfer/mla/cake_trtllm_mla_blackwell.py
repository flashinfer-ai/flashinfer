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

# Generated Blackwell MLA decode backend (``backend="cake"`` of
# ``trtllm_batch_decode_with_kv_cache_mla``): dispatcher + JIT loader.
#
# ``MODULES`` / ``FAMILIES`` / ``ROUTES`` are written by the Cake generated-program
# exporter.  One generated device source per physical schedule serves every listed
# architecture; one compiled family binding per (domain, architecture) selects the
# kernel and launches it.  Route selection uses dtype, shapes, strides and cached
# device facts; the request KV lengths are read once per input identity only for
# the host-known geometries whose production route keys on them
# (``exact_route_candidate``).  Per-row metadata (row -> request, causal KV
# length, page rows, work order) is computed on the device with tensor ops.

from __future__ import annotations

import functools
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch

from ..jit import env as jit_env
from ..jit.core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from ..utils import log2e

MODULES: dict[str, dict[str, Any]] = {
    "cake_trtllm_mla_blackwell_03b4acdad2f17d77e6a8": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_03b4acdad2f17d77e6a8_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a"],
    },
    "cake_trtllm_mla_blackwell_4de3d4bf2f8c5467bf29": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_4de3d4bf2f8c5467bf29_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_5a25f65a14bfec7b2b1e": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_5a25f65a14bfec7b2b1e_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_6a2644cd7c80243f7cc3": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_6a2644cd7c80243f7cc3_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_6ba043a3192bee6bfe95": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_6ba043a3192bee6bfe95_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_71fdced8e63ff5ad0ba4": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_71fdced8e63ff5ad0ba4_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_103a"],
    },
    "cake_trtllm_mla_blackwell_74250d91903d28576cba": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_74250d91903d28576cba_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a"],
    },
    "cake_trtllm_mla_blackwell_79e49e216298d59fce33": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_79e49e216298d59fce33_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_88179fe5ff3ecd4952de": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_88179fe5ff3ecd4952de_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_103a"],
    },
    "cake_trtllm_mla_blackwell_c343762a5d3d9f326db2": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_c343762a5d3d9f326db2_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a"],
    },
    "cake_trtllm_mla_blackwell_d1a68c90cbc4aa0fcd6e": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_d1a68c90cbc4aa0fcd6e_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_d396fd37061fb32d5197": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_d396fd37061fb32d5197_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_103a"],
    },
    "cake_trtllm_mla_blackwell_d6056fe83ccbe21f79b3": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_d6056fe83ccbe21f79b3_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_d7731e89cd4b1ecceeb0": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_d7731e89cd4b1ecceeb0_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a"],
    },
    "cake_trtllm_mla_blackwell_db015a53334184a3005e": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_db015a53334184a3005e_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_trtllm_mla_blackwell_f2fc000e792120064e14": {
        "device": "cake_trtllm_mla_blackwell/cake_trtllm_mla_blackwell_f2fc000e792120064e14_kernel.cu",
        "compile_flags": ["--use_fast_math"],
        "arches": ["sm_103a"],
    },
}
FAMILIES: dict[str, dict[str, Any]] = {
    "cake_trtllm_mla_blackwell_family_3542f15da2f1ae0fc00c": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_3542f15da2f1ae0fc00c_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_6a2644cd7c80243f7cc3"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_vhalf",
    },
    "cake_trtllm_mla_blackwell_family_35a4a08ccde422e44143": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_35a4a08ccde422e44143_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_f2fc000e792120064e14"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_clc_exact007790",
    },
    "cake_trtllm_mla_blackwell_family_3fa9fe12db6254809466": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_3fa9fe12db6254809466_binding.cu",
        "dependencies": [
            "cake_trtllm_mla_blackwell_db015a53334184a3005e",
            "cake_trtllm_mla_blackwell_d1a68c90cbc4aa0fcd6e",
        ],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_half_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["workspace", "partial_output"],
            ["workspace", "partial_lse"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "partial_bmm2_scale"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_split"],
            ["parameter", "total_work_items"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "num_rows"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_native_split8_pdl",
    },
    "cake_trtllm_mla_blackwell_family_58aaa55b459e12fb35b8": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_58aaa55b459e12fb35b8_binding.cu",
        "dependencies": [
            "cake_trtllm_mla_blackwell_db015a53334184a3005e",
            "cake_trtllm_mla_blackwell_d1a68c90cbc4aa0fcd6e",
        ],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_half_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["workspace", "partial_output"],
            ["workspace", "partial_lse"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "partial_bmm2_scale"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_split"],
            ["parameter", "total_work_items"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "num_rows"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_native_split8_pdl",
    },
    "cake_trtllm_mla_blackwell_family_5a587bd20f8138cb26c9": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_5a587bd20f8138cb26c9_binding.cu",
        "dependencies": [
            "cake_trtllm_mla_blackwell_88179fe5ff3ecd4952de",
            "cake_trtllm_mla_blackwell_4de3d4bf2f8c5467bf29",
        ],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "output"],
            ["buffer", "lse"],
            ["buffer", "seq_lens"],
            ["buffer", "work_batch_indices"],
            ["buffer", "page_table"],
            ["workspace", "partial_output"],
            ["workspace", "partial_stats"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_split"],
            ["parameter", "num_queries"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "num_reduce_ctas"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_fp8_page64_pdl",
    },
    "cake_trtllm_mla_blackwell_family_5b3d4b45f82fac17edbd": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_5b3d4b45f82fac17edbd_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_5a25f65a14bfec7b2b1e"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_vquarter",
    },
    "cake_trtllm_mla_blackwell_family_69765950411de111b0b4": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_69765950411de111b0b4_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_6ba043a3192bee6bfe95"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "fp8_lut"],
            ["buffer", "page_table"],
            ["buffer", "sparse_indices"],
            ["buffer", "row_batches"],
            ["buffer", "row_seq_lens"],
            ["buffer", "output"],
            ["buffer", "lse"],
            ["buffer", "sinks"],
            ["parameter", "num_heads"],
            ["parameter", "qk_dim"],
            ["parameter", "value_dim"],
            ["parameter", "kv_stride"],
            ["parameter", "page_size"],
            ["parameter", "page_table_width"],
            ["parameter", "sparse_width"],
            ["parameter", "use_sparse"],
            ["parameter", "softmax_scale"],
            ["parameter", "bmm2_scale"],
            ["parameter", "enable_sink"],
            ["parameter", "write_lse"],
            ["parameter", "num_rows"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_tail",
    },
    "cake_trtllm_mla_blackwell_family_6e7d2a25e57e51cfa70f": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_6e7d2a25e57e51cfa70f_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_6ba043a3192bee6bfe95"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "fp8_lut"],
            ["buffer", "page_table"],
            ["buffer", "sparse_indices"],
            ["buffer", "row_batches"],
            ["buffer", "row_seq_lens"],
            ["buffer", "output"],
            ["buffer", "lse"],
            ["buffer", "sinks"],
            ["parameter", "num_heads"],
            ["parameter", "qk_dim"],
            ["parameter", "value_dim"],
            ["parameter", "kv_stride"],
            ["parameter", "page_size"],
            ["parameter", "page_table_width"],
            ["parameter", "sparse_width"],
            ["parameter", "use_sparse"],
            ["parameter", "softmax_scale"],
            ["parameter", "bmm2_scale"],
            ["parameter", "enable_sink"],
            ["parameter", "write_lse"],
            ["parameter", "num_rows"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_tail",
    },
    "cake_trtllm_mla_blackwell_family_73c2aeebeee855cabb5c": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_73c2aeebeee855cabb5c_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_5a25f65a14bfec7b2b1e"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_vquarter",
    },
    "cake_trtllm_mla_blackwell_family_964aa840394fcd0656c0": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_964aa840394fcd0656c0_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_6a2644cd7c80243f7cc3"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_vhalf",
    },
    "cake_trtllm_mla_blackwell_family_9eb0e53aa4b05afa531b": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_9eb0e53aa4b05afa531b_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_d396fd37061fb32d5197"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_clc_exact007826",
    },
    "cake_trtllm_mla_blackwell_family_a77fd5c07db49c791e35": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_a77fd5c07db49c791e35_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_c343762a5d3d9f326db2"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_clc_exact007826",
    },
    "cake_trtllm_mla_blackwell_family_c325c9f98ac456facec0": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_c325c9f98ac456facec0_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_d6056fe83ccbe21f79b3"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "fp8_lut"],
            ["buffer", "page_table"],
            ["buffer", "sparse_indices"],
            ["buffer", "row_batches"],
            ["buffer", "row_seq_lens"],
            ["buffer", "output"],
            ["buffer", "lse"],
            ["buffer", "sinks"],
            ["parameter", "num_heads"],
            ["parameter", "qk_dim"],
            ["parameter", "value_dim"],
            ["parameter", "kv_stride"],
            ["parameter", "page_size"],
            ["parameter", "page_table_width"],
            ["parameter", "sparse_width"],
            ["parameter", "use_sparse"],
            ["parameter", "softmax_scale"],
            ["parameter", "bmm2_scale"],
            ["parameter", "enable_sink"],
            ["parameter", "write_lse"],
            ["parameter", "num_rows"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_fp8_tail",
    },
    "cake_trtllm_mla_blackwell_family_c461bee2a4bb9454ff11": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_c461bee2a4bb9454ff11_binding.cu",
        "dependencies": [
            "cake_trtllm_mla_blackwell_03b4acdad2f17d77e6a8",
            "cake_trtllm_mla_blackwell_4de3d4bf2f8c5467bf29",
        ],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "output"],
            ["buffer", "lse"],
            ["buffer", "seq_lens"],
            ["buffer", "work_batch_indices"],
            ["buffer", "page_table"],
            ["workspace", "partial_output"],
            ["workspace", "partial_stats"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_split"],
            ["parameter", "num_queries"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "num_reduce_ctas"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_fp8_page64_pdl",
    },
    "cake_trtllm_mla_blackwell_family_cf9531ca88d55eb2813d": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_cf9531ca88d55eb2813d_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_79e49e216298d59fce33"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "output"],
            ["buffer", "completion"],
            ["workspace", "partial_output"],
            ["workspace", "partial_stats"],
            ["parameter", "batch"],
            ["parameter", "q_len"],
            ["parameter", "page_table_stride"],
            ["parameter", "max_num_ctas_q"],
            ["parameter", "max_num_ctas_kv"],
            ["parameter", "bmm1_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_fp8_p32_qk_l2",
    },
    "cake_trtllm_mla_blackwell_family_df126547928223b70b63": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_df126547928223b70b63_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_79e49e216298d59fce33"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "page_table"],
            ["buffer", "seq_lens"],
            ["buffer", "output"],
            ["buffer", "completion"],
            ["workspace", "partial_output"],
            ["workspace", "partial_stats"],
            ["parameter", "batch"],
            ["parameter", "q_len"],
            ["parameter", "page_table_stride"],
            ["parameter", "max_num_ctas_q"],
            ["parameter", "max_num_ctas_kv"],
            ["parameter", "bmm1_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_fp8_p32_qk_l2",
    },
    "cake_trtllm_mla_blackwell_family_df17a573e48e469242d2": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_df17a573e48e469242d2_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_74250d91903d28576cba"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_clc",
    },
    "cake_trtllm_mla_blackwell_family_e958fd69e74d907885f3": {
        "arch": "sm_103a",
        "binding": "cake_trtllm_mla_blackwell/sm_103a/cake_trtllm_mla_blackwell_family_e958fd69e74d907885f3_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_71fdced8e63ff5ad0ba4"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_clc",
    },
    "cake_trtllm_mla_blackwell_family_f0c36de00f7e55ccf2c2": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_f0c36de00f7e55ccf2c2_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_d7731e89cd4b1ecceeb0"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "q_rows"],
            ["buffer", "kv_pages"],
            ["buffer", "output"],
            ["buffer", "seq_lens"],
            ["buffer", "page_table"],
            ["buffer", "sinks"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "total_work_items"],
            ["parameter", "value_split_count"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "enable_sink"],
            ["parameter", "grid_x"],
            ["parameter", "grid_z"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_bf16_clc_exact007790",
    },
    "cake_trtllm_mla_blackwell_family_f662abe7edbd911e086f": {
        "arch": "sm_100a",
        "binding": "cake_trtllm_mla_blackwell/sm_100a/cake_trtllm_mla_blackwell_family_f662abe7edbd911e086f_binding.cu",
        "dependencies": ["cake_trtllm_mla_blackwell_d6056fe83ccbe21f79b3"],
        "compile_flags": ["-std=c++17"],
        "ffi_entry": "run",
        "route_entry": "route_of",
        "arg_plan": [
            ["buffer", "query"],
            ["buffer", "kv"],
            ["buffer", "fp8_lut"],
            ["buffer", "page_table"],
            ["buffer", "sparse_indices"],
            ["buffer", "row_batches"],
            ["buffer", "row_seq_lens"],
            ["buffer", "output"],
            ["buffer", "lse"],
            ["buffer", "sinks"],
            ["parameter", "num_heads"],
            ["parameter", "qk_dim"],
            ["parameter", "value_dim"],
            ["parameter", "kv_stride"],
            ["parameter", "page_size"],
            ["parameter", "page_table_width"],
            ["parameter", "sparse_width"],
            ["parameter", "use_sparse"],
            ["parameter", "softmax_scale"],
            ["parameter", "bmm2_scale"],
            ["parameter", "enable_sink"],
            ["parameter", "write_lse"],
            ["parameter", "num_rows"],
            ["stream", "cuda_stream_ptr"],
        ],
        "domain": "mla_fp8_tail",
    },
}
ROUTES: dict[str, str] = {
    "mla_bf16_clc__sm_100a": "cake_trtllm_mla_blackwell_family_df17a573e48e469242d2",
    "mla_bf16_clc__sm_103a": "cake_trtllm_mla_blackwell_family_e958fd69e74d907885f3",
    "mla_bf16_clc_exact007790__sm_100a": "cake_trtllm_mla_blackwell_family_f0c36de00f7e55ccf2c2",
    "mla_bf16_clc_exact007790__sm_103a": "cake_trtllm_mla_blackwell_family_35a4a08ccde422e44143",
    "mla_bf16_clc_exact007826__sm_100a": "cake_trtllm_mla_blackwell_family_a77fd5c07db49c791e35",
    "mla_bf16_clc_exact007826__sm_103a": "cake_trtllm_mla_blackwell_family_9eb0e53aa4b05afa531b",
    "mla_bf16_native_split8_pdl__sm_100a": "cake_trtllm_mla_blackwell_family_58aaa55b459e12fb35b8",
    "mla_bf16_native_split8_pdl__sm_103a": "cake_trtllm_mla_blackwell_family_3fa9fe12db6254809466",
    "mla_bf16_tail__sm_100a": "cake_trtllm_mla_blackwell_family_69765950411de111b0b4",
    "mla_bf16_tail__sm_103a": "cake_trtllm_mla_blackwell_family_6e7d2a25e57e51cfa70f",
    "mla_bf16_vhalf__sm_100a": "cake_trtllm_mla_blackwell_family_964aa840394fcd0656c0",
    "mla_bf16_vhalf__sm_103a": "cake_trtllm_mla_blackwell_family_3542f15da2f1ae0fc00c",
    "mla_bf16_vquarter__sm_100a": "cake_trtllm_mla_blackwell_family_5b3d4b45f82fac17edbd",
    "mla_bf16_vquarter__sm_103a": "cake_trtllm_mla_blackwell_family_73c2aeebeee855cabb5c",
    "mla_fp8_p32_qk_l2__sm_100a": "cake_trtllm_mla_blackwell_family_cf9531ca88d55eb2813d",
    "mla_fp8_p32_qk_l2__sm_103a": "cake_trtllm_mla_blackwell_family_df126547928223b70b63",
    "mla_fp8_page64_pdl__sm_100a": "cake_trtllm_mla_blackwell_family_c461bee2a4bb9454ff11",
    "mla_fp8_page64_pdl__sm_103a": "cake_trtllm_mla_blackwell_family_5a587bd20f8138cb26c9",
    "mla_fp8_tail__sm_100a": "cake_trtllm_mla_blackwell_family_f662abe7edbd911e086f",
    "mla_fp8_tail__sm_103a": "cake_trtllm_mla_blackwell_family_c325c9f98ac456facec0",
}

_ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
_ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
PACKAGE_DIR = "cake_trtllm_mla_blackwell"


# --- JIT loader -------------------------------------------------------------------


def _csrc_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / PACKAGE_DIR
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "csrc" / PACKAGE_DIR


def _source_path(relative: str) -> Path:
    # Registry paths are relative to the package directory.
    path = _csrc_dir() / Path(relative).relative_to(PACKAGE_DIR)
    if not path.is_file():
        raise FileNotFoundError(
            f"generated TRT-LLM MLA Blackwell source is missing: {path}"
        )
    return path


@functools.cache
def gen_cake_trtllm_mla_blackwell_family(name: str) -> JitSpec:
    """JIT spec of one compiled family: its binding plus every device kernel it links."""
    try:
        family = FAMILIES[name]
    except KeyError as exc:
        raise ValueError(
            f"TRT-LLM MLA Blackwell has no generated family {name!r}"
        ) from exc
    arch = family["arch"]
    sources = [_source_path(family["binding"])]
    flags: list[str] = [*_ARCH_NVCC_FLAGS[arch], *family["compile_flags"]]
    for dependency in family["dependencies"]:
        module = MODULES[dependency]
        sources.append(_source_path(module["device"]))
        for flag in module["compile_flags"]:
            if flag not in flags:
                flags.append(flag)
    csrc_dir = _csrc_dir()
    return gen_jit_spec(
        name=f"cake_trtllm_mla_blackwell_{name}",
        sources=sources,
        extra_cuda_cflags=flags,
        # The generated contract owns the fast-math decision.
        use_fast_math=False,
        extra_include_paths=[csrc_dir, csrc_dir.parent],
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def get_cake_trtllm_mla_blackwell_family(name: str):
    return gen_cake_trtllm_mla_blackwell_family(name).build_and_load()


def route_key(domain: str, arch: str) -> str:
    return f"{domain}__{arch}"


@functools.cache
def family_for(domain: str, arch: str) -> str:
    try:
        return ROUTES[route_key(domain, arch)]
    except KeyError as exc:
        raise ValueError(
            f"TRT-LLM MLA Blackwell has no generated family for domain {domain!r} on {arch}"
        ) from exc


# --- Device facts (cached once per device index) ------------------------------------


@functools.cache
def _device_facts(index: int) -> tuple[str, int]:
    capability = tuple(torch.cuda.get_device_capability(index))
    arch = _ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        raise RuntimeError(
            "TRT-LLM MLA Blackwell requires compute capability 10.0 or 10.3, "
            f"got {capability[0]}.{capability[1]}"
        )
    return arch, int(torch.cuda.get_device_properties(index).multi_processor_count)


# --- Semantic dispatcher ----------------------------------------------------------

_ALIGNED_HEADS = 128
_ALIGNED_QK_DIM = 576
_ALIGNED_VALUE_DIM = 512
_ALIGNED_DIMS = (128, 512, 64)
_CLUSTER_SIZE = 2
_KV_TILE = 128
_MAX_GENERIC_TOKENS = 4096
_MAX_SPLITS = 256
_NATIVE_SPLIT = 8
_P32_KV_CTAS = 2
_P32_HEAD_GROUPS = 32
# Split-KV partials, KV-length reads and the row plans of recent request identities.
_WORKSPACE_CACHE_CAPACITY = 32

DOMAIN_BF16_NATIVE_SPLIT8 = "mla_bf16_native_split8_pdl"
DOMAIN_BF16_VQUARTER = "mla_bf16_vquarter"
DOMAIN_BF16_VHALF = "mla_bf16_vhalf"
DOMAIN_BF16_CLC = "mla_bf16_clc"
DOMAIN_BF16_CLC_EXACT007790 = "mla_bf16_clc_exact007790"
DOMAIN_BF16_CLC_EXACT007826 = "mla_bf16_clc_exact007826"
_CLC_DOMAINS = (
    DOMAIN_BF16_CLC,
    DOMAIN_BF16_CLC_EXACT007790,
    DOMAIN_BF16_CLC_EXACT007826,
)
_ALIGNED_BF16_DOMAINS = (
    DOMAIN_BF16_NATIVE_SPLIT8,
    DOMAIN_BF16_VQUARTER,
    DOMAIN_BF16_VHALF,
    *_CLC_DOMAINS,
)
DOMAIN_BF16_TAIL = "mla_bf16_tail"
DOMAIN_FP8_P32 = "mla_fp8_p32_qk_l2"
DOMAIN_FP8_TAIL = "mla_fp8_tail"
DOMAIN_FP8_PAGE64 = "mla_fp8_page64_pdl"
# Exact-geometry CLC programs: (batch, q_len, max KV length, page-table width) each bakes in.
EXACT_CLC_GEOMETRY = {
    DOMAIN_BF16_CLC_EXACT007790: (64, 2, 8192, 256),
    DOMAIN_BF16_CLC_EXACT007826: (512, 16, 1024, 32),
}

_WORKSPACE_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_WORKSPACE_CACHE_LOCK = threading.Lock()

# (qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, num_heads) tuples with generated programs.
DENSE_DIMENSION_TUPLES = frozenset(
    {
        (128, 512, 64, 128),
        (128, 512, 64, 64),
        (64, 256, 64, 32),
        (512, 512, 64, 128),
    }
)
TOPK_DIMENSION_TUPLES = frozenset(
    {
        (128, 512, 64, 128),
        (128, 512, 64, 64),
        (192, 512, 64, 128),
        (192, 512, 64, 64),
    }
)


def supports_dimension_tuple(
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_heads: int,
    sparse_mla_top_k: int = 0,
) -> bool:
    """Whether this backend has a generated program for the dimension tuple.

    ``backend="cake"`` dispatch uses this to leave other MLA families (for example the
    Kimi-K3 FP8 paged-cache route, ``flashinfer.mla.cake_kimi_k3_mla``) to their own kernels.
    """
    key = (
        int(qk_nope_head_dim),
        int(kv_lora_rank),
        int(qk_rope_head_dim),
        int(num_heads),
    )
    if int(sparse_mla_top_k) > 0:
        return key in TOPK_DIMENSION_TUPLES
    return key in DENSE_DIMENSION_TUPLES


def _pick_num_split(work_items: int, tile_counts: list[int], num_sms: int) -> int:
    """Split-KV count of the aligned BF16 producer for the given per-row KV tiles."""
    live = [count for count in tile_counts if count > 0]
    if not live:
        return 1
    min_tiles = min(live)
    if min_tiles <= 2 or work_items >= max(1, num_sms // _CLUSTER_SIZE):
        return 1
    target = max(1, (num_sms // _CLUSTER_SIZE) // max(1, work_items))
    split = min(target, min_tiles, 16, _MAX_SPLITS)
    while split > 1:
        if all((split - 1) * ((count + split - 1) // split) < count for count in live):
            break
        split -= 1
    return split


def exact_route_candidate(
    *,
    dtype: torch.dtype,
    num_heads: int,
    page_size: int,
    topk: int,
    ragged_query: bool,
    enable_sink: bool,
    skip_softmax: bool,
    batch_size: int,
    q_len: int,
    table_width: int,
) -> bool:
    """Whether the request KV lengths decide the domain.

    Only these host-known geometries can select the native split-KV sequence,
    an exact-geometry CLC program or the FP8 P32 producer; ``seq_lens`` is read
    to the host once per input identity for them and never otherwise.
    """
    if topk != 0 or ragged_query or skip_softmax or num_heads != _ALIGNED_HEADS:
        return False
    if dtype == torch.bfloat16:
        if batch_size == 1 and (
            (q_len == 4 and page_size == 32 and not enable_sink)
            or (q_len == 1 and page_size == 64 and enable_sink)
        ):
            return True
        return page_size == 32 and any(
            (batch_size, q_len, table_width) == (batch, length, width)
            for batch, length, _max_kv, width in EXACT_CLC_GEOMETRY.values()
        )
    return (
        dtype == torch.float8_e4m3fn
        and batch_size == 1
        and q_len == 2
        and page_size == 32
    )


def select_domain(
    *,
    dtype: torch.dtype,
    num_heads: int,
    qk_dim: int,
    value_dim: int,
    qk_nope_head_dim: int,
    page_size: int,
    topk: int,
    uses_shared_paged_kv_idx: bool,
    ragged_query: bool,
    enable_sink: bool,
    skip_softmax: bool,
    wants_lse: bool,
    bmm2_scale: float,
    total_q: int,
    max_seq_len: int,
    batch_size: int,
    q_len: int,
    table_width: int,
    kv_lens: Optional[tuple[int, ...]],
    num_sms: int,
) -> str:
    """Pick the generated domain in the production dispatcher's order.

    ``kv_lens`` is required when ``exact_route_candidate`` holds for the geometry
    (the host read it once for this input identity); ``None`` otherwise, or under
    stream capture, where the runtime-shape domains serve the request.
    """
    aligned = (
        num_heads == _ALIGNED_HEADS
        and qk_dim == _ALIGNED_QK_DIM
        and value_dim == _ALIGNED_VALUE_DIM
        and topk == 0
        and uses_shared_paged_kv_idx
        and not wants_lse
    )
    dims_match = (qk_nope_head_dim, value_dim, qk_dim - value_dim) == _ALIGNED_DIMS
    if dtype == torch.bfloat16 and aligned:
        if kv_lens is not None and dims_match and not ragged_query and not skip_softmax:
            native = (
                batch_size == 1
                and kv_lens == (1024,)
                and (
                    (page_size == 32 and not enable_sink and q_len == 4)
                    or (page_size == 64 and enable_sink and q_len == 1)
                )
            )
            if native:
                rows = [
                    kv - q_len + query + 1 for kv in kv_lens for query in range(q_len)
                ]
                tiles = [(length + _KV_TILE - 1) // _KV_TILE for length in rows]
                if _pick_num_split(total_q, tiles, num_sms) == _NATIVE_SPLIT:
                    return DOMAIN_BF16_NATIVE_SPLIT8
        resident = max(1, num_sms // _CLUSTER_SIZE)
        if enable_sink or total_q * 4 <= resident:
            return DOMAIN_BF16_VQUARTER
        if total_q * 2 <= resident:
            return DOMAIN_BF16_VHALF
        if (
            page_size == 32
            and not ragged_query
            and not skip_softmax
            and bmm2_scale == 1.0
        ):
            if kv_lens is not None:
                max_kv = max(kv_lens)
                for domain, (
                    batch,
                    length,
                    max_kv_len,
                    width,
                ) in EXACT_CLC_GEOMETRY.items():
                    if (batch_size, q_len, table_width) == (batch, length, width) and (
                        max_kv == max_kv_len and kv_lens[-1] == max_kv
                    ):
                        return domain
            return DOMAIN_BF16_CLC
        raise ValueError(
            "TRT-LLM MLA Blackwell has no qualified aligned BF16 domain for this configuration"
        )
    if (
        dtype == torch.float8_e4m3fn
        and aligned
        and kv_lens is not None
        and dims_match
        and not ragged_query
        and page_size == 32
        and not enable_sink
        and not skip_softmax
        and q_len == 2
        and kv_lens == (1024,)
        and all(
            kv - q_len + query + 1 > 256 for kv in kv_lens for query in range(q_len)
        )
    ):
        return DOMAIN_FP8_P32
    if (
        dtype == torch.float8_e4m3fn
        and aligned
        and page_size == 64
        and not ragged_query
        and not enable_sink
        and not skip_softmax
    ):
        return DOMAIN_FP8_PAGE64
    if max_seq_len > _MAX_GENERIC_TOKENS:
        raise ValueError(
            "TRT-LLM MLA Blackwell generic tail supports max_seq_len <= "
            f"{_MAX_GENERIC_TOKENS}"
        )
    return DOMAIN_BF16_TAIL if dtype == torch.bfloat16 else DOMAIN_FP8_TAIL


def plan_num_split(work_items: int, max_seq_len: int, num_sms: int) -> int:
    """Split-K factor of the FP8 page-64 producer from host-known sizes."""
    max_tiles = max(1, (max_seq_len + _KV_TILE - 1) // _KV_TILE)
    cap = max(1, (max_tiles + 1) // 2)
    target = max(1, num_sms // max(1, work_items * _CLUSTER_SIZE))
    return max(2, min(cap, target, max_tiles))


def _workspace_peek(key: tuple[Any, ...]) -> Optional[dict[str, Any]]:
    with _WORKSPACE_CACHE_LOCK:
        state = _WORKSPACE_CACHE.get(key)
        if state is not None:
            _WORKSPACE_CACHE.move_to_end(key)
        return state


def _workspace(key: tuple[Any, ...], build) -> dict[str, Any]:
    with _WORKSPACE_CACHE_LOCK:
        state = _WORKSPACE_CACHE.get(key)
        if state is not None:
            _WORKSPACE_CACHE.move_to_end(key)
            return state
    state = build()
    with _WORKSPACE_CACHE_LOCK:
        existing = _WORKSPACE_CACHE.get(key)
        if existing is not None:
            _WORKSPACE_CACHE.move_to_end(key)
            return existing
        _WORKSPACE_CACHE[key] = state
        while len(_WORKSPACE_CACHE) > _WORKSPACE_CACHE_CAPACITY:
            _WORKSPACE_CACHE.popitem(last=False)
    return state


@functools.cache
def _e4m3_decode_table_values() -> tuple[float, ...]:
    values = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exponent == 0:
            magnitude = mantissa * (2.0**-9)
        elif exponent == 0xF and mantissa == 0x7:
            magnitude = 0.0
        else:
            magnitude = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(sign * magnitude)
    return tuple(values)


def _fp8_lut(device: torch.device) -> torch.Tensor:
    return _workspace(
        ("fp8_lut", device),
        lambda: {
            "lut": torch.tensor(
                _e4m3_decode_table_values(), dtype=torch.float32, device=device
            )
        },
    )["lut"]


def row_metadata(
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    *,
    batch_size: int,
    q_len: int,
    total_q: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per query row: its request index and its causal KV length, on the device."""
    device = seq_lens.device
    if cum_seq_lens_q is None:
        row_batches = torch.arange(batch_size, dtype=torch.int32, device=device)
        row_batches = row_batches.repeat_interleave(q_len)
        offsets = torch.arange(1 - q_len, 1, dtype=torch.int32, device=device).repeat(
            batch_size
        )
        row_seq_lens = seq_lens.repeat_interleave(q_len) + offsets
        return row_batches, row_seq_lens
    cum = cum_seq_lens_q.to(torch.int64)
    q_lens = cum[1:] - cum[:-1]
    row_batches = torch.arange(batch_size, dtype=torch.int64, device=device)
    row_batches = row_batches.repeat_interleave(q_lens, output_size=total_q)
    row_pos = (
        torch.arange(total_q, dtype=torch.int64, device=device) - cum[:-1][row_batches]
    )
    row_seq_lens = (
        seq_lens.to(torch.int64)[row_batches] - q_lens[row_batches] + row_pos + 1
    )
    return row_batches.to(torch.int32), row_seq_lens.to(torch.int32)


def longest_first_order(row_seq_lens: torch.Tensor) -> torch.Tensor:
    """Work ids in descending KV-tile order (ties keep row order), on the device."""
    tiles = (row_seq_lens.to(torch.int64) + (_KV_TILE - 1)) // _KV_TILE
    return torch.argsort(tiles, descending=True, stable=True).to(torch.int32)


def _aligned_page_table(
    table: torch.Tensor,
    row_batches: torch.Tensor,
    row_seq_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Row page table (page-32 halves) followed by the longest-first work permutation.

    The persistent CLC kernel reads ``page_table[num_rows * width + raw_id]`` for the row
    that raw work id ``raw_id`` processes; the other aligned kernels read only the first
    ``num_rows`` rows.  Padding entries hold -1.
    """
    if table.ndim == 3:
        table = table[:, 0]
    rows = table.to(torch.int32).index_select(0, row_batches.to(torch.int64))
    if page_size == 64:
        rows = torch.stack((rows * 2, rows * 2 + 1), dim=-1).reshape(rows.shape[0], -1)
    num_rows, width = int(rows.shape[0]), int(rows.shape[1])
    tail_rows = (num_rows + width - 1) // width
    tail = torch.full((tail_rows * width,), -1, dtype=torch.int32, device=table.device)
    tail[:num_rows] = longest_first_order(row_seq_lens)
    return torch.cat((rows, tail.view(tail_rows, width)), dim=0).contiguous()


def _tensor_identity(tensor: Optional[torch.Tensor]) -> Optional[tuple[Any, ...]]:
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _host_kv_lens(seq_lens: torch.Tensor) -> Optional[tuple[int, ...]]:
    """The request KV lengths, read once per ``seq_lens`` identity.

    Under stream capture an identity read before capture is reused; one not seen
    before cannot be read and returns ``None`` (the runtime-shape domains serve it).
    """
    key = ("kv_lens", seq_lens.device, _tensor_identity(seq_lens))
    state = _workspace_peek(key)
    if state is not None:
        return state["kv_lens"]
    if torch.cuda.is_current_stream_capturing():
        return None
    return _workspace(
        key, lambda: {"kv_lens": tuple(int(v) for v in seq_lens.tolist())}
    )["kv_lens"]


def _row_plan(
    domain: str,
    *,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    batch_size: int,
    q_len: int,
    total_q: int,
    page_size: int,
    topk: int,
) -> dict[str, torch.Tensor]:
    """The per-row metadata a domain's kernels read, built once per input identity.

    As in the production path, the rows are derived when a request's tensors
    (storage address, in-place version counter, shape and strides) first appear
    and reused while they are unchanged, so a warm call launches only the
    attention kernels.  Values rewritten through torch in-place ops bump the
    version counter and rebuild the plan; writes that bypass torch must pass
    fresh tensors.  Under stream capture a plan built before capture is reused;
    a new identity's metadata is recomputed inside the graph and not cached.
    """

    def build() -> dict[str, torch.Tensor]:
        row_batches, row_seq_lens = row_metadata(
            seq_lens,
            cum_seq_lens_q,
            batch_size=batch_size,
            q_len=q_len,
            total_q=total_q,
        )
        state = {"row_batches": row_batches, "row_seq_lens": row_seq_lens}
        if domain in _ALIGNED_BF16_DOMAINS:
            state["table"] = _aligned_page_table(
                block_tables, row_batches, row_seq_lens, page_size
            )
        elif domain == DOMAIN_FP8_PAGE64:
            state["last_kv"] = row_seq_lens - 1
        elif topk > 0:
            state["row_seq_lens"] = torch.full_like(row_seq_lens, topk)
        return state

    key = (
        "plan",
        domain,
        seq_lens.device,
        _tensor_identity(seq_lens),
        _tensor_identity(cum_seq_lens_q),
        _tensor_identity(block_tables),
        batch_size,
        q_len,
        total_q,
        page_size,
        topk,
    )
    cached = _workspace_peek(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return build()
    return _workspace(key, build)


def _check_tensor(
    tensor: torch.Tensor, *, name: str, dtype: torch.dtype, device: torch.device
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _normalize_scale(value: float | torch.Tensor, name: str) -> float:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"TRT-LLM MLA Blackwell requires scalar {name}")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a scalar float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_sinks(
    sinks, *, num_heads: int, device: torch.device
) -> Optional[torch.Tensor]:
    if sinks is None:
        return None
    if isinstance(sinks, (list, tuple)):
        if len(sinks) != 1:
            raise ValueError("TRT-LLM MLA Blackwell expects one sink tensor")
        sinks = sinks[0]
    _check_tensor(sinks, name="sinks", dtype=torch.float32, device=device)
    if tuple(sinks.shape) != (num_heads,):
        raise ValueError(f"sinks must have shape ({num_heads},)")
    return sinks


def trtllm_mla_blackwell_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    *,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    out: Optional[torch.Tensor],
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: Optional[list[torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
    enable_pdl: Optional[bool],
    uses_shared_paged_kv_idx: bool,
    lse: Optional[torch.Tensor],
    return_lse: bool,
    cum_seq_lens_q: Optional[torch.Tensor],
    max_q_len: Optional[int],
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor],
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    enable_dcp: bool,
    backend: Literal["cake"] = "cake",
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Dispatch the generated SM100a/SM103a MLA decode programs."""
    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    if not isinstance(query, torch.Tensor) or not query.is_cuda:
        raise ValueError("query must be a CUDA tensor")
    device = query.device
    arch, num_sms = _device_facts(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    if enable_dcp:
        raise ValueError("TRT-LLM MLA Blackwell does not support DCP")
    if multi_ctas_kv_counter_buffer is not None:
        raise ValueError(
            "TRT-LLM MLA Blackwell does not use a multi-CTA counter buffer"
        )
    if sparse_mla_top_k_lens is not None:
        raise ValueError("TRT-LLM MLA Blackwell does not accept sparse_mla_top_k_lens")
    if seq_lens is None:
        raise ValueError("seq_lens is required for TRT-LLM MLA Blackwell")
    if query.dtype not in {torch.bfloat16, torch.float8_e4m3fn}:
        raise TypeError("TRT-LLM MLA Blackwell query must use BF16 or FP8 E4M3")
    _check_tensor(query, name="query", dtype=query.dtype, device=device)
    _check_tensor(kv_cache, name="kv_cache", dtype=query.dtype, device=device)
    if query.ndim not in {3, 4}:
        raise ValueError(
            "query must be a fixed [B, Q, H, D] or compact [T, H, D] tensor"
        )
    if kv_cache.ndim not in {3, 4} or (kv_cache.ndim == 4 and kv_cache.shape[1] != 1):
        raise ValueError(
            "kv_cache must be a 3D or 4D paged tensor with a singleton head axis"
        )
    page_size = int(kv_cache.shape[-2])
    if page_size not in {32, 64}:
        raise ValueError("TRT-LLM MLA Blackwell requires page size 32 or 64")
    if int(query.shape[-1]) != int(kv_cache.shape[-1]):
        raise ValueError("query and kv_cache head dimensions differ")

    ragged_query = cum_seq_lens_q is not None
    if ragged_query != (query.ndim == 3):
        raise ValueError("compact query and cum_seq_lens_q must be provided together")
    if ragged_query:
        _check_tensor(
            cum_seq_lens_q, name="cum_seq_lens_q", dtype=torch.int32, device=device
        )
        if cum_seq_lens_q.ndim != 1 or cum_seq_lens_q.numel() < 2:
            raise ValueError("cum_seq_lens_q must have shape [batch_size + 1]")
        if max_q_len is None or max_q_len <= 0:
            raise ValueError(
                "max_q_len is required for compact TRT-LLM MLA Blackwell queries"
            )
        batch_size = int(cum_seq_lens_q.numel() - 1)
        q_len = int(max_q_len)
        total_q = int(query.shape[0])
    else:
        batch_size = int(query.shape[0])
        q_len = int(query.shape[1])
        total_q = batch_size * q_len
    if batch_size <= 0 or q_len <= 0 or total_q <= 0:
        raise ValueError(
            "TRT-LLM MLA Blackwell requires nonempty batch and query dimensions"
        )
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    _check_tensor(seq_lens, name="seq_lens", dtype=torch.int32, device=device)
    if tuple(seq_lens.shape) != (batch_size,):
        raise ValueError(f"seq_lens must have shape ({batch_size},)")
    _check_tensor(block_tables, name="block_tables", dtype=torch.int32, device=device)
    topk = int(sparse_mla_top_k)
    if topk > 0:
        expected = (total_q, topk) if ragged_query else (batch_size, q_len, topk)
        if tuple(block_tables.shape) != expected:
            raise ValueError(f"sparse block_tables must have shape {expected}")
    else:
        expected_ndim = 2 if uses_shared_paged_kv_idx else 3
        if block_tables.ndim != expected_ndim or block_tables.shape[0] != batch_size:
            raise ValueError(
                "dense block_tables must match the batch and shared-index layout"
            )

    num_heads = int(query.shape[-2])
    qk_dim = int(query.shape[-1])
    value_dim = int(kv_lora_rank)
    if not supports_dimension_tuple(
        qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, num_heads, topk
    ):
        raise ValueError("unsupported TRT-LLM MLA Blackwell dimension tuple")
    if qk_dim != kv_lora_rank + qk_rope_head_dim:
        raise ValueError("query width must equal kv_lora_rank + qk_rope_head_dim")
    sink = _normalize_sinks(sinks, num_heads=num_heads, device=device)
    bmm1 = _normalize_scale(bmm1_scale, "bmm1_scale")
    bmm2 = _normalize_scale(bmm2_scale, "bmm2_scale")
    skip_softmax = skip_softmax_threshold_scale_factor is not None
    if skip_softmax:
        threshold = float(skip_softmax_threshold_scale_factor)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("skip_softmax_threshold_scale_factor must be positive")
    expected_out_shape = (*query.shape[:-1], value_dim)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=device)
    else:
        _check_tensor(out, name="out", dtype=torch.bfloat16, device=device)
        if tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}")
    if lse is not None:
        _check_tensor(lse, name="lse", dtype=torch.float32, device=device)
        if tuple(lse.shape) not in {(total_q, num_heads), tuple(query.shape[:-1])}:
            raise ValueError("lse shape must match flattened or physical query rows")
    if return_lse and lse is None:
        lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)

    table_width = int(block_tables.shape[-1])
    candidate = exact_route_candidate(
        dtype=query.dtype,
        num_heads=num_heads,
        page_size=page_size,
        topk=topk,
        ragged_query=ragged_query,
        enable_sink=sink is not None,
        skip_softmax=skip_softmax,
        batch_size=batch_size,
        q_len=q_len,
        table_width=table_width,
    )
    domain = select_domain(
        dtype=query.dtype,
        num_heads=num_heads,
        qk_dim=qk_dim,
        value_dim=value_dim,
        qk_nope_head_dim=int(qk_nope_head_dim),
        page_size=page_size,
        topk=topk,
        uses_shared_paged_kv_idx=bool(uses_shared_paged_kv_idx),
        ragged_query=ragged_query,
        enable_sink=sink is not None,
        skip_softmax=skip_softmax,
        wants_lse=lse is not None,
        bmm2_scale=bmm2,
        total_q=total_q,
        max_seq_len=int(max_seq_len),
        batch_size=batch_size,
        q_len=q_len,
        table_width=table_width,
        kv_lens=_host_kv_lens(seq_lens) if candidate else None,
        num_sms=num_sms,
    )
    plan = _row_plan(
        domain,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        block_tables=block_tables,
        batch_size=batch_size,
        q_len=q_len,
        total_q=total_q,
        page_size=page_size,
        topk=topk,
    )
    row_batches, row_seq_lens = plan["row_batches"], plan["row_seq_lens"]
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    run = get_cake_trtllm_mla_blackwell_family(family_for(domain, arch)).run
    softmax_scale_log2 = bmm1 * log2e

    if domain == DOMAIN_BF16_NATIVE_SPLIT8:
        table = plan["table"]
        resident = max(1, num_sms // _CLUSTER_SIZE)
        work = total_q * _NATIVE_SPLIT
        grid_x = min(work, resident) * _CLUSTER_SIZE
        state = _workspace(
            ("native_split8", device, total_q),
            lambda: {
                "partial_output": torch.empty(
                    (total_q, _ALIGNED_HEADS, _NATIVE_SPLIT, _ALIGNED_VALUE_DIM),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "partial_lse": torch.empty(
                    (total_q, _ALIGNED_HEADS, _NATIVE_SPLIT),
                    dtype=torch.float32,
                    device=device,
                ),
            },
        )
        run(
            query.reshape(-1, _ALIGNED_QK_DIM),
            kv_cache.reshape(-1, 32, _ALIGNED_QK_DIM),
            out.reshape(-1, _ALIGNED_VALUE_DIM),
            row_seq_lens,
            table,
            sink if sink is not None else _fp8_lut(device)[:num_heads],
            state["partial_output"],
            state["partial_lse"],
            softmax_scale_log2,
            1.0,
            bmm2,
            _NATIVE_SPLIT,
            work,
            int(table.shape[1]),
            int(sink is not None),
            grid_x,
            total_q,
            stream,
        )
    elif domain in _ALIGNED_BF16_DOMAINS:
        table = plan["table"]
        resident = max(1, num_sms // _CLUSTER_SIZE)
        if domain in _CLC_DOMAINS:
            split, work, grid_x, grid_z = q_len, total_q, 2 * q_len, batch_size
        else:
            split = 4 if domain == DOMAIN_BF16_VQUARTER else 2
            work = total_q * split
            grid_x, grid_z = min(work, resident) * 2, 1
        sinks_arg = sink if sink is not None else _fp8_lut(device)[:num_heads]
        run(
            query.reshape(-1, _ALIGNED_QK_DIM),
            kv_cache.reshape(-1, 32, _ALIGNED_QK_DIM),
            out.reshape(-1, _ALIGNED_VALUE_DIM),
            row_seq_lens,
            table,
            sinks_arg,
            softmax_scale_log2,
            bmm2,
            work,
            split,
            int(table.shape[1]),
            int(sink is not None),
            grid_x,
            grid_z,
            stream,
        )
    elif domain == DOMAIN_FP8_P32:
        reduction_groups = batch_size * _P32_HEAD_GROUPS * q_len
        state = _workspace(
            ("fp8_p32", device, batch_size, q_len),
            lambda: {
                # The producer's same-kernel reduction leaves every counter at zero.
                "completion": torch.zeros(
                    reduction_groups, dtype=torch.uint32, device=device
                ),
                "partial_output": torch.empty(
                    (reduction_groups, _P32_KV_CTAS, 16, 128),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "partial_stats": torch.empty(
                    (reduction_groups, _P32_KV_CTAS, 16, 2),
                    dtype=torch.float32,
                    device=device,
                ),
            },
        )
        run(
            query.reshape(-1, _ALIGNED_QK_DIM).view(torch.uint8),
            kv_cache.reshape(-1, 32, _ALIGNED_QK_DIM).view(torch.uint8),
            block_tables.reshape(-1),
            seq_lens,
            out.reshape(-1),
            state["completion"],
            state["partial_output"],
            state["partial_stats"],
            batch_size,
            q_len,
            table_width,
            q_len,
            _P32_KV_CTAS,
            softmax_scale_log2,
            bmm2,
            stream,
        )
    elif domain == DOMAIN_FP8_PAGE64:
        work_items = total_q
        num_split = plan_num_split(work_items, int(max_seq_len), num_sms)
        reduce_ctas = min(64, max(1, (num_sms * 2) // max(1, work_items * 2)))
        state = _workspace(
            ("fp8_page64", device, work_items, num_split),
            lambda: {
                "partial_output": torch.empty(
                    (work_items, num_split, _ALIGNED_HEADS, _ALIGNED_VALUE_DIM),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                "partial_stats": torch.empty(
                    (work_items, num_split, _ALIGNED_HEADS, 2),
                    dtype=torch.float32,
                    device=device,
                ),
                "lse": torch.empty(
                    (work_items, _ALIGNED_HEADS), dtype=torch.float32, device=device
                ),
            },
        )
        run(
            query.reshape(-1, _ALIGNED_QK_DIM).view(torch.uint8),
            kv_cache.reshape(-1, _ALIGNED_QK_DIM).view(torch.uint8),
            out.reshape(-1, _ALIGNED_VALUE_DIM),
            state["lse"],
            plan["last_kv"],
            row_batches,
            block_tables.reshape(-1),
            state["partial_output"],
            state["partial_stats"],
            softmax_scale_log2,
            bmm2,
            num_split,
            work_items,
            int(block_tables.shape[-1]),
            reduce_ctas,
            stream,
        )
    else:
        kv_stride = int(kv_cache.shape[-1])
        q_rows = query.reshape(-1, qk_dim)
        kv_rows = kv_cache.reshape(-1, kv_stride)
        if query.dtype == torch.float8_e4m3fn:
            q_rows = q_rows.view(torch.uint8)
            kv_rows = kv_rows.view(torch.uint8)
        source_table = (
            block_tables[:, 0] if block_tables.ndim == 3 and topk == 0 else block_tables
        )
        if topk > 0:
            sparse_indices = block_tables.reshape(total_q, topk)
            source_table = sparse_indices
        else:
            sparse_indices = row_batches
        lse_rows = lse.reshape(-1) if lse is not None else _fp8_lut(device)[:1]
        run(
            q_rows,
            kv_rows,
            _fp8_lut(device),
            source_table,
            sparse_indices,
            row_batches,
            row_seq_lens,
            out.reshape(-1, value_dim),
            lse_rows,
            sink if sink is not None else _fp8_lut(device)[:num_heads],
            num_heads,
            qk_dim,
            value_dim,
            kv_stride,
            page_size,
            int(source_table.shape[-1]),
            topk,
            int(topk > 0),
            bmm1,
            bmm2,
            int(sink is not None),
            int(lse is not None),
            total_q,
            stream,
        )
    if return_lse:
        assert lse is not None
        return out, lse
    return out


__all__ = [
    "FAMILIES",
    "MODULES",
    "ROUTES",
    "exact_route_candidate",
    "gen_cake_trtllm_mla_blackwell_family",
    "plan_num_split",
    "row_metadata",
    "select_domain",
    "supports_dimension_tuple",
    "trtllm_mla_blackwell_decode",
]
