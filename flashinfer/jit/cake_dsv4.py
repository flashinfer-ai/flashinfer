"""JIT loader for the Cake-generated DeepSeek V4 sparse MLA kernels.

Every variant is one device kernel plus its TVM-FFI launcher under
``csrc/cake_dsv4/<arch>/`` (``common/`` when both architectures compile the same
source). ``_ARCH_REGISTRATIONS`` is the per-architecture build and argument
contract: the host (``flashinfer.mla.cake_dsv4``) binds every ``arg_plan`` entry
by name and passes the launch grid, so a regenerated registration only needs
names from the host vocabulary.
"""

from __future__ import annotations

import functools
from pathlib import Path

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from .cpp_ext import get_cuda_version, is_cuda_version_at_least


_ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}

# Argument plans shared by the generated launchers: (kind, name) in binding order,
# followed by the launch grid. ``_SLAB`` marks the SM103 bindings that read their
# TMA descriptors from the caller-owned workspace slab.
_GRID = (("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z"))
_SLAB = (("workspace", "tma_descriptor_workspace"),)

_PLAN_BF16_H128_PERSISTENT_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_k"),
    ("tma_buffer", "tmap_compressed_k"),
    ("tma_buffer", "tmap_swa_v"),
    ("tma_buffer", "tmap_compressed_v"),
    ("buffer", "O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "num_query_tokens"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_k"),
    ("tma_buffer", "tmap_compressed_k"),
    ("tma_buffer", "tmap_swa_v"),
    ("tma_buffer", "tmap_compressed_v"),
    ("tma_buffer", "O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "num_query_tokens"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_BF16_H128_SWA_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("buffer", "O"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "num_head_tiles"),
    ("parameter", "has_sinks"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_BF16_H32_MERGE_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("buffer", "partition_arrivals"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "num_splits"),
    ("parameter", "num_head_tiles"),
    ("parameter", "has_sinks"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_BF16_H64_GUARD_TMA_O = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("tma_buffer", "O"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "batch_size"),
    ("parameter", "max_q_len"),
    ("parameter", "ragged_query"),
    ("parameter", "has_sinks"),
)

_PLAN_BF16_H64_PREFILL_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "num_query_tokens"),
    ("parameter", "has_sinks"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_BF16_H64_SPLIT_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "num_splits"),
    ("parameter", "has_sinks"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_BF16_SWA_DECODE_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("buffer", "O"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_FP8_H64_M64_SWA_K_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("tma_buffer", "tmap_swa_k"),
    ("buffer", "O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "num_query_tokens"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "seq_lens"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
)

_PLAN_FP8_LOWHEAD_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "num_query_tokens"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_FP8_PERSISTENT_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("tma_buffer", "tmap_o"),
    ("buffer", "O"),
    ("buffer", "partial_lse"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "num_heads"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "num_query_tokens"),
    ("parameter", "sparse_topk"),
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_H64_REDUCE = (
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("parameter", "num_heads"),
    ("parameter", "num_splits"),
)

_PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
    ("buffer", "swa_indices"),
    ("buffer", "compressed_indices"),
    ("buffer", "sparse_topk_lens"),
    ("buffer", "seq_lens"),
    ("buffer", "cum_seq_lens_q"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("parameter", "swa_index_stride"),
    ("parameter", "compressed_index_stride"),
    ("parameter", "sparse_topk_lens_offset"),
    ("parameter", "sparse_topk"),
    ("parameter", "ragged_query"),
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
    ("parameter", "has_sinks"),
    ("parameter", "num_query_tokens"),
)

_PLAN_SPLIT_REDUCE = (
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("parameter", "num_q_heads"),
    ("parameter", "num_split"),
)

# Populated by the generated-program integration from the resolved bundle; one
# record per variant and architecture.
_PLAN_NVFP4_DECODE = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_out"),
    ("buffer", "q_rows"),
    ("buffer", "main_cache"),
    ("buffer", "extra_cache"),
    ("buffer", "main_indices"),
    ("buffer", "extra_indices"),
    ("buffer", "main_lengths"),
    ("buffer", "extra_lengths"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("buffer", "lse_out"),
    ("parameter", "num_heads"),
    ("parameter", "num_head_tiles"),
    ("parameter", "num_splits"),
    ("parameter", "num_main_tiles"),
    ("parameter", "tiles_per_split"),
    ("parameter", "total_tiles"),
    ("parameter", "main_width"),
    ("parameter", "extra_width"),
    ("parameter", "main_index_stride"),
    ("parameter", "extra_index_stride"),
    ("parameter", "has_main_lengths"),
    ("parameter", "has_extra_lengths"),
    ("parameter", "main_page_shift"),
    ("parameter", "extra_page_shift"),
    ("parameter", "main_page_stride"),
    ("parameter", "extra_page_stride"),
    ("parameter", "has_sinks"),
    ("parameter", "lse_partial_scale"),
    ("parameter", "lse_scale"),
)

_PLAN_NVFP4_G4 = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_out"),
    ("tma_buffer", "tmap_g4d"),
    ("tma_buffer", "tmap_g4f"),
    ("tma_buffer", "tmap_g4dx"),
    ("tma_buffer", "tmap_g4fx"),
    ("buffer", "q_rows"),
    ("buffer", "main_cache"),
    ("buffer", "extra_cache"),
    ("buffer", "main_indices"),
    ("buffer", "extra_indices"),
    ("buffer", "main_lengths"),
    ("buffer", "extra_lengths"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("buffer", "lse_out"),
    ("parameter", "num_heads"),
    ("parameter", "num_head_tiles"),
    ("parameter", "num_splits"),
    ("parameter", "num_main_tiles"),
    ("parameter", "tiles_per_split"),
    ("parameter", "main_width"),
    ("parameter", "extra_width"),
    ("parameter", "main_index_stride"),
    ("parameter", "extra_index_stride"),
    ("parameter", "has_main_lengths"),
    ("parameter", "has_extra_lengths"),
    ("parameter", "main_page_shift"),
    ("parameter", "extra_page_shift"),
    ("parameter", "main_page_stride"),
    ("parameter", "extra_page_stride"),
    ("parameter", "has_sinks"),
    ("parameter", "lse_partial_scale"),
    ("parameter", "lse_scale"),
)

_PLAN_NVFP4_MERGE = (
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("buffer", "lse_out"),
    ("parameter", "num_heads"),
    ("parameter", "num_splits"),
    ("parameter", "heads_per_cta"),
    ("parameter", "lse_scale"),
)

_PLAN_NVFP4_TILE = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_out"),
    ("buffer", "q_rows"),
    ("buffer", "main_cache"),
    ("buffer", "extra_cache"),
    ("buffer", "main_indices"),
    ("buffer", "extra_indices"),
    ("buffer", "main_lengths"),
    ("buffer", "extra_lengths"),
    ("buffer", "sinks"),
    ("buffer", "bmm1_scale"),
    ("buffer", "bmm2_scale"),
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("buffer", "lse_out"),
    ("parameter", "num_heads"),
    ("parameter", "num_head_tiles"),
    ("parameter", "num_splits"),
    ("parameter", "num_main_tiles"),
    ("parameter", "main_width"),
    ("parameter", "extra_width"),
    ("parameter", "main_index_stride"),
    ("parameter", "extra_index_stride"),
    ("parameter", "has_main_lengths"),
    ("parameter", "has_extra_lengths"),
    ("parameter", "main_page_shift"),
    ("parameter", "extra_page_shift"),
    ("parameter", "main_page_stride"),
    ("parameter", "extra_page_stride"),
    ("parameter", "has_sinks"),
    ("parameter", "lse_partial_scale"),
    ("parameter", "lse_scale"),
)

_ARCH_REGISTRATIONS = {
    "sm_100a": {
        "variants": {
            "bf16_h128_prefill_v42": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "5a316b6230275e2de9bdd1fb6942fa1e2df455095204959abc055473b5ec5f82",
                "sources": [
                    "sm_100a/cake_dsv4_860a0ac41a77c1305873_kernel.cu",
                    "sm_100a/cake_dsv4_860a0ac41a77c1305873_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "6a99296cd5d6c0de80d3820b2d31776d197b9beb4b7cb0b592b848f5fd66494d",
                "sources": [
                    "sm_100a/cake_dsv4_8a33d6688c16c25820e9_kernel.cu",
                    "sm_100a/cake_dsv4_8a33d6688c16c25820e9_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "8b3a94fe8503fea96ba4cd20291bbb468c2de50e49015baf0ce68fa076730ba1",
                "sources": [
                    "sm_100a/cake_dsv4_e1c7ffd7ad7de34b2096_kernel.cu",
                    "sm_100a/cake_dsv4_e1c7ffd7ad7de34b2096_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "aeda09841fac016a4b05a94e6fedd25e37223cb96c786996bc82cc02ded2cb99",
                "sources": [
                    "sm_100a/cake_dsv4_7abc1b8f5f3d1d936f1a_kernel.cu",
                    "sm_100a/cake_dsv4_7abc1b8f5f3d1d936f1a_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "01fed2d55eeb0a186ef47e63614c15ea4234feb50b7a810a114f9e29106efb29",
                "sources": [
                    "sm_100a/cake_dsv4_71e0baa44c61b948d82c_kernel.cu",
                    "sm_100a/cake_dsv4_71e0baa44c61b948d82c_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "fc2edc102a92ac9b47d1ff994dc58a799067deff591ff1ebac8fb77e986db2c4",
                "sources": [
                    "sm_100a/cake_dsv4_3af1afa1871d27324303_kernel.cu",
                    "sm_100a/cake_dsv4_3af1afa1871d27324303_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "3ab6b7f0731df770732005f8d1bb4d5db16dacf7f01f8151dc9ace2a2f8d1c04",
                "sources": [
                    "sm_100a/cake_dsv4_fc73aade5a5f4f85a9e8_kernel.cu",
                    "sm_100a/cake_dsv4_fc73aade5a5f4f85a9e8_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "2ccbcccc20cb009f822bf93b5353a81212116bc4aceb0cd4276975a351b319cf",
                "sources": [
                    "sm_100a/cake_dsv4_1b4dfa2ab26fc784f640_kernel.cu",
                    "sm_100a/cake_dsv4_1b4dfa2ab26fc784f640_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "dc9b763a2a0f2197c6f1b00665d178bb6fa4a8a9c2309ec79eacff6b54861d93",
                "sources": [
                    "sm_100a/cake_dsv4_61008e64c0c3bd2e1a83_kernel.cu",
                    "sm_100a/cake_dsv4_61008e64c0c3bd2e1a83_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bbabb2d7088e38e96def36ccea32d82c3f7f907f310caaae66086f6dc38fefd3",
                "sources": [
                    "sm_100a/cake_dsv4_c9156a2e9f65117a0900_kernel.cu",
                    "sm_100a/cake_dsv4_c9156a2e9f65117a0900_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "079202d3f54068e646d22c7a96309644fa48e69b02c993c82648881db4d99ddc",
                "sources": [
                    "sm_100a/cake_dsv4_58df983ae8026a009dab_kernel.cu",
                    "sm_100a/cake_dsv4_58df983ae8026a009dab_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bf037e2abd291d1fec2e2744bbcc6abf5913813885457ef971a1ed28827e1fd0",
                "sources": [
                    "sm_100a/cake_dsv4_efb864796f11291dc52c_kernel.cu",
                    "sm_100a/cake_dsv4_efb864796f11291dc52c_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2835651b803b1e48bdea97a81e39851193af6679690b708e216c4708fa46d225",
                "sources": [
                    "sm_100a/cake_dsv4_cfa285ef8a408579ac99_kernel.cu",
                    "sm_100a/cake_dsv4_cfa285ef8a408579ac99_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6f79d07e8e964e23950fcd200a2338df83751a9faed8cebc0bffca61bb59d1be",
                "sources": [
                    "sm_100a/cake_dsv4_c85537d4ca6a34893cd1_kernel.cu",
                    "sm_100a/cake_dsv4_c85537d4ca6a34893cd1_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5890455023681306e8550ab87996577bab400dc0d8c05cf84239a5da9bbe04c1",
                "sources": [
                    "sm_100a/cake_dsv4_cc8f5e8152e47e5e566a_kernel.cu",
                    "sm_100a/cake_dsv4_cc8f5e8152e47e5e566a_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5a5fdc876904c6efc1cc40a9675f7aa1eb2e7b3c422d3409405c2d385077ffaa",
                "sources": [
                    "sm_100a/cake_dsv4_e3b65d62b4db145adce7_kernel.cu",
                    "sm_100a/cake_dsv4_e3b65d62b4db145adce7_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "99c036a5663cc25e85dfa86b1b3f7997f1901203bde0429c3a375c7a19864100",
                "sources": [
                    "sm_100a/cake_dsv4_2cebeb2f2243d69b2827_kernel.cu",
                    "sm_100a/cake_dsv4_2cebeb2f2243d69b2827_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "db6008ee35a7637151ec89c92e0ccab143ceb667dc09e48e459140887dd96395",
                "sources": [
                    "sm_100a/cake_dsv4_155cefce7e1900a952c3_kernel.cu",
                    "sm_100a/cake_dsv4_155cefce7e1900a952c3_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a98964240d626054db869dee6297d460b53d0131777e9c3f1b8ce86690da8d35",
                "sources": [
                    "sm_100a/cake_dsv4_ca071620f07188f5ea22_kernel.cu",
                    "sm_100a/cake_dsv4_ca071620f07188f5ea22_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "1f14b1c06f518922763964296e537e25904b5d359feb763936585b456539e5bf",
                "sources": [
                    "sm_100a/cake_dsv4_ee710a07143d70b73dc2_kernel.cu",
                    "sm_100a/cake_dsv4_ee710a07143d70b73dc2_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "177daf891c826b472d4dfe1ea629e2707bb5c9c2842cc0598f196f4e257c96cc",
                "sources": [
                    "sm_100a/cake_dsv4_aa5800406f680b6ad9fc_kernel.cu",
                    "sm_100a/cake_dsv4_aa5800406f680b6ad9fc_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "edc54d7de032e113d6777f120904e6cb5cc719ccdfa33bbc4a9d782b84c7309e",
                "sources": [
                    "sm_100a/cake_dsv4_d7c58d0212deccbcaf48_kernel.cu",
                    "sm_100a/cake_dsv4_d7c58d0212deccbcaf48_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "c7244948bb13cfcf30f60287826cfe947d1842209d84c7f45dbd0da1539fa79d",
                "sources": [
                    "sm_100a/cake_dsv4_f98ac384c6e4f7fb86e4_kernel.cu",
                    "sm_100a/cake_dsv4_f98ac384c6e4f7fb86e4_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "95cc9fb8932cc64983cd9473ea365daf3085b1a49519521b08fb76c7d3fadec6",
                "sources": [
                    "sm_100a/cake_dsv4_5d0c8e053e04ac001929_kernel.cu",
                    "sm_100a/cake_dsv4_5d0c8e053e04ac001929_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "0d18f1c06d7edfd3da572053e9104be81d1f1a4f6a1ad40f39984ad450806410",
                "sources": [
                    "sm_100a/cake_dsv4_9c5822db6c2739c63dc1_kernel.cu",
                    "sm_100a/cake_dsv4_9c5822db6c2739c63dc1_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "c094cf8490fece7bca46372f12131e8a5013c98612849c9a5828a4c23321b7c1",
                "sources": [
                    "sm_100a/cake_dsv4_820670a92a2bf9dfb2b0_kernel.cu",
                    "sm_100a/cake_dsv4_820670a92a2bf9dfb2b0_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2971d71708fd20d7a5abc26da66a0763e94d0ca5075207d2acb4d2448c433be7",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_cluster_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_cluster_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3a9d813a27e4dc4836dedb132610f59e4a52b1699b7772714522fd97afc36b1b",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_persistent_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_persistent_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_N16_OC1=1",
                ],
                "identity": "8fbbaa0fdaa5d09550cf79121e73b45196afa8f8c8a01572449288bf1a931b38",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_N32_OC1=1",
                ],
                "identity": "a29be0c65cdb6f9130d4a2d0ece7b40961836731e42aec1afd09142e1ad15af3",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N16_OC1=1",
                ],
                "identity": "ab25a78f1c580de8c4778b404c82284a53b926972d19a07febcdfbb06f955c5b",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N16_OC2=1",
                ],
                "identity": "8c5089f5987cda5dcbbc9651d2830bac06bf926471f240c4087d10a4fe55a12d",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N16_OC4=1",
                ],
                "identity": "338bff5b21dc09aa591aeb024c13a9b3baa3721d0f353e25f2c380b36483e5da",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N32_OC1=1",
                ],
                "identity": "297c8c2ab94d5198bc3786ecbad5bce3e6d1609f984f76dc67b64ea98bd6d406",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N32_OC2=1",
                ],
                "identity": "4aa11c2902cb6d3862ca7352fd4000fa69c2c59c0f2efd2122cd08b76059877d",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N32_OC4=1",
                ],
                "identity": "3df8091570452c070c9c8421b6b6854750f626279d6024f0c1a20619688ffe2d",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "18e16a5c2120d5e03b025dc48ea8b9eed10c99e6243a932c50019a5f41be0b8b",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_h64_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_OC2=1",
                ],
                "identity": "5d6440d249b046d919fd4ebd6b274fd91242d128aae57f78198d325a3d75a64f",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_h64_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_h64_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_tile_h64_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_OC4=1",
                ],
                "identity": "fdd5b98e14d5189edeb263a124d4d6cd4525bee3e0a8d1b232a694574bfdddc5",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_h64_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_h64_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_OC1=1",
                ],
                "identity": "0695d68791bba15e4c80a75fbea861e45f2103925d15d7030bd121524048bf15",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_OC2=1",
                ],
                "identity": "661c3433efcd91b4283b05e896daba4af28f38150446075ede8f30ace469736d",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_OC4=1",
                ],
                "identity": "e4d0208b25b691c5b9f36ba7a0c4f8e2af7e5bb96770a5fee3988a9e5125e674",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0f33036256bd7094ffde8e3b3050808b24c19aed965bbbc0da1b35e2fc1df3c9",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_merge_kernel.cu",
                    "common/cake_dsv4_nvfp4_merge_binding.cu",
                ],
            },
            "split_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "7cea46afc483f21f79017268008dda3281422fde211450ab973bb3b59a32696e",
                "sources": [
                    "sm_100a/cake_dsv4_2b24cab15b398c9b420a_kernel.cu",
                    "sm_100a/cake_dsv4_2b24cab15b398c9b420a_binding.cu",
                ],
            },
        },
    },
    "sm_103a": {
        "variants": {
            "bf16_h128_prefill_v42": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "80d7ac59aa962080cc2d62000ab5f8561a9e04ac36949a24fb0ca177af2102fb",
                "sources": [
                    "sm_103a/cake_dsv4_06cdf778054cc6c67dc5_kernel.cu",
                    "sm_103a/cake_dsv4_06cdf778054cc6c67dc5_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "9b3bb0f42ab31389b7ef02255ddad92416b75cce0b1cb4b7976d26991a8a3709",
                "sources": [
                    "sm_103a/cake_dsv4_a8b112bf2abd1a8ed848_kernel.cu",
                    "sm_103a/cake_dsv4_a8b112bf2abd1a8ed848_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "95ca59e0021b622eb3b961f7200f8eb76d479275e646013e672f9e2a7441d8bb",
                "sources": [
                    "sm_103a/cake_dsv4_8b60cfd54a78d36864f4_kernel.cu",
                    "sm_103a/cake_dsv4_8b60cfd54a78d36864f4_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6a3ef5704ad10c1ad5135f5d6d6a6622775ca013b3bb656d3bfcafc5e0689adc",
                "sources": [
                    "sm_103a/cake_dsv4_f918e4ac3c5b77c4514b_kernel.cu",
                    "sm_103a/cake_dsv4_f918e4ac3c5b77c4514b_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "04b97ac830d37ba297280a69668914de33b552314ca2c99d4b9d634c6dfe03d5",
                "sources": [
                    "sm_103a/cake_dsv4_8c2cf89e71be1ebbd233_kernel.cu",
                    "sm_103a/cake_dsv4_8c2cf89e71be1ebbd233_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "8f000d053f792f4e4537d7370c6ca6e52579ab1b4cca8d8b4fcfb3070f4fba19",
                "sources": [
                    "sm_103a/cake_dsv4_9c1ead828285aaa7f400_kernel.cu",
                    "sm_103a/cake_dsv4_9c1ead828285aaa7f400_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "bb0a0f01dbb6e1fe6a736e9f99cfe996ada9e0759cf32bfb40beee8fb23e116d",
                "sources": [
                    "sm_103a/cake_dsv4_6a34a835c2b39424c20c_kernel.cu",
                    "sm_103a/cake_dsv4_6a34a835c2b39424c20c_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "ddc7f4ed360e21ab8d8d1d2591a3028daeca234fae9c95f9fe848f4e96d0344c",
                "sources": [
                    "sm_103a/cake_dsv4_3b4ec80fd59f1a59eb0d_kernel.cu",
                    "sm_103a/cake_dsv4_3b4ec80fd59f1a59eb0d_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "de1914156f95e7c127f50257df70d17b0f7b9f0b3cb8697771327e2f7a5dc7c9",
                "sources": [
                    "sm_103a/cake_dsv4_a164ee5cbd7fc5b4d14d_kernel.cu",
                    "sm_103a/cake_dsv4_a164ee5cbd7fc5b4d14d_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7d5fe7a24d8596b5bfd3e87ad74ee7d29ae02c3e8ed95ddd512c81537425a5ea",
                "sources": [
                    "sm_103a/cake_dsv4_ebf2dfc077eb990f64fa_kernel.cu",
                    "sm_103a/cake_dsv4_ebf2dfc077eb990f64fa_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "62fa5a24bc71a7176e077b935144961fb1c9743a2b40030fa4e215430099505a",
                "sources": [
                    "sm_103a/cake_dsv4_584c7f1bfa07e4783b08_kernel.cu",
                    "sm_103a/cake_dsv4_584c7f1bfa07e4783b08_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "534205e2faf86f4ae23f9d02f68774c915960013f4de6eb0b013332956122be7",
                "sources": [
                    "sm_103a/cake_dsv4_e900f8c37ff1f0974a39_kernel.cu",
                    "sm_103a/cake_dsv4_e900f8c37ff1f0974a39_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "108a253711ffef47bd770d12b1085673aa9bd1b699899138ed57159bb51f25de",
                "sources": [
                    "sm_103a/cake_dsv4_a293c552f4c6f588106c_kernel.cu",
                    "sm_103a/cake_dsv4_a293c552f4c6f588106c_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "72671cb08b37e1db7844c856831d79c0a86e17f37a1cee8843fb60f241fe3083",
                "sources": [
                    "sm_103a/cake_dsv4_2ada791761e6057628f6_kernel.cu",
                    "sm_103a/cake_dsv4_2ada791761e6057628f6_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "c94422867e63d478cf9cb82d64661a43d499162321b78290f65b65d1ce4b25aa",
                "sources": [
                    "sm_103a/cake_dsv4_34c0d39b1248c1010a15_kernel.cu",
                    "sm_103a/cake_dsv4_34c0d39b1248c1010a15_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "17be1dde22562a78d48fb6120aca8e73e7f4b98f73a38d9d0ea588c0ec3f4ba1",
                "sources": [
                    "sm_103a/cake_dsv4_7383588b37366a763e1e_kernel.cu",
                    "sm_103a/cake_dsv4_7383588b37366a763e1e_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4b25b98298ef653eae8fc4cce699685abfcc48f8e0e7babd95a3630796225a64",
                "sources": [
                    "sm_103a/cake_dsv4_ee00a096a4346bd6ad78_kernel.cu",
                    "sm_103a/cake_dsv4_ee00a096a4346bd6ad78_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "a2040d948500253d55ecac75238b4f82e9cd7165ea6d26925e19cabc807208bb",
                "sources": [
                    "sm_103a/cake_dsv4_1dd9f772bf09c995842d_kernel.cu",
                    "sm_103a/cake_dsv4_1dd9f772bf09c995842d_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0a4d0fabb8d4c56d064752142761aed7f900e6466adc6ef906cb14cb21e0d1a8",
                "sources": [
                    "sm_103a/cake_dsv4_25214a8d235e87a9cc33_kernel.cu",
                    "sm_103a/cake_dsv4_25214a8d235e87a9cc33_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "62abfe11a85b45ca89b9e8d3627def5def3e283bca9ea712c44ddf9e439b5b44",
                "sources": [
                    "sm_103a/cake_dsv4_c09af191b368fc936d16_kernel.cu",
                    "sm_103a/cake_dsv4_c09af191b368fc936d16_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "706de01351b7cd8ceb6888a9427943a99c1fa4cd63a19898ef6faf3f2c750b27",
                "sources": [
                    "sm_103a/cake_dsv4_14df2e7f409bd2c5c501_kernel.cu",
                    "sm_103a/cake_dsv4_14df2e7f409bd2c5c501_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "72dfe6f3cc94735c0e04120b256bd9d1ec0620228039bbecd994c6c1406237f1",
                "sources": [
                    "sm_103a/cake_dsv4_8543cb10ca15edd3a26b_kernel.cu",
                    "sm_103a/cake_dsv4_8543cb10ca15edd3a26b_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4e6224033672232f0d29e1427473fb2d48e8ca8b1ef5c41c9d34c4dc79c59ecd",
                "sources": [
                    "sm_103a/cake_dsv4_e875bc1d8a3b3ff27030_kernel.cu",
                    "sm_103a/cake_dsv4_e875bc1d8a3b3ff27030_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bd0ea36f2b7b1335e2aad245666d9a9d5c264d721724f6b66d4b8b3cc3bb0ca1",
                "sources": [
                    "sm_103a/cake_dsv4_914897d09f87971636b3_kernel.cu",
                    "sm_103a/cake_dsv4_914897d09f87971636b3_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4a71a2732a5d5913f09e1bd418d37cc4c0b0b96371c93ccc45d15bb7f4df956a",
                "sources": [
                    "sm_103a/cake_dsv4_9c1c6548071b6184d8e3_kernel.cu",
                    "sm_103a/cake_dsv4_9c1c6548071b6184d8e3_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9f430a079ba1928ab48b460d2c9d74bc427f24211bf39d4241e6522b7f31f99e",
                "sources": [
                    "sm_103a/cake_dsv4_3b2807f9cb6acd058bdf_kernel.cu",
                    "sm_103a/cake_dsv4_3b2807f9cb6acd058bdf_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3e173e5eefc84548e3acf1e120cf11b2c2e4a7f4f86b501b435cb68489789396",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_cluster_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_cluster_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "48d99563a290f30f2630bb88a4d435862bf698d33a31265be9af3623cf803970",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_persistent_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_persistent_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_N16_OC1=1",
                ],
                "identity": "bd9f55f3a97e53a11dd58a1b2a117f3634548324c44f2c1989a5ce741c4b7e20",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_PV_N32_OC1=1",
                ],
                "identity": "0fa106e698fe3895437eab277fb355ec08730353937c27b3b2161a7851dae895",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N16_OC1=1",
                ],
                "identity": "716ac83e95561b0fee85ffa045e18bc7dd2803dc4a5d67fbcad1ca484a5561d5",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N16_OC2=1",
                ],
                "identity": "7d2061044473896977d4ff13222bfd3b7ed08b6883af99372e755fabd6a2d0cc",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N16_OC4=1",
                ],
                "identity": "d8e2c90f6824540f6f5eb65e749aa51c4595b5e16ba5d105094c07c32a099e26",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N32_OC1=1",
                ],
                "identity": "07e66a239cab04b0bf2d4505866792c61be41da5db938825747ca7969f3b24f2",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N32_OC2=1",
                ],
                "identity": "afe29a88fe519286f7326d526062addfba1a986855bcc51a24b56887597be498",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_SWAP_N32_OC4=1",
                ],
                "identity": "a734596c8d161f8afc8d4f44a12edb533e7e3cf294c8e2e32266bf9125aed4c0",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4d23efd4fd0e5cd8c3a1d96ccaa081be6252db1b057dab909223bde061b25e45",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_h64_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_OC2=1",
                ],
                "identity": "5bae01bb9ad06fde8d50942e0acf82a3e3413b1bf983c058e7f9b5d26f2b02b8",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_h64_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_h64_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_tile_h64_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_H64_OC4=1",
                ],
                "identity": "2b672e4dc1e342169d02aec16f3ea8eb73017a19a5240c3f8682e37aa01945c0",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_h64_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_h64_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_OC1=1",
                ],
                "identity": "ca917b70f66fdcb1c7c160d7f3ad08d0fe17f11a58e7addf157e93dcbb619579",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_OC2=1",
                ],
                "identity": "8a4d832d64ae179b154a7c7f4d55d1b33c28fe11b0b6c01388029cb6dc0d2d56",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_SELECT=1",
                    "-DCAKE_DSV4_NVFP4_DECODE_TILE_OC4=1",
                ],
                "identity": "aceb17c3786648297ddf6f4d151be237e1ae0baa5333515ccbab7ab1309b6f44",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "73050e8bff868aa535af7f8c7c5ef65edbc5b1a40ba6130380400642644a0074",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_merge_kernel.cu",
                    "common/cake_dsv4_nvfp4_merge_binding.cu",
                ],
            },
            "split_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "a17f7b729fe17d899e263d1615cbc2f73fa9b5dcec4d8ac465cd343cbcf9c263",
                "sources": [
                    "sm_103a/cake_dsv4_aa6dd9bfdbb2d261b462_kernel.cu",
                    "sm_103a/cake_dsv4_aa6dd9bfdbb2d261b462_binding.cu",
                ],
            },
        },
    },
}  # type: dict[str, dict[str, dict[str, dict[str, object]]]]


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_dsv4"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_dsv4"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "CAKE DSv4 CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def get_cake_dsv4_spec(variant: str, *, arch: str) -> dict:
    """Return the generated physical build and argument contract."""
    if arch not in _ARCH_REGISTRATIONS:
        raise ValueError(f"unsupported CAKE DSv4 architecture: {arch}")
    try:
        return _ARCH_REGISTRATIONS[arch]["variants"][variant]
    except KeyError as exc:
        raise ValueError(
            f"CAKE DSv4 variant has no generated source contract: {variant}"
        ) from exc


@functools.cache
def gen_cake_dsv4_module(variant: str, *, arch: str) -> JitSpec:
    contract = get_cake_dsv4_spec(variant, arch=arch)
    required = contract.get("min_cuda_version")
    if required is not None and not is_cuda_version_at_least(required):
        raise RuntimeError(
            f"CAKE DSv4 variant {variant} ({arch}) requires CUDA {required} or newer; "
            f"the CUDA toolkit found is {get_cuda_version()}. Its generated kernels spell "
            "the Blackwell QMUL4 as the PTX ISA 9.4 packed multiply "
            "(mul.e4m3x4.e2m1x4), which older nvcc/ptxas cannot assemble."
        )
    csrc_dir = _get_csrc_dir()
    sources = [csrc_dir / name for name in contract["sources"]]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CAKE DSv4 generated sources were not found: "
            + ", ".join(str(path) for path in missing)
        )
    spec = gen_jit_spec(
        name=f"cake_dsv4_{variant}_{arch.replace('_', '')}_{contract['identity']}",
        sources=sources,
        extra_cuda_cflags=[
            *_ARCH_NVCC_FLAGS[arch],
            *contract["compile_flags"],
            *contract.get("host_linkage_flags", ()),
        ],
        # The generated contract owns the fast-math decision.
        use_fast_math=False,
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated CAKE DSv4 {variant} JIT spec: {spec.name}")
    return spec


@functools.cache
def get_cake_dsv4_module(variant: str, *, arch: str):
    loaded = gen_cake_dsv4_module(variant, arch=arch).build_and_load()
    logger.info(f"Loaded CAKE DSv4 {variant} module")
    return loaded


@functools.cache
def gen_cake_dsv4_launch_sequence_module() -> JitSpec:
    """Host-only helper that issues the launches of a multi-kernel route in one FFI call."""
    csrc_dir = _get_csrc_dir()
    return gen_jit_spec(
        name="cake_dsv4_launch_sequence",
        sources=[csrc_dir / "cake_dsv4_launch_sequence.cc"],
        extra_include_paths=[_get_include_dir()],
    )


@functools.cache
def get_cake_dsv4_launch_sequence_module():
    return gen_cake_dsv4_launch_sequence_module().build_and_load()


__all__ = [
    "gen_cake_dsv4_launch_sequence_module",
    "gen_cake_dsv4_module",
    "get_cake_dsv4_launch_sequence_module",
    "get_cake_dsv4_module",
    "get_cake_dsv4_spec",
]
