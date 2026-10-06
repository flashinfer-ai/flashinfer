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
                "identity": "e6ae8bda18ca0ae4f59a007c73ef93819b005164dd2c75618633d137c31d724b",
                "sources": [
                    "sm_100a/cake_dsv4_af8de17d18d6aa78ab6c_kernel.cu",
                    "sm_100a/cake_dsv4_af8de17d18d6aa78ab6c_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "785e18da214e48d48ac83225d4aa3fe06e80cc5840b54c45724d62d1b490c3ea",
                "sources": [
                    "sm_100a/cake_dsv4_738d968ea053f90a32ba_kernel.cu",
                    "sm_100a/cake_dsv4_738d968ea053f90a32ba_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "414e32bd7c75ef2ecab99126445d04dc03eed2c26cbd00705dc59447ab1491b4",
                "sources": [
                    "sm_100a/cake_dsv4_16c0d8281e2d07db722e_kernel.cu",
                    "sm_100a/cake_dsv4_16c0d8281e2d07db722e_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "afb122111e3ae4e961a531da05534a557c6e821b9c5efe6dfe0ef974f2093129",
                "sources": [
                    "common/cake_dsv4_bf16_h128_swa128_kernel.cu",
                    "common/cake_dsv4_bf16_h128_swa128_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "ee3b6eaf83bca2787e5db2486e5dddfe47114804c6fb0e5fd25da55ed8a79d2d",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "2f58edf38982f6281473ea8f08a1d14f765197bf67a0decb4b50af89ccf65fee",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "b4c195be76ce16615b12edaa3354175d94f91044a0604b33806ae5a51b92fd0f",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "f8a200a9772b7087ecd35a45f63cfb0adf64e6810e99f662fd28ec5b696bee28",
                "sources": [
                    "sm_100a/cake_dsv4_9e8f2400154253393820_kernel.cu",
                    "sm_100a/cake_dsv4_9e8f2400154253393820_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "24be03e56f1bebdf8331349118ee4489bd90a532484159a3912ddce4869cfaf7",
                "sources": [
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_kernel.cu",
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0bcd7e68bd7ffc06c37014e558a191b6d804f8cdbf3b66647a37013f95b14850",
                "sources": [
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_kernel.cu",
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "298b15eb193178f30dd993c614e2ad2405307e611e628b730a2a00471ad419f3",
                "sources": [
                    "common/cake_dsv4_bf16_h64_compressed_q8_v38_kernel.cu",
                    "common/cake_dsv4_bf16_h64_compressed_q8_v38_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "afabef5fed786dc6dcf8182924ee6f754d489be0cccfbd08cb4abedf75897176",
                "sources": [
                    "common/cake_dsv4_bf16_h64_compressed_reduce_kernel.cu",
                    "common/cake_dsv4_bf16_h64_compressed_reduce_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d260857621b389e0cb078e9eb3ab1492126a9a1b3148447b66130c84fde421f1",
                "sources": [
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_kernel.cu",
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b7bd09d0b029ed56bdae05215b19941af94c4c51ce3cb2b647daf9b60af3a93b",
                "sources": [
                    "common/cake_dsv4_bf16_h64_prefill_kernel.cu",
                    "common/cake_dsv4_bf16_h64_prefill_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "000350ddfbfb88be633292dd529625dcd24a5c0416e6f322acf2086c4ce6e0c5",
                "sources": [
                    "sm_100a/cake_dsv4_0ee374651ac6d97917b0_kernel.cu",
                    "sm_100a/cake_dsv4_0ee374651ac6d97917b0_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "631eefc7fa934bf23b2167784d5bfebe13c5f0293e9522489632271bd8742684",
                "sources": [
                    "sm_100a/cake_dsv4_b04581bf99c7c33810aa_kernel.cu",
                    "sm_100a/cake_dsv4_b04581bf99c7c33810aa_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0752e57397db9e43e6e99a0fd584752914b95613fa406a12926af183719f77bb",
                "sources": [
                    "sm_100a/cake_dsv4_7c25e5e7e217d9692cb8_kernel.cu",
                    "sm_100a/cake_dsv4_7c25e5e7e217d9692cb8_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "c9cf18cbc27d2198912ddff3e608c73b462bea35ac21a2308bedc2b5994367c7",
                "sources": [
                    "sm_100a/cake_dsv4_6deb5d532f9a66d596a3_kernel.cu",
                    "sm_100a/cake_dsv4_6deb5d532f9a66d596a3_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "eafa1b489d0bfeba330ff43029dee4f7eeebe63046f3ff77de0724f52e833012",
                "sources": [
                    "sm_100a/cake_dsv4_f9c6ffee85a86779e48b_kernel.cu",
                    "sm_100a/cake_dsv4_f9c6ffee85a86779e48b_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3d8b2f227d6d64372fccde9122a871ba30417cfe6326ed90846394ff04ac8f74",
                "sources": [
                    "sm_100a/cake_dsv4_536b3a44f96e5c09fa99_kernel.cu",
                    "sm_100a/cake_dsv4_536b3a44f96e5c09fa99_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "266c11008eed6161c57d9d1e639e02c5596ef65164f28bc5ac3775096674a5cf",
                "sources": [
                    "sm_100a/cake_dsv4_ae42584c56277f4ab8e9_kernel.cu",
                    "sm_100a/cake_dsv4_ae42584c56277f4ab8e9_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "352746d692034c50a35b06d02d65716efe8dd8cc9e71eb6e2f77d51437188fce",
                "sources": [
                    "sm_100a/cake_dsv4_36501bc12985abdd4500_kernel.cu",
                    "sm_100a/cake_dsv4_36501bc12985abdd4500_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7f70315008bbee9382a534190cf232f00850718f51bef0a1e71e0a7a9fc89d71",
                "sources": [
                    "sm_100a/cake_dsv4_703373d90788f8785ed5_kernel.cu",
                    "sm_100a/cake_dsv4_703373d90788f8785ed5_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ad32a0f1926ce4e4f4184f362bff388d2f02b75a56ce939db746c1c833a25758",
                "sources": [
                    "sm_100a/cake_dsv4_8187b1ca82318d5f5f7b_kernel.cu",
                    "sm_100a/cake_dsv4_8187b1ca82318d5f5f7b_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "a2e2a11c720b37a0785f4259389a6b7a1b7d58fd7cf46bf2e617929b21aef71b",
                "sources": [
                    "sm_100a/cake_dsv4_a72b83aefd8e3e928580_kernel.cu",
                    "sm_100a/cake_dsv4_a72b83aefd8e3e928580_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a3333942fb9f9681b6b06b4a4f1a5735947928a8ca41e4e4b680f5c4ab76a5d9",
                "sources": [
                    "sm_100a/cake_dsv4_0aa7fec6b7ea28f44170_kernel.cu",
                    "sm_100a/cake_dsv4_0aa7fec6b7ea28f44170_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "1ce29a8ece68bcfb66f8313a922ba9d6cfaabff85d9bcde5d194604b1c72d32a",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_cluster_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_cluster_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3eae3d018dee00f233a8830a8d354428408a7119868d061a7e35a6f0f80dfb2a",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_persistent_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_persistent_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b4af1a17467d1d24d5286c624fbd4b8133d7a10b33af5d3f654dc1f77e42ca34",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_n16_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6042fb3369d4b3e018a803d39977d73b796d495107722af4c68e448614d9f8f0",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_n32_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9936d7e803755e085331a6024bf75129dd39b34cb7f8f99917bdb99e835c5582",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "fcb59ca8239d4d6c5d44821a6b55f5e47bd6e12b2e4c502f66523295a5416000",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc2_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7364ac70db892e142c03ed46b3e32018ab8fe5a98da77930bac3a07fdc5d20ed",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc4_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "76d0968e7173c4ab3891ed91649701e67a0ceae1b4a85046ce6395307db91272",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "880d7fc07aeeda669d63074a95ace5d80d20680b56acf585705879e265124d3e",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc2_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "cc6c7083873b6c206306166ef4b1a562192808efbd116a45c734417c8a007f3c",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "974789d49b20dc14ccf17a1205ec00c40d0aae8ccb4a9529f3868974947b854f",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a3183ea47eca044f20bc977290006324094f4dd4293ef3ad5ab8460f99a339ef",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4d42108e8502e16caf67366a3004df68fe039e80b138be5089e555af2d1b0eda",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_oc2_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ccc19258874254ae89ed90b67f397245bffd7bac7f68e7776ec6f177e88d8b12",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3de5ac40527c7041bb79d988f56ba8d4ff1aaed31f730665eb27f34e8f774e27",
                "sources": [
                    "common/cake_dsv4_nvfp4_merge_kernel.cu",
                    "common/cake_dsv4_nvfp4_merge_binding.cu",
                ],
            },
            "split_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "2607dcbd97154242c20139ba79133ca6ae8c841b3b1e9bace60cd9856e509496",
                "sources": [
                    "common/cake_dsv4_split_reduce_kernel.cu",
                    "common/cake_dsv4_split_reduce_binding.cu",
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
                "identity": "576ac523d650ed61d942463afb9553f06f9777d37bb4c4e7a4ee25a101cd0c33",
                "sources": [
                    "sm_103a/cake_dsv4_67540695b294db377884_kernel.cu",
                    "sm_103a/cake_dsv4_67540695b294db377884_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "55770323248fee558e28ba532e0b8b032ac3c64cdadfba440e7c443f24ccff63",
                "sources": [
                    "sm_103a/cake_dsv4_61616b294dbb50efcd37_kernel.cu",
                    "sm_103a/cake_dsv4_61616b294dbb50efcd37_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "7c09745168459ca084689c56a25722af295e46198bb6ea17faaa434933d4b9fa",
                "sources": [
                    "sm_103a/cake_dsv4_4ca59064569bce4c30ea_kernel.cu",
                    "sm_103a/cake_dsv4_4ca59064569bce4c30ea_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "afb122111e3ae4e961a531da05534a557c6e821b9c5efe6dfe0ef974f2093129",
                "sources": [
                    "common/cake_dsv4_bf16_h128_swa128_kernel.cu",
                    "common/cake_dsv4_bf16_h128_swa128_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "ee3b6eaf83bca2787e5db2486e5dddfe47114804c6fb0e5fd25da55ed8a79d2d",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "2f58edf38982f6281473ea8f08a1d14f765197bf67a0decb4b50af89ccf65fee",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "b4c195be76ce16615b12edaa3354175d94f91044a0604b33806ae5a51b92fd0f",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "ec2f5d56495e79da389244858ebbce9c7dc21e868da8dd41e5fdb853efc38fe4",
                "sources": [
                    "sm_103a/cake_dsv4_8c925186d1dcbb921252_kernel.cu",
                    "sm_103a/cake_dsv4_8c925186d1dcbb921252_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "24be03e56f1bebdf8331349118ee4489bd90a532484159a3912ddce4869cfaf7",
                "sources": [
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_kernel.cu",
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "84e85db58e3aba8fc89aead5c82b358ad2cd88355fe87931ff8277b000e71567",
                "sources": [
                    "sm_103a/cake_dsv4_8a0ab008d1081aaee107_kernel.cu",
                    "sm_103a/cake_dsv4_8a0ab008d1081aaee107_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "298b15eb193178f30dd993c614e2ad2405307e611e628b730a2a00471ad419f3",
                "sources": [
                    "common/cake_dsv4_bf16_h64_compressed_q8_v38_kernel.cu",
                    "common/cake_dsv4_bf16_h64_compressed_q8_v38_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3aef6a2f4561f504033c8aa780bd72638b3bbb966eb3517087db6950aa35b264",
                "sources": [
                    "sm_103a/cake_dsv4_b9c4afa107e20bf48d99_kernel.cu",
                    "sm_103a/cake_dsv4_b9c4afa107e20bf48d99_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d260857621b389e0cb078e9eb3ab1492126a9a1b3148447b66130c84fde421f1",
                "sources": [
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_kernel.cu",
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b7bd09d0b029ed56bdae05215b19941af94c4c51ce3cb2b647daf9b60af3a93b",
                "sources": [
                    "common/cake_dsv4_bf16_h64_prefill_kernel.cu",
                    "common/cake_dsv4_bf16_h64_prefill_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "acb55785bc0065e5d6f72998f9c42a3332b772a654181d78cdb49fae5ddf99c1",
                "sources": [
                    "sm_103a/cake_dsv4_e66b931b4499db288499_kernel.cu",
                    "sm_103a/cake_dsv4_e66b931b4499db288499_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9ddb772471f40f8ebc858a35d04edfeb37abc8b63446e8bdfc8eacec9f06ad91",
                "sources": [
                    "sm_103a/cake_dsv4_f52f0e6459ef0cca533b_kernel.cu",
                    "sm_103a/cake_dsv4_f52f0e6459ef0cca533b_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f1d8ba749574af6cd080b7e5cd6f554676565739c90ead8b9ba1dccfdad52b68",
                "sources": [
                    "sm_103a/cake_dsv4_27b20805ac082d6893b4_kernel.cu",
                    "sm_103a/cake_dsv4_27b20805ac082d6893b4_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "805699dc486a8e76ac79103bb8686b3f2918f322b65fab52a2941f64df59d4d3",
                "sources": [
                    "sm_103a/cake_dsv4_1130afe607578bd7b1aa_kernel.cu",
                    "sm_103a/cake_dsv4_1130afe607578bd7b1aa_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f2ab3739d0dd06423eef7de1c32d7d932562d1c30da6ab18d5dbe5a7579a166c",
                "sources": [
                    "sm_103a/cake_dsv4_fa1e790288db6dc28069_kernel.cu",
                    "sm_103a/cake_dsv4_fa1e790288db6dc28069_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "680d7f106513ae91d344bf63540390c3b98b0dcd4de2b933a341987a19966bc9",
                "sources": [
                    "sm_103a/cake_dsv4_d8f0c13705d3d18d17b5_kernel.cu",
                    "sm_103a/cake_dsv4_d8f0c13705d3d18d17b5_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f27cc233da350f3beefb457b536041dfe1489db1eaa60b2a1ebebf973dbe209e",
                "sources": [
                    "sm_103a/cake_dsv4_a6efa79d7d8effe3fde4_kernel.cu",
                    "sm_103a/cake_dsv4_a6efa79d7d8effe3fde4_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "352746d692034c50a35b06d02d65716efe8dd8cc9e71eb6e2f77d51437188fce",
                "sources": [
                    "common/cake_dsv4_fp8_h64_source_exact_kernel.cu",
                    "common/cake_dsv4_fp8_h64_source_exact_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5a604e62d8d524c358d7969f79976db8fde2c43670a06e6787dde086fd63ea73",
                "sources": [
                    "sm_103a/cake_dsv4_e0df95dcb157d250a7d5_kernel.cu",
                    "sm_103a/cake_dsv4_e0df95dcb157d250a7d5_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ad32a0f1926ce4e4f4184f362bff388d2f02b75a56ce939db746c1c833a25758",
                "sources": [
                    "common/cake_dsv4_fp8_lowhead_h64_kernel.cu",
                    "common/cake_dsv4_fp8_lowhead_h64_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "77c095ed9d366745ae0e56115a77670b58ecc3fcd05c2e939007da83a7553bd3",
                "sources": [
                    "sm_103a/cake_dsv4_7b76eac0776b050646fa_kernel.cu",
                    "sm_103a/cake_dsv4_7b76eac0776b050646fa_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2d9b5c7a17071bc840cf46c84dd974f3a86699ab642d7aacb673ed0206003616",
                "sources": [
                    "sm_103a/cake_dsv4_2dae64399bc92ecee34e_kernel.cu",
                    "sm_103a/cake_dsv4_2dae64399bc92ecee34e_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0a5235e4d732f4dc96d15065bf5f2624eac8cb715f2fb0bb6596e440b5477892",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_cluster_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_cluster_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "267ea948d1d96fe95cc55d404315b54ef9fdbdc12e9f72c0da9019c77aed3047",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_persistent_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_persistent_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0154d4af0e78766cb717e7cb42fe9ec69566e6dc3b58068c60145a36fe413ee2",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_n16_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7aef5215140548a85a3114174ade99763d9a3f32039f9787d954e4ebfb19ad0a",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_pv_n32_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_pv_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f8bc1227bdaa2fd66c73abc94bd1ebbe590cba37bb67cd18e0d37e4986c88cb2",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "96edb29268d36b0f5c27161445b168888b4da74ef741c84a3dad8f245c16dcba",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc2_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "1c8fd70c2450915134ad6849dd533ff760e7923034963c3b8890473ac8ddf95c",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc4_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n16_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "815854cec013ae41d1b06ef05dc76088fbb240879bdfcad405f62254fae74879",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "fdda0829f93da5cea3812382ba1b2252e6e74ee868ea17e4eb1f3412373d0302",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc2_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "e172ea5b6e9f000f4e430983b5d1984ea3c718ec63e62d376b8563c33c7b0931",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9a3cf6890f5dad197dac40c0c15409550ffe3b0a340e1edeaaee246b28a7e647",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "198be7bcf066ca381c849746a0696c843385002af6b158d76bfafab8b72c04ec",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_oc1_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc1_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "af0b6716baeb170b4240554ecd8eecfa0f1004f28d020f832aad290955c263a5",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_oc2_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc2_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "e5526211a7a378b79cdd2706635a0bc106e8d127c77bc06955608ce1c2f9b232",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f712795e412f650bc6ed2fdb7125f1b6846b36063e6f4c7f17e73d5ec71534d1",
                "sources": [
                    "common/cake_dsv4_nvfp4_merge_kernel.cu",
                    "common/cake_dsv4_nvfp4_merge_binding.cu",
                ],
            },
            "split_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "a530c22fa6529fce877c44fe2492bdc88086574bea6c67f50e740f7e05860a9d",
                "sources": [
                    "sm_103a/cake_dsv4_51c1b1cf8ced19ef9080_kernel.cu",
                    "sm_103a/cake_dsv4_51c1b1cf8ced19ef9080_binding.cu",
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
