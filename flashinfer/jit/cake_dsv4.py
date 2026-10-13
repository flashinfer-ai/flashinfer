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
                "identity": "e3c38b1c1bc01a95065d7875758b3efcfc155b6d0610c3360db70ef168870580",
                "sources": [
                    "sm_100a/cake_dsv4_c602c3f805921b955ae5_kernel.cu",
                    "sm_100a/cake_dsv4_c602c3f805921b955ae5_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "039baa17b1d46e815d65b2ef766ed43edc6be6eb478fa12da90074f5f6d4e17d",
                "sources": [
                    "sm_100a/cake_dsv4_d59492e451e80c110411_kernel.cu",
                    "sm_100a/cake_dsv4_d59492e451e80c110411_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "36c56a943522130196ea2a5d7cb7664adaa1a5dbf4597ef17d16b892ce9589a2",
                "sources": [
                    "sm_100a/cake_dsv4_fa5539cb1afb7555dff5_kernel.cu",
                    "sm_100a/cake_dsv4_fa5539cb1afb7555dff5_binding.cu",
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
                "identity": "a42f748db259b8871bc01232991793aaf1726df43b7bf9ec7aada452f1f69a5f",
                "sources": [
                    "sm_100a/cake_dsv4_b3138c1168ef949b4019_kernel.cu",
                    "sm_100a/cake_dsv4_b3138c1168ef949b4019_binding.cu",
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
                "identity": "ab342562aadfe43e4ce7b2c8e9b66e1832d406ba99942f7b40c785a18b82a84b",
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
                "identity": "28a0e491ed4d19bdec9210900ea94a636d3d7380e75790216d1dbb445b54808b",
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
                "identity": "a4f7a9e1f7ccb2e0bf9bcc5f6a064781de062d444f4d34b320e61e7f16130da1",
                "sources": [
                    "sm_100a/cake_dsv4_74706ad372b69a30483d_kernel.cu",
                    "sm_100a/cake_dsv4_74706ad372b69a30483d_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b06aa58cef3479ddf5616e8c8cf3d6c616edb71684bdeaf51c61926fa090d6ee",
                "sources": [
                    "sm_100a/cake_dsv4_228778aae6bc72c4170d_kernel.cu",
                    "sm_100a/cake_dsv4_228778aae6bc72c4170d_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4c49dae3b8e7d38761883bbfaf49fa60644e66a241d399845cd3054e21c5dcf4",
                "sources": [
                    "sm_100a/cake_dsv4_fd59534f8fb24edd3529_kernel.cu",
                    "sm_100a/cake_dsv4_fd59534f8fb24edd3529_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "7eac249872dc4dc460cc09f0352bd966f8fa5c5aabe6048999970b971ab1f08b",
                "sources": [
                    "sm_100a/cake_dsv4_afe09a82c14f8bd895f4_kernel.cu",
                    "sm_100a/cake_dsv4_afe09a82c14f8bd895f4_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a90bc61850811e6f00c15ba764a53a456be37aa63c76ae653ce3f198e449de81",
                "sources": [
                    "sm_100a/cake_dsv4_4f1bc4343283f1ba5536_kernel.cu",
                    "sm_100a/cake_dsv4_4f1bc4343283f1ba5536_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5e19b41c0584810aa817b584cabf019ec20e0162aca45b3cab8a1b1fc3d5f18b",
                "sources": [
                    "sm_100a/cake_dsv4_352cd296b2fa5c70f5f9_kernel.cu",
                    "sm_100a/cake_dsv4_352cd296b2fa5c70f5f9_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "1f3df891eb5bda4415cf1cccc897e3212f1f459a45471df175e9f80046cd78c8",
                "sources": [
                    "sm_100a/cake_dsv4_746fa3d7b828ce3dd63c_kernel.cu",
                    "sm_100a/cake_dsv4_746fa3d7b828ce3dd63c_binding.cu",
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
                "identity": "4850838c827b535615d6e7e2777ec35000dd1c707c81a468a11a0ef9b7e5f373",
                "sources": [
                    "sm_100a/cake_dsv4_c42c0d4632dca5f570a8_kernel.cu",
                    "sm_100a/cake_dsv4_c42c0d4632dca5f570a8_binding.cu",
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
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "b66a119f696625d5f73a15cf6bb01c6d88f9c20603b0ac323883a511e05e5954",
                "sources": [
                    "sm_100a/cake_dsv4_0472b891257529a5f102_kernel.cu",
                    "sm_100a/cake_dsv4_0472b891257529a5f102_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9becd5eb050721a1c56c69945509d72ea534a05d6450dabb75001030cb0deec5",
                "sources": [
                    "sm_100a/cake_dsv4_2beee9997128cc45a86b_kernel.cu",
                    "sm_100a/cake_dsv4_2beee9997128cc45a86b_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "90fcb1d4fef96bd6530118ff3e4bff8557896974d8c9eef734b517858f74ed6b",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_cluster_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_cluster_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "56cff27e58b865c36ec815127729fed389174b2e82bc422054ae4372788afb71",
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
                "identity": "1ab6f68908e0ec65d3ecafc7a858be3ebd4edc29c783aa86310314afcb322ec8",
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
                "identity": "05aac533eaa644eaef18abe2a70d2b6ba6786cea0f951c6387daaaf0b3c9d164",
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
                "identity": "a826ee5e7616476f82726473f6c8aa62ae16a8df9aa336c138b23bde307185f6",
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
                "identity": "320e783ebde4dccff0c692c4873a2a72eef43851034bbad4cd9d954968813865",
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
                "identity": "65146b0ffd8417de0daea1c1eadb1cb656a9d51df57e7cf916e3ad6282814099",
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
                "identity": "52d8d488ae988c41540b647555017665bc6a41755710104bbcb793371cabaa4c",
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
                "identity": "743970168898bd8715fafbec926ca85851638f36fe98e06c3207159d33ddb965",
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
                "identity": "f473d4c6e6558b9dd496a5434b2b0b379084e4aec88e848e94a7768cbe0877d4",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "112afaf2bf36df2d908dc52d8b5d4a0725b2d2269a271428569bd2f20ee9cc68",
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
                "identity": "71f55cc5b1359e1779563bf6ae77c6a2394fbee75a0e64141530ea5555348fc7",
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
                "identity": "4013e7af9db870f5258440007a656d4c1b48299ec9cf9f10902ef674cbfa7c51",
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
                "identity": "8977d2fc4c310f99eb1f8f12d8d5216c65dd6ed8e46892e7f8a9a2084d5f7f69",
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
                "identity": "4117d74c17266464cf1ad3caefde5a46bbe09629c788efbb0580765ce764bd22",
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
                "identity": "770269d570d69bc70b2a1235850ae4b9b31c8a1407ba58a62afeb100494f355f",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3868c9d64707b2cd86ce99cb43ba8b68a6032a493b99e10f0fbd7d5d3063c5a8",
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
                "identity": "8ccc3d6808d9dcee8c6c4d92e196f8155df2ee933b07ccf06ae380bff2d93b0d",
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
                "identity": "2ece40c3ce98e9d874807f7c208ba606da427386095c431e37b692e876145d72",
                "sources": [
                    "sm_103a/cake_dsv4_c3cf2c612e13bf182910_kernel.cu",
                    "sm_103a/cake_dsv4_c3cf2c612e13bf182910_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "f3c23f97450d95298c4ceff85fadf8744dff25f253f4fb0c9953e5b3cc4d8d76",
                "sources": [
                    "sm_103a/cake_dsv4_8c012a14f26c0b3fc996_kernel.cu",
                    "sm_103a/cake_dsv4_8c012a14f26c0b3fc996_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "033e4b445b803b6b8a105d9aa8a817c06d1ac9335592b0f570614f681004f5f6",
                "sources": [
                    "sm_103a/cake_dsv4_f1ee6630f05cf65f5f69_kernel.cu",
                    "sm_103a/cake_dsv4_f1ee6630f05cf65f5f69_binding.cu",
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
                "identity": "3cb4749458f3444e9d6bc7e671e2924b0ded1f6cfc764b6d7a09309847d62047",
                "sources": [
                    "sm_103a/cake_dsv4_a3e160c0c67071fa998a_kernel.cu",
                    "sm_103a/cake_dsv4_a3e160c0c67071fa998a_binding.cu",
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
                "identity": "ab342562aadfe43e4ce7b2c8e9b66e1832d406ba99942f7b40c785a18b82a84b",
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
                "identity": "28a0e491ed4d19bdec9210900ea94a636d3d7380e75790216d1dbb445b54808b",
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
                "identity": "18b0f19a19c542c099008d70736aef1f036aead4191bb319b2606a896ac5abf0",
                "sources": [
                    "sm_103a/cake_dsv4_985b70c1f95a44b1a915_kernel.cu",
                    "sm_103a/cake_dsv4_985b70c1f95a44b1a915_binding.cu",
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
                "identity": "753d41d305917df72d2e5737bbba291e837aa36b74cdc135fd74d8108493b8b6",
                "sources": [
                    "sm_103a/cake_dsv4_385acfbc8606f3d335b2_kernel.cu",
                    "sm_103a/cake_dsv4_385acfbc8606f3d335b2_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "8958ec1aba11e040f97aad881b6d992155ba003dc9f370fda7dd116303cb7a82",
                "sources": [
                    "sm_103a/cake_dsv4_ac39da4cf9885a0981e8_kernel.cu",
                    "sm_103a/cake_dsv4_ac39da4cf9885a0981e8_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "760689bd629bc3178bee8e54de9e108c1801c03815e5356232964428443231a3",
                "sources": [
                    "sm_103a/cake_dsv4_f4f571730c20919320eb_kernel.cu",
                    "sm_103a/cake_dsv4_f4f571730c20919320eb_binding.cu",
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
                "identity": "f04a654101ef0bde4ffd56b727d2cad7b1ffa134a426a80409e0bbdcf765ce2d",
                "sources": [
                    "sm_103a/cake_dsv4_f3b968292b4c99769293_kernel.cu",
                    "sm_103a/cake_dsv4_f3b968292b4c99769293_binding.cu",
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
                "identity": "669b914522eb2606591ccae76d2e99bc986cd8393c378d2859738b003b8ac06f",
                "sources": [
                    "sm_103a/cake_dsv4_1a99f26bd105cfc54a61_kernel.cu",
                    "sm_103a/cake_dsv4_1a99f26bd105cfc54a61_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d23558ffc6ff0728296296fc6be7a246bae07e07d111ea06a938c126f1504cf8",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_cluster_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_cluster_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "97f66ada72365e7e4d20acc96ce2eb0d81f8fd3fd9cd736d3fd5db167a68f86c",
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
                "identity": "9275dbe72169a9a2881877df4e1dbea4a8f593f6b23cb197faca4bd73bbb0b13",
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
                "identity": "b8b27c41d767d82c5502fd2d3249eaaf0e1ce5e98d336a60fabf0fe25e8cb256",
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
                "identity": "5ea7107ed25486f1a914df6075bd19f17c6670f0f9a134f99d9d26e06d961a97",
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
                "identity": "3806003bd3a30bb085317ea3d03e1a729e930f5a529e3c6c6adbd05c93e47429",
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
                "identity": "0385745419d8c9a36e543060694c1a53d752acf15e28fae311b877f8238c1a27",
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
                "identity": "2686a9baf5d8d8791bdb866989f087fcd8619048738ee9f12d3a43941cb6a569",
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
                "identity": "5eb42d803c7e710fd698cb52bcdc8ffcbfc8f51efe62e334397b3df840df14a2",
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
                "identity": "948363dfaeed4cfe1b23c76ae1b2d21d65f8a2d6e10fa5c18522deceaf4fa32f",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_swap_n32_oc4_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9be19d7c73a37b35a0c0e174c575785087ae89dea9ed082419d849a8a0f9b160",
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
                "identity": "40b2368796b39f6eeb2cff3fdf3a0b6561ccc8079dbb67b83be8fc00c8dc1602",
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
                "identity": "bce172d7ad6ce6fae32984c60f65c165e43b5d85b81a645d5985e8dba6ea3262",
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
                "identity": "64a4fb5ab31dc90a6ac00e6c81552c56c0966c9e32fceea013dca7b6169160c4",
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
                "identity": "92f16b7b832c8b9504985d144c84888ca32fd45da5eecd4cc93fb154906eb216",
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
                "identity": "f5a24fd8fcbf4ec5ddc74fc494db3f6037fb118b9c9067b80fafd873eec9bd36",
                "min_cuda_version": "13.4",
                "sources": [
                    "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
                    "common/cake_dsv4_nvfp4_decode_tile_oc4_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7185d93fad5a359ea5d0a6dde1ba857d6f14b337d2e1e774339eedc198e9f263",
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
                "identity": "8ccc3d6808d9dcee8c6c4d92e196f8155df2ee933b07ccf06ae380bff2d93b0d",
                "sources": [
                    "common/cake_dsv4_split_reduce_kernel.cu",
                    "common/cake_dsv4_split_reduce_binding.cu",
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
