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
                "identity": "62d84ea47c2d3382a35ed4ed1b5dd9882b461393a80d2d61a67b23926aeea90a",
                "sources": [
                    "sm_100a/cake_dsv4_860a0ac41a77c1305873_kernel.cu",
                    "sm_100a/cake_dsv4_860a0ac41a77c1305873_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "67f7ea480ca958a86733932d82a556a0c2696fd406c66913ee64f1a12e8b0258",
                "sources": [
                    "sm_100a/cake_dsv4_8a33d6688c16c25820e9_kernel.cu",
                    "sm_100a/cake_dsv4_8a33d6688c16c25820e9_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "3bcd949aa7a96ca39c2a5cec4d10af6029a8e1e98644de100de2c0c54a1cf6ac",
                "sources": [
                    "sm_100a/cake_dsv4_e1c7ffd7ad7de34b2096_kernel.cu",
                    "sm_100a/cake_dsv4_e1c7ffd7ad7de34b2096_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "547edaa4352398e879bcfc07d49aa2b8a31ca5cc786994c5afc44fd50cab7069",
                "sources": [
                    "sm_100a/cake_dsv4_7abc1b8f5f3d1d936f1a_kernel.cu",
                    "sm_100a/cake_dsv4_7abc1b8f5f3d1d936f1a_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "f7f4c02437cb226f8ea02311e0d8c86d929949b70e94b499fa258d499c0c166e",
                "sources": [
                    "sm_100a/cake_dsv4_71e0baa44c61b948d82c_kernel.cu",
                    "sm_100a/cake_dsv4_71e0baa44c61b948d82c_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "0bed17b1469fe2fe6f6fa228ea8998a725656f67cca38996254610cc66e9d69d",
                "sources": [
                    "sm_100a/cake_dsv4_3af1afa1871d27324303_kernel.cu",
                    "sm_100a/cake_dsv4_3af1afa1871d27324303_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "24c50bef8dca8d253e5f6a7621577ca17e87ace4dbf5e484aa8eb5d49241f948",
                "sources": [
                    "sm_100a/cake_dsv4_fc73aade5a5f4f85a9e8_kernel.cu",
                    "sm_100a/cake_dsv4_fc73aade5a5f4f85a9e8_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "f79b87828ae3ef044fb42d72b8b150252c814b3d195c7c471380e9fc9d429651",
                "sources": [
                    "sm_100a/cake_dsv4_1b4dfa2ab26fc784f640_kernel.cu",
                    "sm_100a/cake_dsv4_1b4dfa2ab26fc784f640_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bb71538cd91c6511fdfedc17dae9809105766dc0de2cd2e22502ef0b5086d459",
                "sources": [
                    "sm_100a/cake_dsv4_61008e64c0c3bd2e1a83_kernel.cu",
                    "sm_100a/cake_dsv4_61008e64c0c3bd2e1a83_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f75d6974f10380ef99aacf7d138451cde1eab443b6531c30c49494ba822a757c",
                "sources": [
                    "sm_100a/cake_dsv4_c9156a2e9f65117a0900_kernel.cu",
                    "sm_100a/cake_dsv4_c9156a2e9f65117a0900_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4705f2ad576d2396b6c6af7345590ef031f6ef793435d00633f5f203a3b7d97e",
                "sources": [
                    "sm_100a/cake_dsv4_58df983ae8026a009dab_kernel.cu",
                    "sm_100a/cake_dsv4_58df983ae8026a009dab_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3a3e434c91f50e1f2ca7cc1ec095a3ddf26d82c3b623401e64439286a559bc13",
                "sources": [
                    "sm_100a/cake_dsv4_efb864796f11291dc52c_kernel.cu",
                    "sm_100a/cake_dsv4_efb864796f11291dc52c_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "207e9fab5a1ede7bab90e9bd91bc844503cfe92498149be1ae08740291bed38d",
                "sources": [
                    "sm_100a/cake_dsv4_cfa285ef8a408579ac99_kernel.cu",
                    "sm_100a/cake_dsv4_cfa285ef8a408579ac99_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "895f8fa2544bed14673c887629c934d700432ad18f91ea778a1a3a42c8f1f468",
                "sources": [
                    "sm_100a/cake_dsv4_c85537d4ca6a34893cd1_kernel.cu",
                    "sm_100a/cake_dsv4_c85537d4ca6a34893cd1_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "8180253e9bc5483c7b05fc7e75f241654170b031aae025996e8b1ed6fddacc74",
                "sources": [
                    "sm_100a/cake_dsv4_cc8f5e8152e47e5e566a_kernel.cu",
                    "sm_100a/cake_dsv4_cc8f5e8152e47e5e566a_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7e9af3e31cb55dcca16b967c42cef1bafcc8ca4ea41a87e5bbfc865a4e248b1d",
                "sources": [
                    "sm_100a/cake_dsv4_e3b65d62b4db145adce7_kernel.cu",
                    "sm_100a/cake_dsv4_e3b65d62b4db145adce7_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3a1cbafd16ef61ea1d6a9b5fa3491bae292d62e7b0fa0bb8f36db1b87d823a94",
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
                "identity": "a7566a70573796c660cf0e4c7da3e16f14649fdde941eefc5112850b5b05af4c",
                "sources": [
                    "sm_100a/cake_dsv4_155cefce7e1900a952c3_kernel.cu",
                    "sm_100a/cake_dsv4_155cefce7e1900a952c3_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "8dc1d9f7d7303b17068ec487cbd397ad1ff12900cbc915c7acd799b5e94441dc",
                "sources": [
                    "sm_100a/cake_dsv4_ca071620f07188f5ea22_kernel.cu",
                    "sm_100a/cake_dsv4_ca071620f07188f5ea22_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "417d9ce02c247fa773800083f288ff84a460fd57803ae8e071a72583d5c05ec1",
                "sources": [
                    "sm_100a/cake_dsv4_ee710a07143d70b73dc2_kernel.cu",
                    "sm_100a/cake_dsv4_ee710a07143d70b73dc2_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b5d3d93e204bde107a71834b79c44cd76194c40096431603f2b9f79ef1f51179",
                "sources": [
                    "sm_100a/cake_dsv4_aa5800406f680b6ad9fc_kernel.cu",
                    "sm_100a/cake_dsv4_aa5800406f680b6ad9fc_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d7bbf52d7745451382bb4b66a9accebb847e5caffba499a970b19f53ade0cd3b",
                "sources": [
                    "sm_100a/cake_dsv4_fcaeb5ddd771372bdb3d_kernel.cu",
                    "sm_100a/cake_dsv4_fcaeb5ddd771372bdb3d_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5ad4cc9ac3a4ba11438c6bb96733b71de3280a9a0a08fa4e75777667c42bf973",
                "sources": [
                    "sm_100a/cake_dsv4_cd52094425e6b8d7102c_kernel.cu",
                    "sm_100a/cake_dsv4_cd52094425e6b8d7102c_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "34f31c87a2329b6f2da527f538680e3f4e55fb58f45faf3df6064db686e6cdfc",
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
                "identity": "205ebc76375ad0427af3eeb50178c1393fdd830a1a91749c8fbaea45e5ebae3e",
                "sources": [
                    "sm_100a/cake_dsv4_9c5822db6c2739c63dc1_kernel.cu",
                    "sm_100a/cake_dsv4_9c5822db6c2739c63dc1_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a2ef15055c43742ba5213b0b075e6e2b9e62b8756a50c1da24510d7d92711008",
                "sources": [
                    "sm_100a/cake_dsv4_b4379be324141d01636c_kernel.cu",
                    "sm_100a/cake_dsv4_b4379be324141d01636c_binding.cu",
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
                "identity": "28e691fa6a52f32ebe7a219b185071e1610f7b0ed3f31a61d2eeb95eace25432",
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
                "identity": "1e14116a460e05c120b92d83b184f9ab922cec7bc5efc84ed1b015f633d97f56",
                "sources": [
                    "sm_103a/cake_dsv4_06cdf778054cc6c67dc5_kernel.cu",
                    "sm_103a/cake_dsv4_06cdf778054cc6c67dc5_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "3826b6dc665765c41b816731c47ef4abfe27e83c06e8d2e54ff1291b39071695",
                "sources": [
                    "sm_103a/cake_dsv4_a8b112bf2abd1a8ed848_kernel.cu",
                    "sm_103a/cake_dsv4_a8b112bf2abd1a8ed848_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "0092534b48dd496fce4f2a12c3bb6c0b447b5ff7ddd8d6803f0f7efa1061e081",
                "sources": [
                    "sm_103a/cake_dsv4_8b60cfd54a78d36864f4_kernel.cu",
                    "sm_103a/cake_dsv4_8b60cfd54a78d36864f4_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "60a8c78dba4d200a5f30be65e0f0756b095b73110e24aabc4054ca61b69bdee1",
                "sources": [
                    "sm_103a/cake_dsv4_f918e4ac3c5b77c4514b_kernel.cu",
                    "sm_103a/cake_dsv4_f918e4ac3c5b77c4514b_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "3eb3212a74d41126e9e210ad337a872f9a2771100cad58beef3d6d19fb09e366",
                "sources": [
                    "sm_103a/cake_dsv4_8c2cf89e71be1ebbd233_kernel.cu",
                    "sm_103a/cake_dsv4_8c2cf89e71be1ebbd233_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "1af3aafafd31d3105b34ffecb7531a9208651fb6aa5a84845c35718ad995816f",
                "sources": [
                    "sm_103a/cake_dsv4_9c1ead828285aaa7f400_kernel.cu",
                    "sm_103a/cake_dsv4_9c1ead828285aaa7f400_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "6401fc666ef0b8b4fbedcc2b39f17484d032a4032c71ce29307a5e41a31e6b80",
                "sources": [
                    "sm_103a/cake_dsv4_6a34a835c2b39424c20c_kernel.cu",
                    "sm_103a/cake_dsv4_6a34a835c2b39424c20c_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "c40b850622115acd5c33f6efc0302e76acceab9e4c0178b51472090c1af66824",
                "sources": [
                    "sm_103a/cake_dsv4_3b4ec80fd59f1a59eb0d_kernel.cu",
                    "sm_103a/cake_dsv4_3b4ec80fd59f1a59eb0d_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f1d591f1dda55d6e69f65eb884214ac484c3b2674a1912ddd8004bbd3cc39864",
                "sources": [
                    "sm_103a/cake_dsv4_a164ee5cbd7fc5b4d14d_kernel.cu",
                    "sm_103a/cake_dsv4_a164ee5cbd7fc5b4d14d_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4eb4e4705d906bba3f910bed9982b17c9b49d57e317a15331c076756a06c5fd2",
                "sources": [
                    "sm_103a/cake_dsv4_ebf2dfc077eb990f64fa_kernel.cu",
                    "sm_103a/cake_dsv4_ebf2dfc077eb990f64fa_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bfcef65f393238ddca3f5820d1d9ce265936416104d6c6d55c86d8d1262da13a",
                "sources": [
                    "sm_103a/cake_dsv4_584c7f1bfa07e4783b08_kernel.cu",
                    "sm_103a/cake_dsv4_584c7f1bfa07e4783b08_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b23d91d08d453704e5a0a409d4765501d3c5583a062ad0509dd8c392d483239d",
                "sources": [
                    "sm_103a/cake_dsv4_e900f8c37ff1f0974a39_kernel.cu",
                    "sm_103a/cake_dsv4_e900f8c37ff1f0974a39_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "cd5cbbee5354e65e06f929402dbb931a25a8a951de821dc7454baed4efbf42e3",
                "sources": [
                    "sm_103a/cake_dsv4_a293c552f4c6f588106c_kernel.cu",
                    "sm_103a/cake_dsv4_a293c552f4c6f588106c_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6908a57716c6092edc9599c148dcc7feeb72994a113ce446b9cae989da62f680",
                "sources": [
                    "sm_103a/cake_dsv4_2ada791761e6057628f6_kernel.cu",
                    "sm_103a/cake_dsv4_2ada791761e6057628f6_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bb52a6ee67f3fedc6d1f33eb212f77aa3b74c7008ec92cfbc6dbc4712228eb1f",
                "sources": [
                    "sm_103a/cake_dsv4_34c0d39b1248c1010a15_kernel.cu",
                    "sm_103a/cake_dsv4_34c0d39b1248c1010a15_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "adaac9d5879c74c27b7037f850006bf721cc538052dc86e01f30d1812f7f6d39",
                "sources": [
                    "sm_103a/cake_dsv4_7383588b37366a763e1e_kernel.cu",
                    "sm_103a/cake_dsv4_7383588b37366a763e1e_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "db41d760e3ea9a9b191cc026df90398f29e05ceb874661e03c7c3296d68a8355",
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
                "identity": "3bcfd48e83508093b5bab1d2da8eea7bb707052bb586d8d9aa83f9d992c45e63",
                "sources": [
                    "sm_103a/cake_dsv4_1dd9f772bf09c995842d_kernel.cu",
                    "sm_103a/cake_dsv4_1dd9f772bf09c995842d_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "40c82e6efd7ff1df1716a14169b773f43f2b1be4716f50cb0bfd27d38792f066",
                "sources": [
                    "sm_103a/cake_dsv4_25214a8d235e87a9cc33_kernel.cu",
                    "sm_103a/cake_dsv4_25214a8d235e87a9cc33_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b99dd4ccb610b4f9fc9adb50a8547bf186c2cfafd7a3569f6bba718fb8735b42",
                "sources": [
                    "sm_103a/cake_dsv4_c09af191b368fc936d16_kernel.cu",
                    "sm_103a/cake_dsv4_c09af191b368fc936d16_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "75573c82f3736e960fb429c9810c387dc1e14285d619ee55ca1774a9570f59ca",
                "sources": [
                    "sm_103a/cake_dsv4_14df2e7f409bd2c5c501_kernel.cu",
                    "sm_103a/cake_dsv4_14df2e7f409bd2c5c501_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "90cba54ab1dfad6ed2629ccd75a0c09c1ddadfd3a428a4080648eaeb63629e73",
                "sources": [
                    "sm_103a/cake_dsv4_7682c095a61495bc66df_kernel.cu",
                    "sm_103a/cake_dsv4_7682c095a61495bc66df_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "85c5fd217211d6d4d073582e78b8aa37db0895d50c8d118a9bc55d65144ab14f",
                "sources": [
                    "sm_103a/cake_dsv4_70a0b1ae882054e5c302_kernel.cu",
                    "sm_103a/cake_dsv4_70a0b1ae882054e5c302_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bdec12e03393c622aa00b8d5912ff6ebd5184e60bdd147207368b9d045a3b967",
                "sources": [
                    "sm_103a/cake_dsv4_914897d09f87971636b3_kernel.cu",
                    "sm_103a/cake_dsv4_914897d09f87971636b3_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9b15425dcf780f49cc5ceef9428732828f0a56f4169a918dea4a6efa0b384373",
                "sources": [
                    "sm_103a/cake_dsv4_9c1c6548071b6184d8e3_kernel.cu",
                    "sm_103a/cake_dsv4_9c1c6548071b6184d8e3_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "765b91210fd3d1a4f1f8a83ae541899c4c6b3a9ec95796682dfecd379dde8e5f",
                "sources": [
                    "sm_103a/cake_dsv4_fc0b78470851912fb4c6_kernel.cu",
                    "sm_103a/cake_dsv4_fc0b78470851912fb4c6_binding.cu",
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
                "identity": "aa4a3fef25dc679115e480518d4930db875f3a030ce5d333c97c425d14e05d50",
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
