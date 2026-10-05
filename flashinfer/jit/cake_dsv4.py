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

_PLAN_BF16_H128_PERSISTENT = (
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
)

_PLAN_BF16_H128_PERSISTENT_TMA_O = (
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
)

_PLAN_BF16_H128_SWA = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("buffer", "O"),
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
    ("parameter", "num_head_tiles"),
    ("parameter", "has_sinks"),
)

_PLAN_BF16_H32_MERGE = (
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

_PLAN_BF16_H64_PREFILL = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
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
    ("parameter", "num_query_tokens"),
    ("parameter", "has_sinks"),
)

_PLAN_BF16_H64_SPLIT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "partial_O"),
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
    ("parameter", "sparse_topk"),
    ("parameter", "num_splits"),
    ("parameter", "has_sinks"),
)

_PLAN_BF16_SWA_DECODE = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("buffer", "O"),
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
    ("parameter", "has_sinks"),
)

_PLAN_FP8_H64_M64_SWA_K = (
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
    ("parameter", "max_q_len"),
    ("parameter", "batch_size"),
)

_PLAN_FP8_H64_SOURCE_EXACT = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("buffer", "O"),
    ("buffer", "cum_seq_lens_q"),
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
    ("parameter", "has_sinks"),
    ("parameter", "total_work_items"),
)

_PLAN_FP8_LOWHEAD = (
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
)

_PLAN_FP8_PERSISTENT = (
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

_PLAN_H8_H16_SOURCE_EXACT_TOKENS = (
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
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "ad4870802b8f734cf01b36392e8046c12eca0dccada373c51c036206af9c9db2",
                "sources": [
                    "sm_100a/cake_dsv4_90aec41cf5294c35cf70_kernel.cu",
                    "sm_100a/cake_dsv4_90aec41cf5294c35cf70_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "cf8790a9d9e7d04d2124f3c3395cdf384db07f5a831d07ddd4e4a666d85f8752",
                "sources": [
                    "sm_100a/cake_dsv4_9ca76f113362f5fdedc5_kernel.cu",
                    "sm_100a/cake_dsv4_9ca76f113362f5fdedc5_binding.cu",
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
                "arg_plan": _PLAN_BF16_H128_SWA + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3058315c3c2fc4e5cc2f96db47f72d5e378233402d75427c9250bcc6b9bb2227",
                "sources": [
                    "common/cake_dsv4_bf16_h128_swa128_kernel.cu",
                    "common/cake_dsv4_bf16_h128_swa128_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "b88b762d209cc80c7dc32f2cfe93026e2f95036498b901b387731ccac8868a17",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "84fcc249e9f11173c6b0d09cbda572c759bbb26bf7b80f97f3bb7edfe8d20eee",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "28dd80135ed8f97e285050fdda4c94ef69fbd6d4a054bc12a1bea621fc16da16",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "272d89532f7750de0ec54880ef50583fd57fa3a88266b576c74fef9b00b257be",
                "sources": [
                    "sm_100a/cake_dsv4_882aa5db5b958b34eb06_kernel.cu",
                    "sm_100a/cake_dsv4_882aa5db5b958b34eb06_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "1c36c865f96e2f1349b86dd09ee821cdcfefd85c8fd2cc932456c769b2e73a5e",
                "sources": [
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_kernel.cu",
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "99cfdce408c0b3b549735e753a993508a84eea7de3a67819c8d6dd7ccf5cd292",
                "sources": [
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_kernel.cu",
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "55dccc8794eae61e55c578e7e218eafbe0d44c48619046e1de6a207e6b052f17",
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
                "identity": "f8d312ced8ca06f8b6004d60c3775a5380bc57afe8d60b99c475d513a9d8b450",
                "sources": [
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_kernel.cu",
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "580a36953060165ada00a47fe5d7297057ff1f8ca64e194e6eff53dd02288a76",
                "sources": [
                    "common/cake_dsv4_bf16_h64_prefill_kernel.cu",
                    "common/cake_dsv4_bf16_h64_prefill_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9c07bfcbcd088bb5da207258431bff9d67fa2ac14298d49376c49aea8f78ca4f",
                "sources": [
                    "sm_100a/cake_dsv4_1c67226e2adc46c7cf30_kernel.cu",
                    "sm_100a/cake_dsv4_1c67226e2adc46c7cf30_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "c611e26f97ff947f9f37ec2a0949a36cc8147695c8a1bdeed80e6e5d37ca3d07",
                "sources": [
                    "sm_100a/cake_dsv4_8088fd03cbf2d583060b_kernel.cu",
                    "sm_100a/cake_dsv4_8088fd03cbf2d583060b_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "aeb3b11e56ccf54bee7226d3f29767d29f53cb8edeed7e5bda58642f79b7acf8",
                "sources": [
                    "sm_100a/cake_dsv4_4290524eda9fb82fc598_kernel.cu",
                    "sm_100a/cake_dsv4_4290524eda9fb82fc598_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "feb617ca836382c74aa05b0860c890f6743154bd063c94a3f4be7936186a795d",
                "sources": [
                    "sm_100a/cake_dsv4_996e197a8a211fac151a_kernel.cu",
                    "sm_100a/cake_dsv4_996e197a8a211fac151a_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "cff402bbb30b363305a6c677bce562fa7d5571b81d018306167ead9a6bab8de3",
                "sources": [
                    "sm_100a/cake_dsv4_8e715cce80e191e2fed8_kernel.cu",
                    "sm_100a/cake_dsv4_8e715cce80e191e2fed8_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bb76e81663c9f212227bf420fcd6c4ff458da3ee04a830ed8101dc7a2ea20b50",
                "sources": [
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_kernel.cu",
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "51dd3bc90186c1f9bfe1e39b2613e0f784dd7fe1c1dbb149f5e508196810f6d8",
                "sources": [
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_multi_tile_kernel.cu",
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_multi_tile_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ede2d6d04e2be39eaf99038c0928e8d4ba1fe18d3067975c0c3ad5fbe9dbc530",
                "sources": [
                    "common/cake_dsv4_fp8_h64_source_exact_kernel.cu",
                    "common/cake_dsv4_fp8_h64_source_exact_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "fc713436f16c21a2526e4fbcbf4c140ebae3ed4fd6064cf5994534ca88797417",
                "sources": [
                    "sm_100a/cake_dsv4_94ff0e8b12ad7c552b63_kernel.cu",
                    "sm_100a/cake_dsv4_94ff0e8b12ad7c552b63_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "66cc8b509e4119cfe999d86e215440e1bdf10fc57958e5c31b4cd19cd465d823",
                "sources": [
                    "common/cake_dsv4_fp8_lowhead_h64_kernel.cu",
                    "common/cake_dsv4_fp8_lowhead_h64_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "6e81b75439ee0432073a4acaa813d21e3986485c9081c54b388d28e72672c705",
                "sources": [
                    "sm_100a/cake_dsv4_fcdb4b752ad083f97527_kernel.cu",
                    "sm_100a/cake_dsv4_fcdb4b752ad083f97527_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d92d31bca740edd94b0470dae30c7f136e1cbcc215b03cf95981999850894ed2",
                "sources": [
                    "sm_100a/cake_dsv4_bdf700268ba7384e2090_kernel.cu",
                    "sm_100a/cake_dsv4_bdf700268ba7384e2090_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "cc43bb8e51a0f515d6b86a5d5027a00dd7b2374a72ef252e952971a5f8ac7abf",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_25ccd210871ec64f56c4_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_25ccd210871ec64f56c4_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6c453fb76ec45f64160a239513244ffe682902a1462e048df0e729fe4cfe1f29",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_41129d29a3b3c6bb72ae_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_41129d29a3b3c6bb72ae_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "e4e8a51ffb16ace18d2e235eb7d03c749bb69ba32e23a1a8445217f9d8581748",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_80ecbd26e4d5abfeb195_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_80ecbd26e4d5abfeb195_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2f5fc9abd94f88af3e794fdc0fd3a5cec89ab4ceb50ead294ba0c7a082d62ed0",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_cf1e703c0218e8bcf5dd_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_cf1e703c0218e8bcf5dd_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "89c949b56e39597c83e2fc2fac87c29ad07a69687f2f7c44881b4144d0278161",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_391d8f29d8a385d7160a_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_391d8f29d8a385d7160a_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a53988b255fd7c9b955ac778a2bff6dfd24d5b1c993e38688bcaa0009d67376f",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_a5261db1f3cbbd407ff9_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_a5261db1f3cbbd407ff9_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9ea70546d7e33d438a1bd94e06f1ed446a59e94ff0f849c705d82b99f1ded0ef",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_68fbc091c87be711be5b_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_68fbc091c87be711be5b_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "536ee44585eac6da4fd948547c66475b0bfd4b04d3f859542a7fe59d2d643c19",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_23e63e46867139f9b7df_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_23e63e46867139f9b7df_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "05a570b04b5e79bcd459c471a2a25b4659840eedfbd31309440eeb26ed54545e",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_7bde67988374d5f8af6f_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_7bde67988374d5f8af6f_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5016269ebe71c7862d3c4322696fbc0880d2ccfe1b89902b01c47fcae733013f",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_bd996d63cddc9576dd79_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_bd996d63cddc9576dd79_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4eaf4e95b8787e5ad319e000eae7836ff6f022562d97c5e148b29f707769e097",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_0a34c729bb35e1e3d7ec_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_0a34c729bb35e1e3d7ec_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "80db91de4d55421e36be56d8958f09bd117b92f69c95f1eb64dc3584094551bf",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_4a5a09dd892c9044c95d_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_4a5a09dd892c9044c95d_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4626d538d7a1bbacc28c01ddf889f9444cf14d6efcbf6269c4690b3c8652b82d",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_b1b667a5de16f73c74b7_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_b1b667a5de16f73c74b7_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "327ba5d93180119f61d3b92bff12c2e5665d143b57005290439ff7b0125324b7",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_a00745ad5c773d13ef49_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_a00745ad5c773d13ef49_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ba735091c9122d840c9f224070cf2e9aba20446cf5e9b78ff2b34b619c88af19",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_4519fb382db814f4d592_kernel.cu",
                    "sm_100a/cake_dsv4_nvfp4_4519fb382db814f4d592_binding.cu",
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
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "94742becd5cc4a933335099cce31fb45d945ec99d3e80d2ac4c1a8fe2ecc0701",
                "sources": [
                    "sm_103a/cake_dsv4_304e0f834046e530860a_kernel.cu",
                    "sm_103a/cake_dsv4_304e0f834046e530860a_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "be6cb2807cc3f15e3e63d5ad9539abaad4d84d0ed092fcf9c951a0de0a5583aa",
                "sources": [
                    "sm_103a/cake_dsv4_c6c1cb5fde431e8906ad_kernel.cu",
                    "sm_103a/cake_dsv4_c6c1cb5fde431e8906ad_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "c39c29f285eedabe6b0f216949ebbe29ec63bee1d5a66656778b4eebf4a17058",
                "sources": [
                    "sm_103a/cake_dsv4_9459f7d9f719c6ca80e8_kernel.cu",
                    "sm_103a/cake_dsv4_9459f7d9f719c6ca80e8_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3058315c3c2fc4e5cc2f96db47f72d5e378233402d75427c9250bcc6b9bb2227",
                "sources": [
                    "common/cake_dsv4_bf16_h128_swa128_kernel.cu",
                    "common/cake_dsv4_bf16_h128_swa128_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "b88b762d209cc80c7dc32f2cfe93026e2f95036498b901b387731ccac8868a17",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "84fcc249e9f11173c6b0d09cbda572c759bbb26bf7b80f97f3bb7edfe8d20eee",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "28dd80135ed8f97e285050fdda4c94ef69fbd6d4a054bc12a1bea621fc16da16",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "e087ec3daf92cc382e6da9ed09369a08968d34625d9438b61960baaff0e40c51",
                "sources": [
                    "sm_103a/cake_dsv4_9cf54f1754fbab3f5669_kernel.cu",
                    "sm_103a/cake_dsv4_9cf54f1754fbab3f5669_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "1c36c865f96e2f1349b86dd09ee821cdcfefd85c8fd2cc932456c769b2e73a5e",
                "sources": [
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_kernel.cu",
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "99cfdce408c0b3b549735e753a993508a84eea7de3a67819c8d6dd7ccf5cd292",
                "sources": [
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_kernel.cu",
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "55dccc8794eae61e55c578e7e218eafbe0d44c48619046e1de6a207e6b052f17",
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
                "identity": "f8d312ced8ca06f8b6004d60c3775a5380bc57afe8d60b99c475d513a9d8b450",
                "sources": [
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_kernel.cu",
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "580a36953060165ada00a47fe5d7297057ff1f8ca64e194e6eff53dd02288a76",
                "sources": [
                    "common/cake_dsv4_bf16_h64_prefill_kernel.cu",
                    "common/cake_dsv4_bf16_h64_prefill_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "370f030bbb7d8e3174accdaca98c18564231e6e4e193849ed4dc908cfa164995",
                "sources": [
                    "sm_103a/cake_dsv4_fba99757a8616176fffe_kernel.cu",
                    "sm_103a/cake_dsv4_fba99757a8616176fffe_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "61e3c46035ae2d1a61c71b8bf0fd7308a5b83f552162f350483ad6158d68f3f9",
                "sources": [
                    "sm_103a/cake_dsv4_8dd329553d3418caecb6_kernel.cu",
                    "sm_103a/cake_dsv4_8dd329553d3418caecb6_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d521f058cdf1e1e45a7a45cb2bf47f52caea8abef8b69ab49495904d6ad45cd6",
                "sources": [
                    "sm_103a/cake_dsv4_336c5113089a9e7c3e9e_kernel.cu",
                    "sm_103a/cake_dsv4_336c5113089a9e7c3e9e_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "a9b0a39f522f690b57f6172f8c0b619a871d0ad39d3400ad0e1ad2ca40ea00dc",
                "sources": [
                    "sm_103a/cake_dsv4_c58b174e83407b66ec0b_kernel.cu",
                    "sm_103a/cake_dsv4_c58b174e83407b66ec0b_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "dba9c7f853342ac8f94a548147b365a1100cc2d49ded2d531501156c630dfcaf",
                "sources": [
                    "sm_103a/cake_dsv4_f0fa94e529de62dd0ac8_kernel.cu",
                    "sm_103a/cake_dsv4_f0fa94e529de62dd0ac8_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "bb76e81663c9f212227bf420fcd6c4ff458da3ee04a830ed8101dc7a2ea20b50",
                "sources": [
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_kernel.cu",
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "51dd3bc90186c1f9bfe1e39b2613e0f784dd7fe1c1dbb149f5e508196810f6d8",
                "sources": [
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_multi_tile_kernel.cu",
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_multi_tile_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ede2d6d04e2be39eaf99038c0928e8d4ba1fe18d3067975c0c3ad5fbe9dbc530",
                "sources": [
                    "common/cake_dsv4_fp8_h64_source_exact_kernel.cu",
                    "common/cake_dsv4_fp8_h64_source_exact_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "11f60425d281b568d5ba59e3a06d77cc8a5b7f738d698807a70d0a213c301715",
                "sources": [
                    "sm_103a/cake_dsv4_31c4530bb8032b0b0ffc_kernel.cu",
                    "sm_103a/cake_dsv4_31c4530bb8032b0b0ffc_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "66cc8b509e4119cfe999d86e215440e1bdf10fc57958e5c31b4cd19cd465d823",
                "sources": [
                    "common/cake_dsv4_fp8_lowhead_h64_kernel.cu",
                    "common/cake_dsv4_fp8_lowhead_h64_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7b0a0b45b06e1b2a5a51c5263c1e55332dfdd87e8b83f5c7d96bd7c91c35b625",
                "sources": [
                    "sm_103a/cake_dsv4_cec56313e7db4039e904_kernel.cu",
                    "sm_103a/cake_dsv4_cec56313e7db4039e904_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "be35f93481488745576192f8402503db32277dc846673fe7e5e92bb62255c877",
                "sources": [
                    "sm_103a/cake_dsv4_d99ab746efb5e8e65525_kernel.cu",
                    "sm_103a/cake_dsv4_d99ab746efb5e8e65525_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "08b5530a122f0d327390ec47a886cf00e3d8861c4c22b3bdca2a32b6fd923e87",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_9884f068105172c7cd44_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_9884f068105172c7cd44_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "61af841f5ebb69bbc5ea3cc44fe2274e60e72438fd1697871be5a8ea923adae5",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_ec607c36cebb9e6c8203_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_ec607c36cebb9e6c8203_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b2cf32596c9c10001d310ceb895b8b0f6d80f027008ee715ac120cbad4016f5e",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_c7252582e0cb3daa3674_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_c7252582e0cb3daa3674_binding.cu",
                ],
            },
            "nvfp4_decode_pv_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a71b9f2e0395242f3c80c9d75b75fe15477613cd0a775864da5be26783f7c3da",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_da01174fe2acd0a1e6fa_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_da01174fe2acd0a1e6fa_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3b9fd81a93d63f6688c111dbf51964fb47c3e74041ceb6d50bc2c81ca1e1c176",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_cf6c97a0107a1f15fc89_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_cf6c97a0107a1f15fc89_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ee60e8cb1ce74801b8947f641c2b0b4ca6f76aab820a1ca937956b7a05e11c53",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_2bfe3d233b63da6ecf13_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_2bfe3d233b63da6ecf13_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n16_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2905bec1d98622c92be1f85e3facbe79b3fed0fe2bff1ca6d2cd25a8ee98c12a",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_0acceb17bb60dce5aabe_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_0acceb17bb60dce5aabe_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a8b5dc049e2093eb318824a020b6eba9c0cf59619139b6a32ba83905bf92008d",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_1543a4025fd7a3698acc_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_1543a4025fd7a3698acc_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b5e801e2ca95ce342859978b81e13906b4d1a4c4a1bcdc52fb4a33c291211423",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_bf522c63cbbe18ff95b8_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_bf522c63cbbe18ff95b8_binding.cu",
                ],
            },
            "nvfp4_decode_swap_n32_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0d130b331e607554c620201407f4a90b3b2a049f35b014e917571c9a5cd72f09",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_b9f9f100d5d98292546e_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_b9f9f100d5d98292546e_binding.cu",
                ],
            },
            "nvfp4_decode_t64_n64_oc1": {
                "arg_plan": _PLAN_NVFP4_G4 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "8e3bf278483d0892ec9c0d75f8817943b4799497a222b922af13009c221814f3",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_a65cb7961dba15898785_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_a65cb7961dba15898785_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc1": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "eb7ee773890481be87f9c37bd15271e2a5dfc13aba02ee1cbc897a5443b5e337",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_6d71bc683b9533041506_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_6d71bc683b9533041506_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc2": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "771e37caf2af776778713563e9961f5131861431ffeed1f376273616180013da",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_16e0ffb22ddfce6f3c6f_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_16e0ffb22ddfce6f3c6f_binding.cu",
                ],
            },
            "nvfp4_decode_tile_oc4": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7d45a62e833eb8547aea73767316b0036dfb49c978b372a9ae254c0d3da993f8",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_4666d25b83dc6f03da20_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_4666d25b83dc6f03da20_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4a1dd5a8701377ce2f901987e1c8e6a60499365ccd27a432ad9c9c42b76bfa04",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_f9c190de33a99c3fd6ff_kernel.cu",
                    "sm_103a/cake_dsv4_nvfp4_f9c190de33a99c3fd6ff_binding.cu",
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
