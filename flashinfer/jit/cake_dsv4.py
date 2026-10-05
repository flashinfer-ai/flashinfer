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
_ARCH_REGISTRATIONS = {
    "sm_100a": {
        "variants": {
            "bf16_h128_prefill_v42": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "0ea9b1b2ec5e19d52cf52bef536d66bd962af241d526450f007d90d5f99854f8",
                "sources": [
                    "sm_100a/cake_dsv4_96dc8329e4cf4f62dea7_kernel.cu",
                    "sm_100a/cake_dsv4_96dc8329e4cf4f62dea7_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "90b7dbb54aeecc3abb6cc801656328f1773ef0ce872c6fbc1f5e05f853c61db0",
                "sources": [
                    "sm_100a/cake_dsv4_705e276df80003bd0d56_kernel.cu",
                    "sm_100a/cake_dsv4_705e276df80003bd0d56_binding.cu",
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
                "identity": "2911434b1f8a9e3840573a315bc8f02c8e9dedf01c770693c29cecaa8a3fd471",
                "sources": [
                    "common/cake_dsv4_bf16_h128_swa128_kernel.cu",
                    "common/cake_dsv4_bf16_h128_swa128_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "bec670c0b6ade7ff84697893f27dbcdcec65b8de26d1b7ac4c8e615d0c3d053a",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "4a8292caf3c8e2d97386d2af79f072604961faa4090d2ab05f3dd1af4fb618bc",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_row_first_vsplit_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "4794a4cd094d0ecfcde824c697ff40716ba4063b63471366c884c59068c26af4",
                "sources": [
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_kernel.cu",
                    "common/cake_dsv4_bf16_h128_topk128x_split4_sm100_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "35b01fab37c8e1a3b90533036e3017d45f41b06453f24c43918138801dc03f4b",
                "sources": [
                    "sm_100a/cake_dsv4_c25384d0c64b44dea081_kernel.cu",
                    "sm_100a/cake_dsv4_c25384d0c64b44dea081_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3f40d29238fd5d99e79e93471b17e1625436437ed5220c7df413bc0573cee24b",
                "sources": [
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_kernel.cu",
                    "common/cake_dsv4_bf16_h16_h32_swa128_v44_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "cbe1111d67fd366065bdda1c14a8ce7dfb68091f87fb8c2adbaea71d76aae04f",
                "sources": [
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_kernel.cu",
                    "common/cake_dsv4_bf16_h32_topk128x_early_v47_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "47a93f4d0b217bda91e70edb703b759d3410f2c82eba787dace58161abb5bac4",
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
                "identity": "c154172f4f343b8e5473f79a4a9463b51ce7e39cf37795b1944f03faf12881cf",
                "sources": [
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_kernel.cu",
                    "common/cake_dsv4_bf16_h64_guard_q_tma_batch_r25_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "83f91ad2a76fb71e6a8ff5483f8399c6bd7aff79c954e6a1cdc47ef562c8a83c",
                "sources": [
                    "common/cake_dsv4_bf16_h64_prefill_kernel.cu",
                    "common/cake_dsv4_bf16_h64_prefill_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9734fdf27f0c0bc60732b98c8db04fd259191c37348df583256b1596c7810b05",
                "sources": [
                    "sm_100a/cake_dsv4_d0f9e9c0533a56645099_kernel.cu",
                    "sm_100a/cake_dsv4_d0f9e9c0533a56645099_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "c8e725fc889732aa30575657c146abe2a4305f5b86041e40955f466e783e1f5e",
                "sources": [
                    "sm_100a/cake_dsv4_e8b01d3cb6d7c3def799_kernel.cu",
                    "sm_100a/cake_dsv4_e8b01d3cb6d7c3def799_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "8b4d137d502b75563e66c24f963518ca4cc73a54c1e386187931e773f807206e",
                "sources": [
                    "sm_100a/cake_dsv4_770afce13624cba9b5f2_kernel.cu",
                    "sm_100a/cake_dsv4_770afce13624cba9b5f2_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "bff7fd443e2e45c5109485523163ab0597de9d0132edc6bc680f6da901851123",
                "sources": [
                    "sm_100a/cake_dsv4_595383408b68af397ae0_kernel.cu",
                    "sm_100a/cake_dsv4_595383408b68af397ae0_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b9a888ab3b63f80518f0e583f9f787fada4ec08d6dfe76bcc1bf25aa7a59f7f7",
                "sources": [
                    "sm_100a/cake_dsv4_cd8cfc679fe1ee72a4f6_kernel.cu",
                    "sm_100a/cake_dsv4_cd8cfc679fe1ee72a4f6_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2708b1de2e598a36267c9ce7b334600cfd1748f97bc96e773e2a2bb072a07ab3",
                "sources": [
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_kernel.cu",
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "8a3bf056f1333d54376409d0cfd68c1e206d1c569b84a61f7599259bc11a2d04",
                "sources": [
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_multi_tile_kernel.cu",
                    "common/cake_dsv4_fp8_h64_prefill_source_persistent_m64_multi_tile_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b4eeb3a0079f3eb90dbc9e1ef832cca41fe81ee990b7f87996acf9de2e26cc3e",
                "sources": [
                    "common/cake_dsv4_fp8_h64_source_exact_kernel.cu",
                    "common/cake_dsv4_fp8_h64_source_exact_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0f744df60a9b4911dedd2f0736fbbd4a9b7247c58cad25b73295230359353890",
                "sources": [
                    "sm_100a/cake_dsv4_3ca2853eb9998a5df58b_kernel.cu",
                    "sm_100a/cake_dsv4_3ca2853eb9998a5df58b_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "c967b8860ac424cecbcec44554df96a5615bdc7abb744a82a0558c287108ada0",
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
                "identity": "ee0ee84c4073833efeab3623700b9053ba2a7f21797a354bdf9e6ceeccf7fed8",
                "sources": [
                    "sm_100a/cake_dsv4_2eda54a17eb6e6d70f22_kernel.cu",
                    "sm_100a/cake_dsv4_2eda54a17eb6e6d70f22_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "581f2a8eb5056352905ca9d360bc03481bd04adafbb4296d056c87b363e8559f",
                "sources": [
                    "sm_100a/cake_dsv4_c47613faf53798343d4b_kernel.cu",
                    "sm_100a/cake_dsv4_c47613faf53798343d4b_binding.cu",
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
                "identity": "0847fd76f7eca4cce33dea67dca9689fe9315ea5473e19aae7023359c1c5dd06",
                "sources": [
                    "sm_103a/cake_dsv4_78f5d1bb05a07d347bbe_kernel.cu",
                    "sm_103a/cake_dsv4_78f5d1bb05a07d347bbe_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "4cf8fa169c7cc3cd3aba47d0b24ef113eb1dc1016b77ae405d2b7c26cafe5c5a",
                "sources": [
                    "sm_103a/cake_dsv4_6b569eae6fbb6ad6fdea_kernel.cu",
                    "sm_103a/cake_dsv4_6b569eae6fbb6ad6fdea_binding.cu",
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
                "arg_plan": _PLAN_BF16_H128_SWA_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "8ea819b4d11a6d2e9dacc53aefe51e437f343153c2786df458f607178ce9957f",
                "sources": [
                    "sm_103a/cake_dsv4_af95a739e06bdfec49e2_kernel.cu",
                    "sm_103a/cake_dsv4_af95a739e06bdfec49e2_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "8e8e0b2065665e5524c04b16127f328e438ea1daf2c1623fcbc4bc59f72d9fa8",
                "sources": [
                    "sm_103a/cake_dsv4_e8ce5523eca0746481e4_kernel.cu",
                    "sm_103a/cake_dsv4_e8ce5523eca0746481e4_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "d02fd31d93a71157a24b2ccc8c58e689ca77879f26f99d36c1257c3ff5a4c999",
                "sources": [
                    "sm_103a/cake_dsv4_5fff196dca32aec63828_kernel.cu",
                    "sm_103a/cake_dsv4_5fff196dca32aec63828_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "139695deadc3ac383bfa77889af00382929c5744aeb792d8a4805055d1c4e855",
                "sources": [
                    "sm_103a/cake_dsv4_5efc31945dfe02568009_kernel.cu",
                    "sm_103a/cake_dsv4_5efc31945dfe02568009_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "c25d6cefaf23093dff9b5761ece2b469689b755258ad57803a2f3376e0e8076e",
                "sources": [
                    "sm_103a/cake_dsv4_40974459bee6dabc37fb_kernel.cu",
                    "sm_103a/cake_dsv4_40974459bee6dabc37fb_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b207d9b29b8a0f884f5f4e35317ada98e2b16eed186bb18e7ad91473ff061cbe",
                "sources": [
                    "sm_103a/cake_dsv4_93b018f4b4ae446494f1_kernel.cu",
                    "sm_103a/cake_dsv4_93b018f4b4ae446494f1_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "af6abb29bd7a145894afd8ffa567ba383c19032e0aac0e09a2c68a7fdde37039",
                "sources": [
                    "sm_103a/cake_dsv4_e92c732e4b632b82ea06_kernel.cu",
                    "sm_103a/cake_dsv4_e92c732e4b632b82ea06_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "17e7d07860a4af915b88c747249c005fc83fbb76b0c76fdefbbf532abde82600",
                "sources": [
                    "sm_103a/cake_dsv4_dd2acd0aa903a7644013_kernel.cu",
                    "sm_103a/cake_dsv4_dd2acd0aa903a7644013_binding.cu",
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
                "identity": "6b833d71415773820dc7244348a4471a964f627d095f391b3404e547baece4c7",
                "sources": [
                    "sm_103a/cake_dsv4_03de55aa1bebf3fed8e9_kernel.cu",
                    "sm_103a/cake_dsv4_03de55aa1bebf3fed8e9_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "33531914e3514a390bb9a2e4bcb397187c386f20e18dc7b2f0aff77250616ae3",
                "sources": [
                    "sm_103a/cake_dsv4_3f4c022671dd451f5d78_kernel.cu",
                    "sm_103a/cake_dsv4_3f4c022671dd451f5d78_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "777b5f6c0a294899743ad88533b25f928a5bdd7377f4fe93cd5dd7867ef3018f",
                "sources": [
                    "sm_103a/cake_dsv4_781ec656d1820e8431fb_kernel.cu",
                    "sm_103a/cake_dsv4_781ec656d1820e8431fb_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6829364ee99ec15d6b8e601d656d660279042ab9ebd04006059cdc11bb526818",
                "sources": [
                    "sm_103a/cake_dsv4_35591811d330ad804c64_kernel.cu",
                    "sm_103a/cake_dsv4_35591811d330ad804c64_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2499ed099c731d4b935bf01e23aa41418900fb641db8793902344d14331fbef9",
                "sources": [
                    "sm_103a/cake_dsv4_aa54d0db80d37dbf14ff_kernel.cu",
                    "sm_103a/cake_dsv4_aa54d0db80d37dbf14ff_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "a9d843dd0daf1bb010f2529a8560745b1f1e4980a71b1f6e73db86ef28a15475",
                "sources": [
                    "sm_103a/cake_dsv4_a6974891b74fdd2967a5_kernel.cu",
                    "sm_103a/cake_dsv4_a6974891b74fdd2967a5_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f7fe10afe92184becf6ff8a1114c865cbceed9f79f1c3ea8474f99880149e475",
                "sources": [
                    "sm_103a/cake_dsv4_7f08dc0e0931cd81aa14_kernel.cu",
                    "sm_103a/cake_dsv4_7f08dc0e0931cd81aa14_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "53e20b378260733dc07033bd08efda076a1ef2f8ab996e12336dd6c624da8ea4",
                "sources": [
                    "sm_103a/cake_dsv4_bf3977abb24a7448b117_kernel.cu",
                    "sm_103a/cake_dsv4_bf3977abb24a7448b117_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0a108d23940ad1444c9ac8836637b935601f0960b4b5ef13405daddde249ca8a",
                "sources": [
                    "sm_103a/cake_dsv4_f89844a692440efb35e9_kernel.cu",
                    "sm_103a/cake_dsv4_f89844a692440efb35e9_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "dd39bb01412444d9661393ebe19ad1407ba3c1ad2f956600bfe0a53c56ac259a",
                "sources": [
                    "sm_103a/cake_dsv4_c49cf0b385b49ac67002_kernel.cu",
                    "sm_103a/cake_dsv4_c49cf0b385b49ac67002_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT_TOKENS_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2910ae9c6c87f4cfe37a3c9d6454a28343850efd1f3cfaa4696bdbd5330cf039",
                "sources": [
                    "sm_103a/cake_dsv4_6399a8c12398f2c810fd_kernel.cu",
                    "sm_103a/cake_dsv4_6399a8c12398f2c810fd_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "c9c228c8559b1a9df97d4fee3c6c6dee173bbbf1673d28ba52bda45414f6232e",
                "sources": [
                    "sm_103a/cake_dsv4_413ead7e285244a38eea_kernel.cu",
                    "sm_103a/cake_dsv4_413ead7e285244a38eea_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "b7277d2f23949f94e4cf735ee25fa74f338b4a76c62b1dd5fabae61f72bb7023",
                "sources": [
                    "sm_103a/cake_dsv4_2b9df22b41894574cc74_kernel.cu",
                    "sm_103a/cake_dsv4_2b9df22b41894574cc74_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD_QLAYOUT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6782dcfc3a1171abc1ecefdb3f176069eee8ecff2bcb482e22dcdc58c6358c0c",
                "sources": [
                    "sm_103a/cake_dsv4_ac2ee20d01ebf2ca8207_kernel.cu",
                    "sm_103a/cake_dsv4_ac2ee20d01ebf2ca8207_binding.cu",
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
