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

_PLAN_BF16_H64_GUARD = (
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

_PLAN_FP8_H64_M64 = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
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

_PLAN_H8_H16_SOURCE_EXACT = (
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

_PLAN_NVFP4_MERGE = (
    ("buffer", "partial_O"),
    ("buffer", "partial_lse"),
    ("buffer", "O"),
    ("buffer", "lse_out"),
    ("parameter", "num_heads"),
    ("parameter", "num_splits"),
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
                "identity": "b3715376b8c6e27f09d941cd1d259130d7a54170640d87e0b534d83708ead0fd",
                "sources": [
                    "sm_100a/cake_dsv4_314d943a39ab7eade2c8_kernel.cu",
                    "sm_100a/cake_dsv4_314d943a39ab7eade2c8_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "24344094f11b37ec4b591eed3a095f38440fc3d66ab7d03357d47093633b9cbb",
                "sources": [
                    "sm_100a/cake_dsv4_d994492f2dfb135a7dbf_kernel.cu",
                    "sm_100a/cake_dsv4_d994492f2dfb135a7dbf_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "1b1cc19373e52f67981ef2cff298918c5579ee69afa319f8c1ed8c1c5cf4ee7c",
                "sources": [
                    "sm_100a/cake_dsv4_d906a03bd66a7488fc29_kernel.cu",
                    "sm_100a/cake_dsv4_d906a03bd66a7488fc29_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "82976f52e6a753f3e2fe9a76552492077e07345543038a013d52d92ae7d037a2",
                "sources": [
                    "sm_100a/cake_dsv4_57e7b93def70f2be356f_kernel.cu",
                    "sm_100a/cake_dsv4_57e7b93def70f2be356f_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "f2200c038a3de20ae07fe52f1a3db30d71941feff6d47fe4db8cd1a45e9d9d44",
                "sources": [
                    "sm_100a/cake_dsv4_e6d521df78ecb146fd49_kernel.cu",
                    "sm_100a/cake_dsv4_e6d521df78ecb146fd49_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "5609e5563448b459fc62af27735acd289478d73d4dfa97cc276009a78b472705",
                "sources": [
                    "sm_100a/cake_dsv4_e1660a7f44d156286fea_kernel.cu",
                    "sm_100a/cake_dsv4_e1660a7f44d156286fea_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "11243ea83b1c9138a3751dda4de6de1d9ba24712eddce8a30c46fa7027f09e4f",
                "sources": [
                    "sm_100a/cake_dsv4_ba9ed6a6233761682cc3_kernel.cu",
                    "sm_100a/cake_dsv4_ba9ed6a6233761682cc3_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "dce42787024ee775d92b9917c6aef6c75769aa31d20cc420a5986533fae9dc29",
                "sources": [
                    "sm_100a/cake_dsv4_a96a3de25becc6c0a422_kernel.cu",
                    "sm_100a/cake_dsv4_a96a3de25becc6c0a422_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7a08a9d6d26f32e66a62402142d1fdb47a1358542fe79be3c76445e4826c6824",
                "sources": [
                    "sm_100a/cake_dsv4_c0e3d0af9a266850de48_kernel.cu",
                    "sm_100a/cake_dsv4_c0e3d0af9a266850de48_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "fe3d4502bb2dfe17a4a4d94fe8f02215e9ee7ae48fd4c70ef3e29326fe7e3640",
                "sources": [
                    "sm_100a/cake_dsv4_422eafcc723b03b8e044_kernel.cu",
                    "sm_100a/cake_dsv4_422eafcc723b03b8e044_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2e14912ba8d257e634816d04ff5cce6abc380b94e8b771da784bb8e3d97e2b53",
                "sources": [
                    "common/cake_dsv4_bf16_h64_compressed_reduce_kernel.cu",
                    "common/cake_dsv4_bf16_h64_compressed_reduce_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "e1b220a2d15a79b981a5e717f4d92d99f416852bcea16cb3c6d377e39c9d9671",
                "sources": [
                    "sm_100a/cake_dsv4_3b3789d6356f15f9cbe3_kernel.cu",
                    "sm_100a/cake_dsv4_3b3789d6356f15f9cbe3_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "53b1e34a7eaa5c285ff343627132fc0cf9a943a2f7860e2bc10cb45797d989f6",
                "sources": [
                    "sm_100a/cake_dsv4_9526f80c89f2b5ef4c6c_kernel.cu",
                    "sm_100a/cake_dsv4_9526f80c89f2b5ef4c6c_binding.cu",
                ],
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "14152b76d08f7df3fc1e57ac1a9c48ee800539333b92cd59c65ec148f1025558",
                "sources": [
                    "sm_100a/cake_dsv4_bf2bffa3152d5dce396f_kernel.cu",
                    "sm_100a/cake_dsv4_bf2bffa3152d5dce396f_binding.cu",
                ],
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d9b38ca3f8c4bf622b0e37322c11c66c33889b7444792e64a4fbb398f0671e71",
                "sources": [
                    "sm_100a/cake_dsv4_88d5662b6147fe74ed0c_kernel.cu",
                    "sm_100a/cake_dsv4_88d5662b6147fe74ed0c_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "db3a18b88f9a22de1bbaf8e18f45b1281ce7ed5f69a7a102ea55af5e8ca5cf24",
                "sources": [
                    "sm_100a/cake_dsv4_4ac65f00a4a032c10ecb_kernel.cu",
                    "sm_100a/cake_dsv4_4ac65f00a4a032c10ecb_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "2c813be0fac81518383683a7faa658c9e452fb1f1966745bdbc4306327d3dee1",
                "sources": [
                    "sm_100a/cake_dsv4_2dd31513440dbd1a0ef2_kernel.cu",
                    "sm_100a/cake_dsv4_2dd31513440dbd1a0ef2_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4dce24ad4f95460dfa2ecc045bf899c67c4e7927a1bb362b2f3cf1d5d24ceb26",
                "sources": [
                    "sm_100a/cake_dsv4_77c162685d381298b786_kernel.cu",
                    "sm_100a/cake_dsv4_77c162685d381298b786_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ca35d52a3fbfe936fa85d8f8e4c62b2f624c9b1043519640803738047d5c17c3",
                "sources": [
                    "sm_100a/cake_dsv4_119728d98fe411f18a22_kernel.cu",
                    "sm_100a/cake_dsv4_119728d98fe411f18a22_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "1dc5dc743aca6bb61cb8956a6906a9cb03997d40adfc53f5eb5308dc16fd279a",
                "sources": [
                    "sm_100a/cake_dsv4_5e5d6648b7bd42bcd0d6_kernel.cu",
                    "sm_100a/cake_dsv4_5e5d6648b7bd42bcd0d6_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7777f8fda7eda670a2733c023b36fd5c05a0cef563e051a8a715ed22d5cc8b79",
                "sources": [
                    "sm_100a/cake_dsv4_fbf138c989dfde9f764f_kernel.cu",
                    "sm_100a/cake_dsv4_fbf138c989dfde9f764f_binding.cu",
                ],
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "3640ee1967a674a977c4ce19a91cfc1c0095d43d59c6ea1fea8cd6c9157331cc",
                "sources": [
                    "sm_100a/cake_dsv4_7771bcc5c13a5ff83890_kernel.cu",
                    "sm_100a/cake_dsv4_7771bcc5c13a5ff83890_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "ef787d9bc5f14ef9eab0e77e05e3e8a3a350341db6b57f23816515a16a92b7d0",
                "sources": [
                    "sm_100a/cake_dsv4_fb6409904b35ed0f7a1a_kernel.cu",
                    "sm_100a/cake_dsv4_fb6409904b35ed0f7a1a_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2a1fc4a24d65057c1db942da73a0d62ca1b02b8a1a1002b05532990638dd2aca",
                "sources": [
                    "sm_100a/cake_dsv4_75827bd38b443de66f67_kernel.cu",
                    "sm_100a/cake_dsv4_75827bd38b443de66f67_binding.cu",
                ],
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f605de7e00c3c688e91148365fd66d73eaee7c467ad023a697395cf45204f1c5",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_714aa5e3b2ca5132cdc5_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_714aa5e3b2ca5132cdc5_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "90a02f5af71ce598d3dae052bbf42b67f2e280c7985db15b865b1f6e5d38955a",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_d331a0051120b347292a_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_d331a0051120b347292a_binding.cu",
                ],
            },
            "nvfp4_decode_tile": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "f036930bb203015479f8b473e475793620b5cd5fc1a643f772c2b952e9560bdb",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_08b25e7f978c5a8e2da6_kernel_portable.cu",
                    "sm_100a/cake_dsv4_nvfp4_08b25e7f978c5a8e2da6_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "e26fd9c5705f3f7e4abbb8e5a3c384dec0c592cff0d19f952fcf428e291c42dc",
                "sources": [
                    "sm_100a/cake_dsv4_nvfp4_18fb28e3685c594fb355_kernel.cu",
                    "sm_100a/cake_dsv4_nvfp4_18fb28e3685c594fb355_binding.cu",
                ],
            },
            "split_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "014512652fd83c055323b34e9c4807829313cd35fbcde3a1d44cd87e89404265",
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
                "identity": "3c18335a89ae686e86a9d8f7e02bedfceda722cf8eca63b648d0bf1871fe4cf5",
                "sources": [
                    "sm_103a/cake_dsv4_a40bd49fee9655d8b1c6_kernel.cu",
                    "sm_103a/cake_dsv4_a40bd49fee9655d8b1c6_binding.cu",
                ],
            },
            "bf16_h128_prefill_v42_snake": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "4ca772248ca31846c587fafbac94c2da2efdf58a844e78832268403caca66abb",
                "sources": [
                    "sm_103a/cake_dsv4_d338c32a69477808a1ef_kernel.cu",
                    "sm_103a/cake_dsv4_d338c32a69477808a1ef_binding.cu",
                ],
            },
            "bf16_h128_split5_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "2a8255d11ab33428fbaa167fb8b2a10bf0267969da6ddfb81d9974cd7b62296f",
                "sources": [
                    "sm_103a/cake_dsv4_230143c89d42c6c89387_kernel.cu",
                    "sm_103a/cake_dsv4_230143c89d42c6c89387_binding.cu",
                ],
            },
            "bf16_h128_swa128": {
                "arg_plan": _PLAN_BF16_H128_SWA + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "55e52ad1bad7c5f6d9f0287db61de94bd0c95041a6cb2933c495784d302f9d68",
                "sources": [
                    "sm_103a/cake_dsv4_f89bc6561b38462e9538_kernel.cu",
                    "sm_103a/cake_dsv4_f89bc6561b38462e9538_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "8fdcc0ef01be0f9758a29d5556b2f7bae6864aa2ed8a79e56200ea722bb60c78",
                "sources": [
                    "sm_103a/cake_dsv4_af6f9d240794a942a124_kernel.cu",
                    "sm_103a/cake_dsv4_af6f9d240794a942a124_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "5bed0c801c9f03c5a4157722eca0768a8457156f1aab8d80ee1d002b00029ca5",
                "sources": [
                    "sm_103a/cake_dsv4_4c41fb8347f137d4d46f_kernel.cu",
                    "sm_103a/cake_dsv4_4c41fb8347f137d4d46f_binding.cu",
                ],
            },
            "bf16_h128_topk4x_v52": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "1ca21a0f7b4c6edfddb5340b989d3cc2964c772e7ecd16fc74bcea8088fd5eec",
                "sources": [
                    "sm_103a/cake_dsv4_125332c2066d89bd7610_kernel.cu",
                    "sm_103a/cake_dsv4_125332c2066d89bd7610_binding.cu",
                ],
            },
            "bf16_h16_h32_swa128_v44": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "181de2841cecdd15e69e910e6afec211743131dcf5fb7fed2023027d0570b314",
                "sources": [
                    "sm_103a/cake_dsv4_ae27071dabfc85fb47c1_kernel.cu",
                    "sm_103a/cake_dsv4_ae27071dabfc85fb47c1_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "456fc2e0220f173e148d85b2fd7b3d73c245f83a64a29847de39a0a29b8a183b",
                "sources": [
                    "sm_103a/cake_dsv4_6b7225f9e37a48354992_kernel.cu",
                    "sm_103a/cake_dsv4_6b7225f9e37a48354992_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9f675bcf369d90feaf54659043953dd0734c40b14e65b5cc2e8db3ad756b6b3f",
                "sources": [
                    "sm_103a/cake_dsv4_fb5ca25f5af355ef89b5_kernel.cu",
                    "sm_103a/cake_dsv4_fb5ca25f5af355ef89b5_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2e14912ba8d257e634816d04ff5cce6abc380b94e8b771da784bb8e3d97e2b53",
                "sources": [
                    "common/cake_dsv4_bf16_h64_compressed_reduce_kernel.cu",
                    "common/cake_dsv4_bf16_h64_compressed_reduce_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD + _SLAB + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "6991e3e7c4210570953f9cdc15c6293f15bf9a0e1798b739abee8df34c6d3551",
                "sources": [
                    "sm_103a/cake_dsv4_c967e7efe220160e28d3_kernel.cu",
                    "sm_103a/cake_dsv4_c967e7efe220160e28d3_binding.cu",
                ],
                "tma_workspace_bytes": 384,
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL + _SLAB + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "aae71071ae4aa66b4bdd00a852c061efda61475dc4a69cdf4b2bd2705f4a4661",
                "sources": [
                    "sm_103a/cake_dsv4_3e5a644d61420c9c7ea1_kernel.cu",
                    "sm_103a/cake_dsv4_3e5a644d61420c9c7ea1_binding.cu",
                ],
                "tma_workspace_bytes": 384,
            },
            "bf16_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT + _SLAB + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ad819a642ec063a95836b08029677d93d1c6516c4b096e3106f9032257a12bd0",
                "sources": [
                    "sm_103a/cake_dsv4_d23652aae3370c30d978_kernel.cu",
                    "sm_103a/cake_dsv4_d23652aae3370c30d978_binding.cu",
                ],
                "tma_workspace_bytes": 384,
            },
            "bf16_h8_swa128_v43": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "12c3d8e97fc54d26fe53f365da8b548e3cfe388c84a0101316c88a55a840ab0e",
                "sources": [
                    "sm_103a/cake_dsv4_fdbe4c4737d7f6dd1d9e_kernel.cu",
                    "sm_103a/cake_dsv4_fdbe4c4737d7f6dd1d9e_binding.cu",
                ],
            },
            "bf16_swa128_single_cta": {
                "arg_plan": _PLAN_BF16_SWA_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5b806976ca4dcb0b14105e2f1b06df0630452af2283a61256c8df96b0ec8b7e4",
                "sources": [
                    "sm_103a/cake_dsv4_9cf30aacb43b71034660_kernel.cu",
                    "sm_103a/cake_dsv4_9cf30aacb43b71034660_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": [
                    "--use_fast_math",
                    "-Xptxas=--register-usage-level=10",
                ],
                "identity": "72c980c50469dd2c8c22342b2384ccd6d6770535e77343c9aea15c9f1f99860f",
                "sources": [
                    "sm_103a/cake_dsv4_85ef6cd9a950ca02c484_kernel.cu",
                    "sm_103a/cake_dsv4_85ef6cd9a950ca02c484_binding.cu",
                ],
            },
            "fp8_h128_prefill_source_persistent_uniform": {
                "arg_plan": _PLAN_FP8_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4cbb2c362b859ca355d32bb1c802aabf58d88fd8f9ee36140273165754e8820c",
                "sources": [
                    "sm_103a/cake_dsv4_32e01d0e8961dfa26bf3_kernel.cu",
                    "sm_103a/cake_dsv4_32e01d0e8961dfa26bf3_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64": {
                "arg_plan": _PLAN_FP8_H64_M64 + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "9499459d9e38d660258217d59ae733e72c9421c68b93e53534379a4353cf4145",
                "sources": [
                    "sm_103a/cake_dsv4_bd3f29f66e3196dc9451_kernel.cu",
                    "sm_103a/cake_dsv4_bd3f29f66e3196dc9451_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a0c09ef4273aa62ee82bc9f0f221af205a14fbbc44509bbdf674ff321f5f66b6",
                "sources": [
                    "sm_103a/cake_dsv4_f55f69afe4ac3ec96817_kernel.cu",
                    "sm_103a/cake_dsv4_f55f69afe4ac3ec96817_binding.cu",
                ],
            },
            "fp8_h8_h16_source_exact": {
                "arg_plan": _PLAN_H8_H16_SOURCE_EXACT + _SLAB + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "d29ebd3f5584b9189fb165159f8117f93dcf919fd9b35bf6ddfccdbbb3f05593",
                "sources": [
                    "sm_103a/cake_dsv4_7a4b9556b3c84ebff6ce_kernel.cu",
                    "sm_103a/cake_dsv4_7a4b9556b3c84ebff6ce_binding.cu",
                ],
                "tma_workspace_bytes": 384,
            },
            "fp8_lowhead_h64": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "7ff0f4e03616e8a841b5726c9bae67e2ec29b56d21c7da962980cf1d0fba8d57",
                "sources": [
                    "sm_103a/cake_dsv4_9e1188b192c9c806d7c6_kernel.cu",
                    "sm_103a/cake_dsv4_9e1188b192c9c806d7c6_binding.cu",
                ],
            },
            "fp8_lowhead_one_partition": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "936e66b0b3c4a12668dcca14881ec70097823c247ca505c455fd0db5da4c99b5",
                "sources": [
                    "sm_103a/cake_dsv4_c20607c57cc11784015e_kernel.cu",
                    "sm_103a/cake_dsv4_c20607c57cc11784015e_binding.cu",
                ],
            },
            "fp8_lowhead_prefill": {
                "arg_plan": _PLAN_FP8_LOWHEAD + _SLAB + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "61781095b5e9377262e277a85c730dcd0bb3588b3633cbe248765b64980e8e23",
                "sources": [
                    "sm_103a/cake_dsv4_a2be51a1fd34ec0d9a3b_kernel.cu",
                    "sm_103a/cake_dsv4_a2be51a1fd34ec0d9a3b_binding.cu",
                ],
                "tma_workspace_bytes": 384,
            },
            "nvfp4_decode_cluster": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ae7c965ff808d58bc7e0a1bce7645906f0356fee8f50459f90b95d36039de702",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_cbeb4120c4b16a2d6308_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_cbeb4120c4b16a2d6308_binding.cu",
                ],
            },
            "nvfp4_decode_persistent": {
                "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "4578abe687866ce72c061383c06e16e316efba8e57f30369095ae33baa7cdffc",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_1db6591ffb95125ea480_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_1db6591ffb95125ea480_binding.cu",
                ],
            },
            "nvfp4_decode_tile": {
                "arg_plan": _PLAN_NVFP4_TILE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "0c683d0a4672008b06801f5a867eca6bceba12cd74079933bf6ecfd474361dd4",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_fb3b5dff7b8e02006f6b_kernel_portable.cu",
                    "sm_103a/cake_dsv4_nvfp4_fb3b5dff7b8e02006f6b_binding.cu",
                ],
            },
            "nvfp4_merge": {
                "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "200de50f171e8f9accf97513a8a864e9ef4f689d43ed82810e54176cfdbedd27",
                "sources": [
                    "sm_103a/cake_dsv4_nvfp4_9516442fcae8fabf2de9_kernel.cu",
                    "sm_103a/cake_dsv4_nvfp4_9516442fcae8fabf2de9_binding.cu",
                ],
            },
            "split_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "014512652fd83c055323b34e9c4807829313cd35fbcde3a1d44cd87e89404265",
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
