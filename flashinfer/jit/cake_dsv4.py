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
                "identity": "9527b6ec69897dc529d1ccdb32e55294e8310dc1a2c7b7ff537bb4076371784c",
                "sources": [
                    "sm_103a/cake_dsv4_1b49b5e869bcfb2e9094_kernel.cu",
                    "sm_103a/cake_dsv4_1b49b5e869bcfb2e9094_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "becf88681c252f775028786c1113d2d8c5e4f66d5eecfb13a73496d93bea1ff2",
                "sources": [
                    "sm_103a/cake_dsv4_f0ab8caa952cc9a88247_kernel.cu",
                    "sm_103a/cake_dsv4_f0ab8caa952cc9a88247_binding.cu",
                ],
            },
            "bf16_h128_topk128x_row_first_vsplit": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "8143dfe0db5b19f8d42604dacd632d09f5854d977d899e57c634c531086c6d5c",
                "sources": [
                    "sm_103a/cake_dsv4_e8da69fb39bcfe8dc9ca_kernel.cu",
                    "sm_103a/cake_dsv4_e8da69fb39bcfe8dc9ca_binding.cu",
                ],
            },
            "bf16_h128_topk128x_split4_sm100": {
                "arg_plan": _PLAN_BF16_H128_PERSISTENT_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "4b57bb56e1c5b4fc794d06f05e8dc52b4b9cc66344d936808537489e31a16c9e",
                "sources": [
                    "sm_103a/cake_dsv4_4f8d14e0bcd15b791276_kernel.cu",
                    "sm_103a/cake_dsv4_4f8d14e0bcd15b791276_binding.cu",
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
                "identity": "c71e24bdaed70076b0fc9b19f51323179fd3d1200b0856f2167b7f08b713dafb",
                "sources": [
                    "sm_103a/cake_dsv4_16db8a119baeaae5a9ce_kernel.cu",
                    "sm_103a/cake_dsv4_16db8a119baeaae5a9ce_binding.cu",
                ],
            },
            "bf16_h32_topk128x_early_v47": {
                "arg_plan": _PLAN_BF16_H32_MERGE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "a5308b1c5e343261a72fd8badcc83e2c88b3be89ad82772c415a759b1d2df43f",
                "sources": [
                    "sm_103a/cake_dsv4_dd52580c965ec3e3e2cf_kernel.cu",
                    "sm_103a/cake_dsv4_dd52580c965ec3e3e2cf_binding.cu",
                ],
            },
            "bf16_h64_compressed_q8_v38": {
                "arg_plan": _PLAN_BF16_H64_SPLIT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "30e403002376efbe27413b75374984132167df2b856ff9cfe577e65aecd815ab",
                "sources": [
                    "sm_103a/cake_dsv4_e0ea9a376eeda12eb2db_kernel.cu",
                    "sm_103a/cake_dsv4_e0ea9a376eeda12eb2db_binding.cu",
                ],
            },
            "bf16_h64_compressed_reduce": {
                "arg_plan": _PLAN_H64_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "2bb0b5c03f6db1c3aada3e1e5c6838f21e1ad4b515c2e85ef9045003793c092c",
                "sources": [
                    "sm_103a/cake_dsv4_17b2e2626e33001a5125_kernel.cu",
                    "sm_103a/cake_dsv4_17b2e2626e33001a5125_binding.cu",
                ],
            },
            "bf16_h64_guard_q_tma_batch_r25": {
                "arg_plan": _PLAN_BF16_H64_GUARD_TMA_O + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "e66082209ddc888d93609afed16c7319877d31fdb332fa371a1aec2478dad86f",
                "sources": [
                    "sm_103a/cake_dsv4_2bc930a5b15355cb7008_kernel.cu",
                    "sm_103a/cake_dsv4_2bc930a5b15355cb7008_binding.cu",
                ],
            },
            "bf16_h64_prefill": {
                "arg_plan": _PLAN_BF16_H64_PREFILL + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "da6cf0ed3af476321bf18df73c139eb94e069ce5f2a3ef9974cd17c0d83e38dc",
                "sources": [
                    "sm_103a/cake_dsv4_f3c14b69f5e723b18f33_kernel.cu",
                    "sm_103a/cake_dsv4_f3c14b69f5e723b18f33_binding.cu",
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
                "identity": "9b414dddf9d1c84dc3e3cd684b0d3008c54f088c44de1caaa8fe7f36c5b8a75a",
                "sources": [
                    "sm_103a/cake_dsv4_88b316b6fe0f24e2fb7e_kernel.cu",
                    "sm_103a/cake_dsv4_88b316b6fe0f24e2fb7e_binding.cu",
                ],
            },
            "fp8_h64_prefill_source_persistent_m64_multi_tile": {
                "arg_plan": _PLAN_FP8_H64_M64_SWA_K + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "5b4cdc853e9f587b08802b4f7f8e1200a7a07f1898855d8b1a0e6de1a2cdda87",
                "sources": [
                    "sm_103a/cake_dsv4_76e682b31bec10da45eb_kernel.cu",
                    "sm_103a/cake_dsv4_76e682b31bec10da45eb_binding.cu",
                ],
            },
            "fp8_h64_source_exact": {
                "arg_plan": _PLAN_FP8_H64_SOURCE_EXACT + _GRID,
                "compile_flags": ["--use_fast_math"],
                "identity": "ec232f3904ca2b28dbd9cc8057a6f7e0372cb607c2c4e96f38f8dd2868d42fc2",
                "sources": [
                    "sm_103a/cake_dsv4_9500966aba6526ef2ebf_kernel.cu",
                    "sm_103a/cake_dsv4_9500966aba6526ef2ebf_binding.cu",
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
                "identity": "7e1cd21e7581be0735a70ded5767af33cff72950a3c53470f7b71a57b688cfc0",
                "sources": [
                    "sm_103a/cake_dsv4_cf247d868a7803a3a029_kernel.cu",
                    "sm_103a/cake_dsv4_cf247d868a7803a3a029_binding.cu",
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
            "split_reduce": {
                "arg_plan": _PLAN_SPLIT_REDUCE + _GRID,
                "compile_flags": ["--use_fast_math"],
                "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
                "identity": "d8f652775016eec9815c4b6183c78e6a79ebc9584a1c40fff998a82ce63fcd3e",
                "sources": [
                    "sm_103a/cake_dsv4_8f432667f00f60acac64_kernel.cu",
                    "sm_103a/cake_dsv4_8f432667f00f60acac64_binding.cu",
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
