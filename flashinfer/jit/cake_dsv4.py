"""JIT loader for the Cake-generated DeepSeek V4 sparse MLA kernels.

Every variant is one device kernel plus its TVM-FFI launcher under
``csrc/cake_dsv4/<arch>/`` (``common/`` when both architectures compile the same
source). ``_ARCH_REGISTRATIONS`` is the per-architecture build and argument
contract: the host (``flashinfer.mla.cake_dsv4``) binds every ``arg_plan`` entry
by name and passes the launch grid, so a regenerated registration only needs
names from the host vocabulary.

The BF16 / FP8 programs are spelled as three literal tables, so a regenerated
delivery rewrites the tables without touching the code that reads them:

* ``ARG_PLANS``: plan name -> ``[kind, name]`` pairs in binding order, the launch
  grid last.
* ``MODULES``: program -> its ``sources`` (relative to ``csrc/cake_dsv4``), its
  ``compile_flags``, the optional ``host_linkage_flags``, its ``arg_plan`` (an
  ``ARG_PLANS`` name), the ``arches`` that compile it and its ``identity`` per
  architecture (the digest that names the JIT module).
* ``VARIANTS``: architecture -> variant key -> program.

``_ARCH_REGISTRATIONS`` is built from those tables and the NVFP4 records.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

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

# The launch grid that ends every argument plan.
_GRID = (("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z"))

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

# The templated members take their retrace knobs as launch arguments after the
# kernel parameters: the launcher switches over the explicit instantiations of its
# kernel unit by (tile_n, o_chunks).
_PLAN_NVFP4_SWAP = _PLAN_NVFP4_TILE + (
    ("parameter", "tile_n"),
    ("parameter", "o_chunks"),
)
_PLAN_NVFP4_TILE_OC = _PLAN_NVFP4_TILE + (("parameter", "o_chunks"),)
_PLAN_NVFP4_PV = _PLAN_NVFP4_G4 + (("parameter", "tile_n"),)

# NVFP4 decode family: one record per member, shared by both architectures (their
# registrations differ only in the identity that names the JIT module). The templated
# members (pv, swap, tile, tile_h64) compile their kernel unit once with every explicit
# instantiation and take the retrace knobs (tile_n, o_chunks) as launch arguments.
_NVFP4_RECORDS = {
    "nvfp4_decode_cluster": {
        "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_decode_cluster_kernel.cu",
            "common/cake_dsv4_nvfp4_decode_cluster_binding.cu",
        ],
    },
    "nvfp4_decode_persistent": {
        "arg_plan": _PLAN_NVFP4_DECODE + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_decode_persistent_kernel.cu",
            "common/cake_dsv4_nvfp4_decode_persistent_binding.cu",
        ],
    },
    "nvfp4_decode_pv": {
        "arg_plan": _PLAN_NVFP4_PV + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_decode_pv_kernel.cu",
            "common/cake_dsv4_nvfp4_decode_pv_binding.cu",
        ],
    },
    "nvfp4_decode_swap": {
        "arg_plan": _PLAN_NVFP4_SWAP + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_decode_swap_kernel.cu",
            "common/cake_dsv4_nvfp4_decode_swap_binding.cu",
        ],
    },
    "nvfp4_decode_t64_n64_oc1": {
        "arg_plan": _PLAN_NVFP4_G4 + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_kernel.cu",
            "common/cake_dsv4_nvfp4_decode_t64_n64_oc1_binding.cu",
        ],
    },
    "nvfp4_decode_tile": {
        "arg_plan": _PLAN_NVFP4_TILE_OC + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_decode_tile_kernel.cu",
            "common/cake_dsv4_nvfp4_decode_tile_binding.cu",
        ],
    },
    "nvfp4_decode_tile_h64": {
        "arg_plan": _PLAN_NVFP4_TILE_OC + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_decode_tile_h64_kernel.cu",
            "common/cake_dsv4_nvfp4_decode_tile_h64_binding.cu",
        ],
    },
    "nvfp4_merge": {
        "arg_plan": _PLAN_NVFP4_MERGE + _GRID,
        "sources": [
            "common/cake_dsv4_nvfp4_merge_kernel.cu",
            "common/cake_dsv4_nvfp4_merge_binding.cu",
        ],
    },
}
_NVFP4_IDENTITIES = {
    "sm_100a": {
        "nvfp4_decode_cluster": "2971d71708fd20d7a5abc26da66a0763e94d0ca5075207d2acb4d2448c433be7",
        "nvfp4_decode_persistent": "3a9d813a27e4dc4836dedb132610f59e4a52b1699b7772714522fd97afc36b1b",
        "nvfp4_decode_pv": "e9d03a609e9a79e8647463bea12b930419bf01c0254734a9cc98a9ae985df2ff",
        "nvfp4_decode_swap": "fe90a8150f671f46ca1d26df290fe3e394252f123aab729f7a4a726b9548ef66",
        "nvfp4_decode_t64_n64_oc1": "18e16a5c2120d5e03b025dc48ea8b9eed10c99e6243a932c50019a5f41be0b8b",
        "nvfp4_decode_tile": "fd63f71fbe4f07219ae65174f962117f5bfa3a51eda8f21e0f1959ce18b07e01",
        "nvfp4_decode_tile_h64": "86d495adf8b3bcbd4af9b3f855d95e3b05b3804590fa42b322474d5f034e01a3",
        "nvfp4_merge": "0c4146f78e275850deedeaa5eecd9c6cfa790b7b62329e995d261b5b937881b7",
    },
    "sm_103a": {
        "nvfp4_decode_cluster": "3e173e5eefc84548e3acf1e120cf11b2c2e4a7f4f86b501b435cb68489789396",
        "nvfp4_decode_persistent": "48d99563a290f30f2630bb88a4d435862bf698d33a31265be9af3623cf803970",
        "nvfp4_decode_pv": "c02c5b7def113df4aa306b88fbcd1d1b7d5ec1135be392164b566424501a1121",
        "nvfp4_decode_swap": "a2ac56c674df1f3bcfcfcc4156f93559d44333f3d929240149a16509cefc611e",
        "nvfp4_decode_t64_n64_oc1": "4d23efd4fd0e5cd8c3a1d96ccaa081be6252db1b057dab909223bde061b25e45",
        "nvfp4_decode_tile": "5063120aaf31dd1d6866c3b1172e3f66a37f050519c9f801bd136c0dc40225b1",
        "nvfp4_decode_tile_h64": "5dd82fc89ef96a01498cb2c3ea6c7044ee038b85ebbcdab2c64f95e24e9b0725",
        "nvfp4_merge": "2c3aa9fe35accdfdf40b7985fbe76e42aab2974cf575a1aacea0b92d6f0d9d4a",
    },
}


def _nvfp4_variants(arch: str) -> dict[str, dict[str, object]]:
    """The NVFP4 records of one architecture (CUDA 13.4: PTX ISA 9.4 QMUL4)."""
    return {
        variant: {
            **record,
            "compile_flags": ["--use_fast_math"],
            "identity": _NVFP4_IDENTITIES[arch][variant],
            "min_cuda_version": "13.4",
        }
        for variant, record in _NVFP4_RECORDS.items()
    }


# The BF16 / FP8 argument plans, programs and variant routes (see the module
# docstring). Keep each table a plain literal (no name, call or operator inside)
# so that it can be read and rewritten as data.
ARG_PLANS: dict[str, list[list[str]]] = {
    "bf16_h128_persistent_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_k"],
        ["tma_buffer", "tmap_compressed_k"],
        ["tma_buffer", "tmap_swa_v"],
        ["tma_buffer", "tmap_compressed_v"],
        ["buffer", "O"],
        ["buffer", "partial_lse"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "num_query_tokens"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "has_sinks"],
        ["parameter", "total_work_items"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "bf16_h128_persistent_tma_o_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_k"],
        ["tma_buffer", "tmap_compressed_k"],
        ["tma_buffer", "tmap_swa_v"],
        ["tma_buffer", "tmap_compressed_v"],
        ["tma_buffer", "O"],
        ["buffer", "partial_lse"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "num_query_tokens"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "has_sinks"],
        ["parameter", "total_work_items"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "bf16_h128_swa_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["buffer", "O"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "num_head_tiles"],
        ["parameter", "has_sinks"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "bf16_h32_merge_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["buffer", "partial_O"],
        ["buffer", "partial_lse"],
        ["buffer", "O"],
        ["buffer", "partition_arrivals"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "num_splits"],
        ["parameter", "num_head_tiles"],
        ["parameter", "has_sinks"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "bf16_h64_guard_tma_o": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["tma_buffer", "O"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "batch_size"],
        ["parameter", "max_q_len"],
        ["parameter", "ragged_query"],
        ["parameter", "has_sinks"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "bf16_h64_prefill_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["buffer", "O"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "num_query_tokens"],
        ["parameter", "has_sinks"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "bf16_h64_split_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["buffer", "partial_O"],
        ["buffer", "partial_lse"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "num_splits"],
        ["parameter", "has_sinks"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "bf16_swa_decode_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["buffer", "O"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "has_sinks"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_h64_m64_swa_k_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["tma_buffer", "tmap_swa_k"],
        ["buffer", "O"],
        ["buffer", "partial_lse"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "num_query_tokens"],
        ["parameter", "sparse_topk"],
        ["parameter", "has_sinks"],
        ["parameter", "total_work_items"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_h64_source_exact_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["buffer", "O"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "seq_lens"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["parameter", "has_sinks"],
        ["parameter", "total_work_items"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_lowhead_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["buffer", "O"],
        ["buffer", "partial_lse"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "num_query_tokens"],
        ["parameter", "sparse_topk"],
        ["parameter", "has_sinks"],
        ["parameter", "total_work_items"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_persistent_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["tma_buffer", "tmap_o"],
        ["buffer", "O"],
        ["buffer", "partial_lse"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "num_heads"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "num_query_tokens"],
        ["parameter", "sparse_topk"],
        ["parameter", "has_sinks"],
        ["parameter", "total_work_items"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "h64_reduce": [
        ["buffer", "partial_O"],
        ["buffer", "partial_lse"],
        ["buffer", "O"],
        ["parameter", "num_heads"],
        ["parameter", "num_splits"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "h8_h16_source_exact_tokens_qlayout": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["buffer", "O"],
        ["buffer", "swa_indices"],
        ["buffer", "compressed_indices"],
        ["buffer", "sparse_topk_lens"],
        ["buffer", "seq_lens"],
        ["buffer", "cum_seq_lens_q"],
        ["buffer", "sinks"],
        ["buffer", "bmm1_scale"],
        ["buffer", "bmm2_scale"],
        ["parameter", "swa_index_stride"],
        ["parameter", "compressed_index_stride"],
        ["parameter", "sparse_topk_lens_offset"],
        ["parameter", "sparse_topk"],
        ["parameter", "ragged_query"],
        ["parameter", "max_q_len"],
        ["parameter", "batch_size"],
        ["parameter", "has_sinks"],
        ["parameter", "num_query_tokens"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "split_reduce": [
        ["buffer", "partial_O"],
        ["buffer", "partial_lse"],
        ["buffer", "O"],
        ["parameter", "num_q_heads"],
        ["parameter", "num_split"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

MODULES: dict[str, dict[str, Any]] = {
    "cake_dsv4_009fa784c7206613d7b7": {
        "sources": [
            "sm_100a/cake_dsv4_009fa784c7206613d7b7_kernel.cu",
            "sm_100a/cake_dsv4_009fa784c7206613d7b7_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_persistent_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "dbc975825805c1ed5b4264421711817b45d81ef025941ff9be4be1b435f4c981",
        },
    },
    "cake_dsv4_13091b5bf0612783f02d": {
        "sources": [
            "sm_103a/cake_dsv4_13091b5bf0612783f02d_kernel.cu",
            "sm_103a/cake_dsv4_13091b5bf0612783f02d_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "360a43167438aa6a30d1bf22a05714251a71a66dba5b5384927ffdaa0e16dbfc",
        },
    },
    "cake_dsv4_13f0ca13f90a32070cfe": {
        "sources": [
            "sm_103a/cake_dsv4_13f0ca13f90a32070cfe_kernel.cu",
            "sm_103a/cake_dsv4_13f0ca13f90a32070cfe_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_swa_decode_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "3f64f1cb5db5832198c942deeda53649bf13a20363d8bf7705c716c9caf3ac90",
        },
    },
    "cake_dsv4_16749fca629bdea1d909": {
        "sources": [
            "common/cake_dsv4_16749fca629bdea1d909_kernel.cu",
            "common/cake_dsv4_16749fca629bdea1d909_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_h128_swa_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "eecbea87366fe396c72da546e4bbb1b34027965c06d851fb8041bfe6646104bc",
            "sm_103a": "621306e47d1eb3bf715dd8400a1ff3c61a0d055004538bae069d5d740989abcf",
        },
    },
    "cake_dsv4_1aefb39e2f65a24b1628": {
        "sources": [
            "sm_103a/cake_dsv4_1aefb39e2f65a24b1628_kernel.cu",
            "sm_103a/cake_dsv4_1aefb39e2f65a24b1628_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_tma_o_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "9ae633cce8741795babfaf5912348720c7f0e8c9309fbef76d4bdce174b67e0b",
        },
    },
    "cake_dsv4_1cd047de4a9a94e4f214": {
        "sources": [
            "common/cake_dsv4_1cd047de4a9a94e4f214_kernel.cu",
            "common/cake_dsv4_1cd047de4a9a94e4f214_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_h64_guard_tma_o",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "0532d16cc54ca4386865fd4d774849cf79948647db5edcf8a207cc1234bccda0",
            "sm_103a": "e5fecce40a647b75e935879b7f15fbe69116019cac20b5551c10501c5a2e8c7e",
        },
    },
    "cake_dsv4_1e2529ed7823ff5927df": {
        "sources": [
            "sm_103a/cake_dsv4_1e2529ed7823ff5927df_kernel.cu",
            "sm_103a/cake_dsv4_1e2529ed7823ff5927df_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "7902ae3300933ecdd9795bd273d53806046fede89acdde3bc75db151c8d86123",
        },
    },
    "cake_dsv4_289be7e9a77a8bb87264": {
        "sources": [
            "sm_100a/cake_dsv4_289be7e9a77a8bb87264_kernel.cu",
            "sm_100a/cake_dsv4_289be7e9a77a8bb87264_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "split_reduce",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "793778812017819a450fc6bb83c6fa86bbdcfb731fcba674313f230845f96f5f",
        },
    },
    "cake_dsv4_30ff1ef182cbae9bfba3": {
        "sources": [
            "sm_100a/cake_dsv4_30ff1ef182cbae9bfba3_kernel.cu",
            "sm_100a/cake_dsv4_30ff1ef182cbae9bfba3_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "9c5c537b7f73afb40a1f698063646e3d595733991bd82bd986dc314b1a0e1812",
        },
    },
    "cake_dsv4_36c80272414ea39a56d2": {
        "sources": [
            "common/cake_dsv4_36c80272414ea39a56d2_kernel.cu",
            "common/cake_dsv4_36c80272414ea39a56d2_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "8933a25c10455d1c266216d5fe0069367b3f3578d45c4c2bc2877311961b7267",
            "sm_103a": "25e23b357ee5a9141c42b933e73f0f639af8ed1bd695505068e7a82fe77da22a",
        },
    },
    "cake_dsv4_3fce080f96f6dc5e9f8c": {
        "sources": [
            "sm_103a/cake_dsv4_3fce080f96f6dc5e9f8c_kernel.cu",
            "sm_103a/cake_dsv4_3fce080f96f6dc5e9f8c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_m64_swa_k_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "548f8a0dfd23947bcdd7fe854a34b779cd37dfbd838ce45d6f8e2022ac4d1251",
        },
    },
    "cake_dsv4_40932cb442b76026206f": {
        "sources": [
            "sm_103a/cake_dsv4_40932cb442b76026206f_kernel.cu",
            "sm_103a/cake_dsv4_40932cb442b76026206f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_m64_swa_k_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "3e13dbcd568df4f6fa2b01d5deafbeb8a847e277350b4c085e80ed393018f1ed",
        },
    },
    "cake_dsv4_42cbf7d70ac869d2a473": {
        "sources": [
            "sm_100a/cake_dsv4_42cbf7d70ac869d2a473_kernel.cu",
            "sm_100a/cake_dsv4_42cbf7d70ac869d2a473_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "888325ac377098ab61f593baf8778a40a36fab21b6a32da2a537a4878c2fa7be",
        },
    },
    "cake_dsv4_430c388f48f1e672627c": {
        "sources": [
            "sm_103a/cake_dsv4_430c388f48f1e672627c_kernel.cu",
            "sm_103a/cake_dsv4_430c388f48f1e672627c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "565f892b7461dd4f5eec85db7deb1e5f9feb1f7b178f80d267c4187559ac15c5",
        },
    },
    "cake_dsv4_4869e52499bb527f3e84": {
        "sources": [
            "sm_100a/cake_dsv4_4869e52499bb527f3e84_kernel.cu",
            "sm_100a/cake_dsv4_4869e52499bb527f3e84_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_swa_decode_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "56e39a54a6c54528942d4755a15a8e1e8e4e00d1cfbb3656da64042c9cb255ca",
        },
    },
    "cake_dsv4_4c389d42544dcc6552cd": {
        "sources": [
            "common/cake_dsv4_4c389d42544dcc6552cd_kernel.cu",
            "common/cake_dsv4_4c389d42544dcc6552cd_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "5354ee41c49debe4450353fda5623721c7ba465b16f3bd4242f0e4b92e074844",
            "sm_103a": "3a3db28e3299b0a8e14932b9a73b0aa203f80593cd4ad09983a1a5411afb6f46",
        },
    },
    "cake_dsv4_50a2c6ca93259d30db6b": {
        "sources": [
            "common/cake_dsv4_50a2c6ca93259d30db6b_kernel.cu",
            "common/cake_dsv4_50a2c6ca93259d30db6b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h64_reduce",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "254641199b600346536f0178bd47d30e956c6a3bddcf968102de840ec92c3f6f",
            "sm_103a": "7c66d36bbb6e1e4e7aa1ffc910352b2a20936732a9766ad55e47dd14e35ee73e",
        },
    },
    "cake_dsv4_51eee53d2d4f8b5344e9": {
        "sources": [
            "common/cake_dsv4_51eee53d2d4f8b5344e9_kernel.cu",
            "common/cake_dsv4_51eee53d2d4f8b5344e9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "996d876b90eb9a70eb3d0a749b6bb365cc7ca986890d74de1bd9be534405544e",
            "sm_103a": "97c93ae7199cc266045d634d4ac0019f43a413b75dd4aec2e016ab1b7a9c0bfc",
        },
    },
    "cake_dsv4_5a02e8137c1b1bc7c58a": {
        "sources": [
            "sm_100a/cake_dsv4_5a02e8137c1b1bc7c58a_kernel.cu",
            "sm_100a/cake_dsv4_5a02e8137c1b1bc7c58a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_m64_swa_k_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "07ae543b81e4af48710c939727a1aa748c79608d3560680842c2e890aca1c336",
        },
    },
    "cake_dsv4_5f640d72462f26d33803": {
        "sources": [
            "common/cake_dsv4_5f640d72462f26d33803_kernel.cu",
            "common/cake_dsv4_5f640d72462f26d33803_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "split_reduce",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "00c1f03b2b547713d533e73462ebe845b856a21adc732d2ca3c02b0e7b9527c0",
            "sm_103a": "56980013bb3fe4bba3c55e9c737cc2c2ef6a172363e374108b2b77d67013c23a",
        },
    },
    "cake_dsv4_695ead762879465130c3": {
        "sources": [
            "sm_100a/cake_dsv4_695ead762879465130c3_kernel.cu",
            "sm_100a/cake_dsv4_695ead762879465130c3_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "6e72083d080959652436a2a8f198cd63f05704734acabf11ff46750848959bf9",
        },
    },
    "cake_dsv4_750835d54fe97cfc7ad4": {
        "sources": [
            "common/cake_dsv4_750835d54fe97cfc7ad4_kernel.cu",
            "common/cake_dsv4_750835d54fe97cfc7ad4_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_swa_decode_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "4226959c60d3389663a4382d82ffe98f225311b3d55e10781bc33112d48454e7",
            "sm_103a": "5dbb36979e65f66eaf611bffc5a839bdc6da18da9acd7ead3e62bedf7f09e66c",
        },
    },
    "cake_dsv4_76a317e04008767a3737": {
        "sources": [
            "sm_100a/cake_dsv4_76a317e04008767a3737_kernel.cu",
            "sm_100a/cake_dsv4_76a317e04008767a3737_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_lowhead_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "fbbb8614dda940a466b6fa489be93c5d2a515b0e66bc3b73e586ff2e9eeaca62",
        },
    },
    "cake_dsv4_7850190a53cc07aba85a": {
        "sources": [
            "sm_103a/cake_dsv4_7850190a53cc07aba85a_kernel.cu",
            "sm_103a/cake_dsv4_7850190a53cc07aba85a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_swa_decode_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "06fd4da46281f1c4a30e5944bc86f1126c2584f2b09fa1b7800536908c8fc272",
        },
    },
    "cake_dsv4_7c51b4b6d5b345d4fe03": {
        "sources": [
            "sm_103a/cake_dsv4_7c51b4b6d5b345d4fe03_kernel.cu",
            "sm_103a/cake_dsv4_7c51b4b6d5b345d4fe03_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_persistent_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "4bbaab7255e1b31033b12c0c13d62aa12fe0cddf0c43ab839941f82e89be9299",
        },
    },
    "cake_dsv4_7d6d09e037a1a9e99d87": {
        "sources": [
            "common/cake_dsv4_7d6d09e037a1a9e99d87_kernel.cu",
            "common/cake_dsv4_7d6d09e037a1a9e99d87_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_h64_prefill_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "8261fffc7484344ede9441c1c7e6326e43234bb51f0f527e4cfacf82b60d7703",
            "sm_103a": "4370ff7b8f887d3169f5c53b17aa4349343535acfe96d748bdd7fb5727f01e1d",
        },
    },
    "cake_dsv4_82ca6fee14338e07ece4": {
        "sources": [
            "common/cake_dsv4_82ca6fee14338e07ece4_kernel.cu",
            "common/cake_dsv4_82ca6fee14338e07ece4_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_h64_split_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "2f99b2a7afaaebd7155d2f7a9eb549cd251617145789bcfec11db0eb325baf70",
            "sm_103a": "bddd4c2730d1e8ff49a6499206753612675f3349a2ead78db4c100590f761eb3",
        },
    },
    "cake_dsv4_88757d413062636b5640": {
        "sources": [
            "sm_100a/cake_dsv4_88757d413062636b5640_kernel.cu",
            "sm_100a/cake_dsv4_88757d413062636b5640_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_tma_o_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "4290339dd401a14f7bed380842fadadbc7b670f9542f1d4af4088f929422ccc3",
        },
    },
    "cake_dsv4_8bcdc9f41a3d74231d15": {
        "sources": [
            "sm_103a/cake_dsv4_8bcdc9f41a3d74231d15_kernel.cu",
            "sm_103a/cake_dsv4_8bcdc9f41a3d74231d15_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_persistent_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "6a434dff40724f43052e22265792539ee4b4cd0c5a7ce6b165669f89d0e55b47",
        },
    },
    "cake_dsv4_9daa9c5ff43490ad80ac": {
        "sources": [
            "sm_100a/cake_dsv4_9daa9c5ff43490ad80ac_kernel.cu",
            "sm_100a/cake_dsv4_9daa9c5ff43490ad80ac_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "1d5ffeeaa9987a7064927cfc2f184984605d45ff8521c27ad86d325b7a8bbbeb",
        },
    },
    "cake_dsv4_9f75cb272ce8994420ff": {
        "sources": [
            "sm_100a/cake_dsv4_9f75cb272ce8994420ff_kernel.cu",
            "sm_100a/cake_dsv4_9f75cb272ce8994420ff_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_persistent_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "a7afade48c41e04c0db732a909e4b814c5063d5b7b5ab80f00e53f311b2b63bc",
        },
    },
    "cake_dsv4_b0835b18bf6b2cf1d833": {
        "sources": [
            "sm_100a/cake_dsv4_b0835b18bf6b2cf1d833_kernel.cu",
            "sm_100a/cake_dsv4_b0835b18bf6b2cf1d833_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "61a4ede3c66e23c858992e9a5e6d27b37d85caf6d4e34ee76e8f7fe74d6c2650",
        },
    },
    "cake_dsv4_c05cf124b389eee7c55a": {
        "sources": [
            "sm_103a/cake_dsv4_c05cf124b389eee7c55a_kernel.cu",
            "sm_103a/cake_dsv4_c05cf124b389eee7c55a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "a07c4fcd9c353c82db1ae0ddfd3358506266f9747b150fca335f2b4ee623cd10",
        },
    },
    "cake_dsv4_cf7e7f261a5e1c42f98f": {
        "sources": [
            "sm_103a/cake_dsv4_cf7e7f261a5e1c42f98f_kernel.cu",
            "sm_103a/cake_dsv4_cf7e7f261a5e1c42f98f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "f2d06b61a833bf2e255a7d37066f4f676ffd1afc6b55920f6312bc6e72b529ec",
        },
    },
    "cake_dsv4_d0233d0d9fe44a645212": {
        "sources": [
            "sm_100a/cake_dsv4_d0233d0d9fe44a645212_kernel.cu",
            "sm_100a/cake_dsv4_d0233d0d9fe44a645212_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_swa_decode_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "cad88341403732c7ae0b50f2ec501f2a7e9fcaebfb1b06577216e05800289429",
        },
    },
    "cake_dsv4_dd34c7d71da6b6f66030": {
        "sources": [
            "sm_103a/cake_dsv4_dd34c7d71da6b6f66030_kernel.cu",
            "sm_103a/cake_dsv4_dd34c7d71da6b6f66030_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "b1371372d5dd1ea5ba69716849e7294889d52629e4384855414b07e185aaf85d",
        },
    },
    "cake_dsv4_e5e9d7f2a2798aec29ff": {
        "sources": [
            "common/cake_dsv4_e5e9d7f2a2798aec29ff_kernel.cu",
            "common/cake_dsv4_e5e9d7f2a2798aec29ff_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_source_exact_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "34284d78f03b0284b881b9720674d1a4813c6b087a742acca08cc5f123e12364",
            "sm_103a": "d993119b34840042a04139a493421b6bb1fad516a895b869c08da8710fe4053a",
        },
    },
    "cake_dsv4_e880e7c12a67fe035a37": {
        "sources": [
            "sm_103a/cake_dsv4_e880e7c12a67fe035a37_kernel.cu",
            "sm_103a/cake_dsv4_e880e7c12a67fe035a37_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "split_reduce",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "1712237fc719cb0c2f32d750eacd1dda44c264a5fd3a2f330c46883c30f4eaee",
        },
    },
    "cake_dsv4_f0c84ff1c7e6bce419fb": {
        "sources": [
            "common/cake_dsv4_f0c84ff1c7e6bce419fb_kernel.cu",
            "common/cake_dsv4_f0c84ff1c7e6bce419fb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_tma_o_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "4345573e8f8f61ba8b04d830f8928ceee64c6e2f8c30e8c50dbcd17761f454e0",
            "sm_103a": "3dd55506a2708cd6e3287c1fd10ffc4002333f53dae1f9b8a11e3e352da9dc46",
        },
    },
    "cake_dsv4_f647b239789ffbbdbd90": {
        "sources": [
            "sm_100a/cake_dsv4_f647b239789ffbbdbd90_kernel.cu",
            "sm_100a/cake_dsv4_f647b239789ffbbdbd90_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_m64_swa_k_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "159bd348c1481ad010a0c317a9ff532939001f03be889bb5fcf2f7683114f7cb",
        },
    },
    "cake_dsv4_f81486848bf3b4e05349": {
        "sources": [
            "common/cake_dsv4_f81486848bf3b4e05349_kernel.cu",
            "common/cake_dsv4_f81486848bf3b4e05349_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "bf16_h32_merge_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "e8b2b84eb0152f83cf97526d156d5619b819e67c79827eeb1a83d5b564d07020",
            "sm_103a": "b4a3936030565653e447cc2706a138b67cf8158707b131f4abf53ecfc32a1a1c",
        },
    },
}

VARIANTS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "bf16_h128_prefill_v42": "cake_dsv4_30ff1ef182cbae9bfba3",
        "bf16_h128_prefill_v42_snake": "cake_dsv4_9daa9c5ff43490ad80ac",
        "bf16_h128_split5_reduce": "cake_dsv4_289be7e9a77a8bb87264",
        "bf16_h128_swa128": "cake_dsv4_16749fca629bdea1d909",
        "bf16_h128_topk128x_row_first": "cake_dsv4_36c80272414ea39a56d2",
        "bf16_h128_topk128x_row_first_vsplit": "cake_dsv4_51eee53d2d4f8b5344e9",
        "bf16_h128_topk128x_split4_sm100": "cake_dsv4_f0c84ff1c7e6bce419fb",
        "bf16_h128_topk4x_v52": "cake_dsv4_88757d413062636b5640",
        "bf16_h16_h32_swa128_v44": "cake_dsv4_750835d54fe97cfc7ad4",
        "bf16_h32_topk128x_early_v47": "cake_dsv4_f81486848bf3b4e05349",
        "bf16_h64_compressed_q8_v38": "cake_dsv4_82ca6fee14338e07ece4",
        "bf16_h64_compressed_reduce": "cake_dsv4_50a2c6ca93259d30db6b",
        "bf16_h64_guard_q_tma_batch_r25": "cake_dsv4_1cd047de4a9a94e4f214",
        "bf16_h64_prefill": "cake_dsv4_7d6d09e037a1a9e99d87",
        "bf16_h8_h16_source_exact": "cake_dsv4_b0835b18bf6b2cf1d833",
        "bf16_h8_swa128_v43": "cake_dsv4_4869e52499bb527f3e84",
        "bf16_swa128_single_cta": "cake_dsv4_d0233d0d9fe44a645212",
        "fp8_h128_prefill_source_persistent": "cake_dsv4_9f75cb272ce8994420ff",
        "fp8_h128_prefill_source_persistent_uniform": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h64_prefill_source_persistent_m64": "cake_dsv4_5a02e8137c1b1bc7c58a",
        "fp8_h64_prefill_source_persistent_m64_multi_tile": "cake_dsv4_f647b239789ffbbdbd90",
        "fp8_h64_source_exact": "cake_dsv4_e5e9d7f2a2798aec29ff",
        "fp8_h8_h16_source_exact": "cake_dsv4_695ead762879465130c3",
        "fp8_lowhead_h64": "cake_dsv4_4c389d42544dcc6552cd",
        "fp8_lowhead_one_partition": "cake_dsv4_76a317e04008767a3737",
        "fp8_lowhead_prefill": "cake_dsv4_42cbf7d70ac869d2a473",
        "split_reduce": "cake_dsv4_5f640d72462f26d33803",
    },
    "sm_103a": {
        "bf16_h128_prefill_v42": "cake_dsv4_13091b5bf0612783f02d",
        "bf16_h128_prefill_v42_snake": "cake_dsv4_cf7e7f261a5e1c42f98f",
        "bf16_h128_split5_reduce": "cake_dsv4_e880e7c12a67fe035a37",
        "bf16_h128_swa128": "cake_dsv4_16749fca629bdea1d909",
        "bf16_h128_topk128x_row_first": "cake_dsv4_36c80272414ea39a56d2",
        "bf16_h128_topk128x_row_first_vsplit": "cake_dsv4_51eee53d2d4f8b5344e9",
        "bf16_h128_topk128x_split4_sm100": "cake_dsv4_f0c84ff1c7e6bce419fb",
        "bf16_h128_topk4x_v52": "cake_dsv4_1aefb39e2f65a24b1628",
        "bf16_h16_h32_swa128_v44": "cake_dsv4_750835d54fe97cfc7ad4",
        "bf16_h32_topk128x_early_v47": "cake_dsv4_f81486848bf3b4e05349",
        "bf16_h64_compressed_q8_v38": "cake_dsv4_82ca6fee14338e07ece4",
        "bf16_h64_compressed_reduce": "cake_dsv4_50a2c6ca93259d30db6b",
        "bf16_h64_guard_q_tma_batch_r25": "cake_dsv4_1cd047de4a9a94e4f214",
        "bf16_h64_prefill": "cake_dsv4_7d6d09e037a1a9e99d87",
        "bf16_h8_h16_source_exact": "cake_dsv4_1e2529ed7823ff5927df",
        "bf16_h8_swa128_v43": "cake_dsv4_7850190a53cc07aba85a",
        "bf16_swa128_single_cta": "cake_dsv4_13f0ca13f90a32070cfe",
        "fp8_h128_prefill_source_persistent": "cake_dsv4_7c51b4b6d5b345d4fe03",
        "fp8_h128_prefill_source_persistent_uniform": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h64_prefill_source_persistent_m64": "cake_dsv4_40932cb442b76026206f",
        "fp8_h64_prefill_source_persistent_m64_multi_tile": "cake_dsv4_3fce080f96f6dc5e9f8c",
        "fp8_h64_source_exact": "cake_dsv4_e5e9d7f2a2798aec29ff",
        "fp8_h8_h16_source_exact": "cake_dsv4_dd34c7d71da6b6f66030",
        "fp8_lowhead_h64": "cake_dsv4_4c389d42544dcc6552cd",
        "fp8_lowhead_one_partition": "cake_dsv4_c05cf124b389eee7c55a",
        "fp8_lowhead_prefill": "cake_dsv4_430c388f48f1e672627c",
        "split_reduce": "cake_dsv4_5f640d72462f26d33803",
    },
}


def _registration(arch: str, variant: str) -> dict[str, object]:
    """The registration of one BF16 / FP8 variant on ``arch``, read from the tables."""
    program = VARIANTS[arch][variant]
    record = MODULES[program]
    if arch not in record["arches"]:
        raise ValueError(
            f"CAKE DSv4 variant {variant} ({arch}) names {program}, "
            f"which is built for {record['arches']} only"
        )
    registration: dict[str, object] = {
        "arg_plan": tuple(tuple(pair) for pair in ARG_PLANS[record["arg_plan"]]),
        "compile_flags": list(record["compile_flags"]),
    }
    if "host_linkage_flags" in record:
        registration["host_linkage_flags"] = list(record["host_linkage_flags"])
    registration["identity"] = record["identity"][arch]
    registration["sources"] = list(record["sources"])
    return registration


def _arch_registration(arch: str) -> dict[str, dict[str, dict[str, object]]]:
    """Every variant of ``arch`` (BF16 / FP8 and NVFP4), in key order."""
    variants = {variant: _registration(arch, variant) for variant in VARIANTS[arch]}
    nvfp4 = _nvfp4_variants(arch)
    shared = sorted(set(variants) & set(nvfp4))
    if shared:
        raise ValueError(f"CAKE DSv4 variant keys registered twice on {arch}: {shared}")
    variants.update(nvfp4)
    return {"variants": dict(sorted(variants.items()))}


_ARCH_REGISTRATIONS = {arch: _arch_registration(arch) for arch in _ARCH_NVCC_FLAGS}


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
