"""JIT loader for the Cake-generated DeepSeek V4 sparse MLA kernels.

Every variant is one device kernel plus its TVM-FFI launcher under
``csrc/cake_dsv4/<arch>/`` (``common/`` when both SM100 and SM103 compile the
same source). SM107 (``sm_107a``) routes only the FP8 decode variants whose Cake
programs build for it, each from its own ``sm_107a/`` sources, and has no NVFP4
records. ``_ARCH_REGISTRATIONS`` is the per-architecture build and argument
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
    sm107a_nvcc_flags,
)
from .cpp_ext import get_cuda_version, is_cuda_version_at_least


_ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}

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
# NVFP4 prefill bodies: the caller-packed NVFP4 query through two tensor maps (data and scale rows),
# the pools through 32-byte-pitch row views (data and footer rows); all write the final O and the
# base-2 LSE directly. The query-block bodies (16 / 32 heads, window-only rows) store O through the
# ``tmap_o`` map alone (a 16-row box map is encoded from the same tensor); the per-token H <= 64 body
# takes ``O`` as a pointer and, last, the ``tmap_o`` map its box builds store through; the two-CTA
# H128 body takes both.
_PLAN_NVFP4_QB_PREFILL = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_q_sf"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("tma_buffer", "tmap_swa_sf"),
    ("tma_buffer", "tmap_compressed_sf"),
    ("tma_buffer", "tmap_o"),
    ("buffer", "LSE"),
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
    ("parameter", "has_sinks"),
    ("parameter", "swa_page_log2"),
    ("parameter", "swa_pitch_units"),
    ("parameter", "swa_footer_units"),
    ("parameter", "compressed_page_log2"),
    ("parameter", "compressed_pitch_units"),
    ("parameter", "compressed_footer_units"),
    ("parameter", "swa_width"),
    ("parameter", "compressed_width"),
    ("buffer", "extra_topk_lens"),
)
_PLAN_NVFP4_H64_PREFILL = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_q_sf"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("tma_buffer", "tmap_swa_sf"),
    ("tma_buffer", "tmap_compressed_sf"),
    ("buffer", "O"),
    ("buffer", "LSE"),
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
    ("parameter", "has_sinks"),
    ("parameter", "swa_page_log2"),
    ("parameter", "swa_pitch_units"),
    ("parameter", "swa_footer_units"),
    ("parameter", "compressed_page_log2"),
    ("parameter", "compressed_pitch_units"),
    ("parameter", "compressed_footer_units"),
    ("parameter", "swa_width"),
    ("parameter", "compressed_width"),
    ("buffer", "extra_topk_lens"),
    ("tma_buffer", "tmap_o"),
)
_PLAN_NVFP4_H128_PREFILL = (
    ("tma_buffer", "tmap_q"),
    ("tma_buffer", "tmap_q_sf"),
    ("tma_buffer", "tmap_swa_kv"),
    ("tma_buffer", "tmap_compressed_kv"),
    ("tma_buffer", "tmap_swa_sf"),
    ("tma_buffer", "tmap_compressed_sf"),
    ("tma_buffer", "tmap_o"),
    ("buffer", "O"),
    ("buffer", "LSE"),
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
    ("parameter", "max_q_len"),
    ("parameter", "swa_page_log2"),
    ("parameter", "swa_pitch_units"),
    ("parameter", "swa_footer_units"),
    ("parameter", "compressed_page_log2"),
    ("parameter", "compressed_pitch_units"),
    ("parameter", "compressed_footer_units"),
    ("parameter", "swa_width"),
    ("parameter", "compressed_width"),
    ("buffer", "extra_topk_lens"),
)

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

# NVFP4 prefill family (Cake ``cake_dsv4_nvfp4_query_block_prefill`` and
# ``flashinfer_blackwell_trtllm_sparse_mla_dsv4_nvfp4_{h64,h128}_prefill_source_v1``): every
# query-block and per-token build is one source for both architectures; the two-CTA H128 body
# traces a different schedule per architecture (score reduction), so each has its own.
_NVFP4_PREFILL_RECORDS = {
    "sm_100a": {
        "nvfp4_h128_prefill": {
            "arg_plan": _PLAN_NVFP4_H128_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "bdf9ef79e705fc93daf38fb1ec6e87cadb12299fd350fc723f20dd949336e41a",
            "sources": [
                "sm_100a/cake_dsv4_nvfp4_h128_prefill_kernel.cu",
                "sm_100a/cake_dsv4_nvfp4_h128_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_early_gather_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "7e91d763042f530abc6b9c841e509c67207c7df2d7b5f48a5043964db2bcafa7",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_early_gather_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_early_gather_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_o_boxes_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "334950df8fa71d834bc63b134fe8fda8e006e577eef4d21e447f4caa6be17e3f",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_o_boxes_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_o_boxes_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_q_prefetch_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "31c39fd7028d454021cd8fb7448f266a0cf9ee6eb576547dbec95362b302d452",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_q_prefetch_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_q_prefetch_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_tmem_v_early_gather_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "3dd4d7634d90d1c6e0307047bb7788b8ca49249d0a2812d6fa9cfa87a4c11b2a",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_tmem_v_early_gather_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_tmem_v_early_gather_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_tmem_v_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "2d26ae3a41a6cfcde2faa77df100ca8152b65dca438747e880896d47802aad60",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_tmem_v_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_tmem_v_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h16_breaks_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "3dc76e9741876defe00c794e8d13e67da682194a0ffdb56e08c03b44be27de6f",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h16_breaks_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h16_breaks_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h16_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "4a91e246cd958c8be9889538ade592ab27d8d5927a01ab65b3c29a495744c176",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h16_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h16_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h32_late_claim_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "d9203869bbc18c44376f4d122f23c824725c36c10264acaf3a8b3db96d8bd0e4",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h32_late_claim_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h32_late_claim_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h32_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "276d2888199f80656f5292c8c6fd9301fd3c52abbebded52db269c26f850632b",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h32_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h32_prefill_binding.cu",
            ],
        },
    },
    "sm_103a": {
        "nvfp4_h128_prefill": {
            "arg_plan": _PLAN_NVFP4_H128_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "ed7af9522b27ceebfda98a94ff6a539d6ab4e4fbb496c7233b1454950f8a2f05",
            "sources": [
                "sm_103a/cake_dsv4_nvfp4_h128_prefill_kernel.cu",
                "sm_103a/cake_dsv4_nvfp4_h128_prefill_binding.cu",
            ],
        },
        "nvfp4_h128_v_unit_pairs_prefill": {
            "arg_plan": _PLAN_NVFP4_H128_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "b2d59a9f2b335ba02323931d610231f3ecedffa333f01cc2ad99aec7824e87e9",
            "sources": [
                "sm_103a/cake_dsv4_nvfp4_h128_v_unit_pairs_prefill_kernel.cu",
                "sm_103a/cake_dsv4_nvfp4_h128_v_unit_pairs_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_early_gather_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "7e91d763042f530abc6b9c841e509c67207c7df2d7b5f48a5043964db2bcafa7",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_early_gather_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_early_gather_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_o_boxes_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "334950df8fa71d834bc63b134fe8fda8e006e577eef4d21e447f4caa6be17e3f",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_o_boxes_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_o_boxes_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_q_prefetch_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "31c39fd7028d454021cd8fb7448f266a0cf9ee6eb576547dbec95362b302d452",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_q_prefetch_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_q_prefetch_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_tmem_v_early_gather_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "3dd4d7634d90d1c6e0307047bb7788b8ca49249d0a2812d6fa9cfa87a4c11b2a",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_tmem_v_early_gather_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_tmem_v_early_gather_prefill_binding.cu",
            ],
        },
        "nvfp4_h64_tmem_v_prefill": {
            "arg_plan": _PLAN_NVFP4_H64_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "2d26ae3a41a6cfcde2faa77df100ca8152b65dca438747e880896d47802aad60",
            "sources": [
                "common/cake_dsv4_nvfp4_h64_tmem_v_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_h64_tmem_v_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h16_breaks_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "3dc76e9741876defe00c794e8d13e67da682194a0ffdb56e08c03b44be27de6f",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h16_breaks_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h16_breaks_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h16_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "4a91e246cd958c8be9889538ade592ab27d8d5927a01ab65b3c29a495744c176",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h16_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h16_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h32_late_claim_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "d9203869bbc18c44376f4d122f23c824725c36c10264acaf3a8b3db96d8bd0e4",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h32_late_claim_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h32_late_claim_prefill_binding.cu",
            ],
        },
        "nvfp4_qb_h32_prefill": {
            "arg_plan": _PLAN_NVFP4_QB_PREFILL + _GRID,
            "compile_flags": ["--use_fast_math"],
            "identity": "276d2888199f80656f5292c8c6fd9301fd3c52abbebded52db269c26f850632b",
            "sources": [
                "common/cake_dsv4_nvfp4_qb_h32_prefill_kernel.cu",
                "common/cake_dsv4_nvfp4_qb_h32_prefill_binding.cu",
            ],
        },
    },
}


def _nvfp4_variants(arch: str) -> dict[str, dict[str, object]]:
    """The NVFP4 records of one architecture (CUDA 13.4: PTX ISA 9.4 QMUL4)."""
    variants: dict[str, dict[str, object]] = {
        variant: {
            **record,
            "compile_flags": ["--use_fast_math"],
            "identity": _NVFP4_IDENTITIES[arch][variant],
            "min_cuda_version": "13.4",
        }
        for variant, record in _NVFP4_RECORDS.items()
    }
    for variant, record in _NVFP4_PREFILL_RECORDS[arch].items():
        variants[variant] = {**record, "min_cuda_version": "13.4"}
    return variants


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
    "fp8_h128_prefill_source_persistent_no_pdl": [
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
    "fp8_h128_prefill_source_persistent_uniform": [
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
        ["launch", "enable_pdl"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_h128_prefill_source_persistent_uniform_k64_split": [
        ["tma_buffer", "tmap_q"],
        ["tma_buffer", "tmap_swa_kv"],
        ["tma_buffer", "tmap_compressed_kv"],
        ["tma_buffer", "tmap_o"],
        ["tma_buffer", "tmap_partial_o"],
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
        ["parameter", "num_whole_items"],
        ["parameter", "num_splits"],
        ["launch", "enable_pdl"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce": [
        ["buffer", "partial_O"],
        ["buffer", "partial_lse"],
        ["buffer", "O"],
        ["parameter", "num_heads"],
        ["parameter", "num_splits"],
        ["launch", "enable_pdl"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_h64_prefill_source_persistent_m64_multi_tile": [
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
        ["launch", "enable_pdl"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_l2pf": [
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
        ["buffer", "compressed_KV_cache"],
        ["parameter", "compressed_KV_rows"],
        ["launch", "enable_pdl"],
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
        ["parameter", "num_query_tokens"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "fp8_lowhead_prefill": [
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
        ["launch", "enable_pdl"],
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
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "8287bd3483e9062481662126c66084f46bc4d7e05eebf9b9405e95fd3147c59d",
        },
    },
    "cake_dsv4_089147c4b5a5062fab1b": {
        "sources": [
            "sm_103a/cake_dsv4_089147c4b5a5062fab1b_kernel.cu",
            "sm_103a/cake_dsv4_089147c4b5a5062fab1b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "723d365e77682461f22ae691dee3ad828a7a71656d4e47c9dbabc2edd5aa7cdd",
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
    "cake_dsv4_132f453f8df4a8300d5c": {
        "sources": [
            "sm_103a/cake_dsv4_132f453f8df4a8300d5c_kernel.cu",
            "sm_103a/cake_dsv4_132f453f8df4a8300d5c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_h128_prefill_source_persistent_no_pdl",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "37774c143b12b6115f2d994229deeae6ecc1af06d63a49bcd15b53da12aa3d69",
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
            "sm_103a": "091b8f6bfe8cf234842a92a89c28682895fb7d827a1d6baf9ae769b196d83a5e",
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
            "sm_100a": "714a2d5448946e564c05763ab2e3e544c54aad5debc776816a56164e4a83bc2c",
            "sm_103a": "cc76fb79e0c274e4268ad05b2da855b22cfab8cc2475ef6c7b2226c0debac50f",
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
    "cake_dsv4_25addc7a80a1e13c3ece": {
        "sources": [
            "sm_107a/cake_dsv4_25addc7a80a1e13c3ece_kernel.cu",
            "sm_107a/cake_dsv4_25addc7a80a1e13c3ece_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_prefill",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "af63049e241f0893c206390c9119f3be022bf32513fb73a58f1f6a221a8e1ba2",
        },
    },
    "cake_dsv4_27a5d41ecb0fd5ecec7b": {
        "sources": [
            "sm_107a/cake_dsv4_27a5d41ecb0fd5ecec7b_kernel.cu",
            "sm_107a/cake_dsv4_27a5d41ecb0fd5ecec7b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "380d1866f35e1ee80c88cbe92d4d95a59e513d55684e6829923236b5013444e3",
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
    "cake_dsv4_384b6be5c7906a440e61": {
        "sources": [
            "sm_107a/cake_dsv4_384b6be5c7906a440e61_kernel.cu",
            "sm_107a/cake_dsv4_384b6be5c7906a440e61_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "3330858123876de4edfe2d533dbfe2cede2316e6c042f33edad13be9e428fe67",
        },
    },
    "cake_dsv4_3eaffaa117944ce5e14e": {
        "sources": [
            "common/cake_dsv4_3eaffaa117944ce5e14e_kernel.cu",
            "common/cake_dsv4_3eaffaa117944ce5e14e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "host_linkage_flags": ["--device-entity-has-hidden-visibility=false"],
        "arg_plan": "bf16_h128_persistent_tma_o_qlayout",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "02d6144a16ca4404bced3edf9b68d7cb427c3bce22c658f1f497c89eb30cb049",
            "sm_103a": "ac53ab1e19758d2c0156cec8c965169f58ca0a12020bdb3a331866dba7f913f2",
        },
    },
    "cake_dsv4_3fce080f96f6dc5e9f8c": {
        "sources": [
            "sm_103a/cake_dsv4_3fce080f96f6dc5e9f8c_kernel.cu",
            "sm_103a/cake_dsv4_3fce080f96f6dc5e9f8c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "147e82a37af4791b7565ff78a576b41ffbbf525226f841b3d67002c4a118750e",
        },
    },
    "cake_dsv4_40932cb442b76026206f": {
        "sources": [
            "sm_103a/cake_dsv4_40932cb442b76026206f_kernel.cu",
            "sm_103a/cake_dsv4_40932cb442b76026206f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "bcf8fc2f3320ca8a1f2a6fa0991e0cd402ee461bf286e8fcd92b4ae3df24a689",
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
    "cake_dsv4_4a6718b7a2ca825d8800": {
        "sources": [
            "sm_107a/cake_dsv4_4a6718b7a2ca825d8800_kernel.cu",
            "sm_107a/cake_dsv4_4a6718b7a2ca825d8800_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "954684e567b70122d1e315f5b6ed7a98f1ee4b272f7586676147da5cfbdfa388",
        },
    },
    "cake_dsv4_506e02d3d583a84bacdd": {
        "sources": [
            "sm_100a/cake_dsv4_506e02d3d583a84bacdd_kernel.cu",
            "sm_100a/cake_dsv4_506e02d3d583a84bacdd_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_h128_prefill_source_persistent_no_pdl",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "e89081fea2e8534c42cfee9b5eceb944bc861b6a9ef4b8dd692016a829ff5a91",
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
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "cb4870c193fc1110579dc4b36247f42d7fcffbdb0f18b674ed3cd5033a4d406d",
        },
    },
    "cake_dsv4_5b9d14c8ac3031e13487": {
        "sources": [
            "sm_107a/cake_dsv4_5b9d14c8ac3031e13487_kernel.cu",
            "sm_107a/cake_dsv4_5b9d14c8ac3031e13487_binding.cu",
        ],
        "compile_flags": [],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "73032944ab872ff1ac8943d65f3222154d47e111c2c7857789476a400df08237",
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
    "cake_dsv4_7505ea6a749435f33f86": {
        "sources": [
            "sm_107a/cake_dsv4_7505ea6a749435f33f86_kernel.cu",
            "sm_107a/cake_dsv4_7505ea6a749435f33f86_binding.cu",
        ],
        "compile_flags": [],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "80ec9856b7d8aedb27d60b4b80481640121ca6668b5933ebf2d5b294d886720d",
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
            "sm_100a": "25c1f4c3f2c15aa0deead6e08dfd9695d2d41dae60459d3e5f9d48bda66aeaf9",
            "sm_103a": "e6e5b87fd8a983b27441821bef344529d5ee4590768197309b782583f469a393",
        },
    },
    "cake_dsv4_77fc7066e2ba5bebb062": {
        "sources": [
            "sm_107a/cake_dsv4_77fc7066e2ba5bebb062_kernel.cu",
            "sm_107a/cake_dsv4_77fc7066e2ba5bebb062_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "e1e02c8bf09123ce9837336807bb88ecd36a0509d1ba018a72ffc2c179065474",
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
            "sm_103a": "f628532d65dd1875e01123a77a8871a269168fdc6d63dfa6ffaf76b1d841a58e",
        },
    },
    "cake_dsv4_7a104353fbcabeae6af8": {
        "sources": [
            "sm_107a/cake_dsv4_7a104353fbcabeae6af8_kernel.cu",
            "sm_107a/cake_dsv4_7a104353fbcabeae6af8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_l2pf",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "d4ce9398f53bd222d2368a09a1ecde458a371cdc91091c7872c44b1915bfa3d0",
        },
    },
    "cake_dsv4_7c51b4b6d5b345d4fe03": {
        "sources": [
            "sm_103a/cake_dsv4_7c51b4b6d5b345d4fe03_kernel.cu",
            "sm_103a/cake_dsv4_7c51b4b6d5b345d4fe03_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "ac44c636adbd64873e4ca15fa914f27e95c8c90c6ec571f663ee5c3446c58e8d",
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
            "sm_100a": "fac9b6db2449cb8fa4785c8aeca7d00afe97b124b658e270722501c329897125",
            "sm_103a": "6b8c4d478b03aeef864d7ddf8afcc8d33aa8e5261a64bac2cd0a66e58ffb6be2",
        },
    },
    "cake_dsv4_8625d6529808d0d51fc2": {
        "sources": [
            "sm_107a/cake_dsv4_8625d6529808d0d51fc2_kernel.cu",
            "sm_107a/cake_dsv4_8625d6529808d0d51fc2_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_prefill",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "e524e06c9e670a135bc145055678ae3274c88f3f94f4e2695c12069b101a52db",
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
    "cake_dsv4_8afa079cbcc688b8b5b9": {
        "sources": [
            "sm_107a/cake_dsv4_8afa079cbcc688b8b5b9_kernel.cu",
            "sm_107a/cake_dsv4_8afa079cbcc688b8b5b9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "56a081bf317abfb7ff19f1270b24eb569d5bee93181c46f8131f5462a4badede",
        },
    },
    "cake_dsv4_8bcdc9f41a3d74231d15": {
        "sources": [
            "sm_103a/cake_dsv4_8bcdc9f41a3d74231d15_kernel.cu",
            "sm_103a/cake_dsv4_8bcdc9f41a3d74231d15_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "0409e5938c08aec662525ad558b0c68bbb2509892019bc75f69bf2cfcbe43c1a",
        },
    },
    "cake_dsv4_8e21f3bf00b9778e08c2": {
        "sources": [
            "sm_100a/cake_dsv4_8e21f3bf00b9778e08c2_kernel.cu",
            "sm_100a/cake_dsv4_8e21f3bf00b9778e08c2_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "a7e68188ce894ea1ae2c0ba4babb168b040877227741c7da89d7689561757d2d",
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
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "eed23e81f47b3907beab754067a37f0f5b40d6902187e8f2a5dd5a0210d61ec5",
        },
    },
    "cake_dsv4_b78d35b605cb10daf20f": {
        "sources": [
            "sm_103a/cake_dsv4_b78d35b605cb10daf20f_kernel.cu",
            "sm_103a/cake_dsv4_b78d35b605cb10daf20f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "31977e5437b6c3f2792cc92d967c421a8307d8e139f222dd522846103e5c22c1",
        },
    },
    "cake_dsv4_b7c345c94f3a216b24e5": {
        "sources": [
            "sm_100a/cake_dsv4_b7c345c94f3a216b24e5_kernel.cu",
            "sm_100a/cake_dsv4_b7c345c94f3a216b24e5_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "-Xptxas=--register-usage-level=10"],
        "arg_plan": "fp8_lowhead_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "fe482c106d746eb25e14b0e92414600c7894c848f56e6e2397c560edf962beec",
        },
    },
    "cake_dsv4_c51bf1f09fbe07ac3be6": {
        "sources": [
            "sm_107a/cake_dsv4_c51bf1f09fbe07ac3be6_kernel.cu",
            "sm_107a/cake_dsv4_c51bf1f09fbe07ac3be6_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_prefill",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "be235b6174a4e79388b4ab2221712ff4ed8006954ab3fa6fa366e3574e100ece",
        },
    },
    "cake_dsv4_cbf3168c696847fd3f69": {
        "sources": [
            "common/cake_dsv4_cbf3168c696847fd3f69_kernel.cu",
            "common/cake_dsv4_cbf3168c696847fd3f69_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_prefill",
        "arches": ["sm_100a", "sm_103a"],
        "identity": {
            "sm_100a": "1805f23a2001fe8a9e6bfa4efdc9104c9df345bbc6a59d051483372e126a5534",
            "sm_103a": "1408f1cf42e4f32c795d4f4bcaefcba78e6ddc86d6037717ffdc4df8b49e0a04",
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
    "cake_dsv4_d01ae8250de69f1360ee": {
        "sources": [
            "sm_107a/cake_dsv4_d01ae8250de69f1360ee_kernel.cu",
            "sm_107a/cake_dsv4_d01ae8250de69f1360ee_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform_k64_split",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "6a6fc3dd2ff14e8001d87ccf001159cc6c7397a870c614bc2abc0082e54cc4d8",
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
            "sm_100a": "717635d85e956dc8f0332111940fa97d38f58e7f5ef22506b91e148cfb9e6a8c",
        },
    },
    "cake_dsv4_d9d623599c9697a9ccf3": {
        "sources": [
            "sm_100a/cake_dsv4_d9d623599c9697a9ccf3_kernel.cu",
            "sm_100a/cake_dsv4_d9d623599c9697a9ccf3_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_prefill",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "b1b2c3108dcfb7ea8b54a57f7dde80f561fff8fd5010c93fa4b1ee3a738bdb18",
        },
    },
    "cake_dsv4_df5d069562864950e48b": {
        "sources": [
            "sm_107a/cake_dsv4_df5d069562864950e48b_kernel.cu",
            "sm_107a/cake_dsv4_df5d069562864950e48b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "e8e84bca5329e94abf29f10e4e48b2748eb9fb4900579c27dfae0ccbce55464e",
        },
    },
    "cake_dsv4_e1ae8472ce68a65facf8": {
        "sources": [
            "sm_107a/cake_dsv4_e1ae8472ce68a65facf8_kernel.cu",
            "sm_107a/cake_dsv4_e1ae8472ce68a65facf8_binding.cu",
        ],
        "compile_flags": [],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "2b5d6bd75ea859582fbfc9629e52009155c43d289b6b4e6cc4f6f8bced57a328",
        },
    },
    "cake_dsv4_e21760e2024d7c1e85c9": {
        "sources": [
            "sm_103a/cake_dsv4_e21760e2024d7c1e85c9_kernel.cu",
            "sm_103a/cake_dsv4_e21760e2024d7c1e85c9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_lowhead_prefill",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "bbaa5da7cb0f8fc63684d3daf1d5d2156b9c441e87068d2c666b6361df61b840",
        },
    },
    "cake_dsv4_e2c23db7ef9c580df1c5": {
        "sources": [
            "sm_107a/cake_dsv4_e2c23db7ef9c580df1c5_kernel.cu",
            "sm_107a/cake_dsv4_e2c23db7ef9c580df1c5_binding.cu",
        ],
        "compile_flags": [],
        "arg_plan": "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "8d8a0dcef58fed3bff5e1726e717dd789a10ecb2a3d8458f9ac4e5935cf495fc",
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
            "sm_100a": "13f4a547c0061816d07cdbe46b7ee77dfcad4f12a4ac9b6f9de44f32038530a4",
            "sm_103a": "16676f60a97e85568291e8efbd7eec3750189512899b6b3721c8bd98eabf3dc1",
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
    "cake_dsv4_f647b239789ffbbdbd90": {
        "sources": [
            "sm_100a/cake_dsv4_f647b239789ffbbdbd90_kernel.cu",
            "sm_100a/cake_dsv4_f647b239789ffbbdbd90_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "fec7317ee77f3bdab4feb97b72525afa540dd36c2d5855511e3e7d99d2b1be72",
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
    "cake_dsv4_fee38be0a1a558bc4cf4": {
        "sources": [
            "sm_100a/cake_dsv4_fee38be0a1a558bc4cf4_kernel.cu",
            "sm_100a/cake_dsv4_fee38be0a1a558bc4cf4_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_100a"],
        "identity": {
            "sm_100a": "b00e7ecf066310de780d1784a65726bdc931ec8e237c9a47a0e3ea7a065ad199",
        },
    },
    "cake_dsv4_ff4aa022aa97fded6032": {
        "sources": [
            "sm_103a/cake_dsv4_ff4aa022aa97fded6032_kernel.cu",
            "sm_103a/cake_dsv4_ff4aa022aa97fded6032_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "h8_h16_source_exact_tokens_qlayout",
        "arches": ["sm_103a"],
        "identity": {
            "sm_103a": "832dacc9ff00ef248b7be596da51588a6aca20c85fd17b4fdd25cbf7064f09b2",
        },
    },
    "cake_dsv4_ffa4216b35dae8d46543": {
        "sources": [
            "sm_107a/cake_dsv4_ffa4216b35dae8d46543_kernel.cu",
            "sm_107a/cake_dsv4_ffa4216b35dae8d46543_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "arg_plan": "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "arches": ["sm_107a"],
        "identity": {
            "sm_107a": "0f5ba42603e09cc45e0da9e7cd30c76c47d630c64d0bd6018cdb2c8b5a09caea",
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
        "bf16_h128_topk128x_split4_sm100": "cake_dsv4_3eaffaa117944ce5e14e",
        "bf16_h128_topk4x_v52": "cake_dsv4_88757d413062636b5640",
        "bf16_h16_h32_swa128_v44": "cake_dsv4_750835d54fe97cfc7ad4",
        "bf16_h32_topk128x_early_v47": "cake_dsv4_f81486848bf3b4e05349",
        "bf16_h64_compressed_q8_v38": "cake_dsv4_82ca6fee14338e07ece4",
        "bf16_h64_compressed_reduce": "cake_dsv4_50a2c6ca93259d30db6b",
        "bf16_h64_guard_q_tma_batch_r25": "cake_dsv4_1cd047de4a9a94e4f214",
        "bf16_h64_prefill": "cake_dsv4_7d6d09e037a1a9e99d87",
        "bf16_h8_h16_source_exact": "cake_dsv4_8e21f3bf00b9778e08c2",
        "bf16_h8_swa128_v43": "cake_dsv4_4869e52499bb527f3e84",
        "bf16_swa128_single_cta": "cake_dsv4_d0233d0d9fe44a645212",
        "fp8_h128_prefill_source_persistent": "cake_dsv4_9f75cb272ce8994420ff",
        "fp8_h128_prefill_source_persistent_no_pdl": "cake_dsv4_506e02d3d583a84bacdd",
        "fp8_h128_prefill_source_persistent_uniform": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h128_prefill_source_persistent_uniform_k64": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h128_prefill_source_persistent_uniform_k64_split": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c5": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c9": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_s3": "cake_dsv4_009fa784c7206613d7b7",
        "fp8_h64_prefill_source_persistent_m64": "cake_dsv4_5a02e8137c1b1bc7c58a",
        "fp8_h64_prefill_source_persistent_m64_multi_tile": "cake_dsv4_f647b239789ffbbdbd90",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4": "cake_dsv4_f647b239789ffbbdbd90",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_bcl": "cake_dsv4_f647b239789ffbbdbd90",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_l2pf": "cake_dsv4_f647b239789ffbbdbd90",
        "fp8_h64_source_exact": "cake_dsv4_e5e9d7f2a2798aec29ff",
        "fp8_h8_h16_source_exact": "cake_dsv4_fee38be0a1a558bc4cf4",
        "fp8_lowhead_h64": "cake_dsv4_cbf3168c696847fd3f69",
        "fp8_lowhead_one_partition": "cake_dsv4_b7c345c94f3a216b24e5",
        "fp8_lowhead_prefill": "cake_dsv4_d9d623599c9697a9ccf3",
        "fp8_lowhead_prefill_overlap": "cake_dsv4_d9d623599c9697a9ccf3",
        "fp8_lowhead_prefill_vblock": "cake_dsv4_d9d623599c9697a9ccf3",
        "split_reduce": "cake_dsv4_5f640d72462f26d33803",
    },
    "sm_103a": {
        "bf16_h128_prefill_v42": "cake_dsv4_13091b5bf0612783f02d",
        "bf16_h128_prefill_v42_snake": "cake_dsv4_cf7e7f261a5e1c42f98f",
        "bf16_h128_split5_reduce": "cake_dsv4_e880e7c12a67fe035a37",
        "bf16_h128_swa128": "cake_dsv4_16749fca629bdea1d909",
        "bf16_h128_topk128x_row_first": "cake_dsv4_36c80272414ea39a56d2",
        "bf16_h128_topk128x_row_first_vsplit": "cake_dsv4_51eee53d2d4f8b5344e9",
        "bf16_h128_topk128x_split4_sm100": "cake_dsv4_3eaffaa117944ce5e14e",
        "bf16_h128_topk4x_v52": "cake_dsv4_1aefb39e2f65a24b1628",
        "bf16_h16_h32_swa128_v44": "cake_dsv4_750835d54fe97cfc7ad4",
        "bf16_h32_topk128x_early_v47": "cake_dsv4_f81486848bf3b4e05349",
        "bf16_h64_compressed_q8_v38": "cake_dsv4_82ca6fee14338e07ece4",
        "bf16_h64_compressed_reduce": "cake_dsv4_50a2c6ca93259d30db6b",
        "bf16_h64_guard_q_tma_batch_r25": "cake_dsv4_1cd047de4a9a94e4f214",
        "bf16_h64_prefill": "cake_dsv4_7d6d09e037a1a9e99d87",
        "bf16_h8_h16_source_exact": "cake_dsv4_089147c4b5a5062fab1b",
        "bf16_h8_swa128_v43": "cake_dsv4_7850190a53cc07aba85a",
        "bf16_swa128_single_cta": "cake_dsv4_13f0ca13f90a32070cfe",
        "fp8_h128_prefill_source_persistent": "cake_dsv4_7c51b4b6d5b345d4fe03",
        "fp8_h128_prefill_source_persistent_no_pdl": "cake_dsv4_132f453f8df4a8300d5c",
        "fp8_h128_prefill_source_persistent_uniform": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h128_prefill_source_persistent_uniform_k64": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h128_prefill_source_persistent_uniform_k64_split": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c5": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c9": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_s3": "cake_dsv4_8bcdc9f41a3d74231d15",
        "fp8_h64_prefill_source_persistent_m64": "cake_dsv4_40932cb442b76026206f",
        "fp8_h64_prefill_source_persistent_m64_multi_tile": "cake_dsv4_3fce080f96f6dc5e9f8c",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4": "cake_dsv4_3fce080f96f6dc5e9f8c",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_bcl": "cake_dsv4_3fce080f96f6dc5e9f8c",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_l2pf": "cake_dsv4_3fce080f96f6dc5e9f8c",
        "fp8_h64_source_exact": "cake_dsv4_e5e9d7f2a2798aec29ff",
        "fp8_h8_h16_source_exact": "cake_dsv4_ff4aa022aa97fded6032",
        "fp8_lowhead_h64": "cake_dsv4_cbf3168c696847fd3f69",
        "fp8_lowhead_one_partition": "cake_dsv4_b78d35b605cb10daf20f",
        "fp8_lowhead_prefill": "cake_dsv4_e21760e2024d7c1e85c9",
        "fp8_lowhead_prefill_overlap": "cake_dsv4_e21760e2024d7c1e85c9",
        "fp8_lowhead_prefill_vblock": "cake_dsv4_e21760e2024d7c1e85c9",
        "split_reduce": "cake_dsv4_5f640d72462f26d33803",
    },
    "sm_107a": {
        "fp8_h128_prefill_source_persistent": "cake_dsv4_77fc7066e2ba5bebb062",
        "fp8_h128_prefill_source_persistent_uniform": "cake_dsv4_4a6718b7a2ca825d8800",
        "fp8_h128_prefill_source_persistent_uniform_k64": "cake_dsv4_384b6be5c7906a440e61",
        "fp8_h128_prefill_source_persistent_uniform_k64_split": "cake_dsv4_d01ae8250de69f1360ee",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce": "cake_dsv4_e2c23db7ef9c580df1c5",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c5": "cake_dsv4_5b9d14c8ac3031e13487",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_c9": "cake_dsv4_7505ea6a749435f33f86",
        "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce_s3": "cake_dsv4_e1ae8472ce68a65facf8",
        "fp8_h64_prefill_source_persistent_m64": "cake_dsv4_df5d069562864950e48b",
        "fp8_h64_prefill_source_persistent_m64_multi_tile": "cake_dsv4_ffa4216b35dae8d46543",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4": "cake_dsv4_8afa079cbcc688b8b5b9",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_bcl": "cake_dsv4_27a5d41ecb0fd5ecec7b",
        "fp8_h64_prefill_source_persistent_m64_multi_tile_kv4_l2pf": "cake_dsv4_7a104353fbcabeae6af8",
        "fp8_lowhead_prefill": "cake_dsv4_8625d6529808d0d51fc2",
        "fp8_lowhead_prefill_overlap": "cake_dsv4_25addc7a80a1e13c3ece",
        "fp8_lowhead_prefill_vblock": "cake_dsv4_c51bf1f09fbe07ac3be6",
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
    """Every variant of ``arch`` (BF16 / FP8 and, where it has them, NVFP4), in key order."""
    variants = {
        variant: _registration(arch, variant) for variant in VARIANTS.get(arch, {})
    }
    nvfp4 = _nvfp4_variants(arch) if arch in _NVFP4_IDENTITIES else {}
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
            f"CAKE DSv4 variant has no generated source contract: {variant} ({arch})"
            + (
                " (SM107 runs the FP8 decode route's persistent, M64 and low-head "
                "prefill programs only)"
                if arch == "sm_107a"
                else ""
            )
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
