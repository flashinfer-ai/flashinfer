"""JIT loader for the CAKE-generated Kimi-K3 MLA FP8 paged-attention kernels (SM100 / SM103).

The generated CUDA lives in ``csrc/cake_kimi_k3_mla/<arch>/`` (one ``*_kernel.cu`` +
``*_binding.cu`` pair per physical module).  ``MODULES`` and ``ROUTES`` are written by the
Cake generated-program exporter (``exports/kimi_k3_mla_fp8_paged_attention/export.py``):
``MODULES`` holds one record per physical module (sources, compile flags, FFI entry and
argument plan) and ``ROUTES`` maps ``"<kind>__<arch>"`` (``main_rt16`` .. ``main_rt96``,
``reduce_w1`` / ``reduce_w2`` / ``reduce_w4``, ``reduce_cta``) to its module name.
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

_ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
PACKAGE_DIR = "cake_kimi_k3_mla"

# Filled by the exporter's ``integrate`` step (see module docstring).
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_mla_fp8_paged_attention_0e323712fcaac09bab09": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_0e323712fcaac09bab09_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_0e323712fcaac09bab09_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "ba2e65b4cc0f80217ab596f1f173728b706c52fe54ea511c295e8ac15199921d",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_153d31a72ba8d9e7f725": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_153d31a72ba8d9e7f725_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_153d31a72ba8d9e7f725_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "e01127cdd160b659cd70d4433de334e8e869e1c2ce21ab71e53e259d779edb7c",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_1881f24e3e9012bfe20f": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_1881f24e3e9012bfe20f_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_1881f24e3e9012bfe20f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "c3debd8639babe93a8e4bd507c7f22ce43beefdfaa1a583197d2800b4a5c13b8",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_1982a7ffbfdaaf134793": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_1982a7ffbfdaaf134793_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_1982a7ffbfdaaf134793_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "a2d550ec3d4f1242124b7b5e2781831a4670cc9787e729cb3a7379c56d8368eb",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_259656ea112d42e5b002": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_259656ea112d42e5b002_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_259656ea112d42e5b002_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "50097bfd9ede7e78eb1a5c41e2d7f038f029c298025a401dc4506864fd6061b3",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_26109da8d76ccbc875e7": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_26109da8d76ccbc875e7_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_26109da8d76ccbc875e7_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "2f76e9417bc81f18c5f77281bf300b4793c94f90ef4d50d9203a42e02e66b1c2",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_30b82d3b12c32ae2ff40": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_30b82d3b12c32ae2ff40_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_30b82d3b12c32ae2ff40_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "fefb132b53b3f6fb82c3dcd7c6b1393a59c21140f7d493fec26a48cc4559daab",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_5284c5cd1ac2f078e483": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_5284c5cd1ac2f078e483_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_5284c5cd1ac2f078e483_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "108348710ef0029bc493962b7d9a52b71f52a195f62ee6401486d00e568ed5ca",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_6afd9bdce713a7f1c00a": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_6afd9bdce713a7f1c00a_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_6afd9bdce713a7f1c00a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "8670acacd844ec8b9cbf4d22b92be48d5ef799a579cc965dfb4904b9377759ec",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_70d89deb9d99461debde": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_70d89deb9d99461debde_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_70d89deb9d99461debde_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "ed6fa2b53ead09d1cb48c2363edad049fc0fd8637dd4baf467b0e6c6a1b7f38a",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_71bb33f073a8b4697e59": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_71bb33f073a8b4697e59_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_71bb33f073a8b4697e59_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_kr"],
            ["tma_buffer", "tmap_v"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "328fb22979fe8a3a4d028dec95c4e786656911b0d5231ef572db1d77a5b4b24c",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_99c9fdf5fb12f13bb732": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_99c9fdf5fb12f13bb732_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_99c9fdf5fb12f13bb732_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "e5f5de6de3206d502b572614e4d201c71fa08ebbe1d2e640306739b5e7575d7c",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_9d753c4cf061cdaaf506": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_9d753c4cf061cdaaf506_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_9d753c4cf061cdaaf506_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "ca4bc0f92058d4822f8f894fb8143a39e88c72d7edb273e55fd6d08557fd44c0",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_c086a35e2e2b2554f26c": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_c086a35e2e2b2554f26c_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_c086a35e2e2b2554f26c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "3f5ed28d69a178e7e809b0ffa142e1d178c4d0df1017bac4c958c7a81f9436a6",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_c43301e3628e995cecb9": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_c43301e3628e995cecb9_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_c43301e3628e995cecb9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_kr"],
            ["tma_buffer", "tmap_v"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "13b595099c1daf931aa7504435b5dc6dfd958e0cbfc3692890d187cd397526b9",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_c7e09044a1a802b6bf2e": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_c7e09044a1a802b6bf2e_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_c7e09044a1a802b6bf2e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "92f8dced6920c1e2cc8d67e32f8a4942cf5c2f85579b8833883b5edc5072fa44",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_db439aa58c8882632cbb": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_db439aa58c8882632cbb_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_db439aa58c8882632cbb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "7bd5ac164ab2cc7a0b29003c60ed9ff49ef14353eeba146df2bcbc643e5056fa",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_dfaae2a940c898594c4e": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_dfaae2a940c898594c4e_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_dfaae2a940c898594c4e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "e5f416a9753015a1e75b1aba23d6e9b658fe89ee3ff08f67349fbb04446d148a",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_e3b92e674f2431f72452": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_e3b92e674f2431f72452_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_e3b92e674f2431f72452_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "04214e2998fa1e66ee761a1de24c080e2aa0352040f37829e152fef83afb089f",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_f5e19ae39137a64d0367": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_f5e19ae39137a64d0367_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_f5e19ae39137a64d0367_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "e531f29780829428123ac95f2a4ef8a98de4d8a6711a1dddc98b7511a2a34d03",
        "tma_workspace_bytes": 0,
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "main_rt16__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt16",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_26109da8d76ccbc875e7",
    },
    "main_rt16__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt16",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_c7e09044a1a802b6bf2e",
    },
    "main_rt32__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt32",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_70d89deb9d99461debde",
    },
    "main_rt32__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt32",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_259656ea112d42e5b002",
    },
    "main_rt48__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt48",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_153d31a72ba8d9e7f725",
    },
    "main_rt48__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt48",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_e3b92e674f2431f72452",
    },
    "main_rt64__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt64",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_6afd9bdce713a7f1c00a",
    },
    "main_rt64__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt64",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_f5e19ae39137a64d0367",
    },
    "main_rt96__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt96",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_c086a35e2e2b2554f26c",
    },
    "main_rt96__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt96",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_1881f24e3e9012bfe20f",
    },
    "main_wide__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_wide",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_c43301e3628e995cecb9",
    },
    "main_wide__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_wide",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_71bb33f073a8b4697e59",
    },
    "reduce_cta__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_cta",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_99c9fdf5fb12f13bb732",
    },
    "reduce_cta__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_cta",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_30b82d3b12c32ae2ff40",
    },
    "reduce_w1__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_w1",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_1982a7ffbfdaaf134793",
    },
    "reduce_w1__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_w1",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_9d753c4cf061cdaaf506",
    },
    "reduce_w2__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_w2",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_5284c5cd1ac2f078e483",
    },
    "reduce_w2__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_w2",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_0e323712fcaac09bab09",
    },
    "reduce_w4__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_w4",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_dfaae2a940c898594c4e",
    },
    "reduce_w4__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_w4",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_db439aa58c8882632cbb",
    },
}


def route_key(kind: str, arch: str) -> str:
    return f"{kind}__{arch}"


def get_cake_kimi_k3_mla_route(kind: str, *, arch: str) -> dict[str, Any]:
    """Return the physical module record selected for ``kind`` on ``arch``."""
    try:
        route = ROUTES[route_key(kind, arch)]
    except KeyError as exc:
        raise ValueError(
            f"CAKE Kimi-K3 MLA has no generated route {kind!r} for {arch}"
        ) from exc
    module = MODULES[route["module"]]
    if module["arch"] != arch:
        raise ValueError(
            f"CAKE Kimi-K3 MLA route {kind!r} resolved to a {module['arch']} module on {arch}"
        )
    return dict(module, name=route["module"])


def _get_csrc_dir(arch: str) -> Path:
    if arch not in _ARCH_NVCC_FLAGS:
        raise ValueError(f"unsupported CAKE Kimi-K3 MLA architecture: {arch}")
    installed = jit_env.FLASHINFER_CSRC_DIR / PACKAGE_DIR / arch
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / PACKAGE_DIR / arch
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "CAKE Kimi-K3 MLA CUDA sources were not found. Checked:\n"
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


@functools.cache
def gen_cake_kimi_k3_mla_module(name: str) -> JitSpec:
    """JIT spec of one physical generated module (device + binding translation units)."""
    try:
        contract = MODULES[name]
    except KeyError as exc:
        raise ValueError(f"CAKE Kimi-K3 MLA has no generated module {name!r}") from exc
    arch = contract["arch"]
    csrc_dir = _get_csrc_dir(arch)
    sources = [csrc_dir / Path(src).name for src in contract["sources"]]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CAKE Kimi-K3 MLA generated sources were not found: "
            + ", ".join(str(path) for path in missing)
        )
    spec = gen_jit_spec(
        name=f"cake_kimi_k3_mla_{name}_{arch.replace('_', '')}",
        sources=sources,
        extra_cuda_cflags=[
            *_ARCH_NVCC_FLAGS[arch],
            *contract["compile_flags"],
            *contract.get("host_linkage_flags", ()),
        ],
        # The generated contract owns the fast-math decision.
        use_fast_math=False,
        extra_include_paths=[csrc_dir, csrc_dir.parent.parent, _get_include_dir()],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated CAKE Kimi-K3 MLA {name} JIT spec: {spec.name}")
    return spec


@functools.cache
def get_cake_kimi_k3_mla_module(name: str):
    loaded = gen_cake_kimi_k3_mla_module(name).build_and_load()
    logger.info(f"Loaded CAKE Kimi-K3 MLA {name} module")
    return loaded


__all__ = [
    "MODULES",
    "ROUTES",
    "gen_cake_kimi_k3_mla_module",
    "get_cake_kimi_k3_mla_module",
    "get_cake_kimi_k3_mla_route",
    "route_key",
]
