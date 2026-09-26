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
    "cake_kimi_k3_mla_fp8_paged_attention_0ce2970cff662167c83a": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_0ce2970cff662167c83a_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_0ce2970cff662167c83a_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "5d37daf5bb12daed77a9bb1ac514e1cbfe042ad98a0e5209c4df70112cd01f96",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_0d9120b51c4ef32af3bf": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_0d9120b51c4ef32af3bf_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_0d9120b51c4ef32af3bf_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "25f2a96518d795f18efc2c9241f6a262f6b7fd08a6b7a768290a180b926474d4",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_155582f3d600c0a23cf7": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_155582f3d600c0a23cf7_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_155582f3d600c0a23cf7_binding.cu",
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
        "closure_sha256": "dcd3d8431497eec426674fd5479e4454ed7cf47d9b6ccdfadfe752bbcd93867a",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_2e58e3f7d166b883b583": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_2e58e3f7d166b883b583_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_2e58e3f7d166b883b583_binding.cu",
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
        "closure_sha256": "70060460c351429a59d91a36887e10ebadfcfb3e166596bcf8ed17ea0e88ce02",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_352ed2333cff56936cbc": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_352ed2333cff56936cbc_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_352ed2333cff56936cbc_binding.cu",
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
        "closure_sha256": "42aa01797ff1303d959a893fc01324971641620703a7188f0fd90dbd4cb9a9d8",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_3a41df07e915e77ec73e": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_3a41df07e915e77ec73e_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_3a41df07e915e77ec73e_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "003439918dcb4bc96ab98de0f4c592462a4571da3656accca1d3c7bb41253c0b",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_3c2eb2f47a24a9cda244": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_3c2eb2f47a24a9cda244_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_3c2eb2f47a24a9cda244_binding.cu",
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
        "closure_sha256": "25e21976f8c1b94a823b85340d64a45e4e78a08cace7dceb9b800c0f0af03c8e",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_3e4b71b082cf5d3abceb": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_3e4b71b082cf5d3abceb_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_3e4b71b082cf5d3abceb_binding.cu",
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
        "closure_sha256": "410b56d5763eb165a938e27f267f9bf47160656d26e359e9edea6ac0d0d36290",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_41f768abc88f639eb2ff": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_41f768abc88f639eb2ff_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_41f768abc88f639eb2ff_binding.cu",
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
        "closure_sha256": "dc8ca0a8595b2b9d1482a126413d38c9336ed96d5a6ef46f851cba694282103b",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_5efd28ecc1adf10f89cd": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_5efd28ecc1adf10f89cd_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_5efd28ecc1adf10f89cd_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "ad0fc73ce1e23390bdb888c885d827dc9fceb5b124e3ad961ddf2a3e2aeb9c47",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_6dc0632503e227b97d1d": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_6dc0632503e227b97d1d_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_6dc0632503e227b97d1d_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "1f40c4314263c0e12a81f6cbff55f2fb512ffaadd1103ae8676706cc9f1b7fa3",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_8aed522f087f6e81c77a": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_8aed522f087f6e81c77a_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_8aed522f087f6e81c77a_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "9a75192bc5216aa072f0ab5496d99d27fca1db5f67de8e8aa1c97099c34bfad1",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_9a78cca8b8ba1f758ed5": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_9a78cca8b8ba1f758ed5_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_9a78cca8b8ba1f758ed5_binding.cu",
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
        "closure_sha256": "37fea6da22bba3aac2d362006b587a4ce75745c2a963c501d789629444af5456",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_a5d38bf637ec40bc8083": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_a5d38bf637ec40bc8083_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_a5d38bf637ec40bc8083_binding.cu",
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
        "closure_sha256": "913f6b09f335b755807a2abc2218a293328315dd8833f130825abf7d3ae0d530",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_a68072b28bedc16b32b6": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_a68072b28bedc16b32b6_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_a68072b28bedc16b32b6_binding.cu",
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
        "closure_sha256": "2008db061159fe55aa4a013eed1da2ceff7bf34e9478a069755803477e6b4403",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_b181f1276ae9155fbf01": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_b181f1276ae9155fbf01_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_b181f1276ae9155fbf01_binding.cu",
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
        "closure_sha256": "d8c3e24a6e8ffa5e652d56746d8e8e5fcf6bc6face93a64dc115a70862789fbc",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_cb26f447f8114959d885": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_cb26f447f8114959d885_kernel.cu",
            "cake_kimi_k3_mla/sm_103a/cake_kimi_k3_mla_fp8_paged_attention_cb26f447f8114959d885_binding.cu",
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
        "closure_sha256": "af9eac7a38b65e193c5200d6d710136ed6b26d6aa8aadf547d98818039afcec9",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_d051ba05067b9b4188a6": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_d051ba05067b9b4188a6_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_d051ba05067b9b4188a6_binding.cu",
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
        "closure_sha256": "1ec8ecf2074981d3c902d786715b56b30921c8505a89a92e18340636b928fce5",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_d135c689aa2149f59eba": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_d135c689aa2149f59eba_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_d135c689aa2149f59eba_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "da28f8fac7cf00db645ff0545faf586cd24e4185dd3330f82acf443cef04efa6",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_mla_fp8_paged_attention_dcea66af29f10a1c20a5": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_dcea66af29f10a1c20a5_kernel.cu",
            "cake_kimi_k3_mla/sm_100a/cake_kimi_k3_mla_fp8_paged_attention_dcea66af29f10a1c20a5_binding.cu",
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "07695241703f6c1bd66238b6bc1774a9a0072949862538c6f8eff8444def081a",
        "tma_workspace_bytes": 0,
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "main_rt16__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt16",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_352ed2333cff56936cbc",
    },
    "main_rt16__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt16",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_b181f1276ae9155fbf01",
    },
    "main_rt32__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt32",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_a5d38bf637ec40bc8083",
    },
    "main_rt32__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt32",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_cb26f447f8114959d885",
    },
    "main_rt48__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt48",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_d051ba05067b9b4188a6",
    },
    "main_rt48__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt48",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_3c2eb2f47a24a9cda244",
    },
    "main_rt64__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt64",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_2e58e3f7d166b883b583",
    },
    "main_rt64__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt64",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_3e4b71b082cf5d3abceb",
    },
    "main_rt96__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_rt96",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_9a78cca8b8ba1f758ed5",
    },
    "main_rt96__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_rt96",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_155582f3d600c0a23cf7",
    },
    "main_wide__sm_100a": {
        "arch": "sm_100a",
        "kind": "main_wide",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_a68072b28bedc16b32b6",
    },
    "main_wide__sm_103a": {
        "arch": "sm_103a",
        "kind": "main_wide",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_41f768abc88f639eb2ff",
    },
    "reduce_cta__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_cta",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_d135c689aa2149f59eba",
    },
    "reduce_cta__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_cta",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_8aed522f087f6e81c77a",
    },
    "reduce_w1__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_w1",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_6dc0632503e227b97d1d",
    },
    "reduce_w1__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_w1",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_3a41df07e915e77ec73e",
    },
    "reduce_w2__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_w2",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_0ce2970cff662167c83a",
    },
    "reduce_w2__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_w2",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_5efd28ecc1adf10f89cd",
    },
    "reduce_w4__sm_100a": {
        "arch": "sm_100a",
        "kind": "reduce_w4",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_dcea66af29f10a1c20a5",
    },
    "reduce_w4__sm_103a": {
        "arch": "sm_103a",
        "kind": "reduce_w4",
        "module": "cake_kimi_k3_mla_fp8_paged_attention_0d9120b51c4ef32af3bf",
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
