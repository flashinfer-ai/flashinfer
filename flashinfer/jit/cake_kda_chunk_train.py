"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Native JIT loading for the generated chunked KDA training-backward kernels.

SM100a / SM103a.  The registry is filled mechanically from the resolved
production builds: one device/binding source pair per backward stage and
architecture.  Bindings encode TMA descriptors on the host on every call into a
caller-owned device workspace, so tensor addresses may change between calls.
"""

from functools import cache
from pathlib import Path

import torch

from . import env as jit_env
from .core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

STAGES = (
    "prep",
    "wy",
    "fwdh",
    "dav",
    "dhu",
    "dqkg",
    "intra",
    "gate_epilogue",
    "qk_epilogue",
    "finalize",
)

# MODULES[stage][arch] -> generated module record (mechanically filled).
MODULES = {
    "prep": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_72f262735fdac648a5a4",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_72f262735fdac648a5a4_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_72f262735fdac648a5a4",
            "arg_plan": (
                ("buffer", "g_raw"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "v"),
                ("buffer", "beta"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "aqk"),
                ("buffer", "gk_out"),
                ("buffer", "vb_out"),
                ("buffer", "kb_out"),
                ("buffer", "qg_out"),
                ("buffer", "kg_out"),
                ("buffer", "ke_out"),
                ("buffer", "qe_out"),
                ("buffer", "aqk_tril"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_heads"),
                ("parameter", "group"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_72f262735fdac648a5a4_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_72f262735fdac648a5a4_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 4096,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_04552a349834008d18cf",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_04552a349834008d18cf_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_04552a349834008d18cf",
            "arg_plan": (
                ("buffer", "g_raw"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "v"),
                ("buffer", "beta"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "aqk"),
                ("buffer", "gk_out"),
                ("buffer", "vb_out"),
                ("buffer", "kb_out"),
                ("buffer", "qg_out"),
                ("buffer", "kg_out"),
                ("buffer", "ke_out"),
                ("buffer", "qe_out"),
                ("buffer", "aqk_tril"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_heads"),
                ("parameter", "group"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_04552a349834008d18cf_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_04552a349834008d18cf_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 4096,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "wy": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_365cc414e3ad612e9634",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_365cc414e3ad612e9634_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_365cc414e3ad612e9634",
            "arg_plan": (
                ("tma_buffer", "akk_tma"),
                ("tma_buffer", "vb_tma"),
                ("tma_buffer", "kb_tma"),
                ("buffer", "u_out"),
                ("buffer", "w_out"),
                ("parameter", "num_heads"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_365cc414e3ad612e9634_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_365cc414e3ad612e9634_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_d32d54edcf58e0ee45f5",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_d32d54edcf58e0ee45f5_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_d32d54edcf58e0ee45f5",
            "arg_plan": (
                ("tma_buffer", "akk_tma"),
                ("tma_buffer", "vb_tma"),
                ("tma_buffer", "kb_tma"),
                ("buffer", "u_out"),
                ("buffer", "w_out"),
                ("parameter", "num_heads"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_d32d54edcf58e0ee45f5_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_d32d54edcf58e0ee45f5_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "fwdh": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_f058bceb35ac77dd8277",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_f058bceb35ac77dd8277_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_f058bceb35ac77dd8277",
            "arg_plan": (
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "kg_tma"),
                ("buffer", "u"),
                ("buffer", "gk"),
                ("buffer", "h_out"),
                ("buffer", "v_new"),
                ("parameter", "num_heads"),
                ("parameter", "seq_len"),
                ("parameter", "num_chunks"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_f058bceb35ac77dd8277_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_f058bceb35ac77dd8277_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 91136,
            "tma_workspace_bytes": 256,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_6d7d29fb2f4abc9f0dcf",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_6d7d29fb2f4abc9f0dcf_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_6d7d29fb2f4abc9f0dcf",
            "arg_plan": (
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "kg_tma"),
                ("buffer", "u"),
                ("buffer", "gk"),
                ("buffer", "h_out"),
                ("buffer", "v_new"),
                ("parameter", "num_heads"),
                ("parameter", "seq_len"),
                ("parameter", "num_chunks"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_6d7d29fb2f4abc9f0dcf_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_6d7d29fb2f4abc9f0dcf_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 91136,
            "tma_workspace_bytes": 256,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "dav": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_efd1f91318b9eb777c56",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_efd1f91318b9eb777c56_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_efd1f91318b9eb777c56",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vnew_tma"),
                ("tma_buffer", "aqk_tma"),
                ("buffer", "dAqk"),
                ("buffer", "dv1"),
                ("parameter", "num_heads"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_efd1f91318b9eb777c56_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_efd1f91318b9eb777c56_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_45f943d52b2424e3267d",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_45f943d52b2424e3267d_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_45f943d52b2424e3267d",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vnew_tma"),
                ("tma_buffer", "aqk_tma"),
                ("buffer", "dAqk"),
                ("buffer", "dv1"),
                ("parameter", "num_heads"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_45f943d52b2424e3267d_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_45f943d52b2424e3267d_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "dhu": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_75b231015bb20368c6df",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_75b231015bb20368c6df_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_75b231015bb20368c6df",
            "arg_plan": (
                ("tma_buffer", "kg_tma"),
                ("tma_buffer", "qg_tma"),
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "do_tma"),
                ("buffer", "dv1"),
                ("buffer", "gk"),
                ("buffer", "dh_out"),
                ("buffer", "dv2"),
                ("parameter", "num_heads"),
                ("parameter", "seq_len"),
                ("parameter", "num_chunks"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_75b231015bb20368c6df_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_75b231015bb20368c6df_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 140288,
            "tma_workspace_bytes": 512,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_19070e33f884bc4f33bf",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_19070e33f884bc4f33bf_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_19070e33f884bc4f33bf",
            "arg_plan": (
                ("tma_buffer", "kg_tma"),
                ("tma_buffer", "qg_tma"),
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "do_tma"),
                ("buffer", "dv1"),
                ("buffer", "gk"),
                ("buffer", "dh_out"),
                ("buffer", "dv2"),
                ("parameter", "num_heads"),
                ("parameter", "seq_len"),
                ("parameter", "num_chunks"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_19070e33f884bc4f33bf_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_19070e33f884bc4f33bf_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 140288,
            "tma_workspace_bytes": 512,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "dqkg": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_f4121c1ac1fcc00ddc64",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_f4121c1ac1fcc00ddc64_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_f4121c1ac1fcc00ddc64",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vn_tma"),
                ("tma_buffer", "dv2_tma"),
                ("tma_buffer", "v_tma"),
                ("tma_buffer", "h_tma"),
                ("tma_buffer", "dh_tma"),
                ("tma_buffer", "akk_tma"),
                ("buffer", "h_ptr"),
                ("buffer", "dh_ptr"),
                ("buffer", "k_ptr"),
                ("buffer", "q_ptr"),
                ("buffer", "v_ptr"),
                ("buffer", "gk"),
                ("buffer", "beta"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("buffer", "dAkk_out"),
                ("buffer", "dv_out"),
                ("parameter", "num_heads"),
                ("parameter", "num_chunks"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_f4121c1ac1fcc00ddc64_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_f4121c1ac1fcc00ddc64_binding.cu",
            ),
            "block": (320, 1, 1),
            "dynamic_smem_bytes": 174080,
            "tma_workspace_bytes": 896,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_20f35b9f77e23e9f5283",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_20f35b9f77e23e9f5283_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_20f35b9f77e23e9f5283",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vn_tma"),
                ("tma_buffer", "dv2_tma"),
                ("tma_buffer", "v_tma"),
                ("tma_buffer", "h_tma"),
                ("tma_buffer", "dh_tma"),
                ("tma_buffer", "akk_tma"),
                ("buffer", "h_ptr"),
                ("buffer", "dh_ptr"),
                ("buffer", "k_ptr"),
                ("buffer", "q_ptr"),
                ("buffer", "v_ptr"),
                ("buffer", "gk"),
                ("buffer", "beta"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("buffer", "dAkk_out"),
                ("buffer", "dv_out"),
                ("parameter", "num_heads"),
                ("parameter", "num_chunks"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_20f35b9f77e23e9f5283_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_20f35b9f77e23e9f5283_binding.cu",
            ),
            "block": (320, 1, 1),
            "dynamic_smem_bytes": 174080,
            "tma_workspace_bytes": 896,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "intra": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_8f902df21bdc21f29ba4",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_8f902df21bdc21f29ba4_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_8f902df21bdc21f29ba4",
            "arg_plan": (
                ("buffer", "dAqk"),
                ("buffer", "dAkk"),
                ("buffer", "gk"),
                ("buffer", "k_e"),
                ("buffer", "q_e"),
                ("buffer", "beta"),
                ("buffer", "dq_f"),
                ("buffer", "dk_f"),
                ("buffer", "dg_f"),
                ("buffer", "db_f"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("parameter", "num_heads"),
                ("parameter", "num_qk_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_8f902df21bdc21f29ba4_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_8f902df21bdc21f29ba4_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 103680,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_2b8071d9377000d0e36b",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_2b8071d9377000d0e36b_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_2b8071d9377000d0e36b",
            "arg_plan": (
                ("buffer", "dAqk"),
                ("buffer", "dAkk"),
                ("buffer", "gk"),
                ("buffer", "k_e"),
                ("buffer", "q_e"),
                ("buffer", "beta"),
                ("buffer", "dq_f"),
                ("buffer", "dk_f"),
                ("buffer", "dg_f"),
                ("buffer", "db_f"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("parameter", "num_heads"),
                ("parameter", "num_qk_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_2b8071d9377000d0e36b_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_2b8071d9377000d0e36b_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 103680,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "gate_epilogue": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_0ad4626786c206c7f6d6",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_0ad4626786c206c7f6d6_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_0ad4626786c206c7f6d6",
            "arg_plan": (
                ("buffer", "dg_intra"),
                ("buffer", "g_raw"),
                ("buffer", "db_total"),
                ("buffer", "beta_raw"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "dg_out"),
                ("buffer", "dbeta"),
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("parameter", "num_heads"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_0ad4626786c206c7f6d6_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_0ad4626786c206c7f6d6_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 12288,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_34e0c3e6404417af0349",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_34e0c3e6404417af0349_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_34e0c3e6404417af0349",
            "arg_plan": (
                ("buffer", "dg_intra"),
                ("buffer", "g_raw"),
                ("buffer", "db_total"),
                ("buffer", "beta_raw"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "dg_out"),
                ("buffer", "dbeta"),
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("parameter", "num_heads"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_34e0c3e6404417af0349_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_34e0c3e6404417af0349_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 12288,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "qk_epilogue": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_d086ff5b482109313f4f",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_d086ff5b482109313f4f_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_d086ff5b482109313f4f",
            "arg_plan": (
                ("buffer", "dq_intra"),
                ("buffer", "dk_intra"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "q_rstd"),
                ("buffer", "k_rstd"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_v_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_d086ff5b482109313f4f_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_d086ff5b482109313f4f_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 0,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_c05735dd9f5a1a7890f3",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_c05735dd9f5a1a7890f3_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_c05735dd9f5a1a7890f3",
            "arg_plan": (
                ("buffer", "dq_intra"),
                ("buffer", "dk_intra"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "q_rstd"),
                ("buffer", "k_rstd"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_v_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_c05735dd9f5a1a7890f3_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_c05735dd9f5a1a7890f3_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 0,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "finalize": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_516c96214ba5df6fc734",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_516c96214ba5df6fc734_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_516c96214ba5df6fc734",
            "arg_plan": (
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("buffer", "dA_log"),
                ("buffer", "dt_bias_grad"),
                ("parameter", "num_chunks"),
                ("parameter", "num_heads"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_516c96214ba5df6fc734_kernel.cu",
                "csrc/kda/chunk_train/sm_100a/cake_kda_chunk_train_516c96214ba5df6fc734_binding.cu",
            ),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_93b2b013aa41ffaa96dc",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_93b2b013aa41ffaa96dc_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_93b2b013aa41ffaa96dc",
            "arg_plan": (
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("buffer", "dA_log"),
                ("buffer", "dt_bias_grad"),
                ("parameter", "num_chunks"),
                ("parameter", "num_heads"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_93b2b013aa41ffaa96dc_kernel.cu",
                "csrc/kda/chunk_train/sm_103a/cake_kda_chunk_train_93b2b013aa41ffaa96dc_binding.cu",
            ),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
}


def device_arch(device=None):
    """Map the current CUDA device to the exported architecture tag."""
    capability = torch.cuda.get_device_capability(device)
    targets = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
    if capability not in targets:
        raise NotImplementedError(
            "the chunked KDA training backward requires SM100a or SM103a"
        )
    return targets[capability]


def _source_path(relative):
    installed = jit_env.FLASHINFER_CSRC_DIR / Path(relative).relative_to("csrc")
    if installed.is_file():
        return installed
    return Path(__file__).resolve().parents[2] / relative


@cache
def spec(stage, arch):
    record = MODULES[stage][arch]
    flags = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]
    include = jit_env.FLASHINFER_INCLUDE_DIR
    if not include.is_dir():
        include = Path(__file__).resolve().parents[2] / "include"
    sources = [_source_path(path) for path in record["sources"]]
    return gen_jit_spec(
        name=record["cache_name"],
        sources=sources,
        extra_cuda_cflags=[*flags, *record["compile_flags"]],
        extra_include_paths=[
            sources[-1].parent,
            _source_path("csrc/tvm_ffi_utils.h").parent,
            include,
        ],
    )


@cache
def load(stage, arch):
    return spec(stage, arch).build_and_load()


class NativeKernel:
    """One generated stage; named bindings are mapped onto the exported argument plan.

    ``launch(grid=(x, y, z), **bindings)`` binds tensors and scalars by their
    exported names.  The caller-owned TMA descriptor workspace and the grid
    are filled in automatically; descriptors are re-encoded on every call.
    """

    def __init__(self, stage, arch=None, device=None):
        self.stage = stage
        self.arch = arch or device_arch(device)
        record = MODULES[stage][self.arch]
        self.record = record
        native = load(stage, self.arch)
        self._call = getattr(native, record["ffi_entry"])
        self._arg_plan = tuple(tuple(item) for item in record["arg_plan"])
        workspace_bytes = int(record["tma_workspace_bytes"])
        self.descriptor_storage = (
            torch.empty(
                workspace_bytes,
                dtype=torch.uint8,
                device=device if device is not None else "cuda",
            )
            if workspace_bytes
            else None
        )

    def launch(self, *, grid, **bindings):
        grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
        args = []
        used = set()
        for kind, name in self._arg_plan:
            if kind == "grid":
                args.append(grid[("grid_x", "grid_y", "grid_z").index(name)])
            elif kind == "workspace":
                args.append(self.descriptor_storage)
            else:
                if name not in bindings:
                    raise KeyError(f"{self.stage}: missing binding {name!r}")
                args.append(bindings[name])
                used.add(name)
        unexpected = set(bindings) - used
        if unexpected:
            raise KeyError(f"{self.stage}: unexpected bindings {sorted(unexpected)!r}")
        return self._call(*args)


@cache
def kernel(stage, arch):
    """Return the process-wide NativeKernel for ``stage`` on ``arch``."""
    return NativeKernel(stage, arch)
