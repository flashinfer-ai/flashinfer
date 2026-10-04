# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration and rendered route chains of the generated Cake batch DeepGEMM FP8 programs.

``PROGRAMS`` lists every generated program once: its two translation units under
``csrc/cake_batch_deepgemm_fp8`` (one source shared by every architecture it runs
on), the exact compile flags of the source build, the tvm-ffi entry, the kernel
symbol and the id of its positional argument plan in ``ARG_PLANS``.  ``MODULES``
names one JIT module per (program, architecture).  ``SERVING_PROGRAMS`` maps a
packed-scale serving configuration to its program.  ``DISPATCH_TABLES`` holds, per
architecture and SM count, the ordered first-match route chain of the Cake
dispatcher rendered to Python: every predicate, launch grid, argument binding and
workspace shape is an expression over the five scalars ``(B, M, N, K,
expected_m)``; a string argument names a family tensor or workspace, an integer is
a scalar kernel argument.  All tables are written by the generated-program export;
do not edit them by hand.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

ARG_PLANS: dict[str, list[list[str]]] = {
    "plan0": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "SFA_bits"],
        ["buffer", "SFB_bits"],
        ["buffer", "masked_m"],
        ["tma_buffer", "C_tma"],
        ["parameter", "num_groups"],
        ["parameter", "shape_m"],
        ["parameter", "shape_n"],
        ["parameter", "grid_n"],
        ["parameter", "k_tiles"],
        ["parameter", "sf_cols"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan1": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C_tma"],
        ["tma_buffer", "SFA_packed"],
        ["tma_buffer", "SFB_packed"],
        ["buffer", "masked_m"],
        ["parameter", "num_groups"],
        ["parameter", "shape_m"],
        ["parameter", "N"],
        ["parameter", "K"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan2": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "SFA_bits"],
        ["buffer", "SFB_bits"],
        ["buffer", "masked_m"],
        ["buffer", "C"],
        ["parameter", "num_groups"],
        ["parameter", "shape_m"],
        ["parameter", "grid_n"],
        ["parameter", "k_tiles"],
        ["parameter", "sf_cols"],
        ["parameter", "scheduled_pair_blocks"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan3": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["tma_buffer", "C_tma"],
        ["buffer", "SFA_packed"],
        ["buffer", "SFB_packed"],
        ["buffer", "masked_m"],
        ["parameter", "num_groups"],
        ["parameter", "shape_m"],
        ["parameter", "compute_m_cap"],
        ["parameter", "N"],
        ["parameter", "K"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan4": [
        ["buffer", "SFA_bits"],
        ["buffer", "SFB_bits"],
        ["buffer", "SFA_packed"],
        ["buffer", "SFB_packed"],
        ["parameter", "num_groups"],
        ["parameter", "shape_m"],
        ["parameter", "N"],
        ["parameter", "K"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan5": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "A_scale"],
        ["buffer", "B_scale"],
        ["buffer", "masked_m"],
        ["buffer", "C"],
        ["parameter", "batch_size"],
        ["parameter", "shape_m"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan6": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B"],
        ["buffer", "SFA_bits"],
        ["buffer", "SFB_bits"],
        ["buffer", "masked_m"],
        ["buffer", "C"],
        ["parameter", "num_groups"],
        ["parameter", "shape_m"],
        ["parameter", "grid_n"],
        ["parameter", "k_tiles"],
        ["parameter", "sf_cols"],
        ["parameter", "scheduled_m_blocks"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "plan7": [
        ["tma_buffer", "A"],
        ["tma_buffer", "B_tensor"],
        ["tma_buffer", "C_tma"],
        ["buffer", "A_scale"],
        ["buffer", "B_scale"],
        ["buffer", "masked_m"],
        ["parameter", "num_groups"],
        ["parameter", "M"],
        ["parameter", "num_n_tiles"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_batch_deepgemm_fp8_2382d782c0bafc3a9878": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_2382d782c0bafc3a9878",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2382d782c0bafc3a9878_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2382d782c0bafc3a9878_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_29a2b409934055f8cdd0": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_29a2b409934055f8cdd0_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_29a2b409934055f8cdd0_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_29a8930160c38b72cff6": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_29a8930160c38b72cff6",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_29a8930160c38b72cff6_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_29a8930160c38b72cff6_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_2cd08ce62ec175db050b": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_2cd08ce62ec175db050b",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2cd08ce62ec175db050b_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2cd08ce62ec175db050b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_36afbebf9dfd01967336": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_36afbebf9dfd01967336",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_36afbebf9dfd01967336_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_36afbebf9dfd01967336_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_389e6c89e61b7d49a076": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_389e6c89e61b7d49a076_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_389e6c89e61b7d49a076_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_3dfe31e7f71ee9976e3e": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_3dfe31e7f71ee9976e3e",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_3dfe31e7f71ee9976e3e_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_3dfe31e7f71ee9976e3e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_103a"],
    },
    "cake_batch_deepgemm_fp8_508491c54eb3f469b37e": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_508491c54eb3f469b37e",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_508491c54eb3f469b37e_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_508491c54eb3f469b37e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_57c401152579ac3ac327": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_57c401152579ac3ac327",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_57c401152579ac3ac327_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_57c401152579ac3ac327_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_638db43d71890e22c8fe": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_638db43d71890e22c8fe",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_638db43d71890e22c8fe_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_638db43d71890e22c8fe_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_733a94c236b2ee6a8aee": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_733a94c236b2ee6a8aee",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_733a94c236b2ee6a8aee_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_733a94c236b2ee6a8aee_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_74557d5c854d22263fab": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_74557d5c854d22263fab",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_74557d5c854d22263fab_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_74557d5c854d22263fab_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan3",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_8730531b6d09e9544c01": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_8730531b6d09e9544c01",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_8730531b6d09e9544c01_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_8730531b6d09e9544c01_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_87a3c50a96af69fbae87": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_87a3c50a96af69fbae87",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_87a3c50a96af69fbae87_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_87a3c50a96af69fbae87_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan4",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_b974999e103cd7640cc7": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_b974999e103cd7640cc7",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_b974999e103cd7640cc7_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_b974999e103cd7640cc7_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan3",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_ba92e96364e30012640e": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_ba92e96364e30012640e",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ba92e96364e30012640e_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ba92e96364e30012640e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan5",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_bf96f77ec8235b35deac": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_bf96f77ec8235b35deac",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_bf96f77ec8235b35deac_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_bf96f77ec8235b35deac_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_c41147f6dbdf4c321445": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_c41147f6dbdf4c321445",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c41147f6dbdf4c321445_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c41147f6dbdf4c321445_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_ca298547e7b496acb4d3": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_ca298547e7b496acb4d3",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ca298547e7b496acb4d3_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ca298547e7b496acb4d3_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_d4262883ad8c2c6cef94": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_d4262883ad8c2c6cef94",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d4262883ad8c2c6cef94_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d4262883ad8c2c6cef94_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_103a"],
    },
    "cake_batch_deepgemm_fp8_d662bd14d9ad55d45263": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_d662bd14d9ad55d45263",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d662bd14d9ad55d45263_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d662bd14d9ad55d45263_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_e460881a1cdcaccf5860": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_e460881a1cdcaccf5860",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_e460881a1cdcaccf5860_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_e460881a1cdcaccf5860_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_eb727f8d280f83b52730": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_eb727f8d280f83b52730",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_eb727f8d280f83b52730_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_eb727f8d280f83b52730_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan6",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_ec43ea789e823a74035e": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_ec43ea789e823a74035e",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ec43ea789e823a74035e_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ec43ea789e823a74035e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan3",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan7",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_f376efa5168590657e71": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_f376efa5168590657e71",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_f376efa5168590657e71_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_f376efa5168590657e71_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan4",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_f8971314a3ef775f9c58": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_f8971314a3ef775f9c58",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_f8971314a3ef775f9c58_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_f8971314a3ef775f9c58_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_fa4617e53e638c775af8": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_fa4617e53e638c775af8",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_fa4617e53e638c775af8_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_fa4617e53e638c775af8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
}

MODULES: dict[str, dict[str, str]] = {
    "cake_batch_deepgemm_fp8_2382d782c0bafc3a9878_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_2382d782c0bafc3a9878",
    },
    "cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91",
    },
    "cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91",
    },
    "cake_batch_deepgemm_fp8_29a2b409934055f8cdd0_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
    },
    "cake_batch_deepgemm_fp8_29a2b409934055f8cdd0_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
    },
    "cake_batch_deepgemm_fp8_29a8930160c38b72cff6_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_29a8930160c38b72cff6",
    },
    "cake_batch_deepgemm_fp8_29a8930160c38b72cff6_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_29a8930160c38b72cff6",
    },
    "cake_batch_deepgemm_fp8_2cd08ce62ec175db050b_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_2cd08ce62ec175db050b",
    },
    "cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b",
    },
    "cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b",
    },
    "cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9",
    },
    "cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9",
    },
    "cake_batch_deepgemm_fp8_36afbebf9dfd01967336_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_36afbebf9dfd01967336",
    },
    "cake_batch_deepgemm_fp8_389e6c89e61b7d49a076_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
    },
    "cake_batch_deepgemm_fp8_389e6c89e61b7d49a076_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
    },
    "cake_batch_deepgemm_fp8_3dfe31e7f71ee9976e3e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_3dfe31e7f71ee9976e3e",
    },
    "cake_batch_deepgemm_fp8_508491c54eb3f469b37e_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_508491c54eb3f469b37e",
    },
    "cake_batch_deepgemm_fp8_508491c54eb3f469b37e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_508491c54eb3f469b37e",
    },
    "cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3",
    },
    "cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3",
    },
    "cake_batch_deepgemm_fp8_57c401152579ac3ac327_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_57c401152579ac3ac327",
    },
    "cake_batch_deepgemm_fp8_57c401152579ac3ac327_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_57c401152579ac3ac327",
    },
    "cake_batch_deepgemm_fp8_638db43d71890e22c8fe_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_638db43d71890e22c8fe",
    },
    "cake_batch_deepgemm_fp8_638db43d71890e22c8fe_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_638db43d71890e22c8fe",
    },
    "cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
    },
    "cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
    },
    "cake_batch_deepgemm_fp8_733a94c236b2ee6a8aee_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_733a94c236b2ee6a8aee",
    },
    "cake_batch_deepgemm_fp8_74557d5c854d22263fab_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_74557d5c854d22263fab",
    },
    "cake_batch_deepgemm_fp8_8730531b6d09e9544c01_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_8730531b6d09e9544c01",
    },
    "cake_batch_deepgemm_fp8_8730531b6d09e9544c01_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_8730531b6d09e9544c01",
    },
    "cake_batch_deepgemm_fp8_87a3c50a96af69fbae87_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_87a3c50a96af69fbae87",
    },
    "cake_batch_deepgemm_fp8_b974999e103cd7640cc7_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_b974999e103cd7640cc7",
    },
    "cake_batch_deepgemm_fp8_ba92e96364e30012640e_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_ba92e96364e30012640e",
    },
    "cake_batch_deepgemm_fp8_ba92e96364e30012640e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_ba92e96364e30012640e",
    },
    "cake_batch_deepgemm_fp8_bf96f77ec8235b35deac_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_bf96f77ec8235b35deac",
    },
    "cake_batch_deepgemm_fp8_bf96f77ec8235b35deac_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_bf96f77ec8235b35deac",
    },
    "cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8",
    },
    "cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8",
    },
    "cake_batch_deepgemm_fp8_c41147f6dbdf4c321445_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_c41147f6dbdf4c321445",
    },
    "cake_batch_deepgemm_fp8_c41147f6dbdf4c321445_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_c41147f6dbdf4c321445",
    },
    "cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a",
    },
    "cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a",
    },
    "cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
    },
    "cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
    },
    "cake_batch_deepgemm_fp8_ca298547e7b496acb4d3_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_ca298547e7b496acb4d3",
    },
    "cake_batch_deepgemm_fp8_ca298547e7b496acb4d3_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_ca298547e7b496acb4d3",
    },
    "cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13",
    },
    "cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13",
    },
    "cake_batch_deepgemm_fp8_d4262883ad8c2c6cef94_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_d4262883ad8c2c6cef94",
    },
    "cake_batch_deepgemm_fp8_d662bd14d9ad55d45263_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_d662bd14d9ad55d45263",
    },
    "cake_batch_deepgemm_fp8_d662bd14d9ad55d45263_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_d662bd14d9ad55d45263",
    },
    "cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5",
    },
    "cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5",
    },
    "cake_batch_deepgemm_fp8_e460881a1cdcaccf5860_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_e460881a1cdcaccf5860",
    },
    "cake_batch_deepgemm_fp8_e460881a1cdcaccf5860_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_e460881a1cdcaccf5860",
    },
    "cake_batch_deepgemm_fp8_eb727f8d280f83b52730_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_eb727f8d280f83b52730",
    },
    "cake_batch_deepgemm_fp8_eb727f8d280f83b52730_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_eb727f8d280f83b52730",
    },
    "cake_batch_deepgemm_fp8_ec43ea789e823a74035e_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_ec43ea789e823a74035e",
    },
    "cake_batch_deepgemm_fp8_ec43ea789e823a74035e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_ec43ea789e823a74035e",
    },
    "cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523",
    },
    "cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523",
    },
    "cake_batch_deepgemm_fp8_f376efa5168590657e71_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_f376efa5168590657e71",
    },
    "cake_batch_deepgemm_fp8_f376efa5168590657e71_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_f376efa5168590657e71",
    },
    "cake_batch_deepgemm_fp8_f8971314a3ef775f9c58_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_f8971314a3ef775f9c58",
    },
    "cake_batch_deepgemm_fp8_f8971314a3ef775f9c58_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_f8971314a3ef775f9c58",
    },
    "cake_batch_deepgemm_fp8_fa4617e53e638c775af8_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_fa4617e53e638c775af8",
    },
    "cake_batch_deepgemm_fp8_fa4617e53e638c775af8_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_fa4617e53e638c775af8",
    },
}

SERVING_PROGRAMS: dict[str, str] = {
    "n4096_k7168_g32_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_2fce665a907c0d48ab4b",
    "n4096_k7168_g32_s6e1_evict_normal_l1": "cake_batch_deepgemm_fp8_bf96f77ec8235b35deac",
    "n4096_k7168_g64_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_638db43d71890e22c8fe",
    "n4096_k7168_g64_s6e1_evict_normal_l1": "cake_batch_deepgemm_fp8_f8971314a3ef775f9c58",
    "n7168_k2048_g32_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_c216805f6296a1f0b0e8",
    "n7168_k2048_g32_s5e3_evict_normal_l8": "cake_batch_deepgemm_fp8_e460881a1cdcaccf5860",
    "n7168_k2048_g64_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_2ffc5a909a9e79353fa9",
    "n7168_k2048_g64_s5e3_evict_normal_l8": "cake_batch_deepgemm_fp8_29a8930160c38b72cff6",
}

ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


@dataclass(frozen=True)
class StagePlan:
    """One kernel launch of a route: the program, its grid and its bound arguments."""

    program: str
    grid: tuple[int, int, int]
    args: tuple[Any, ...]


@dataclass(frozen=True)
class RoutePlan:
    """The selected route: ordered stages and the exact shapes of the workspaces they bind."""

    route: str
    parent: str
    stages: tuple[StagePlan, ...]
    workspaces: dict[str, tuple[int, ...]]


RouteChain = Callable[[int, int, int, int, int], Optional[RoutePlan]]


def _source_dirs() -> tuple[Path, Path]:
    checkout = Path(__file__).resolve().parents[3]
    source_root = checkout / "csrc"
    include_root = checkout / "include"
    if not (source_root / "tvm_ffi_utils.h").is_file():
        source_root = jit_env.FLASHINFER_CSRC_DIR
        include_root = jit_env.FLASHINFER_INCLUDE_DIR
    if not (source_root / "tvm_ffi_utils.h").is_file():
        raise FileNotFoundError("FlashInfer binding headers were not found")
    return source_root, include_root


@functools.cache
def device_arch(device_index: int) -> str | None:
    """Generated-program architecture of CUDA device ``device_index`` (``None`` when unsupported)."""
    import torch

    return SUPPORTED_COMPUTE_CAPABILITIES.get(
        torch.cuda.get_device_capability(device_index)
    )


@functools.cache
def device_sm_count(device_index: int) -> int:
    """Streaming-multiprocessor count of CUDA device ``device_index`` (queried once)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def route_chain(arch: str, sm_count: int) -> RouteChain | None:
    """The rendered route chain for ``(arch, sm_count)``, or ``None`` when none was exported."""
    return DISPATCH_TABLES.get(arch, {}).get(sm_count)


def generated_program_available(device) -> bool:
    """True when this checkout registers programs and a route chain for ``device``."""
    import torch

    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    arch = device_arch(index)
    return (
        arch is not None
        and route_chain(arch, device_sm_count(index)) is not None
        and any(r["arch"] == arch for r in MODULES.values())
    )


def module_name(arch: str, program: str) -> str:
    """Return the registered JIT module name for one ``(arch, program)`` pair."""
    if program not in PROGRAMS or arch not in PROGRAMS[program]["arches"]:
        raise NotImplementedError(
            f"no generated Cake batch DeepGEMM FP8 program {program!r} is registered for {arch}"
        )
    return f"{program}_{arch}"


@functools.cache
def gen_cake_batch_deepgemm_fp8_module(name: str) -> JitSpec:
    record = MODULES[name]
    program = PROGRAMS[record["program"]]
    source_root, include_root = _source_dirs()
    sources = [source_root / relative for relative in program["sources"]]
    return gen_jit_spec(
        name=name,
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *program["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            source_root,
            *sorted({p.parent for p in sources}),
            include_root,
        ],
        use_fast_math=False,
    )


@functools.cache
def load_cake_batch_deepgemm_fp8_module(name: str):
    return gen_cake_batch_deepgemm_fp8_module(name).build_and_load()


def _routes_sm_100a_148(
    B: int, M: int, N: int, K: int, expected_m: int
) -> Optional[RoutePlan]:
    """Ordered first-match route chain of the Cake dispatcher for sm_100a devices with 148 SMs."""
    if ((((B == 1) and (M == 128)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 102
    ):
        return RoutePlan(
            route="seed_masked_bn__b1_m128_n4096_k7168_em102",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 4) and (M == 256)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 141
    ):
        return RoutePlan(
            route="seed_masked_bn__b4_m256_n4096_k7168_em141",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 191) // 192)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 191) // 192),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 128)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 53
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m128_n4096_k7168_em53",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 256)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 175
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m256_n4096_k7168_em175",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 223) // 224)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 223) // 224),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 4) and (M == 128)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 21
    ):
        return RoutePlan(
            route="seed_masked_bn__b4_m128_n7168_k2048_em21",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 4) and (M == 256)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 115
    ):
        return RoutePlan(
            route="seed_masked_bn__b4_m256_n7168_k2048_em115",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_fa4617e53e638c775af8",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 143) // 144)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 143) // 144),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 128)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 44
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m128_n7168_k2048_em44",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 95) // 96)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 95) // 96),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 256)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 117
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m256_n7168_k2048_em117",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 223) // 224)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 223) // 224),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 64) and (M == 128)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 68
    ):
        return RoutePlan(
            route="seed_masked_bn__b64_m128_n7168_k2048_em68",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_36afbebf9dfd01967336",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 143) // 144)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 143) // 144),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 128) and (M == 128)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 59
    ):
        return RoutePlan(
            route="seed_masked_bn__b128_m128_n7168_k2048_em59",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_36afbebf9dfd01967336",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 143) // 144)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 143) // 144),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 1) and (M == 8192)) and (N == 128)) and (K == 512)) and (
        expected_m == 5116
    ):
        return RoutePlan(
            route="seed_masked_bn__b1_m8192_n128_k512_em5116",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_2cd08ce62ec175db050b",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 63) // 64)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 63) // 64),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 4) and (M == 1024)) and (N == 128)) and (K == 512)) and (
        expected_m == 487
    ):
        return RoutePlan(
            route="seed_masked_bn__b4_m1024_n128_k512_em487",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_2382d782c0bafc3a9878",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 1024)) and (N == 128)) and (K == 512)) and (
        expected_m == 380
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m1024_n128_k512_em380",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_2382d782c0bafc3a9878",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and ((B * ((expected_m + 127) // 128)) <= 1):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn32",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_508491c54eb3f469b37e",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        ((B * ((expected_m + 127) // 128)) >= 2)
        and ((B * ((expected_m + 127) // 128)) <= 2)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn64",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 63) // 64)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 63) // 64),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        ((B * ((expected_m + 127) // 128)) >= 3)
        and ((B * ((expected_m + 127) // 128)) <= 3)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn96",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 95) // 96)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 95) // 96),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        ((B * ((expected_m + 127) // 128)) >= 4)
        and ((B * ((expected_m + 127) // 128)) <= 4)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn112",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        ((B * ((expected_m + 127) // 128)) >= 9)
        and ((B * ((expected_m + 127) // 128)) <= 9)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn128",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ca298547e7b496acb4d3",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 127) // 128)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 127) // 128),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        (
            ((B * ((expected_m + 127) // 128)) >= 5)
            and ((B * ((expected_m + 127) // 128)) <= 5)
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 10)
            and ((B * ((expected_m + 127) // 128)) <= 10)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn144",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_fa4617e53e638c775af8",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 143) // 144)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 143) // 144),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ((B * ((expected_m + 127) // 128)) >= 6)
                                        and ((B * ((expected_m + 127) // 128)) <= 6)
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 11)
                                        and ((B * ((expected_m + 127) // 128)) <= 11)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 12)
                                    and ((B * ((expected_m + 127) // 128)) <= 12)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 13)
                                and ((B * ((expected_m + 127) // 128)) <= 13)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 17)
                            and ((B * ((expected_m + 127) // 128)) <= 17)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 18)
                        and ((B * ((expected_m + 127) // 128)) <= 18)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 19)
                    and ((B * ((expected_m + 127) // 128)) <= 20)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 25)
                and ((B * ((expected_m + 127) // 128)) <= 26)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 33)
            and ((B * ((expected_m + 127) // 128)) <= 33)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn192",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 191) // 192)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 191) // 192),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ((B * ((expected_m + 127) // 128)) >= 7)
                                        and ((B * ((expected_m + 127) // 128)) <= 7)
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 14)
                                        and ((B * ((expected_m + 127) // 128)) <= 14)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 21)
                                    and ((B * ((expected_m + 127) // 128)) <= 22)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 27)
                                and ((B * ((expected_m + 127) // 128)) <= 29)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 34)
                            and ((B * ((expected_m + 127) // 128)) <= 37)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 42)
                        and ((B * ((expected_m + 127) // 128)) <= 44)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 50)
                    and ((B * ((expected_m + 127) // 128)) <= 51)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 58)
                and ((B * ((expected_m + 127) // 128)) <= 59)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 66)
            and ((B * ((expected_m + 127) // 128)) <= 66)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn208",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c41147f6dbdf4c321445",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 207) // 208)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 207) // 208),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            >= 15
                                                                        )
                                                                        and (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            <= 15
                                                                        )
                                                                    )
                                                                    or (
                                                                        (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            >= 23
                                                                        )
                                                                        and (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            <= 23
                                                                        )
                                                                    )
                                                                )
                                                                or (
                                                                    (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        >= 30
                                                                    )
                                                                    and (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        <= 31
                                                                    )
                                                                )
                                                            )
                                                            or (
                                                                (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    >= 38
                                                                )
                                                                and (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    <= 38
                                                                )
                                                            )
                                                        )
                                                        or (
                                                            (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                >= 45
                                                            )
                                                            and (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                <= 46
                                                            )
                                                        )
                                                    )
                                                    or (
                                                        (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            >= 52
                                                        )
                                                        and (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            <= 54
                                                        )
                                                    )
                                                )
                                                or (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 60
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 62
                                                    )
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 67
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 70
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 75)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 77
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 83)
                                        and ((B * ((expected_m + 127) // 128)) <= 85)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 91)
                                    and ((B * ((expected_m + 127) // 128)) <= 93)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 99)
                                and ((B * ((expected_m + 127) // 128)) <= 101)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 107)
                            and ((B * ((expected_m + 127) // 128)) <= 109)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 116)
                        and ((B * ((expected_m + 127) // 128)) <= 116)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 124)
                    and ((B * ((expected_m + 127) // 128)) <= 124)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 132)
                and ((B * ((expected_m + 127) // 128)) <= 132)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 140)
            and ((B * ((expected_m + 127) // 128)) <= 140)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn224",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 223) // 224)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 223) // 224),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 8192))))
            and (not ((B == 1) and (M == 16384)))
        )
        and (not ((B == 4) and (M == 1024)))
    ) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                >= 8
                                                                            )
                                                                            and (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                <= 8
                                                                            )
                                                                        )
                                                                        or (
                                                                            (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                >= 16
                                                                            )
                                                                            and (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                <= 16
                                                                            )
                                                                        )
                                                                    )
                                                                    or (
                                                                        (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            >= 24
                                                                        )
                                                                        and (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            <= 24
                                                                        )
                                                                    )
                                                                )
                                                                or (
                                                                    (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        >= 32
                                                                    )
                                                                    and (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        <= 32
                                                                    )
                                                                )
                                                            )
                                                            or (
                                                                (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    >= 39
                                                                )
                                                                and (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    <= 41
                                                                )
                                                            )
                                                        )
                                                        or (
                                                            (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                >= 47
                                                            )
                                                            and (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                <= 49
                                                            )
                                                        )
                                                    )
                                                    or (
                                                        (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            >= 55
                                                        )
                                                        and (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            <= 57
                                                        )
                                                    )
                                                )
                                                or (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 63
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 65
                                                    )
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 71
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 74
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 78)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 82
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 86)
                                        and ((B * ((expected_m + 127) // 128)) <= 90)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 94)
                                    and ((B * ((expected_m + 127) // 128)) <= 98)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 102)
                                and ((B * ((expected_m + 127) // 128)) <= 106)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 110)
                            and ((B * ((expected_m + 127) // 128)) <= 115)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 117)
                        and ((B * ((expected_m + 127) // 128)) <= 123)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 125)
                    and ((B * ((expected_m + 127) // 128)) <= 131)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 133)
                and ((B * ((expected_m + 127) // 128)) <= 139)
            )
        )
        or ((B * ((expected_m + 127) // 128)) >= 141)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn240",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 239) // 240)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 239) // 240),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 16384)))) and (
        (B * ((expected_m + 127) // 128)) <= 1
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn64",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 63) // 64)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 63) // 64),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 16384)))) and (
        ((B * ((expected_m + 127) // 128)) >= 2)
        and ((B * ((expected_m + 127) // 128)) <= 2)
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn112",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 16384)))) and (
        ((B * ((expected_m + 127) // 128)) >= 5)
        and ((B * ((expected_m + 127) // 128)) <= 5)
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn128",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ca298547e7b496acb4d3",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 127) // 128)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 127) // 128),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            ((B * ((expected_m + 127) // 128)) >= 3)
                            and ((B * ((expected_m + 127) // 128)) <= 3)
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 6)
                            and ((B * ((expected_m + 127) // 128)) <= 6)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 7)
                        and ((B * ((expected_m + 127) // 128)) <= 7)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 10)
                    and ((B * ((expected_m + 127) // 128)) <= 10)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 11)
                and ((B * ((expected_m + 127) // 128)) <= 11)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 15)
            and ((B * ((expected_m + 127) // 128)) <= 15)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn192",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 191) // 192)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 191) // 192),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            ((B * ((expected_m + 127) // 128)) >= 4)
                            and ((B * ((expected_m + 127) // 128)) <= 4)
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 8)
                            and ((B * ((expected_m + 127) // 128)) <= 8)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 12)
                        and ((B * ((expected_m + 127) // 128)) <= 12)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 16)
                    and ((B * ((expected_m + 127) // 128)) <= 16)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 20)
                and ((B * ((expected_m + 127) // 128)) <= 21)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 25)
            and ((B * ((expected_m + 127) // 128)) <= 25)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn208",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c41147f6dbdf4c321445",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 207) // 208)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 207) // 208),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 9
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 9
                                                    )
                                                )
                                                or (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 13
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 13
                                                    )
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 17
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 18
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 22)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 23
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 26)
                                        and ((B * ((expected_m + 127) // 128)) <= 27)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 30)
                                    and ((B * ((expected_m + 127) // 128)) <= 32)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 35)
                                and ((B * ((expected_m + 127) // 128)) <= 37)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 40)
                            and ((B * ((expected_m + 127) // 128)) <= 41)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 45)
                        and ((B * ((expected_m + 127) // 128)) <= 46)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 50)
                    and ((B * ((expected_m + 127) // 128)) <= 50)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 55)
                and ((B * ((expected_m + 127) // 128)) <= 55)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 60)
            and ((B * ((expected_m + 127) // 128)) <= 60)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn224",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 223) // 224)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 223) // 224),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 14
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 14
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 19
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 19
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 24)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 24
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 28)
                                        and ((B * ((expected_m + 127) // 128)) <= 29)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 33)
                                    and ((B * ((expected_m + 127) // 128)) <= 34)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 38)
                                and ((B * ((expected_m + 127) // 128)) <= 39)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 42)
                            and ((B * ((expected_m + 127) // 128)) <= 44)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 47)
                        and ((B * ((expected_m + 127) // 128)) <= 49)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 51)
                    and ((B * ((expected_m + 127) // 128)) <= 54)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 56)
                and ((B * ((expected_m + 127) // 128)) <= 59)
            )
        )
        or ((B * ((expected_m + 127) // 128)) >= 61)
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn240",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 239) // 240)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 239) // 240),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        ((B * ((expected_m + 127) // 128)) <= 18)
        or (
            ((B * ((expected_m + 127) // 128)) >= 19)
            and ((B * ((expected_m + 127) // 128)) <= 37)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn32",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        ((B * ((expected_m + 127) // 128)) >= 38)
        and ((B * ((expected_m + 127) // 128)) <= 49)
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn48",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_8730531b6d09e9544c01",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 47) // 48)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 47) // 48),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        ((B * ((expected_m + 127) // 128)) >= 50)
        and ((B * ((expected_m + 127) // 128)) <= 74)
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn64",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 63) // 64)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 63) // 64),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        (B * ((expected_m + 127) // 128)) >= 75
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn128",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_d662bd14d9ad55d45263",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 127) // 128)
                            ),
                            148,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 127) // 128),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 128) and (K == 512):
        return RoutePlan(
            route="seed_n128_k512",
            parent="seed_n128_k512",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ba92e96364e30012640e",
                    grid=((min((B * ((M + 255) // 256)), 74) * 2), 1, 1),
                    args=(
                        "a",
                        "b",
                        "a_scale",
                        "b_scale",
                        "masked_m",
                        "out",
                        B,
                        M,
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 512) and (K == 128):
        return RoutePlan(
            route="seed_n512_k128",
            parent="seed_n512_k128",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523",
                    grid=(((B * (((expected_m + 127) // 128) + 1)) * (N // 128)), 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "a_scale",
                        "b_scale",
                        "masked_m",
                        B,
                        M,
                        (N // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 32) and (M == 4096)) and (N == 6144)) and (K == 7168)) and (
        expected_m == 24
    ):
        return RoutePlan(
            route="seed_swap_m32_fp32_sm100_b32",
            parent="seed_swap_m32_fp32_sm100_b32",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_87a3c50a96af69fbae87",
                    grid=(
                        (B * ((((M + 48) - 1) // 48) + (((N + 48) - 1) // 48))),
                        1,
                        1,
                    ),
                    args=(
                        "a_scale_bits",
                        "b_scale_bits",
                        "sfa_m32_packed",
                        "sfb_m32_packed",
                        B,
                        M,
                        N,
                        K,
                    ),
                ),
                StagePlan(
                    program="cake_batch_deepgemm_fp8_733a94c236b2ee6a8aee",
                    grid=(132, 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "sfa_m32_packed",
                        "sfb_m32_packed",
                        "masked_m",
                        B,
                        M,
                        N,
                        K,
                    ),
                ),
            ),
            workspaces={
                "sfa_m32_packed": (
                    (B * (K // 512)),
                    M,
                ),
                "sfb_m32_packed": (
                    (B * (K // 512)),
                    N,
                ),
            },
        )
    if ((((B == 1) and (M == 16384)) and (N == 4096)) and (K == 7168)) or (
        (((B == 1) and (M == 8192)) and (N == 4096)) and (K == 7168)
    ):
        return RoutePlan(
            route="seed_swap_m224_packed_l2_16",
            parent="seed_swap_m224_packed_l2_16",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_f376efa5168590657e71",
                    grid=(
                        (
                            B
                            * ((((M + 48) - 1) // 48) + ((((N // 128) + 48) - 1) // 48))
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a_scale_bits",
                        "b_scale_bits",
                        "sfa_packed",
                        "sfb_packed",
                        B,
                        M,
                        N,
                        K,
                    ),
                ),
                StagePlan(
                    program="cake_batch_deepgemm_fp8_b974999e103cd7640cc7",
                    grid=(148, 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "sfa_packed",
                        "sfb_packed",
                        "masked_m",
                        B,
                        M,
                        M,
                        N,
                        K,
                    ),
                ),
            ),
            workspaces={
                "sfa_packed": (
                    B,
                    (K // 512),
                    M,
                ),
                "sfb_packed": (
                    B,
                    (K // 512),
                    (N // 128),
                ),
            },
        )
    if ((((B == 6) and (M == 4096)) and (N == 4096)) and (K == 4096)) and (
        expected_m == 1228
    ):
        return RoutePlan(
            route="seed_swap_m224_packed_sm100_b6_n4096",
            parent="seed_swap_m224_packed_sm100_b6_n4096",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_f376efa5168590657e71",
                    grid=(
                        (
                            B
                            * ((((M + 48) - 1) // 48) + ((((N // 128) + 48) - 1) // 48))
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a_scale_bits",
                        "b_scale_bits",
                        "sfa_packed",
                        "sfb_packed",
                        B,
                        M,
                        N,
                        K,
                    ),
                ),
                StagePlan(
                    program="cake_batch_deepgemm_fp8_74557d5c854d22263fab",
                    grid=(148, 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "sfa_packed",
                        "sfb_packed",
                        "masked_m",
                        B,
                        M,
                        M,
                        N,
                        K,
                    ),
                ),
            ),
            workspaces={
                "sfa_packed": (
                    B,
                    (K // 512),
                    M,
                ),
                "sfb_packed": (
                    B,
                    (K // 512),
                    (N // 128),
                ),
            },
        )
    if ((((B == 6) and (M == 4096)) and (N == 6144)) and (K == 7168)) and (
        expected_m == 1228
    ):
        return RoutePlan(
            route="seed_swap_m224_packed_sm100_b6_n6144",
            parent="seed_swap_m224_packed_sm100_b6_n6144",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_f376efa5168590657e71",
                    grid=(
                        (
                            B
                            * ((((M + 48) - 1) // 48) + ((((N // 128) + 48) - 1) // 48))
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a_scale_bits",
                        "b_scale_bits",
                        "sfa_packed",
                        "sfb_packed",
                        B,
                        M,
                        N,
                        K,
                    ),
                ),
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ec43ea789e823a74035e",
                    grid=(148, 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "sfa_packed",
                        "sfb_packed",
                        "masked_m",
                        B,
                        M,
                        M,
                        N,
                        K,
                    ),
                ),
            ),
            workspaces={
                "sfa_packed": (
                    B,
                    (K // 512),
                    M,
                ),
                "sfb_packed": (
                    B,
                    (K // 512),
                    (N // 128),
                ),
            },
        )
    if (
        ((N == 4096) and (K == 7168))
        and (not (((B == 1) and (M == 16384)) or ((B == 8) and (M == 1024))))
    ) and (
        not (
            ((((B == 1) and (M == 8192)) and (N == 4096)) and (K == 7168))
            or ((((B == 4) and (M == 1024)) and (N == 4096)) and (K == 7168))
        )
    ):
        raise NotImplementedError(
            "route seed_n4096_k7168 of the Cake dispatcher has no exported program"
        )
    if (((N == 7168) and (K == 2048)) and (M <= 256)) or (
        (expected_m == 24)
        and (
            (
                (
                    (((N == 7168) and (K == 2048)) or ((N == 6144) and (K == 7168)))
                    or ((N == 7168) and (K == 3072))
                )
                or ((N == 4096) and (K == 4096))
            )
            or ((N == 4096) and (K == 2048))
        )
    ):
        return RoutePlan(
            route="seed_large_nk_cta1",
            parent="seed_large_nk_cta1",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_eb727f8d280f83b52730",
                    grid=(
                        min(
                            ((B * (((expected_m + 127) // 128) + 1)) * (N // 128)), 148
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 127) // 128) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (
                (
                    (
                        (
                            (
                                ((expected_m == 230) or (expected_m == 1228))
                                and (
                                    (
                                        (
                                            ((N == 6144) and (K == 7168))
                                            or ((N == 7168) and (K == 3072))
                                        )
                                        or ((N == 4096) and (K == 4096))
                                    )
                                    or ((N == 4096) and (K == 2048))
                                )
                            )
                            or (
                                (((B == 1) and (M == 8192)) and (N == 7168))
                                and (K == 2048)
                            )
                        )
                        or (
                            (((B == 1) and (M == 16384)) and (N == 4096))
                            and (K == 7168)
                        )
                    )
                    or ((((B == 8) and (M == 1024)) and (N == 4096)) and (K == 7168))
                )
                or ((((B == 1) and (M == 8192)) and (N == 4096)) and (K == 7168))
            )
            or ((((B == 1) and (M == 16384)) and (N == 7168)) and (K == 2048))
        )
        or ((((B == 4) and (M == 1024)) and (N == 4096)) and (K == 7168))
    ) or ((((B == 8) and (M == 1024)) and (N == 7168)) and (K == 2048)):
        return RoutePlan(
            route="seed_swap_m224_packed",
            parent="seed_swap_m224_packed",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_f376efa5168590657e71",
                    grid=(
                        (
                            B
                            * ((((M + 48) - 1) // 48) + ((((N // 128) + 48) - 1) // 48))
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a_scale_bits",
                        "b_scale_bits",
                        "sfa_packed",
                        "sfb_packed",
                        B,
                        M,
                        N,
                        K,
                    ),
                ),
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ec43ea789e823a74035e",
                    grid=(148, 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "sfa_packed",
                        "sfb_packed",
                        "masked_m",
                        B,
                        M,
                        M,
                        N,
                        K,
                    ),
                ),
            ),
            workspaces={
                "sfa_packed": (
                    B,
                    (K // 512),
                    M,
                ),
                "sfb_packed": (
                    B,
                    (K // 512),
                    (N // 128),
                ),
            },
        )
    if (N == 7168) and (K == 2048):
        return RoutePlan(
            route="seed_n7168_k2048",
            parent="seed_n7168_k2048",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                74,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 6144) and (K == 7168):
        return RoutePlan(
            route="seed_large_nk_n6144_k7168",
            parent="seed_large_nk_n6144_k7168",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                74,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 7168) and (K == 3072):
        return RoutePlan(
            route="seed_large_nk_n7168_k3072",
            parent="seed_large_nk_n7168_k3072",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                74,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 4096) and (K == 4096):
        return RoutePlan(
            route="seed_large_nk_n4096_k4096",
            parent="seed_large_nk_n4096_k4096",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                74,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 4096) and (K == 2048):
        return RoutePlan(
            route="seed_large_nk_n4096_k2048",
            parent="seed_large_nk_n4096_k2048",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                74,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    return None


def _routes_sm_103a_152(
    B: int, M: int, N: int, K: int, expected_m: int
) -> Optional[RoutePlan]:
    """Ordered first-match route chain of the Cake dispatcher for sm_103a devices with 152 SMs."""
    if ((((B == 1) and (M == 128)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 26
    ):
        return RoutePlan(
            route="seed_masked_bn__b1_m128_n4096_k7168_em26",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 4) and (M == 1024)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 303
    ):
        return RoutePlan(
            route="seed_masked_bn__b4_m1024_n4096_k7168_em303",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 256)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 101
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m256_n4096_k7168_em101",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 1024)) and (N == 4096)) and (K == 7168)) and (
        expected_m == 595
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m1024_n4096_k7168_em595",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 239) // 240)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 239) // 240),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 8) and (M == 256)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 44
    ):
        return RoutePlan(
            route="seed_masked_bn__b8_m256_n7168_k2048_em44",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 64) and (M == 128)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 68
    ):
        return RoutePlan(
            route="seed_masked_bn__b64_m128_n7168_k2048_em68",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 191) // 192)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 191) // 192),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((((B == 128) and (M == 128)) and (N == 7168)) and (K == 2048)) and (
        expected_m == 60
    ):
        return RoutePlan(
            route="seed_masked_bn__b128_m128_n7168_k2048_em60",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_d4262883ad8c2c6cef94",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 239) // 240)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 239) // 240),
                        (K // 256),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        (B * ((expected_m + 127) // 128)) <= 1
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn32",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_508491c54eb3f469b37e",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        ((B * ((expected_m + 127) // 128)) >= 2)
        and ((B * ((expected_m + 127) // 128)) <= 2)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn64",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c42f3895c1e079f0e36a",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 63) // 64)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 63) // 64),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        ((B * ((expected_m + 127) // 128)) >= 3)
        and ((B * ((expected_m + 127) // 128)) <= 3)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn96",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 95) // 96)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 95) // 96),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        ((B * ((expected_m + 127) // 128)) >= 4)
        and ((B * ((expected_m + 127) // 128)) <= 4)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn112",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_389e6c89e61b7d49a076",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 111) // 112)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 111) // 112),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        ((B * ((expected_m + 127) // 128)) >= 9)
        and ((B * ((expected_m + 127) // 128)) <= 9)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn128",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ca298547e7b496acb4d3",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 127) // 128)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 127) // 128),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        (
            ((B * ((expected_m + 127) // 128)) >= 5)
            and ((B * ((expected_m + 127) // 128)) <= 5)
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 10)
            and ((B * ((expected_m + 127) // 128)) <= 10)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn144",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_fa4617e53e638c775af8",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 143) // 144)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 143) // 144),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ((B * ((expected_m + 127) // 128)) >= 6)
                                        and ((B * ((expected_m + 127) // 128)) <= 6)
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 11)
                                        and ((B * ((expected_m + 127) // 128)) <= 11)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 12)
                                    and ((B * ((expected_m + 127) // 128)) <= 12)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 13)
                                and ((B * ((expected_m + 127) // 128)) <= 13)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 17)
                            and ((B * ((expected_m + 127) // 128)) <= 17)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 18)
                        and ((B * ((expected_m + 127) // 128)) <= 19)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 20)
                    and ((B * ((expected_m + 127) // 128)) <= 20)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 26)
                and ((B * ((expected_m + 127) // 128)) <= 27)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 34)
            and ((B * ((expected_m + 127) // 128)) <= 34)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn192",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 191) // 192)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 191) // 192),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        ((B * ((expected_m + 127) // 128)) >= 7)
                                        and ((B * ((expected_m + 127) // 128)) <= 7)
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 14)
                                        and ((B * ((expected_m + 127) // 128)) <= 15)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 21)
                                    and ((B * ((expected_m + 127) // 128)) <= 22)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 28)
                                and ((B * ((expected_m + 127) // 128)) <= 30)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 35)
                            and ((B * ((expected_m + 127) // 128)) <= 38)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 43)
                        and ((B * ((expected_m + 127) // 128)) <= 45)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 51)
                    and ((B * ((expected_m + 127) // 128)) <= 53)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 60)
                and ((B * ((expected_m + 127) // 128)) <= 60)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 68)
            and ((B * ((expected_m + 127) // 128)) <= 68)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn208",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c41147f6dbdf4c321445",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 207) // 208)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 207) // 208),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                >= 8
                                                                            )
                                                                            and (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                <= 8
                                                                            )
                                                                        )
                                                                        or (
                                                                            (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                >= 16
                                                                            )
                                                                            and (
                                                                                (
                                                                                    B
                                                                                    * (
                                                                                        (
                                                                                            expected_m
                                                                                            + 127
                                                                                        )
                                                                                        // 128
                                                                                    )
                                                                                )
                                                                                <= 16
                                                                            )
                                                                        )
                                                                    )
                                                                    or (
                                                                        (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            >= 23
                                                                        )
                                                                        and (
                                                                            (
                                                                                B
                                                                                * (
                                                                                    (
                                                                                        expected_m
                                                                                        + 127
                                                                                    )
                                                                                    // 128
                                                                                )
                                                                            )
                                                                            <= 24
                                                                        )
                                                                    )
                                                                )
                                                                or (
                                                                    (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        >= 31
                                                                    )
                                                                    and (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        <= 32
                                                                    )
                                                                )
                                                            )
                                                            or (
                                                                (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    >= 39
                                                                )
                                                                and (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    <= 40
                                                                )
                                                            )
                                                        )
                                                        or (
                                                            (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                >= 46
                                                            )
                                                            and (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                <= 48
                                                            )
                                                        )
                                                    )
                                                    or (
                                                        (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            >= 54
                                                        )
                                                        and (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            <= 56
                                                        )
                                                    )
                                                )
                                                or (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 61
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 64
                                                    )
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 69
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 72
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 77)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 80
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 85)
                                        and ((B * ((expected_m + 127) // 128)) <= 88)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 93)
                                    and ((B * ((expected_m + 127) // 128)) <= 96)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 102)
                                and ((B * ((expected_m + 127) // 128)) <= 104)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 110)
                            and ((B * ((expected_m + 127) // 128)) <= 112)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 119)
                        and ((B * ((expected_m + 127) // 128)) <= 120)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 127)
                    and ((B * ((expected_m + 127) // 128)) <= 128)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 136)
                and ((B * ((expected_m + 127) // 128)) <= 136)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 144)
            and ((B * ((expected_m + 127) // 128)) <= 144)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn224",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 223) // 224)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 223) // 224),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 4096) and (K == 7168)) and (not ((B == 1) and (M == 16384)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        >= 25
                                                                    )
                                                                    and (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        <= 25
                                                                    )
                                                                )
                                                                or (
                                                                    (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        >= 33
                                                                    )
                                                                    and (
                                                                        (
                                                                            B
                                                                            * (
                                                                                (
                                                                                    expected_m
                                                                                    + 127
                                                                                )
                                                                                // 128
                                                                            )
                                                                        )
                                                                        <= 33
                                                                    )
                                                                )
                                                            )
                                                            or (
                                                                (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    >= 41
                                                                )
                                                                and (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    <= 42
                                                                )
                                                            )
                                                        )
                                                        or (
                                                            (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                >= 49
                                                            )
                                                            and (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                <= 50
                                                            )
                                                        )
                                                    )
                                                    or (
                                                        (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            >= 57
                                                        )
                                                        and (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            <= 59
                                                        )
                                                    )
                                                )
                                                or (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 65
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 67
                                                    )
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 73
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 76
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 81)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 84
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 89)
                                        and ((B * ((expected_m + 127) // 128)) <= 92)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 97)
                                    and ((B * ((expected_m + 127) // 128)) <= 101)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 105)
                                and ((B * ((expected_m + 127) // 128)) <= 109)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 113)
                            and ((B * ((expected_m + 127) // 128)) <= 118)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 121)
                        and ((B * ((expected_m + 127) // 128)) <= 126)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 129)
                    and ((B * ((expected_m + 127) // 128)) <= 135)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 137)
                and ((B * ((expected_m + 127) // 128)) <= 143)
            )
        )
        or ((B * ((expected_m + 127) // 128)) >= 145)
    ):
        return RoutePlan(
            route="seed_masked_bn__n4096_k7168_bn240",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 239) // 240)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 239) // 240),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 8192)))) and (
        (B * ((expected_m + 127) // 128)) <= 1
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn48",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_3dfe31e7f71ee9976e3e",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 47) // 48)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 47) // 48),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 8192)))) and (
        ((B * ((expected_m + 127) // 128)) >= 2)
        and ((B * ((expected_m + 127) // 128)) <= 2)
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn96",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_2458ef4d346ca1b65c91",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 95) // 96)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 95) // 96),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 8192)))) and (
        (
            ((B * ((expected_m + 127) // 128)) >= 3)
            and ((B * ((expected_m + 127) // 128)) <= 3)
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 6)
            and ((B * ((expected_m + 127) // 128)) <= 6)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn144",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_fa4617e53e638c775af8",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 143) // 144)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 143) // 144),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 8192)))) and (
        (
            (
                (
                    (
                        (
                            ((B * ((expected_m + 127) // 128)) >= 4)
                            and ((B * ((expected_m + 127) // 128)) <= 4)
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 7)
                            and ((B * ((expected_m + 127) // 128)) <= 7)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 8)
                        and ((B * ((expected_m + 127) // 128)) <= 8)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 11)
                    and ((B * ((expected_m + 127) // 128)) <= 11)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 12)
                and ((B * ((expected_m + 127) // 128)) <= 12)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 16)
            and ((B * ((expected_m + 127) // 128)) <= 16)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn192",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_29a2b409934055f8cdd0",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 191) // 192)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 191) // 192),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 8192)))) and (
        (
            (
                (
                    ((B * ((expected_m + 127) // 128)) >= 13)
                    and ((B * ((expected_m + 127) // 128)) <= 13)
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 17)
                    and ((B * ((expected_m + 127) // 128)) <= 17)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 21)
                and ((B * ((expected_m + 127) // 128)) <= 21)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 26)
            and ((B * ((expected_m + 127) // 128)) <= 26)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn208",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c41147f6dbdf4c321445",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 207) // 208)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 207) // 208),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 8192)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                >= 9
                                                            )
                                                            and (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                <= 9
                                                            )
                                                        )
                                                        or (
                                                            (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                >= 14
                                                            )
                                                            and (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                <= 14
                                                            )
                                                        )
                                                    )
                                                    or (
                                                        (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            >= 18
                                                        )
                                                        and (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            <= 19
                                                        )
                                                    )
                                                )
                                                or (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 22
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 23
                                                    )
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 27
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 28
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 31)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 33
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 36)
                                        and ((B * ((expected_m + 127) // 128)) <= 38)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 41)
                                    and ((B * ((expected_m + 127) // 128)) <= 42)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 46)
                                and ((B * ((expected_m + 127) // 128)) <= 47)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 51)
                            and ((B * ((expected_m + 127) // 128)) <= 52)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 56)
                        and ((B * ((expected_m + 127) // 128)) <= 57)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 61)
                    and ((B * ((expected_m + 127) // 128)) <= 61)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 66)
                and ((B * ((expected_m + 127) // 128)) <= 66)
            )
        )
        or (
            ((B * ((expected_m + 127) // 128)) >= 71)
            and ((B * ((expected_m + 127) // 128)) <= 71)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn224",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_c4f1500b3e772eb3e02f",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 223) // 224)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 223) // 224),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 7168) and (K == 2048)) and (not ((B == 1) and (M == 8192)))) and (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    >= 5
                                                                )
                                                                and (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    <= 5
                                                                )
                                                            )
                                                            or (
                                                                (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    >= 10
                                                                )
                                                                and (
                                                                    (
                                                                        B
                                                                        * (
                                                                            (
                                                                                expected_m
                                                                                + 127
                                                                            )
                                                                            // 128
                                                                        )
                                                                    )
                                                                    <= 10
                                                                )
                                                            )
                                                        )
                                                        or (
                                                            (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                >= 15
                                                            )
                                                            and (
                                                                (
                                                                    B
                                                                    * (
                                                                        (
                                                                            expected_m
                                                                            + 127
                                                                        )
                                                                        // 128
                                                                    )
                                                                )
                                                                <= 15
                                                            )
                                                        )
                                                    )
                                                    or (
                                                        (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            >= 20
                                                        )
                                                        and (
                                                            (
                                                                B
                                                                * (
                                                                    (expected_m + 127)
                                                                    // 128
                                                                )
                                                            )
                                                            <= 20
                                                        )
                                                    )
                                                )
                                                or (
                                                    (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        >= 24
                                                    )
                                                    and (
                                                        (
                                                            B
                                                            * (
                                                                (expected_m + 127)
                                                                // 128
                                                            )
                                                        )
                                                        <= 25
                                                    )
                                                )
                                            )
                                            or (
                                                (
                                                    (B * ((expected_m + 127) // 128))
                                                    >= 29
                                                )
                                                and (
                                                    (B * ((expected_m + 127) // 128))
                                                    <= 30
                                                )
                                            )
                                        )
                                        or (
                                            ((B * ((expected_m + 127) // 128)) >= 34)
                                            and (
                                                (B * ((expected_m + 127) // 128)) <= 35
                                            )
                                        )
                                    )
                                    or (
                                        ((B * ((expected_m + 127) // 128)) >= 39)
                                        and ((B * ((expected_m + 127) // 128)) <= 40)
                                    )
                                )
                                or (
                                    ((B * ((expected_m + 127) // 128)) >= 43)
                                    and ((B * ((expected_m + 127) // 128)) <= 45)
                                )
                            )
                            or (
                                ((B * ((expected_m + 127) // 128)) >= 48)
                                and ((B * ((expected_m + 127) // 128)) <= 50)
                            )
                        )
                        or (
                            ((B * ((expected_m + 127) // 128)) >= 53)
                            and ((B * ((expected_m + 127) // 128)) <= 55)
                        )
                    )
                    or (
                        ((B * ((expected_m + 127) // 128)) >= 58)
                        and ((B * ((expected_m + 127) // 128)) <= 60)
                    )
                )
                or (
                    ((B * ((expected_m + 127) // 128)) >= 62)
                    and ((B * ((expected_m + 127) // 128)) <= 65)
                )
            )
            or (
                ((B * ((expected_m + 127) // 128)) >= 67)
                and ((B * ((expected_m + 127) // 128)) <= 70)
            )
        )
        or ((B * ((expected_m + 127) // 128)) >= 72)
    ):
        return RoutePlan(
            route="seed_masked_bn__n7168_k2048_bn240",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_63a1a48b4d5f268a5163",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 239) // 240)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 239) // 240),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        ((B * ((expected_m + 127) // 128)) <= 19)
        or (
            ((B * ((expected_m + 127) // 128)) >= 20)
            and ((B * ((expected_m + 127) // 128)) <= 38)
        )
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn32",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_573a2b54ea777a0d43c3",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 31) // 32)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 31) // 32),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        ((B * ((expected_m + 127) // 128)) >= 39)
        and ((B * ((expected_m + 127) // 128)) <= 50)
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn48",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_8730531b6d09e9544c01",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 47) // 48)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 47) // 48),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        ((B * ((expected_m + 127) // 128)) >= 51)
        and ((B * ((expected_m + 127) // 128)) <= 76)
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn64",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_d9620b70a0bd6b92d5f5",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 63) // 64)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 63) // 64),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (((N == 128) and (K == 512)) and (B <= 8)) and (
        (B * ((expected_m + 127) // 128)) >= 77
    ):
        return RoutePlan(
            route="seed_masked_bn__n128_k512_bn128",
            parent="seed_masked_bn",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_d662bd14d9ad55d45263",
                    grid=(
                        min(
                            (
                                (B * (((expected_m + 127) // 128) + 1))
                                * ((N + 127) // 128)
                            ),
                            152,
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        N,
                        ((N + 127) // 128),
                        (K // 128),
                        (K // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 128) and (K == 512):
        return RoutePlan(
            route="seed_n128_k512",
            parent="seed_n128_k512",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ba92e96364e30012640e",
                    grid=((min((B * ((M + 255) // 256)), 76) * 2), 1, 1),
                    args=(
                        "a",
                        "b",
                        "a_scale",
                        "b_scale",
                        "masked_m",
                        "out",
                        B,
                        M,
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 512) and (K == 128):
        return RoutePlan(
            route="seed_n512_k128",
            parent="seed_n512_k128",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_f2d3d7cdd838fe7ef523",
                    grid=(((B * (((expected_m + 127) // 128) + 1)) * (N // 128)), 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "a_scale",
                        "b_scale",
                        "masked_m",
                        B,
                        M,
                        (N // 128),
                    ),
                ),
            ),
            workspaces={},
        )
    if ((N == 4096) and (K == 7168)) and (
        not (((B == 1) and (M == 16384)) or ((B == 8) and (M == 1024)))
    ):
        raise NotImplementedError(
            "route seed_n4096_k7168 of the Cake dispatcher has no exported program"
        )
    if (((N == 7168) and (K == 2048)) and (M <= 256)) or (
        (expected_m == 24)
        and (
            (
                (
                    (((N == 7168) and (K == 2048)) or ((N == 6144) and (K == 7168)))
                    or ((N == 7168) and (K == 3072))
                )
                or ((N == 4096) and (K == 4096))
            )
            or ((N == 4096) and (K == 2048))
        )
    ):
        return RoutePlan(
            route="seed_large_nk_cta1",
            parent="seed_large_nk_cta1",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_eb727f8d280f83b52730",
                    grid=(
                        min(
                            ((B * (((expected_m + 127) // 128) + 1)) * (N // 128)), 152
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 127) // 128) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (
        (
            (
                ((expected_m == 230) or (expected_m == 1228))
                and (
                    (
                        (((N == 6144) and (K == 7168)) or ((N == 7168) and (K == 3072)))
                        or ((N == 4096) and (K == 4096))
                    )
                    or ((N == 4096) and (K == 2048))
                )
            )
            or ((((B == 1) and (M == 8192)) and (N == 7168)) and (K == 2048))
        )
        or ((((B == 1) and (M == 16384)) and (N == 4096)) and (K == 7168))
    ) or ((((B == 8) and (M == 1024)) and (N == 4096)) and (K == 7168)):
        return RoutePlan(
            route="seed_swap_m224_packed",
            parent="seed_swap_m224_packed",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_f376efa5168590657e71",
                    grid=(
                        (
                            B
                            * ((((M + 48) - 1) // 48) + ((((N // 128) + 48) - 1) // 48))
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a_scale_bits",
                        "b_scale_bits",
                        "sfa_packed",
                        "sfb_packed",
                        B,
                        M,
                        N,
                        K,
                    ),
                ),
                StagePlan(
                    program="cake_batch_deepgemm_fp8_ec43ea789e823a74035e",
                    grid=(152, 1, 1),
                    args=(
                        "a",
                        "b",
                        "out",
                        "sfa_packed",
                        "sfb_packed",
                        "masked_m",
                        B,
                        M,
                        M,
                        N,
                        K,
                    ),
                ),
            ),
            workspaces={
                "sfa_packed": (
                    B,
                    (K // 512),
                    M,
                ),
                "sfb_packed": (
                    B,
                    (K // 512),
                    (N // 128),
                ),
            },
        )
    if (N == 7168) and (K == 2048):
        return RoutePlan(
            route="seed_n7168_k2048",
            parent="seed_n7168_k2048",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                76,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 6144) and (K == 7168):
        return RoutePlan(
            route="seed_large_nk_n6144_k7168",
            parent="seed_large_nk_n6144_k7168",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                76,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 7168) and (K == 3072):
        return RoutePlan(
            route="seed_large_nk_n7168_k3072",
            parent="seed_large_nk_n7168_k3072",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                76,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 4096) and (K == 4096):
        return RoutePlan(
            route="seed_large_nk_n4096_k4096",
            parent="seed_large_nk_n4096_k4096",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                76,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    if (N == 4096) and (K == 2048):
        return RoutePlan(
            route="seed_large_nk_n4096_k2048",
            parent="seed_large_nk_n4096_k2048",
            stages=(
                StagePlan(
                    program="cake_batch_deepgemm_fp8_57c401152579ac3ac327",
                    grid=(
                        (
                            min(
                                ((B * (((expected_m + 255) // 256) + 1)) * (N // 128)),
                                76,
                            )
                            * 2
                        ),
                        1,
                        1,
                    ),
                    args=(
                        "a",
                        "b",
                        "a_scale_bits",
                        "b_scale_bits",
                        "masked_m",
                        "out",
                        B,
                        M,
                        (N // 128),
                        (K // 256),
                        (K // 128),
                        (((expected_m + 255) // 256) + 1),
                    ),
                ),
            ),
            workspaces={},
        )
    return None


DISPATCH_TABLES: dict[str, dict[int, RouteChain]] = {
    "sm_100a": {148: _routes_sm_100a_148},
    "sm_103a": {152: _routes_sm_103a_152},
}

__all__ = [
    "ARG_PLANS",
    "DISPATCH_TABLES",
    "MODULES",
    "PROGRAMS",
    "RouteChain",
    "RoutePlan",
    "SERVING_PROGRAMS",
    "StagePlan",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "device_arch",
    "device_sm_count",
    "gen_cake_batch_deepgemm_fp8_module",
    "generated_program_available",
    "load_cake_batch_deepgemm_fp8_module",
    "module_name",
    "route_chain",
]
