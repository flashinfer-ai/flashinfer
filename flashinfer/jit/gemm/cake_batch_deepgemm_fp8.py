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
    "plan3": [
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
    "cake_batch_deepgemm_fp8_02cd3d43724acdbda982": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_02cd3d43724acdbda982_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_02cd3d43724acdbda982_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_1405732af0d574df0139": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_1405732af0d574df0139",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1405732af0d574df0139_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1405732af0d574df0139_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_1b7b5278ee7b1be99f15": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_1b7b5278ee7b1be99f15",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1b7b5278ee7b1be99f15_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1b7b5278ee7b1be99f15_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_103a"],
    },
    "cake_batch_deepgemm_fp8_1bb4c4c3d68e35871c2f": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_1bb4c4c3d68e35871c2f",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1bb4c4c3d68e35871c2f_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1bb4c4c3d68e35871c2f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_1eaf7d9c7921b59547f6": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_1eaf7d9c7921b59547f6",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1eaf7d9c7921b59547f6_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_1eaf7d9c7921b59547f6_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_215f20f0a5f44903608b": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_215f20f0a5f44903608b",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_215f20f0a5f44903608b_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_215f20f0a5f44903608b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_2ac9e23059edba242a92": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_2ac9e23059edba242a92",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2ac9e23059edba242a92_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2ac9e23059edba242a92_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_2c120bce2db6a19bf130": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_2c120bce2db6a19bf130",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2c120bce2db6a19bf130_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_2c120bce2db6a19bf130_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_364ce5d12a45a1d49602": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_364ce5d12a45a1d49602",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_364ce5d12a45a1d49602_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_364ce5d12a45a1d49602_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_54ec41340241b946c14e": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_54ec41340241b946c14e",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_54ec41340241b946c14e_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_54ec41340241b946c14e_binding.cu",
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
        "arg_plan": "plan3",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_59edc7df733ec03c53d5": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_59edc7df733ec03c53d5",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_59edc7df733ec03c53d5_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_59edc7df733ec03c53d5_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4_binding.cu",
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
    "cake_batch_deepgemm_fp8_8c4067fce6c805ee9300": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_8c4067fce6c805ee9300",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_8c4067fce6c805ee9300_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_8c4067fce6c805ee9300_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_90284e378e31f0382475": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_90284e378e31f0382475",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_90284e378e31f0382475_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_90284e378e31f0382475_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_929a5561abc99caa7f4f": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_929a5561abc99caa7f4f",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_929a5561abc99caa7f4f_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_929a5561abc99caa7f4f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_b53863e736461b0c669d": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_b53863e736461b0c669d",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_b53863e736461b0c669d_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_b53863e736461b0c669d_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a", "sm_103a"],
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
    "cake_batch_deepgemm_fp8_c1541da977c9a379e49b": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_c1541da977c9a379e49b",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c1541da977c9a379e49b_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_c1541da977c9a379e49b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_ce9db47474bd124735cc": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_ce9db47474bd124735cc",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ce9db47474bd124735cc_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_ce9db47474bd124735cc_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan1",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_d22ea585c19c028e29ae": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_d22ea585c19c028e29ae",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d22ea585c19c028e29ae_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_d22ea585c19c028e29ae_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_e306f272ed478b53bd16": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_e306f272ed478b53bd16",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_e306f272ed478b53bd16_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_e306f272ed478b53bd16_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a"],
    },
    "cake_batch_deepgemm_fp8_e6e78deccdcbf45fa870": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_e6e78deccdcbf45fa870",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_e6e78deccdcbf45fa870_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_e6e78deccdcbf45fa870_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_103a"],
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
    "cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan0",
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_batch_deepgemm_fp8_fffc74e89022d23621dd": {
        "kernel": "kernel_cake_batch_deepgemm_fp8_fffc74e89022d23621dd",
        "sources": [
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_fffc74e89022d23621dd_kernel.cu",
            "cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_fffc74e89022d23621dd_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "tma_abi": "grid_constant",
        "arg_plan": "plan2",
        "arches": ["sm_100a"],
    },
}

MODULES: dict[str, dict[str, str]] = {
    "cake_batch_deepgemm_fp8_02cd3d43724acdbda982_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
    },
    "cake_batch_deepgemm_fp8_02cd3d43724acdbda982_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
    },
    "cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88",
    },
    "cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b",
    },
    "cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b",
    },
    "cake_batch_deepgemm_fp8_1405732af0d574df0139_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_1405732af0d574df0139",
    },
    "cake_batch_deepgemm_fp8_1405732af0d574df0139_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_1405732af0d574df0139",
    },
    "cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a",
    },
    "cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a",
    },
    "cake_batch_deepgemm_fp8_1b7b5278ee7b1be99f15_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_1b7b5278ee7b1be99f15",
    },
    "cake_batch_deepgemm_fp8_1bb4c4c3d68e35871c2f_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_1bb4c4c3d68e35871c2f",
    },
    "cake_batch_deepgemm_fp8_1eaf7d9c7921b59547f6_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_1eaf7d9c7921b59547f6",
    },
    "cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
    },
    "cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
    },
    "cake_batch_deepgemm_fp8_215f20f0a5f44903608b_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_215f20f0a5f44903608b",
    },
    "cake_batch_deepgemm_fp8_2ac9e23059edba242a92_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_2ac9e23059edba242a92",
    },
    "cake_batch_deepgemm_fp8_2ac9e23059edba242a92_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_2ac9e23059edba242a92",
    },
    "cake_batch_deepgemm_fp8_2c120bce2db6a19bf130_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_2c120bce2db6a19bf130",
    },
    "cake_batch_deepgemm_fp8_2c120bce2db6a19bf130_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_2c120bce2db6a19bf130",
    },
    "cake_batch_deepgemm_fp8_364ce5d12a45a1d49602_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_364ce5d12a45a1d49602",
    },
    "cake_batch_deepgemm_fp8_364ce5d12a45a1d49602_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_364ce5d12a45a1d49602",
    },
    "cake_batch_deepgemm_fp8_54ec41340241b946c14e_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_54ec41340241b946c14e",
    },
    "cake_batch_deepgemm_fp8_54ec41340241b946c14e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_54ec41340241b946c14e",
    },
    "cake_batch_deepgemm_fp8_57c401152579ac3ac327_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_57c401152579ac3ac327",
    },
    "cake_batch_deepgemm_fp8_57c401152579ac3ac327_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_57c401152579ac3ac327",
    },
    "cake_batch_deepgemm_fp8_59edc7df733ec03c53d5_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_59edc7df733ec03c53d5",
    },
    "cake_batch_deepgemm_fp8_59edc7df733ec03c53d5_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_59edc7df733ec03c53d5",
    },
    "cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4",
    },
    "cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4",
    },
    "cake_batch_deepgemm_fp8_87a3c50a96af69fbae87_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_87a3c50a96af69fbae87",
    },
    "cake_batch_deepgemm_fp8_8c4067fce6c805ee9300_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_8c4067fce6c805ee9300",
    },
    "cake_batch_deepgemm_fp8_8c4067fce6c805ee9300_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_8c4067fce6c805ee9300",
    },
    "cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6",
    },
    "cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6",
    },
    "cake_batch_deepgemm_fp8_90284e378e31f0382475_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_90284e378e31f0382475",
    },
    "cake_batch_deepgemm_fp8_90284e378e31f0382475_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_90284e378e31f0382475",
    },
    "cake_batch_deepgemm_fp8_929a5561abc99caa7f4f_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_929a5561abc99caa7f4f",
    },
    "cake_batch_deepgemm_fp8_929a5561abc99caa7f4f_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_929a5561abc99caa7f4f",
    },
    "cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
    },
    "cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
    },
    "cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2",
    },
    "cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2",
    },
    "cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a",
    },
    "cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a",
    },
    "cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a",
    },
    "cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a",
    },
    "cake_batch_deepgemm_fp8_b53863e736461b0c669d_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_b53863e736461b0c669d",
    },
    "cake_batch_deepgemm_fp8_b53863e736461b0c669d_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_b53863e736461b0c669d",
    },
    "cake_batch_deepgemm_fp8_ba92e96364e30012640e_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_ba92e96364e30012640e",
    },
    "cake_batch_deepgemm_fp8_ba92e96364e30012640e_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_ba92e96364e30012640e",
    },
    "cake_batch_deepgemm_fp8_c1541da977c9a379e49b_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_c1541da977c9a379e49b",
    },
    "cake_batch_deepgemm_fp8_c1541da977c9a379e49b_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_c1541da977c9a379e49b",
    },
    "cake_batch_deepgemm_fp8_ce9db47474bd124735cc_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_ce9db47474bd124735cc",
    },
    "cake_batch_deepgemm_fp8_ce9db47474bd124735cc_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_ce9db47474bd124735cc",
    },
    "cake_batch_deepgemm_fp8_d22ea585c19c028e29ae_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_d22ea585c19c028e29ae",
    },
    "cake_batch_deepgemm_fp8_d22ea585c19c028e29ae_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_d22ea585c19c028e29ae",
    },
    "cake_batch_deepgemm_fp8_e306f272ed478b53bd16_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_e306f272ed478b53bd16",
    },
    "cake_batch_deepgemm_fp8_e6e78deccdcbf45fa870_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_e6e78deccdcbf45fa870",
    },
    "cake_batch_deepgemm_fp8_eb727f8d280f83b52730_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_eb727f8d280f83b52730",
    },
    "cake_batch_deepgemm_fp8_eb727f8d280f83b52730_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_eb727f8d280f83b52730",
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
    "cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43",
    },
    "cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43_sm_103a": {
        "arch": "sm_103a",
        "program": "cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43",
    },
    "cake_batch_deepgemm_fp8_fffc74e89022d23621dd_sm_100a": {
        "arch": "sm_100a",
        "program": "cake_batch_deepgemm_fp8_fffc74e89022d23621dd",
    },
}

SERVING_PROGRAMS: dict[str, str] = {
    "n4096_k7168_g32_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_1405732af0d574df0139",
    "n4096_k7168_g32_s6e1_evict_normal_l1": "cake_batch_deepgemm_fp8_1602ca00dfa78ba89d6a",
    "n4096_k7168_g64_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_ce9db47474bd124735cc",
    "n4096_k7168_g64_s6e1_evict_normal_l1": "cake_batch_deepgemm_fp8_364ce5d12a45a1d49602",
    "n7168_k2048_g32_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_8f4bdd22c19c1467e2e6",
    "n7168_k2048_g32_s5e3_evict_normal_l8": "cake_batch_deepgemm_fp8_a7617f8e2deb15f1eb4a",
    "n7168_k2048_g64_s5e2_evict_normal_l8": "cake_batch_deepgemm_fp8_0fbe46cd4a67fa93336b",
    "n7168_k2048_g64_s5e3_evict_normal_l8": "cake_batch_deepgemm_fp8_90284e378e31f0382475",
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
                    program="cake_batch_deepgemm_fp8_2ac9e23059edba242a92",
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
                    program="cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_8c4067fce6c805ee9300",
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
                    program="cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43",
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
                    program="cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
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
                    program="cake_batch_deepgemm_fp8_e306f272ed478b53bd16",
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
                    program="cake_batch_deepgemm_fp8_e306f272ed478b53bd16",
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
                    program="cake_batch_deepgemm_fp8_1eaf7d9c7921b59547f6",
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
                    program="cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88",
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
                    program="cake_batch_deepgemm_fp8_0924c6ddd05c4e52de88",
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
                    program="cake_batch_deepgemm_fp8_929a5561abc99caa7f4f",
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
                    program="cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2",
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
                    program="cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4",
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
                    program="cake_batch_deepgemm_fp8_8c4067fce6c805ee9300",
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
                    program="cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
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
                    program="cake_batch_deepgemm_fp8_59edc7df733ec03c53d5",
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
                    program="cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
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
                    program="cake_batch_deepgemm_fp8_54ec41340241b946c14e",
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
                    program="cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4",
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
                    program="cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
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
                    program="cake_batch_deepgemm_fp8_59edc7df733ec03c53d5",
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
                    program="cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
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
                    program="cake_batch_deepgemm_fp8_54ec41340241b946c14e",
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
                    program="cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a",
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
                    program="cake_batch_deepgemm_fp8_d22ea585c19c028e29ae",
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
                    program="cake_batch_deepgemm_fp8_2c120bce2db6a19bf130",
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
                    program="cake_batch_deepgemm_fp8_c1541da977c9a379e49b",
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
                    program="cake_batch_deepgemm_fp8_215f20f0a5f44903608b",
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
                    program="cake_batch_deepgemm_fp8_fffc74e89022d23621dd",
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
                    program="cake_batch_deepgemm_fp8_1bb4c4c3d68e35871c2f",
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
                    program="cake_batch_deepgemm_fp8_b53863e736461b0c669d",
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
                    program="cake_batch_deepgemm_fp8_b53863e736461b0c669d",
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
                    program="cake_batch_deepgemm_fp8_2ac9e23059edba242a92",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_54ec41340241b946c14e",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
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
                    program="cake_batch_deepgemm_fp8_1b7b5278ee7b1be99f15",
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
                    program="cake_batch_deepgemm_fp8_929a5561abc99caa7f4f",
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
                    program="cake_batch_deepgemm_fp8_a51ac2146dc44e7c0fd2",
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
                    program="cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43",
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
                    program="cake_batch_deepgemm_fp8_9cb48a4615c58feb00f6",
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
                    program="cake_batch_deepgemm_fp8_741ad4b48907eb89c3f4",
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
                    program="cake_batch_deepgemm_fp8_8c4067fce6c805ee9300",
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
                    program="cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
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
                    program="cake_batch_deepgemm_fp8_59edc7df733ec03c53d5",
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
                    program="cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
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
                    program="cake_batch_deepgemm_fp8_54ec41340241b946c14e",
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
                    program="cake_batch_deepgemm_fp8_e6e78deccdcbf45fa870",
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
                    program="cake_batch_deepgemm_fp8_fe5a14cba1d988faaf43",
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
                    program="cake_batch_deepgemm_fp8_8c4067fce6c805ee9300",
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
                    program="cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9",
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
                    program="cake_batch_deepgemm_fp8_59edc7df733ec03c53d5",
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
                    program="cake_batch_deepgemm_fp8_02cd3d43724acdbda982",
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
                    program="cake_batch_deepgemm_fp8_54ec41340241b946c14e",
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
                    program="cake_batch_deepgemm_fp8_a6a50eabd6d1caa8ec6a",
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
                    program="cake_batch_deepgemm_fp8_d22ea585c19c028e29ae",
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
                    program="cake_batch_deepgemm_fp8_2c120bce2db6a19bf130",
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
                    program="cake_batch_deepgemm_fp8_c1541da977c9a379e49b",
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
                    program="cake_batch_deepgemm_fp8_b53863e736461b0c669d",
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
