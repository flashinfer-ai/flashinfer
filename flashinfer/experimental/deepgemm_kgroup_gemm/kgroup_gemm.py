"""Prepared grouped packed FP4 E2M1 GEMM with BF16/FP32 output on SM100a/SM103a.

``MODULES`` (one record per generated program, shared by every architecture)
and ``ROUTES`` (schedule tier and output mode -> program) are written by the
generated-program export; do not edit them by hand. Tiers: ``general_s1`` /
``general_s2`` (N128, one or two store stages), ``n256``, ``swapped`` (BM240),
``small`` (BM128 x BN16, runtime tile geometry) and ``small_2x8x3`` (the same
schedule with the 2 x 8 x 3 tile geometry compiled in).
"""

from __future__ import annotations

import functools
from typing import Any

MODULES: dict[str, dict[str, Any]] = {
    "cake_deepgemm_kgroup_gemm_0742ea5c6ecc95a608b9": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_0742ea5c6ecc95a608b9_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_0742ea5c6ecc95a608b9_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_0f18ee590e94e1f764dd": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_0f18ee590e94e1f764dd_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_0f18ee590e94e1f764dd_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_26c5ac585619a31d6037": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_26c5ac585619a31d6037_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_26c5ac585619a31d6037_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_4af7ea7eac0d5742e076": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_4af7ea7eac0d5742e076_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_4af7ea7eac0d5742e076_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_5aca7ac0c071c1894ca5": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_5aca7ac0c071c1894ca5_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_5aca7ac0c071c1894ca5_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_5c35d5418b2ca3c43c18": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_5c35d5418b2ca3c43c18_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_5c35d5418b2ca3c43c18_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_72e983a00369bcf1e7c9": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_72e983a00369bcf1e7c9_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_72e983a00369bcf1e7c9_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_731a5a2ebd3e006d7154": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_731a5a2ebd3e006d7154_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_731a5a2ebd3e006d7154_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_81e492072a5f28bf5037": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_81e492072a5f28bf5037_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_81e492072a5f28bf5037_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_8a4b0a083bd7c973917b": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_8a4b0a083bd7c973917b_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_8a4b0a083bd7c973917b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a"],
    },
    "cake_deepgemm_kgroup_gemm_8c7c168642ff93b4bded": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_8c7c168642ff93b4bded_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_8c7c168642ff93b4bded_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_a2a735d32f4d0116c2ec": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_a2a735d32f4d0116c2ec_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_a2a735d32f4d0116c2ec_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_a6fc1f535d0e01ddac6a": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_a6fc1f535d0e01ddac6a_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_a6fc1f535d0e01ddac6a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_bac08f1a9f3d96b9b6cd": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_bac08f1a9f3d96b9b6cd_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_bac08f1a9f3d96b9b6cd_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_cc5b14b295a81421d291": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_cc5b14b295a81421d291_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_cc5b14b295a81421d291_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_dba5ed7a5804ae4d1c7b": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_dba5ed7a5804ae4d1c7b_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_dba5ed7a5804ae4d1c7b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_e790c4cfc1c3a686dd42": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_e790c4cfc1c3a686dd42_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_e790c4cfc1c3a686dd42_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_ef57d87e2e86c40f0b6d": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_ef57d87e2e86c40f0b6d_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_ef57d87e2e86c40f0b6d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_kgroup_gemm_fb4cdee6c050c1b2c52d": {
        "sources": [
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_fb4cdee6c050c1b2c52d_kernel.cu",
            "csrc/experimental/deepgemm_kgroup_gemm/generated/cake_deepgemm_kgroup_gemm_fb4cdee6c050c1b2c52d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "C_tma"],
            ["buffer", "grouped_layout"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "num_groups"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
}
ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "general_s1:bf16": "cake_deepgemm_kgroup_gemm_a2a735d32f4d0116c2ec",
        "general_s1:fp32": "cake_deepgemm_kgroup_gemm_0742ea5c6ecc95a608b9",
        "general_s1:fp32_acc": "cake_deepgemm_kgroup_gemm_0f18ee590e94e1f764dd",
        "general_s2:bf16": "cake_deepgemm_kgroup_gemm_fb4cdee6c050c1b2c52d",
        "general_s2:fp32": "cake_deepgemm_kgroup_gemm_ef57d87e2e86c40f0b6d",
        "general_s2:fp32_acc": "cake_deepgemm_kgroup_gemm_cc5b14b295a81421d291",
        "n256:bf16": "cake_deepgemm_kgroup_gemm_5aca7ac0c071c1894ca5",
        "n256:fp32": "cake_deepgemm_kgroup_gemm_8a4b0a083bd7c973917b",
        "n256:fp32_acc": "cake_deepgemm_kgroup_gemm_81e492072a5f28bf5037",
        "small:bf16": "cake_deepgemm_kgroup_gemm_8c7c168642ff93b4bded",
        "small:fp32": "cake_deepgemm_kgroup_gemm_dba5ed7a5804ae4d1c7b",
        "small:fp32_acc": "cake_deepgemm_kgroup_gemm_e790c4cfc1c3a686dd42",
        "small_2x8x3:bf16": "cake_deepgemm_kgroup_gemm_bac08f1a9f3d96b9b6cd",
        "small_2x8x3:fp32": "cake_deepgemm_kgroup_gemm_4af7ea7eac0d5742e076",
        "small_2x8x3:fp32_acc": "cake_deepgemm_kgroup_gemm_731a5a2ebd3e006d7154",
        "swapped:fp32": "cake_deepgemm_kgroup_gemm_72e983a00369bcf1e7c9",
    },
    "sm_103a": {
        "general_s1:bf16": "cake_deepgemm_kgroup_gemm_a2a735d32f4d0116c2ec",
        "general_s1:fp32": "cake_deepgemm_kgroup_gemm_0742ea5c6ecc95a608b9",
        "general_s1:fp32_acc": "cake_deepgemm_kgroup_gemm_0f18ee590e94e1f764dd",
        "general_s2:bf16": "cake_deepgemm_kgroup_gemm_fb4cdee6c050c1b2c52d",
        "general_s2:fp32": "cake_deepgemm_kgroup_gemm_ef57d87e2e86c40f0b6d",
        "general_s2:fp32_acc": "cake_deepgemm_kgroup_gemm_cc5b14b295a81421d291",
        "n256:bf16": "cake_deepgemm_kgroup_gemm_5aca7ac0c071c1894ca5",
        "n256:fp32": "cake_deepgemm_kgroup_gemm_5c35d5418b2ca3c43c18",
        "n256:fp32_acc": "cake_deepgemm_kgroup_gemm_81e492072a5f28bf5037",
        "small:bf16": "cake_deepgemm_kgroup_gemm_8c7c168642ff93b4bded",
        "small:fp32": "cake_deepgemm_kgroup_gemm_dba5ed7a5804ae4d1c7b",
        "small:fp32_acc": "cake_deepgemm_kgroup_gemm_e790c4cfc1c3a686dd42",
        "small_2x8x3:bf16": "cake_deepgemm_kgroup_gemm_bac08f1a9f3d96b9b6cd",
        "small_2x8x3:fp32": "cake_deepgemm_kgroup_gemm_4af7ea7eac0d5742e076",
        "small_2x8x3:fp32_acc": "cake_deepgemm_kgroup_gemm_731a5a2ebd3e006d7154",
        "swapped:bf16": "cake_deepgemm_kgroup_gemm_26c5ac585619a31d6037",
        "swapped:fp32": "cake_deepgemm_kgroup_gemm_72e983a00369bcf1e7c9",
        "swapped:fp32_acc": "cake_deepgemm_kgroup_gemm_a6fc1f535d0e01ddac6a",
    },
}

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# Measured L2 geometry of the BM240 swapped schedule: every output mode on
# 152 SMs, FP32 non-accumulating on 148 SMs.
_SWAPPED_GEOMETRIES = frozenset({(5120, 2304)})
_TILE_M = {"general": 128, "n256": 128, "swapped": 240, "small": 128}
_TILE_N = {"general": 128, "n256": 256, "swapped": 128, "small": 16}
# BN16 pays off once a group streams at least this many 256-wide k-blocks
# (2 k-blocks tie with N128 at 256x128, 9 k-blocks run 1.4x faster).
_SMALL_MIN_K_BLOCKS = 4
_SWIZZLE_M_BLOCKS = 8  # BM128 x BN16 (small) swizzle width for every SM count
# (grid_m, grid_n, num_groups) BN16 tile geometries served by an exact-geometry
# program (tile count and scheduler divisors compiled in; the group layout stays
# a runtime value): the 256x128 three-group rows.
_SMALL_EXACT_GEOMETRIES = frozenset({(2, 8, 3)})


def fast_division(divisor: int) -> tuple[int, int]:
    """``(multiplier, shift)`` with ``(x * multiplier) >> shift == x // divisor`` for ``0 <= x < 2**31``."""
    if divisor < 1 or divisor >= 1 << 30:
        raise ValueError(f"fast_division needs 1 <= divisor < 2**30, got {divisor}")
    shift = 31 + (divisor - 1).bit_length()
    return -(-(1 << shift) // divisor), shift


def swizzle_m_blocks(sm_count: int, tile_n: int) -> int:
    """M tiles per swizzle group (8 or 16): the strict minimum of the per-wave A+B footprint, 8 on a tie."""
    return min((8, 16), key=lambda blocks: blocks * 128 + ((sm_count + blocks - 1) // blocks) * tile_n)


def division_values(tier: str, grid_m: int, grid_n: int, sm_count: int) -> list[int]:
    """Host-precomputed tile-scheduler divisors of one tier: six int32 slots appended
    to the group metadata buffer, (multiplier, shift) for tiles per group, tiles
    per swizzle group and the tail swizzle group's tile count.

    ``n256`` swizzles M in groups of ``swizzle_m_blocks(sm_count, 256)`` tiles (16
    on 148 and 152 SMs), ``small`` in groups of 8; ``swapped`` swizzles N in groups
    of 16 tiles and reads only the swizzle and tail slots; ``general`` and the
    exact-geometry ``small_<grid_m>x<grid_n>x<groups>`` programs read no
    divisors. Multipliers above 2**31 are stored as their int32 bit pattern (the
    kernel reads the slots as u32).
    """
    if tier in ("n256", "small"):
        swizzle = swizzle_m_blocks(sm_count, 256) if tier == "n256" else _SWIZZLE_M_BLOCKS
        divisors = (grid_m * grid_n, grid_n * swizzle, grid_m % swizzle or swizzle)
    elif tier == "swapped":
        divisors = (1, grid_m * 16, grid_n % 16 or 16)
    else:
        return []
    values: list[int] = []
    for divisor in divisors:
        multiplier, shift = fast_division(divisor)
        values += [multiplier - (1 << 32) if multiplier >= 1 << 31 else multiplier, shift]
    return values


@functools.cache
def device_facts(device_index: int) -> tuple[str, int]:
    """``(arch, sm_count)`` of one CUDA device, queried once per process."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = _ARCHES.get(capability)
    if arch is None:
        raise RuntimeError(
            f"Grouped FP4 GEMM has generated programs for {sorted(_ARCHES.values())}, "
            f"not compute capability {capability}"
        )
    return arch, torch.cuda.get_device_properties(device_index).multi_processor_count


def output_mode(output_dtype: str, accumulate: bool) -> str:
    return f"{output_dtype}_acc" if accumulate else output_dtype


def schedule_tier(m: int, n: int, num_groups: int, sm_count: int, mode: str, max_k_blocks: int,
                  swapped: bool = True) -> str:
    """Physical schedule for one problem.

    Only the tile counts and the largest group's k-block count enter the
    decision; every schedule reads the group layout itself at runtime.
    ``swapped=False`` decides as if the BM240 schedule did not exist.
    """
    if (
        swapped
        and sm_count in (148, 152)
        and (m, n) in _SWAPPED_GEOMETRIES
        and (sm_count == 152 or mode == "fp32")
    ):
        return "swapped"
    physical_m = (m + 255) // 256 * 256
    general_tiles = (physical_m // 128) * (n // 128) * num_groups
    if n % 256 == 0 and (physical_m // 128) * (n // 256) * num_groups >= sm_count:
        return "n256"
    if general_tiles * 8 <= sm_count and max_k_blocks >= _SMALL_MIN_K_BLOCKS:
        return "small"
    return "general"


def route_candidates(m, n, group_ks, sm_count, output_dtype, accumulate, k_alignment=256) -> list[str]:
    """Routes for one problem, preferred first.

    The first entry is the schedule the decision chain picks for this SM
    count; the following entries are the schedules it would pick if the
    earlier ones did not exist (BM240 -> N256 / BN16 / N128, exact BN16
    geometry -> runtime BN16, then both N128 store-stage variants). The N128
    programs serve any problem, so the list always ends in a route every
    architecture carries.
    """
    mode = output_mode(output_dtype, accumulate)
    padded_ks = [(k + k_alignment - 1) // k_alignment * k_alignment for k in group_ks]
    max_k_blocks = max((k + 255) // 256 for k in padded_ks)
    physical_m = (m + 255) // 256 * 256
    tiers: list[str] = []
    tier = schedule_tier(m, n, len(group_ks), sm_count, mode, max_k_blocks)
    if tier == "swapped":
        tiers.append("swapped")
        tier = schedule_tier(m, n, len(group_ks), sm_count, mode, max_k_blocks, swapped=False)
    if tier == "n256":
        tiers.append("n256")
    elif tier == "small":
        grid_m, grid_n, _grid = launch_geometry(tier, physical_m, n, len(group_ks), sm_count)
        if (grid_m, grid_n, len(group_ks)) in _SMALL_EXACT_GEOMETRIES:
            tiers.append(f"small_{grid_m}x{grid_n}x{len(group_ks)}")
        tiers.append("small")
    k_blocks = sum(padded_ks) // len(group_ks) // 256
    threshold = {"bf16": 16, "fp32_acc": 24, "fp32": 32}[mode]
    tiers += ["general_s1", "general_s2"] if k_blocks >= threshold else ["general_s2", "general_s1"]
    return [f"{tier}:{mode}" for tier in tiers]


def route_key(m, n, group_ks, sm_count, output_dtype, accumulate, k_alignment=256) -> str:
    """The preferred route of one problem (``route_candidates(...)[0]``)."""
    return route_candidates(m, n, group_ks, sm_count, output_dtype, accumulate, k_alignment)[0]


def select_route(arch, m, n, group_ks, sm_count, output_dtype, accumulate, k_alignment=256) -> tuple[str, str]:
    """``(route, program)`` for one device: the first candidate route whose
    program the export measured on ``arch``.

    The route table lists, per architecture, the schedules the export reached
    on the catalog devices (148 SMs on SM100a, 152 SMs on SM103a). A device
    whose SM count reaches a schedule the export never measured on its
    architecture (the BM240 schedule in BF16 / FP32-accumulate on a 152-SM
    SM100a part such as GB200) takes the next schedule of the decision chain
    instead of failing the lookup.
    """
    table = ROUTES.get(arch, ROUTES)
    candidates = route_candidates(m, n, group_ks, sm_count, output_dtype, accumulate, k_alignment)
    for route in candidates:
        program = table.get(route)
        if program is not None:
            return route, program
    raise RuntimeError(f"no generated program on {arch} for any of {candidates}")


def launch_geometry(tier: str, physical_m: int, n: int, num_groups: int, sm_count: int):
    """``(grid_m, grid_n, grid)``: tile counts and the persistent two-CTA cluster grid."""
    base = tier.split("_")[0]
    grid_m = (physical_m + _TILE_M[base] - 1) // _TILE_M[base]
    grid_n = n // _TILE_N[base]
    tiles = num_groups * grid_m * grid_n
    if base == "small":
        return grid_m, grid_n, (sm_count // 2 * 2, 1, 1)
    return grid_m, grid_n, (min(sm_count // 2 * 2, (tiles + 1) // 2 * 2), 1, 1)


def grouped_layout_values(group_ks, k_alignment: int) -> list[int]:
    """Kernel group metadata: logical K ends over 256-aligned groups.

    A coarser ``k_alignment`` pads every group to a multiple of it; the padding
    holds encoded zero with valid scales, so the padded end is reported and the
    products are unchanged.
    """
    ends, cursor = [], 0
    for logical_k in group_ks:
        padded_k = (logical_k + k_alignment - 1) // k_alignment * k_alignment
        ends.append(cursor + (logical_k if k_alignment == 256 else padded_k))
        cursor += padded_k
    return ends


def _nvcc_flags(arch: str):
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


@functools.cache
def program_spec(arch: str, name: str):
    """JIT build specification of one generated program for ``arch``."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = MODULES[name]
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=[env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]],
        extra_cuda_cflags=[
            *_nvcc_flags(arch),
            *record["compile_flags"],
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,
    )


@functools.cache
def load_program(arch: str, name: str):
    return program_spec(arch, name).build_and_load()


class GroupedFP4Plan:
    """Prepared independent K-group products, with optional in-place FP32 addition.

    Each group owns out[group]. Padding contains encoded zero and valid scales.
    run() uses the current stream; one output has one owner. For all-empty
    groups, overwrite clears the output and accumulation preserves it.
    """

    def __init__(
        self,
        a,
        b,
        a_scales,
        b_scales,
        *,
        m,
        group_ks,
        k_alignment=256,
        use_psum_layout=True,
        output_dtype="bf16",
        accumulate=False,
        num_stages=7,
        out=None,
    ):
        import torch

        group_ks = tuple(int(k) for k in group_ks)
        if m < 1 or not group_ks or any(k < 0 for k in group_ks):
            raise ValueError("m must be positive and group_ks a nonempty list of nonnegative K sizes")
        if k_alignment < 256 or k_alignment % 256:
            raise ValueError("k_alignment must be a positive multiple of 256")
        if output_dtype not in ("bf16", "fp32") or (accumulate and output_dtype != "fp32"):
            raise ValueError("Output must be bf16/fp32; accumulation requires fp32")
        if num_stages != 7:
            raise ValueError("The generated schedules use seven load stages")
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.dtype not in (torch.int8, torch.uint8)
            or b.dtype not in (torch.int8, torch.uint8)
        ):
            raise ValueError("A and B must contain packed E2M1 int8/uint8 bytes")
        if a.device.type != "cuda":
            raise RuntimeError("Grouped FP4 GEMM requires a CUDA device")
        n = b.shape[0]
        if n < 1 or n % 128:
            raise ValueError("N must be positive and divisible by 128")
        physical_m = (m + 255) // 256 * 256
        padded_ks = [(k + k_alignment - 1) // k_alignment * k_alignment for k in group_ks]
        total_k = sum(padded_ks)
        if tuple(a.shape) != (physical_m, total_k // 2) or tuple(b.shape) != (n, total_k // 2):
            raise ValueError(
                "A/B must use independently padded concatenated groups and physical M aligned to 256"
            )
        for tensor, shape in (
            (a_scales, ((total_k + 127) // 128, physical_m)),
            (b_scales, ((total_k + 127) // 128, n)),
        ):
            if tensor.dtype not in (torch.int32, torch.uint32) or tuple(tensor.shape) != shape:
                raise ValueError(f"Packed UE8M0 scales must be int32/uint32 with shape {shape}")
        dtype = torch.bfloat16 if output_dtype == "bf16" else torch.float32
        if out is None:
            if accumulate:
                raise ValueError("In-place accumulation requires caller-provided initialized out")
            out = torch.empty((len(group_ks), physical_m, n), dtype=dtype, device=a.device)
        if out.dtype != dtype or tuple(out.shape) != (len(group_ks), physical_m, n):
            raise ValueError("out must match output_dtype and [groups, physical_M, N]")
        tensors = (a, b, a_scales, b_scales, out)
        if any(t.device != a.device or not t.is_contiguous() for t in tensors):
            raise ValueError("Operands, scales and output must be contiguous on one CUDA device")
        arch, sm_count = device_facts(a.device.index if a.device.index is not None else torch.cuda.current_device())
        self.route, self.program = select_route(arch, m, n, group_ks, sm_count, output_dtype, accumulate, k_alignment)
        self.storage, self.output = out, out[:, :m]
        self.empty, self.accumulate = total_k == 0, bool(accumulate)
        self.options = dict(
            M=m, N=n, group_ks=group_ks, k_alignment=k_alignment, use_psum_layout=bool(use_psum_layout),
            output_dtype=output_dtype, accumulate=bool(accumulate), num_stages=num_stages,
        )
        if self.empty:
            self._retained = tensors
            return
        tier = self.route.split(":")[0]
        grid_m, grid_n, grid = launch_geometry(tier, physical_m, n, len(group_ks), sm_count)
        self.grid = grid
        grouped_layout = torch.tensor(
            grouped_layout_values(group_ks, k_alignment) + division_values(tier, grid_m, grid_n, sm_count),
            dtype=torch.int32, device=a.device,
        )
        bindings = dict(
            A=a.view(torch.uint8), B=b.view(torch.uint8), SFA=a_scales.view(torch.uint32),
            SFB=b_scales.view(torch.uint32), C_tma=out, grouped_layout=grouped_layout,
            M=physical_m, N=n, K=total_k, grid_m=grid_m, grid_n=grid_n, num_groups=len(group_ks),
        )
        grid_by_axis = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
        record = MODULES[self.program]
        args = []
        for kind, name in record["arg_plan"]:
            if kind == "grid":
                args.append(grid_by_axis[name])
            elif name in bindings:
                args.append(bindings[name])
            else:
                raise RuntimeError(f"generated program {self.program} binds {name!r}, which this plan does not declare")
        module = load_program(arch, self.program)
        self._submission = getattr(module, record["ffi_entry"]), tuple(args)
        self._retained = (module, tensors, grouped_layout)

    def run(self):
        import tvm_ffi

        if self.empty:
            if not self.accumulate:
                self.storage.zero_()
            return self.output
        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.output
