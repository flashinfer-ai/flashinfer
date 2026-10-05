"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Verified source-only JIT loading for the Cake BF16 RMSNorm training kernels
(SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding that FlashInfer's JIT compiles
together for exactly one architecture.  ``MODULES``, ``ROUTES`` and
``HIDDEN_SIZES`` are filled by the exporter from the complete verified program
bundle.  The loader checks every source file's SHA-256 before its first build,
so a modified or partially delivered source tree fails closed instead of
silently running a different kernel.

A route is the ordered list of modules one public call launches:
``fwd`` / ``fwd_residual`` launch one kernel; ``bwd`` / ``bwd_residual``
launch the fused dx + deterministic dw kernel(s).  Token counts are runtime
scalars, so a changed number of rows never recompiles anything.

A module record is one (kernel, launch policy) pair.  A kernel launched under
several policies -- a different persistent grid or row range per stage -- has
one record per policy (``<kernel>__<stage>`` after the first), all sharing the
kernel's sources, digests, flags and JIT cache name, so it is compiled once.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

import tvm_ffi

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

# Filled mechanically from the complete verified program bundle.
MODULES: dict[str, dict[str, Any]] = {
    "cake_rmsnorm_train_01036297dc6e647b9e56": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "g"),
            ("buffer", "x"),
            ("buffer", "w"),
            ("buffer", "r"),
            ("buffer", "dx"),
            ("buffer", "dw"),
            ("buffer", "partial"),
            ("buffer", "counters"),
            ("buffer", "g_h"),
            ("parameter", "T"),
            ("parameter", "g_stride"),
            ("parameter", "x_stride"),
            ("parameter", "gh_stride"),
            ("parameter", "n_chunks"),
            ("parameter", "rows_per_chunk"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_01036297dc6e647b9e56_sm_100a",
            "sm_103a": "cake_rmsnorm_train_01036297dc6e647b9e56_sm_103a",
            "sm_107a": "cake_rmsnorm_train_01036297dc6e647b9e56_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "sm_scaled_bounded",
            "launch_bound_ctas_per_sm": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_01036297dc6e647b9e56",
        "kernel_symbol": "kernel_cake_rmsnorm_train_01036297dc6e647b9e56",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 21120,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "counters": "counters",
            "dw": "dw",
            "dx": "dx",
            "g": "g",
            "g_h": "g_h",
            "g_stride": "g_stride",
            "gh_stride": "g_h_stride",
            "n_chunks": "n_chunks",
            "partial": "partial",
            "r": "r",
            "rows_per_chunk": "rows_per_chunk",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
        },
        "role": "bwd_main",
        "select_rule": None,
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_01036297dc6e647b9e56_binding.cu": "92401216c621c08dbba01e613049aeca03d12d969172b0d03286939b201eba3c",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_01036297dc6e647b9e56_kernel.cu": "4ada94254a359a64c527322034ee5d61af41e54910826fcdcf6f1e0c4043e444",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_01036297dc6e647b9e56_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_01036297dc6e647b9e56_binding.cu",
        ],
        "workspace_rule": {
            "counters": 16,
            "kind": "persistent_grid",
        },
    },
    "cake_rmsnorm_train_0daf7b50565f2c0846a4": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_0daf7b50565f2c0846a4_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_0daf7b50565f2c0846a4",
        "kernel_symbol": "kernel_cake_rmsnorm_train_0daf7b50565f2c0846a4",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_0daf7b50565f2c0846a4_binding.cu": "858eeccab14727f1b08f76a0ae022e4dd5c4b079a317dc909d6ecd5949f76b66",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_0daf7b50565f2c0846a4_kernel.cu": "9ae66263d8863a61742d47b3ab7fa8f9a02f053e67a95b850d989f4ffbae9161",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_0daf7b50565f2c0846a4_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_0daf7b50565f2c0846a4_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_124a3bbdc1d5a6069326": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_124a3bbdc1d5a6069326_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 32,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_124a3bbdc1d5a6069326",
        "kernel_symbol": "kernel_cake_rmsnorm_train_124a3bbdc1d5a6069326",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_large",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_binding.cu": "f9b05122b6cf648fb02c3caef52022278adec63e029a4ba582c5454073d81afe",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_kernel.cu": "a68c70a5dbf539619db8771e4d3a04044a13256679369882315dc21abb6d9adb",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_124a3bbdc1d5a6069326__wide": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_124a3bbdc1d5a6069326_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 64,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_124a3bbdc1d5a6069326",
        "kernel_symbol": "kernel_cake_rmsnorm_train_124a3bbdc1d5a6069326",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_wide",
        "select_rule": {
            "above": 12288,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_binding.cu": "f9b05122b6cf648fb02c3caef52022278adec63e029a4ba582c5454073d81afe",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_kernel.cu": "a68c70a5dbf539619db8771e4d3a04044a13256679369882315dc21abb6d9adb",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_124a3bbdc1d5a6069326_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1af0069a0ec96fb8b742": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_1af0069a0ec96fb8b742_sm_100a",
            "sm_103a": "cake_rmsnorm_train_1af0069a0ec96fb8b742_sm_103a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_1af0069a0ec96fb8b742",
        "kernel_symbol": "kernel_cake_rmsnorm_train_1af0069a0ec96fb8b742",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1af0069a0ec96fb8b742_binding.cu": "6ff64d646383083bbf543fb69c1214e0c940c03ed73c8aefd846ddb0103a06b0",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1af0069a0ec96fb8b742_kernel.cu": "4fde9d1a11b73305f05ffef303c73a3f45cad1eedeac23de4cbf194b2ccae03f",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1af0069a0ec96fb8b742_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1af0069a0ec96fb8b742_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1fe87dab637f77d6d1ec": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_1fe87dab637f77d6d1ec_sm_100a",
            "sm_103a": "cake_rmsnorm_train_1fe87dab637f77d6d1ec_sm_103a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_1fe87dab637f77d6d1ec",
        "kernel_symbol": "kernel_cake_rmsnorm_train_1fe87dab637f77d6d1ec",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1fe87dab637f77d6d1ec_binding.cu": "145f3453afa1add9d4180f4f283b0e6d89cd571b794f91fa140eaa4fe4a687d9",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1fe87dab637f77d6d1ec_kernel.cu": "8ffcd8edf0c79ea03b4c69ca3bf7b0a85a7fa6aad57251db87e29f5d4f010112",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1fe87dab637f77d6d1ec_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_1fe87dab637f77d6d1ec_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_209e86d94c24fbe2ad17": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "g"),
            ("buffer", "x"),
            ("buffer", "w"),
            ("buffer", "r"),
            ("buffer", "dx"),
            ("buffer", "dw"),
            ("buffer", "partial"),
            ("buffer", "counters"),
            ("buffer", "g_h"),
            ("parameter", "T"),
            ("parameter", "g_stride"),
            ("parameter", "x_stride"),
            ("parameter", "gh_stride"),
            ("parameter", "n_chunks"),
            ("parameter", "rows_per_chunk"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_209e86d94c24fbe2ad17_sm_100a",
            "sm_103a": "cake_rmsnorm_train_209e86d94c24fbe2ad17_sm_103a",
            "sm_107a": "cake_rmsnorm_train_209e86d94c24fbe2ad17_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "sm_scaled_bounded",
            "launch_bound_ctas_per_sm": 2,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_209e86d94c24fbe2ad17",
        "kernel_symbol": "kernel_cake_rmsnorm_train_209e86d94c24fbe2ad17",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 21248,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "counters": "counters",
            "dw": "dw",
            "dx": "dx",
            "g": "g",
            "g_h": "g_h",
            "g_stride": "g_stride",
            "gh_stride": "g_h_stride",
            "n_chunks": "n_chunks",
            "partial": "partial",
            "r": "r",
            "rows_per_chunk": "rows_per_chunk",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
        },
        "role": "bwd_main",
        "select_rule": None,
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_209e86d94c24fbe2ad17_binding.cu": "103a81f00e7e66d3796bfa210139fe203c10df3ccc3fcbb7daab687f0f8474ec",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_209e86d94c24fbe2ad17_kernel.cu": "043088fba29bcea7e4b94e08f75d83268cc256a1792439d000b9a78b207a39ad",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_209e86d94c24fbe2ad17_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_209e86d94c24fbe2ad17_binding.cu",
        ],
        "workspace_rule": {
            "counters": 64,
            "kind": "persistent_grid",
        },
    },
    "cake_rmsnorm_train_238d1fcaf82b12cf76ce": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_238d1fcaf82b12cf76ce_sm_100a",
            "sm_103a": "cake_rmsnorm_train_238d1fcaf82b12cf76ce_sm_103a",
            "sm_107a": "cake_rmsnorm_train_238d1fcaf82b12cf76ce_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_238d1fcaf82b12cf76ce",
        "kernel_symbol": "kernel_cake_rmsnorm_train_238d1fcaf82b12cf76ce",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_main_residual",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_binding.cu": "8ec2e7ffbb0b84f35baf58ccc09430f6565602078caeb32620d32294d3ba00bb",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_kernel.cu": "c4b26a1196d07eb9311bc11a850547a361d2f7e55cfe34870a6e73684d90f9e3",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_238d1fcaf82b12cf76ce__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_238d1fcaf82b12cf76ce_sm_100a",
            "sm_103a": "cake_rmsnorm_train_238d1fcaf82b12cf76ce_sm_103a",
            "sm_107a": "cake_rmsnorm_train_238d1fcaf82b12cf76ce_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_238d1fcaf82b12cf76ce",
        "kernel_symbol": "kernel_cake_rmsnorm_train_238d1fcaf82b12cf76ce",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_binding.cu": "8ec2e7ffbb0b84f35baf58ccc09430f6565602078caeb32620d32294d3ba00bb",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_kernel.cu": "c4b26a1196d07eb9311bc11a850547a361d2f7e55cfe34870a6e73684d90f9e3",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_238d1fcaf82b12cf76ce_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_30aadcc325a2374f6dc7": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_30aadcc325a2374f6dc7_sm_100a",
            "sm_103a": "cake_rmsnorm_train_30aadcc325a2374f6dc7_sm_103a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_30aadcc325a2374f6dc7",
        "kernel_symbol": "kernel_cake_rmsnorm_train_30aadcc325a2374f6dc7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_30aadcc325a2374f6dc7_binding.cu": "e55f930412255605426aebdbe05df24bd8ae77fdc232c9133ea2c285786932f2",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_30aadcc325a2374f6dc7_kernel.cu": "bd53dbec8b818a3d9cf7cb1bd6b666390b45d8a400f83ad7e3803446c5d04ad7",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_30aadcc325a2374f6dc7_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_30aadcc325a2374f6dc7_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3553c65ba62d5346923c": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_3553c65ba62d5346923c_sm_100a",
            "sm_103a": "cake_rmsnorm_train_3553c65ba62d5346923c_sm_103a",
            "sm_107a": "cake_rmsnorm_train_3553c65ba62d5346923c_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_3553c65ba62d5346923c",
        "kernel_symbol": "kernel_cake_rmsnorm_train_3553c65ba62d5346923c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_3553c65ba62d5346923c_binding.cu": "76cb94a15f342d08d321ae6cc76bd94eb0f5914b0bfbdc537e1b6c7b67448d71",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_3553c65ba62d5346923c_kernel.cu": "36bbeb3b17c65647b2b4333c03ef452f20fe417ef840c1796fe6a512a80b3663",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_3553c65ba62d5346923c_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_3553c65ba62d5346923c_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_36dcbfe9ce26c2a5d859": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_103a": "cake_rmsnorm_train_36dcbfe9ce26c2a5d859_sm_103a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_36dcbfe9ce26c2a5d859",
        "kernel_symbol": "kernel_cake_rmsnorm_train_36dcbfe9ce26c2a5d859",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_36dcbfe9ce26c2a5d859_binding.cu": "a246689717fa8427eed66f3f61411fd5128482892bb0cb98becd6c35ba885bc3",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_36dcbfe9ce26c2a5d859_kernel.cu": "36359245687c911ff1ef038d2c84807e4f97763646d357adbe9862167926f4b4",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_36dcbfe9ce26c2a5d859_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_36dcbfe9ce26c2a5d859_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_39325dca02d9cc2ad87d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_39325dca02d9cc2ad87d_sm_100a",
            "sm_103a": "cake_rmsnorm_train_39325dca02d9cc2ad87d_sm_103a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_39325dca02d9cc2ad87d",
        "kernel_symbol": "kernel_cake_rmsnorm_train_39325dca02d9cc2ad87d",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_39325dca02d9cc2ad87d_binding.cu": "c47fe4a24e6e5678b84c0b5601925695ca21d87aadd8d4f6a5758e32f1aad729",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_39325dca02d9cc2ad87d_kernel.cu": "d9a9485486f2947e762b636935ef578453e900304e873fb19e8c005c429da68a",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_39325dca02d9cc2ad87d_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_39325dca02d9cc2ad87d_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4171ae444c864ce774b5": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_4171ae444c864ce774b5_sm_100a",
            "sm_103a": "cake_rmsnorm_train_4171ae444c864ce774b5_sm_103a",
            "sm_107a": "cake_rmsnorm_train_4171ae444c864ce774b5_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_4171ae444c864ce774b5",
        "kernel_symbol": "kernel_cake_rmsnorm_train_4171ae444c864ce774b5",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_quad",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_binding.cu": "a977268d11bdac3d762bb39b6449bd9eb1869cbd4d1e8db6b41d6c0207fc15be",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_kernel.cu": "173ce36c29c80a87c410fa7070117486bb3b56526162c0289869210f5b869f67",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4171ae444c864ce774b5__quad": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_4171ae444c864ce774b5_sm_100a",
            "sm_103a": "cake_rmsnorm_train_4171ae444c864ce774b5_sm_103a",
            "sm_107a": "cake_rmsnorm_train_4171ae444c864ce774b5_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_4171ae444c864ce774b5",
        "kernel_symbol": "kernel_cake_rmsnorm_train_4171ae444c864ce774b5",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_quad",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_binding.cu": "a977268d11bdac3d762bb39b6449bd9eb1869cbd4d1e8db6b41d6c0207fc15be",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_kernel.cu": "173ce36c29c80a87c410fa7070117486bb3b56526162c0289869210f5b869f67",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_4171ae444c864ce774b5_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_46d2fd4b879b5c5926a3": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_46d2fd4b879b5c5926a3_sm_100a",
            "sm_103a": "cake_rmsnorm_train_46d2fd4b879b5c5926a3_sm_103a",
            "sm_107a": "cake_rmsnorm_train_46d2fd4b879b5c5926a3_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_46d2fd4b879b5c5926a3",
        "kernel_symbol": "kernel_cake_rmsnorm_train_46d2fd4b879b5c5926a3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_mid_residual",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_46d2fd4b879b5c5926a3_binding.cu": "d6f421bbe2e4da538acf6b0c7dbc24faf19fae0102c5b21a2bf9f9363bf74086",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_46d2fd4b879b5c5926a3_kernel.cu": "17ae68bf186e8fb75f57ca11ae79f1d45fad1ca737da3020213fb21a1c8848b3",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_46d2fd4b879b5c5926a3_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_46d2fd4b879b5c5926a3_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7cbb889e8e3e28efde83": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "g"),
            ("buffer", "x"),
            ("buffer", "w"),
            ("buffer", "r"),
            ("buffer", "dx"),
            ("buffer", "dw"),
            ("buffer", "partial"),
            ("buffer", "counters"),
            ("buffer", "g_h"),
            ("parameter", "T"),
            ("parameter", "g_stride"),
            ("parameter", "x_stride"),
            ("parameter", "gh_stride"),
            ("parameter", "n_chunks"),
            ("parameter", "rows_per_chunk"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_7cbb889e8e3e28efde83_sm_100a",
            "sm_103a": "cake_rmsnorm_train_7cbb889e8e3e28efde83_sm_103a",
            "sm_107a": "cake_rmsnorm_train_7cbb889e8e3e28efde83_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "sm_scaled_bounded",
            "launch_bound_ctas_per_sm": 2,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_7cbb889e8e3e28efde83",
        "kernel_symbol": "kernel_cake_rmsnorm_train_7cbb889e8e3e28efde83",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 29952,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "counters": "counters",
            "dw": "dw",
            "dx": "dx",
            "g": "g",
            "g_h": "g_h",
            "g_stride": "g_stride",
            "gh_stride": "g_h_stride",
            "n_chunks": "n_chunks",
            "partial": "partial",
            "r": "r",
            "rows_per_chunk": "rows_per_chunk",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
        },
        "role": "bwd_main",
        "select_rule": None,
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_7cbb889e8e3e28efde83_binding.cu": "63efbb5a0cd46de35472e355b63e5173381cd15ff61d42d1d5a0f79ed2952bc0",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_7cbb889e8e3e28efde83_kernel.cu": "4c1153e005a4564c612d75ea71df633460583d4c40a816bac1f5a5bad5609f19",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_7cbb889e8e3e28efde83_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_7cbb889e8e3e28efde83_binding.cu",
        ],
        "workspace_rule": {
            "counters": 192,
            "kind": "persistent_grid",
        },
    },
    "cake_rmsnorm_train_842800c4a5e2c849515e": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "g"),
            ("buffer", "x"),
            ("buffer", "w"),
            ("buffer", "r"),
            ("buffer", "dx"),
            ("buffer", "dw"),
            ("buffer", "partial"),
            ("buffer", "counters"),
            ("buffer", "g_h"),
            ("parameter", "T"),
            ("parameter", "g_stride"),
            ("parameter", "x_stride"),
            ("parameter", "gh_stride"),
            ("parameter", "n_chunks"),
            ("parameter", "rows_per_chunk"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_842800c4a5e2c849515e_sm_100a",
            "sm_103a": "cake_rmsnorm_train_842800c4a5e2c849515e_sm_103a",
            "sm_107a": "cake_rmsnorm_train_842800c4a5e2c849515e_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "sm_scaled_bounded",
            "launch_bound_ctas_per_sm": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_842800c4a5e2c849515e",
        "kernel_symbol": "kernel_cake_rmsnorm_train_842800c4a5e2c849515e",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 7168,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "counters": "counters",
            "dw": "dw",
            "dx": "dx",
            "g": "g",
            "g_h": "g_h",
            "g_stride": "g_stride",
            "gh_stride": "g_h_stride",
            "n_chunks": "n_chunks",
            "partial": "partial",
            "r": "r",
            "rows_per_chunk": "rows_per_chunk",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
        },
        "role": "bwd_main_residual",
        "select_rule": None,
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_842800c4a5e2c849515e_binding.cu": "6bc04f8db5d5a988f3de999b7e5070d1e6a87c54a7e3b963e707a5665032ec2b",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_842800c4a5e2c849515e_kernel.cu": "b0dfb7e7f9212b4f0b0f6130ddcc4961de1c4a819accfc8ec056b2de48552366",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_842800c4a5e2c849515e_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_842800c4a5e2c849515e_binding.cu",
        ],
        "workspace_rule": {
            "counters": 96,
            "kind": "persistent_grid",
        },
    },
    "cake_rmsnorm_train_a4c187b2830114c1a279": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_a4c187b2830114c1a279_sm_100a",
            "sm_103a": "cake_rmsnorm_train_a4c187b2830114c1a279_sm_103a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a4c187b2830114c1a279",
        "kernel_symbol": "kernel_cake_rmsnorm_train_a4c187b2830114c1a279",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_a4c187b2830114c1a279_binding.cu": "86abb016c0a67901556d9742c90121e34c858660baef2522fa07fbf23204dc18",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_a4c187b2830114c1a279_kernel.cu": "7baf13320a67e546e551fb61c69e9e54a4e4e7fdb0797ccee7d1c8394fb01edd",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_a4c187b2830114c1a279_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_a4c187b2830114c1a279_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b6ac896edca3b49b37a9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_100a",
            "sm_103a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_103a",
            "sm_107a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b6ac896edca3b49b37a9",
        "kernel_symbol": "kernel_cake_rmsnorm_train_b6ac896edca3b49b37a9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_binding.cu": "186c4f9f262d3ab791a95963ffc648983e6a34cb91c2bb7c4f194882033fd370",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_kernel.cu": "386dcb4aa914c4251201224e6079de8f40affc08e3f7ef9cca4afa8ee6d32a5c",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b6ac896edca3b49b37a9__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_100a",
            "sm_103a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_103a",
            "sm_107a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b6ac896edca3b49b37a9",
        "kernel_symbol": "kernel_cake_rmsnorm_train_b6ac896edca3b49b37a9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_binding.cu": "186c4f9f262d3ab791a95963ffc648983e6a34cb91c2bb7c4f194882033fd370",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_kernel.cu": "386dcb4aa914c4251201224e6079de8f40affc08e3f7ef9cca4afa8ee6d32a5c",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b6ac896edca3b49b37a9__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_100a",
            "sm_103a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_103a",
            "sm_107a": "cake_rmsnorm_train_b6ac896edca3b49b37a9_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_b6ac896edca3b49b37a9",
        "kernel_symbol": "kernel_cake_rmsnorm_train_b6ac896edca3b49b37a9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_binding.cu": "186c4f9f262d3ab791a95963ffc648983e6a34cb91c2bb7c4f194882033fd370",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_kernel.cu": "386dcb4aa914c4251201224e6079de8f40affc08e3f7ef9cca4afa8ee6d32a5c",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_b6ac896edca3b49b37a9_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d44010e821a6c74587a6": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_d44010e821a6c74587a6_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d44010e821a6c74587a6",
        "kernel_symbol": "kernel_cake_rmsnorm_train_d44010e821a6c74587a6",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_d44010e821a6c74587a6_binding.cu": "af86b0f9d16e87d1648b1fbaf6b3ca92561e41a7a6c7ba70387e782cfda10f42",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_d44010e821a6c74587a6_kernel.cu": "d101c553bc43bdef213df05d516a459b4ae61e6d6cdfc59f1bac031d89c08cf8",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_d44010e821a6c74587a6_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_d44010e821a6c74587a6_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e19d033974aaa63cfbe2": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_e19d033974aaa63cfbe2_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "kernel_symbol": "kernel_cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu": "f789c352cdfa302bd878486205919354920c9f7a748ccf65ffa43f21b97905bc",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu": "72fc1d0e79fae488390da3cdd62f4b601a6175d54612c1bdd98e1624fd93231d",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e19d033974aaa63cfbe2__large": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_e19d033974aaa63cfbe2_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 16,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "kernel_symbol": "kernel_cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu": "f789c352cdfa302bd878486205919354920c9f7a748ccf65ffa43f21b97905bc",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu": "72fc1d0e79fae488390da3cdd62f4b601a6175d54612c1bdd98e1624fd93231d",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e19d033974aaa63cfbe2__main": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_e19d033974aaa63cfbe2_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 64,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "kernel_symbol": "kernel_cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_main",
        "select_rule": {
            "above": 30720,
            "at_most": None,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu": "f789c352cdfa302bd878486205919354920c9f7a748ccf65ffa43f21b97905bc",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu": "72fc1d0e79fae488390da3cdd62f4b601a6175d54612c1bdd98e1624fd93231d",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e19d033974aaa63cfbe2__wide": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_e19d033974aaa63cfbe2_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 32,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "kernel_symbol": "kernel_cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_wide",
        "select_rule": {
            "above": 12288,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu": "f789c352cdfa302bd878486205919354920c9f7a748ccf65ffa43f21b97905bc",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu": "72fc1d0e79fae488390da3cdd62f4b601a6175d54612c1bdd98e1624fd93231d",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e19d033974aaa63cfbe2__xwide": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_e19d033974aaa63cfbe2_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "kernel_symbol": "kernel_cake_rmsnorm_train_e19d033974aaa63cfbe2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_xwide",
        "select_rule": {
            "above": 24576,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu": "f789c352cdfa302bd878486205919354920c9f7a748ccf65ffa43f21b97905bc",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu": "72fc1d0e79fae488390da3cdd62f4b601a6175d54612c1bdd98e1624fd93231d",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e19d033974aaa63cfbe2_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e6b2419224108211616a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_e6b2419224108211616a_sm_100a",
            "sm_103a": "cake_rmsnorm_train_e6b2419224108211616a_sm_103a",
            "sm_107a": "cake_rmsnorm_train_e6b2419224108211616a_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_e6b2419224108211616a",
        "kernel_symbol": "kernel_cake_rmsnorm_train_e6b2419224108211616a",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_binding.cu": "72f8b5388e14a0a9d45269c6c6329d96810104b60e85770469072cf77d2f3bb5",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_kernel.cu": "91920641b20a90f92d94c2682f4dc8e8b21a8fbc9e1185b571d9b47c1ac1b7a5",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e6b2419224108211616a__main": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_100a": "cake_rmsnorm_train_e6b2419224108211616a_sm_100a",
            "sm_103a": "cake_rmsnorm_train_e6b2419224108211616a_sm_103a",
            "sm_107a": "cake_rmsnorm_train_e6b2419224108211616a_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_e6b2419224108211616a",
        "kernel_symbol": "kernel_cake_rmsnorm_train_e6b2419224108211616a",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_binding.cu": "72f8b5388e14a0a9d45269c6c6329d96810104b60e85770469072cf77d2f3bb5",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_kernel.cu": "91920641b20a90f92d94c2682f4dc8e8b21a8fbc9e1185b571d9b47c1ac1b7a5",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_e6b2419224108211616a_binding.cu",
        ],
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_f9a6840d24c353c6303a": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "u"),
            ("buffer", "w"),
            ("buffer", "y"),
            ("buffer", "h_new"),
            ("buffer", "r"),
            ("parameter", "T"),
            ("parameter", "x_stride"),
            ("parameter", "u_stride"),
            ("parameter", "eps"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_names": {
            "sm_107a": "cake_rmsnorm_train_f9a6840d24c353c6303a_sm_107a",
        },
        "compile_flags": [],
        "ffi_entry": "run",
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_f9a6840d24c353c6303a",
        "kernel_symbol": "kernel_cake_rmsnorm_train_f9a6840d24c353c6303a",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "persistent_ctas_per_sm": 1,
            "use_pdl": False,
        },
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "u_stride": "u_stride",
            "w": "w",
            "x": "x",
            "x_stride": "x_stride",
            "y": "y",
        },
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "source_sha256": {
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_f9a6840d24c353c6303a_binding.cu": "64cf1ce9c110ca8e86617df511e2d9d6639b6ffd472c5bd0ead9bc324a7946f6",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_f9a6840d24c353c6303a_kernel.cu": "c9b7c23733537a5de7c4e843dd9ab35259141607cea19897f8c28f6757168ae9",
        },
        "sources": [
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_f9a6840d24c353c6303a_kernel.cu",
            "csrc/cake_rmsnorm_train/cake_rmsnorm_train_f9a6840d24c353c6303a_binding.cu",
        ],
        "workspace_rule": None,
    },
}
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd_h2048": [
            "cake_rmsnorm_train_209e86d94c24fbe2ad17",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_01036297dc6e647b9e56",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_7cbb889e8e3e28efde83",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_842800c4a5e2c849515e",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b6ac896edca3b49b37a9",
            "cake_rmsnorm_train_e6b2419224108211616a",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_1af0069a0ec96fb8b742",
            "cake_rmsnorm_train_4171ae444c864ce774b5",
            "cake_rmsnorm_train_1fe87dab637f77d6d1ec",
            "cake_rmsnorm_train_39325dca02d9cc2ad87d",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_3553c65ba62d5346923c",
            "cake_rmsnorm_train_a4c187b2830114c1a279",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_30aadcc325a2374f6dc7",
            "cake_rmsnorm_train_46d2fd4b879b5c5926a3",
            "cake_rmsnorm_train_238d1fcaf82b12cf76ce",
        ],
    },
    "sm_103a": {
        "bwd_h2048": [
            "cake_rmsnorm_train_209e86d94c24fbe2ad17",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_01036297dc6e647b9e56",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_7cbb889e8e3e28efde83",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_842800c4a5e2c849515e",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_36dcbfe9ce26c2a5d859",
            "cake_rmsnorm_train_b6ac896edca3b49b37a9__mid",
            "cake_rmsnorm_train_e6b2419224108211616a",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_1af0069a0ec96fb8b742",
            "cake_rmsnorm_train_4171ae444c864ce774b5",
            "cake_rmsnorm_train_1fe87dab637f77d6d1ec",
            "cake_rmsnorm_train_39325dca02d9cc2ad87d",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_3553c65ba62d5346923c",
            "cake_rmsnorm_train_a4c187b2830114c1a279",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_30aadcc325a2374f6dc7",
            "cake_rmsnorm_train_46d2fd4b879b5c5926a3",
            "cake_rmsnorm_train_238d1fcaf82b12cf76ce",
        ],
    },
    "sm_107a": {
        "bwd_h2048": [
            "cake_rmsnorm_train_209e86d94c24fbe2ad17",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_01036297dc6e647b9e56",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_7cbb889e8e3e28efde83",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_842800c4a5e2c849515e",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b6ac896edca3b49b37a9__small",
            "cake_rmsnorm_train_d44010e821a6c74587a6",
            "cake_rmsnorm_train_124a3bbdc1d5a6069326",
            "cake_rmsnorm_train_124a3bbdc1d5a6069326__wide",
            "cake_rmsnorm_train_e6b2419224108211616a__main",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_4171ae444c864ce774b5__quad",
            "cake_rmsnorm_train_0daf7b50565f2c0846a4",
            "cake_rmsnorm_train_f9a6840d24c353c6303a",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_3553c65ba62d5346923c",
            "cake_rmsnorm_train_e19d033974aaa63cfbe2",
            "cake_rmsnorm_train_e19d033974aaa63cfbe2__large",
            "cake_rmsnorm_train_e19d033974aaa63cfbe2__wide",
            "cake_rmsnorm_train_e19d033974aaa63cfbe2__xwide",
            "cake_rmsnorm_train_e19d033974aaa63cfbe2__main",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_238d1fcaf82b12cf76ce__small",
            "cake_rmsnorm_train_46d2fd4b879b5c5926a3",
            "cake_rmsnorm_train_238d1fcaf82b12cf76ce",
        ],
    },
}
HIDDEN_SIZES: tuple[int, ...] = (512, 2048, 6144)

SOURCE_PACKAGE = "cake_rmsnorm_train"
# One verified build per architecture: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set; there is no family fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
_ARCH_NVCC_FLAGS: dict[str, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}
# FP32 statistics contract: IEEE division and square root, no flush-to-zero
# and no fast-math rewriting.  The source build carries no fast-math flag; the
# JIT build states the precise-math defaults explicitly and never adds
# ``-use_fast_math``.
PRECISE_MATH_FLAGS = ("--prec-div=true", "--prec-sqrt=true", "--ftz=false")
FAST_MATH_FLAGS = frozenset(
    {
        "-use_fast_math",
        "--use_fast_math",
        "--ftz=true",
        "-ftz=true",
        "--prec-div=false",
        "-prec-div=false",
        "--prec-sqrt=false",
        "-prec-sqrt=false",
    }
)
KINDS = ("fwd", "bwd")
# Row-strided BF16 inputs are read with 16-byte vector accesses; the forward
# launcher additionally requires 32-byte aligned row storage (its 256-bit load
# configuration), the backward 16 bytes.  Row strides are multiples of eight
# elements.
ROW_ALIGNMENT_ELEMENTS = 8
FORWARD_ALIGNMENT_BYTES = 32
BACKWARD_ALIGNMENT_BYTES = 16
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "y",
    "r",
    "u",
    "h_new",
    "g",
    "g_h",
    "dx",
    "dw",
    "partial",
    "counters",
)
LOGICAL_SCALARS = (
    "rows",
    "eps",
    "x_stride",
    "u_stride",
    "g_stride",
    "g_h_stride",
    "n_chunks",
    "rows_per_chunk",
)


def route_key(hidden: int, kind: str, residual: bool) -> str:
    """Name the route of one public call: ``<kind>[_residual]_h<hidden>``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return f"{kind}{'_residual' if residual else ''}_h{int(hidden)}"


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """Hidden sizes with a complete fwd + bwd route on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(
        hidden
        for hidden in HIDDEN_SIZES
        if all(route_key(hidden, kind, False) in routes for kind in KINDS)
    )


def route_applies(
    *, device_capability: Sequence[int], hidden: int, kind: str, residual: bool
) -> bool:
    """Whether the verified export owns one training call."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return route_key(hidden, kind, residual) in ROUTES.get(arch, {})


def route_modules(arch: str, hidden: int, kind: str, residual: bool) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch)."""

    key = route_key(hidden, kind, residual)
    names = ROUTES.get(arch, {}).get(key)
    if not names:
        raise ValueError(
            f"unsupported Cake RMSNorm training route {key!r} on {arch!r}; exported "
            f"hidden sizes on this architecture: {supported_hidden_sizes(arch)}"
        )
    return tuple(names)


def _selects(rule: Optional[dict[str, Any]], rows: int) -> bool:
    if rule is None:
        return True
    kind = rule["kind"]
    if kind == "rows_between":
        at_most = rule["at_most"]
        return rows > int(rule["above"]) and (at_most is None or rows <= int(at_most))
    raise ValueError(f"unknown Cake RMSNorm training selection rule: {kind!r}")


def selected_modules(
    arch: str, hidden: int, kind: str, residual: bool, *, rows: int
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``rows`` tokens.

    A route may carry several programs of one kernel: the forward keeps
    latency programs for small and mid row counts next to its throughput
    program and switches on the row count.  Each module's recorded
    ``select_rule`` names the row range it owns (``rows_between``: ``above <
    rows <= at_most``, ``at_most`` ``None`` = unbounded; ``None`` = every call)
    and exactly the modules whose rule admits ``rows`` are launched, so the
    exported call runs the same program the source launcher selects.
    """

    rows = int(rows)
    names = tuple(
        name
        for name in route_modules(arch, hidden, kind, residual)
        if _selects(MODULES[name]["select_rule"], rows)
    )
    if not names:
        raise ValueError(
            f"no exported module of the {route_key(hidden, kind, residual)!r} route on "
            f"{arch!r} admits {rows} rows"
        )
    return names


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens under its recorded grid rule.

    ``rows_per_cta`` modules launch ``ceil(rows / rows_per_cta)`` CTAs, capped
    at ``grid_limit`` when the rule records a positive limit, else at
    ``grid_per_sm * SM count`` of the device when that is positive (persistent
    grid-stride row loop in the kernel).  ``fixed`` modules launch exactly
    ``ctas`` CTAs.  ``sm_scaled_bounded`` modules are the persistent backward
    grid: ``min(ctas_per_sm * SM count, rows)`` CTAs, at least one, where
    ``ctas_per_sm`` is the production grid factor capped at the co-residency
    the program's launch bound declares (a cooperative grid must be fully
    resident); the kernel's static row chunks follow that count, so the
    deterministic ``dw`` order is a pure function of ``(rows, device)``.
    ``rows`` must be positive; the public API returns empty outputs for zero
    rows without launching.
    """

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "rows_per_cta":
        grid_x = _ceil_div(rows, int(rule["rows_per_cta"]))
        limit = int(rule.get("grid_limit", 0))
        per_sm = int(rule.get("grid_per_sm", 0))
        if limit <= 0 and per_sm > 0:
            limit = per_sm * sm_count(device_index)
        if limit > 0:
            grid_x = min(grid_x, limit)
    elif kind == "fixed":
        grid_x = int(rule["ctas"])
    elif kind == "sm_scaled_bounded":
        grid_x = min(int(rule["ctas_per_sm"]) * sm_count(device_index), rows)
    else:
        raise ValueError(f"unknown Cake RMSNorm training grid rule: {kind!r}")
    return (max(grid_x, 1), 1, 1)


def chunking(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int]:
    """``(n_chunks, rows_per_chunk)`` of the backward partition for ``rows`` tokens.

    The chunk count equals the persistent grid of the main backward module and
    ``rows_per_chunk = ceil(rows / n_chunks)``; both are host-known and never
    depend on scheduling.
    """

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "persistent_grid":
        raise ValueError(f"module {record['kernel_symbol']} owns no backward workspace")
    n_chunks = launch_grid(record, rows, device_index=device_index)[0]
    return n_chunks, _ceil_div(int(rows), n_chunks)


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // int(multiple)) * int(multiple)


def max_chunks(record: dict[str, Any], *, device_index: int) -> int:
    """Largest chunk count the main backward module launches on ``device_index``."""

    rule = record["grid_rule"]
    if rule["kind"] == "sm_scaled_bounded":
        return int(rule["ctas_per_sm"]) * sm_count(device_index)
    if rule["kind"] == "fixed":
        return int(rule["ctas"])
    raise ValueError(f"grid rule {rule['kind']!r} has no bounded chunk count")


def workspace_layout(
    record: dict[str, Any], rows: int, *, device_index: int
) -> dict[str, int]:
    """Byte layout of the caller-owned backward workspace of one module.

    The workspace is one CUDA ``uint8`` buffer: the FP32 ``[n_chunks, hidden]``
    per-chunk dw partials first, then the ``uint32`` completion counters at a
    256-byte aligned offset.  It is sized for the largest chunk count of the
    device (``max_chunks``), so one buffer serves every token count; the
    counters must be zero when the buffer is first used and the kernels leave
    them zero, so no host write is needed between launches or CUDA Graph
    replays.
    """

    rule = record["workspace_rule"]
    n_chunks, rows_per_chunk = chunking(record, rows, device_index=device_index)
    hidden = int(record["hidden"])
    capacity = max_chunks(record, device_index=device_index)
    partial_bytes = 4 * capacity * hidden
    counters = int(rule["counters"])
    counters_offset = _round_up(partial_bytes, 256)
    total = counters_offset + _round_up(4 * max(counters, 1), 256)
    return {
        "n_chunks": n_chunks,
        "rows_per_chunk": rows_per_chunk,
        "capacity_chunks": capacity,
        "partial_bytes": partial_bytes,
        "counters": counters,
        "counters_offset": counters_offset,
        "total_bytes": total,
    }


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake RMSNorm training CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the RMSNorm training package: {relative!r}"
        )
    return _source_dir().joinpath(*parts[2:])


@functools.cache
def verified_sources(name: str) -> tuple[Path, ...]:
    """Resolve one module's sources and verify their recorded SHA-256 digests."""

    record = MODULES[name]
    paths = []
    for relative in record["sources"]:
        path = _source_path(relative)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Cake RMSNorm training source is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = record["source_sha256"][relative]
        if digest != expected:
            raise RuntimeError(
                f"Cake RMSNorm training source {relative} does not match its exported "
                f"digest (expected {expected}, found {digest})"
            )
        paths.append(path)
    return tuple(paths)


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake RMSNorm training target"
        )
    return arch


def module_cuda_cflags(name: str, arch: str) -> list[str]:
    """Exact nvcc flags of one module for ``arch``: the architecture, the recorded
    source build flags and the precise-math contract."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake RMSNorm training module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    flags = list(record["compile_flags"])
    fast = sorted(FAST_MATH_FLAGS.intersection(flags))
    if fast:
        raise RuntimeError(
            f"Cake RMSNorm training module {name} carries fast-math flags {fast}; "
            "the FP32 statistics contract forbids them"
        )
    return [*_ARCH_NVCC_FLAGS[arch], *flags, *PRECISE_MATH_FLAGS]


@functools.cache
def spec(name: str, arch: str) -> JitSpec:
    """JIT spec of one module built for ``arch`` (its own cache identity per architecture)."""

    record = MODULES[name]
    return gen_jit_spec(
        name=record["cache_names"][arch],
        sources=list(verified_sources(name)),
        extra_cuda_cflags=module_cuda_cflags(name, arch),
        extra_include_paths=[_source_dir().parent],
        use_fast_math=False,
    )


@functools.cache
def load(name: str, arch: str):
    return spec(name, arch).build_and_load()


def specs_for_arch(arch: str) -> list[JitSpec]:
    """Every kernel build of one architecture (for ahead-of-time warm-up); records
    that share a kernel (several launch policies) contribute one spec."""

    specs, seen = [], set()
    for name, record in MODULES.items():
        if arch in record["arches"] and record["cache_names"][arch] not in seen:
            seen.add(record["cache_names"][arch])
            specs.append(spec(name, arch))
    return specs


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid follows
    the module's recorded grid rule for ``rows`` tokens and the kernel runs on
    the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, rows, device_index=device_index)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "FAST_MATH_FLAGS",
    "HIDDEN_SIZES",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PRECISE_MATH_FLAGS",
    "ROUTES",
    "BACKWARD_ALIGNMENT_BYTES",
    "FORWARD_ALIGNMENT_BYTES",
    "ROW_ALIGNMENT_ELEMENTS",
    "arch_for_capability",
    "chunking",
    "device_arch",
    "launch_grid",
    "load",
    "max_chunks",
    "module_cuda_cflags",
    "route_applies",
    "route_key",
    "route_modules",
    "run_module",
    "selected_modules",
    "sm_count",
    "spec",
    "specs_for_arch",
    "supported_hidden_sizes",
    "verified_sources",
    "workspace_layout",
]
