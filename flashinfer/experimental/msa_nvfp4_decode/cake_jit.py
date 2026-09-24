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
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated programs.  One record per
# (architecture, split-KV factor); each record carries the single physical
# stage (the persistent decode kernel) with its translation units, compile
# flags, FFI entry and argument plan, plus the resident-CTA count per SM the
# host uses to size the persistent grid.  Populated verbatim by the
# generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_msa_nvfp4_decode_sm_100a_split1": {
        "arch": "sm_100a",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_a3d3c78c43ded90b9719",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_a3d3c78c43ded90b9719_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_a3d3c78c43ded90b9719_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "6ed2106833bafe051cb900e2352b60998b1ae727c53f0af1b6a710b2b3ed94ab",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "6ed2106833bafe051cb900e2352b60998b1ae727c53f0af1b6a710b2b3ed94ab",
    },
    "cake_msa_nvfp4_decode_sm_100a_split2": {
        "arch": "sm_100a",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_71065bb124bf5466a6a2",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_71065bb124bf5466a6a2_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_71065bb124bf5466a6a2_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "4cf89672a102d35cf9fdd6d28563b619183f607f0d35a2f25b785667ab2a4890",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "4cf89672a102d35cf9fdd6d28563b619183f607f0d35a2f25b785667ab2a4890",
    },
    "cake_msa_nvfp4_decode_sm_100a_split4": {
        "arch": "sm_100a",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_1c7e327b9cc422697ef7",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_1c7e327b9cc422697ef7_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_1c7e327b9cc422697ef7_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "8314a9ef2acd1903884057bbf3848bb710d912e018517ebc35891ef1e5ee4a84",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "8314a9ef2acd1903884057bbf3848bb710d912e018517ebc35891ef1e5ee4a84",
    },
    "cake_msa_nvfp4_decode_sm_100a_split8": {
        "arch": "sm_100a",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_a2134dc5a661176b49ca",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_a2134dc5a661176b49ca_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_a2134dc5a661176b49ca_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "5915fb37c0698fdaba717b99b7c03dab913d02abf88e47df1393daab6e4de1a2",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "5915fb37c0698fdaba717b99b7c03dab913d02abf88e47df1393daab6e4de1a2",
    },
    "cake_msa_nvfp4_decode_sm_103a_split1": {
        "arch": "sm_103a",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_1793bc756284c9a798c2",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_1793bc756284c9a798c2_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_1793bc756284c9a798c2_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "3741c1f1f22256a1e5afa736cf5b44d0024a00d96030d1bbccd5d9b9c258ad5d",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "3741c1f1f22256a1e5afa736cf5b44d0024a00d96030d1bbccd5d9b9c258ad5d",
    },
    "cake_msa_nvfp4_decode_sm_103a_split2": {
        "arch": "sm_103a",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_a79a3423b034108b7158",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_a79a3423b034108b7158_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_a79a3423b034108b7158_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "f58ed4568196631dfc308547b6e149fd4f4796b68811a52e766e381f3e5618b8",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "f58ed4568196631dfc308547b6e149fd4f4796b68811a52e766e381f3e5618b8",
    },
    "cake_msa_nvfp4_decode_sm_103a_split4": {
        "arch": "sm_103a",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_c2f8f2db3c611545cebe",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_c2f8f2db3c611545cebe_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_c2f8f2db3c611545cebe_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "57a57a27e18ae485dc6db435644b6a3f7c0cfe5dd56dfcd3be94575b3ae29d28",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "57a57a27e18ae485dc6db435644b6a3f7c0cfe5dd56dfcd3be94575b3ae29d28",
    },
    "cake_msa_nvfp4_decode_sm_103a_split8": {
        "arch": "sm_103a",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_cdc636fab48db4b8d4bd",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_cdc636fab48db4b8d4bd_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_cdc636fab48db4b8d4bd_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "K_scale"],
                ["tma_buffer", "V"],
                ["tma_buffer", "V_scale"],
                ["buffer", "O"],
                ["buffer", "msa_lse"],
                ["buffer", "partial_O"],
                ["buffer", "partial_M"],
                ["buffer", "partial_D"],
                ["buffer", "split_completion"],
                ["buffer", "kv_indices"],
                ["buffer", "kv_indptr"],
                ["buffer", "task_kind"],
                ["buffer", "task_request"],
                ["buffer", "task_kv_head"],
                ["parameter", "total_q"],
                ["parameter", "seqlen_q"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "output_scale"],
                ["parameter", "msa_max_pages"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "56a9d1891c6ee68c953116f0b0241660ab9ff95198dee05ad90d10435aead92a",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "56a9d1891c6ee68c953116f0b0241660ab9ff95198dee05ad90d10435aead92a",
    },
}

STAGES = ("main",)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def registered_split_factors(arch: str) -> tuple[int, ...]:
    """Split-KV factors with a registered program for ``arch``."""
    return tuple(
        sorted(
            int(record["splits"])
            for record in MODULES.values()
            if record["arch"] == arch
        )
    )


def select_module(arch: str, splits: int) -> str:
    """Return the registered module name for ``arch`` and split factor ``splits``."""
    for name, record in MODULES.items():
        if record["arch"] == arch and int(record["splits"]) == int(splits):
            return name
    raise NotImplementedError(
        "The generated NVFP4 MSA decode program for "
        f"{arch} with split factor {splits} is not registered in this checkout"
    )


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_msa_nvfp4_decode_module(name, stage):
    record = MODULES[name]
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_msa_nvfp4_decode_module(name, stage):
    return gen_cake_msa_nvfp4_decode_module(name, stage).build_and_load()
