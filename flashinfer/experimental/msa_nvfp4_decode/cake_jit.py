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
            "module": "cake_msa_nvfp4_decode_20224cc59640fc6610a6",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_20224cc59640fc6610a6_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_20224cc59640fc6610a6_binding.cu",
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
            "closure_sha256": "1b5fa54a3eaabaa982f482b59b1d82cd574a54a0d2b1713447f27c760a2833d9",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "1b5fa54a3eaabaa982f482b59b1d82cd574a54a0d2b1713447f27c760a2833d9",
    },
    "cake_msa_nvfp4_decode_sm_100a_split2": {
        "arch": "sm_100a",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_f3a34fa64cc2bf6f4624",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_f3a34fa64cc2bf6f4624_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_f3a34fa64cc2bf6f4624_binding.cu",
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
            "closure_sha256": "f676413cbc04bf6b3240fd524fed16611a8a2e9085c0ac98f944797d047dcf46",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "f676413cbc04bf6b3240fd524fed16611a8a2e9085c0ac98f944797d047dcf46",
    },
    "cake_msa_nvfp4_decode_sm_100a_split4": {
        "arch": "sm_100a",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_9ddf0d6ef957f0cbafc2",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_9ddf0d6ef957f0cbafc2_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_9ddf0d6ef957f0cbafc2_binding.cu",
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
            "closure_sha256": "55d868c25f54b1828ab5dea8c93a53579a4e1f57343160955ce83d1a0848b560",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "55d868c25f54b1828ab5dea8c93a53579a4e1f57343160955ce83d1a0848b560",
    },
    "cake_msa_nvfp4_decode_sm_100a_split8": {
        "arch": "sm_100a",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_1d2221f7da7db60808b3",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_1d2221f7da7db60808b3_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_1d2221f7da7db60808b3_binding.cu",
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
            "closure_sha256": "fd10d34fab01e74ded55a5aa8e47be60020f73217f3d501c7b844c1e54986f4d",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "fd10d34fab01e74ded55a5aa8e47be60020f73217f3d501c7b844c1e54986f4d",
    },
    "cake_msa_nvfp4_decode_sm_103a_split1": {
        "arch": "sm_103a",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_327f6606c686359b3132",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_327f6606c686359b3132_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_327f6606c686359b3132_binding.cu",
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
            "closure_sha256": "47ae6750050821b1b6ff154744193db0f5156993e06c46ad8acbf96249ebe059",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "47ae6750050821b1b6ff154744193db0f5156993e06c46ad8acbf96249ebe059",
    },
    "cake_msa_nvfp4_decode_sm_103a_split2": {
        "arch": "sm_103a",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_a1e3934da22bd12305d4",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_a1e3934da22bd12305d4_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_a1e3934da22bd12305d4_binding.cu",
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
            "closure_sha256": "c241f9c01d7a6c8bea2f5f6e146e33e98c9b290b84d21f221f15d72dadff6420",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "c241f9c01d7a6c8bea2f5f6e146e33e98c9b290b84d21f221f15d72dadff6420",
    },
    "cake_msa_nvfp4_decode_sm_103a_split4": {
        "arch": "sm_103a",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_ca59e2e03e061299947d",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_ca59e2e03e061299947d_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_ca59e2e03e061299947d_binding.cu",
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
            "closure_sha256": "f451b6974744c8663f1359996b7a8ffe693e7e21ec5a5f806ccfe3f570f9a6a1",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "f451b6974744c8663f1359996b7a8ffe693e7e21ec5a5f806ccfe3f570f9a6a1",
    },
    "cake_msa_nvfp4_decode_sm_103a_split8": {
        "arch": "sm_103a",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_1a203cb3ba1cbb4a7bef",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_1a203cb3ba1cbb4a7bef_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_1a203cb3ba1cbb4a7bef_binding.cu",
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
            "closure_sha256": "d5759716a6dab2581d00d42222a64964f807723383853c1b235af80730b39efa",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "d5759716a6dab2581d00d42222a64964f807723383853c1b235af80730b39efa",
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
