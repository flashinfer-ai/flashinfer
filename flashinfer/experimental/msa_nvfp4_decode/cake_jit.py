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
            "module": "cake_msa_nvfp4_decode_326252f6052975ed929d",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_326252f6052975ed929d_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_326252f6052975ed929d_binding.cu",
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
            "closure_sha256": "8f047f71d7e71b9ac68bc9633e3acce192cbdff7e521d7eefecb3dfaa4b601e7",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "8f047f71d7e71b9ac68bc9633e3acce192cbdff7e521d7eefecb3dfaa4b601e7",
    },
    "cake_msa_nvfp4_decode_sm_100a_split2": {
        "arch": "sm_100a",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_b6dd8f68bf17bd4c83b7",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_b6dd8f68bf17bd4c83b7_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_b6dd8f68bf17bd4c83b7_binding.cu",
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
            "closure_sha256": "069ad387df000671035f9f6ba3cd1fea69f30266b2d6df2ebbb3cae7092ed484",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "069ad387df000671035f9f6ba3cd1fea69f30266b2d6df2ebbb3cae7092ed484",
    },
    "cake_msa_nvfp4_decode_sm_100a_split4": {
        "arch": "sm_100a",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_88c9b2ce933138f28324",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_88c9b2ce933138f28324_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_88c9b2ce933138f28324_binding.cu",
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
            "closure_sha256": "7f1ac737e3ad1c0d0b1bf9c88f1f7c7ae6feca28ac9b52ff0ada4a064917b098",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "7f1ac737e3ad1c0d0b1bf9c88f1f7c7ae6feca28ac9b52ff0ada4a064917b098",
    },
    "cake_msa_nvfp4_decode_sm_100a_split8": {
        "arch": "sm_100a",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_d3882204d0f2f0d132df",
            "sources": [
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_d3882204d0f2f0d132df_kernel.cu",
                "cake_msa_nvfp4_decode/sm_100a/cake_msa_nvfp4_decode_d3882204d0f2f0d132df_binding.cu",
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
            "closure_sha256": "1112fae5bf8fd798675f3a5f0700237117329d4110d9f137c303f40ad9253bc6",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "1112fae5bf8fd798675f3a5f0700237117329d4110d9f137c303f40ad9253bc6",
    },
    "cake_msa_nvfp4_decode_sm_103a_split1": {
        "arch": "sm_103a",
        "splits": 1,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_88c1ff97b28714cec8fc",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_88c1ff97b28714cec8fc_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_88c1ff97b28714cec8fc_binding.cu",
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
            "closure_sha256": "248eb68f290624284636422afc3f77825332a67d2e7aff96c935b246f774c723",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "248eb68f290624284636422afc3f77825332a67d2e7aff96c935b246f774c723",
    },
    "cake_msa_nvfp4_decode_sm_103a_split2": {
        "arch": "sm_103a",
        "splits": 2,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_79af024671768d4a3c5b",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_79af024671768d4a3c5b_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_79af024671768d4a3c5b_binding.cu",
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
            "closure_sha256": "e583340a698e574d95242cd7130ef2828e5e876cacc52b769ec8ba1eb538844a",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "e583340a698e574d95242cd7130ef2828e5e876cacc52b769ec8ba1eb538844a",
    },
    "cake_msa_nvfp4_decode_sm_103a_split4": {
        "arch": "sm_103a",
        "splits": 4,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_cccc953ed2d2606cf222",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_cccc953ed2d2606cf222_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_cccc953ed2d2606cf222_binding.cu",
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
            "closure_sha256": "0c49956effd3e378c3d9f7e9e7eae08af91d052606aba144786e90d9b59ceb8c",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "0c49956effd3e378c3d9f7e9e7eae08af91d052606aba144786e90d9b59ceb8c",
    },
    "cake_msa_nvfp4_decode_sm_103a_split8": {
        "arch": "sm_103a",
        "splits": 8,
        "ctas_per_sm": 1,
        "main": {
            "module": "cake_msa_nvfp4_decode_4886a6e0e7257779240d",
            "sources": [
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_4886a6e0e7257779240d_kernel.cu",
                "cake_msa_nvfp4_decode/sm_103a/cake_msa_nvfp4_decode_4886a6e0e7257779240d_binding.cu",
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
            "closure_sha256": "8c9ba215f7fa70be8bd59b389e9869031e2db1a46f20b1a177e6edd8af808f31",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "8c9ba215f7fa70be8bd59b389e9869031e2db1a46f20b1a177e6edd8af808f31",
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
