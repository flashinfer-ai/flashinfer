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

# Explicit target-owned registration of the generated program.  One record per
# architecture; each record carries the two physical stages (the persistent
# decode kernel and the split-KV combine kernel) as separate extensions with
# their own translation units, compile flags, FFI entry and argument plan.
# Populated verbatim by the generated-program export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_nvfp4_mla_decode_sm_100a": {
        "arch": "sm_100a",
        "main": {
            "module": "cake_nvfp4_mla_decode_8345960a39a614c055f6",
            "sources": [
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_8345960a39a614c055f6_kernel.cu",
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_8345960a39a614c055f6_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "QS"],
                ["tma_buffer", "KV"],
                ["tma_buffer", "KVS"],
                ["buffer", "O"],
                ["buffer", "LSE"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "work_table"],
                ["buffer", "unit_first"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens"],
                ["buffer", "q_indptr"],
                ["buffer", "sinks"],
                ["parameter", "num_heads"],
                ["parameter", "q_len"],
                ["parameter", "max_pages"],
                ["parameter", "max_splits"],
                ["parameter", "scale_log2"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "9293963ed1e6a5cf61f4c72719686d5ed154b6e0022cba0f09ea40adf9e18e51",
            "tma_workspace_bytes": 0,
        },
        "reduce": {
            "module": "cake_nvfp4_mla_decode_dc4be86c62fbaf46c704",
            "sources": [
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_dc4be86c62fbaf46c704_kernel.cu",
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_dc4be86c62fbaf46c704_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "o"],
                ["buffer", "lse"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "row_splits"],
                ["parameter", "num_heads"],
                ["parameter", "max_splits"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ee7d2ed9a8034f2429a5e76aa9e1b8f7bcd1e03f0e2b3264411de0a7de8c2694",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "ccd228fffe51d693dc4dc0993e1ca82d186c25cbd326ee50f38fa5af8be3654b",
    },
    "cake_nvfp4_mla_decode_sm_103a": {
        "arch": "sm_103a",
        "main": {
            "module": "cake_nvfp4_mla_decode_5032fa112bb7a6ce884c",
            "sources": [
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_5032fa112bb7a6ce884c_kernel.cu",
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_5032fa112bb7a6ce884c_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "QS"],
                ["tma_buffer", "KV"],
                ["tma_buffer", "KVS"],
                ["buffer", "O"],
                ["buffer", "LSE"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "work_table"],
                ["buffer", "unit_first"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens"],
                ["buffer", "q_indptr"],
                ["buffer", "sinks"],
                ["parameter", "num_heads"],
                ["parameter", "q_len"],
                ["parameter", "max_pages"],
                ["parameter", "max_splits"],
                ["parameter", "scale_log2"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "d37bcf6e06e06d1b9cf54f264aa2fed2d0274daf9e87f56e6a3f1ed39f73642c",
            "tma_workspace_bytes": 0,
        },
        "reduce": {
            "module": "cake_nvfp4_mla_decode_658a7a88207e25d82f4f",
            "sources": [
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_658a7a88207e25d82f4f_kernel.cu",
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_658a7a88207e25d82f4f_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "o"],
                ["buffer", "lse"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "row_splits"],
                ["parameter", "num_heads"],
                ["parameter", "max_splits"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "c551d5cde4237296decc64a9d3c45764f7e7924dd45bb3861244dd1277bdc560",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "4a264853eca1013e3c669d94dcefd575b0527583732fe1111f47a0ebbcf1b587",
    },
}

STAGES = ("main", "reduce")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch`` (``sm_100a`` / ``sm_103a``)."""
    for name, record in MODULES.items():
        if record["arch"] == arch:
            return name
    raise NotImplementedError(
        "The generated NVFP4 MLA decode program for "
        f"{arch} is not registered in this checkout yet "
        "(see flashinfer-ai/flashinfer#5403)"
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
def gen_cake_nvfp4_mla_decode_module(name, stage):
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
def load_cake_nvfp4_mla_decode_module(name, stage):
    return gen_cake_nvfp4_mla_decode_module(name, stage).build_and_load()
