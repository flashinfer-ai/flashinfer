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
            "module": "cake_nvfp4_mla_decode_c33925bbab8964a3547b",
            "sources": [
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_c33925bbab8964a3547b_kernel.cu",
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_c33925bbab8964a3547b_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "QS"],
                ["tma_buffer", "KV"],
                ["tma_buffer", "KVS"],
                ["tma_buffer", "OT"],
                ["tma_buffer", "PO"],
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
                ["parameter", "rows_total"],
                ["parameter", "num_heads"],
                ["parameter", "q_len"],
                ["parameter", "max_pages"],
                ["parameter", "max_splits"],
                ["parameter", "scale_log2"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "5f35ec151e3a8e867ca3294d4804c9b73bbdaced5fb94a5298c102df355d788e",
            "tma_workspace_bytes": 0,
        },
        "reduce": {
            "module": "cake_nvfp4_mla_decode_5c02ef0c72ec6887b7f5",
            "sources": [
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_5c02ef0c72ec6887b7f5_kernel.cu",
                "cake_nvfp4_mla_decode/sm_100a/cake_nvfp4_mla_decode_5c02ef0c72ec6887b7f5_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "o"],
                ["buffer", "lse"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "row_splits"],
                ["parameter", "total_q"],
                ["parameter", "num_heads"],
                ["parameter", "max_splits"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "f536377f9a2be25fd7e086698d51b67ddaefc26aa62fb26c65e9e15639820dd4",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "deda31f47a7e96d958dca5db306f7babebe692936d7fc517d2062b6eff1f049d",
    },
    "cake_nvfp4_mla_decode_sm_103a": {
        "arch": "sm_103a",
        "main": {
            "module": "cake_nvfp4_mla_decode_6260e249381d96576baa",
            "sources": [
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_6260e249381d96576baa_kernel.cu",
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_6260e249381d96576baa_binding.cu",
            ],
            "compile_flags": ["--use_fast_math"],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "QS"],
                ["tma_buffer", "KV"],
                ["tma_buffer", "KVS"],
                ["tma_buffer", "OT"],
                ["tma_buffer", "PO"],
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
                ["parameter", "rows_total"],
                ["parameter", "num_heads"],
                ["parameter", "q_len"],
                ["parameter", "max_pages"],
                ["parameter", "max_splits"],
                ["parameter", "scale_log2"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "8c031ac48c9399d58aae84ea32f72e55ca82ada6562d85b1c8d799ead893a517",
            "tma_workspace_bytes": 0,
        },
        "reduce": {
            "module": "cake_nvfp4_mla_decode_ff76ef6d8e6453ca5e3c",
            "sources": [
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_ff76ef6d8e6453ca5e3c_kernel.cu",
                "cake_nvfp4_mla_decode/sm_103a/cake_nvfp4_mla_decode_ff76ef6d8e6453ca5e3c_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "o"],
                ["buffer", "lse"],
                ["buffer", "partial_o"],
                ["buffer", "partial_lse"],
                ["buffer", "row_splits"],
                ["parameter", "total_q"],
                ["parameter", "num_heads"],
                ["parameter", "max_splits"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "01f7776b52a67f27bf3f0e6b52e9f873a60ee08aa7c7a9d914bb53c952bad38d",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "ef1f4375f89cf17dbc10123a8ef49492023420681cf79404eabbf78cee6432ec",
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
