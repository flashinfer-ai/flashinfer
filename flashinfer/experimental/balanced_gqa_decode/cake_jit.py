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
# architecture; each record carries the single physical stage (the persistent
# balanced decode kernel) with its own translation units, compile flags, FFI
# entry and argument plan.  Populated verbatim by the generated-program
# export; do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_balanced_gqa_decode_mtp32_sm_100a": {
        "arch": "sm_100a",
        "kind": "mtp32",
        "main": {
            "module": "cake_balanced_gqa_decode_34b49a05ba3fbe366caa",
            "sources": [
                "cake_balanced_gqa_decode/sm_100a/cake_balanced_gqa_decode_34b49a05ba3fbe366caa_kernel.cu",
                "cake_balanced_gqa_decode/sm_100a/cake_balanced_gqa_decode_34b49a05ba3fbe366caa_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "V"],
                ["buffer", "O_ptr"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens_kv"],
                ["buffer", "partial_o"],
                ["buffer", "partial_stats"],
                ["buffer", "tile_counters"],
                ["buffer", "queue_counters"],
                ["parameter", "max_pages_per_seq"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "batch_size"],
                ["parameter", "q_len"],
                ["parameter", "max_items"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "bb9dee11182b3843e1f7a768c7f2adca5d2f590cc8dd832d4cbf8a72e40efb19",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "bb9dee11182b3843e1f7a768c7f2adca5d2f590cc8dd832d4cbf8a72e40efb19",
    },
    "cake_balanced_gqa_decode_mtp32_sm_103a": {
        "arch": "sm_103a",
        "kind": "mtp32",
        "main": {
            "module": "cake_balanced_gqa_decode_9d653e956b812e3af73b",
            "sources": [
                "cake_balanced_gqa_decode/sm_103a/cake_balanced_gqa_decode_9d653e956b812e3af73b_kernel.cu",
                "cake_balanced_gqa_decode/sm_103a/cake_balanced_gqa_decode_9d653e956b812e3af73b_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "V"],
                ["buffer", "O_ptr"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens_kv"],
                ["buffer", "partial_o"],
                ["buffer", "partial_stats"],
                ["buffer", "tile_counters"],
                ["buffer", "queue_counters"],
                ["parameter", "max_pages_per_seq"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "batch_size"],
                ["parameter", "q_len"],
                ["parameter", "max_items"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ae6b62cddb935ce227b432b02300503f3ba206b57c1b6bdab3fb62aa64fee2aa",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "ae6b62cddb935ce227b432b02300503f3ba206b57c1b6bdab3fb62aa64fee2aa",
    },
    "cake_balanced_gqa_decode_mtp64_sm_100a": {
        "arch": "sm_100a",
        "kind": "mtp64",
        "main": {
            "module": "cake_balanced_gqa_decode_590799352820c65c9ac0",
            "sources": [
                "cake_balanced_gqa_decode/sm_100a/cake_balanced_gqa_decode_590799352820c65c9ac0_kernel.cu",
                "cake_balanced_gqa_decode/sm_100a/cake_balanced_gqa_decode_590799352820c65c9ac0_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "V"],
                ["buffer", "O_ptr"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens_kv"],
                ["buffer", "partial_o"],
                ["buffer", "partial_stats"],
                ["buffer", "tile_counters"],
                ["buffer", "queue_counters"],
                ["parameter", "max_pages_per_seq"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "batch_size"],
                ["parameter", "q_len"],
                ["parameter", "max_items"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "19ee2736626d0bd4a71bfcac3b061d618412834824820902b76eeb4b63c67773",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "19ee2736626d0bd4a71bfcac3b061d618412834824820902b76eeb4b63c67773",
    },
    "cake_balanced_gqa_decode_mtp64_sm_103a": {
        "arch": "sm_103a",
        "kind": "mtp64",
        "main": {
            "module": "cake_balanced_gqa_decode_2783fca0e4f7b585ba5c",
            "sources": [
                "cake_balanced_gqa_decode/sm_103a/cake_balanced_gqa_decode_2783fca0e4f7b585ba5c_kernel.cu",
                "cake_balanced_gqa_decode/sm_103a/cake_balanced_gqa_decode_2783fca0e4f7b585ba5c_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Q"],
                ["tma_buffer", "K"],
                ["tma_buffer", "V"],
                ["buffer", "O_ptr"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens_kv"],
                ["buffer", "partial_o"],
                ["buffer", "partial_stats"],
                ["buffer", "tile_counters"],
                ["buffer", "queue_counters"],
                ["parameter", "max_pages_per_seq"],
                ["parameter", "softmax_scale_log2"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "batch_size"],
                ["parameter", "q_len"],
                ["parameter", "max_items"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "2577686fd80c1a427b4282fcde163027914d1b19926b2292541b77eb982eb456",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "2577686fd80c1a427b4282fcde163027914d1b19926b2292541b77eb982eb456",
    },
    "cake_balanced_gqa_decode_sm_100a": {
        "arch": "sm_100a",
        "kind": "row",
        "main": {
            "module": "cake_balanced_gqa_decode_db1624c38dc2f7ee3831",
            "sources": [
                "cake_balanced_gqa_decode/sm_100a/cake_balanced_gqa_decode_db1624c38dc2f7ee3831_kernel.cu",
                "cake_balanced_gqa_decode/sm_100a/cake_balanced_gqa_decode_db1624c38dc2f7ee3831_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Qt"],
                ["tma_buffer", "K"],
                ["tma_buffer", "V"],
                ["buffer", "O_ptr"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens_kv"],
                ["buffer", "partial_o"],
                ["buffer", "partial_stats"],
                ["buffer", "tile_counters"],
                ["buffer", "queue_counters"],
                ["parameter", "max_pages_per_seq"],
                ["parameter", "softmax_scale"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "group_ratio"],
                ["parameter", "batch_size"],
                ["parameter", "q_len"],
                ["parameter", "max_items"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "d3daccb022ced5249ad4f32175714650db940a32865c0ba4bda5993dca6681ca",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "d3daccb022ced5249ad4f32175714650db940a32865c0ba4bda5993dca6681ca",
    },
    "cake_balanced_gqa_decode_sm_103a": {
        "arch": "sm_103a",
        "kind": "row",
        "main": {
            "module": "cake_balanced_gqa_decode_e781b959832d27370afa",
            "sources": [
                "cake_balanced_gqa_decode/sm_103a/cake_balanced_gqa_decode_e781b959832d27370afa_kernel.cu",
                "cake_balanced_gqa_decode/sm_103a/cake_balanced_gqa_decode_e781b959832d27370afa_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "Qt"],
                ["tma_buffer", "K"],
                ["tma_buffer", "V"],
                ["buffer", "O_ptr"],
                ["buffer", "page_table"],
                ["buffer", "seq_lens_kv"],
                ["buffer", "partial_o"],
                ["buffer", "partial_stats"],
                ["buffer", "tile_counters"],
                ["buffer", "queue_counters"],
                ["parameter", "max_pages_per_seq"],
                ["parameter", "softmax_scale"],
                ["parameter", "num_q_heads"],
                ["parameter", "num_kv_heads"],
                ["parameter", "group_ratio"],
                ["parameter", "batch_size"],
                ["parameter", "q_len"],
                ["parameter", "max_items"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "9e0a8a3d38d3db854c3d875e9aaa284d02c9e3cf8458d50e8d062e255d5d436b",
            "tma_workspace_bytes": 0,
        },
        "closure_sha256": "9e0a8a3d38d3db854c3d875e9aaa284d02c9e3cf8458d50e8d062e255d5d436b",
    },
}

STAGES = ("main",)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


# Program kinds: "row" = the row-tile kernel (any q_len_per_req), "mtp32" /
# "mtp64" = the packed-row MTP kernel instances (32- and 64-row tiles).  Records
# without a "kind" field are the row-tile program.
def select_module(arch: str, kind: str = "row") -> str:
    """Return the registered module name for ``arch`` and program ``kind``."""
    for name, record in MODULES.items():
        if record["arch"] == arch and record.get("kind", "row") == kind:
            return name
    raise NotImplementedError(
        f"The generated balanced GQA decode program ({kind}) for "
        f"{arch} is not registered in this checkout yet "
        "(see flashinfer-ai/flashinfer#4832)"
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
def gen_cake_balanced_gqa_decode_module(name, stage):
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
def load_cake_balanced_gqa_decode_module(name, stage):
    return gen_cake_balanced_gqa_decode_module(name, stage).build_and_load()
