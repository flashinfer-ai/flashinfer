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

# Explicit target-owned registration of the generated programs: one record per
# program (its translation units, FFI entry and argument plan) and the
# architectures that compile it.  Written by the generated-program export; do
# not edit by hand.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_balanced_gqa_decode_36ecd3362289b95340d6": {
        "kind": "mtp32",
        "sources": [
            "cake_balanced_gqa_decode/cake_balanced_gqa_decode_36ecd3362289b95340d6_kernel.cu",
            "cake_balanced_gqa_decode/cake_balanced_gqa_decode_36ecd3362289b95340d6_binding.cu",
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
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_balanced_gqa_decode_41e174c38744ae345da2": {
        "kind": "row",
        "sources": [
            "cake_balanced_gqa_decode/cake_balanced_gqa_decode_41e174c38744ae345da2_kernel.cu",
            "cake_balanced_gqa_decode/cake_balanced_gqa_decode_41e174c38744ae345da2_binding.cu",
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
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_balanced_gqa_decode_470794ded0549b0c6037": {
        "kind": "mtp64",
        "sources": [
            "cake_balanced_gqa_decode/cake_balanced_gqa_decode_470794ded0549b0c6037_kernel.cu",
            "cake_balanced_gqa_decode/cake_balanced_gqa_decode_470794ded0549b0c6037_binding.cu",
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
        "arches": ["sm_100a", "sm_103a"],
    },
}

# Program kind -> program.  "row" is the row-tile kernel (one 8-head query row
# per work item, any q_len_per_req); "mtp32" / "mtp64" are the packed-row MTP
# instances (32- and 64-row tiles for q_len_per_req 3..4 and 5..8).  The same
# program serves every architecture it lists.
KINDS: dict[str, str] = {
    "mtp32": "cake_balanced_gqa_decode_36ecd3362289b95340d6",
    "mtp64": "cake_balanced_gqa_decode_470794ded0549b0c6037",
    "row": "cake_balanced_gqa_decode_41e174c38744ae345da2",
}

ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def select_module(arch: str, kind: str = "row") -> str:
    """Return the registered program for ``arch`` and program ``kind``."""
    name = KINDS.get(kind)
    if name is None or arch not in PROGRAMS[name]["arches"]:
        raise NotImplementedError(
            f"The generated balanced GQA decode program ({kind}) for "
            f"{arch} is not registered in this checkout "
            "(see flashinfer-ai/flashinfer#4832)"
        )
    return name


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
def gen_cake_balanced_gqa_decode_module(name: str, arch: str):
    """JIT spec of program ``name`` compiled for ``arch`` (one library per architecture)."""
    record = PROGRAMS[name]
    if arch not in record["arches"]:
        raise NotImplementedError(f"program {name} is not built for {arch}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*ARCH_NVCC_FLAGS[arch], *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_balanced_gqa_decode_module(name: str, arch: str):
    return gen_cake_balanced_gqa_decode_module(name, arch).build_and_load()
