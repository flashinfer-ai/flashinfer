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
from ...jit.core import gen_jit_spec, sm103a_nvcc_flags

# The one generated program of this backend: its device and binding
# translation units (relative to ``csrc/``), the compile flags of its source
# build, and the physical argument order of its ``run`` entry. The same source
# compiles for every listed architecture; the loader keys the cached library
# by architecture.
PROGRAM: dict[str, Any] = {
    "name": "cake_nvfp4_attention_6045de810304deb488d9",
    "arches": ["sm_103a"],
    "sources": [
        "cake_nvfp4_attention/cake_nvfp4_attention_6045de810304deb488d9_kernel.cu",
        "cake_nvfp4_attention/cake_nvfp4_attention_6045de810304deb488d9_binding.cu",
    ],
    "compile_flags": ["--use_fast_math"],
    "ffi_entry": "run",
    "arg_plan": [
        ["tma_buffer", "Q"],
        ["tma_buffer", "K"],
        ["tma_buffer", "Vt"],
        ["tma_buffer", "SFQ"],
        ["tma_buffer", "SFK"],
        ["tma_buffer", "SFVtLo"],
        ["tma_buffer", "SFVtHi"],
        ["tma_buffer", "O"],
        ["parameter", "seqlen_q"],
        ["parameter", "seqlen_kv"],
        ["parameter", "q_stride"],
        ["parameter", "kv_stride"],
        ["parameter", "softmax_scale_log2"],
        ["parameter", "total_bh"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
}

ARCHES = tuple(PROGRAM["arches"])
_NVCC_FLAGS = {"sm_103a": sm103a_nvcc_flags}


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
def gen_cake_nvfp4_attention_module(arch):
    if arch not in ARCHES:
        raise ValueError(f"NVFP4 attention is generated for {ARCHES}, not {arch}")
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in PROGRAM["sources"]]
    return gen_jit_spec(
        name=f"{PROGRAM['name']}_{arch}",
        sources=sources,
        extra_cuda_cflags=[*_NVCC_FLAGS[arch], *PROGRAM["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            root,
            *dict.fromkeys(p.parent for p in sources),
            *_header_dirs(),
        ],
        use_fast_math=False,
    )


@functools.cache
def load_cake_nvfp4_attention_module(arch):
    return gen_cake_nvfp4_attention_module(arch).build_and_load()
