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

# Explicit target-owned registration of the generated programs.
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): translation units, compile flags, FFI entry, argument
# plan and closure identity.  ``ROUTES`` maps ``"<variant>__<arch>"`` to the
# ordered stage -> module assignment of that program family:
#
# * ``bf16``          stages ``("attention",)``
# * ``nvfp4_fp4pv``   stages ``("quantize", "attention")``
# * ``nvfp4_fp8pv``   stages ``("quantize", "attention")``
#
# ``quantize`` is one fused single-launch quantizer per PV mode
# (``minimax_h3_varlen_nvfp4_quantize_qkv`` / ``..._quantize_qk_fp8v``); the
# ``attention`` binding of the NVFP4 routes carries the programmatic
# dependent launch attribute.  Every module is an exact-arch program (the
# sm_100a NVFP4 attention uses the hybrid exp2 recipe, sm_103a
# ``tcgen05.ld.red``).  Both literals are populated verbatim by the
# generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_minimax_h3_varlen_attention_0df56330b21e1d259c21": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_0df56330b21e1d259c21_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_0df56330b21e1d259c21_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["buffer", "Q_raw"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["buffer", "O"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "unit_table"],
            ["parameter", "total_tiles"],
            ["parameter", "num_heads"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "aeba79ff326986c99785d8e1d1d59c821c3a047b10c0dee11e25b461d041d5be",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_1664f7cb7998b7ec20a6": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_1664f7cb7998b7ec20a6_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_1664f7cb7998b7ec20a6_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "b668daef5f095672a7762e5f1b6b9027242a5a7f27c02e36332485eb70f821dc",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_2281e14d5960c8e399ed": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_2281e14d5960c8e399ed_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_2281e14d5960c8e399ed_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["buffer", "Q_raw"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["buffer", "O"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "unit_table"],
            ["parameter", "total_tiles"],
            ["parameter", "num_heads"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "30056ac2d75e4bb78a3f8bbba92e8e9365ee96cb0aced261a1e070363ce11154",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_576c36c016845b71ba04": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_576c36c016845b71ba04_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_576c36c016845b71ba04_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp4_t"],
            ["buffer", "v_scale_lo"],
            ["buffer", "v_scale_hi"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "76fb2acc56506446693850695628304f7a59101af5b4a23d25742c1eb9849635",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_867261f6b4079dd3db64": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_867261f6b4079dd3db64_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_867261f6b4079dd3db64_binding.cu",
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
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "2a48abcebae9079e392421fc3f3fdbba834ff7a23afcdcff20fe8a69ee40591e",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_c330e02790f956ad0b08": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_c330e02790f956ad0b08_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_c330e02790f956ad0b08_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp4_t"],
            ["buffer", "v_scale_lo"],
            ["buffer", "v_scale_hi"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "f0ce1048a69b3a3a41b75dc6684879c3356a6829099079741e9b69eaefbd4297",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_c33c81e009ba5bcc8244": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_c33c81e009ba5bcc8244_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_c33c81e009ba5bcc8244_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "SFQ"],
            ["tma_buffer", "SFK"],
            ["buffer", "O"],
            ["buffer", "v_amax"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "a5d136f9453edfb9f8315acfd4ba7db439052f753f282ccbd858228b67cc72b5",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_cca6dc137573738ca929": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_cca6dc137573738ca929_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_cca6dc137573738ca929_binding.cu",
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
            ["buffer", "O"],
            ["buffer", "cl_head"],
            ["buffer", "cl_seg_begin"],
            ["buffer", "cl_seg_len"],
            ["buffer", "cl_kv_base"],
            ["buffer", "cl_q_block"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "dc80a2073d937b3186f8b819b5642d99743e0899229f42f8de7b94699e31a6ae",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_e9349d858cf7daf87d83": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_e9349d858cf7daf87d83_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_100a/cake_minimax_h3_varlen_attention_e9349d858cf7daf87d83_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp8"],
            ["buffer", "v_amax"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6012fd3f891aba70d2e3de973fb2af98fdfbea7493dc08c5e539876a5557c9fd",
        "tma_workspace_bytes": 0,
    },
    "cake_minimax_h3_varlen_attention_f71a833f43b9a3f3aa0b": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_f71a833f43b9a3f3aa0b_kernel.cu",
            "cake_minimax_h3_varlen_attention/sm_103a/cake_minimax_h3_varlen_attention_f71a833f43b9a3f3aa0b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k"],
            ["buffer", "v"],
            ["buffer", "q_fp4"],
            ["buffer", "k_fp4"],
            ["buffer", "q_scale"],
            ["buffer", "k_scale"],
            ["buffer", "v_fp8"],
            ["buffer", "v_amax"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "27c8e1b0fa6e88cdc539cec9423ff1c84ce6c05247f907b7a1f1b8b9554ffa8a",
        "tma_workspace_bytes": 0,
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "bf16__sm_100a": {
        "arch": "sm_100a",
        "variant": "bf16",
        "stages": ["attention"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_0df56330b21e1d259c21",
        },
    },
    "bf16__sm_103a": {
        "arch": "sm_103a",
        "variant": "bf16",
        "stages": ["attention"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_2281e14d5960c8e399ed",
        },
    },
    "nvfp4_fp4pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_c330e02790f956ad0b08",
            "attention": "cake_minimax_h3_varlen_attention_cca6dc137573738ca929",
        },
    },
    "nvfp4_fp4pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_576c36c016845b71ba04",
            "attention": "cake_minimax_h3_varlen_attention_867261f6b4079dd3db64",
        },
    },
    "nvfp4_fp8pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp8pv",
        "stages": ["quantize", "attention"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_e9349d858cf7daf87d83",
            "attention": "cake_minimax_h3_varlen_attention_1664f7cb7998b7ec20a6",
        },
    },
    "nvfp4_fp8pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp8pv",
        "stages": ["quantize", "attention"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_f71a833f43b9a3f3aa0b",
            "attention": "cake_minimax_h3_varlen_attention_c33c81e009ba5bcc8244",
        },
    },
}

VARIANTS = ("bf16", "nvfp4_fp4pv", "nvfp4_fp8pv")
STAGES = {
    "bf16": ("attention",),
    "nvfp4_fp4pv": ("quantize", "attention"),
    "nvfp4_fp8pv": ("quantize", "attention"),
}
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def route_name(variant: str, arch: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown MiniMax-H3 varlen attention variant {variant!r}")
    return f"{variant}__{arch}"


def select_route(variant: str, arch: str) -> dict[str, Any]:
    """Return the registered route record for ``variant`` on ``arch``."""
    record = ROUTES.get(route_name(variant, arch))
    if record is None:
        raise NotImplementedError(
            f"The generated MiniMax-H3 packed-varlen {variant} attention program "
            f"for {arch} is not registered in this checkout yet "
            "(see flashinfer-ai/flashinfer#4532)"
        )
    if tuple(record["stages"]) != STAGES[variant]:
        raise RuntimeError(
            f"registered route {route_name(variant, arch)!r} has stages "
            f"{tuple(record['stages'])!r}, expected {STAGES[variant]!r}"
        )
    return record


def route_available(variant: str, arch: str) -> bool:
    return route_name(variant, arch) in ROUTES


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
def gen_cake_minimax_h3_varlen_attention_module(name: str):
    record = MODULES[name]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    return gen_jit_spec(
        name=f"{name}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_minimax_h3_varlen_attention_module(name: str):
    return gen_cake_minimax_h3_varlen_attention_module(name).build_and_load()
