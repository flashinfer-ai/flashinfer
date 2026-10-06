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
# ``MODULES`` holds one record per physical generated program (one kernel
# source plus its host binding, shared by every architecture in ``arches``):
# translation units, compile flags, FFI entry, argument plan and closure
# identity.  Architecture-specific lowering (the SM103 ``tcgen05.ld.red``
# score drain against the SM100 packed-polynomial exp2 emulation, the BF16
# per-architecture schedule) lives inside the one source under exact
# ``__CUDA_ARCH__`` regions, so a program is compiled once per exact target
# (``sm100a_nvcc_flags`` / ``sm103a_nvcc_flags``) and the JIT caches one
# library per (program, arch).  ``ROUTES`` maps ``"<variant>__<arch>"`` to the
# ordered stage -> program assignment of that program family:
#
# * ``bf16``          stages ``("attention", "combine")``
# * ``nvfp4_fp4pv``   stages ``("quantize", "attention", "attention_split", "combine")``
# * ``nvfp4_fp8pv``   stages ``("amax", "quantize", "attention", "attention_split", "combine")``
#
# ``quantize`` is one fused single-launch quantizer per PV mode
# (``minimax_h3_varlen_nvfp4_quantize_qkv`` / ``..._quantize_qk_fp8v``); the
# fp8 route precedes it with ``amax`` (``minimax_h3_varlen_v_amax_partial``,
# one partial ``max|V|`` per CTA, folded by the quantizer, which is that
# kernel's programmatic dependent launch); the NVFP4 routes carry two
# attention programs, the dense ``attention`` (no K/V-split code; bound for
# plans without split units) and ``attention_split`` (reads the unit's K/V
# block range and writes partial rows; bound when the plan has split units)
# -- the runner binds exactly one of them per plan -- and both bindings carry
# the programmatic dependent launch attribute; ``combine`` is the shared
# K/V-split merge kernel (``minimax_h3_varlen_split_combine``) that finishes
# the units the host planner split over their K/V range (skipped by the
# runner when a plan has no split units).  Both literals are populated
# verbatim by the generated-program export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_minimax_h3_varlen_attention_08efd015f99ac63b11df": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_08efd015f99ac63b11df_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_08efd015f99ac63b11df_binding.cu",
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
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "0bf03ea23d1a255f58296fd338f1422bcaa0197f7ec19952bb30b7efba5fab3d",
            "sm_103a": "efa021e8e01e0ccfc6649a6bc821768e484474f2a3b2540b1688a121669002cb",
        },
    },
    "cake_minimax_h3_varlen_attention_14cc208e0531054e28fc": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_14cc208e0531054e28fc_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_14cc208e0531054e28fc_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "1b1822451a3332fb6784c0bd20df2a0932148b02c63c22bcad2750eac3883783",
            "sm_103a": "8c1821e65e84d351798d96cd92f086722bb51405a073d5878ec7157b1dbfe5f7",
        },
    },
    "cake_minimax_h3_varlen_attention_4cfed868802f277c0494": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_4cfed868802f277c0494_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_4cfed868802f277c0494_binding.cu",
        ],
        "compile_flags": ["--use_fast_math", "--ptxas-options=--opt-level=1"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "Q"],
            ["tma_buffer", "K"],
            ["tma_buffer", "V"],
            ["tma_buffer", "O"],
            ["buffer", "O_raw"],
            ["buffer", "unit_table"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "total_tiles"],
            ["parameter", "num_heads"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "4d0b9c545a1b8ff090afb47e452d5c017b6f8fa204838b27c3cdae26e9c6e8f4",
            "sm_103a": "ee5646d48f4b89b53b34f86195e3aa5def82359fcbf01a61517ddb76b5e37814",
        },
    },
    "cake_minimax_h3_varlen_attention_998a9624b757b0a48881": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_998a9624b757b0a48881_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_998a9624b757b0a48881_binding.cu",
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
            ["buffer", "cl_kv_begin"],
            ["buffer", "cl_kv_blocks"],
            ["buffer", "cl_ws_slot"],
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["parameter", "num_tiles"],
            ["parameter", "total_clusters"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "19604b64527baf31a95957e6f9398d4efbb5c08b494164f761a47a705367337c",
            "sm_103a": "444c3c0e4543b89d775d42f449d783270b641871f7cb464e1b2ac45758d7aa7c",
        },
    },
    "cake_minimax_h3_varlen_attention_d0676c68ea89faded6b6": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d0676c68ea89faded6b6_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d0676c68ea89faded6b6_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "c7fc5cda7db4151350e9a1e6822b295572fce4bd8b74720cefc0a4f87c6bcdfc",
            "sm_103a": "b18003f835d2314df41541517f54a52702b9c1380419f5fbc5053df28de65c2a",
        },
    },
    "cake_minimax_h3_varlen_attention_d18ed0a702ce625d75a8": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d18ed0a702ce625d75a8_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d18ed0a702ce625d75a8_binding.cu",
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
        "closure_sha256": {
            "sm_100a": "7781593e72bf188beabf3d16cb935363f7734f36aae25e47ba8049e449a0db0f",
            "sm_103a": "68e5757439f7a35bda1de8f4e0440960702a641589ffc525c8e5644222335023",
        },
    },
    "cake_minimax_h3_varlen_attention_d685cb8f9d9ceebf67f6": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d685cb8f9d9ceebf67f6_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_d685cb8f9d9ceebf67f6_binding.cu",
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
            ["buffer", "v_amax_partial"],
            ["buffer", "block_token"],
            ["buffer", "block_valid"],
            ["parameter", "heads"],
            ["parameter", "PB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "7335a0bc7db0f1b533a091442ef4c9513fd24c98e4927f187669521930841aa6",
            "sm_103a": "c2af3dc62a7af657bca199ee2cd0e66e85f87a3dffc404bb6b5a514a6ab773c6",
        },
    },
    "cake_minimax_h3_varlen_attention_fbf0a620ff74617e2a6a": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_fbf0a620ff74617e2a6a_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_fbf0a620ff74617e2a6a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "v"],
            ["buffer", "v_amax_partial"],
            ["parameter", "num_vectors"],
            ["parameter", "num_chunks"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "b4b8cb3de1cd2041c9403bbd3b9905f836ce318d1c88c3bcbe65534a7e9e4f65",
            "sm_103a": "c97556c789b8e281b4ac564ff02b2bac3c6c1129392879bf36a191b80eb525af",
        },
    },
    "cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_ML"],
            ["buffer", "combine_table"],
            ["buffer", "seg_begin"],
            ["buffer", "seg_len"],
            ["buffer", "O"],
            ["parameter", "num_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": {
            "sm_100a": "53d2bb7afbc03004717aeda331e5180378f588b9aef52472d17f546ff6c525e2",
            "sm_103a": "6386589c7168890cd3b90958f177e36e9d7027e53ed22ff438d412d6e696d64e",
        },
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "bf16__sm_100a": {
        "arch": "sm_100a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_4cfed868802f277c0494",
            "combine": "cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e",
        },
    },
    "bf16__sm_103a": {
        "arch": "sm_103a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_4cfed868802f277c0494",
            "combine": "cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e",
        },
    },
    "nvfp4_fp4pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_d18ed0a702ce625d75a8",
            "attention": "cake_minimax_h3_varlen_attention_d0676c68ea89faded6b6",
            "attention_split": "cake_minimax_h3_varlen_attention_08efd015f99ac63b11df",
            "combine": "cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e",
        },
    },
    "nvfp4_fp4pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_d18ed0a702ce625d75a8",
            "attention": "cake_minimax_h3_varlen_attention_d0676c68ea89faded6b6",
            "attention_split": "cake_minimax_h3_varlen_attention_08efd015f99ac63b11df",
            "combine": "cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e",
        },
    },
    "nvfp4_fp8pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_fbf0a620ff74617e2a6a",
            "quantize": "cake_minimax_h3_varlen_attention_d685cb8f9d9ceebf67f6",
            "attention": "cake_minimax_h3_varlen_attention_14cc208e0531054e28fc",
            "attention_split": "cake_minimax_h3_varlen_attention_998a9624b757b0a48881",
            "combine": "cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e",
        },
    },
    "nvfp4_fp8pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_fbf0a620ff74617e2a6a",
            "quantize": "cake_minimax_h3_varlen_attention_d685cb8f9d9ceebf67f6",
            "attention": "cake_minimax_h3_varlen_attention_14cc208e0531054e28fc",
            "attention_split": "cake_minimax_h3_varlen_attention_998a9624b757b0a48881",
            "combine": "cake_minimax_h3_varlen_attention_ff9654b6e96277c07d5e",
        },
    },
}

VARIANTS = ("bf16", "nvfp4_fp4pv", "nvfp4_fp8pv")
STAGES = {
    "bf16": ("attention", "combine"),
    "nvfp4_fp4pv": ("quantize", "attention", "attention_split", "combine"),
    "nvfp4_fp8pv": ("amax", "quantize", "attention", "attention_split", "combine"),
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
    for stage, module in record["modules"].items():
        if arch not in MODULES[module]["arches"]:
            raise RuntimeError(
                f"route {route_name(variant, arch)!r} stage {stage!r} names program "
                f"{module!r}, which is not built for {arch}"
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
def gen_cake_minimax_h3_varlen_attention_module(name: str, arch: str):
    """JIT spec of program ``name`` compiled for the exact target ``arch``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(
            f"program {name!r} is not built for {arch!r}: {record['arches']}"
        )
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    # ``closure_sha256[arch]`` is the export's module receipt identity for this target.
    return gen_jit_spec(
        name=f"{name}_{arch}_" + record["closure_sha256"][arch][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_minimax_h3_varlen_attention_module(name: str, arch: str):
    return gen_cake_minimax_h3_varlen_attention_module(name, arch).build_and_load()
