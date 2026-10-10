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
    "cake_minimax_h3_varlen_attention_01e6cf096b287e594899": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_01e6cf096b287e594899_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_01e6cf096b287e594899_binding.cu",
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
            "sm_100a": "8c83e336653ee1ad7e1e1cf511b8eca103105292cb8f32cf496493cd9f846f4f",
            "sm_103a": "c9e88c7d55e736c4cad208e93e7611ea2936b1a1e46b22b28b36a29bd63c9f93",
        },
    },
    "cake_minimax_h3_varlen_attention_4b8d9db1ca486cd41383": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_4b8d9db1ca486cd41383_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_4b8d9db1ca486cd41383_binding.cu",
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
            "sm_100a": "3c3f5863bb8bb4d0f01a1a842afbad99a4be5914c7d5915cd0126e210e7a2809",
            "sm_103a": "bf9a78b305dd3152f37aadc6a4be54e9d87c876ee8e48dae71a03ae8502bdd0b",
        },
    },
    "cake_minimax_h3_varlen_attention_571c1ac46e3fe6d8842d": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_571c1ac46e3fe6d8842d_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_571c1ac46e3fe6d8842d_binding.cu",
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
            "sm_100a": "abd39baa0afe00d5ddbb9752a8314fd30ea9d8d72c8d1b4e8fe4f9273d16adbd",
            "sm_103a": "c06597a8c68f9c5750f341b879188b2e2ab7bb1c767aafbb48c67542de7d77df",
        },
    },
    "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_579b1fc2756eec97f194_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_579b1fc2756eec97f194_binding.cu",
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
            "sm_100a": "e729c4593edec16a979ade9343c74d5bc9ae78c2dcc9ce8b29783fb5b37eff25",
            "sm_103a": "5f4a057726f7ee87ff8eec114dfa03f126c4e243e2eeea01bbf41089da6d5af5",
        },
    },
    "cake_minimax_h3_varlen_attention_660bd18545042b7ed2e1": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_660bd18545042b7ed2e1_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_660bd18545042b7ed2e1_binding.cu",
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
            "sm_100a": "1ff1b0d3eaccbd42891f463069867b465ea3d29e5dbab251660b33f4043625ef",
            "sm_103a": "4da55f5f819b1999f62be18cdbd682de8951ea1a82d6136b41aa0ae8467bf588",
        },
    },
    "cake_minimax_h3_varlen_attention_a0106039a66d9a80b786": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_a0106039a66d9a80b786_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_a0106039a66d9a80b786_binding.cu",
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
            "sm_100a": "b31bc7104955486ff395125f4e252f4801d14f0775d3849c03f28b3bca7f3cd5",
            "sm_103a": "029683984a07debbc51c24e2589cdab89d855f817a38fc5b69855c9ae130d04a",
        },
    },
    "cake_minimax_h3_varlen_attention_ccba90ddf663625098b1": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_ccba90ddf663625098b1_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_ccba90ddf663625098b1_binding.cu",
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
            "sm_100a": "b3e69d79a9f4f05c528a67d6ed3e2689b5114290ca4e88516c10e0fa48188fdc",
            "sm_103a": "dd35e411f44b5fbbf1c89f2cf09a16f82ca78170d5650e306e83953271899f61",
        },
    },
    "cake_minimax_h3_varlen_attention_cd9b0982c132645f17d1": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_cd9b0982c132645f17d1_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_cd9b0982c132645f17d1_binding.cu",
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
            "sm_100a": "3c84cfa1ee35251cffe7f8695d3795b8921964bf2fa8897d869269fa9e0ac519",
            "sm_103a": "0893acbff67a2c6d92bdfe676df10324e631426e2edf1263ab14b1a90e81da7c",
        },
    },
    "cake_minimax_h3_varlen_attention_ee96b07d39fc362d3fe5": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_ee96b07d39fc362d3fe5_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_ee96b07d39fc362d3fe5_binding.cu",
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
            "sm_100a": "5f577d5ea5e8cbfa2c3c988b4a704c16429c5ab86821f2ede332ca4169c3ebda",
            "sm_103a": "614453c2a4f50cf135bb31ecfb862c8ff5d5ac5974544a7f75b4b7df1893fc35",
        },
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "bf16__sm_100a": {
        "arch": "sm_100a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_ee96b07d39fc362d3fe5",
            "combine": "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194",
        },
    },
    "bf16__sm_103a": {
        "arch": "sm_103a",
        "variant": "bf16",
        "stages": ["attention", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_ee96b07d39fc362d3fe5",
            "combine": "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194",
        },
    },
    "nvfp4_fp4pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_cd9b0982c132645f17d1",
            "attention": "cake_minimax_h3_varlen_attention_01e6cf096b287e594899",
            "attention_split": "cake_minimax_h3_varlen_attention_660bd18545042b7ed2e1",
            "combine": "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194",
        },
    },
    "nvfp4_fp4pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp4pv",
        "stages": ["quantize", "attention", "attention_split", "combine"],
        "modules": {
            "quantize": "cake_minimax_h3_varlen_attention_cd9b0982c132645f17d1",
            "attention": "cake_minimax_h3_varlen_attention_01e6cf096b287e594899",
            "attention_split": "cake_minimax_h3_varlen_attention_660bd18545042b7ed2e1",
            "combine": "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194",
        },
    },
    "nvfp4_fp8pv__sm_100a": {
        "arch": "sm_100a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_571c1ac46e3fe6d8842d",
            "quantize": "cake_minimax_h3_varlen_attention_ccba90ddf663625098b1",
            "attention": "cake_minimax_h3_varlen_attention_4b8d9db1ca486cd41383",
            "attention_split": "cake_minimax_h3_varlen_attention_a0106039a66d9a80b786",
            "combine": "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194",
        },
    },
    "nvfp4_fp8pv__sm_103a": {
        "arch": "sm_103a",
        "variant": "nvfp4_fp8pv",
        "stages": ["amax", "quantize", "attention", "attention_split", "combine"],
        "modules": {
            "amax": "cake_minimax_h3_varlen_attention_571c1ac46e3fe6d8842d",
            "quantize": "cake_minimax_h3_varlen_attention_ccba90ddf663625098b1",
            "attention": "cake_minimax_h3_varlen_attention_4b8d9db1ca486cd41383",
            "attention_split": "cake_minimax_h3_varlen_attention_a0106039a66d9a80b786",
            "combine": "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194",
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
