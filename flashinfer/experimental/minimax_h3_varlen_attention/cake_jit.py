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
# * ``bf16``          stages ``("attention", "attention_dyn", "attention_sread", "combine")``
#                     on sm_100a and ``("attention", "attention_dyn", "combine")`` on sm_103a
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
# the programmatic dependent launch attribute; the BF16 route carries three
# bitwise-identical attention programs selected per plan by the host planner
# (``cake_backend.build_bf16_segment_plan``): ``attention`` (the static
# schedule), ``attention_dyn`` (the dynamic whole-unit ticket scheduler of the
# long multi-wave plans) and, on ``BF16_SREAD_ARCHES`` only, ``attention_sread``
# (the publish-tail cut of the remaining plans); ``combine`` is the shared
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
            "sm_100a": "c44ed454ac7baeb18aba978ce11a468c45401acf273b481ecee53ef68053c9f2",
            "sm_103a": "91d4a124e1c3efc8e845c2f6ec93615517d9165770d00a451860a2eb83c20a28",
        },
    },
    "cake_minimax_h3_varlen_attention_08e428db13e7c3896d64": {
        "arches": ["sm_100a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_08e428db13e7c3896d64_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_08e428db13e7c3896d64_binding.cu",
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
            "sm_100a": "0d0898df7073da04ae63d270a4118f25a3057e6d35d973606f4ad13bf917167f",
        },
    },
    "cake_minimax_h3_varlen_attention_1350b7d87b8bb2dc9c46": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_1350b7d87b8bb2dc9c46_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_1350b7d87b8bb2dc9c46_binding.cu",
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
            "sm_100a": "c110396f0072bdefdafea3c20787d9f8235df7b2296654b6828bd9b2a63a2da1",
            "sm_103a": "8163c27a8faf1136b10e365321a36845fe04ab623b9099e223194a99fdbe2e57",
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
            "sm_100a": "f6474a8b636bea0a652822fe24bc49f9e2607ffa03d95f84650678101a741d47",
            "sm_103a": "32b00613ab7c158e4b203836b9b2ac4715f46ee52d56443fb44a9e96286feaf7",
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
            "sm_100a": "392eb57d169f6b6d2625ad7ec6fa7b3133582cd0cc8712b72ec0c6b00d10c070",
            "sm_103a": "2b66442c37e47563d1fdcad9bb864209ac3adaf90191157c821fe91deb0b3e04",
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
            "sm_100a": "803ba298e93e90880a693406d8e9f86ee7bc0a281a469e97e46c13b88aca4afa",
            "sm_103a": "6c9355cdeea1fb6824e89ea90adf4776e6f4d445e9fcfb212656689660c2d960",
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
            "sm_100a": "d002516a0825cc26cf0cb7e1f2d9a727cd5ec231a268c2401bcbe43246b9ef57",
            "sm_103a": "a12402fd78c910fda087d75cd0720382f66abe243a71dfcabd319d9db23ce3a8",
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
            "sm_100a": "fb4721104c044955e6ba990546c19ee671d85345711e47e42420602eac5b35e6",
            "sm_103a": "632fa6d57eeb8619e444e07d3c3c85a685e81a56ef361a7ae548a1392a8dd575",
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
            "sm_100a": "271e14bdbe5bad5caaca4847a34e421c7bfd9e9c5fd54b2ae326325de86560e1",
            "sm_103a": "7473fd6cd22069bb035c7e0b476a5be3ba90019634f6370ed7680341742b1dd8",
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
            "sm_100a": "def5c9688aa9a26e1e3c8e732e219664495d9a420aa1ec2622f7381658eeb608",
            "sm_103a": "033b1bc990e878eb2162996cbd6ee547162a46354f0f460ecb7398fa7aabe034",
        },
    },
    "cake_minimax_h3_varlen_attention_dcc6cea1370b3bfaec30": {
        "arches": ["sm_100a", "sm_103a"],
        "role": "kernel",
        "sources": [
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_dcc6cea1370b3bfaec30_kernel.cu",
            "cake_minimax_h3_varlen_attention/cake_minimax_h3_varlen_attention_dcc6cea1370b3bfaec30_binding.cu",
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
            "sm_100a": "9016b513e9c6a9fa213ccd64dab904a7ced02a96e40fd892010cc1e99d4e0255",
            "sm_103a": "1e57f1ee2d9bdfa5b0ea1ab99c720fb9beeb84702ad1040ef9f3c30cf79905c4",
        },
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "bf16__sm_100a": {
        "arch": "sm_100a",
        "variant": "bf16",
        "stages": ["attention", "attention_dyn", "attention_sread", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_dcc6cea1370b3bfaec30",
            "attention_dyn": "cake_minimax_h3_varlen_attention_1350b7d87b8bb2dc9c46",
            "attention_sread": "cake_minimax_h3_varlen_attention_08e428db13e7c3896d64",
            "combine": "cake_minimax_h3_varlen_attention_579b1fc2756eec97f194",
        },
    },
    "bf16__sm_103a": {
        "arch": "sm_103a",
        "variant": "bf16",
        "stages": ["attention", "attention_dyn", "combine"],
        "modules": {
            "attention": "cake_minimax_h3_varlen_attention_dcc6cea1370b3bfaec30",
            "attention_dyn": "cake_minimax_h3_varlen_attention_1350b7d87b8bb2dc9c46",
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
    "bf16": ("attention", "attention_dyn", "attention_sread", "combine"),
    "nvfp4_fp4pv": ("quantize", "attention", "attention_split", "combine"),
    "nvfp4_fp8pv": ("amax", "quantize", "attention", "attention_split", "combine"),
}
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
# Architectures whose BF16 route carries the ``attention_sread`` program (the
# Cake planner's ``SREAD_UNIFIED_DISPATCH_ARCHS``); the other routes bind the
# static program for the plans that are not dynamically scheduled.
BF16_SREAD_ARCHES = ("sm_100a",)


def stages_for(variant: str, arch: str) -> tuple[str, ...]:
    """The ordered stages of ``variant`` on ``arch``."""
    stages = STAGES[variant]
    if variant == "bf16" and arch not in BF16_SREAD_ARCHES:
        stages = tuple(s for s in stages if s != "attention_sread")
    return stages


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
    if tuple(record["stages"]) != stages_for(variant, arch):
        raise RuntimeError(
            f"registered route {route_name(variant, arch)!r} has stages "
            f"{tuple(record['stages'])!r}, expected {stages_for(variant, arch)!r}"
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
