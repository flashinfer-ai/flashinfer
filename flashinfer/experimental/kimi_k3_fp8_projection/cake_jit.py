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

# Explicit target-owned registration of the generated programs of the Kimi-K3
# serialized ``FP8_PB_WO`` projection GEMMs (SM100 / SM103).
#
# ``MODULES`` holds one record per physical generated module (a kernel plus
# its host binding): translation units, compile flags, FFI entry, argument
# plan and closure identity.  ``KERNELS`` maps ``"<arch>"`` to the logical
# kernel key -> module assignment the host dispatcher resolves at preparation:
#
# * ``quant:u<units>``                 the per-token 1x128 E4M3 / UE8M0
#   quantization launch with ``units`` K blocks per half warp (1 for M <= 256,
#   2 / 4 for the large-M rows, chosen by ``cake_backend.quant_units``);
# * ``gemm``                            the persistent 2-CTA block-scaled
#   tcgen05 GEMM (256 output columns per CTA pair, M > 256);
# * ``decode:t<tok>_p<stages>[_fused][_res]``   the swap-AB split-K decode
#   kernel for one token-tile width, TMA pipeline depth, in-CTA quantization
#   (``_fused``) and resident token tiles (``_res``), as the measured dispatch
#   table selects per ``(N, K, M bucket)`` (M <= 256).
#
# Every module is an exact-architecture program (tcgen05 / TMEM, ``cta_group::2``
# for the GEMM).  Both literals are populated verbatim by the generated-program
# export; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_fp8_projection_051176a8b997498ebd85": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_051176a8b997498ebd85_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_051176a8b997498ebd85_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "3083fa297c1937e7b8898d68aa8696de4637e39ab826a191a00d9922db22b431",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_1828d2db2a9f93281d7c": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_1828d2db2a9f93281d7c_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_1828d2db2a9f93281d7c_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "a_q"],
            ["buffer", "a_sf"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "units_per_row"],
            ["parameter", "sf_k_sets"],
            ["parameter", "sf_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6abf1cff9cd9a7a9ff5a528aca5e24de27afd41bd4552e21af709904ccba4959",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_1e24e68aaf7e16d0a8f7": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_1e24e68aaf7e16d0a8f7_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_1e24e68aaf7e16d0a8f7_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "0db637e52663d7dd969282d1ca2481229267849ffa6360543c6cb537319b38be",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_2b0936a7df1ad289fb9b": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_2b0936a7df1ad289fb9b_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_2b0936a7df1ad289fb9b_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "a_q"],
            ["buffer", "a_sf"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "units_per_row"],
            ["parameter", "sf_k_sets"],
            ["parameter", "sf_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "7f1a41b0681efbe6556da7ca8fd868174c6569c4b9eb64fce58c53b142503037",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_308f3d8891d1a06f6a22": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_308f3d8891d1a06f6a22_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_308f3d8891d1a06f6a22_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6dfb27ffe991b52aed37267d602fbd15b311c8048ac6fe89d43252e7cc957f86",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_43c581301f63f1d4d202": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_43c581301f63f1d4d202_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_43c581301f63f1d4d202_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6e2593fa1141284da283a5b8144dd856b522550ea6de08aa3b1c40ecdf4ae981",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_44fadbba0ec9b8db66de": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_44fadbba0ec9b8db66de_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_44fadbba0ec9b8db66de_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "196f3db65bfb302a1a89222c565a14805d6f751629f13e0de13c920ef9f68900",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_477d522e4806d9867ae0": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_477d522e4806d9867ae0_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_477d522e4806d9867ae0_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "6a36d34ae4eca99affd8d6776f091444a520f59db0d8759502ee94a7f365d0df",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_4a3952d0c334f9089442": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_4a3952d0c334f9089442_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_4a3952d0c334f9089442_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "e6b0b6b50f08c19f7652b498be421c87bce9c2e9fbf200eb649c6321d4586924",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_506c67a1240bfc89102b": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_506c67a1240bfc89102b_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_506c67a1240bfc89102b_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "m_tiles"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "store_vec"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "e436a1ec915344fc7a5e481a20f9ab02d3551bb8ac5c11c59e4fe3d927df7f9f",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_532e06fab70e2220b93c": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_532e06fab70e2220b93c_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_532e06fab70e2220b93c_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "8ec920506dd5659e5ec174609dcce0eb68de0b404efbfe2f76f2aaca745cb182",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_5c73f66bbd1ff88360af": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_5c73f66bbd1ff88360af_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_5c73f66bbd1ff88360af_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "a_q"],
            ["buffer", "a_sf"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "units_per_row"],
            ["parameter", "sf_k_sets"],
            ["parameter", "sf_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "7a047898bf10662fed4d42128dac187781d9ff8cac933fc9a03ed05da77667a7",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_62f91929e21f24c1887e": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_62f91929e21f24c1887e_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_62f91929e21f24c1887e_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "dede7434973d6a2767604fb6c94ae0c5043fbec1ffedaff2d72fd1492e43e4c5",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_661e344d21eb916344b7": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_661e344d21eb916344b7_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_661e344d21eb916344b7_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "4e5d8ba713f8d9753d373cd0f328e8fe8583f01add83277569722f895ed5cbf5",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_797208c881b67b391b13": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_797208c881b67b391b13_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_797208c881b67b391b13_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "43429a7799b631a9b04acef5047f06e3c8f059d82dd4acfa1011bd95e123a2f8",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_8487e1857c1b5f183195": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_8487e1857c1b5f183195_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_8487e1857c1b5f183195_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "d727cf7b674a2eb7266dcd163e7a7fc511a298e44d434a222b65576ecbfac2b9",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_8a9abe5873ce50b3758f": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_8a9abe5873ce50b3758f_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_8a9abe5873ce50b3758f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "00a79b4b317d818068e6a38011e9299af4213de9639aa1d0bdf24523a8e18bc3",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_8c391ffb3474d5240aba": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_8c391ffb3474d5240aba_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_8c391ffb3474d5240aba_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "a_q"],
            ["buffer", "a_sf"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "units_per_row"],
            ["parameter", "sf_k_sets"],
            ["parameter", "sf_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "0177a984e5edd5d9c2a8370d818fa3f54537cf4a7bc6792b64642a72abe98ee1",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_975fd6ef843b8ee047b6": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_975fd6ef843b8ee047b6_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_975fd6ef843b8ee047b6_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "W"],
            ["tma_buffer", "X"],
            ["tma_buffer", "SFW"],
            ["tma_buffer", "SFX"],
            ["buffer", "out"],
            ["buffer", "partials"],
            ["buffer", "counters"],
            ["parameter", "M"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["parameter", "split"],
            ["parameter", "tok_per_cta"],
            ["parameter", "total_work"],
            ["parameter", "store_vec"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["tma_buffer", "XB"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "c7e2d9db7f94b87dffe6edaeb90069ca5f4899e05961b0b392d66845667e5774",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_adb0ef322619250bd1cb": {
        "arch": "sm_103a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_adb0ef322619250bd1cb_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_103a/cake_kimi_k3_fp8_projection_adb0ef322619250bd1cb_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "a_q"],
            ["buffer", "a_sf"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "units_per_row"],
            ["parameter", "sf_k_sets"],
            ["parameter", "sf_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "74e7f35156e40f508e7c2b8a3fdfb20df4ba848f62ba481a806615ca27cbd7b4",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_b5bb63c3e98bea8d714f": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_b5bb63c3e98bea8d714f_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_b5bb63c3e98bea8d714f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["buffer", "out"],
            ["parameter", "M"],
            ["parameter", "m_tiles"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "store_vec"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "0da89a75a60abbabbce1287ac33dbddb2c62c144c7727d19655309f5ed43a28f",
        "tma_workspace_bytes": 0,
    },
    "cake_kimi_k3_fp8_projection_e2390e342628e8e998a4": {
        "arch": "sm_100a",
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_e2390e342628e8e998a4_kernel.cu",
            "cake_kimi_k3_fp8_projection/sm_100a/cake_kimi_k3_fp8_projection_e2390e342628e8e998a4_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "a_q"],
            ["buffer", "a_sf"],
            ["parameter", "M"],
            ["parameter", "K"],
            ["parameter", "units_per_row"],
            ["parameter", "sf_k_sets"],
            ["parameter", "sf_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "c589eeb485fa340a7c706f72f5ad969333ecc23189c781908a572eae23a0a2b3",
        "tma_workspace_bytes": 0,
    },
}
KERNELS: dict[str, dict[str, str]] = {
    "sm_100a": {
        "decode:t128_p3": "cake_kimi_k3_fp8_projection_051176a8b997498ebd85",
        "decode:t16_p4": "cake_kimi_k3_fp8_projection_975fd6ef843b8ee047b6",
        "decode:t16_p4_fused": "cake_kimi_k3_fp8_projection_308f3d8891d1a06f6a22",
        "decode:t32_p3_fused_res": "cake_kimi_k3_fp8_projection_44fadbba0ec9b8db66de",
        "decode:t32_p4": "cake_kimi_k3_fp8_projection_1e24e68aaf7e16d0a8f7",
        "decode:t64_p2_fused_res": "cake_kimi_k3_fp8_projection_4a3952d0c334f9089442",
        "decode:t64_p4": "cake_kimi_k3_fp8_projection_477d522e4806d9867ae0",
        "gemm": "cake_kimi_k3_fp8_projection_b5bb63c3e98bea8d714f",
        "quant:u1": "cake_kimi_k3_fp8_projection_e2390e342628e8e998a4",
        "quant:u2": "cake_kimi_k3_fp8_projection_5c73f66bbd1ff88360af",
        "quant:u4": "cake_kimi_k3_fp8_projection_2b0936a7df1ad289fb9b",
    },
    "sm_103a": {
        "decode:t128_p3": "cake_kimi_k3_fp8_projection_8a9abe5873ce50b3758f",
        "decode:t16_p4": "cake_kimi_k3_fp8_projection_8487e1857c1b5f183195",
        "decode:t16_p4_fused": "cake_kimi_k3_fp8_projection_532e06fab70e2220b93c",
        "decode:t32_p3_fused_res": "cake_kimi_k3_fp8_projection_62f91929e21f24c1887e",
        "decode:t32_p4": "cake_kimi_k3_fp8_projection_43c581301f63f1d4d202",
        "decode:t64_p2_fused_res": "cake_kimi_k3_fp8_projection_661e344d21eb916344b7",
        "decode:t64_p4": "cake_kimi_k3_fp8_projection_797208c881b67b391b13",
        "gemm": "cake_kimi_k3_fp8_projection_506c67a1240bfc89102b",
        "quant:u1": "cake_kimi_k3_fp8_projection_8c391ffb3474d5240aba",
        "quant:u2": "cake_kimi_k3_fp8_projection_adb0ef322619250bd1cb",
        "quant:u4": "cake_kimi_k3_fp8_projection_1828d2db2a9f93281d7c",
    },
}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}

GEMM_KERNEL_KEY = "gemm"


def quant_kernel_key(units: int) -> str:
    return f"quant:u{int(units)}"


def decode_kernel_key(tok: int, stages: int, fused: bool, resident: bool) -> str:
    key = f"decode:t{int(tok)}_p{int(stages)}"
    if fused:
        key += "_fused"
    if resident:
        key += "_res"
    return key


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when ``arch`` is registered and carries every key in ``required_keys``."""
    table = KERNELS.get(arch)
    return table is not None and all(key in table for key in required_keys)


def kernel_module_name(arch: str, key: str) -> str:
    """Return the registered physical module for ``key`` on ``arch``."""
    table = KERNELS.get(arch)
    if table is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 FP8 projection programs for {arch} are not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#4568)"
        )
    name = table.get(key)
    if name is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 FP8 projection kernel {key!r} for {arch} is not "
            "registered in this checkout (see flashinfer-ai/flashinfer#4568)"
        )
    record = MODULES[name]
    if record["arch"] != arch:
        raise RuntimeError(
            f"registered module {name!r} is an {record['arch']} program bound to {arch}"
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
def gen_cake_kimi_k3_fp8_projection_module(name: str):
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
def load_cake_kimi_k3_fp8_projection_module(name: str):
    return gen_cake_kimi_k3_fp8_projection_module(name).build_and_load()
