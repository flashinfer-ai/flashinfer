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

# Registration of the generated programs of the Kimi-K3 serialized ``FP8_PB_WO``
# projection GEMMs (SM100 / SM103).
#
# ``MODULES`` holds one record per physical program (a kernel source plus its
# host launcher): the two translation units under ``csrc/``, the architectures
# the source compiles for, compile flags, FFI entry, argument plan and the
# closure identity (SHA-256 over the translation units and the shared device /
# host headers they include).  One source serves every listed architecture;
# the architecture-specific lowering lines sit behind ``__CUDA_ARCH__`` guards
# inside the source, and the loader compiles it with the exact flag set of the
# device it runs on.
#
# ``KERNELS`` maps the logical kernel key the host dispatcher resolves at
# preparation to its program and the compile-line ``defines`` of that
# instantiation (``-DNAME=value``):
#
# * ``quant:u<units>``                 the per-token 1x128 E4M3 / UE8M0
#   quantization launch; one program, ``units`` K blocks per half warp is the
#   ``QUANT_UNITS`` define (1 for M <= 256, 2 / 4 for the large-M rows, chosen
#   by ``cake_backend.quant_units``);
# * ``gemm`` / ``gemm_rstaged`` / ``gemm_tstore``   the persistent 2-CTA
#   block-scaled tcgen05 GEMM (256 output columns per CTA pair, M > 256) with
#   the register, staged-register or TMA-store epilogue;
# * ``decode:t<tok>_p<stages>[_fused][_res][_r<xb>][_q<lanes>][_cs<C>]``   the
#   swap-AB split-K decode kernel for one token-tile width, TMA pipeline
#   depth, in-CTA quantization (``_fused``), resident token tiles (``_res``),
#   a decoupled ``xb``-deep BF16 token ring (``_r``), narrow quantization
#   units of ``lanes`` lanes (``_q``; absent = half-warp units) and a
#   ``C``-CTA cluster split-K exchange (``_cs``), as the measured dispatch
#   table selects per ``(N, K, M bucket)``.
#
# Both literals are populated by the generated-program export; do not edit
# them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_fp8_projection_3a1c6f876002d18348e1": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_3a1c6f876002d18348e1_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_3a1c6f876002d18348e1_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "d73db36c8f35f4b7824a57b05fabe9c7511bbf270fb70623e66507f6823451a4",
    },
    "cake_kimi_k3_fp8_projection_3dd4e1661d410d5aa058": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_3dd4e1661d410d5aa058_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_3dd4e1661d410d5aa058_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "d7db084c0906972440c574412cf75444cd86104a8819b219b9a0964c62618eaf",
    },
    "cake_kimi_k3_fp8_projection_3e01c578aa78b25be948": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_3e01c578aa78b25be948_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_3e01c578aa78b25be948_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "ac2a3a5895b7ccfe351079ef46f9f9d2523e1fa2e81e8f2df461d90206da6b6b",
    },
    "cake_kimi_k3_fp8_projection_4d3a62cc50bfed1ea087": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_4d3a62cc50bfed1ea087_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_4d3a62cc50bfed1ea087_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["buffer", "out"],
            ["tma_buffer", "OUT"],
            ["parameter", "M"],
            ["parameter", "m_tiles"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "store_vec"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "efbe9d85360d883d8255782404edf717df98b22b53087cd365bb86d125ea15a2",
    },
    "cake_kimi_k3_fp8_projection_4dd5f78f40291f4e92ba": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_4dd5f78f40291f4e92ba_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_4dd5f78f40291f4e92ba_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "d3a68b5118dfc111b9944664b45ae63d4502b6d715c6e0b8b6a86d3358723b3c",
    },
    "cake_kimi_k3_fp8_projection_5d7e5d80d3e6d87d8fcf": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_5d7e5d80d3e6d87d8fcf_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_5d7e5d80d3e6d87d8fcf_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "12f487128b762ab278c0a608c653fe7c28aac182ddf08126f1fd6a5f9f4b73f8",
    },
    "cake_kimi_k3_fp8_projection_603ff3671178b2df6a72": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_603ff3671178b2df6a72_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_603ff3671178b2df6a72_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "021782d2337d6b39dfa6084d7c28dfc9e1ffbb90fc497858d0b35f09c0cf12f1",
    },
    "cake_kimi_k3_fp8_projection_72dfec1fcac0e3ff00e1": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_72dfec1fcac0e3ff00e1_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_72dfec1fcac0e3ff00e1_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "55b3c049fe03762214b61dabc8d84773305ef267aae95ea3f556ab929b6c71e4",
    },
    "cake_kimi_k3_fp8_projection_7f685e504e1332af920b": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_7f685e504e1332af920b_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_7f685e504e1332af920b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["buffer", "out"],
            ["tma_buffer", "OUT"],
            ["parameter", "M"],
            ["parameter", "m_tiles"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "store_vec"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "4686f74dd4d3374712f7e54921a53ee38365fa5da5b01da77a72b16c74993298",
    },
    "cake_kimi_k3_fp8_projection_8dd9f65d19a59ab10ade": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_8dd9f65d19a59ab10ade_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_8dd9f65d19a59ab10ade_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "268e6d48b76a20384d6bd7a1232e94974da81b8fa4d821ce1018a8bfad3c3090",
    },
    "cake_kimi_k3_fp8_projection_9de5e1067ddd78594579": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_9de5e1067ddd78594579_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_9de5e1067ddd78594579_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "f72bfc7a34a1628148a75b6201398f29455ac05eadc1ae122b85cdebb53149bc",
    },
    "cake_kimi_k3_fp8_projection_a7db8da23a75c236a86f": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_a7db8da23a75c236a86f_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_a7db8da23a75c236a86f_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "00d80023a76b5ac3bde2eaba28d6e9a045da835527cf62cc8ad9cad5329e950c",
    },
    "cake_kimi_k3_fp8_projection_a93fb42ab501b9687989": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_a93fb42ab501b9687989_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_a93fb42ab501b9687989_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "fedf899fb95f9138fa66df9bb2768fc8d0eb35e2a24d6d277206d3912902bfbd",
    },
    "cake_kimi_k3_fp8_projection_af01a031e88c4f4a6dd1": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_af01a031e88c4f4a6dd1_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_af01a031e88c4f4a6dd1_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "b71a076cc97a398c66a740f31f093a790cb155696725f0df7f0820a05c927fde",
    },
    "cake_kimi_k3_fp8_projection_cb26b23aa6b8a479fefe": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_cb26b23aa6b8a479fefe_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_cb26b23aa6b8a479fefe_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "0cec06b092e4aa87a053d26b69f24abe837046ae649556391dc44ae98e1d23b6",
    },
    "cake_kimi_k3_fp8_projection_d9f9b8fe3cca45d25b8b": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_d9f9b8fe3cca45d25b8b_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_d9f9b8fe3cca45d25b8b_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "170192b396886985312bb885e6b1f1ba296fabb6186523ac80a2f75e7fa69ca8",
    },
    "cake_kimi_k3_fp8_projection_e6960e4014976bd37cd5": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_e6960e4014976bd37cd5_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_e6960e4014976bd37cd5_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["buffer", "out"],
            ["tma_buffer", "OUT"],
            ["parameter", "M"],
            ["parameter", "m_tiles"],
            ["parameter", "n_tiles"],
            ["parameter", "n_valid"],
            ["parameter", "ldo"],
            ["parameter", "store_vec"],
            ["parameter", "num_k_iters"],
            ["parameter", "sf_k_tiles"],
            ["buffer", "x"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "9498669442f37f4a52dd76b06c3d508f30ada3de54aaf9edd9a85677271e9900",
    },
    "cake_kimi_k3_fp8_projection_e89394df4fef8c5b6500": {
        "role": "kernel",
        "sources": [
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_e89394df4fef8c5b6500_kernel.cu",
            "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_e89394df4fef8c5b6500_binding.cu",
        ],
        "arches": ["sm_100a", "sm_103a"],
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
        "closure_sha256": "120dc1db5b3610214766382a5543eff0878f8f1745fd9c538a0c6c0209729b3c",
    },
}
KERNELS: dict[str, dict[str, Any]] = {
    "decode:t128_p3": {
        "module": "cake_kimi_k3_fp8_projection_d9f9b8fe3cca45d25b8b",
        "defines": {},
    },
    "decode:t16_p4": {
        "module": "cake_kimi_k3_fp8_projection_8dd9f65d19a59ab10ade",
        "defines": {},
    },
    "decode:t16_p4_fused": {
        "module": "cake_kimi_k3_fp8_projection_cb26b23aa6b8a479fefe",
        "defines": {},
    },
    "decode:t16_p4_fused_cs2": {
        "module": "cake_kimi_k3_fp8_projection_5d7e5d80d3e6d87d8fcf",
        "defines": {},
    },
    "decode:t16_p4_fused_cs4": {
        "module": "cake_kimi_k3_fp8_projection_af01a031e88c4f4a6dd1",
        "defines": {},
    },
    "decode:t32_p3_fused_cs8": {
        "module": "cake_kimi_k3_fp8_projection_3a1c6f876002d18348e1",
        "defines": {},
    },
    "decode:t32_p3_fused_r5_q4": {
        "module": "cake_kimi_k3_fp8_projection_3dd4e1661d410d5aa058",
        "defines": {},
    },
    "decode:t32_p3_fused_res": {
        "module": "cake_kimi_k3_fp8_projection_4dd5f78f40291f4e92ba",
        "defines": {},
    },
    "decode:t32_p4": {
        "module": "cake_kimi_k3_fp8_projection_603ff3671178b2df6a72",
        "defines": {},
    },
    "decode:t32_p4_cs4": {
        "module": "cake_kimi_k3_fp8_projection_9de5e1067ddd78594579",
        "defines": {},
    },
    "decode:t32_p4_cs8": {
        "module": "cake_kimi_k3_fp8_projection_a93fb42ab501b9687989",
        "defines": {},
    },
    "decode:t64_p2_fused_r3_q4": {
        "module": "cake_kimi_k3_fp8_projection_e89394df4fef8c5b6500",
        "defines": {},
    },
    "decode:t64_p2_fused_res": {
        "module": "cake_kimi_k3_fp8_projection_3e01c578aa78b25be948",
        "defines": {},
    },
    "decode:t64_p4": {
        "module": "cake_kimi_k3_fp8_projection_72dfec1fcac0e3ff00e1",
        "defines": {},
    },
    "gemm": {
        "module": "cake_kimi_k3_fp8_projection_4d3a62cc50bfed1ea087",
        "defines": {},
    },
    "gemm_rstaged": {
        "module": "cake_kimi_k3_fp8_projection_e6960e4014976bd37cd5",
        "defines": {},
    },
    "gemm_tstore": {
        "module": "cake_kimi_k3_fp8_projection_7f685e504e1332af920b",
        "defines": {},
    },
    "quant:u1": {
        "module": "cake_kimi_k3_fp8_projection_a7db8da23a75c236a86f",
        "defines": {"QUANT_UNITS": 1},
    },
    "quant:u2": {
        "module": "cake_kimi_k3_fp8_projection_a7db8da23a75c236a86f",
        "defines": {"QUANT_UNITS": 2},
    },
    "quant:u4": {
        "module": "cake_kimi_k3_fp8_projection_a7db8da23a75c236a86f",
        "defines": {"QUANT_UNITS": 4},
    },
}

ARCHES = ("sm_100a", "sm_103a")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}

GEMM_KERNEL_KEY = "gemm"  # register epilogue (any even output row stride)
GEMM_RSTAGED_KERNEL_KEY = "gemm_rstaged"  # staged row-coalesced register epilogue (8-byte aligned output rows)
GEMM_TSTORE_KERNEL_KEY = (
    "gemm_tstore"  # TMA-store epilogue (16-byte output base, row stride, column edge)
)

Defines = tuple[tuple[str, int], ...]


def quant_kernel_key(units: int) -> str:
    return f"quant:u{int(units)}"


def decode_kernel_key(
    tok: int,
    stages: int,
    fused: bool,
    resident: bool,
    xb_stages: int = 0,
    qlanes: int = 16,
    csplit: int = 1,
    cs_alias: bool = False,
) -> str:
    key = f"decode:t{int(tok)}_p{int(stages)}"
    if fused:
        key += "_fused"
    if resident:
        key += "_res"
    if xb_stages:
        key += f"_r{int(xb_stages)}"
    if qlanes != 16:
        key += f"_q{int(qlanes)}"
    if int(csplit) > 1:
        key += f"_cs{int(csplit)}"  # K split across the CTAs of one cluster, DSM partial exchange
        if cs_alias:
            key += "a"  # the exchange inbox aliases the dead pipeline stages (one round; one work item per CTA)
    return key


def kernel_program(arch: str, key: str) -> tuple[str, Defines]:
    """``(program, compile-line defines)`` of logical kernel ``key`` on ``arch``."""
    entry = KERNELS.get(key)
    if entry is None:
        raise NotImplementedError(
            f"The generated Kimi-K3 FP8 projection kernel {key!r} is not registered "
            "in this checkout (see flashinfer-ai/flashinfer#4568)"
        )
    name = entry["module"]
    if arch not in MODULES[name]["arches"]:
        raise NotImplementedError(
            f"The generated Kimi-K3 FP8 projection program of {key!r} is not built "
            f"for {arch} (registered: {MODULES[name]['arches']})"
        )
    return name, tuple(sorted((str(k), int(v)) for k, v in entry["defines"].items()))


def route_available(arch: str, required_keys: tuple[str, ...] = ()) -> bool:
    """True when every key in ``required_keys`` has a program built for ``arch``."""
    if arch not in ARCHES:
        return False
    for key in required_keys:
        entry = KERNELS.get(key)
        if entry is None or arch not in MODULES[entry["module"]]["arches"]:
            return False
    return True


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
def gen_cake_kimi_k3_fp8_projection_module(name: str, arch: str, defines: Defines = ()):
    """JIT spec of program ``name`` compiled for ``arch`` with the compile-line ``defines``.

    The architecture and the defines are part of the spec name, so two
    architectures (or two instantiations) never share one cached library; the
    closure digest covers the translation units and their shared headers."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise ValueError(
            f"program {name!r} is not built for {arch} ({record['arches']})"
        )
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in record["sources"]]
    suffix = "".join(f"_{key}{value}" for key, value in defines)
    return gen_jit_spec(
        name=f"{name}_{arch}{suffix}_" + record["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
            *[f"-D{key}={value}" for key, value in defines],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            root,
            *dict.fromkeys(p.parent for p in sources),
            *_header_dirs(),
        ],
        use_fast_math=False,
    )


@functools.cache
def load_cake_kimi_k3_fp8_projection_module(
    name: str, arch: str, defines: Defines = ()
):
    return gen_cake_kimi_k3_fp8_projection_module(name, arch, defines).build_and_load()
