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
# ``ARG_PLANS`` holds the five argument orders of the generated host launchers
# as ``[kind, name]`` pairs (``tma_buffer`` / ``buffer`` tensors, ``parameter``
# scalars and the ``grid`` dims; ``cake_backend._bind`` orders a call by
# name): ``decode`` and ``decode_tso`` (the swap-AB decode programs without /
# with the TMA-store ``OUT`` map), ``gemm`` and ``gemm_skp`` (the persistent
# GEMM programs; the prefetched stream-K form takes three more scalars) and
# ``quant``.  ``_PROGRAMS`` holds one row per physical program (a kernel
# source plus its host launcher), keyed by the 20-hex digest of the module
# name ``cake_kimi_k3_fp8_projection_<digest>``: the architectures the source
# compiles for, the ``ARG_PLANS`` key, the exporter's closure digest (SHA-256
# over the exported translation units and the shared device / host headers
# they include; the JIT library is named by its first 20 hex digits) and,
# where a record differs from the family default, the digest of the program
# whose kernel translation unit it builds (``kernel``; default: its own
# ``<digest>_kernel.cu``) and its extra nvcc flags (``compile_flags``;
# default ``--use_fast_math``).  ``MODULES`` expands the rows into one record
# per module name: ``sources`` (the kernel translation unit and the record's
# own ``<digest>_binding.cu`` under ``csrc/``), ``arches``, ``compile_flags``,
# ``arg_plan`` and ``closure_sha256``; every launcher exports the ``run``,
# ``run_plan``, ``run_launch`` and ``run_chain`` entries.  One source serves
# every listed architecture; the architecture-specific lowering lines sit
# behind ``__CUDA_ARCH__`` guards inside the source, and the loader compiles
# it with the exact flag set of the device it runs on.
#
# ``KERNELS`` maps the logical kernel key the host dispatcher resolves at
# preparation to its program and the compile-line ``defines`` of that
# instantiation (``-DNAME=value``):
#
# * ``quant:u<units>``                 the per-token 1x128 E4M3 / UE8M0
#   quantization launch; one program, ``units`` K blocks per half warp is the
#   ``QUANT_UNITS`` define (the ``cake_backend.quant_units`` rule: 1 for
#   M <= 256, 2 / 4 for the large-M rows; or the width a measured table cell
#   pins with ``quant_units``);
# * ``gemm`` / ``gemm_rstaged`` / ``gemm_tstore``   the persistent 2-CTA
#   block-scaled tcgen05 GEMM (256 output columns per CTA pair, M > 256) with
#   the register, staged-register or TMA-store epilogue; the instance tokens
#   follow the stem in the order ``[_n192|_n384][_g64][_k128][_cs<S>|_st|_gs][_s<n>][_ts<n>][_sk|_skf<n>|_sks<n>][_pdlb][_pf<D>]``
#   (``_n384``: the 128 x 384 pair-tile form of the TMA-store / staged-register
#   programs on the tabulated ``gemm_bn`` 384 rows, round 10 -- three 128-wide
#   accumulators share one activation stage; always ``_k128``, never with the
#   stage / stream-K / prefetch tokens; same output bits; ``_g64``: the 64-row-tile
#   raster group of those two epilogues on the tabulated ``gemm_gm`` rows, round 10;
#   same output bits; ``_k128``: the 128-K operand hand-off form of the TMA-store / staged-register
#   programs on the tabulated ``gemm_bk`` rows -- every 128-K half of an operand
#   stage is loaded and multiplied under its own barrier pair; same output bits;
#   ``_s<n>`` / ``_ts<n>``: operand stages / staging slots per epilogue group of
#   the TMA-store programs on the tabulated ``gemm_stages`` / ``gemm_ts_slots``
#   rows, round 9 -- two stages and four slots on the short-K large-M rows; same
#   output bits; ``_pdlb``: the PDL-wait-after-weights form of the TMA-store /
#   staged-register programs on the tabulated ``gemm_pdlb`` rows, round 11 --
#   the loader waits for the quantization launch once per CTA right before its
#   first activation box, after that stage's weight boxes; closes the stem;
#   same output bits; ``_st``: the static one-tile form of the plain ``_n192_k128``
#   TMA-store / staged-register programs on the tabulated ``gemm_static`` rows,
#   round 11 -- one output tile per CTA pair over 2 x tiles CTAs, no
#   cluster-launch-control work ring, admissible when the tiles fit the resident
#   pairs; the ``_cs<S>`` slot; same output bits; ``_gs``: the static grid-stride
#   form of the plain ``_k128`` / ``_n192_k128`` TMA-store programs on the
#   tabulated ``gemm_gs`` rows, round 11 -- pair p of 2 min(T, resident pairs)
#   CTAs owns tiles p, p + P', ..., no cluster-launch-control work ring; the same
#   slot; same output bits);
# * ``decode:t<tok>_p<stages>[_fused][_a<n>][_e<n>][_res][_r<xb>][_q<lanes>][_cs<C>]...[_tso][_dpf][_pi<n>][_xp]``
#   the swap-AB split-K decode kernel for one token-tile width, TMA pipeline
#   depth, in-CTA quantization (``_fused``), ``n`` TMEM accumulator sets
#   (``_a``) and ``n`` epilogue warp groups (``_e``; both absent at 1),
#   resident token tiles (``_res``),
#   a decoupled ``xb``-deep BF16 token ring (``_r``), narrow quantization
#   units of ``lanes`` lanes (``_q``; absent = half-warp units) and a
#   ``C``-CTA cluster split-K exchange (``_cs``), as the measured dispatch
#   table selects per ``(N, K, M bucket)``; the key closes with the lever
#   tokens in the Cake symbol order -- the TMA-store epilogue (``_tso``), the
#   tensor-map descriptor prefetch in the kernel prelude (``_dpf``), the
#   cross-item L2 prefetch (``_pi<n>``), the first-item weight-stream L2
#   prefetch before the PDL wait (``_pw<n>``, round 8, adopted in round 11)
#   and the parallel-issue DSM exchange (``_xp``); ``decode_kernel_key``
#   lists every token.
#
# The rows, ``ARG_PLANS`` and ``KERNELS`` are populated by the generated-program
# export; do not edit them by hand. Two hand-maintained exceptions, pending
# regeneration by the export: a row whose kernel source is another record's
# translation unit (a prefetch-lever or raster-width twin) names that program
# in ``kernel`` and carries in ``compile_flags`` the lever define
# (``-DCAKE_LEVER_<X>=<n>`` or ``-DGROUP_M=<n>``) and ``-DCAKE_PROGRAM_KERNEL``
# / ``-DCAKE_PROGRAM_SHIM`` naming its own kernel symbol and host-shim
# namespace (the shared kernel source carries the matching guarded defaults
# and ``#if`` arms); and the row form itself is a hand fold of the exported
# per-record literal.
ARG_PLANS: dict[str, list[list[str]]] = {
    "decode": [
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
    "decode_tso": [
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
        ["tma_buffer", "OUT"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm": [
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
        ["buffer", "sk_partials"],
        ["buffer", "sk_flags"],
        ["parameter", "sk_pairs"],
        ["parameter", "sk_rem"],
        ["parameter", "sk_ksplit"],
        ["parameter", "sk_dp"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "gemm_skp": [
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
        ["buffer", "sk_partials"],
        ["buffer", "sk_flags"],
        ["parameter", "sk_pairs"],
        ["parameter", "sk_rem"],
        ["parameter", "sk_ksplit"],
        ["parameter", "sk_dp"],
        ["parameter", "sk_l"],
        ["parameter", "sk_slack"],
        ["parameter", "sk_b"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ],
    "quant": [
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
}
# Row vocabulary: the architecture lists a program is compiled for, the
# family's default nvcc flags and the path stem of the generated sources.
_SM100 = ("sm_100a",)
_SM103 = ("sm_103a",)
_BOTH = ("sm_100a", "sm_103a")
_FAST_MATH = ("--use_fast_math",)
_SOURCE_PREFIX = "cake_kimi_k3_fp8_projection/cake_kimi_k3_fp8_projection_"
_PROGRAMS: dict[str, dict[str, Any]] = {
    "021cc0171d570f358807": {
        "arches": _BOTH,
        "arg_plan": "decode_tso",
        "closure_sha256": "4c9f08f21e588f9aed237ce84f8f0a98f170ae4edbbb2ba025d32940751905e6",
    },
    "09ef673497e9adb01b7e": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "32005b54524c9fed9ef4146122e4eddaa23388d0bf1ad58f10438d6f882a11dc",
    },
    "0cdb8cc10eb1192d3635": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "58dfece0362905fed5bf54eeb6d1b8e2de12582e808697b8c7851834c52da30f",
    },
    "0f46cbbf5011f4371f92": {
        "kernel": "baa922f3fad64f788380",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_0f46cbbf5011f4371f92",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_a1c520d06c82b077",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "e8ec29024c4ce5c2ce6ed10415ca8982faccfd3cc6b68793f31ad8d1b4a484bb",
    },
    "11cbad4e31342b47dc08": {
        "arches": _SM100,
        "arg_plan": "gemm",
        "closure_sha256": "162879a68df08839e2f912db94900185614e1987b5fd8cd0b8e0f7bbe6bf2fb7",
    },
    "20ea24437d297722619d": {
        "kernel": "5196dc524176f72bfdb6",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_20ea24437d297722619d",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_22d7bf24b2a10e03",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "d78df97e4a06f8ac3808e020d55bad06de133ef99c1bb2c08c0d658bb67b98d0",
    },
    "220aae40317b8e398117": {
        "arches": _SM103,
        "arg_plan": "gemm",
        "closure_sha256": "ebc0807c4dc69e3de892eb730d25a2f28fdcc278d1b9d05fda627af15d19fe09",
    },
    "24a91d5a85a8465ab6ca": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "d0a112bb1b8e16a2c5a7f9bd4c92f90ff3a5c1bda01c15474f7cdb9ce75b7e8a",
    },
    "28d3f834bafa2bf36d00": {
        "kernel": "baa922f3fad64f788380",
        "arches": _SM100,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_28d3f834bafa2bf36d00",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_78e8c78ada438ce9",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "5f4e0c4c2e341c202711600469d006dca6d5eed57200d4acf01f47bb1a4f6173",
    },
    "2ec639014a8fa1272b0a": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "01411f8b6d8d3b0a753c4f8d676f5ce7093731d6bf9810cb944976743f95fb36",
    },
    "3467fab6a69e46ade21d": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "0e8a960ea62fffdfd7d4b18eb8576031ec28fe59bfce8d462c451b137feed1f6",
    },
    "35deb0e5eea243f77d81": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "86eb9fed4f3bfda2eca45e7e2e884f88685cd47ea2cb04308e0e18c220edb807",
    },
    "3654f2f7a859afea0aa6": {
        "kernel": "24a91d5a85a8465ab6ca",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PX=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_3654f2f7a859afea0aa6",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_11296ebf54354318",
        ],
        "arg_plan": "decode",
        "closure_sha256": "084e8f5bbc110877908e201178e021d32087a44a9da9ad7793bad7bb0541ef4a",
    },
    "39fd3fc41b3d7ca27508": {
        "kernel": "71c5eadf708248a0d43c",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PW=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_39fd3fc41b3d7ca27508",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_b11387199d0dd7c3",
        ],
        "arg_plan": "decode_tso",
        "closure_sha256": "2047dc73ebf40ce9700ffa6248978fc3116908e0980971c6b8032e199ee3866c",
    },
    "3e38b6f34453a3f8849d": {
        "arches": _SM100,
        "arg_plan": "decode_tso",
        "closure_sha256": "574af7017aaac22a6be2bea0f7d1f050b1353b34779616ac782d9245f5bc7459",
    },
    "3fa6af62a780530c1810": {
        "arches": _SM103,
        "arg_plan": "decode_tso",
        "closure_sha256": "46f07b934bc4beb73ce3c6bcd59008a6763962d6664743985110035d488d7acf",
    },
    "43d3d6683770c0bf8bad": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "59939f4b6d1060c9209f0f3175016745438fba6cd70dc818946d6f326a8b3704",
    },
    "46c66f5e27a55602bfd7": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "7e3e46ece57c4be2355cbb74414d4c2004518d2b503bff2ca4a12f5fdfd241df",
    },
    "46f4c0a44682680afa00": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "7363c51e8bd3a3d86bb8d2ba16e9f9b99ee3d63a28c58b19c7b2e5089b44f500",
    },
    "4be8750b7ef4b9419aaf": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "41754110ec6a7aeeb5e10ae4675119667e00fb95bb224cf37097e17c3fc72be7",
    },
    "4da95d45aada36775de5": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "42a8b6ecfd8d6162104fb827c7b70f2cd142cddbb34cd6ae4e891e2534bd199d",
    },
    "504b13e2236557baa266": {
        "kernel": "a8222cc117f456537bdc",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_504b13e2236557baa266",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_0bb5ea91c93581fe",
        ],
        "arg_plan": "decode",
        "closure_sha256": "e5650c6490d16d7259816f0f47f112dece9832c659b1d231eb91164cea09411e",
    },
    "5196dc524176f72bfdb6": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "ebff74149d3a2ad79fc47e2df0c301d413d00350923486e9a0709bcf79afa1f9",
    },
    "53557e910bed0b361877": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "d3abe2a952d33268729ad975d6853559077ec6997ea20b59a0a84cedc9d095e3",
    },
    "55edd49d3fc4d8738f68": {
        "kernel": "3fa6af62a780530c1810",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=3",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_55edd49d3fc4d8738f68",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_c12a23595302b755",
        ],
        "arg_plan": "decode_tso",
        "closure_sha256": "f52ea48e3d5a313f0c9b77f5751ad449c36a5d7b1e3715c08329761af5b59e52",
    },
    "5660a6b1b752dc71dc15": {
        "kernel": "ee2e0fe9754d301492e2",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=4",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_5660a6b1b752dc71dc15",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_97fab907207c981b",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "009d53bd30dee1746e7a3798c083381f4b8e243d51b360139a4069029bb16f4d",
    },
    "5a04cd74fb0cbb150e8f": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "f3fbe96627c756fbd5c7ea70d4a1f194eb74dd95c02cc979444f3b2f7dc01f01",
    },
    "5dbbe3415920815e812e": {
        "arches": _BOTH,
        "arg_plan": "decode_tso",
        "closure_sha256": "e045f1c7ac789ea829e2534c660312054e50a3137640a84921c7243213d50076",
    },
    "6055768204261cb211ec": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "39cf640a6f339f420bfd170fb06ec63241ad71c0566478e5b077a2e37afa85e7",
    },
    "60d7ceeca1565284fa84": {
        "kernel": "0cdb8cc10eb1192d3635",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=3",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_60d7ceeca1565284fa84",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_6da2c2551ea588de",
        ],
        "arg_plan": "decode",
        "closure_sha256": "a5762c131a13a02e8e18c508e9d0cab69be89bcadab898c6afb80ebea3e1e842",
    },
    "61d0bd00b93effa9ce40": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "06f28e5dec321c27ceb72ae5f5e75f6f68464ecab0d64eee4da73ab00bb2d00c",
    },
    "62ed5b839ef485e277bc": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "94f7e615508ed157b5dfddbda225f2354a1bf1a77aa9e82347857a0fdec8232f",
    },
    "66015597a7e7c29045d7": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "3083c082a1437d431c237331e9118bdc3ae989a3bb9ccb8253c91700c7f77a5a",
    },
    "67b79a99dbca18833ff9": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "51c7a2003f1c2b0aae634c880263d04c1a33e9bb6c1a53ea8beea4c6a73f9e17",
    },
    "6965aeafa116c7f6ef89": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "f914bd43ad469c0ddd318ea84393608184c51872ff8a933e247f68d9da7521c3",
    },
    "6c9831e79f12b9d92832": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "b28e994ee223e21faf81e7c92f8680d14c6c682f6142ce0e165b0c866d3d8dab",
    },
    "6f14858293e3d9f61c93": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "1ebc91dc6f46fc21bf17d78e67b91b90324d5a7cb1f0f61ba20c596af854ea91",
    },
    "718b9dd8552d49e4f782": {
        "kernel": "f13cdc4b7875b5c1ca74",
        "arches": _SM100,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_718b9dd8552d49e4f782",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_6be44e6374d5dbfe",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "8219a2604a89ee105486e3bd2328b62920cde744041a985256fc329d593e847b",
    },
    "71c5eadf708248a0d43c": {
        "arches": _BOTH,
        "arg_plan": "decode_tso",
        "closure_sha256": "c217025f80f7ee8c954b63260c9ff69d8c8cde617544ffa2ec1253a2d1b6000b",
    },
    "7518d0ddd0f8eb901098": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "b70f824217706a34ebfd6fd90f5d03b3f262671d8c664d8651e6675d10945266",
    },
    "75e18bf3c6d16aa5ba9a": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "31a2127013dc6b76517eeb3113de8928d9d180702ca009c50137c1ec0508ce38",
    },
    "76f152b19e7a843ba820": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "3ede74f7ee98cb072f91782cb95e8659723a6ed1f9706a66d5ee53452e79669f",
    },
    "79f0e568c12ad3ab2f9a": {
        "kernel": "b66413033ce4e9b54af0",
        "arches": _SM100,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_79f0e568c12ad3ab2f9a",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_31eaf8dabc5b9205",
        ],
        "arg_plan": "decode_tso",
        "closure_sha256": "2232ffabb2a976cf11957005e4969b349e700ff3e23ec1424d340c89898662bd",
    },
    "7a477ba7758c4954b1ff": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "1b8e46d2f352df1c1a57eee9c4cb36dd61514688ac6810e9ff258c190bece231",
    },
    "7a9e8f32c95b21cd3ba3": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "c84dc0871d110538d0eccf7861aa42c530f8bc32979b284bfbd334ed4f5c19aa",
    },
    "7abdff4851061bffe5cb": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "673a9771c2d2812aa02d4c9948b143d6863ee95158945cb8ae9267878d2db72a",
    },
    "7c0c2078f3110d338d76": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "25c4b34dc85ff91ae6ee7c741c8abeac3c832bda5b8cffce07df894086c31b20",
    },
    "81f8643b8577aafe1a7d": {
        "arches": _SM100,
        "arg_plan": "gemm",
        "closure_sha256": "52244accbf055f973350a77b5e8ace66f3166caa23a054d8474696e4abf47bd2",
    },
    "82375137e249e01d15c2": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "4f0e6c31d9122c7246b0886d76f2bf87e08c029076e1f5eb55b35bca8cb6bfca",
    },
    "83c6d5b5d853851ec87a": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "4221aed597b8e6a4f832c099ed9db8ac993c35afd9cd305bffe7c57b7d10851a",
    },
    "88eaaccbac79ef8cf184": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "c3563db7b75650cd035aff834f748ccbfe6801306adbd4f85c257132290627c7",
    },
    "8b3063e42a11bfdd5253": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "88359eb3a76ef280f18e5d569fd37cf9d63d276bf537a52d85c303caca135af5",
    },
    "8b6f92bd50579503f035": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "eaf1b540049635a76eb78463c051d29c13b99a5ff8e02b9a851e4be689e01b31",
    },
    "8becf9f889fc4a657c22": {
        "kernel": "5196dc524176f72bfdb6",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_8becf9f889fc4a657c22",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_6ba646259e9cadc4",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "f4702403472d7996382d3054ecadfbfeb19abd71652aca23c5d26c91a8e59b94",
    },
    "8cceae711ab4461f4135": {
        "arches": _SM100,
        "arg_plan": "decode_tso",
        "closure_sha256": "b6b7f7df4775cb24c219e8a009cc1958551e7d910128fe59b5902bf238539db6",
    },
    "922236e00dea3cbba602": {
        "kernel": "83c6d5b5d853851ec87a",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PX=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_922236e00dea3cbba602",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_4492bd3ae544c0e2",
        ],
        "arg_plan": "decode",
        "closure_sha256": "6aab7dee3cdf250144b3fe59249c7803bafd8587634774bb8c6d059d6290cb06",
    },
    "9265c88bf850d89b0c01": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "dc16c1f0f36015cfc4c9d26f3a8f2a1147beafa3d0abb563541661a114a1594c",
    },
    "93fd6a71010eb8fb34f5": {
        "kernel": "88eaaccbac79ef8cf184",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_DPF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_93fd6a71010eb8fb34f5",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_e8c2a330a7db152e",
        ],
        "arg_plan": "decode",
        "closure_sha256": "40dd0c5676312182c33b73cb435a7a8fd728674c613c4ab57069bbb7d0662139",
    },
    "96a3edd7d4db71c390df": {
        "kernel": "53557e910bed0b361877",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_96a3edd7d4db71c390df",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_38175979a2656920",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "240ae0103081acfa7f38b7ee15976f22adf8070536bf80f1ab6104131e659034",
    },
    "974015cffe52e2f67773": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "e9c3dced6784768e91b3e2447be8e2df426251c029e736576ddff409887233cb",
    },
    "9d3611069a6cf0422f05": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "4d1297ad4f6114acf4cbf8d066812a5a860f5497171cb212f12bb2c948d01780",
    },
    "9f4331dc1b45870d0d95": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "532ae8bde9285f1e766dc415f83fbf1ca651dc185d2f4036aa65ef1091279d82",
    },
    "a1ee4a9c1f1f7ea1917c": {
        "kernel": "09ef673497e9adb01b7e",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_a1ee4a9c1f1f7ea1917c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_1f06dccd0c0a72f1",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "4413bf43e20077c9eda77247747e24169b542d3e604bc32cf4eb9629f50f64d9",
    },
    "a2e46d7b502767bca64c": {
        "kernel": "71c5eadf708248a0d43c",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_a2e46d7b502767bca64c",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_ea34dee130223833",
        ],
        "arg_plan": "decode_tso",
        "closure_sha256": "5e5defcbfc031d502263b8d733a2e8f8e5b51a229451f36821036c141f15bca9",
    },
    "a8222cc117f456537bdc": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "3885ac9c57adcc2d3b5dbb7bf4f123119307beae8143e6e2aab2915296fa79e4",
    },
    "a9484fb9b47a717a2207": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "9ab6cc9366d57925194c84b52f0f3f6733679cdcf81aaf28ec5b6c268619d9df",
    },
    "ac0039f3b84b450bea6d": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "a4ba33c2f2b601b44835cb1bed7c22226fbf68bc799068fc7e93a41ad8bccdd7",
    },
    "ada93e51754d04322ff1": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "897448cfeff9842e20943e70eb064ceece74f77a997eb3d216b12dc634769ed6",
    },
    "ae4ec3a3d90aac71ea20": {
        "kernel": "83c6d5b5d853851ec87a",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_ae4ec3a3d90aac71ea20",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_0c748b922871df7e",
        ],
        "arg_plan": "decode",
        "closure_sha256": "268cc7c6499cdc92af88a68cef04636c18311301c00d4b1501fb18a689be1fbc",
    },
    "ae66e1c9615449d1921b": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "e3bbbaf3fb95d68b13507c03e362fd0c39217c4752c892af78cd4a971f9e29b5",
    },
    "af4862ae5f0dee9355a7": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "4f9ef4eb8a51aaf068df18707c1bb212383db7a9072f9c1f03bcde83bf5dc626",
    },
    "b2e2ef1b2a5df0f83b1b": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "0092411a1d4ee7f37068c97aef5e2beb90e585a60298091f65f5c94dbea65d58",
    },
    "b658c4971aa78e9c9165": {
        "kernel": "021cc0171d570f358807",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_b658c4971aa78e9c9165",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_5f5331e70dd7f128",
        ],
        "arg_plan": "decode_tso",
        "closure_sha256": "c3edac63039f5a323fc77faae26b05845793787f988ac20134f1ad1d0c789475",
    },
    "b66413033ce4e9b54af0": {
        "arches": _BOTH,
        "arg_plan": "decode_tso",
        "closure_sha256": "44520811651aa5fd83a2d8b708dab7769f5a74e30b5f960681132c900b01ad03",
    },
    "baa922f3fad64f788380": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "45c997450e00dc4ecced2625d89d96488080d8f3cd151e2ad6deb82eac312323",
    },
    "bac8891ceedeac1efaf2": {
        "kernel": "09ef673497e9adb01b7e",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_bac8891ceedeac1efaf2",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_761ee9710b90f91f",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "3331927d52791492681f0f671ac947546b563ec92f5f3496282727637033c5d8",
    },
    "c068dc77b0968fcc0a15": {
        "kernel": "83c6d5b5d853851ec87a",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_DPF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_c068dc77b0968fcc0a15",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3f012b9ad612b7c9",
        ],
        "arg_plan": "decode",
        "closure_sha256": "72974c3f331689229d8b3a2471d6509104f8ba0181b25840f6f674ebdd745bb3",
    },
    "c22cadffa1705cdcb3b8": {
        "kernel": "82375137e249e01d15c2",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DGROUP_M=64",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_c22cadffa1705cdcb3b8",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_4d04809c3755ce1f",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "664cdf2d88095a56b11b65fc3f916576f85b28358ffd15a3c208ae70dbee3be8",
    },
    "c2b4ed9c6a005c71cf4b": {
        "arches": _SM103,
        "arg_plan": "gemm",
        "closure_sha256": "fb9baa705123d9ab66d489084066a40482e42e994055068c9334fffcd9ca5b3c",
    },
    "c4486703f068891e6898": {
        "kernel": "6c9831e79f12b9d92832",
        "arches": _SM100,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PX=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_c4486703f068891e6898",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_ffbfb6caf29869bd",
        ],
        "arg_plan": "decode",
        "closure_sha256": "2d8eac81161051ad6e2b0a49324cd6b443e63a686eebab0581f4cb749fadba90",
    },
    "c5da9b7926ca218a3b36": {
        "kernel": "62ed5b839ef485e277bc",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PX=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_c5da9b7926ca218a3b36",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_f5bbd342029b1c1b",
        ],
        "arg_plan": "decode",
        "closure_sha256": "1a4421c6489ec9b7176373091ae1210a349eac4689f9d11c90b37fc12ad8dce1",
    },
    "c9822f9e834e3240b188": {
        "arches": _SM100,
        "arg_plan": "decode_tso",
        "closure_sha256": "1df7b89ba235c8a1b9dccad244ded159bdca9d9f30b9c019ac6c44aa0c813acc",
    },
    "ca4fbacb9d82722ff4ce": {
        "arches": _BOTH,
        "compile_flags": [],
        "arg_plan": "quant",
        "closure_sha256": "d5d39facb312fabff52152738f3d1a20d91ed2525532d3672c2dc0636c8f0de1",
    },
    "cc335af0b89da542ea91": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "d2bfe95c5e51a89dc6717980797c24ffa6a4bdaf9e033f385ccc956b84cf890c",
    },
    "cced2467167c42643d15": {
        "kernel": "53557e910bed0b361877",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_cced2467167c42643d15",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_430d9070efba3939",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "b2db5da664713b6c2c0f9605b618e7a651ff6c8f17a490c824984d7bb321fefe",
    },
    "ceeb47741a53f199cb00": {
        "arches": _SM103,
        "arg_plan": "decode",
        "closure_sha256": "52d0db6ab35c0f6fe634e3faee342b62bc32728230324df637f831c28da8efe3",
    },
    "cf10903c711b1dd4f7af": {
        "kernel": "8cceae711ab4461f4135",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_DPF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_cf10903c711b1dd4f7af",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_267677c142dd3880",
        ],
        "arg_plan": "decode_tso",
        "closure_sha256": "6c8f3abb77dc45a57891565d0df17f50e8267f0fd4c6f58278a2d53e471ac74b",
    },
    "d1ac3a99961e78557b6f": {
        "kernel": "a8222cc117f456537bdc",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PW=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_d1ac3a99961e78557b6f",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_b0fff8ce01adde74",
        ],
        "arg_plan": "decode",
        "closure_sha256": "32ee37fa6aa838169f01128fa470f7508004124af5dcd131cffbccc3caafe0ea",
    },
    "d4cfd6a99b1a5f08bd75": {
        "kernel": "f90460931d7c9a77f1a1",
        "arches": _SM100,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=1",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_d4cfd6a99b1a5f08bd75",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_f3481f1632c2f219",
        ],
        "arg_plan": "decode",
        "closure_sha256": "cc08d2895fa803203c0abe599ccb8038dabcf2aa21fef501c97a2c0b3b0a0b3b",
    },
    "d4d43a9ba6f62cb3d829": {
        "arches": _SM103,
        "arg_plan": "gemm",
        "closure_sha256": "9904e266d5a05866e205108c5a95f292707fe3a96a1f6cbb60dea9ea7a3ccc58",
    },
    "e319dfda1d4453cff859": {
        "kernel": "baa922f3fad64f788380",
        "arches": _SM103,
        "compile_flags": [
            "--use_fast_math",
            "-DGROUP_M=64",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_e319dfda1d4453cff859",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3fdcd009bdf3f5b7",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "5f3001db0a0598ad612948149f8e270aac005948d29439c84fd4abe409f6e478",
    },
    "e796bf37781af8d66734": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "baf78527096a945b95c340cd7145639370327896fdeb87b98e37b479ea91f3da",
    },
    "e985ccc1acc9a1b0c018": {
        "kernel": "6965aeafa116c7f6ef89",
        "arches": _BOTH,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_e985ccc1acc9a1b0c018",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_0a67a5bb2b4a7eda",
        ],
        "arg_plan": "decode",
        "closure_sha256": "822b7dbcc8495628df38ee5526555424e0c06f1ff83f4a61324c3361fdcffdb8",
    },
    "e9af5f2d6bdaec79f30a": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "5a011ecc8e36d0b8a44112107e4199acd51d21b267cd70b45ce39735c39648c0",
    },
    "ea0109448a87bf68aebb": {
        "arches": _SM100,
        "arg_plan": "gemm_skp",
        "closure_sha256": "3c4440091a98ebd638e2f3e943cd28dac7f8ef6d020b1ab7b3da9211bf178f20",
    },
    "ea8d2ae10f643d1b0341": {
        "kernel": "6055768204261cb211ec",
        "arches": _SM100,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PW=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_ea8d2ae10f643d1b0341",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_3eb3e28f2802b3d2",
        ],
        "arg_plan": "decode",
        "closure_sha256": "937f9d14d06ffbee7063f2a24f5b4746c36079e61569877d7d337284026b4efe",
    },
    "eb810f44b85f3a53dea5": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "0114e1bac9a2758e6e538f3b85d1aa0d4a5a250bfd91d1d45685b2a7ec6eb293",
    },
    "ee2e0fe9754d301492e2": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "78dd9d72a2a189eaf9b0a26a7cbe789f7a115bf8508889f10db50ba039aeffc9",
    },
    "f01f38424f66eefb47b0": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "8f9e3d81a8a541ce7fcea7203be018ccacd8c4c9df6fc9552c3e6da02f18c0c3",
    },
    "f13cdc4b7875b5c1ca74": {
        "arches": _BOTH,
        "arg_plan": "gemm",
        "closure_sha256": "eaff4d3dd61e38f790c42ca06625e594110c49936a9bf340b6dfca7099cd5a7e",
    },
    "f65b17236fc8c8df0336": {
        "arches": _BOTH,
        "arg_plan": "decode_tso",
        "closure_sha256": "e16379487b2313416618796b5564e3d275967347e6adf6428cb8e81dc9d3c4f5",
    },
    "f90460931d7c9a77f1a1": {
        "arches": _BOTH,
        "arg_plan": "decode",
        "closure_sha256": "99073e683c39a9c7011cb71c4263d87f686c66162ef3f835b762f899e074621e",
    },
    "fcc485f4025b5d964562": {
        "arches": _BOTH,
        "arg_plan": "decode_tso",
        "closure_sha256": "4923c887e86dc8ceae838bf9017a34e6fef782d571c269cd0b65530f24e7c87f",
    },
    "fcd2cccaffa526675b6d": {
        "kernel": "e9af5f2d6bdaec79f30a",
        "arches": _SM100,
        "compile_flags": [
            "--use_fast_math",
            "-DCAKE_LEVER_PF=2",
            "-DCAKE_PROGRAM_KERNEL=kernel_cake_kimi_k3_fp8_projection_fcd2cccaffa526675b6d",
            "-DCAKE_PROGRAM_SHIM=cake_host_shim_1aad653a89cd8fd9",
        ],
        "arg_plan": "gemm",
        "closure_sha256": "da9f9d0e2d775c3ca88106c327a8b282fca99bb55136672cf5c0375f3e995f68",
    },
    "fd419903fff14b0b8e10": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "a9d679f6a77f4cf3ea0ba236eaecbaf20f80ae88bbfc04b9b836300c7663bb1e",
    },
    "fdb1ab7e8ccbb2e8e1d8": {
        "arches": _SM100,
        "arg_plan": "gemm",
        "closure_sha256": "03443eff21977e77314cb947617f4569a6984e82cb5aa124d548f5b6922699e8",
    },
    "fe01dd81c571f1f74124": {
        "arches": _SM100,
        "arg_plan": "decode",
        "closure_sha256": "410700dbf2dd3b5dfe2e441f74497323156bda15bcc7d2342a3876d08fe21aa5",
    },
}


def _module(digest: str, row: dict[str, Any]) -> dict[str, Any]:
    """The ``MODULES`` record of program ``digest`` from its ``_PROGRAMS`` row."""
    return {
        "sources": [
            f"{_SOURCE_PREFIX}{row.get('kernel', digest)}_kernel.cu",
            f"{_SOURCE_PREFIX}{digest}_binding.cu",
        ],
        "arches": list(row["arches"]),
        "compile_flags": list(row.get("compile_flags", _FAST_MATH)),
        "arg_plan": row["arg_plan"],
        "closure_sha256": row["closure_sha256"],
    }


MODULES: dict[str, dict[str, Any]] = {
    f"cake_kimi_k3_fp8_projection_{digest}": _module(digest, row)
    for digest, row in _PROGRAMS.items()
}
KERNELS: dict[str, dict[str, Any]] = {
    "decode:t128_p3": {
        "module": "cake_kimi_k3_fp8_projection_f90460931d7c9a77f1a1",
        "defines": {},
    },
    "decode:t128_p3_mc2_pf1": {
        "module": "cake_kimi_k3_fp8_projection_0cdb8cc10eb1192d3635",
        "defines": {},
    },
    "decode:t128_p3_mc2_pf1_tso": {
        "module": "cake_kimi_k3_fp8_projection_3fa6af62a780530c1810",
        "defines": {},
    },
    "decode:t128_p3_mc2_pf3": {
        "module": "cake_kimi_k3_fp8_projection_60d7ceeca1565284fa84",
        "defines": {},
    },
    "decode:t128_p3_mc2_pf3_tso": {
        "module": "cake_kimi_k3_fp8_projection_55edd49d3fc4d8738f68",
        "defines": {},
    },
    "decode:t128_p3_pf1": {
        "module": "cake_kimi_k3_fp8_projection_d4cfd6a99b1a5f08bd75",
        "defines": {},
    },
    "decode:t128_p3_pf1_tso": {
        "module": "cake_kimi_k3_fp8_projection_79f0e568c12ad3ab2f9a",
        "defines": {},
    },
    "decode:t128_p3_tso": {
        "module": "cake_kimi_k3_fp8_projection_b66413033ce4e9b54af0",
        "defines": {},
    },
    "decode:t16_p3_fused": {
        "module": "cake_kimi_k3_fp8_projection_61d0bd00b93effa9ce40",
        "defines": {},
    },
    "decode:t16_p3_fused_cs14_px1": {
        "module": "cake_kimi_k3_fp8_projection_af4862ae5f0dee9355a7",
        "defines": {},
    },
    "decode:t16_p3_fused_cs14_px1_xp": {
        "module": "cake_kimi_k3_fp8_projection_3654f2f7a859afea0aa6",
        "defines": {},
    },
    "decode:t16_p3_fused_cs14_xp": {
        "module": "cake_kimi_k3_fp8_projection_24a91d5a85a8465ab6ca",
        "defines": {},
    },
    "decode:t16_p3_fused_r2_xh_q8_cs14_px1_xp": {
        "module": "cake_kimi_k3_fp8_projection_7a477ba7758c4954b1ff",
        "defines": {},
    },
    "decode:t16_p3_fused_r2_xh_qe_q8_cs14_xp": {
        "module": "cake_kimi_k3_fp8_projection_eb810f44b85f3a53dea5",
        "defines": {},
    },
    "decode:t16_p3_fused_r4_x4_q8_cs6": {
        "module": "cake_kimi_k3_fp8_projection_fd419903fff14b0b8e10",
        "defines": {},
    },
    "decode:t16_p4": {
        "module": "cake_kimi_k3_fp8_projection_6055768204261cb211ec",
        "defines": {},
    },
    "decode:t16_p4_fused": {
        "module": "cake_kimi_k3_fp8_projection_83c6d5b5d853851ec87a",
        "defines": {},
    },
    "decode:t16_p4_fused_cs2": {
        "module": "cake_kimi_k3_fp8_projection_46c66f5e27a55602bfd7",
        "defines": {},
    },
    "decode:t16_p4_fused_cs4": {
        "module": "cake_kimi_k3_fp8_projection_6c9831e79f12b9d92832",
        "defines": {},
    },
    "decode:t16_p4_fused_cs4_px1": {
        "module": "cake_kimi_k3_fp8_projection_c4486703f068891e6898",
        "defines": {},
    },
    "decode:t16_p4_fused_dpf": {
        "module": "cake_kimi_k3_fp8_projection_c068dc77b0968fcc0a15",
        "defines": {},
    },
    "decode:t16_p4_fused_pf1": {
        "module": "cake_kimi_k3_fp8_projection_ae4ec3a3d90aac71ea20",
        "defines": {},
    },
    "decode:t16_p4_fused_px1": {
        "module": "cake_kimi_k3_fp8_projection_922236e00dea3cbba602",
        "defines": {},
    },
    "decode:t16_p4_fused_q8": {
        "module": "cake_kimi_k3_fp8_projection_62ed5b839ef485e277bc",
        "defines": {},
    },
    "decode:t16_p4_fused_q8_px1": {
        "module": "cake_kimi_k3_fp8_projection_c5da9b7926ca218a3b36",
        "defines": {},
    },
    "decode:t16_p4_fused_r2_xh_qe_q8": {
        "module": "cake_kimi_k3_fp8_projection_9d3611069a6cf0422f05",
        "defines": {},
    },
    "decode:t16_p4_fused_r2_xh_qe_x4_q8_cs6_px1": {
        "module": "cake_kimi_k3_fp8_projection_8b3063e42a11bfdd5253",
        "defines": {},
    },
    "decode:t16_p4_fused_r3_cs4_pf1_px1": {
        "module": "cake_kimi_k3_fp8_projection_9f4331dc1b45870d0d95",
        "defines": {},
    },
    "decode:t16_p4_fused_r3_cs7": {
        "module": "cake_kimi_k3_fp8_projection_b2e2ef1b2a5df0f83b1b",
        "defines": {},
    },
    "decode:t16_p4_pw2": {
        "module": "cake_kimi_k3_fp8_projection_ea8d2ae10f643d1b0341",
        "defines": {},
    },
    "decode:t32_p3_fused_r5_q4": {
        "module": "cake_kimi_k3_fp8_projection_a9484fb9b47a717a2207",
        "defines": {},
    },
    "decode:t32_p3_fused_r5_q4_tso": {
        "module": "cake_kimi_k3_fp8_projection_f65b17236fc8c8df0336",
        "defines": {},
    },
    "decode:t32_p3_fused_res": {
        "module": "cake_kimi_k3_fp8_projection_7518d0ddd0f8eb901098",
        "defines": {},
    },
    "decode:t32_p3_fused_res_tso": {
        "module": "cake_kimi_k3_fp8_projection_5dbbe3415920815e812e",
        "defines": {},
    },
    "decode:t32_p4": {
        "module": "cake_kimi_k3_fp8_projection_5a04cd74fb0cbb150e8f",
        "defines": {},
    },
    "decode:t32_p4_cs3": {
        "module": "cake_kimi_k3_fp8_projection_974015cffe52e2f67773",
        "defines": {},
    },
    "decode:t32_p4_cs4_pf1": {
        "module": "cake_kimi_k3_fp8_projection_7c0c2078f3110d338d76",
        "defines": {},
    },
    "decode:t32_p4_cs8_xp": {
        "module": "cake_kimi_k3_fp8_projection_e796bf37781af8d66734",
        "defines": {},
    },
    "decode:t32_p4_tso": {
        "module": "cake_kimi_k3_fp8_projection_fcc485f4025b5d964562",
        "defines": {},
    },
    "decode:t32_p5_c16": {
        "module": "cake_kimi_k3_fp8_projection_6965aeafa116c7f6ef89",
        "defines": {},
    },
    "decode:t32_p5_c16_pf2": {
        "module": "cake_kimi_k3_fp8_projection_e985ccc1acc9a1b0c018",
        "defines": {},
    },
    "decode:t32_p5_c16_pf2_tso": {
        "module": "cake_kimi_k3_fp8_projection_b658c4971aa78e9c9165",
        "defines": {},
    },
    "decode:t32_p5_c16_tso": {
        "module": "cake_kimi_k3_fp8_projection_021cc0171d570f358807",
        "defines": {},
    },
    "decode:t64_p1_fused_w16_e2_r3_xh_qe_q4_cs6_pf1_xp": {
        "module": "cake_kimi_k3_fp8_projection_f01f38424f66eefb47b0",
        "defines": {},
    },
    "decode:t64_p2": {
        "module": "cake_kimi_k3_fp8_projection_ac0039f3b84b450bea6d",
        "defines": {},
    },
    "decode:t64_p2_fused_r2_xh_q8": {
        "module": "cake_kimi_k3_fp8_projection_6f14858293e3d9f61c93",
        "defines": {},
    },
    "decode:t64_p2_fused_r2_xh_q8_tso": {
        "module": "cake_kimi_k3_fp8_projection_3e38b6f34453a3f8849d",
        "defines": {},
    },
    "decode:t64_p2_fused_res": {
        "module": "cake_kimi_k3_fp8_projection_88eaaccbac79ef8cf184",
        "defines": {},
    },
    "decode:t64_p2_fused_res_dpf": {
        "module": "cake_kimi_k3_fp8_projection_93fd6a71010eb8fb34f5",
        "defines": {},
    },
    "decode:t64_p2_fused_res_tso": {
        "module": "cake_kimi_k3_fp8_projection_8cceae711ab4461f4135",
        "defines": {},
    },
    "decode:t64_p2_fused_res_tso_dpf": {
        "module": "cake_kimi_k3_fp8_projection_cf10903c711b1dd4f7af",
        "defines": {},
    },
    "decode:t64_p2_fused_w16_r2_xh_q8_cs4_xp": {
        "module": "cake_kimi_k3_fp8_projection_ae66e1c9615449d1921b",
        "defines": {},
    },
    "decode:t64_p2_fused_w16_r3_xh_qe_q4_cs6a": {
        "module": "cake_kimi_k3_fp8_projection_ceeb47741a53f199cb00",
        "defines": {},
    },
    "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_pi2": {
        "module": "cake_kimi_k3_fp8_projection_9265c88bf850d89b0c01",
        "defines": {},
    },
    "decode:t64_p2_fused_w16_r3_xh_qe_q4_pf4_px1_tso_pi2": {
        "module": "cake_kimi_k3_fp8_projection_c9822f9e834e3240b188",
        "defines": {},
    },
    "decode:t64_p3_a2_e2_cs3": {
        "module": "cake_kimi_k3_fp8_projection_75e18bf3c6d16aa5ba9a",
        "defines": {},
    },
    "decode:t64_p3_cs8": {
        "module": "cake_kimi_k3_fp8_projection_46f4c0a44682680afa00",
        "defines": {},
    },
    "decode:t64_p3_cs8_xp": {
        "module": "cake_kimi_k3_fp8_projection_fe01dd81c571f1f74124",
        "defines": {},
    },
    "decode:t64_p3_e2_cs3": {
        "module": "cake_kimi_k3_fp8_projection_8b6f92bd50579503f035",
        "defines": {},
    },
    "decode:t64_p4": {
        "module": "cake_kimi_k3_fp8_projection_a8222cc117f456537bdc",
        "defines": {},
    },
    "decode:t64_p4_a2_cs2a": {
        "module": "cake_kimi_k3_fp8_projection_7abdff4851061bffe5cb",
        "defines": {},
    },
    "decode:t64_p4_cs2a": {
        "module": "cake_kimi_k3_fp8_projection_67b79a99dbca18833ff9",
        "defines": {},
    },
    "decode:t64_p4_pf1": {
        "module": "cake_kimi_k3_fp8_projection_504b13e2236557baa266",
        "defines": {},
    },
    "decode:t64_p4_pf1_tso": {
        "module": "cake_kimi_k3_fp8_projection_a2e46d7b502767bca64c",
        "defines": {},
    },
    "decode:t64_p4_pw2": {
        "module": "cake_kimi_k3_fp8_projection_d1ac3a99961e78557b6f",
        "defines": {},
    },
    "decode:t64_p4_pw2_tso": {
        "module": "cake_kimi_k3_fp8_projection_39fd3fc41b3d7ca27508",
        "defines": {},
    },
    "decode:t64_p4_tso": {
        "module": "cake_kimi_k3_fp8_projection_71c5eadf708248a0d43c",
        "defines": {},
    },
    "gemm": {
        "module": "cake_kimi_k3_fp8_projection_4be8750b7ef4b9419aaf",
        "defines": {},
    },
    "gemm_rstaged": {
        "module": "cake_kimi_k3_fp8_projection_7a9e8f32c95b21cd3ba3",
        "defines": {},
    },
    "gemm_rstaged_g64_k128": {
        "module": "cake_kimi_k3_fp8_projection_c22cadffa1705cdcb3b8",
        "defines": {},
    },
    "gemm_rstaged_k128": {
        "module": "cake_kimi_k3_fp8_projection_82375137e249e01d15c2",
        "defines": {},
    },
    "gemm_rstaged_n192": {
        "module": "cake_kimi_k3_fp8_projection_2ec639014a8fa1272b0a",
        "defines": {},
    },
    "gemm_rstaged_n192_k128": {
        "module": "cake_kimi_k3_fp8_projection_76f152b19e7a843ba820",
        "defines": {},
    },
    "gemm_rstaged_n192_k128_cs2": {
        "module": "cake_kimi_k3_fp8_projection_c2b4ed9c6a005c71cf4b",
        "defines": {},
    },
    "gemm_rstaged_n192_k128_pdlb": {
        "module": "cake_kimi_k3_fp8_projection_d4d43a9ba6f62cb3d829",
        "defines": {},
    },
    "gemm_rstaged_n192_k128_st": {
        "module": "cake_kimi_k3_fp8_projection_11cbad4e31342b47dc08",
        "defines": {},
    },
    "gemm_rstaged_n384_k128": {
        "module": "cake_kimi_k3_fp8_projection_4da95d45aada36775de5",
        "defines": {},
    },
    "gemm_tstore": {
        "module": "cake_kimi_k3_fp8_projection_5196dc524176f72bfdb6",
        "defines": {},
    },
    "gemm_tstore_g64_k128": {
        "module": "cake_kimi_k3_fp8_projection_e319dfda1d4453cff859",
        "defines": {},
    },
    "gemm_tstore_k128": {
        "module": "cake_kimi_k3_fp8_projection_baa922f3fad64f788380",
        "defines": {},
    },
    "gemm_tstore_k128_gs": {
        "module": "cake_kimi_k3_fp8_projection_3467fab6a69e46ade21d",
        "defines": {},
    },
    "gemm_tstore_k128_pf1": {
        "module": "cake_kimi_k3_fp8_projection_0f46cbbf5011f4371f92",
        "defines": {},
    },
    "gemm_tstore_k128_pf2": {
        "module": "cake_kimi_k3_fp8_projection_28d3f834bafa2bf36d00",
        "defines": {},
    },
    "gemm_tstore_k128_s2_ts4": {
        "module": "cake_kimi_k3_fp8_projection_f13cdc4b7875b5c1ca74",
        "defines": {},
    },
    "gemm_tstore_k128_s2_ts4_pf2": {
        "module": "cake_kimi_k3_fp8_projection_718b9dd8552d49e4f782",
        "defines": {},
    },
    "gemm_tstore_k128_skf": {
        "module": "cake_kimi_k3_fp8_projection_66015597a7e7c29045d7",
        "defines": {},
    },
    "gemm_tstore_n192_k128": {
        "module": "cake_kimi_k3_fp8_projection_09ef673497e9adb01b7e",
        "defines": {},
    },
    "gemm_tstore_n192_k128_cs2": {
        "module": "cake_kimi_k3_fp8_projection_220aae40317b8e398117",
        "defines": {},
    },
    "gemm_tstore_n192_k128_gs": {
        "module": "cake_kimi_k3_fp8_projection_cc335af0b89da542ea91",
        "defines": {},
    },
    "gemm_tstore_n192_k128_pf1": {
        "module": "cake_kimi_k3_fp8_projection_bac8891ceedeac1efaf2",
        "defines": {},
    },
    "gemm_tstore_n192_k128_pf2": {
        "module": "cake_kimi_k3_fp8_projection_a1ee4a9c1f1f7ea1917c",
        "defines": {},
    },
    "gemm_tstore_n192_k128_skf": {
        "module": "cake_kimi_k3_fp8_projection_35deb0e5eea243f77d81",
        "defines": {},
    },
    "gemm_tstore_n192_k128_skf_pf1": {
        "module": "cake_kimi_k3_fp8_projection_fdb1ab7e8ccbb2e8e1d8",
        "defines": {},
    },
    "gemm_tstore_n192_k128_st": {
        "module": "cake_kimi_k3_fp8_projection_81f8643b8577aafe1a7d",
        "defines": {},
    },
    "gemm_tstore_n192_pf1": {
        "module": "cake_kimi_k3_fp8_projection_e9af5f2d6bdaec79f30a",
        "defines": {},
    },
    "gemm_tstore_n192_pf2": {
        "module": "cake_kimi_k3_fp8_projection_fcd2cccaffa526675b6d",
        "defines": {},
    },
    "gemm_tstore_n192_s2_ts4": {
        "module": "cake_kimi_k3_fp8_projection_ada93e51754d04322ff1",
        "defines": {},
    },
    "gemm_tstore_n384_k128": {
        "module": "cake_kimi_k3_fp8_projection_43d3d6683770c0bf8bad",
        "defines": {},
    },
    "gemm_tstore_pf1": {
        "module": "cake_kimi_k3_fp8_projection_8becf9f889fc4a657c22",
        "defines": {},
    },
    "gemm_tstore_pf2": {
        "module": "cake_kimi_k3_fp8_projection_20ea24437d297722619d",
        "defines": {},
    },
    "gemm_tstore_s2_ts4": {
        "module": "cake_kimi_k3_fp8_projection_ee2e0fe9754d301492e2",
        "defines": {},
    },
    "gemm_tstore_s2_ts4_pf4": {
        "module": "cake_kimi_k3_fp8_projection_5660a6b1b752dc71dc15",
        "defines": {},
    },
    "gemm_tstore_skf": {
        "module": "cake_kimi_k3_fp8_projection_53557e910bed0b361877",
        "defines": {},
    },
    "gemm_tstore_skf_pf1": {
        "module": "cake_kimi_k3_fp8_projection_cced2467167c42643d15",
        "defines": {},
    },
    "gemm_tstore_skf_pf2": {
        "module": "cake_kimi_k3_fp8_projection_96a3edd7d4db71c390df",
        "defines": {},
    },
    "gemm_tstore_skp4": {
        "module": "cake_kimi_k3_fp8_projection_ea0109448a87bf68aebb",
        "defines": {},
    },
    "quant:u1": {
        "module": "cake_kimi_k3_fp8_projection_ca4fbacb9d82722ff4ce",
        "defines": {"QUANT_UNITS": 1},
    },
    "quant:u2": {
        "module": "cake_kimi_k3_fp8_projection_ca4fbacb9d82722ff4ce",
        "defines": {"QUANT_UNITS": 2},
    },
    "quant:u4": {
        "module": "cake_kimi_k3_fp8_projection_ca4fbacb9d82722ff4ce",
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
    epi_chunk: int = 0,
    pf: int = 0,
    mc: int = 1,
    tstore: bool = False,
    dpf: bool = False,
    pfx: int = 0,
    pfi: int = 0,
    qwarps: int = 8,
    xbh: bool = False,
    qer: bool = False,
    xp: bool = False,
    xq_stages: int = 0,
    acc_bufs: int = 1,
    epi_groups: int = 1,
    pw: int = 0,
) -> str:
    key = f"decode:t{int(tok)}_p{int(stages)}"
    if fused:
        key += "_fused"
        if int(qwarps) != 8:
            key += f"_w{int(qwarps)}"  # round 6 continuation 8 (lever QW16): quantizing warps of the fused instance (table key ``qwarps``; 8 = default)
    if int(acc_bufs) != 1:
        key += f"_a{int(acc_bufs)}"  # round 8 (lever ACC_BUFS): TMEM accumulator sets of the instance (table key ``acc_bufs``; after the quantizing-warp count, before ``_res``: the Cake symbol order)
    if int(epi_groups) != 1:
        key += f"_e{int(epi_groups)}"  # round 8 (lever EPI_GROUPS): epilogue warp groups (table key ``epi_groups``; 64/128-token tiles only)
    if resident:
        key += "_res"
    if xb_stages:
        key += f"_r{int(xb_stages)}"
        if xbh:
            key += "_xh"  # round 6 continuation 9 (lever XBH): half-slot BF16 ring (table key ``xbh``; per-128-K-block loads and releases)
            if qer:
                key += "_qe"  # round 6 continuation 10 (lever QER): the quantizing warps release each half slot right after their register loads (table key ``qer``)
    if xb_stages and xq_stages:
        key += f"_x{int(xq_stages)}"  # round 4 (lever F): the FP8 token tiles stream through their own ring (table key ``xq_stages``; needs the BF16 ring)
    if qlanes != 16:
        key += f"_q{int(qlanes)}"
    if epi_chunk and int(epi_chunk) != min(32, int(tok)):
        key += f"_c{int(epi_chunk)}"  # round 6: epilogue staging rows per flush (a 16-row chunk frees SMEM for a 5th stage)
    if int(csplit) > 1:
        key += f"_cs{int(csplit)}"  # round 5: K split across the CTAs of one cluster, DSM partial exchange
        if cs_alias:
            key += "a"  # the exchange inbox aliases the dead pipeline stages (one round; one work item per CTA)
    if int(mc) > 1:
        key += f"_mc{int(mc)}"  # round 6 (lever M): the C m tiles of one N tile run as a cluster and share the W stage (TMA multicast)
    if int(pf) > 0:
        key += f"_pf{int(pf)}"  # round 6 (lever P): weight tiles prefetched into L2 pf stages ahead of their TMA load
    if int(pw) > 0:
        key += f"_pw{int(pw)}"  # round 8 (lever PW; adopted round 11 C20 on the sm_103a 192,2,64 cell): the first work item's W / SFW stream is L2-prefetched pw stages deep before the PDL wait (table key ``pw``; after ``_pf<D>``, before ``_px<D>`` / ``_tso``: the Cake symbol order; no data-path, launch-argument or SMEM effect)
    if int(pfx) > 0:
        key += f"_px{int(pfx)}"  # round 6 next loop (lever PX): the BF16 token tile prefetched into L2 pfx stages ahead of its TMA load
    if tstore:
        key += "_tso"  # round 6 (lever E1): the split-1 epilogue stores BF16 through TMA (16-byte-aligned output views)
    if dpf:
        key += "_dpf"  # round 6 (lever Q1; adopted round 8): the tensor-map descriptors are prefetched in the kernel prelude before the first TMA issue (table key ``dpf``; after ``_tso``, before ``_pi<n>``: the Cake symbol order; no data-path or launch-argument effect)
    if int(pfi) > 0:
        key += f"_pi{int(pfi)}"  # round 6 continuation 7 (lever PI-W): the load warp prefetches the next work item's first W/SFW tiles into L2 (needs pf > 0)
    if xp and int(csplit) > 1:
        key += "_xp"  # round 7 (lever XP): the cluster split-K exchange issues its DSM copies from one lane per peer in parallel (table key ``xp``)
    return key


def gemm_kernel_key(base: str, pf: int = 0) -> str:
    """``gemm`` / ``gemm_tstore`` / ``gemm_rstaged`` (with the ``_n192`` / ``_n384`` / ``_g64`` / ``_k128`` /
    ``_cs<S>[r]`` / ``_st`` / ``_gs`` / ``_s<n>`` / ``_ts<n>`` / ``_sk`` / ``_skf`` / ``_pdlb`` instance suffixes, in that stem order; ``_n384`` /
    ``_g64`` are the round-10 384-wide pair tile (table key ``gemm_bn`` 384) and 64-row-tile raster group (table key
    ``gemm_gm``); ``_k128`` is the 128-K operand hand-off form, table key ``gemm_bk``; ``_cs<S>[r]`` the cluster
    split-K form of the ``_n192_k128`` TMA-store / staged-register programs, table key ``gemm_cs``, round 10; ``_st`` the
    static one-tile form of the same programs (one tile per CTA pair, no CLC work ring), table key ``gemm_static``, round 11; ``_gs``
    the static grid-stride form of the ``_k128`` / ``_n192_k128`` TMA-store programs (pair p owns tiles p + i P' over 2 P' CTAs, no
    CLC work ring), table key ``gemm_gs``, round 11;
    ``_s<n>`` / ``_ts<n>`` the operand-stage / staging-slot form of the TMA-store programs, table keys
    ``gemm_stages`` / ``gemm_ts_slots``, round 9; ``_pdlb`` the PDL-wait-after-weights form of the TMA-store /
    staged-register programs, table key ``gemm_pdlb``, round 11, closing the stem) plus ``_pf<D>`` when the shape's table row prefetches the weight
    tiles ``D`` stages ahead (round 6, lever GP; table key ``gemm_pf``)."""
    return base + (f"_pf{int(pf)}" if int(pf) > 0 else "")


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
