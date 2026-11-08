# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-scale fp8 GEMM (the class ``gemm.e4m3.kblock128`` behind
:func:`flashinfer.gemm_fp8_nt_groupwise`): one TVM-FFI object per (knobs, persistent grid, output format) with dynamic
M, N, K; cache module ``batch_invariant_gemm_fp8_blockscale`` in FlashInfer's CuTe-DSL disk cache (its own module: an edit to
the block-scale body leaves the dense and the block-scaled modules' objects valid and vice versa). Operand conventions:
A (M, K) and B (N, K) e4m3 with K contiguous (the NT operator's row-major weight), SFA (M, K / 128) and SFB
(ceil(N / 128), K / 128) fp32 with any strides (the ``K`` and ``MN`` scale layouts are one kernel), C (M, N) N-major in
the output format; fp32 accumulate, the block scales applied by the epilogue warps once per K block.
"""

from __future__ import annotations

import dataclasses
import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..generator import Knobs
from ..spec import BLOCK_K
from .blockscale_fp8 import IN_FMT, BlockScaleFp8Gemm
from .operands import ALIGN, IN_DT, OUT_DT

MODULE = "batch_invariant_gemm_fp8_blockscale"


def kernel_name(cfg: Knobs, clusters: int, out_fmt: str, native_k: bool = False) -> str:
    """``clusters`` is the compiled persistent grid in clusters (the knobs' own ``grid_clusters`` when set, else the launch
    device's co-resident count), keyed once as ``_g<clusters>``; ``native_k`` (the K-native scale staging) as ``_nk``."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"e4m3bs_{out_fmt}_{knobs.describe()}_g{clusters}" + ("_nk" if native_k else "")
    )


@functools.cache
def get_compiled_fp8bs(
    cfg: Knobs,
    clusters: int,
    out_fmt: str = "bf16",
    family: str = "",
    native_k: bool = False,
):
    """``kernel(a3, b3, sfa, sfb, c3)``: A as (M, K, 1), B as (N, K, 1), the scale tensors as (rows, K / 128) views with
    their own strides, C as (M, N, 1). ``clusters``, ``family`` and ``native_k`` (the K-native scale staging: the
    scale tensors read as they are with 16-byte-aligned bases) are part of the cache key as for the dense kernels
    (``blocks/compile.py``): placements, never the bits."""
    assert cfg.well_formed(IN_FMT), cfg
    in_dt, al = IN_DT[IN_FMT], ALIGN[IN_FMT]
    out_dt, oal = OUT_DT[out_fmt], ALIGN[out_fmt]
    grid = cfg.grid_clusters or clusters
    assert grid > 0, (cfg, clusters)
    name = kernel_name(cfg, grid, out_fmt, native_k)

    def compile_fn():
        # K a multiple of 128 (one whole scale block per smem stage), the operand rows 16-byte aligned; the scale tensors
        # with free strides (the kernel's producer warp gathers them element by element, or four at a time from a
        # 16-byte-aligned base with the K-native staging)
        sal = 16 if native_k else 4
        sym_m = cute.sym_int(32)
        sym_n = cute.sym_int(32, divisibility=oal)
        sym_k = cute.sym_int(32, divisibility=BLOCK_K)
        sym_kb, sym_nb = cute.sym_int(32), cute.sym_int(32)
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_k, 1),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, 1),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        sfa_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32,
            (sym_m, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=sal,
        )
        sfb_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32,
            (sym_nb, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=sal,
        )
        c_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_n, 1),
            (cute.sym_int64(divisibility=oal), 1, cute.sym_int64(divisibility=oal)),
            assumed_align=16,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            BlockScaleFp8Gemm(cfg, out_fmt, native_k),
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            c_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
