# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-scaled GEMM (the MXFP8 classes): one TVM-FFI object per (knobs, chunks, persistent grid,
formats) with dynamic M, N, K; cache module ``batch_invariant_gemm_blockscaled`` in FlashInfer's CuTe-DSL disk cache (its
own module: an edit to the block-scaled body leaves the dense module's objects valid and vice versa). Operand conventions:
A (M, K) and B (N, K) e4m3 with K contiguous (the block-scaled instruction takes K-major operands only), SFA / SFB flat
uint8 buffers in the 128x4 swizzled layout, C (M, N) N-major -- or, under the swapped decode record, the kernel's A is the
weight, its B the tokens and its C the M-major (N, M) view of the output -- fp32 accumulate.
"""

from __future__ import annotations

import dataclasses
import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..generator import Knobs
from ..spec import SF_ATOM_KBLOCKS, SF_ATOM_ROWS
from .blockscaled import BlockScaledGemm
from .operands import ALIGN, IN_DT, OUT_DT

MODULE = "batch_invariant_gemm_blockscaled"
SF_BUFFER_DIVISIBILITY = (
    SF_ATOM_ROWS * SF_ATOM_KBLOCKS
)  # a scale-factor buffer is a whole number of 512-byte atoms


def kernel_name(
    cfg: Knobs, chunks: int, clusters: int, in_fmt: str, out_fmt: str
) -> str:
    """``clusters`` is the compiled persistent grid in clusters (the knobs' own ``grid_clusters`` when set, else the launch
    device's co-resident count), keyed once as ``_g<clusters>``; the knobs' ``describe()`` carries the in-cluster split
    factor, the swapped decode record and the in-CTA chunk fold."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"{in_fmt}_{out_fmt}_bk_C{chunks}_{knobs.describe()}_g{clusters}"
    )


@functools.cache
def get_compiled_blockscaled(
    cfg: Knobs,
    chunks: int,
    clusters: int,
    in_fmt: str,
    out_fmt: str = "bf16",
    family: str = "",
):
    """``kernel(a3, b3, sfa, sfb, c3)`` for the format's class (``mxfp8``: the sm_100 / sm_103 instruction; ``mxfp8_k64``: the
    sm_107 one). The direct stores -- the unsplit class (L = 1), the in-cluster split (``cfg.cluster_k = chunks``: whole
    operands, the chunk fold in the cluster) and the in-CTA chunk fold (``cfg.cta_chunks``: the split class's ``(rows, Kc,
    C)`` operand views, the output with L = 1) -- convert in the epilogue; the fixed-chunk split (batch index = K chunk)
    stores fp32 partials that the dense module's fixup folds and converts. ``clusters`` and ``family`` are part of the cache
    key as for the dense kernels (``blocks/compile.py``): placements, never the bits."""
    in_dt, al = IN_DT[in_fmt], ALIGN[in_fmt]
    swap_ab = cfg.swap_ab
    in_cluster = cfg.cluster_k > 1
    in_cta = cfg.cta_chunks and chunks > 1
    assert not in_cluster or chunks == cfg.cluster_k, (cfg.describe(), chunks)
    direct = chunks == 1 or in_cluster or in_cta
    assert not (swap_ab and not direct), "the swapped decode record stores directly"
    out_dt = OUT_DT[out_fmt] if direct else cutlass.Float32
    oal = ALIGN[out_fmt] if direct else ALIGN["f32"]
    L = (
        1 if in_cluster else chunks
    )  # the operands' batch extent: the K chunk of the split views
    Lc = 1 if in_cta else L
    grid = cfg.grid_clusters or clusters
    assert grid > 0, (cfg, clusters)
    name = kernel_name(cfg, chunks, grid, in_fmt, out_fmt)

    def compile_fn():
        # K elements per row, a multiple of 16 (K is a multiple of 32: one whole scale block); the row and chunk strides
        # are 16-byte multiples too. The kernel's N extent is the output's contiguous extent (aligned), except under the
        # swapped record, where it is the token count (any value) and the kernel's M extent (the weights' N) is the aligned one.
        sym_m = cute.sym_int(32)
        sym_n = cute.sym_int(32) if swap_ab else cute.sym_int(32, divisibility=oal)
        sym_k = cute.sym_int(32, divisibility=al)
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_k, L),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, L),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        sfa_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
            (1,),
            assumed_align=16,
        )
        sfb_fake = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(32, divisibility=SF_BUFFER_DIVISIBILITY),),
            (1,),
            assumed_align=16,
        )
        if swap_ab:
            c_fake = cute.runtime.make_fake_tensor(
                out_dt,
                (cute.sym_int(32, divisibility=oal), sym_n, Lc),
                (1, cute.sym_int64(divisibility=oal), cute.sym_int64(divisibility=oal)),
                assumed_align=16,
            )
        else:
            c_fake = cute.runtime.make_fake_tensor(
                out_dt,
                (sym_m, sym_n, Lc),
                (cute.sym_int64(divisibility=oal), 1, cute.sym_int64(divisibility=oal)),
                assumed_align=16,
            )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            BlockScaledGemm(cfg, in_fmt, out_fmt if direct else "f32", L),
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
