# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-scaled GEMM (the MXFP8, NVFP4 and MXFP4 classes): one TVM-FFI object per (knobs, chunks,
persistent grid, formats) with dynamic M, N, K; cache module ``batch_invariant_gemm_blockscaled`` in FlashInfer's CuTe-DSL
disk cache (its own module: an edit to the block-scaled body leaves the dense module's objects valid and vice versa). Operand
conventions: A (M, K) and B (N, K) e4m3 with K contiguous (the block-scaled instruction takes K-major operands only), or for
the 4-bit formats the byte-packed (M, K / 2) and (N, K / 2) uint8 tensors; SFA / SFB flat uint8 buffers in the 128x4 swizzled
layout; C (M, N) N-major -- or, under the swapped decode record, the kernel's A is the weight, its B the tokens and its C
the M-major (N, M) view of the output -- fp32 accumulate; the 4-bit classes' direct stores take ``alpha`` as a 1-element
fp32 tensor.
"""

from __future__ import annotations

import dataclasses
import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..generator import Knobs
from ..spec import ALPHA_GIVEN, ELEM_BITS_OF, SF_ATOM_KBLOCKS, SF_ATOM_ROWS
from .blockscaled import BlockScaledGemm, ScaledBlockScaledGemm
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
    """``kernel(a3, b3, sfa, sfb, c3)`` for the format's class (``mxfp8`` / ``nvfp4`` / ``mxfp4``: the sm_100 / sm_103
    instructions; ``mxfp8_k64`` / ``nvfp4_k128`` / ``mxfp4_k128``: the sm_107 ones), ``kernel(a3, b3, sfa, sfb, c3, alpha)``
    for the 4-bit classes' direct stores (the alpha chain). The direct stores -- the unsplit class (L = 1), the in-cluster split (``cfg.cluster_k = chunks``: whole
    operands, the chunk fold in the cluster) and the in-CTA chunk fold (``cfg.cta_chunks``: the split class's ``(rows, Kc,
    C)`` operand views, the output with L = 1) -- convert in the epilogue; the fixed-chunk split (batch index = K chunk)
    stores fp32 partials that the dense module's fixup folds and converts. ``clusters`` and ``family`` are part of the cache
    key as for the dense kernels (``blocks/compile.py``): placements, never the bits."""
    in_dt, al = IN_DT[in_fmt], ALIGN[in_fmt]
    packed = (
        ELEM_BITS_OF[in_fmt] == 4
    )  # the 4-bit operands arrive byte-packed: K / 2 bytes per row
    fake_dt = cutlass.Uint8 if packed else in_dt
    al_k = (
        al // 2 if packed else al
    )  # the operand row's divisibility in its own units (16 bytes either way)
    swap_ab = cfg.swap_ab
    in_cluster = cfg.cluster_k > 1
    in_cta = cfg.cta_chunks and chunks > 1
    assert not in_cluster or chunks == cfg.cluster_k, (cfg.describe(), chunks)
    direct = chunks == 1 or in_cluster or in_cta
    scaled = (
        in_fmt in ALPHA_GIVEN and direct
    )  # the partials of the workspace split are unscaled (the fixup applies alpha)
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
        sym_k = cute.sym_int(32, divisibility=al_k)
        a_fake = cute.runtime.make_fake_tensor(
            fake_dt,
            (sym_m, sym_k, L),
            (cute.sym_int64(divisibility=al_k), 1, cute.sym_int64(divisibility=al_k)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            fake_dt,
            (sym_n, sym_k, L),
            (cute.sym_int64(divisibility=al_k), 1, cute.sym_int64(divisibility=al_k)),
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
        body_out = out_fmt if direct else "f32"
        if scaled:
            alpha_fake = cute.runtime.make_fake_tensor(
                cutlass.Float32, (1,), (1,), assumed_align=4
            )
            return cute.compile(
                ScaledBlockScaledGemm(cfg, in_fmt, body_out, L),
                a_fake,
                b_fake,
                sfa_fake,
                sfb_fake,
                c_fake,
                alpha_fake,
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        return cute.compile(
            BlockScaledGemm(cfg, in_fmt, body_out, L),
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
