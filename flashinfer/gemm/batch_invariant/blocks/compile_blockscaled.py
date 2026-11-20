# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-scaled GEMM (the MXFP8, NVFP4 and MXFP4 classes): one TVM-FFI object per (compile-time
knobs, split or not, formats) with dynamic M, N, K; cache module ``batch_invariant_gemm_blockscaled`` in FlashInfer's CuTe-DSL
disk cache (its own module: an edit to the block-scaled body leaves the dense module's objects valid and vice versa). The
persistent grid, the swizzle size and the raster order are Int32 arguments of every object (``blocks/scheduler.py``), the
split classes' chunk count the operands' dynamic batch extent (``blocks/compile.py``). Operand
conventions: A (M, K) and B (N, K) e4m3 with K contiguous (the block-scaled instruction takes K-major operands only), or for
the 4-bit formats the byte-packed (M, K / 2) and (N, K / 2) uint8 tensors; SFA / SFB flat uint8 buffers in the 128x4 swizzled
layout; C (M, N) N-major -- or, under the swapped decode record, the kernel's A is the weight, its B the tokens and its C
the M-major (N, M) view of the output -- fp32 accumulate; the 4-bit classes' direct stores take ``alpha`` as a 1-element
fp32 tensor. :func:`compile_blockscaled` is the shared compile closure: the mixed classes' module
(``compile_blockscaled_mixed.py``) runs it with its own plans, each operand in its own storage.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..generator import Knobs
from ..spec import ALPHA_GIVEN, SF_ATOM_KBLOCKS, SF_ATOM_ROWS
from .blockscaled import PLANS, BlockScaledGemm, ScaledBlockScaledGemm
from .compile import (
    GRID_PLACEHOLDER,
    RASTER_PLACEHOLDER,
    SWIZZLE_PLACEHOLDER,
    object_record,
)
from .operands import ALIGN, OUT_DT

MODULE = "batch_invariant_gemm_blockscaled"
SF_BUFFER_DIVISIBILITY = (
    SF_ATOM_ROWS * SF_ATOM_KBLOCKS
)  # a scale-factor buffer is a whole number of 512-byte atoms


def kernel_name(
    cfg: Knobs,
    chunks: int,
    in_fmt: str,
    out_fmt: str,
    enable_pdl: bool = False,
) -> str:
    """``C1``: a batch extent of one (the unsplit class, the in-cluster split whose factor the knobs' ``_ck`` carries);
    ``CL``: the split classes with the chunk count as the operands' dynamic batch extent; the knobs' ``describe_object()``
    carries the swapped decode record and the in-CTA chunk fold; the dependent-launch variant is keyed ``_pdl``."""
    dyn_l = chunks > 1 and cfg.cluster_k == 1
    return jit.sanitize_symbol_name(
        f"{in_fmt}_{out_fmt}_bk_C{'L' if dyn_l else 1}_{object_record(cfg).describe_object()}"
        + ("_pdl" if enable_pdl else "")
    )


def object_chunks(cfg: Knobs, chunks: int) -> int:
    """The chunk count an object is compiled for: the in-cluster split's factor (static), else 1 (unsplit) or 2 (any
    split: the count is the operands' dynamic batch extent)."""
    if cfg.cluster_k > 1:
        assert chunks == cfg.cluster_k, (cfg.describe(), chunks)
        return chunks
    return 2 if chunks > 1 else 1


def get_compiled_blockscaled(
    cfg: Knobs,
    chunks: int,
    in_fmt: str,
    out_fmt: str = "bf16",
    family: str = "",
):
    """``kernel(a3, b3, sfa, sfb, c3, grid, swizzle, raster_m)`` for the format's class (``mxfp8`` / ``nvfp4`` / ``mxfp4``:
    the sm_100 / sm_103 instructions; ``mxfp8_k64`` / ``nvfp4_k128`` / ``mxfp4_k128``: the sm_107 ones), ``kernel(a3, b3,
    sfa, sfb, c3, alpha, grid, swizzle, raster_m)`` for the 4-bit classes' direct stores (the alpha chain). The direct
    stores -- the unsplit class (L = 1), the in-cluster split (``cfg.cluster_k = chunks``: whole operands, the chunk fold
    in the cluster) and the in-CTA chunk fold (``cfg.cta_chunks``: the split class's ``(rows, Kc, C)`` operand views, the
    output with L = 1) -- convert in the epilogue; the fixed-chunk split (batch index = K chunk) stores fp32 partials that
    the dense module's fixup folds and converts. The trailing Int32 arguments and ``family`` are as for the dense kernels
    (``blocks/compile.py``): placements, never the bits."""
    return _compiled_blockscaled(
        object_record(cfg), object_chunks(cfg, chunks), in_fmt, out_fmt, family
    )


@functools.cache
def _compiled_blockscaled(
    cfg: Knobs, chunks: int, in_fmt: str, out_fmt: str, family: str
):
    return compile_blockscaled(cfg, chunks, in_fmt, out_fmt, MODULE)


def compile_blockscaled(
    cfg: Knobs,
    chunks: int,
    in_fmt: str,
    out_fmt: str,
    module: str,
    plan_cls=None,
    enable_pdl: bool = False,
):
    """The compile closure behind :func:`get_compiled_blockscaled`, in cache module ``module`` with the record's plan class
    (default: the format's in ``PLANS``; the mixed classes pass their own, the swapped records' with the operand formats
    exchanged) and, with ``enable_pdl``, the kernel compiled as a programmatic dependent launch (a placement: the same
    bits). Each operand's fake tensor takes the kernel-side operand's storage: its own dtype for an 8-bit element,
    bytes (two elements each) for a 4-bit one; a row's extent and strides are 16-byte multiples in either."""
    cfg = object_record(cfg)
    plan_cls = plan_cls or PLANS[in_fmt]
    a_packed, b_packed = plan_cls.A_DT.width == 4, plan_cls.B_DT.width == 4
    a_fake_dt = cutlass.Uint8 if a_packed else plan_cls.A_DT
    b_fake_dt = cutlass.Uint8 if b_packed else plan_cls.B_DT
    al = ALIGN[in_fmt]
    al_k = (
        al // 2 if (a_packed or b_packed) else al
    )  # the operand row's divisibility in its own units (16 bytes either way)
    swap_ab = cfg.swap_ab
    in_cluster = cfg.cluster_k > 1
    in_cta = cfg.cta_chunks and chunks > 1
    assert not in_cluster or chunks == cfg.cluster_k, (cfg.describe(), chunks)
    direct = chunks == 1 or in_cluster or in_cta
    scaled = (
        in_fmt in ALPHA_GIVEN and direct
    )  # the partials of the workspace split are unscaled (the fixup applies alpha)
    # a swapped record in the split classes writes its chunk partials like the dense one: the kernel's (N, M, C) view of the
    # fp32 workspace with the kernel's M contiguous (the same M-major store as its direct output), then the fixup
    assert not (swap_ab and (in_cluster or in_cta)), (
        "the swapped record splits through the workspace only"
    )
    out_dt = OUT_DT[out_fmt] if direct else cutlass.Float32
    oal = ALIGN[out_fmt] if direct else ALIGN["f32"]
    name = kernel_name(cfg, chunks, in_fmt, out_fmt, enable_pdl)

    def compile_fn(**compile_options):
        # the operands' batch extent: the K chunk of the split views, a runtime value (1 for the unsplit class and the
        # in-cluster split, which folds whole operands)
        L = cute.sym_int(32) if (chunks > 1 and not in_cluster) else 1
        Lc = 1 if in_cta else L
        # K elements per row, a multiple of 16 (K is a multiple of 32: one whole scale block); the row and chunk strides
        # are 16-byte multiples too. The kernel's N extent is the output's contiguous extent (aligned), except under the
        # swapped record, where it is the token count (any value) and the kernel's M extent (the weights' N) is the aligned one.
        # The mixed classes' operands have different K extents in their own units (K bytes against K / 2): one symbol each.
        sym_m = cute.sym_int(32)
        sym_n = cute.sym_int(32) if swap_ab else cute.sym_int(32, divisibility=oal)
        sym_ka = cute.sym_int(32, divisibility=al_k)
        sym_kb = cute.sym_int(32, divisibility=al_k) if a_packed != b_packed else sym_ka
        a_fake = cute.runtime.make_fake_tensor(
            a_fake_dt,
            (sym_m, sym_ka, L),
            (cute.sym_int64(divisibility=al_k), 1, cute.sym_int64(divisibility=al_k)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            b_fake_dt,
            (sym_n, sym_kb, L),
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
        tail = (GRID_PLACEHOLDER, SWIZZLE_PLACEHOLDER, RASTER_PLACEHOLDER, stream_fake)
        if scaled:
            alpha_fake = cute.runtime.make_fake_tensor(
                cutlass.Float32, (1,), (1,), assumed_align=4
            )
            return cute.compile(
                ScaledBlockScaledGemm(
                    cfg,
                    in_fmt,
                    body_out,
                    cta_chunks=in_cta,
                    plan_cls=plan_cls,
                    enable_pdl=enable_pdl,
                ),
                a_fake,
                b_fake,
                sfa_fake,
                sfb_fake,
                c_fake,
                alpha_fake,
                *tail,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        return cute.compile(
            BlockScaledGemm(
                cfg,
                in_fmt,
                body_out,
                cta_chunks=in_cta,
                plan_cls=plan_cls,
                enable_pdl=enable_pdl,
            ),
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            c_fake,
            *tail,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(module, name, compile_fn)
