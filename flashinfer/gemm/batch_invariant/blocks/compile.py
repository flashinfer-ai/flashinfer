# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-built GEMM: one TVM-FFI object per (compile-time knobs, B major, split or not, formats)
with dynamic M, N, K; cache module ``batch_invariant_gemm`` in FlashInfer's CuTe-DSL disk cache (both formats share one
source closure). The persistent grid, the swizzle size and the raster order are Int32 arguments of every object
(``blocks/scheduler.py``), as is the split classes' chunk count (the operands' batch extent), so the records of one tile /
cluster bucket share one object (:func:`object_record`) and the launch passes the record's values (:func:`launch_args`).
Operand conventions: A (M, K) K-major, B (K, N) N-major (torch row-major) or K-major (a transposed weight), C (M, N)
N-major, fp32 accumulate; the fp8 kernels take the two per-tensor scales as 1-element fp32 tensors, and the batched
unsplit fp8 kernel has a dynamic batch extent (``bmm_fp8``: the batch is the operands' third mode); the biased bf16
body takes the (N,) bf16 bias as one more operand.
"""

from __future__ import annotations

import dataclasses
import functools

import cutlass
import cutlass.cute as cute

from ....batch_invariant import jit
from ..generator import RASTERS, SWIZZLES, Knobs
from .kernel import BiasBlockGemm, BlockGemm, ScaledBlockGemm
from .operands import ALIGN, IN_DT, OUT_DT, can_implement  # noqa: F401  (re-exported)

MODULE = "batch_invariant_gemm"
#: compile-time values of the Int32 launch arguments (grid, swizzle, raster flag): never compiled in
GRID_PLACEHOLDER = 1
SWIZZLE_PLACEHOLDER = 1
RASTER_PLACEHOLDER = 1


def max_active_clusters(cfg: Knobs, sm_count: int) -> int:
    """The fallback persistent grid in clusters, one CTA per SM: what the launch device can host at most, not what it
    can host at once (a GPC holds whole clusters only, so the co-resident count is below ``SMs // size`` for clusters of
    more than one CTA; ``api.persistent_clusters`` asks the device). A launch parameter: the bits never depend on it."""
    return sm_count // (cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1])


@functools.lru_cache(maxsize=None)
def object_record(cfg: Knobs) -> Knobs:
    """The record a compiled object stands for: the knobs with the launch arguments (raster order, swizzle size, grid)
    at their defaults, so records differing only in them share one object."""
    return dataclasses.replace(cfg, raster_along="m", swizzle_size=1, grid_clusters=0)


def launch_args(cfg: Knobs, clusters: int) -> tuple:
    """The trailing Int32 arguments of an object's call: (persistent grid in clusters, swizzle size, raster-along-m
    flag); ``clusters`` is the launch device's co-resident count, overridden by the knobs' own ``grid_clusters``. The
    values reach the kernel as runtime arguments, so they are checked here (the static objects checked them at compile)."""
    grid = int(cfg.grid_clusters or clusters)
    if grid <= 0:
        raise ValueError(
            f"batch-invariant GEMM: the persistent grid must be positive, got {grid} clusters"
        )
    if cfg.swizzle_size not in SWIZZLES or cfg.raster_along not in RASTERS:
        raise ValueError(
            f"batch-invariant GEMM: swizzle {cfg.swizzle_size!r} / raster {cfg.raster_along!r} outside "
            f"{SWIZZLES} / {RASTERS}"
        )
    return (grid, int(cfg.swizzle_size), 1 if cfg.raster_along == "m" else 0)


def kernel_name(
    cfg: Knobs,
    b_major: str,
    chunks: int,
    in_fmt: str = "bf16",
    out_fmt: str = "bf16",
    batched: bool = False,
    bias: bool = False,
) -> str:
    """``C1``: the unsplit class (batch extent 1; ``_bL`` when the batch is dynamic); ``CL``: the split classes, whose chunk
    count is the dynamic batch extent of the operands (one object for every count)."""
    return jit.sanitize_symbol_name(
        f"{in_fmt}_{out_fmt}_b{b_major}_C{'L' if chunks > 1 else 1}_{object_record(cfg).describe_object()}"
        + ("_bL" if batched else "")
        + ("_bias" if bias else "")
    )


def get_compiled_gemm(
    cfg: Knobs,
    b_major: str,
    chunks: int,
    in_fmt: str = "bf16",
    out_fmt: str = "bf16",
    batched: bool = False,
    family: str = "",
    bias: bool = False,
):
    """bf16: ``kernel(a3, b3, c3, grid, swizzle, raster_m)``; fp8: ``kernel(a3, b3, c3, a_scale, b_scale, grid, swizzle,
    raster_m)`` with 1-element fp32 scales, for the direct stores: the unsplit class and the in-CTA chunk fold, whose
    chain multiplies by their product. The unsplit class has L = 1, or a dynamic L (the batch) with ``batched``; the
    fixed-chunk split (batch index = K chunk, L dynamic) stores fp32 partials through the UNSCALED body (the fixup
    applies the chain) or, with ``cfg.cta_chunks``, folds the chunks in the CTA (the same operand views, the output
    direct with L = 1). The trailing Int32 arguments (:func:`launch_args`) are the persistent grid in clusters, the
    swizzle size and the raster flag: placements, never the bits. ``family``, the launch device's architecture family,
    is part of the cache key (devices of two architectures in one process get their own callables; the module on disk
    is per architecture already). ``bias``: the biased bf16 chain, ``kernel(a3, b3, c3, bias, grid, swizzle, raster_m)``
    with the (N,) bf16 bias, for the direct stores only (the split's partial kernel never takes it: its fixup adds the
    bias)."""
    return _compiled_gemm(
        object_record(cfg), b_major, chunks > 1, in_fmt, out_fmt, batched, family, bias
    )


@functools.cache
def _compiled_gemm(
    cfg: Knobs,
    b_major: str,
    split: bool,
    in_fmt: str,
    out_fmt: str,
    batched: bool,
    family: str,
    bias: bool,
):
    in_dt, al = IN_DT[in_fmt], ALIGN[in_fmt]
    in_cta = cfg.cta_chunks and split
    direct = not split or in_cta
    scaled = in_fmt != "bf16" and direct
    assert not batched or not split, "the batched kernel is the unsplit class's"
    assert not bias or (direct and not scaled), (
        "the bias: a direct store of the bf16 class"
    )
    out_dt = OUT_DT[out_fmt] if direct else cutlass.Float32
    oal = ALIGN[out_fmt] if direct else ALIGN["f32"]
    # N's declared divisibility is that of what N makes contiguous: B's rows for an N-major B (the input alignment), else
    # only the output's rows (its alignment); the typed checks require the same, so a K-major fp8 weight with N a multiple
    # of 8 but not 16 is a shape of the class
    nal = al if b_major == "n" else oal
    name = kernel_name(cfg, b_major, 2 if split else 1, in_fmt, out_fmt, batched, bias)

    def compile_fn(**compile_options):
        sym_m, sym_n, sym_kc = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=nal),
            cute.sym_int(32, divisibility=al),
        )
        # the batch extent: 1 for the unsplit class, dynamic for the batched kernel and the split classes (the chunk count)
        L = cute.sym_int(32) if (batched or split) else 1
        Lc = 1 if in_cta else L
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_kc, L),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        if b_major == "n":  # (K, N) row-major viewed as (n, kc, l)
            b_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (sym_n, sym_kc, L),
                (1, cute.sym_int64(divisibility=al), cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )
        else:  # (N, K) row-major, a transposed weight
            b_fake = cute.runtime.make_fake_tensor(
                in_dt,
                (sym_n, sym_kc, L),
                (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
                assumed_align=16,
            )
        c_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_n, Lc),
            (cute.sym_int64(divisibility=oal), 1, cute.sym_int64(divisibility=oal)),
            assumed_align=16,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        body_out = out_fmt if direct else "f32"
        tail = (GRID_PLACEHOLDER, SWIZZLE_PLACEHOLDER, RASTER_PLACEHOLDER, stream_fake)
        if bias:
            bias_fake = cute.runtime.make_fake_tensor(
                in_dt, (sym_n,), (1,), assumed_align=16
            )
            return cute.compile(
                BiasBlockGemm(cfg, in_fmt, body_out, cta_chunks=in_cta),
                a_fake,
                b_fake,
                c_fake,
                bias_fake,
                *tail,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        if not scaled:
            return cute.compile(
                BlockGemm(cfg, in_fmt, body_out, cta_chunks=in_cta),
                a_fake,
                b_fake,
                c_fake,
                *tail,
                options="--enable-tvm-ffi",
                **compile_options,
            )
        sa_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (1,), (1,), assumed_align=4
        )
        sb_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (1,), (1,), assumed_align=4
        )
        return cute.compile(
            ScaledBlockGemm(cfg, in_fmt, body_out, cta_chunks=in_cta),
            a_fake,
            b_fake,
            c_fake,
            sa_fake,
            sb_fake,
            *tail,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(MODULE, name, compile_fn)
