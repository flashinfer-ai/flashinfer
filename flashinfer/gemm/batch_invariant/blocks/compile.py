# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the block-built GEMM: one TVM-FFI object per (knobs, B major, chunks, persistent grid, formats) with dynamic
M, N, K; cache module ``batch_invariant_gemm`` in FlashInfer's CuTe-DSL disk cache (both formats share one source closure).
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
from ..generator import Knobs
from .kernel import BiasBlockGemm, BlockGemm, ScaledBlockGemm
from .operands import ALIGN, IN_DT, OUT_DT, can_implement  # noqa: F401  (re-exported)

MODULE = "batch_invariant_gemm"


def max_active_clusters(cfg: Knobs, sm_count: int) -> int:
    """The fallback persistent grid in clusters, one CTA per SM: what the launch device can host at most, not what it
    can host at once (a GPC holds whole clusters only, so the co-resident count is below ``SMs // size`` for clusters of
    more than one CTA; ``api.persistent_clusters`` asks the device). A launch parameter: the bits never depend on it."""
    return sm_count // (cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1])


def kernel_name(
    cfg: Knobs,
    b_major: str,
    chunks: int,
    clusters: int,
    in_fmt: str = "bf16",
    out_fmt: str = "bf16",
    batched: bool = False,
    bias: bool = False,
) -> str:
    """``clusters`` is the compiled persistent grid in clusters (the knobs' own ``grid_clusters`` when set, else the
    launch device's co-resident count), keyed once as ``_g<clusters>``."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"{in_fmt}_{out_fmt}_b{b_major}_C{chunks}_{knobs.describe()}_g{clusters}"
        + ("_bL" if batched else "")
        + ("_bias" if bias else "")
    )


@functools.cache
def get_compiled_gemm(
    cfg: Knobs,
    b_major: str,
    chunks: int,
    clusters: int,
    in_fmt: str = "bf16",
    out_fmt: str = "bf16",
    batched: bool = False,
    family: str = "",
    bias: bool = False,
):
    """bf16: ``kernel(a3, b3, c3)``; fp8: ``kernel(a3, b3, c3, a_scale, b_scale)`` with 1-element fp32 scales, for the
    direct stores: the unsplit class and the in-CTA chunk fold, whose chain multiplies by their product. The unsplit class
    has L = 1, or a dynamic L (the batch) with ``batched``; the fixed-chunk split (batch index = K chunk) stores fp32
    partials through the UNSCALED body (the fixup applies the chain) or, with ``cfg.cta_chunks``, folds the chunks in the
    CTA (the same operand views, the output direct with L = 1). ``clusters`` is the persistent grid in clusters the body is compiled for: the launch
    device's co-resident cluster count for the knobs' cluster shape (``api.persistent_clusters``), unless the knobs
    force one (``cfg.grid_clusters``); it is part of the cache key, as is ``family``, the launch device's architecture
    family (devices of two architectures in one process get their own callables; the module on disk is per architecture
    already). A placement: which CTA computes which tile, never the bits. ``bias``: the biased bf16 chain,
    ``kernel(a3, b3, c3, bias)`` with the (N,) bf16 bias, for the direct stores only (the split's partial kernel never
    takes it: its fixup adds the bias)."""
    in_dt, al = IN_DT[in_fmt], ALIGN[in_fmt]
    in_cta = cfg.cta_chunks and chunks > 1
    direct = chunks == 1 or in_cta
    scaled = in_fmt != "bf16" and direct
    assert not batched or chunks == 1, "the batched kernel is the unsplit class's"
    assert not bias or (direct and not scaled), (
        "the bias: a direct store of the bf16 class"
    )
    out_dt = OUT_DT[out_fmt] if direct else cutlass.Float32
    oal = ALIGN[out_fmt] if direct else ALIGN["f32"]
    # N's declared divisibility is that of what N makes contiguous: B's rows for an N-major B (the input alignment), else
    # only the output's rows (its alignment); the typed checks require the same, so a K-major fp8 weight with N a multiple
    # of 8 but not 16 is a shape of the class
    nal = al if b_major == "n" else oal
    grid = cfg.grid_clusters or clusters
    name = kernel_name(cfg, b_major, chunks, grid, in_fmt, out_fmt, batched, bias)

    def compile_fn():
        sym_m, sym_n, sym_kc = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=nal),
            cute.sym_int(32, divisibility=al),
        )
        L = (
            cute.sym_int(32) if batched else chunks
        )  # the batch extent is dynamic for the batched kernel
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
        grid = cfg.grid_clusters or clusters
        assert grid > 0, (cfg, clusters)
        body_out = out_fmt if direct else "f32"
        if bias:
            bias_fake = cute.runtime.make_fake_tensor(
                in_dt, (sym_n,), (1,), assumed_align=16
            )
            return cute.compile(
                BiasBlockGemm(cfg, in_fmt, body_out, chunks),
                a_fake,
                b_fake,
                c_fake,
                bias_fake,
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        if not scaled:
            return cute.compile(
                BlockGemm(cfg, in_fmt, body_out, chunks),
                a_fake,
                b_fake,
                c_fake,
                grid,
                stream_fake,
                options="--enable-tvm-ffi",
            )
        sa_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (1,), (1,), assumed_align=4
        )
        sb_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (1,), (1,), assumed_align=4
        )
        return cute.compile(
            ScaledBlockGemm(cfg, in_fmt, body_out, chunks),
            a_fake,
            b_fake,
            c_fake,
            grid,
            stream_fake,
            sa_fake,
            sb_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
