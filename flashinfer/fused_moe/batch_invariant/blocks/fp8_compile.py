# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile wrapper for the grouped fp8 GEMM (:mod:`.fp8_kernel`): one TVM-FFI object per (knobs, chain) with dynamic Tk, N, K
and E; cache module ``batch_invariant_moe_fp8`` in FlashInfer's CuTe-DSL disk cache (this file is its root: the fp8 body and
epilogue, the bf16 blocks they compose and the dense GEMM blocks are its closure; the bf16 module's key does not move). The
persistent grid is a dynamic argument of the compiled kernel, as for the bf16 module.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from ....batch_invariant import jit
from ....gemm.batch_invariant.blocks.operands import ALIGN, IN_DT, OUT_DT
from ....gemm.batch_invariant.generator import TILES_N, Knobs
from .compile import GRID_PLACEHOLDER
from .epilogue import GROUP
from .fp8_epilogue import GEMM2_CHAIN, Fp8Chain
from .fp8_kernel import IN_FMT, ScaledGroupedBlockGemmFp8
from .scheduler import MAX_E

MODULE = "batch_invariant_moe_fp8"
#: The e4m3 output is stored as bytes (the chain's paired conversion fills byte registers).
OUT_DT_FP8 = dict(OUT_DT, e4m3=cutlass.Uint8)
OUT_ALIGN_FP8 = dict(ALIGN, e4m3=16)


def kernel_name(cfg: Knobs, chain: Fp8Chain) -> str:
    """The object's name: the chain (kind, activation, scale, output format, parameters) and the record without its
    ``grid_clusters``."""
    knobs = dataclasses.replace(cfg, grid_clusters=0)
    return jit.sanitize_symbol_name(
        f"grouped_e4m3_{chain.describe()}_{knobs.describe()}"
    )


def can_implement_grouped_fp8(
    knobs: Knobs, N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN
) -> bool:
    """Acceptance of the grouped fp8 body: well-formed knobs for the fp8 format at swizzle 1 (the group scheduler has no
    swizzle), the dense tiles' N (the narrow decode tiles have no staged-epilogue layout), 16-byte-aligned operands (K and N
    multiples of 16 e4m3 elements), the output rows aligned to its format, at most ``MAX_E`` experts, the gated chain on whole
    ``2 x GROUP`` column blocks -- plus the static TMEM thread-value check done at trace time."""
    if (
        not knobs.well_formed(IN_FMT)
        or knobs.swizzle_size != 1
        or knobs.cluster_k > 1
        or knobs.swap_ab
        or knobs.cta_chunks
    ):
        return False
    if knobs.tile_n not in TILES_N:
        return False
    if chain.pair and N % (2 * GROUP):
        return False
    al = ALIGN[IN_FMT]
    n_out = N // 2 if chain.pair else N
    return (
        K % al == 0
        and N % al == 0
        and n_out % OUT_ALIGN_FP8[chain.out_fmt] == 0
        and 1 <= E <= MAX_E
    )


def compile_grouped_fp8_fn(cfg: Knobs, chain: Fp8Chain) -> Callable[[], Any]:
    """The zero-argument compile closure of one grouped fp8 specialisation."""
    in_dt, al = IN_DT[IN_FMT], ALIGN[IN_FMT]
    out_dt, oal = OUT_DT_FP8[chain.out_fmt], OUT_ALIGN_FP8[chain.out_fmt]

    def compile_fn():
        sym_m, sym_n, sym_k, sym_e = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32, divisibility=al),
            cute.sym_int(32),
        )
        sym_no = cute.sym_int(32, divisibility=oal)
        a_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_m, sym_k, 1),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        b_fake = cute.runtime.make_fake_tensor(
            in_dt,
            (sym_n, sym_k, sym_e),
            (cute.sym_int64(divisibility=al), 1, cute.sym_int64(divisibility=al)),
            assumed_align=16,
        )
        o_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_no),
            (cute.sym_int64(divisibility=oal), 1),
            assumed_align=16,
        )
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        alpha_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (sym_e,), (1,), assumed_align=4
        )
        sa_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (1,), (1,), assumed_align=4
        )
        si_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (1,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        grid = GRID_PLACEHOLDER
        return cute.compile(
            ScaledGroupedBlockGemmFp8(cfg, chain),
            a_fake,
            b_fake,
            o_fake,
            c_fake,
            alpha_fake,
            sa_fake,
            si_fake,
            grid,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return compile_fn


@functools.cache
def get_compiled_grouped_fp8(cfg: Knobs, chain: Fp8Chain) -> Any:
    """``kernel(a3, b3, out2, counts, alpha, sa, si, grid)``: A ``(Tk, K, 1)`` K-major e4m3, B ``(N, K, E)`` K-major e4m3, out
    ``(Tk, Nout)`` row-major (uint8 bytes for the e4m3 chains, bf16 / f32 otherwise), counts ``(E,)`` int32, ``alpha (E,)`` the
    per-expert weight scales, ``sa (1,)`` the A operand's scale, ``si (1,)`` the intermediate multiplier, ``grid`` the persistent
    grid in clusters."""
    return jit.build_and_load(
        MODULE, kernel_name(cfg, chain), compile_grouped_fp8_fn(cfg, chain)
    )
