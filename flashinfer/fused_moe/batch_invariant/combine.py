# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Combine kernel: ``out[t, h] = cvt( fold_{r in slot order} add(acc, mul(f32(y[dest[t, r], h]), w[t, r])) )`` from
``+0``. The fold is the engine's plan of ``spec.combine_reduction`` (one register-level ``slot`` fold, fused with the
loads by ``canon.init_state``); ``wpt`` warps per token (a placement: warp ``w`` of a token owns the 32-vector slabs
``w, w + wpt, ...`` of the row; at small ``T`` the row is spread over several warps so the k dependent loads per vector
are not serialised in one warp), lanes stride over 16-byte vectors of ``h``; every op through ``CuteF32``. A dropped slot
(``dest[t, r] = -1``: an expert id outside ``[0, E)``) contributes the term ``+0``, whatever its weight. Launched with
programmatic dependent launch (waits for GEMM2's grid before reading ``y``)."""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ...batch_invariant import canon, jit
from ...batch_invariant.fp_interface.cute import get_cutlass_dtype
from ...batch_invariant.quantization.cute_ext import CuteF32
from . import spec as S
from . import pdl

MODULE = "batch_invariant_moe"
WARPS = 4
LANES = 32
VEC = 8


def _select_f32(cond, a, b):
    """``a`` where ``cond`` else ``b`` (``arith.select``, one ``selp.f32``: no arithmetic)."""
    from cutlass._mlir.dialects import arith

    return Float32(
        arith.select(cond.ir_value(), Float32(a).ir_value(), Float32(b).ir_value())
    )


def _fold_slots(plan, frags, ws, oks, i):
    """The engine's slot fold for element ``i`` of the vectors: ``init_state`` fuses the loads ``mul(f32(y_r), w_r)``
    with the fold steps in slot order (a plain Python helper, so the staged loop body assigns only device values); a
    dropped slot (``oks[r]`` false) enters as the term ``+0``."""
    state, start = canon.init_state(
        plan,
        CuteF32,
        lambda key: _select_f32(
            oks[key[0]],
            CuteF32.mul(frags[key[0]][i].to(Float32), ws[key[0]]),
            CuteF32.const(0.0),
        ),
    )
    return canon.run(plan, state, start, None, CuteF32)


class CombineKernel:
    def __init__(
        self, k: int, out_fmt: str = "bf16", wpt: int = 1, w_round: bool = False
    ):
        assert wpt >= 1 and wpt & (wpt - 1) == 0, wpt
        self.k, self.out_fmt, self.wpt, self.w_round = k, out_fmt, wpt, w_round
        red = S.combine_reduction(k, lambda fp, r: None)
        self.plan = canon.plan(red, canon.Placement.make({"slot": [(k, "reg")]}))

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mDest: cute.Tensor,
        mW: cute.Tensor,
        mOut: cute.Tensor,
        T: Int32,
        H: Int32,
        stream,
    ):
        self.kernel(mY, mDest, mW, mOut, T, H).launch(
            grid=[cute.ceil_div(T * Int32(self.wpt), WARPS), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mY: cute.Tensor,
        mDest: cute.Tensor,
        mW: cute.Tensor,
        mOut: cute.Tensor,
        T: Int32,
        H: Int32,
    ):
        k = self.k
        wpt = self.wpt
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        g_raw = (
            bidx * WARPS + tidx // LANES
        )  # the global warp index: token t_raw, part w of its row
        t_raw = g_raw // Int32(wpt)
        part = g_raw % Int32(wpt)
        valid = t_raw < T
        t = cutlass.min(
            t_raw, T - Int32(1)
        )  # out-of-range warps recompute a valid token and store nothing
        ps = [mDest[t, r] for r in range(k)]
        ws = [mW[t, r] for r in range(k)]
        if cutlass.const_expr(self.w_round):
            # pre-routed weights given as f32 enter as their bf16 rounding (the packed word's value): the engine's
            # conversion here and the value's own exact widening, instead of two host passes over the slots
            ws = [CuteF32.cvt_out(w_, "bf16").to(Float32) for w_ in ws]
        # a dropped slot (dest -1: an expert id outside [0, E)) has no expert row: its loads take row 0 and its term is +0
        oks = [p >= Int32(0) for p in ps]
        ps = [cutlass.max(p, Int32(0)) for p in ps]
        atom_in = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=VEC * mY.element_type.width,
        )
        atom_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mOut.element_type,
            num_bits_per_copy=VEC * mOut.element_type.width,
        )
        n_vec = H // Int32(VEC)
        frags = [cute.make_rmem_tensor((VEC,), mY.element_type) for _ in range(k)]
        out = cute.make_rmem_tensor((VEC,), mOut.element_type)
        rows_y = [
            mY[(ps[r], None)] for r in range(k)
        ]  # 1-D row slices (a (1, VEC) tile mis-vectorises)
        row_out = mOut[(t, None)]
        # this warp's vectors: the 32-vector slabs part, part + wpt, ...; j = (jj * wpt + part) * 32 + lane < n_vec
        n_slabs = (n_vec + Int32(LANES - 1)) // Int32(LANES)
        n_mine = (n_slabs - part + Int32(wpt - 1)) // Int32(wpt)
        for jj in cutlass.range(0, n_mine, 1):
            j = (jj * Int32(wpt) + part) * LANES + lane
            jc = cutlass.min(
                j, n_vec - Int32(1)
            )  # the last slab's lanes past the row recompute a valid vector and store nothing
            for r in cutlass.range_constexpr(k):
                cute.copy(atom_in, cute.local_tile(rows_y[r], (VEC,), (jc,)), frags[r])
            for i in cutlass.range_constexpr(VEC):
                out[i] = CuteF32.cvt_out(
                    _fold_slots(self.plan, frags, ws, oks, i), self.out_fmt
                )
            if valid & (j < n_vec):
                cute.copy(atom_out, out, cute.local_tile(row_out, (VEC,), (jc,)))
        pdl.trigger_one_wave()


WARPS_PER_SM_TARGET = 32  # the combine's warp count target per SM: one warp's serial slabs of a row are latency-bound


def warps_per_token(T: int, H: int, sms: int) -> int:
    """The combine's placement at (T, H): warps per token doubled while the launch holds fewer than ``WARPS_PER_SM_TARGET``
    warps per SM (a long token count alone fills the machine and is bandwidth-bound with one warp per token); a power of
    two, at most 16 and at most one 32-vector slab per warp (H / 256 for bf16 rows). ``sms``: the device's SM count."""
    n_vec = max(1, H // VEC)
    cap = max(1, min(16, -(-n_vec // LANES)))
    w = 1
    while w * 2 <= cap and T * w * 2 <= WARPS_PER_SM_TARGET * sms:
        w *= 2
    return w


def combine_kernel_name(
    k: int, out_fmt: str, y_fmt: str, wpt: int, w_round: bool = False
) -> str:
    return jit.sanitize_symbol_name(
        f"moe_combine_k{k}_{y_fmt}_{out_fmt}"
        + (f"_w{wpt}" if wpt > 1 else "")
        + ("_wr" if w_round else "")
    )


@functools.cache
def get_compiled_combine(
    k: int,
    out_fmt: str = "bf16",
    y_fmt: str = "bf16",
    wpt: int = 1,
    w_round: bool = False,
) -> Any:
    """``kernel(y [rows, H], dest [T, k] i32, w [T, k] f32, out [T, H], T, H)``; ``wpt`` warps per token (``warps_per_token``);
    ``w_round``: the weights enter as their bf16 rounding (the pre-routed packed-word convention)."""
    y_dt, o_dt = get_cutlass_dtype(y_fmt), get_cutlass_dtype(out_fmt)

    def compile_fn():
        sym_t, sym_tk, sym_h = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=8),
        )
        y_fake = cute.runtime.make_fake_tensor(
            y_dt, (sym_tk, sym_h), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        d_fake = cute.runtime.make_fake_tensor(
            Int32, (sym_t, k), (k, 1), assumed_align=4
        )
        w_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_t, k), (k, 1), assumed_align=4
        )
        o_fake = cute.runtime.make_fake_tensor(
            o_dt, (sym_t, sym_h), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            CombineKernel(k, out_fmt, wpt, w_round),
            y_fake,
            d_fake,
            w_fake,
            o_fake,
            Int32(1),
            Int32(8),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE, combine_kernel_name(k, out_fmt, y_fmt, wpt, w_round), compile_fn
    )
