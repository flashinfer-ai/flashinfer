# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 class's scatter (plumbing, no floating point): the quantized activation rows and their UE8M0 block scales moved
into the padded group layout the grouped block-scaled GEMM reads.

``x_perm[dest[i], :] = xq[i // k, :]`` for every expanded row ``i`` of the ``T * k`` (token, slot) pairs: one warp per live
row, nothing written to the padding rows of the 128-row atoms (their bytes stay unspecified: the MMA is row-local and no tile
stores a padding row, so they never reach a stored value); the scale bytes of token ``t`` are read from the **linear** layout
the class takes its activations in (``xsf [T, K / 32]``: the MXFP8 MoE activation pack) and written to the **128x4
swizzled** layout of the ``(Tk_pad, K)`` operand (byte ``(r // 128) * 128 * pcols + (j // 4) * 512 + (r % 32) * 16 +
((r % 128) // 32) * 4 + j % 4``), the padding columns ``j >= K / 32`` of a live row zero. 16-byte vectors for the values
(lanes stride over the row), one lane per scale byte. Every copy is an unpredicated ``cute.copy`` under a plain branch (a
predicated copy lowers to element-wise stores), launched with programmatic dependent launch.
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32

from ...batch_invariant import jit
from . import pdl
from .permute_nvfp4 import SF_COL_TILE, scatter_grid, scatter_lanes

MODULE = "batch_invariant_moe"  # the MoE kernels' one module (the scatter shares the bodies' cache key)
WARPS = 4
LANES = 32
VEC_BYTES = 16


@cute.jit
def _sf_byte_offset(row: Int32, col: Int32, pcols: Int32) -> Int64:
    """Byte offset of scale column ``col`` of ``row`` in the 128x4 swizzled layout of a buffer with ``pcols`` scale columns."""
    r = row.to(Int64)
    r128 = (r % 128).to(Int32)
    inner = (col // 4) * Int32(512) + (r128 % 32) * 16 + (r128 // 32) * 4 + (col % 4)
    return (r // 128) * (Int32(128) * pcols).to(Int64) + inner.to(Int64)


class ScatterMxKernel:
    """``kernel(xq [T, B] u8, xsf [T * nb] u8 (linear), dest [Tk] i32, dest_sf [Tk] i32, x_perm [rows, B] u8, sf_perm
    [sf_numel(sf_rows, K)] u8, Tk, B, nb, pcols)``: ``B`` bytes per row (``B % 16 == 0``), ``nb = K / 32`` scale bytes per
    row, ``pcols`` the padded scale columns of the destination (a multiple of 4); ``dest[i]`` the destination row of
    expanded row ``i`` (token ``i // k``), ``dest_sf[i]`` the row of its scale bytes (``dest`` itself in the padded
    packing; the padded position of the dense row in the dense one). The row scatter's placement (the fp4 class's): ``wpr``
    warps per expanded row (``permute.warps_per_row``), the row groups of a capped grid striding over the rows
    (``scatter_grid``), a warp's lanes striding over its 32-chunk slabs of the 16-byte row chunks, then over its slabs of the
    4-byte scale groups of the swizzled atom (each assembled from the row's linear bytes, zero past the row's blocks); a row's
    loads are all issued before its stores (``cpl`` / ``spl`` chunks and groups per lane, static bounds from the row width:
    :func:`permute_nvfp4.scatter_lanes`; the tail predicated) and the next row's positions are loaded while this row moves.
    The padding rows of the 128-row atoms are never written (unspecified bytes the row-local MMA reads into rows no tile
    stores)."""

    def __init__(self, k: int, wpr: int = 1, cpl: int = 1, spl: int = 1):
        assert wpr >= 1 and wpr & (wpr - 1) == 0, wpr
        assert cpl >= 1 and spl >= 1, (cpl, spl)
        self.k, self.wpr, self.cpl, self.spl = k, wpr, cpl, spl

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSf: cute.Tensor,
        mDest: cute.Tensor,
        mDestSf: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        nb: Int32,
        pcols: Int32,
        stream,
    ):
        self.kernel(mX, mSf, mDest, mDestSf, mXp, mSfp, Tk, B, nb, pcols).launch(
            grid=[scatter_grid(Tk * Int32(self.wpr)), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mSf: cute.Tensor,
        mDest: cute.Tensor,
        mDestSf: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        nb: Int32,
        pcols: Int32,
    ):
        wpr = Int32(self.wpr)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        pdl.trigger_one_wave()
        g = bidx * WARPS + tidx // LANES
        w = g // wpr  # this warp's row group
        part = g % wpr  # its slab phase within the row
        stride = (gdim * WARPS) // wpr  # row groups per sweep
        # this group's rows: w, w + stride, ...
        n_rows = cutlass.max((Tk - w + stride - Int32(1)) // stride, Int32(0))
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=8 * VEC_BYTES
        )
        n_vec = B // Int32(VEC_BYTES)
        step = wpr * Int32(LANES)  # chunk j = (jj * wpr + part) * LANES + lane
        frags = [cute.make_rmem_tensor((VEC_BYTES,), Uint8) for _ in range(self.cpl)]
        dst32 = cute.make_tensor(
            cute.recast_ptr(mSfp.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSfp) // 4,)),
        )
        n_groups = pcols // Int32(SF_COL_TILE)
        # this lane's chunks and scale groups of a row: slab c of its phase (static bounds, the tail predicated), and their
        # dynamic counts for the chunks past the bounds
        js = [(Int32(c) * wpr + part) * Int32(LANES) + lane for c in range(self.cpl)]
        gs = [(Int32(c) * wpr + part) * Int32(LANES) + lane for c in range(self.spl)]
        n_mine = (n_vec - lane - part * Int32(LANES) + step - Int32(1)) // step
        n_sf = (n_groups - lane - part * Int32(LANES) + step - Int32(1)) // step
        if w < stride:
            # the positions of the next row are loaded while this row moves (the index clamped: in bounds past the last row)
            last = Tk - Int32(1)
            p_next = mDest[cutlass.min(w, last)]
            ps_next = mDestSf[cutlass.min(w, last)]
            for r in cutlass.range(0, n_rows, 1):
                i = w + r * stride
                t = i // Int32(self.k)
                p = p_next
                ps = ps_next
                i_next = cutlass.min(i + stride, last)
                p_next = mDest[i_next]
                ps_next = mDestSf[i_next]
                # a dropped slot (dest -1: an expert id outside [0, E)) has no row: nothing is stored for it
                live = p >= Int32(0)
                p = cutlass.max(p, Int32(0))
                ps = cutlass.max(ps, Int32(0))
                if live:
                    row_x = mX[(t, None)]
                    row_p = mXp[(p, None)]
                    for c in cutlass.range_constexpr(self.cpl):
                        if js[c] < n_vec:
                            cute.copy(
                                atom,
                                cute.local_tile(row_x, (VEC_BYTES,), (js[c],)),
                                frags[c],
                            )
                    words = []
                    for c in cutlass.range_constexpr(self.spl):
                        words.append(_sf_word(mSf, t, nb, gs[c], gs[c] < n_groups))
                    for c in cutlass.range_constexpr(self.cpl):
                        if js[c] < n_vec:
                            cute.copy(
                                atom,
                                frags[c],
                                cute.local_tile(row_p, (VEC_BYTES,), (js[c],)),
                            )
                    for c in cutlass.range_constexpr(self.spl):
                        if gs[c] < n_groups:
                            dst32[
                                _sf_byte_offset(ps, Int32(SF_COL_TILE) * gs[c], pcols)
                                // 4
                            ] = words[c]
                    # the chunks and groups past the static bounds (a caller that did not size them): one at a time
                    for jj in cutlass.range(Int32(self.cpl), n_mine, 1):
                        j = (jj * wpr + part) * Int32(LANES) + lane
                        cute.copy(
                            atom, cute.local_tile(row_x, (VEC_BYTES,), (j,)), frags[0]
                        )
                        cute.copy(
                            atom, frags[0], cute.local_tile(row_p, (VEC_BYTES,), (j,))
                        )
                    for jj in cutlass.range(Int32(self.spl), n_sf, 1):
                        gg = (jj * wpr + part) * Int32(LANES) + lane
                        dst32[
                            _sf_byte_offset(ps, Int32(SF_COL_TILE) * gg, pcols) // 4
                        ] = _sf_word(mSf, t, nb, gg, cutlass.Boolean(True))


@cute.jit
def _sf_word(mSf: cute.Tensor, t: Int32, nb: Int32, g: Int32, ok) -> Uint32:
    """The 4-byte group ``g`` of token ``t``'s scale row assembled little-endian from its linear bytes (``nb`` per row; zero
    past the row's blocks and when ``ok`` is false)."""
    word = Uint32(0)
    if ok:
        base = t.to(Int64) * nb.to(Int64) + Int64(SF_COL_TILE) * g.to(Int64)
        for b in cutlass.range_constexpr(SF_COL_TILE):
            col = Int32(SF_COL_TILE) * g + Int32(b)
            if col < nb:
                word = (word | (mSf[base + Int64(b)].to(Uint32) << Uint32(8 * b))).to(
                    Uint32
                )
    return word


def scatter_mx_kernel_name(k: int, wpr: int = 1, cpl: int = 1, spl: int = 1) -> str:
    return jit.sanitize_symbol_name(f"moe_scatter_mx_k{k}_w{wpr}_c{cpl}_s{spl}")


def scatter_mx_lanes(B: int, pcols: int, wpr: int) -> dict:
    """The static per-lane bounds of the MX row scatter (``cpl`` 16-byte chunks, ``spl`` 4-byte scale groups): the fp4
    class's rule at ``B`` bytes per row."""
    return scatter_lanes(B, pcols, wpr)


@functools.cache
def get_compiled_scatter_mx(k: int, wpr: int = 1, cpl: int = 1, spl: int = 1) -> Any:
    """``kernel(xq [T, B] u8, xsf [n_src] u8, dest [Tk] i32, dest_sf [Tk] i32, x_perm [rows, B] u8, sf_perm [n_dst] u8, Tk, B,
    nb, pcols)``; ``wpr`` warps per row (``permute.warps_per_row(T k, B / 16, sms)``), ``cpl`` / ``spl`` the lane's static
    bounds (:func:`scatter_mx_lanes`)."""

    def compile_fn(**compile_options):
        sym_t, sym_n, sym_p, sym_b = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=16),
        )
        sym_s, sym_sp = cute.sym_int(32), cute.sym_int(32, divisibility=512)
        x_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_t, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sf_fake = cute.runtime.make_fake_tensor(Uint8, (sym_s,), (1,), assumed_align=16)
        dest_fake = cute.runtime.make_fake_tensor(
            Int32, (sym_p,), (1,), assumed_align=4
        )
        dest_sf_fake = cute.runtime.make_fake_tensor(
            Int32, (sym_p,), (1,), assumed_align=4
        )
        xp_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_n, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sfp_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_sp,), (1,), assumed_align=16
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            ScatterMxKernel(k, wpr, cpl, spl),
            x_fake,
            sf_fake,
            dest_fake,
            dest_sf_fake,
            xp_fake,
            sfp_fake,
            Int32(128),
            Int32(16),
            Int32(1),
            Int32(4),
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(
        MODULE, scatter_mx_kernel_name(k, wpr, cpl, spl), compile_fn
    )
