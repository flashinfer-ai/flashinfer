# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE classes' row moves (cache module ``batch_invariant_moe``, with the bodies): the scatter of the per-token quantized
activations into the grouped layout (the packed fp4 rows to their dense destinations, the 128x4 swizzled scale factors and
the per-token scales to the padded positions), and the re-tiling of the intermediate's scale factors and row scales from the
dense rows to the padded positions. Byte moves only: no arithmetic, a placement of the class's bytes.

The scale-factor buffers are FlashInfer's 128x4 swizzled layout (:mod:`flashinfer.batch_invariant.quantization.layout`):
the four scales of columns ``4g .. 4g + 3`` of a row are contiguous at byte ``g * 512 + (row % 32) * 16 + ((row % 128) //
32) * 4 + (row // 128) * 128 * pcols``, so a row's scales move as 4-byte groups.
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32

from ...batch_invariant import jit
from . import pdl
from .permute import LANES, WARPS

MODULE = "batch_invariant_moe"
SF_COL_TILE = 4  # scale columns per contiguous 4-byte group of the 128x4 layout
# blocks past which the scatter kernels' warps stride over the rows: the block launch rate, not the bytes, bounds a
# one-row-per-warp grid at the large batches
SCATTER_GRID_CAP = 4096


def _sf_group_offset(row: Int32, g: Int32, pcols: Int32) -> Int64:
    """Byte offset of the first scale of column group ``g`` of ``row`` in the 128x4 swizzled layout."""
    r = row.to(Int64)
    r128 = (r % 128).to(Int32)
    inner = g * Int32(512) + (r128 % 32) * 16 + (r128 // 32) * 4
    return (r // 128) * (Int32(128) * pcols).to(Int64) + inner.to(Int64)


class ScatterFp4Kernel:
    """The packed fp4 row of token ``t = i // k`` to row ``dest[i]`` (``B`` bytes), its swizzled scale-factor groups and its
    per-token scale to row ``dest_sf[i]`` (the padded position; the same tensor as ``dest`` for the padded packing). ``wpr``
    warps per expanded row (a placement, as the row gather: ``warps_per_row``), the row groups of a capped grid striding over
    the rows (``SCATTER_GRID_CAP``: a block per row would be bound by the block launch rate at the large batches); a warp's
    lanes stride over its 32-chunk slabs of the 16-byte row chunks, then over its slabs of the 4-byte scale groups; lane 0 of
    the row's first warp moves the per-token scale."""

    def __init__(
        self,
        k: int,
        wpr: int = 1,
        sf_src: str = "128x4",
        unit_pts: bool = False,
        no_pts: bool = False,
        block: int = 16,
        cpl: int = 1,
        spl: int = 1,
    ):
        """``sf_src``: the source scale factors' layout -- ``"128x4"`` (the quantizer's swizzled buffer) or ``"linear"`` (row-major
        ``[T, K / block]`` bytes, the layout of the per-tensor activation pack of the TRTLLM backends: the kernel assembles each
        4-byte group of the destination atom from the row's bytes, a zero past the row's last block); ``unit_pts``: the rows'
        per-token scales are one (the per-tensor pack: nothing to read, the destination gets ``1.0``); ``no_pts``: the class
        carries no per-token scale (the MX classes: nothing is read or stored); ``block``: the elements per scale of the
        linear source layout; ``cpl`` / ``spl``: a lane's 16-byte chunks and 4-byte scale groups per row (static bounds from
        the row width and ``wpr``, :func:`scatter_lanes`; the tail predicated), so a row's loads are all issued before its
        stores. Placements of the same bytes."""
        assert wpr >= 1 and wpr & (wpr - 1) == 0, wpr
        assert sf_src in ("128x4", "linear"), sf_src
        assert cpl >= 1 and spl >= 1, (cpl, spl)
        self.k, self.wpr, self.sf_src, self.unit_pts = k, wpr, sf_src, unit_pts
        self.no_pts, self.block, self.cpl, self.spl = no_pts, block, cpl, spl

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSf: cute.Tensor,
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mDestSf: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        pcols: Int32,
        stream,
    ):
        self.kernel(
            mX, mSf, mPts, mDest, mDestSf, mXp, mSfp, mPtsp, Tk, B, pcols
        ).launch(
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
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mDestSf: cute.Tensor,
        mXp: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        B: Int32,
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
            cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=128
        )
        n_vec = B // Int32(16)
        step = wpr * Int32(LANES)  # chunk j = (jj * wpr + part) * LANES + lane
        frags = [cute.make_rmem_tensor((16,), Uint8) for _ in range(self.cpl)]
        src32 = cute.make_tensor(
            cute.recast_ptr(mSf.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSf) // 4,)),
        )
        dst32 = cute.make_tensor(
            cute.recast_ptr(mSfp.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSfp) // 4,)),
        )
        n_groups = pcols // Int32(SF_COL_TILE)
        nb = B // Int32(
            self.block // 2
        )  # scale blocks per row (K / block): the linear row's bytes
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
                    if cutlass.const_expr(not self.no_pts):
                        if (lane == 0) & (part == 0):
                            if cutlass.const_expr(self.unit_pts):
                                mPtsp[ps] = Float32(1.0)
                            else:
                                mPtsp[ps] = mPts[t]
                    row_x = mX[(t, None)]
                    row_p = mXp[(p, None)]
                    for c in cutlass.range_constexpr(self.cpl):
                        if js[c] < n_vec:
                            cute.copy(
                                atom, cute.local_tile(row_x, (16,), (js[c],)), frags[c]
                            )
                    words = []
                    for c in cutlass.range_constexpr(self.spl):
                        word = Uint32(0)
                        if gs[c] < n_groups:
                            if cutlass.const_expr(self.sf_src == "linear"):
                                # the row's bytes 4gg .. 4gg + 3 assembled little-endian into the atom's 4-byte group (zero past
                                # the row's blocks)
                                base = t.to(Int64) * nb.to(Int64) + Int64(4) * gs[c].to(
                                    Int64
                                )
                                for b in cutlass.range_constexpr(4):
                                    col = Int32(4) * gs[c] + Int32(b)
                                    if col < nb:
                                        word = (
                                            word
                                            | (
                                                mSf[base + Int64(b)].to(Uint32)
                                                << Uint32(8 * b)
                                            )
                                        ).to(Uint32)
                            else:
                                word = src32[_sf_group_offset(t, gs[c], pcols) // 4].to(
                                    Uint32
                                )
                        words.append(word)
                    for c in cutlass.range_constexpr(self.cpl):
                        if js[c] < n_vec:
                            cute.copy(
                                atom, frags[c], cute.local_tile(row_p, (16,), (js[c],))
                            )
                    for c in cutlass.range_constexpr(self.spl):
                        if gs[c] < n_groups:
                            dst32[_sf_group_offset(ps, gs[c], pcols) // 4] = words[c]
                    # the chunks and groups past the static bounds (a caller that did not size them): one at a time
                    for jj in cutlass.range(Int32(self.cpl), n_mine, 1):
                        j = (jj * wpr + part) * Int32(LANES) + lane
                        cute.copy(atom, cute.local_tile(row_x, (16,), (j,)), frags[0])
                        cute.copy(atom, frags[0], cute.local_tile(row_p, (16,), (j,)))
                    for jj in cutlass.range(Int32(self.spl), n_sf, 1):
                        gg = (jj * wpr + part) * Int32(LANES) + lane
                        word = Uint32(0)
                        if cutlass.const_expr(self.sf_src == "linear"):
                            base = t.to(Int64) * nb.to(Int64) + Int64(4) * gg.to(Int64)
                            for b in cutlass.range_constexpr(4):
                                col = Int32(4) * gg + Int32(b)
                                if col < nb:
                                    word = (
                                        word
                                        | (
                                            mSf[base + Int64(b)].to(Uint32)
                                            << Uint32(8 * b)
                                        )
                                    ).to(Uint32)
                        else:
                            word = src32[_sf_group_offset(t, gg, pcols) // 4].to(Uint32)
                        dst32[_sf_group_offset(ps, gg, pcols) // 4] = word


class ScatterSfKernel:
    """The scale factors of row ``t = i // k`` moved to row ``dest[i]`` (4-byte groups of the 128x4 layout; the source and
    the destination with the same ``pcols``), and ``pts_perm[dest[i]] = pts[t]``: the re-tiling of the intermediate's scale
    factors and row scales from the dense rows to the padded positions (``k = 1``, ``dest`` the padded position of every
    dense row). One lane per 4-byte group, one warp per row, the warps of a capped grid striding over the rows."""

    def __init__(self, k: int, no_pts: bool = False, spl: int = 1):
        """``no_pts``: the class carries no per-token scale (the MX classes): the row scales are neither read nor stored;
        ``spl``: a lane's 4-byte groups per row (a static bound, :func:`scatter_sf_lanes`; the tail predicated)."""
        assert spl >= 1, spl
        self.k, self.no_pts, self.spl = k, no_pts, spl

    @cute.jit
    def __call__(
        self,
        mSf: cute.Tensor,
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        pcols: Int32,
        stream,
    ):
        self.kernel(mSf, mPts, mDest, mSfp, mPtsp, Tk, pcols).launch(
            grid=[scatter_grid(Tk), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mSf: cute.Tensor,
        mPts: cute.Tensor,
        mDest: cute.Tensor,
        mSfp: cute.Tensor,
        mPtsp: cute.Tensor,
        Tk: Int32,
        pcols: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        pdl.trigger_one_wave()
        w = bidx * WARPS + tidx // LANES
        stride = gdim * WARPS
        n_rows = cutlass.max((Tk - w + stride - Int32(1)) // stride, Int32(0))
        src32 = cute.make_tensor(
            cute.recast_ptr(mSf.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSf) // 4,)),
        )
        dst32 = cute.make_tensor(
            cute.recast_ptr(mSfp.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mSfp) // 4,)),
        )
        n_groups = pcols // Int32(SF_COL_TILE)
        gs = [Int32(c) * Int32(LANES) + lane for c in range(self.spl)]
        n_mine = (n_groups - lane + Int32(LANES - 1)) // Int32(LANES)
        if w < stride:
            last = Tk - Int32(1)
            p_next = mDest[cutlass.min(w, last)]
            for r in cutlass.range(0, n_rows, 1):
                i = w + r * stride
                t = i // Int32(self.k)
                p = p_next
                p_next = mDest[cutlass.min(i + stride, last)]
                # a row without a padded position (the plan marks it -1: a dropped slot's share of the buffer) stores nothing
                live = p >= Int32(0)
                p = cutlass.max(p, Int32(0))
                if live:
                    if cutlass.const_expr(not self.no_pts):
                        if lane == 0:
                            mPtsp[p] = mPts[t]
                    words = []
                    for c in cutlass.range_constexpr(self.spl):
                        word = Uint32(0)
                        if gs[c] < n_groups:
                            word = src32[_sf_group_offset(t, gs[c], pcols) // 4].to(
                                Uint32
                            )
                        words.append(word)
                    for c in cutlass.range_constexpr(self.spl):
                        if gs[c] < n_groups:
                            dst32[_sf_group_offset(p, gs[c], pcols) // 4] = words[c]
                    for jj in cutlass.range(
                        Int32(self.spl), n_mine, 1
                    ):  # the groups past the static bound
                        g = jj * Int32(LANES) + lane
                        dst32[_sf_group_offset(p, g, pcols) // 4] = src32[
                            _sf_group_offset(t, g, pcols) // 4
                        ]


def scatter_lanes(B: int, pcols: int, wpr: int) -> dict:
    """The static per-lane counts of a row scatter at ``B`` bytes per row, ``pcols`` scale columns and ``wpr`` warps per row:
    ``cpl`` 16-byte chunks and ``spl`` 4-byte scale groups per lane (the ceilings; a placement of the same bytes)."""
    return {
        "cpl": max(1, -(-(B // 16) // (wpr * LANES))),
        "spl": max(1, -(-(pcols // SF_COL_TILE) // (wpr * LANES))),
    }


def scatter_sf_lanes(pcols: int) -> int:
    """The static 4-byte groups per lane of the scale-factor re-tiling (one warp per row)."""
    return max(1, -(-(pcols // SF_COL_TILE) // LANES))


def scatter_grid(Tk: Int32) -> Int32:
    """Blocks of a scatter launch: one warp per row up to ``SCATTER_GRID_CAP`` blocks, beyond which the warps stride over
    the rows (a placement: the same bytes move whatever the grid)."""
    return cutlass.min(cute.ceil_div(Tk, WARPS), Int32(SCATTER_GRID_CAP))


@functools.cache
def get_compiled_scatter_fp4(
    k: int,
    wpr: int = 1,
    sf_src: str = "128x4",
    unit_pts: bool = False,
    no_pts: bool = False,
    block: int = 16,
    cpl: int = 1,
    spl: int = 1,
) -> Any:
    """``kernel(x4 [T, B] u8, sf [n_src] u8, pts [T] f32, dest [T k] i32, dest_sf [T k] i32, x_perm [rows, B] u8, sf_perm
    [n_dst] u8, pts_perm [rows_sf] f32, Tk, B, pcols)``, ``B % 16 == 0``; ``wpr`` warps per row (``warps_per_row``); ``sf_src``
    the source scale factors' layout (``"128x4"`` / ``"linear"``: ``sf`` then the row-major ``[T, K / block]`` bytes, any count);
    ``unit_pts``: the per-token scales are one (``pts`` unread, a 1-element placeholder); ``no_pts``: no per-token scale at all
    (``pts`` / ``pts_perm`` unread, 1-element placeholders); ``block``: the elements per scale of the linear layout."""

    def compile_fn(**compile_options):
        sym_t, sym_n, sym_p, sym_b = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=16),
        )
        sym_s = (
            cute.sym_int(32)
            if sf_src == "linear"
            else cute.sym_int(32, divisibility=512)
        )
        sym_sp, sym_pp = (
            cute.sym_int(32, divisibility=512),
            cute.sym_int(32),
        )
        x_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_t, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sf_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_s,), (1,), assumed_align=16 if sf_src == "128x4" else 1
        )
        pts_fake = cute.runtime.make_fake_tensor(
            Float32,
            (cute.sym_int(32) if (unit_pts or no_pts) else sym_t,),
            (1,),
            assumed_align=4,
        )
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        ds_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        xp_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_p, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sfp_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_sp,), (1,), assumed_align=16
        )
        ptsp_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_pp,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            ScatterFp4Kernel(k, wpr, sf_src, unit_pts, no_pts, block, cpl, spl),
            x_fake,
            sf_fake,
            pts_fake,
            d_fake,
            ds_fake,
            xp_fake,
            sfp_fake,
            ptsp_fake,
            Int32(1),
            Int32(16),
            Int32(4),
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(
        MODULE,
        jit.sanitize_symbol_name(
            f"moe_scatter_fp4_k{k}"
            + (f"_w{wpr}" if wpr > 1 else "")
            + f"_c{cpl}_s{spl}"
            + ("_lin" if sf_src == "linear" else "")
            + ("_unit" if unit_pts else "")
            + ("_nopts" if no_pts else "")
            + (f"_b{block}" if block != 16 else "")
        ),
        compile_fn,
    )


@functools.cache
def get_compiled_scatter_sf(k: int, no_pts: bool = False, spl: int = 1) -> Any:
    """``kernel(sf [n_src] u8, pts [T] f32, dest [T k] i32, sf_perm [n_dst] u8, pts_perm [rows_sf] f32, Tk, pcols)``;
    ``no_pts``: the per-token scales are neither read nor stored (1-element placeholders); ``spl``: a lane's groups per row
    (:func:`scatter_sf_lanes`)."""

    def compile_fn(**compile_options):
        sym_s, sym_t, sym_n, sym_sp, sym_p = (
            cute.sym_int(32, divisibility=512),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=512),
            cute.sym_int(32),
        )
        sf_fake = cute.runtime.make_fake_tensor(Uint8, (sym_s,), (1,), assumed_align=16)
        pts_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_t,), (1,), assumed_align=4
        )
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        sfp_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_sp,), (1,), assumed_align=16
        )
        ptsp_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_p,), (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            ScatterSfKernel(k, no_pts, spl),
            sf_fake,
            pts_fake,
            d_fake,
            sfp_fake,
            ptsp_fake,
            Int32(1),
            Int32(4),
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(
        MODULE,
        jit.sanitize_symbol_name(
            f"moe_scatter_sf_k{k}" + ("_nopts" if no_pts else "") + f"_s{spl}"
        ),
        compile_fn,
    )
