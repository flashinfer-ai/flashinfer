# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE classes' per-token quantizations with destination maps (cache module ``batch_invariant_moe_nvfp4_quant``):
the per-token NVFP4 quantization class (:mod:`flashinfer.batch_invariant.quantization.nvfp4_kernel`: the same engine plan,
chains and primitives, term for term) with its stores redirected. :class:`RowMappedNvFP4QuantKernel` stores the scale
factors and per-row scale of input row ``r`` at row ``map[r]`` (the 128-row-padded position of the dense packing) while the
e2m1 codes stay at row ``r``: one launch in place of the quantizer plus the scale re-tiling of the intermediate.
:class:`ScatterQuantNvFP4Quantizer` quantizes token ``t`` once and stores its codes at the ``k`` dense rows ``dest[t k + j]``
and its scale factors and per-token scale at the padded positions ``dest_pad[t k + j]``: one launch in place of the
quantizer plus the fp4 scatter of the input. Both are placements of the class's bytes (the same codes, scale bytes and row
scales, at the rows the grouped GEMMs read).
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32

from ....batch_invariant import canon, jit
from .. import pdl
from ....batch_invariant.quantization.cute_ext import CuteF32
from ....batch_invariant.quantization.layout import sanitize_symbol_name
from ....batch_invariant.quantization.nvfp4_kernel import (
    VEC_BYTES,
    NvFP4QuantKernel,
    _CutePeers,
    _Loader,
    _predicate,
    _quantize_block,
    _Span,
    _tiled_copy,
)
from ....batch_invariant.quantization.nvfp4_schedule import (
    BLOCK,
    LANES,
    SF_COL_TILE,
    QuantConfig,
)
from ....batch_invariant.quantization.nvfp4_spec import PER_TOKEN, row_scale_chain

MODULE = "batch_invariant_moe_nvfp4_quant"


class RowMappedNvFP4QuantKernel(NvFP4QuantKernel):
    """The per-token row kernel of :class:`NvFP4QuantKernel` (the 128x4 layout) with the scale factors and the per-row scale
    of row ``r`` stored at row ``map[r]``: the loads, the reductions, the chains and the code stores are the parent's; the
    rows past ``M`` have no map entry and store nothing."""

    def __init__(self, dtype: cutlass.Numeric, Kbp: int, cfg: QuantConfig):
        super().__init__(dtype, Kbp, cfg, PER_TOKEN, "128x4")
        assert not self.tile, "the row map serves the per-token row kernel"

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mP: cute.Tensor,
        mG: cute.Tensor,
        mR: cute.Tensor,
        M: Int64,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel_mapped(
            mX,
            mY,
            mS,
            mP,
            mG,
            mR,
            M,
            K,
            enable_pdl,
            self._tv_in(),
            self._tv_out(),
            (self.rows, self.span),
            (self.rows, self.span // 2),
        ).launch(
            grid=[cute.ceil_div(M, self.rows), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel_mapped(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mP: cute.Tensor,
        mG: cute.Tensor,
        mR: cute.Tensor,
        M: Int64,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_in: cute.Layout,
        tv_out: cute.Layout,
        tiler_in: cute.Shape,
        tiler_out: cute.Shape,
    ):
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
            pdl.trigger_one_wave()
        W, Lr, rows, bpt, V = self.W, self.Lr, self.rows, self.bpt, self.V
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        # ---- unit coordinates: (row in CTA, warp in row, lane in row) ----------------
        if cutlass.const_expr(self.cfg.threads_per_row >= LANES):
            row_in_cta = warp_idx // W
            w = warp_idx % W
            t = lane_idx
        else:
            row_in_cta = warp_idx * self.rows_per_warp + lane_idx // Lr
            w = Int32(0)
            t = lane_idx % Lr
        vtid = t + Lr * row_in_cta
        coords = {"lane": t, "warp": w, "cta": 0, "grid": 0}
        row = bidx.to(Int64) * rows + row_in_cta.to(Int64)
        nb = K // BLOCK
        pcols = ((nb + (SF_COL_TILE - 1)) // SF_COL_TILE) * SF_COL_TILE
        # the destination row of the scale factors and the per-row scale: the row's map entry (a row past M has none and
        # stores nothing: the store bound is M)
        srow = mR[cutlass.min(row, M - Int64(1))].to(Int64)
        # a row without a map entry (the plan marks the rows past the live ones -1: the dropped slots' share of the
        # buffer) keeps its codes in place and stores no scale factor or row scale
        mapped = (row < M) & (srow >= Int64(0))
        srow = cutlass.max(srow, Int64(0))

        xbuf = None
        if cutlass.const_expr(self.needs_xbuf):
            smem = cutlass.utils.SmemAllocator()
            xbuf = smem.allocate_tensor(
                Float32, cute.make_layout((rows, 1, W, LANES)), byte_alignment=16
            )
        peers = _CutePeers(xbuf, row_in_cta, w, lane_idx)

        # ---- loads: the thread's whole span, zero-filled beyond (M, K) ------------------
        thr_x, atom_x = _tiled_copy(
            mX, self.VEC * mX.element_type.width, tv_in, tiler_in, vtid
        )
        thr_y, atom_y = _tiled_copy(mY, (V // 2) * 8, tv_out, tiler_out, vtid)
        idX = cute.make_identity_tensor(mX.shape)
        gX = thr_x.partition_S(cute.local_tile(mX, tiler_in, (bidx, w)))
        pX = _predicate(
            thr_x.partition_S(cute.local_tile(idX, tiler_in, (bidx, w))), M, K
        )

        src = _Span(self, gX, atom_x, pX, mX.element_type)
        frag = src.load()

        # ===== the amax reductions: driven by the plan ==================================
        state, start = canon.init_state(self.plan, fp, _Loader(self, frag))
        state, nxt = canon.run_steps(
            self.plan, state, start, peers, fp, stop_before="blk"
        )
        amax = [state[self._key_for(kk)] for kk in range(bpt)]

        g = mG[0]
        ramax = canon.run(self.plan, dict(state), nxt, peers, fp)
        g, pts = row_scale_chain(fp, ramax, g)
        if (t == 0) & (w == 0) & mapped:
            mP[srow] = pts
        if cutlass.const_expr(self.cfg.stage == "reread"):
            frag = src.load()

        # ===== outputs: per block the scale chain, the element chain, the stores ===========
        idY = cute.make_identity_tensor(mY.shape)
        gY = thr_y.partition_S(cute.local_tile(mY, tiler_out, (bidx, w)))
        pY = _predicate(
            thr_y.partition_S(cute.local_tile(idY, tiler_out, (bidx, w))), M, K // 2
        )
        fy_words = cute.make_rmem_tensor(cute.make_layout(((V // 8) * bpt,)), Uint32)
        fy = cute.make_tensor(
            cute.recast_ptr(fy_words.iterator, dtype=Uint8),
            cute.make_fragment_like(gY).layout,
        )
        half0 = (t % self.lpb) == 0
        for kk in cutlass.range_constexpr(bpt):
            sf_byte = _quantize_block(self, fp, frag, kk, amax[kk], g, fy_words)
            blk = self.plan.digit_value("blk", coords, self._key_for(kk))
            if half0 & (blk < pcols) & mapped:
                mS[self._sf_off(srow, blk, pcols)] = sf_byte.to(Uint8)
        cute.copy(atom_y, fy, gY, pred=pY)
        if cutlass.const_expr(self.Kbp < SF_COL_TILE):
            # the 128x4 layout pads scale columns to 4; a bucket of 1 or 2 blocks owns none of columns Kbp..3
            if (t == 0) & (w == 0) & mapped:
                for c in cutlass.range_constexpr(self.Kbp, SF_COL_TILE):
                    mS[self._sf_off(srow, Int32(c), pcols)] = Uint8(0)


class ScatterQuantNvFP4Quantizer(NvFP4QuantKernel):
    """The per-token row kernel of :class:`NvFP4QuantKernel` (the 128x4 layout) storing token ``t``'s codes at the ``k``
    dense rows ``dest[t k + j]`` of the grouped operand and its scale factors and per-token scale at the padded positions
    ``dest_pad[t k + j]``: the loads, the reductions and the chains are the parent's; the code bytes of a thread's block are
    stored as words at the destination row (the parent's tiled store addresses the token row)."""

    def __init__(self, dtype: cutlass.Numeric, Kbp: int, cfg: QuantConfig, k: int):
        super().__init__(dtype, Kbp, cfg, PER_TOKEN, "128x4")
        assert not self.tile, "the scatter serves the per-token row kernel"
        assert self.V % 8 == 0 and k >= 1
        self.k = k

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mP: cute.Tensor,
        mG: cute.Tensor,
        mD: cute.Tensor,
        mDP: cute.Tensor,
        M: Int64,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel_scatter(
            mX,
            mY,
            mS,
            mP,
            mG,
            mD,
            mDP,
            M,
            K,
            enable_pdl,
            self._tv_in(),
            (self.rows, self.span),
        ).launch(
            grid=[cute.ceil_div(M, self.rows), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel_scatter(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mP: cute.Tensor,
        mG: cute.Tensor,
        mD: cute.Tensor,
        mDP: cute.Tensor,
        M: Int64,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_in: cute.Layout,
        tiler_in: cute.Shape,
    ):
        fp = CuteF32
        bidx, _, _ = cute.arch.block_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
            pdl.trigger_one_wave()
        W, Lr, rows, bpt, V = self.W, self.Lr, self.rows, self.bpt, self.V
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        if cutlass.const_expr(self.cfg.threads_per_row >= LANES):
            row_in_cta = warp_idx // W
            w = warp_idx % W
            t = lane_idx
        else:
            row_in_cta = warp_idx * self.rows_per_warp + lane_idx // Lr
            w = Int32(0)
            t = lane_idx % Lr
        vtid = t + Lr * row_in_cta
        coords = {"lane": t, "warp": w, "cta": 0, "grid": 0}
        row = bidx.to(Int64) * rows + row_in_cta.to(Int64)
        nb = K // BLOCK
        pcols = ((nb + (SF_COL_TILE - 1)) // SF_COL_TILE) * SF_COL_TILE
        valid = row < M

        xbuf = None
        if cutlass.const_expr(self.needs_xbuf):
            smem = cutlass.utils.SmemAllocator()
            xbuf = smem.allocate_tensor(
                Float32, cute.make_layout((rows, 1, W, LANES)), byte_alignment=16
            )
        peers = _CutePeers(xbuf, row_in_cta, w, lane_idx)

        thr_x, atom_x = _tiled_copy(
            mX, self.VEC * mX.element_type.width, tv_in, tiler_in, vtid
        )
        idX = cute.make_identity_tensor(mX.shape)
        gX = thr_x.partition_S(cute.local_tile(mX, tiler_in, (bidx, w)))
        pX = _predicate(
            thr_x.partition_S(cute.local_tile(idX, tiler_in, (bidx, w))), M, K
        )
        src = _Span(self, gX, atom_x, pX, mX.element_type)
        frag = src.load()

        state, start = canon.init_state(self.plan, fp, _Loader(self, frag))
        state, nxt = canon.run_steps(
            self.plan, state, start, peers, fp, stop_before="blk"
        )
        amax = [state[self._key_for(kk)] for kk in range(bpt)]
        g = mG[0]
        ramax = canon.run(self.plan, dict(state), nxt, peers, fp)
        g, pts = row_scale_chain(fp, ramax, g)
        if cutlass.const_expr(self.cfg.stage == "reread"):
            frag = src.load()

        # the token's codes and scale bytes, computed once
        fy_words = cute.make_rmem_tensor(cute.make_layout(((V // 8) * bpt,)), Uint32)
        sf_bytes = cute.make_rmem_tensor(cute.make_layout((bpt,)), Uint8)
        for kk in cutlass.range_constexpr(bpt):
            sf_bytes[kk] = _quantize_block(
                self, fp, frag, kk, amax[kk], g, fy_words
            ).to(Uint8)
        # the stores: the k expanded rows of the token (the dense row for the codes, the padded position for the scales)
        y32 = cute.make_tensor(
            cute.recast_ptr(mY.iterator, dtype=Uint32),
            cute.make_layout((cute.size(mY) // 4,)),
        )
        row_words = mY.stride[0] // 4  # the dense operand's row stride in words
        half = V // 2
        lane_off = (t % self.lpb) * half
        half0 = (t % self.lpb) == 0
        last = M * Int64(self.k) - Int64(1)
        for j in cutlass.range(0, self.k, 1):
            i = cutlass.min(row * Int64(self.k) + Int64(j), last)
            d = mD[i].to(Int64)
            dp = mDP[i].to(Int64)
            # a dropped slot (an expert id outside [0, E)) has no position: the plan marks it -1 and the slot stores nothing
            live = valid & (d >= Int64(0))
            d = cutlass.max(d, Int64(0))
            dp = cutlass.max(dp, Int64(0))
            if (t == 0) & (w == 0) & live:
                mP[dp] = pts
            for kk in cutlass.range_constexpr(bpt):
                blk = self.plan.digit_value("blk", coords, self._key_for(kk))
                if half0 & (blk < pcols) & live:
                    mS[self._sf_off(dp, blk, pcols)] = sf_bytes[kk]
                if (blk < nb) & live:
                    base = d * Int64(row_words) + (
                        (blk * (BLOCK // 2) + lane_off) // 4
                    ).to(Int64)
                    for q in cutlass.range_constexpr(V // 8):
                        y32[base + Int64(q)] = fy_words[kk * (V // 8) + q]
            if cutlass.const_expr(self.Kbp < SF_COL_TILE):
                if (t == 0) & (w == 0) & live:
                    for c in cutlass.range_constexpr(self.Kbp, SF_COL_TILE):
                        mS[self._sf_off(dp, Int32(c), pcols)] = Uint8(0)


def get_compiled_rowmap_quant(
    dtype_str: str, K: int, cfg: QuantConfig, enable_pdl: bool = True
) -> Any:
    """``kernel(x [M, K], fp4 [M, K/2] u8, sf_pad flat u8, pts_pad f32, gs_inv [1] f32, row_map [M] i32, M, K)``: one kernel
    per (dtype, Kbp bucket, config, pdl), as the quantizer's."""
    return _compile_bucket(dtype_str, cfg.bucket(K), cfg, bool(enable_pdl))


@functools.cache
def _compile_bucket(dtype_str: str, Kbp: int, cfg: QuantConfig, enable_pdl: bool):
    from ....batch_invariant.fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = sanitize_symbol_name(
        f"moe_quant_pertoken_rowmap_{dtype_str}_kbp{Kbp}_{cfg.describe()}_pdl{int(enable_pdl)}"
    )

    def compile_fn(**compile_options):
        sym_m, sym_k, sym_k2, sym_n, sym_m2, sym_r = (
            cute.sym_int(64),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(64),
            cute.sym_int(64),
            cute.sym_int(64),
        )
        x = cute.runtime.make_fake_tensor(
            dtype,
            (sym_m, sym_k),
            (cute.sym_int64(divisibility=VEC_BYTES // (dtype.width // 8)), 1),
            assumed_align=16,
        )
        y = cute.runtime.make_fake_tensor(
            Uint8, (sym_m, sym_k2), (cute.sym_int64(divisibility=8), 1), assumed_align=8
        )
        s = cute.runtime.make_fake_tensor(Uint8, (sym_n,), (1,), assumed_align=16)
        p = cute.runtime.make_fake_tensor(Float32, (sym_m2,), (1,), assumed_align=4)
        g = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=16)
        r = cute.runtime.make_fake_tensor(Int32, (sym_r,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            RowMappedNvFP4QuantKernel(dtype, Kbp, cfg),
            x,
            y,
            s,
            p,
            g,
            r,
            Int64(1),
            Int32(BLOCK),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(MODULE, name, compile_fn)


def get_compiled_scatter_quant(
    dtype_str: str, K: int, cfg: QuantConfig, k: int, enable_pdl: bool = True
) -> Any:
    """``kernel(x [T, K], x_perm4 [rows, K/2] u8, sf_pad flat u8, pts_pad f32, gs_inv [1] f32, dest [T k] i32, dest_pad
    [T k] i32, T, K)``: one kernel per (dtype, Kbp bucket, config, k, pdl)."""
    return _compile_scatter_bucket(dtype_str, cfg.bucket(K), cfg, k, bool(enable_pdl))


@functools.cache
def _compile_scatter_bucket(
    dtype_str: str, Kbp: int, cfg: QuantConfig, k: int, enable_pdl: bool
):
    from ....batch_invariant.fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = sanitize_symbol_name(
        f"moe_quant_pertoken_scatter_{dtype_str}_kbp{Kbp}_k{k}_{cfg.describe()}_pdl{int(enable_pdl)}"
    )

    def compile_fn(**compile_options):
        sym_m, sym_k, sym_k2, sym_n, sym_m2, sym_r, sym_p = (
            cute.sym_int(64),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(64),
            cute.sym_int(64),
            cute.sym_int(64),
            cute.sym_int(64),
        )
        x = cute.runtime.make_fake_tensor(
            dtype,
            (sym_m, sym_k),
            (cute.sym_int64(divisibility=VEC_BYTES // (dtype.width // 8)), 1),
            assumed_align=16,
        )
        y = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_p, sym_k2),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        s = cute.runtime.make_fake_tensor(Uint8, (sym_n,), (1,), assumed_align=16)
        p = cute.runtime.make_fake_tensor(Float32, (sym_m2,), (1,), assumed_align=4)
        g = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=16)
        d = cute.runtime.make_fake_tensor(Int32, (sym_r,), (1,), assumed_align=4)
        dp = cute.runtime.make_fake_tensor(Int32, (sym_r,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            ScatterQuantNvFP4Quantizer(dtype, Kbp, cfg, k),
            x,
            y,
            s,
            p,
            g,
            d,
            dp,
            Int64(1),
            Int32(BLOCK),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(MODULE, name, compile_fn)
