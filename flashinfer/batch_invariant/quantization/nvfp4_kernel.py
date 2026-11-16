# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 quantization kernel on the canon engine: the per-tensor class ``nvfp4.quant`` and the per-token class
``nvfp4.quant.pertoken`` (``nvfp4_spec.Variant``), bf16 / f16 / fp32 inputs, the scale factors in the 128x4 swizzled,
the 8x4 swizzled or the linear layout (``layout``: a placement of the store).

The reductions are not written here: the kernel builds a :class:`canon.Plan` from the config's placement of the row
amax reduction (digits ``elem``, ``blk``), loads the leaves through ``canon.init_state``, runs the ``elem`` digit's
steps with ``canon.run_steps`` (the block amaxes, one per owned block) and, for the per-token class, the ``blk``
digit's steps with ``canon.run`` (the row amax) through a hardware :class:`canon.Peers` (shuffles, smem exchange). The
chains are ``nvfp4_spec.sf_value`` / ``inv_scale`` / ``quant_pre`` / ``row_scale_chain`` through ``CuteF32``; the two
conversions are the declared opaque primitives :func:`cvt_f32_to_e4m3` (byte + the exactly widened value) and
``blocks.cvt_f32x8_to_e2m1x8``, pinned by the IR test and verified against the CPU model over every binary32 input.
Kernel-specific is only the plumbing: tiled copies, predicates, the e2m1 pack, the scale store.

``K`` is a runtime argument; a kernel is compiled per ``Kbp`` bucket (``padded_blocks(K)``). Blocks beyond ``K / 16``
and rows beyond ``M`` are zero-filled by the predicated loads, which is exact for the max of absolute values
(``Digit.identity_note``) and produces the zero scale bytes the padded layouts require; the e2m1 stores of those blocks
and rows are predicated off. The tile placement (a 128-row x column-tile CTA that stages the scale bytes in shared memory
and writes each 512-byte scale tile with 16-byte stores) serves the per-tensor class in the 128x4 layout.
"""

from __future__ import annotations

from typing import List, Optional

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .. import canon, jit
from .blocks import (
    byte_address,
    cvt_f32x8_to_e2m1x8,
    sf_offset_8x4_dev,
    sf_offset_128x4_dev,
    sf_offset_linear_dev,
    st_global_v4_u32,
)
from .cute_ext import CuteF32
from .layout import layout_name, sanitize_symbol_name
from .nvfp4_schedule import BLOCK, LANES, SF_COL_TILE, SF_ROW_TILE, QuantConfig
from .nvfp4_spec import (
    PER_TENSOR,
    Variant,
    inv_scale,
    leaf_value,
    quant_pre,
    row_reduction,
    row_scale_chain,
    sf_value,
)

NVFP4_QUANT_MODULE = "batch_invariant_nvfp4_quant"  # the on-disk cache module of every NVFP4 quantizer kernel
_LAYOUT_TAG = {"128x4": "sw", "8x4": "8x4", "linear": "lin"}


def nvfp4_quant_kernel_name(
    variant_name: str,
    dtype_str: str,
    Kbp: int,
    cfg_describe: str,
    layout: str,
    enable_pdl: bool,
) -> str:
    """The cache name of one specialization: the class, the input dtype, the Kbp bucket, the configuration, the
    scale-factor layout and PDL; never M (one specialization serves every row count)."""
    return sanitize_symbol_name(
        f"{variant_name}_{dtype_str}_Kbp{Kbp}_{cfg_describe}_{_LAYOUT_TAG[layout]}_pdl{int(enable_pdl)}"
    )


VEC_BYTES = 16  # bytes per input vector; elements per vector = 16 / element bytes (8 for bf16 / f16, 4 for fp32)


# ----------------------------------------------------------------------------- declared conversions
@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None):
    """``cvt.rn.satfinite.e4m3x2.f32`` of one value. Returns ``(byte, value)``: the e4m3 byte (in a
    ``Uint32``) and the byte's value widened back to f32 exactly (``cvt.rn.f16x2.e4m3x2`` then
    ``cvt.f32.f16``: every e4m3 value is an f16 value is an f32 value)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Float32(a).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 pair, lo, hi;
            .reg .b32 h2;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 pair, zero, $2;
            cvt.rn.f16x2.e4m3x2 h2, pair;
            mov.b32 {lo, hi}, h2;
            cvt.f32.f16 $1, lo;
            cvt.u32.u16 $0, pair;
        }
        """,
        "=r,=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


# ----------------------------------------------------------------------------- plumbing
@cute.jit
def _predicate(tXcX: cute.Tensor, M: Int64, N: Int32) -> cute.Tensor:
    """Per-vector predicate: (row, col) < (M, N)."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k], (M, N)
            )
    return tXpX


def _tiled_copy(mT, bits, tv_layout, tiler_mn, vtid):
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), mT.element_type, num_bits_per_copy=bits
    )
    return cute.make_tiled_copy(atom, tv_layout, tiler_mn).get_slice(vtid), atom


class _CutePeers(canon.Peers):
    """Hardware peers: xor shuffles for lane steps, an smem exchange for warp steps (one entry per
    live key; the quantizer's warp step sees one)."""

    def __init__(self, xbuf, group, warp_id, lane_id):
        self.xbuf, self.group, self.warp_id, self.lane_id = (
            xbuf,
            group,
            warp_id,
            lane_id,
        )
        self.n_exchanges = 0

    def shuffle_xor(self, key, value, mask):
        if value is None:
            return None
        return cute.arch.shuffle_sync_bfly(value, offset=mask)

    def exchange(self, step, state):
        keys = [k for k, v in state.items() if v is not None]
        if self.n_exchanges:
            cute.arch.barrier()
        for e, k in enumerate(keys):
            self.xbuf[self.group, e, self.warp_id, self.lane_id] = state[k]
        cute.arch.barrier()
        self.n_exchanges += 1
        if step.warp_stride == 1 and step.radix == step.warp_total:
            sib_ids = list(range(step.radix))
        else:
            my = (self.warp_id // step.warp_stride) % step.radix
            sib_ids = [
                self.warp_id + (s - my) * step.warp_stride for s in range(step.radix)
            ]
        out = {k: [None] * step.radix for k in state}
        for e, k in enumerate(keys):
            out[k] = [self.xbuf[self.group, e, sid, self.lane_id] for sid in sib_ids]
        return out


class NvFP4QuantKernel:
    """NVFP4 quantization for one ``Kbp`` bucket (K is a runtime argument). Outputs: packed e2m1 bytes
    (M, K/2), the 128x4 swizzled scale bytes (flat, ``padded_M * padded_cols``) and, for the per-token
    class, the per-row scale (M,) f32."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        Kbp: int,
        cfg: QuantConfig,
        variant: Variant = PER_TENSOR,
        layout: str = "128x4",
    ):
        K = (
            Kbp * BLOCK
        )  # a representative K of the bucket: everything below depends on K only via Kbp
        cfg.validate(K, dtype.width // 8)
        self.VEC = (
            VEC_BYTES // (dtype.width // 8)
        )  # input elements per 16-byte vector (the fp32 input is the same class: the leaf is the fp32 value)
        self.layout = layout_name(layout)
        self.tile = cfg.mode == "tile"
        assert not (self.tile and variant.per_token), (
            "the tile placement serves the per-tensor class"
        )
        assert not self.tile or Kbp == cfg.tile_blocks
        assert not self.tile or self.layout == "128x4", (
            "the tile placement writes the 128x4 layout"
        )
        self.dtype, self.K, self.Kbp, self.cfg, self.variant = (
            dtype,
            K,
            Kbp,
            cfg,
            variant,
        )
        self.red = row_reduction(K, leaf_x=None)  # the leaves come from the loader
        self.plan = canon.plan(self.red, cfg.placement(K))
        assert self.plan.units["cta"] == 1 and self.plan.units["grid"] == 1
        self.V = cfg.elems_per_thread
        self.ri_elem = self.plan.reg_index("elem")
        # the block digit's register factor is innermost (pos 0) or second (pos 1) depending on blk_order
        self.ri_blk = next(
            (i for i, (d, _, _) in enumerate(self.plan.reg_factors) if d == "blk"), None
        )
        self.num_threads = cfg.num_threads
        if self.tile:
            # a warp streams tile_passes rows of a 128-row x block_lanes-block tile, one row per pass
            # (the threads_per_row = 32 placement: lpb lanes per block, bl blocks per warp)
            self.W, self.Lr, self.lpb, self.bl, self.bpt = (
                cfg.tile_warps,
                LANES,
                cfg.lanes_per_block,
                cfg.block_lanes,
                cfg.tile_bpt,
            )
            self.rows, self.rows_per_warp = 1, 1
            self.passes = cfg.tile_passes
            self.lane_major = True
            self.needs_xbuf = False
            self.sf_tile_bytes = (
                SF_ROW_TILE * cfg.tile_blocks
            )  # the CTA's scale region: tile_blocks / 4 contiguous 512-byte tiles
        else:
            self.W = self.plan.units["warp"]
            self.Lr = self.plan.units["lane"]
            assert self.Lr == cfg.lanes_per_row
            self.lpb, self.bl = cfg.lanes_per_block, cfg.block_lanes
            self.bpt = cfg.blocks_per_thread(K)
            self.rows, self.rows_per_warp = cfg.rows_per_cta, cfg.rows_per_warp
            self.lane_major = cfg.blk_order == "lane"
            self.needs_xbuf = variant.per_token and self.W > 1
        self.span = BLOCK * self.bl * self.bpt  # elements of a row one warp covers

    # -- layouts ---------------------------------------------------------------------
    def _tv_in(self):
        rows, V, lpb, bl, bpt, VEC = (
            self.rows,
            self.V,
            self.lpb,
            self.bl,
            self.bpt,
            self.VEC,
        )
        bi_stride, kk_stride = (
            (BLOCK, BLOCK * bl) if self.lane_major else (BLOCK * bpt, BLOCK)
        )
        shape = ((lpb, bl, rows), (VEC, V // VEC, bpt))
        stride = ((V * rows, bi_stride * rows, 1), (rows, VEC * rows, kk_stride * rows))
        return cute.make_layout(shape, stride=stride)

    def _tv_out(self):
        rows, V, lpb, bl, bpt = self.rows, self.V, self.lpb, self.bl, self.bpt
        half = V // 2  # output bytes per thread per block
        bi_stride, kk_stride = (
            (BLOCK // 2, BLOCK // 2 * bl)
            if self.lane_major
            else (BLOCK // 2 * bpt, BLOCK // 2)
        )
        shape = ((lpb, bl, rows), (half, 1, bpt))
        stride = (
            (half * rows, bi_stride * rows, 1),
            (rows, half * rows, kk_stride * rows),
        )
        return cute.make_layout(shape, stride=stride)

    def _smem_bytes(self) -> int:
        return self.rows * self.W * LANES * 4 if self.needs_xbuf else 0

    def _key_for(self, kk: int) -> tuple:
        key: List[Optional[int]] = [None] * self.plan.n_reg
        if self.ri_blk is not None:
            key[self.ri_blk] = kk
        return tuple(key)

    def _frag_index(self, kk: int, v: int) -> int:
        return v + self.V * kk

    def _sf_off(self, row, col, pcols):
        """The byte offset of (row, col) in the scale buffer under the kernel's layout (``pcols``: the padded column count of
        the swizzled layouts, the column count of the linear one)."""
        if self.layout == "128x4":
            return sf_offset_128x4_dev(row, col, pcols)
        if self.layout == "8x4":
            return sf_offset_8x4_dev(row, col, pcols)
        return sf_offset_linear_dev(row, col, pcols)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mP: cute.Tensor,
        mG: cute.Tensor,
        M: Int64,
        K: Int32,
        padded_M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        if cutlass.const_expr(self.tile):
            self.kernel_tile(
                mX,
                mY,
                mS,
                mG,
                M,
                K,
                padded_M,
                enable_pdl,
                self._tv_in(),
                self._tv_out(),
                (self.rows, self.span),
                (self.rows, self.span // 2),
            ).launch(
                grid=[
                    cute.ceil_div(K // BLOCK, self.bl * self.bpt),
                    cute.ceil_div(padded_M, SF_ROW_TILE),
                    1,
                ],
                block=[self.num_threads, 1, 1],
                smem=self.sf_tile_bytes + 128,
                stream=stream,
                use_pdl=enable_pdl,
            )
        else:
            self.kernel(
                mX,
                mY,
                mS,
                mP,
                mG,
                M,
                K,
                padded_M,
                enable_pdl,
                self._tv_in(),
                self._tv_out(),
                (self.rows, self.span),
                (self.rows, self.span // 2),
            ).launch(
                grid=[cute.ceil_div(padded_M, self.rows), 1, 1],
                block=[self.num_threads, 1, 1],
                smem=self._smem_bytes(),
                stream=stream,
                use_pdl=enable_pdl,
            )

    @cute.kernel
    def kernel_tile(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mG: cute.Tensor,
        M: Int64,
        K: Int32,
        padded_M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        tv_in: cute.Layout,
        tv_out: cute.Layout,
        tiler_in: cute.Shape,
        tiler_out: cute.Shape,
    ):
        """Tile placement (per-tensor class): CTA (bidx, bidy) = column tile bidx (bl blocks) of row tile
        bidy (128 rows). Warp w streams rows w, w + W, ... of the tile; a row pass is the
        ``threads_per_row = 32`` placement (lpb lanes per block): the same engine plan and chains as the
        row kernel. The scale bytes are staged in shared memory in the 128x4 layout and the tile's
        contiguous ``128 * bl``-byte scale region is written with one 16-byte store per thread."""
        fp = CuteF32
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        W, V, lpb, bl, bpt = self.W, self.V, self.lpb, self.bl, self.bpt
        tb = bl * bpt  # blocks per column tile
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        nb = K // BLOCK
        pcols = ((nb + (SF_COL_TILE - 1)) // SF_COL_TILE) * SF_COL_TILE
        coords = {"lane": lane_idx, "warp": 0, "cta": 0, "grid": 0}
        smem = cutlass.utils.SmemAllocator()
        sbuf = smem.allocate_tensor(
            Uint8, cute.make_layout((self.sf_tile_bytes,)), byte_alignment=16
        )
        thr_x, atom_x = _tiled_copy(
            mX, self.VEC * mX.element_type.width, tv_in, tiler_in, lane_idx
        )
        thr_y, atom_y = _tiled_copy(mY, (V // 2) * 8, tv_out, tiler_out, lane_idx)
        idX = cute.make_identity_tensor(mX.shape)
        idY = cute.make_identity_tensor(mY.shape)
        gs = mG[0]
        half0 = (lane_idx % lpb) == 0
        sf_lane_off = _tile_sf_offsets(self, coords)
        for p in cutlass.range_constexpr(self.passes):
            r = warp_idx + W * p  # row within the 128-row tile
            row = bidy.to(Int64) * SF_ROW_TILE + r.to(
                Int64
            )  # global row (tile coordinate: rows = 1)
            gX = thr_x.partition_S(cute.local_tile(mX, tiler_in, (row, bidx)))
            pX = _predicate(
                thr_x.partition_S(cute.local_tile(idX, tiler_in, (row, bidx))), M, K
            )
            frag = _Span(self, gX, atom_x, pX, mX.element_type).load()
            state, start = canon.init_state(self.plan, fp, _Loader(self, frag))
            state, _ = canon.run_steps(
                self.plan,
                state,
                start,
                _CutePeers(None, 0, 0, lane_idx),
                fp,
                stop_before="blk",
            )
            fy_words = cute.make_rmem_tensor(
                cute.make_layout(((V // 8) * bpt,)), Uint32
            )
            gY = thr_y.partition_S(cute.local_tile(mY, tiler_out, (row, bidx)))
            fy = cute.make_tensor(
                cute.recast_ptr(fy_words.iterator, dtype=Uint8),
                cute.make_fragment_like(gY).layout,
            )
            r_off = (r % 32) * 16 + (r // 32) * 4
            for kk in cutlass.range_constexpr(bpt):
                sf_byte = _quantize_block(
                    self, fp, frag, kk, state[self._key_for(kk)], gs, fy_words
                )
                if half0:
                    sbuf[sf_lane_off[kk] + r_off] = sf_byte.to(Uint8)
            pY = _predicate(
                thr_y.partition_S(cute.local_tile(idY, tiler_out, (row, bidx))),
                M,
                K // 2,
            )
            cute.copy(atom_y, fy, gY, pred=pY)
        cute.arch.barrier()
        # copy-out: the tile's scale region is contiguous in global memory (tb / 4 consecutive 512-byte
        # tiles); 16 bytes per thread; groups beyond the last real column group do not exist
        s32 = cute.make_tensor(
            cute.recast_ptr(sbuf.iterator, dtype=Uint32),
            cute.make_layout((self.sf_tile_bytes // 4,)),
        )
        n_chunks = self.sf_tile_bytes // 16
        ncg = pcols // SF_COL_TILE
        base = bidy.to(Int64) * (SF_ROW_TILE * pcols).to(Int64) + (
            bidx * (SF_ROW_TILE * tb)
        ).to(Int64)
        for c0 in cutlass.range_constexpr(0, n_chunks, self.num_threads):
            c = c0 + tidx
            if (c < n_chunks) & (bidx * (tb // SF_COL_TILE) + c // 32 < ncg):
                st_global_v4_u32(
                    byte_address(mS, base + (c * 16).to(Int64)),
                    s32[4 * c],
                    s32[4 * c + 1],
                    s32[4 * c + 2],
                    s32[4 * c + 3],
                )
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        mP: cute.Tensor,
        mG: cute.Tensor,
        M: Int64,
        K: Int32,
        padded_M: Int64,
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
        if cutlass.const_expr(self.layout == "linear"):
            pcols = nb  # the linear layout has no column padding
        else:
            pcols = ((nb + (SF_COL_TILE - 1)) // SF_COL_TILE) * SF_COL_TILE
        srow = row
        store_bound = padded_M  # the padded rows store their (zero) scale bytes too

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
        amax = [
            state[self._key_for(kk)] for kk in range(bpt)
        ]  # the block amaxes (complete)

        g = mG[0]
        if cutlass.const_expr(self.variant.per_token):
            ramax = canon.run(self.plan, dict(state), nxt, peers, fp)
            g, pts = row_scale_chain(fp, ramax, g)
            if (t == 0) & (w == 0) & (row < M):
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
            if half0 & (blk < pcols) & (row < store_bound):
                mS[self._sf_off(srow, blk, pcols)] = sf_byte.to(Uint8)
        cute.copy(atom_y, fy, gY, pred=pY)
        if cutlass.const_expr(self.Kbp < SF_COL_TILE and self.layout != "linear"):
            # the swizzled layouts pad scale columns to 4; a bucket of 1 or 2 blocks owns none of columns Kbp..3
            if (t == 0) & (w == 0) & (row < store_bound):
                for c in cutlass.range_constexpr(self.Kbp, SF_COL_TILE):
                    mS[self._sf_off(srow, Int32(c), pcols)] = Uint8(0)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


class _Span:
    """A thread's whole row span as one zero-filled, predicated register fragment."""

    def __init__(self, k, gX, atom, pred, dtype):
        self.k, self.gX, self.atom, self.pred, self.dtype = k, gX, atom, pred, dtype

    def load(self):
        frag = cute.make_fragment_like(self.gX)
        frag.store(cute.zeros_like(frag, dtype=self.dtype))
        cute.copy(self.atom, self.gX, frag, pred=self.pred)
        return frag


class _Loader:
    """``canon.init_state`` leaf loader: the spec's leaf (``fabs`` of the widened element) for the
    register key ``(elem value, block register index)``."""

    def __init__(self, k, frag):
        self.k, self.frag = k, frag

    def __call__(self, key):
        k = self.k
        kk = key[k.ri_blk] if k.ri_blk is not None else 0
        return leaf_value(
            CuteF32, self.frag[k._frag_index(kk, key[k.ri_elem])].to(Float32)
        )


def _quantize_block(k, fp, frag, kk: int, amax, g, fy_words):
    """The per-block chain (``spec.sf_value`` -> e4m3 -> ``spec.inv_scale``) and the element chain
    (``spec.quant_pre`` -> e2m1 pack) of block ``kk`` of the span; writes the packed words, returns
    the scale byte."""
    V = k.V
    sf_byte, sf_d = cvt_f32_to_e4m3(sf_value(fp, amax, g))
    inv = inv_scale(fp, sf_d, g)
    ys = [quant_pre(fp, frag[k._frag_index(kk, v)].to(Float32), inv) for v in range(V)]
    for q in range(V // 8):
        fy_words[kk * (V // 8) + q] = cvt_f32x8_to_e2m1x8(*ys[8 * q : 8 * q + 8])
    return sf_byte


def _tile_sf_offsets(k, coords):
    """Tile mode: for each of the lane's blocks within the column tile, the smem byte offset of its scale
    in the 128x4 layout before the row term (``(blk // 4) * 512 + blk % 4``)."""
    out = []
    for kk in range(k.bpt):
        blk_local = k.plan.digit_value("blk", coords, k._key_for(kk))
        out.append((blk_local // SF_COL_TILE) * 512 + (blk_local % SF_COL_TILE))
    return out


# ----------------------------------------------------------------------------- compile
def get_compiled_kernel(
    dtype_str: str,
    K: int,
    cfg: QuantConfig,
    variant: Variant,
    enable_pdl: bool,
    layout: str = "128x4",
):
    """The compiled kernel serving row length ``K``: one kernel per (dtype, Kbp bucket, config, variant, layout, pdl);
    K, M and the row strides are runtime arguments (tile-mode kernels serve every K).
    ``kernel(x, fp4, sf, pts, gs, M, K, padded_M)``."""
    return _compile_bucket(
        dtype_str, cfg.bucket(K), cfg, variant, enable_pdl, layout_name(layout)
    )


@functools.cache
def _compile_bucket(
    dtype_str: str,
    Kbp: int,
    cfg: QuantConfig,
    variant: Variant,
    enable_pdl: bool,
    layout: str,
):
    from ..fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    name = nvfp4_quant_kernel_name(
        variant.name, dtype_str, Kbp, cfg.describe(), layout, enable_pdl
    )

    def compile_fn():
        sym_m, sym_k, sym_k2, sym_n, sym_m2 = (
            cute.sym_int(64),
            cute.sym_int(32),
            cute.sym_int(32),
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
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        obj = NvFP4QuantKernel(dtype, Kbp, cfg, variant, layout)
        return cute.compile(
            obj,
            x,
            y,
            s,
            p,
            g,
            Int64(1),
            Int32(BLOCK),
            Int64(128),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(NVFP4_QUANT_MODULE, name, compile_fn)
