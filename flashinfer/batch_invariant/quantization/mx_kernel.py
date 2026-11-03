"""The MX quantization kernel on the canon engine: the class ``mxfp8.quant`` (``mx_spec.MXFormat``) for bf16 / f16 / fp32
inputs, scale factors in the 128x4 swizzled or the linear
layout (a placement of the store).

The reductions are not written here: the kernel builds a :class:`canon.Plan` from the config's placement of the
kernels' reduction (digits ``elem``, ``blk``), loads the leaves through ``canon.init_state`` and runs the ``elem``
digit's steps with ``canon.run_steps`` (the block amaxes, one per owned block; the ``blk`` digit fixes the ownership
and is never combined). The chains are ``mx_spec.scale_norm`` -> ``blocks.cvt_f32_to_ue8m0`` -> ``mx_spec.inv_scale``
and ``mx_spec.quant_pre`` -> ``mx_spec.saturate`` -> ``blocks.pack_block_codes`` through ``CuteF32``. Kernel-specific
is only the plumbing: tiled copies, predicates, the code pack, the scale store of the chosen layout.

``K`` is a runtime argument; a kernel is compiled per ``Kbp`` bucket (``padded_blocks(K)``); ``M`` and the row strides
are runtime arguments of every kernel (the batch dimension is never a compile-time constant). Blocks beyond ``K / 32``
and rows beyond ``M`` are zero-filled by the predicated loads, which is exact for the max of absolute values
(``Digit.identity_note``) and produces the zero scale bytes the padded 128x4 layout requires; the element stores of
those blocks and rows are predicated off.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32

from .. import canon, jit
from .cute_ext import CuteF32
from .blocks import (
    LanePeers,
    SfTileStage,
    Span,
    cvt_f32_to_ue8m0,
    pack_block_codes,
    predicate,
    sf_offset_128x4_dev,
    sf_offset_8x4_dev,
    sf_offset_linear_dev,
    tiled_copy,
)
from .layout import SF_COL_TILE, SF_ROW_TILE, layout_name, sanitize_symbol_name
from .mx_schedule import BLOCK, LANES, MXQuantConfig
from .mx_spec import (
    DEFAULT_RULE,
    MXFP8,
    MXFormat,
    ScaleRule,
    inv_scale,
    leaf_value,
    quant_pre,
    row_reduction,
    saturate,
    scale_norm,
)

VEC_BYTES = 16  # bytes per input / output vector copy
MX_QUANT_MODULE = (
    "batch_invariant_mx_quant"  # the on-disk cache module of every MX quantizer kernel
)
_LAYOUT_TAG = {"128x4": "sw", True: "sw", "8x4": "8x4", "linear": "lin", False: "lin"}


def mx_quant_kernel_name(
    fmt_name: str,
    rule_name: str,
    dtype_str: str,
    Kbp: int,
    cfg_describe: str,
    swizzled,
    enable_pdl: bool,
) -> str:
    """The cache name of one specialization: the format, the recipe constants' name, the input dtype, the Kbp bucket,
    the configuration, the scale-factor layout (``sw`` = 128x4, ``8x4``, ``lin``) and PDL; never M (one specialization
    serves every row count)."""
    tag = _LAYOUT_TAG[swizzled]
    return sanitize_symbol_name(
        f"{fmt_name}_{rule_name}_{dtype_str}_Kbp{Kbp}_{cfg_describe}_{tag}_pdl{int(enable_pdl)}"
    )


class _Loader:
    """``canon.init_state`` leaf loader: the spec's leaf (``fabs`` of the widened element) for the register key
    ``(elem value, block register index)``."""

    def __init__(self, k, frag):
        self.k, self.frag = k, frag

    def __call__(self, key):
        k = self.k
        kk = key[k.ri_blk] if k.ri_blk is not None else 0
        return leaf_value(
            CuteF32, self.frag[k._frag_index(kk, key[k.ri_elem])].to(Float32)
        )


class MXQuantKernel:
    """MX quantization for one ``Kbp`` bucket (K is a runtime argument). Outputs: the element codes (e4m3 bytes, (M, K))
    and the UE8M0 scale bytes in the chosen layout (flat)."""

    def __init__(
        self,
        dtype: cutlass.Numeric,
        Kbp: int,
        cfg: MXQuantConfig,
        fmt: MXFormat = MXFP8,
        rule: ScaleRule = DEFAULT_RULE,
        swizzled="128x4",
    ):
        """``swizzled``: the scale-factor layout, ``"128x4"`` (or ``True``), ``"8x4"``, ``"linear"`` (or ``False``)."""
        K = (
            Kbp * BLOCK
        )  # a representative K of the bucket: everything below depends on K only via Kbp
        self.elem_bytes = dtype.width // 8
        self.layout = layout_name(swizzled)
        cfg.validate(K, self.elem_bytes, self.layout)
        self.dtype, self.K, self.Kbp, self.cfg, self.fmt, self.rule = (
            dtype,
            K,
            Kbp,
            cfg,
            fmt,
            rule,
        )
        self.swizzled = (
            self.layout != "linear"
        )  # padded rows / columns exist (both swizzles)
        self.VEC = (
            VEC_BYTES // self.elem_bytes
        )  # input elements per 16-byte vector (8 or 4)
        self.tile = cfg.mode == "tile"
        assert not self.tile or (self.layout == "128x4" and Kbp == cfg.tile_blocks)
        self.red = row_reduction(K, leaf_x=None)  # the leaves come from the loader
        self.plan = canon.plan(self.red, cfg.placement(K))
        assert (
            self.plan.units["cta"] == 1
            and self.plan.units["grid"] == 1
            and self.plan.units["warp"] == cfg.warps_per_row
        )
        self.V = cfg.elems_per_thread
        self.ri_elem = self.plan.reg_index("elem")
        self.ri_blk = next(
            (i for i, (d, _, _) in enumerate(self.plan.reg_factors) if d == "blk"), None
        )
        self.num_threads = cfg.num_threads
        self.lpb, self.bl = cfg.lanes_per_block, cfg.block_lanes
        self.ob = (
            self.V * fmt.elem_bits // 8
        )  # output bytes per thread per block (32 / 16 / 8 or 16 / 8 / 4)
        self.out_vec = min(VEC_BYTES, self.ob)  # bytes per output vector copy
        self.words_per_block = self.ob // 4
        self.block_bytes = (
            BLOCK * fmt.elem_bits // 8
        )  # output bytes per block (32 or 16)
        if self.tile:
            self.W, self.Lr, self.bpt = cfg.tile_warps, LANES, cfg.tile_bpt
            self.rows, self.rows_per_warp = 1, 1
            self.passes = cfg.tile_passes
            self.lane_major = True
            self.stage = SfTileStage(cfg.tile_blocks, self.num_threads)
        else:
            self.W, self.Lr = self.plan.units["warp"], self.plan.units["lane"]
            assert self.Lr == cfg.lanes_per_row
            self.bpt = cfg.blocks_per_thread(K)
            self.rows, self.rows_per_warp = cfg.rows_per_cta, cfg.rows_per_warp
            self.lane_major = cfg.blk_order == "lane"
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
        rows, lpb, bl, bpt, ob, ov, bb = (
            self.rows,
            self.lpb,
            self.bl,
            self.bpt,
            self.ob,
            self.out_vec,
            self.block_bytes,
        )
        bi_stride, kk_stride = (bb, bb * bl) if self.lane_major else (bb * bpt, bb)
        shape = ((lpb, bl, rows), (ov, ob // ov, bpt))
        stride = ((ob * rows, bi_stride * rows, 1), (rows, ov * rows, kk_stride * rows))
        return cute.make_layout(shape, stride=stride)

    def _key_for(self, kk: int) -> tuple:
        key: list = [None] * self.plan.n_reg
        if self.ri_blk is not None:
            key[self.ri_blk] = kk
        return tuple(key)

    def _frag_index(self, kk: int, v: int) -> int:
        return v + self.V * kk

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        K: Int32,
        padded_M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tiler_in, tiler_out = (
            (self.rows, self.span),
            (self.rows, self.span * self.fmt.elem_bits // 8),
        )
        if cutlass.const_expr(self.tile):
            self.kernel_tile(
                mX,
                mY,
                mS,
                M,
                K,
                padded_M,
                enable_pdl,
                self._tv_in(),
                self._tv_out(),
                tiler_in,
                tiler_out,
            ).launch(
                grid=[
                    cute.ceil_div(K // BLOCK, self.bl * self.bpt),
                    cute.ceil_div(padded_M, SF_ROW_TILE),
                    1,
                ],
                block=[self.num_threads, 1, 1],
                smem=self.stage.nbytes + 128,
                stream=stream,
                use_pdl=enable_pdl,
            )
        else:
            grid_rows = padded_M if cutlass.const_expr(self.swizzled) else M
            self.kernel(
                mX,
                mY,
                mS,
                M,
                K,
                padded_M,
                enable_pdl,
                self._tv_in(),
                self._tv_out(),
                tiler_in,
                tiler_out,
            ).launch(
                grid=[cute.ceil_div(grid_rows, self.rows), 1, 1],
                block=[self.num_threads, 1, 1],
                smem=0,
                stream=stream,
                use_pdl=enable_pdl,
            )

    @cute.kernel
    def kernel_tile(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        K: Int32,
        padded_M: Int64,
        enable_pdl: cutlass.Constexpr[bool],
        tv_in: cute.Layout,
        tv_out: cute.Layout,
        tiler_in: cute.Shape,
        tiler_out: cute.Shape,
    ):
        """Tile placement (128x4 layout): CTA (bidx, bidy) = column tile bidx of row tile bidy (128 rows). Warp w
        streams rows w, w + W, ... of the tile; a row pass is the ``threads_per_row = 32`` placement. The scale bytes are
        staged in shared memory in the 128x4 layout and the tile's contiguous scale region is written with one 16-byte
        store per thread."""
        fp = CuteF32
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
        W, lpb, bpt = self.W, self.lpb, self.bpt
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        nb = K // BLOCK
        pcols = ((nb + (SF_COL_TILE - 1)) // SF_COL_TILE) * SF_COL_TILE
        coords = {"lane": lane_idx, "warp": 0, "cta": 0, "grid": 0}
        smem = cutlass.utils.SmemAllocator()
        sbuf = smem.allocate_tensor(
            Uint8, cute.make_layout((self.stage.nbytes,)), byte_alignment=16
        )
        thr_x, atom_x = tiled_copy(
            mX, self.VEC * mX.element_type.width, tv_in, tiler_in, lane_idx
        )
        thr_y, atom_y = tiled_copy(mY, self.out_vec * 8, tv_out, tiler_out, lane_idx)
        idX = cute.make_identity_tensor(mX.shape)
        idY = cute.make_identity_tensor(mY.shape)
        part0 = (lane_idx % lpb) == 0
        sf_lane_off = [
            self.stage.local_offset(
                self.plan.digit_value("blk", coords, self._key_for(kk)), 0
            )
            for kk in range(bpt)
        ]
        for p in cutlass.range_constexpr(self.passes):
            r = warp_idx + W * p  # row within the 128-row tile
            row = bidy.to(Int64) * SF_ROW_TILE + r.to(Int64)
            gX = thr_x.partition_S(cute.local_tile(mX, tiler_in, (row, bidx)))
            pX = predicate(
                thr_x.partition_S(cute.local_tile(idX, tiler_in, (row, bidx))), M, K
            )
            frag = Span(gX, atom_x, pX, mX.element_type).load()
            state, start = canon.init_state(self.plan, fp, _Loader(self, frag))
            state, _ = canon.run_steps(
                self.plan, state, start, LanePeers(), fp, stop_before="blk"
            )
            fy_words = cute.make_rmem_tensor(
                cute.make_layout((self.words_per_block * bpt,)), Uint32
            )
            gY = thr_y.partition_S(cute.local_tile(mY, tiler_out, (row, bidx)))
            fy = cute.make_tensor(
                cute.recast_ptr(fy_words.iterator, dtype=Uint8),
                cute.make_fragment_like(gY).layout,
            )
            r_off = (r % 32) * 16 + (r // 32) * 4
            for kk in cutlass.range_constexpr(bpt):
                sf_byte = _quantize_block(
                    self, fp, frag, kk, state[self._key_for(kk)], fy_words
                )
                if part0:
                    sbuf[sf_lane_off[kk] + r_off] = sf_byte.to(Uint8)
            pY = predicate(
                thr_y.partition_S(cute.local_tile(idY, tiler_out, (row, bidx))),
                M,
                K * self.fmt.elem_bits // 8,
            )
            cute.copy(atom_y, fy, gY, pred=pY)
        cute.arch.barrier()
        self.stage.copy_out(sbuf, mS, tidx, bidy, bidx, pcols)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
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
        W, Lr, rows, bpt, lpb = self.W, self.Lr, self.rows, self.bpt, self.lpb
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

        # ---- loads: the thread's whole span, zero-filled beyond (M, K) ------------------
        thr_x, atom_x = tiled_copy(
            mX, self.VEC * mX.element_type.width, tv_in, tiler_in, vtid
        )
        thr_y, atom_y = tiled_copy(mY, self.out_vec * 8, tv_out, tiler_out, vtid)
        idX = cute.make_identity_tensor(mX.shape)
        gX = thr_x.partition_S(cute.local_tile(mX, tiler_in, (bidx, w)))
        pX = predicate(
            thr_x.partition_S(cute.local_tile(idX, tiler_in, (bidx, w))), M, K
        )
        frag = Span(gX, atom_x, pX, mX.element_type).load()

        # ===== the block amaxes: driven by the plan ======================================
        state, start = canon.init_state(self.plan, fp, _Loader(self, frag))
        state, _ = canon.run_steps(
            self.plan, state, start, LanePeers(), fp, stop_before="blk"
        )

        # ===== outputs: per block the scale chain, the element chain, the stores ===========
        idY = cute.make_identity_tensor(mY.shape)
        gY = thr_y.partition_S(cute.local_tile(mY, tiler_out, (bidx, w)))
        pY = predicate(
            thr_y.partition_S(cute.local_tile(idY, tiler_out, (bidx, w))),
            M,
            K * self.fmt.elem_bits // 8,
        )
        fy_words = cute.make_rmem_tensor(
            cute.make_layout((self.words_per_block * bpt,)), Uint32
        )
        fy = cute.make_tensor(
            cute.recast_ptr(fy_words.iterator, dtype=Uint8),
            cute.make_fragment_like(gY).layout,
        )
        part0 = (t % lpb) == 0
        for kk in cutlass.range_constexpr(bpt):
            sf_byte = _quantize_block(
                self, fp, frag, kk, state[self._key_for(kk)], fy_words
            )
            blk = self.plan.digit_value("blk", coords, self._key_for(kk))
            if cutlass.const_expr(self.layout == "128x4"):
                if part0 & (blk < pcols) & (row < padded_M):
                    mS[sf_offset_128x4_dev(row, blk, pcols)] = sf_byte.to(Uint8)
            elif cutlass.const_expr(self.layout == "8x4"):
                if part0 & (blk < pcols) & (row < padded_M):
                    mS[sf_offset_8x4_dev(row, blk, pcols)] = sf_byte.to(Uint8)
            else:
                if part0 & (blk < nb) & (row < M):
                    mS[sf_offset_linear_dev(row, blk, nb)] = sf_byte.to(Uint8)
        cute.copy(atom_y, fy, gY, pred=pY)
        if cutlass.const_expr(self.swizzled and self.Kbp < SF_COL_TILE):
            # the swizzled layouts pad scale columns to 4; a bucket of 1 or 2 blocks owns none of columns Kbp..3
            if (t == 0) & (w == 0) & (row < padded_M):
                for c in cutlass.range_constexpr(self.Kbp, SF_COL_TILE):
                    if cutlass.const_expr(self.layout == "128x4"):
                        mS[sf_offset_128x4_dev(row, Int32(c), pcols)] = Uint8(0)
                    else:
                        mS[sf_offset_8x4_dev(row, Int32(c), pcols)] = Uint8(0)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


def _quantize_block(k, fp, frag, kk: int, amax, fy_words):
    """The per-block scale chain (``mx_spec.scale_norm`` -> ``cvt_f32_to_ue8m0`` -> ``mx_spec.inv_scale``) and the
    element chain (``quant_pre`` -> ``saturate`` -> the declared conversion) of block ``kk`` of the span; writes the
    packed words, returns the scale byte."""
    sf_byte, sf_d = cvt_f32_to_ue8m0(scale_norm(fp, amax, k.fmt, k.rule), k.rule.round)
    inv = inv_scale(fp, sf_d, k.rule)
    ys = [
        saturate(
            fp,
            quant_pre(fp, frag[k._frag_index(kk, v)].to(Float32), inv, k.rule),
            k.fmt,
            k.rule,
        )
        for v in range(k.V)
    ]
    for w, word in enumerate(pack_block_codes(fp, ys, k.fmt.elem_fmt)):
        fy_words[kk * k.words_per_block + w] = word
    return sf_byte


# ----------------------------------------------------------------------------- compile
def get_compiled_kernel(
    dtype_str: str,
    K: int,
    cfg: MXQuantConfig,
    fmt: MXFormat,
    rule: ScaleRule,
    swizzled,
    enable_pdl: bool,
):
    """The compiled kernel serving row length ``K``: one kernel per (dtype, Kbp bucket, config, format, rule, layout,
    pdl); K, M and the row strides are runtime arguments (tile-mode kernels serve every K). ``swizzled`` names the layout.
    ``kernel(x, codes, sf, M, K, padded_M)``."""
    return _compile_bucket(
        dtype_str, cfg.bucket(K), cfg, fmt, rule, layout_name(swizzled), enable_pdl
    )


@functools.cache
def _compile_bucket(
    dtype_str: str,
    Kbp: int,
    cfg: MXQuantConfig,
    fmt: MXFormat,
    rule: ScaleRule,
    swizzled: str,
    enable_pdl: bool,
):
    from ..fp_interface.cute import get_cutlass_dtype

    dtype = get_cutlass_dtype(dtype_str)
    elem_bytes = dtype.width // 8
    name = mx_quant_kernel_name(
        fmt.name, rule.name, dtype_str, Kbp, cfg.describe(), swizzled, enable_pdl
    )

    def compile_fn():
        sym_m, sym_k, sym_k2, sym_n = (
            cute.sym_int(64),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(64),
        )
        x = cute.runtime.make_fake_tensor(
            dtype,
            (sym_m, sym_k),
            (cute.sym_int64(divisibility=VEC_BYTES // elem_bytes), 1),
            assumed_align=16,
        )
        y = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_m, sym_k2),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        s = cute.runtime.make_fake_tensor(Uint8, (sym_n,), (1,), assumed_align=16)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        obj = MXQuantKernel(dtype, Kbp, cfg, fmt, rule, swizzled)
        return cute.compile(
            obj,
            x,
            y,
            s,
            Int64(1),
            Int32(BLOCK),
            Int64(128),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MX_QUANT_MODULE, name, compile_fn)
