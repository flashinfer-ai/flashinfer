"""VC-Attention-QK16 operand preparation kernels (CuTe DSL).

Two small kernels turn bf16/fp16 ``[B, S, H, D]`` K/V into the prims_ts
VC-Attention-QK16 operands with each tensor read once (V twice) and written once:

* :class:`VcKvPass1`: per (batch*head, 128-token K/V tile) gathers the permuted
  tokens, writes the permuted K rows, and writes the V tile mean and the
  residual amax;
* :class:`VcKvPass2`: after the per-channel V residual scale is known, writes
  the E4M3 V residuals and the packed bf16 tile-mean UMMA operand.

Following the paper, V residuals get one E4M3 scale per (batch, head,
channel) and the block means are stored divided by that scale. Q and K stay
in their input dtype.

Every kernel runs 128 threads. Thread ``t`` owns head_dim columns
``(t % 8) * 16 .. + 16`` (one 32-byte vector) of rows ``t // 8 + 16 * i``
for ``i < 8``.
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ..vc_attention import (
    E4M3_MAX,
    VC_K_BLOCK_SIZE,
    VC_MEAN_MMA_K,
    VC_MEAN_GROUP_TILES,
    VC_MEAN_OPERANDS,
    vc_mean_group_shape,
    vc_mean_operand_shape,
)
from .fmha_decode.fmha_decode_resources.helpers_common import _pack_float2_to_bf16
from .fmha_decode.fmha_decode_resources.helpers_softmax import _pack_float4_to_fp8_e4m3

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
# The kernels are specialized for head_dim 128.
_D = 128
_THREADS = 128
_TILE = VC_K_BLOCK_SIZE
_COLS_PER_THREAD = 16
_COL_GROUPS = _D // _COLS_PER_THREAD
_ROW_GROUPS = _THREADS // _COL_GROUPS
_ROWS_PER_THREAD = _TILE // _ROW_GROUPS
_MU_ELEMS = math.prod(vc_mean_operand_shape(_D))
_HADAMARD_NORM = 1.0 / math.sqrt(_D)


@cute.jit
def _abs_f32(x: Float32) -> Float32:
    return cute.arch.fmax(x, -x)


@cute.jit
def _hadamard128_row(vals, base: cutlass.Constexpr[int], cg: Int32):
    """In-place normalised 128-point Walsh-Hadamard transform of one row whose
    16 columns ``vals[base : base + 16]`` live in this thread and whose other
    112 columns live in the 7 neighbouring lanes ``cg ^ {1, 2, 4}``."""
    for h in (1, 2, 4, 8):
        for i in cutlass.range_constexpr(_COLS_PER_THREAD):
            if (i & h) == 0:
                a = vals[base + i]
                b = vals[base + (i ^ h)]
                vals[base + i] = a + b
                vals[base + (i ^ h)] = a - b
    for sbit in cutlass.range_constexpr(3):
        stride = 1 << sbit
        # Lanes with the bit clear take (mine + other); lanes with it set take (other - mine).
        sign = Float32(1.0) - Float32(2.0) * Float32((cg >> sbit) & 1)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            mine = vals[base + j]
            other = cute.arch.shuffle_sync_bfly(mine, stride)
            vals[base + j] = sign * mine + other
    for j in cutlass.range_constexpr(_COLS_PER_THREAD):
        vals[base + j] = vals[base + j] * Float32(_HADAMARD_NORM)


@cute.jit
def _block_max(v: Float32, red_smem, tidx: Int32) -> Float32:
    """Max over the 128 threads of the CTA (4 warps)."""
    for shift in cutlass.range_constexpr(5):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, 1 << shift))
    if tidx % 32 == 0:
        red_smem[tidx // 32] = v
    cute.arch.sync_threads()
    r = red_smem[0]
    for w in cutlass.range_constexpr(1, 4):
        r = cute.arch.fmax(r, red_smem[w])
    cute.arch.sync_threads()
    return r


@cute.jit
def _flat_scale_offset(
    bh: Int32, num_heads: Int32, num_batch_heads: Int32, seq_len: Int32, block: Int32
) -> Int64:
    """Element offset of block ``block`` of sequence ``bh // num_heads`` in the flat
    ``[H, ceil(B*S / 128) + B - 1]`` scale layout (``sage.flat_scale_slot``)."""
    b_idx = bh // num_heads
    h_idx = bh % num_heads
    n_batch = num_batch_heads // num_heads
    numel = ((n_batch * seq_len + _TILE - 1) // _TILE) + n_batch - 1
    slot = ((b_idx * seq_len) // _TILE) + b_idx + block
    return Int64(h_idx) * Int64(numel) + Int64(slot)


@cute.jit
def _load_row16(base_addr: Int64, is_bf16: cutlass.Constexpr[bool]):
    """16 bf16/fp16 elements at ``base_addr`` as a Vector of 16 Float32."""
    regs = cutlass.inttoptr(base_addr, mem_space=1, dtype=Int32).load(
        count=8, alignment=32
    )
    if cutlass.const_expr(is_bf16):
        return regs.bitcast(cutlass.BFloat16).to(Float32)
    return regs.bitcast(cutlass.Float16).to(Float32)


@cute.jit
def _store_row16_fp8(base_addr: Int64, vals, scale_inv: Float32):
    """Quantize 16 Float32 values (Array or Vector) by ``scale_inv`` and store 16 E4M3 bytes."""
    packed = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(4):
        packed[j] = _pack_float4_to_fp8_e4m3(
            vals[4 * j] * scale_inv,
            vals[4 * j + 1] * scale_inv,
            vals[4 * j + 2] * scale_inv,
            vals[4 * j + 3] * scale_inv,
        )
    cutlass.inttoptr(base_addr, mem_space=1, dtype=Int32).store(
        packed.data_ptr().load(count=4, alignment=16), alignment=16
    )


@cute.jit
def _load_row16_f32(base_addr: Int64):
    return cutlass.inttoptr(base_addr, mem_space=1, dtype=Float32).load(
        count=16, alignment=64
    )


class VcKvPass1:
    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mKPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVAmax: cute.Tensor,
        mKMean: cute.Tensor,
        mK8: cute.Tensor,
        mKScale: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        num_batch_heads: Int32,
        demean: cutlass.Constexpr[bool],
        qk_fp8: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // num_heads
        h = bh % num_heads
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS_PER_THREAD

        smem = cutlass.utils.SmemAllocator()
        col_sums = smem.allocate_array(Float32, _ROW_GROUPS * _TILE)
        red = smem.allocate_array(Float32, 4)

        k_base = mK.iterator.toint()
        v_base = mV.iterator.toint()
        kp_base = mKPerm.iterator.toint()
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(seq_len) * 4
        row_bytes = Int64(num_heads) * (_D * 2)

        toks = cutlass.Array(Int32, _ROWS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            tok = Int32(0)
            if pos < seq_len:
                tok = cutlass.inttoptr(
                    perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                ).load()
            toks[i] = tok

        if cutlass.const_expr(qk_fp8):
            # K, VC-Attention-QK8: centre by the per-head channel mean, rotate the
            # permuted rows by the Hadamard matrix, quantize to E4M3 with one
            # scale per tile in the flat scale layout.
            kmean = _load_row16_f32(
                mKMean.iterator.toint() + (Int64(bh) * _D + col0) * 4
            )
            kvals = cutlass.Array(
                Float32,
                _ROWS_PER_THREAD * _COLS_PER_THREAD,
                space=cutlass.AddressSpace.rmem,
            )
            kmax = Float32(0.0)
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if pos < seq_len:
                    addr = (
                        k_base
                        + (Int64(b) * Int64(seq_len) + Int64(toks[i])) * row_bytes
                        + (Int64(h) * _D + col0) * 2
                    )
                    row = _load_row16(addr, self.is_bf16)
                    for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                        kvals[i * _COLS_PER_THREAD + j] = row[j] - kmean[j]
                else:
                    for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                        kvals[i * _COLS_PER_THREAD + j] = Float32(0.0)
            # The rotation needs all 8 lanes of a row, also for padded rows.
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                _hadamard128_row(kvals, i * _COLS_PER_THREAD, cg)
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if pos < seq_len:
                    for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                        kmax = cute.arch.fmax(
                            kmax, _abs_f32(kvals[i * _COLS_PER_THREAD + j])
                        )
            kmax = _block_max(kmax, red, tidx)
            kscale = cute.arch.fmax(kmax / E4M3_MAX, Float32(1e-12))
            kscale_inv = Float32(1.0) / kscale
            if tidx == 0:
                cutlass.inttoptr(
                    mKScale.iterator.toint()
                    + _flat_scale_offset(bh, num_heads, num_batch_heads, seq_len, t)
                    * 4,
                    mem_space=1,
                    dtype=Float32,
                ).store(kscale)
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if pos < seq_len:
                    addr = (
                        mK8.iterator.toint()
                        + (Int64(b) * Int64(seq_len) + Int64(pos))
                        * Int64(num_heads)
                        * _D
                        + Int64(h) * _D
                        + col0
                    )
                    sub = cutlass.Array(
                        Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                    )
                    for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                        sub[j] = kvals[i * _COLS_PER_THREAD + j]
                    _store_row16_fp8(addr, sub, kscale_inv)
        else:
            # K, copy the permuted rows.
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if pos < seq_len:
                    src = (
                        k_base
                        + (Int64(b) * Int64(seq_len) + Int64(toks[i])) * row_bytes
                        + (Int64(h) * _D + col0) * 2
                    )
                    dst = (
                        kp_base
                        + (Int64(b) * Int64(seq_len) + Int64(pos)) * row_bytes
                        + (Int64(h) * _D + col0) * 2
                    )
                    regs = cutlass.inttoptr(src, mem_space=1, dtype=Int32).load(
                        count=8, alignment=32
                    )
                    cutlass.inttoptr(dst, mem_space=1, dtype=Int32).store(
                        regs, alignment=32
                    )

        # V, tile mean and residual amax.
        vvals = cutlass.Array(
            Float32,
            _ROWS_PER_THREAD * _COLS_PER_THREAD,
            space=cutlass.AddressSpace.rmem,
        )
        psum = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            psum[j] = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                addr = (
                    v_base
                    + (Int64(b) * Int64(seq_len) + Int64(toks[i])) * row_bytes
                    + (Int64(h) * _D + col0) * 2
                )
                row = _load_row16(addr, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    vvals[i * _COLS_PER_THREAD + j] = row[j]
                    psum[j] = psum[j] + row[j]
            else:
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    vvals[i * _COLS_PER_THREAD + j] = Float32(0.0)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            col_sums[rg * _TILE + col0 + j] = psum[j]
        cute.arch.sync_threads()
        valid_rows = seq_len - t * _TILE
        if valid_rows > _TILE:
            valid_rows = Int32(_TILE)
        inv_count = Float32(1.0) / Float32(valid_rows)
        mean = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            acc = Float32(0.0)
            for g in cutlass.range_constexpr(_ROW_GROUPS):
                acc = acc + col_sums[g * _TILE + col0 + j]
            mean[j] = acc * inv_count
            if cutlass.const_expr(not demean):
                mean[j] = Float32(0.0)
        if rg == 0:
            mean_addr = (
                mMean.iterator.toint()
                + ((Int64(bh) * Int64(num_tiles) + t) * _D + col0) * 4
            )
            cutlass.inttoptr(mean_addr, mem_space=1, dtype=Float32).store(
                mean.data_ptr().load(count=16, alignment=64), alignment=64
            )
        # Per-channel residual amax of this tile (reduced over tiles on the host).
        cmax = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            cmax[j] = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    cmax[j] = cute.arch.fmax(
                        cmax[j], _abs_f32(vvals[i * _COLS_PER_THREAD + j] - mean[j])
                    )
        cute.arch.sync_threads()  # everyone is done reading col_sums as sums
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            col_sums[rg * _TILE + col0 + j] = cmax[j]
        cute.arch.sync_threads()
        if rg == 0:
            for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                acc = Float32(0.0)
                for g in cutlass.range_constexpr(_ROW_GROUPS):
                    acc = cute.arch.fmax(acc, col_sums[g * _TILE + col0 + j])
                cmax[j] = acc
            amax_addr = (
                mVAmax.iterator.toint()
                + ((Int64(bh) * Int64(num_tiles) + t) * _D + col0) * 4
            )
            cutlass.inttoptr(amax_addr, mem_space=1, dtype=Float32).store(
                cmax.data_ptr().load(count=16, alignment=64), alignment=64
            )

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mKPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVAmax: cute.Tensor,
        mKMean: cute.Tensor,
        mK8: cute.Tensor,
        mKScale: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        num_batch_heads: Int32,
        demean: cutlass.Constexpr[bool],
        qk_fp8: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(
            mK,
            mV,
            mPerm,
            mKPerm,
            mMean,
            mVAmax,
            mKMean,
            mK8,
            mKScale,
            seq_len,
            num_heads,
            num_tiles,
            num_batch_heads,
            demean,
            qk_fp8,
        ).launch(
            grid=[num_tiles, num_batch_heads, 1],
            block=[_THREADS, 1, 1],
            smem=(_ROW_GROUPS * _TILE + 4) * 4,
            stream=stream,
        )


class VcKvPass2:
    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // num_heads
        h = bh % num_heads
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS_PER_THREAD
        v_base = mV.iterator.toint()
        v8_base = mV8.iterator.toint()
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(seq_len) * 4
        row_bytes = Int64(num_heads) * (_D * 2)
        v8_row_bytes = Int64(num_heads) * _D
        mean_base = mMean.iterator.toint() + (Int64(bh) * Int64(num_tiles) + t) * (
            _D * 4
        )

        vscale_base = mVScale.iterator.toint() + Int64(bh) * (_D * 4)  # [B, H, D] fp32
        vs = _load_row16_f32(vscale_base + col0 * 4)
        vs_inv = cutlass.Array(
            Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            vs_inv[j] = Float32(1.0) / vs[j]
        mean = _load_row16_f32(mean_base + col0 * 4)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                tok = cutlass.inttoptr(
                    perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                ).load()
                addr = (
                    v_base
                    + (Int64(b) * Int64(seq_len) + Int64(tok)) * row_bytes
                    + (Int64(h) * _D + col0) * 2
                )
                row = _load_row16(addr, self.is_bf16)
                res = cutlass.Array(
                    Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    res[j] = (row[j] - mean[j]) * vs_inv[j]
                out_addr = (
                    v8_base
                    + (Int64(b) * Int64(seq_len) + Int64(pos)) * v8_row_bytes
                    + Int64(h) * _D
                    + col0
                )
                _store_row16_fp8(out_addr, res, Float32(1.0))

        # Mean operands of this tile's group in core-matrix order, g = 8-row
        # group, c = K core matrix, r = row, k = K index. Tile 8o+i of the group
        # owns K slots 2i and 2i+1 of every row of operand o; the operands are
        # zeroed by the host, so each thread stores only its two rows' slots.
        k_per_core = VC_MEAN_MMA_K // 2
        idx0 = tidx * (2 * k_per_core)
        g = idx0 // (8 * VC_MEAN_MMA_K)
        c = (idx0 // (8 * k_per_core)) % 2
        r0 = (idx0 // k_per_core) % 8
        d0 = g * 8 + r0
        tile_in_group = t % VC_MEAN_GROUP_TILES
        operand = tile_in_group // k_per_core
        slot = 2 * (tile_in_group % k_per_core)
        c_slot = slot // k_per_core
        m0 = cutlass.inttoptr(
            mean_base + Int64(d0) * 4, mem_space=1, dtype=Float32
        ).load()
        m1 = cutlass.inttoptr(
            mean_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
        ).load()
        s0 = cutlass.inttoptr(
            vscale_base + Int64(d0) * 4, mem_space=1, dtype=Float32
        ).load()
        s1 = cutlass.inttoptr(
            vscale_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
        ).load()
        if c == c_slot:
            # Means are stored divided by the per-channel value scale (paper, Appendix B).
            num_groups = (num_tiles + VC_MEAN_GROUP_TILES - 1) // VC_MEAN_GROUP_TILES
            dst = (
                mMu.iterator.toint()
                + (
                    (
                        (Int64(bh) * Int64(num_groups) + t // VC_MEAN_GROUP_TILES)
                        * VC_MEAN_OPERANDS
                        + operand
                    )
                    * _MU_ELEMS
                    + idx0
                    + slot % k_per_core
                )
                * 2
            )
            cutlass.inttoptr(dst, mem_space=1, dtype=Int32).store(
                _pack_float2_to_bf16(m0 / s0, m0 / s0)
            )
            cutlass.inttoptr(dst + k_per_core * 2, mem_space=1, dtype=Int32).store(
                _pack_float2_to_bf16(m1 / s1, m1 / s1)
            )

    @cute.jit
    def __call__(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        num_batch_heads: Int32,
        stream,
    ):
        self.kernel(
            mV, mPerm, mMean, mVScale, mV8, mMu, seq_len, num_heads, num_tiles
        ).launch(
            grid=[num_tiles, num_batch_heads, 1], block=[_THREADS, 1, 1], stream=stream
        )


class VcQPass:
    """VC-Attention-QK8 Q pass: rotate each 128-token block by the Hadamard matrix
    and quantize it to E4M3 with one scale per block in the flat scale layout."""

    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQ8: cute.Tensor,
        mQScale: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_batch_heads: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        nb, bh, _ = cute.arch.block_idx()
        b = bh // num_heads
        h = bh % num_heads
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS_PER_THREAD
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_array(Float32, 4)
        q_base = mQ.iterator.toint()
        row_bytes = Int64(num_heads) * (_D * 2)
        qvals = cutlass.Array(
            Float32,
            _ROWS_PER_THREAD * _COLS_PER_THREAD,
            space=cutlass.AddressSpace.rmem,
        )
        qmax = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                addr = (
                    q_base
                    + (Int64(b) * Int64(seq_len) + Int64(pos)) * row_bytes
                    + (Int64(h) * _D + col0) * 2
                )
                row = _load_row16(addr, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    qvals[i * _COLS_PER_THREAD + j] = row[j]
            else:
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    qvals[i * _COLS_PER_THREAD + j] = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            _hadamard128_row(qvals, i * _COLS_PER_THREAD, cg)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    qmax = cute.arch.fmax(
                        qmax, _abs_f32(qvals[i * _COLS_PER_THREAD + j])
                    )
        qmax = _block_max(qmax, red, tidx)
        qscale = cute.arch.fmax(qmax / E4M3_MAX, Float32(1e-12))
        qscale_inv = Float32(1.0) / qscale
        if tidx == 0:
            cutlass.inttoptr(
                mQScale.iterator.toint()
                + _flat_scale_offset(bh, num_heads, num_batch_heads, seq_len, nb) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(qscale)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                addr = (
                    mQ8.iterator.toint()
                    + (Int64(b) * Int64(seq_len) + Int64(pos)) * Int64(num_heads) * _D
                    + Int64(h) * _D
                    + col0
                )
                sub = cutlass.Array(
                    Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    sub[j] = qvals[i * _COLS_PER_THREAD + j]
                _store_row16_fp8(addr, sub, qscale_inv)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mQ8: cute.Tensor,
        mQScale: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_blocks: Int32,
        num_batch_heads: Int32,
        stream,
    ):
        self.kernel(mQ, mQ8, mQScale, seq_len, num_heads, num_batch_heads).launch(
            grid=[num_blocks, num_batch_heads, 1],
            block=[_THREADS, 1, 1],
            smem=16,
            stream=stream,
        )


# ---------------------------------------------------------------------------
# Compilation cache (TVM-FFI entry points taking torch tensors directly)
# ---------------------------------------------------------------------------
def _fake1d(dtype, align=32):
    return cute.runtime.make_fake_compact_tensor(
        dtype, (cute.sym_int(),), assumed_align=align
    )


def _in_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16, True
    if dtype == torch.float16:
        return cutlass.Float16, False
    raise TypeError(f"VC fused preparation needs bf16/fp16 inputs, got {dtype}")


@functools.lru_cache(maxsize=None)
def _compiled(dtype: torch.dtype, demean: bool, qk_fp8: bool = False):
    cdt, is_bf16 = _in_dtype(dtype)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    f32 = lambda: _fake1d(Float32, 64)  # noqa: E731
    i32 = lambda: _fake1d(Int32, 16)  # noqa: E731
    e4m3 = lambda: _fake1d(cutlass.Float8E4M3FN, 16)  # noqa: E731
    p1 = cute.compile(
        VcKvPass1(is_bf16),
        _fake1d(cdt),
        _fake1d(cdt),
        i32(),
        _fake1d(cdt),
        f32(),
        f32(),
        f32(),
        e4m3(),
        f32(),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        demean,
        qk_fp8,
        stream,
        options=_COMPILE_OPTIONS,
    )
    pq = None
    if qk_fp8:
        pq = cute.compile(
            VcQPass(is_bf16),
            _fake1d(cdt),
            e4m3(),
            f32(),
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(1),
            stream,
            options=_COMPILE_OPTIONS,
        )
    p2 = cute.compile(
        VcKvPass2(is_bf16),
        _fake1d(cdt),
        i32(),
        f32(),
        f32(),
        e4m3(),
        _fake1d(cutlass.BFloat16),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        stream,
        options=_COMPILE_OPTIONS,
    )
    return p1, p2, pq


@torch.no_grad()
def vc_prepare(
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    *,
    demean: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Run the two preparation kernels.

    ``k``, ``v``: contiguous ``[B, S, H, D]`` bf16/fp16; ``perm``:
    ``[B, H, S_k]`` int32/int64. ``demean=False`` keeps the tile means at zero
    (V-Smooth off, plain per-channel E4M3 V). Returns ``(k_perm, v8, v_scale,
    mean, mu)`` in the ``VCAttentionOperands`` layouts; ``v_scale`` is
    ``[B, H, D]``.
    """
    return _prepare(None, k, v, perm, demean=demean)[:5]


@torch.no_grad()
def vc_prepare_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    *,
    demean: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Run the three VC-Attention-QK8 preparation kernels.

    ``q``, ``k``, ``v``: contiguous ``[B, S, H, D]`` bf16/fp16 (one Q block scale
    per 128 tokens); ``perm``: ``[B, H, S_k]`` int32/int64. Returns ``(q8, k8,
    v8, q_scale, k_scale, v_scale, mean, mu)`` in the ``VCAttentionOperands``
    layouts, the Q/K scales in the flat scale layout.
    """
    if q.shape[-1] != _D or not q.is_contiguous():
        raise ValueError(f"VC fused preparation needs contiguous head_dim {_D} Q")
    k8, v8, v_scale, mean, mu, q8, q_scale, k_scale = _prepare(
        q, k, v, perm, demean=demean
    )
    return q8, k8, v8, q_scale, k_scale, v_scale, mean, mu


def _prepare(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    *,
    demean: bool,
) -> tuple[torch.Tensor, ...]:
    """Run the passes; ``q`` selects the QK8 recipe. Returns ``(k_out, v8,
    v_scale, mean, mu, q8, q_scale, k_scale)`` with the QK8 outputs ``None``
    under QK16."""
    b, s_k, h, d = k.shape
    if d != _D:
        raise ValueError(f"VC fused preparation supports head_dim {_D} only")
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("k and v must be contiguous")
    dev = k.device
    t = (s_k + _TILE - 1) // _TILE
    bh = b * h
    qk_fp8 = q is not None
    p1, p2, pq = _compiled(k.dtype, bool(demean), qk_fp8)
    perm32 = perm.to(torch.int32).contiguous()
    dummy = torch.zeros((1,), dtype=torch.float32, device=dev)
    if qk_fp8:
        k_perm = torch.empty((1,), dtype=k.dtype, device=dev)
        k_mean = k.float().mean(dim=1).contiguous()  # [B, H, D]
        k8 = torch.empty((b, s_k, h, d), dtype=torch.float8_e4m3fn, device=dev)
        # Flat scale layout (sage.flat_scale_numel), [H, ceil(B*S / 128) + B - 1].
        k_scale = torch.ones(
            (h, (b * s_k + _TILE - 1) // _TILE + b - 1), dtype=torch.float32, device=dev
        )
    else:
        k_perm = torch.empty_like(k)
        k_mean = dummy
        k8 = torch.empty((1,), dtype=torch.float8_e4m3fn, device=dev)
        k_scale = dummy
    v8 = torch.empty((b, s_k, h, d), dtype=torch.float8_e4m3fn, device=dev)
    mean = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    vamax = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    mu = torch.zeros(
        (
            b,
            h,
            (t + VC_MEAN_GROUP_TILES - 1) // VC_MEAN_GROUP_TILES,
            *vc_mean_group_shape(d),
        ),
        dtype=torch.bfloat16,
        device=dev,
    )
    p1(
        k.view(-1),
        v.view(-1),
        perm32.view(-1),
        k_perm.view(-1),
        mean.view(-1),
        vamax.view(-1),
        k_mean.view(-1),
        k8.view(-1),
        k_scale.view(-1),
        s_k,
        h,
        t,
        bh,
    )
    v_scale = (vamax.amax(dim=2) / E4M3_MAX).clamp_min(1e-12).contiguous()  # [B, H, D]
    p2(
        v.view(-1),
        perm32.view(-1),
        mean.view(-1),
        v_scale.view(-1),
        v8.view(-1),
        mu.view(-1),
        s_k,
        h,
        t,
        bh,
    )
    if not qk_fp8:
        return k_perm, v8, v_scale, mean, mu, None, None, None
    s_q = q.shape[1]
    q8 = torch.empty((b, s_q, h, d), dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.ones(
        (h, (b * s_q + _TILE - 1) // _TILE + b - 1), dtype=torch.float32, device=dev
    )
    pq(
        q.view(-1),
        q8.view(-1),
        q_scale.view(-1),
        s_q,
        h,
        (s_q + _TILE - 1) // _TILE,
        bh,
    )
    return k8, v8, v_scale, mean, mu, q8, q_scale, k_scale
