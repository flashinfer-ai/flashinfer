"""VC-Attention-QK16 operand preparation kernels (CuTe DSL).

Two small kernels turn bf16/fp16 ``[B, S, H, D]`` K/V into the prims_ts
VC-Attention-QK16 operands with each tensor read once (V twice) and written once:

* :class:`VcKvPass1`: per (batch*head, 128-token K/V tile) gathers the permuted
  tokens, writes the permuted K rows, and writes the V tile mean and the
  residual amax;
* :class:`VcKvPass2`: after the per-channel V residual scale is known, writes
  the E4M3 V residuals and the packed bf16 tile-mean UMMA operand.

With ``repair`` the token order is the input order, the partial last tile
moves ``tail_shift`` rows down to leave room for the V repair tiles, and pass
2 writes each token's residual energy instead of the mean operand.

Following the paper, V residuals get one E4M3 scale per (batch, head,
channel) and the block means are stored divided by that scale. Q and K stay
in their input dtype.

Every kernel runs 256 threads. Thread ``t`` owns head_dim columns
``(t % 16) * 8 .. + 8`` (one 16-byte vector) of rows ``t // 16 + 16 * i``
for ``i < 8``. Rows stream through registers, never a tile-sized slab, and the
per-column state is 8 wide: the pre-pass is HBM-latency-bound and needs the
occupancy (a 128-value fp32 slab per thread capped it at 2 CTAs/SM and 17% of
DRAM bandwidth; 16-wide column state at 96 registers still at 30%).
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
from ..vc_attention import vc_repair_kv_len

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
# The kernels are specialized for head_dim 128.
_D = 128
_THREADS = 256
_TILE = VC_K_BLOCK_SIZE
_COLS_PER_THREAD = 8
_COL_GROUPS = _D // _COLS_PER_THREAD
_ROW_GROUPS = _THREADS // _COL_GROUPS
_ROWS_PER_THREAD = _TILE // _ROW_GROUPS
_MU_ELEMS = math.prod(vc_mean_operand_shape(_D))


@cute.jit
def _coords(num_heads: Int32):
    """This CTA's tile, batch and head, this thread's row group and first column."""
    tidx, _, _ = cute.arch.thread_idx()
    t, bh, _ = cute.arch.block_idx()
    return (
        t,
        bh,
        bh // num_heads,
        bh % num_heads,
        tidx // _COL_GROUPS,
        (tidx % _COL_GROUPS) * _COLS_PER_THREAD,
    )


@cute.jit
def _row_addr(
    base: Int64,
    b: Int32,
    rows: Int32,
    row: Int32,
    h: Int32,
    col0: Int32,
    row_bytes: Int64,
    elem_bytes: cutlass.Constexpr[int],
) -> Int64:
    """Address of this thread's columns of row ``row`` of batch ``b`` in a ``[B, rows, H, D]`` tensor."""
    return (
        base
        + (Int64(b) * Int64(rows) + Int64(row)) * row_bytes
        + (Int64(h) * _D + col0) * elem_bytes
    )


@cute.jit
def _load_row(base_addr: Int64, is_bf16: cutlass.Constexpr[bool]):
    """This thread's 8 bf16 or fp16 elements at ``base_addr`` as a Vector of Float32."""
    regs = cutlass.inttoptr(base_addr, mem_space=1, dtype=Int32).load(
        count=_COLS_PER_THREAD // 2, alignment=16
    )
    if cutlass.const_expr(is_bf16):
        return regs.bitcast(cutlass.BFloat16).to(Float32)
    return regs.bitcast(cutlass.Float16).to(Float32)


@cute.jit
def _load_row_f32(base_addr: Int64):
    """This thread's 8 Float32 elements at ``base_addr``."""
    return cutlass.inttoptr(base_addr, mem_space=1, dtype=Float32).load(
        count=_COLS_PER_THREAD, alignment=32
    )


@cute.jit
def _pack_row_fp8(vals):
    """This thread's 8 Float32 values as two words of E4M3 codes."""
    packed = cutlass.Array(
        Int32, _COLS_PER_THREAD // 4, space=cutlass.AddressSpace.rmem
    )
    for j in cutlass.range_constexpr(_COLS_PER_THREAD // 4):
        packed[j] = _pack_float4_to_fp8_e4m3(
            vals[4 * j], vals[4 * j + 1], vals[4 * j + 2], vals[4 * j + 3]
        )
    return packed.data_ptr().load(count=_COLS_PER_THREAD // 4, alignment=8)


@cute.jit
def _reduce_cols(
    col_sums,
    rg: Int32,
    col0: Int32,
    vals,
    use_max: cutlass.Constexpr[bool],
    leader_only: cutlass.Constexpr[bool],
):
    """Sum or max of ``vals`` over the row groups of the CTA, per column, into ``vals``.

    With ``leader_only`` just row group 0 reduces, for a result only it stores.
    """
    for j in cutlass.range_constexpr(_COLS_PER_THREAD):
        col_sums[rg * _TILE + col0 + j] = vals[j]
    cute.arch.sync_threads()
    if cutlass.const_expr(not leader_only) or rg == 0:
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            acc = Float32(0.0)
            for g in cutlass.range_constexpr(_ROW_GROUPS):
                if cutlass.const_expr(use_max):
                    acc = cute.arch.fmax(acc, col_sums[g * _TILE + col0 + j])
                else:
                    acc = acc + col_sums[g * _TILE + col0 + j]
            vals[j] = acc


@cute.jit
def _store_tile_cols(
    base: Int64, bh: Int32, num_tiles: Int32, t: Int32, col0: Int32, vals
):
    """Store this thread's 8 columns of tile ``t`` of a ``[B, H, num_tiles, D]`` fp32 tensor."""
    cutlass.inttoptr(
        base + ((Int64(bh) * Int64(num_tiles) + t) * _D + col0) * 4,
        mem_space=1,
        dtype=Float32,
    ).store(vals.data_ptr().load(count=_COLS_PER_THREAD, alignment=32), alignment=32)


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
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        demean: cutlass.Constexpr[bool],
        repair: cutlass.Constexpr[bool],
    ):
        t, bh, b, h, rg, col0 = _coords(num_heads)
        row_bytes = Int64(num_heads) * (_D * 2)
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(seq_len) * 4
        smem = cutlass.utils.SmemAllocator()
        col_sums = smem.allocate_array(Float32, _ROW_GROUPS * _TILE)

        toks = cutlass.Array(Int32, _ROWS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            toks[i] = Int32(0)
            if pos < seq_len:
                toks[i] = pos
                if cutlass.const_expr(not repair):
                    toks[i] = cutlass.inttoptr(
                        perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                    ).load()

        # K, copy the permuted rows.
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                src = _row_addr(
                    mK.iterator.toint(), b, seq_len, toks[i], h, col0, row_bytes, 2
                )
                dst = _row_addr(
                    mKPerm.iterator.toint(),
                    b,
                    out_len,
                    pos + tail_shift * Int32(pos >= tail_start),
                    h,
                    col0,
                    row_bytes,
                    2,
                )
                regs = cutlass.inttoptr(src, mem_space=1, dtype=Int32).load(
                    count=_COLS_PER_THREAD // 2, alignment=16
                )
                cutlass.inttoptr(dst, mem_space=1, dtype=Int32).store(
                    regs, alignment=16
                )

        # V, tile mean and residual amax. The rows stream through registers.
        # Per-column sum, max and min are enough, since
        # max_i |v_i - mean| == max(vmax - mean, mean - vmin) exactly.
        psum = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        pmax = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        pmin = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            psum[j] = Float32(0.0)
            pmax[j] = -Float32.inf
            pmin[j] = Float32.inf
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                row = _load_row(
                    _row_addr(
                        mV.iterator.toint(), b, seq_len, toks[i], h, col0, row_bytes, 2
                    ),
                    self.is_bf16,
                )
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    psum[j] = psum[j] + row[j]
                    pmax[j] = cute.arch.fmax(pmax[j], row[j])
                    pmin[j] = cute.arch.fmin(pmin[j], row[j])
        _reduce_cols(col_sums, rg, col0, psum, False, False)
        valid_rows = cute.math.min(seq_len - t * _TILE, Int32(_TILE))
        inv_count = Float32(1.0) / Float32(valid_rows)
        mean = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            mean[j] = psum[j] * inv_count
            if cutlass.const_expr(not demean):
                mean[j] = Float32(0.0)
        if rg == 0:
            _store_tile_cols(mMean.iterator.toint(), bh, num_tiles, t, col0, mean)
        # Per-channel residual amax of this tile, reduced over tiles on the host.
        # Threads without a valid row keep -inf and inf and contribute 0.
        cmax = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            cmax[j] = Float32(0.0)
            if pmax[j] >= pmin[j]:
                cmax[j] = cute.arch.fmax(pmax[j] - mean[j], mean[j] - pmin[j])
        cute.arch.sync_threads()  # everyone is done reading col_sums as sums
        _reduce_cols(col_sums, rg, col0, cmax, True, True)
        if rg == 0:
            _store_tile_cols(mVAmax.iterator.toint(), bh, num_tiles, t, col0, cmax)

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mKPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVAmax: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        num_batch_heads: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        demean: cutlass.Constexpr[bool],
        repair: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(
            mK,
            mV,
            mPerm,
            mKPerm,
            mMean,
            mVAmax,
            seq_len,
            num_heads,
            num_tiles,
            out_len,
            tail_start,
            tail_shift,
            demean,
            repair,
        ).launch(
            grid=[num_tiles, num_batch_heads, 1],
            block=[_THREADS, 1, 1],
            smem=_ROW_GROUPS * _TILE * 4,
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
        mEnergy: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        repair: cutlass.Constexpr[bool],
    ):
        t, bh, b, h, rg, col0 = _coords(num_heads)
        row_bytes = Int64(num_heads) * (_D * 2)
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(seq_len) * 4
        mean_base = mMean.iterator.toint() + (Int64(bh) * Int64(num_tiles) + t) * (
            _D * 4
        )
        vscale_base = mVScale.iterator.toint() + Int64(bh) * (_D * 4)

        smem = cutlass.utils.SmemAllocator()
        # V repair, per-thread partial residual energies of 8 channels for each of 8 rows.
        energy_part = smem.allocate_array(Float32, _THREADS * _ROWS_PER_THREAD)

        vs = _load_row_f32(vscale_base + col0 * 4)
        vs_inv = cutlass.Array(
            Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            vs_inv[j] = Float32(1.0) / vs[j]
        mean = _load_row_f32(mean_base + col0 * 4)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            energy = Float32(0.0)
            if pos < seq_len:
                tok = pos
                if cutlass.const_expr(not repair):
                    tok = cutlass.inttoptr(
                        perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                    ).load()
                row = _load_row(
                    _row_addr(
                        mV.iterator.toint(),
                        b,
                        seq_len,
                        tok,
                        h,
                        col0,
                        row_bytes,
                        2,
                    ),
                    self.is_bf16,
                )
                res = cutlass.Array(
                    Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    res[j] = (row[j] - mean[j]) * vs_inv[j]
                words = _pack_row_fp8(res)
                if cutlass.const_expr(repair):
                    # Residual energy of the E4M3 rounding in value units.
                    codes = words.bitcast(cutlass.Float8E4M3FN).to(Float32)
                    for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                        err = (res[j] - codes[j]) * vs[j]
                        energy = energy + err * err
                cutlass.inttoptr(
                    _row_addr(
                        mV8.iterator.toint(),
                        b,
                        out_len,
                        pos + tail_shift * Int32(pos >= tail_start),
                        h,
                        col0,
                        row_bytes // 2,
                        1,
                    ),
                    mem_space=1,
                    dtype=Int32,
                ).store(words, alignment=8)
            if cutlass.const_expr(repair):
                energy_part[
                    (rg + _ROW_GROUPS * i) * _COL_GROUPS + col0 // _COLS_PER_THREAD
                ] = energy
        if cutlass.const_expr(repair):
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if (col0 == 0) & (pos < seq_len):
                    acc = Float32(0.0)
                    for c in cutlass.range_constexpr(_COL_GROUPS):
                        acc = (
                            acc + energy_part[(rg + _ROW_GROUPS * i) * _COL_GROUPS + c]
                        )
                    cutlass.inttoptr(
                        mEnergy.iterator.toint()
                        + (Int64(bh) * Int64(seq_len) + Int64(pos)) * 4,
                        mem_space=1,
                        dtype=Float32,
                    ).store(acc)

        # Mean operands of this tile's group in core-matrix order, g = 8-row
        # group, c = K core matrix, r = row, k = K index. Tile 8o+i of the group
        # owns K slots 2i and 2i+1 of every row of operand o; the operands are
        # zeroed by the host, so each thread stores only its two rows' slots.
        # The operand mapping below was written for 128 threads, so the upper
        # half of the CTA has nothing to store.
        k_per_core = VC_MEAN_MMA_K // 2
        tidx = rg * _COL_GROUPS + col0 // _COLS_PER_THREAD
        idx0 = tidx * (2 * k_per_core)
        g = idx0 // (8 * VC_MEAN_MMA_K)
        c = (idx0 // (8 * k_per_core)) % 2
        d0 = g * 8 + (idx0 // k_per_core) % 8
        tile_in_group = t % VC_MEAN_GROUP_TILES
        operand = tile_in_group // k_per_core
        slot = 2 * (tile_in_group % k_per_core)
        if cutlass.const_expr(not repair) and (tidx < 128) & (c == slot // k_per_core):
            # Means are stored divided by the per-channel value scale, paper Appendix B.
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
            for r in cutlass.range_constexpr(2):
                m = cutlass.inttoptr(
                    mean_base + Int64(d0 + r) * 4, mem_space=1, dtype=Float32
                ).load()
                sc = cutlass.inttoptr(
                    vscale_base + Int64(d0 + r) * 4, mem_space=1, dtype=Float32
                ).load()
                cutlass.inttoptr(
                    dst + r * k_per_core * 2, mem_space=1, dtype=Int32
                ).store(_pack_float2_to_bf16(m / sc, m / sc))

    @cute.jit
    def __call__(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor,
        mEnergy: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        num_batch_heads: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        repair: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(
            mV,
            mPerm,
            mMean,
            mVScale,
            mV8,
            mMu,
            mEnergy,
            seq_len,
            num_heads,
            num_tiles,
            out_len,
            tail_start,
            tail_shift,
            repair,
        ).launch(
            grid=[num_tiles, num_batch_heads, 1],
            block=[_THREADS, 1, 1],
            smem=_THREADS * _ROWS_PER_THREAD * 4,
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
def _compiled(dtype: torch.dtype, demean: bool, repair: bool = False):
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
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        demean,
        repair,
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
        f32(),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        repair,
        stream,
        options=_COMPILE_OPTIONS,
    )
    return p1, p2


@torch.no_grad()
def vc_prepare(
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor | None,
    *,
    demean: bool = True,
    repair_tiles: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Run the two preparation kernels.

    ``k``, ``v``: contiguous ``[B, S, H, D]`` bf16/fp16; ``perm``:
    ``[B, H, S_k]`` int32/int64. ``demean=False`` keeps the tile means at zero
    (V-Smooth off, plain per-channel E4M3 V). Returns ``(k_perm, v8, v_scale,
    mean, mu, energy)`` in the ``VCAttentionOperands`` layouts; ``v_scale`` is
    ``[B, H, D]``. With ``repair_tiles`` the token order is the input order,
    ``k_perm`` and ``v8`` use the V repair layout with the repair tiles left
    unwritten, and ``energy`` holds each token's residual energy ``[B, H, S]``
    in place of the mean operand.
    """
    b, s_k, h, d = k.shape
    if d != _D:
        raise ValueError(f"VC fused preparation supports head_dim {_D} only")
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("k and v must be contiguous")
    dev = k.device
    t = (s_k + _TILE - 1) // _TILE
    repair = repair_tiles > 0
    rows = vc_repair_kv_len(s_k, repair_tiles) if repair else s_k
    tail_start = s_k // _TILE * _TILE if repair else s_k
    p1, p2 = _compiled(k.dtype, bool(demean), repair)
    perm32 = (
        torch.zeros((1,), dtype=torch.int32, device=dev)
        if repair
        else perm.to(torch.int32).contiguous()
    )
    k_perm = torch.empty((b, rows, h, d), dtype=k.dtype, device=dev)
    v8 = torch.empty((b, rows, h, d), dtype=torch.float8_e4m3fn, device=dev)
    mean = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    vamax = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    mu = torch.zeros(
        (1,)
        if repair
        else (
            b,
            h,
            (t + VC_MEAN_GROUP_TILES - 1) // VC_MEAN_GROUP_TILES,
            *vc_mean_group_shape(d),
        ),
        dtype=torch.bfloat16,
        device=dev,
    )
    energy = torch.empty(
        (b, h, s_k) if repair else (1,), dtype=torch.float32, device=dev
    )
    p1(
        k.view(-1),
        v.view(-1),
        perm32.view(-1),
        k_perm.view(-1),
        mean.view(-1),
        vamax.view(-1),
        s_k,
        h,
        t,
        b * h,
        rows,
        tail_start,
        repair_tiles * _TILE,
    )
    v_scale = (vamax.amax(dim=2) / E4M3_MAX).clamp_min(1e-12).contiguous()  # [B, H, D]
    p2(
        v.view(-1),
        perm32.view(-1),
        mean.view(-1),
        v_scale.view(-1),
        v8.view(-1),
        mu.view(-1),
        energy.view(-1),
        s_k,
        h,
        t,
        b * h,
        rows,
        tail_start,
        repair_tiles * _TILE,
    )
    return k_perm, v8, v_scale, mean, mu, energy
