"""GDN prefix-cache state materialization (persistent grid, single layer).

Reconstructs a request's SSM state at a chosen token in the recent past and
writes it to a SEPARATE pool slot, leaving the source checkpoint and the ring
untouched.  This is what a prefix cache needs: it wants the state at a round
token boundary (say every 128 tokens), which essentially never coincides with
where the u-cache checkpoint happens to sit.

    S_C = exp(G_C) * S0 + sum_{j<C} exp(G_C - g_j) * u_j (x) k_j,   G_C = g_{C-1}

Relationship to the decode kernels
----------------------------------
This is the FOLD of the u-cache decode kernels and nothing else.
Dropped: q load / L2-norm, the Gram matrices, the WY inverse, the output
tokens, the ring appends, and the cursor commit.  Two things change:

  - PARTIAL.  The decode fold always replays the whole window ``hist_len``;
    here the caller picks ``count <= hist_len``.  Because each ``u_j`` is the
    true per-token delta-rule correction (the WY machinery is only a batched
    way to compute them), truncating to any prefix is exact.
  - NON-DESTRUCTIVE.  Reads ``src_slots``, writes ``dst_slots`` -- same pool,
    different slot -- so the live request keeps decoding against its own
    checkpoint and ring.

NOT the fold-absorb variant.  The STP decode kernel deliberately pulls the
current token into the checkpoint and re-weights the GEMM by ``a0`` to match
vLLM's flush semantics.  Materialization wants the plain prefix fold, so there
is no ``a0`` here and no row-``P`` absorb.

Serves BOTH decode paths.  The MTP/spec-decode ring (``..._flush.py``, 32
slots) and the STP ring (``..._stp.py``, 16) differ only in depth -- element
layouts, dtypes and ``g`` semantics are identical.  The depth is derived from
``k_cache.shape[2]`` (never accepted from the caller; see the wrapper) and the
wrapper compiles ONE SPECIALIZATION PER DEPTH: the DLPack descriptor bakes the
ring dimension's stride, so a single compiled kernel cannot serve both.
Their other differences (fold-absorb commit semantics, single-row vs T-row
appends) are decode-side behaviours this kernel does not participate in: MTP's
filler rows sit past the live window, so a prefix of ``count <= hist_len``
never reads them.

No ``beta`` and no per-head ``A``: both are consumed at decode time when the
correction ``u`` is formed, so the ring already carries them.  16-bit state
only -- the 8-bit two-pass scale-discovery path is out of scope.

Scheduling
----------
A fixed pool of CTAs -- ``min(B * HV, SMs * 8)``, sized to the GPU, not the
batch -- grid-strides over (request, value head) work items, indirecting
through ``active_request_indices`` (live batch rows first, ``-1`` tail).  The
loop's trip count is ``num_active[0] * HV``, read from a device scalar, so a
launch with nothing to do retires after one load per CTA with no host-device
sync (CUDA-graph capturable).  Serving batches are padded to a fixed size for
graph capture, so at the realistic operating point almost every batch row is
idle; a one-CTA-per-item launch pays CTA creation for every padded row (32 us
of pure overhead at B=256/HV=64 on B200, vs ~2.5 us for this schedule).
Single layer per launch; the per-layer pointer-table extension is follow-up.

Tensors (entry point ``gdn_prefix_materialize``)
------------------------------------------------
  Name          Shape                  Dtype   Dir    Meaning
  ------------  ---------------------  ------  -----  -----------------------
  state         [pool, HV, V, K]       STATE   in/out both src and dst live here
  src_slots     [B]                    int32   in     read checkpoint from
  dst_slots     [B]                    int32   in     write result to
  k_cache       [pool, H,  R, K]        RING    in     ring: L2-normalized keys
  u_cache       [pool, HV, R, V]        RING    in     ring: corrections
  g_cache       [pool, HV, R]           f32     in     ring: cumulative log-decay
  cache_base    [B]                    int32   in     ring window origin
  count         [B]                    int32   in     entries to replay, <= 16
  active_request_indices [B]           int32   in     live rows first, then -1
  num_active    [1]                    int32   in     device scalar: live rows

The wrapper derives ``active_request_indices`` / ``num_active`` from ``count``
when omitted (host-side; fine for eager calls).  A CUDA-graph caller should
build both on-device and pass them in -- that sync-free path is the reason
they are tensors at all.  ``num_active`` holds the number of LIVE REQUESTS
(not items); it need not be tight for correctness, only cover every live
entry -- the per-item skip guard stands alone.  ``-1`` entries are SKIPPED,
not a hard stop (see the contract note in the kernel body).

``R`` is the ring depth, read from the tensors: 32 for MTP, 16 for STP.  The
replay window is capped at ``W_RING`` (16, one MMA tile) either way -- MTP's
legal ``hist_len`` reaches 16, STP's 15.

A (request) is skipped when ``src_slots`` or ``dst_slots`` is negative, when
``count`` is negative, or when ``count`` exceeds the window depth -- matching
the mamba materialize kernel, which also skips rather than raising.

``count == 0`` is an exact copy.  It runs through the same GEMM path with
``bdec = 1`` and an all-zero weight vector, which is bit-exact with a byte copy
(the state round-trips through f32 unchanged) and keeps to one code path.
"""

from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.experimental  # noqa: F401  # registers cute.experimental.jit
import cutlass.utils as utils
from cutlass.cute.arch import sync_threads
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64

# The decode kernel owns the PTX helpers, the dtype selection (import-time env
# vars) and the swizzle/MMA geometry.  Importing them keeps the two files in
# lockstep: a change to the fold's arithmetic or the SW128 layout lands here
# automatically instead of drifting in a copy.
#
# Deliberately the MTP/spec-decode module (PR #4081, merged on main) rather than
# the STP fork (PR #4974, still open): this file then depends only on landed
# code and can be reviewed and merged on its own.  STP is still served -- the
# two rings differ ONLY in depth, which is a runtime argument here, and the fork
# left the fold machinery and element layouts untouched.
from .gdn_decode_bf16_wy_ucache_flush import (
    K_DIM,
    K_HALF,
    K_PADDED,
    T,
    THREADS,
    TK_PAD,
    V_DIM_C,
    V_PADDED,
    W_RING,
    WARP,
    _cp_async_bf16x8_cg,
    _cp_async_commit_group,
    _cp_async_wait_group_0,
    _exp_approx_f32,
    _fold_fma_bf16x2,
    _lds_b32,
    _lds_v4_b32,
    _ldmatrix_x4_trans,
    _make_sH_sw128_layout_half,
    _mul_rg2_f32,
    _qtv_4mma_rg,
    _st_global_v4_b32,
    _sts_st2_f32,
    _sw128_xor,
    f32,
    io,
    state_ty,
)

__all__ = ["gdn_prefix_materialize"]


class GdnPrefixMaterializeKernel:
    """Replay a ring prefix onto a checkpoint, into a different pool slot."""

    # min_blocks_per_mp is a REGISTER budget, not just a hint: without it the
    # compiler spends ~256 regs/thread and occupancy collapses to 2 CTAs/SM
    # (12.5%), leaving the kernel latency-bound at ~1.65 TB/s. At 8 the SMEM
    # bound (~25 KB/CTA on a 228 KB SM) is met, the B=256 full-fold time drops
    # 385 -> 188 us, and the count=0 copy hits 174 us -- the same 1.07 GB
    # slot-to-slot copy measured at 172 us with a plain torch contiguous copy,
    # i.e. the kernel sits on the memory floor. The 2 default this file
    # originally inherited is correct for the T=4 decode kernel (whose SMEM
    # footprint only fits 2 CTAs/SM) and wrong here.
    def __init__(self, min_blocks_per_mp=8):
        self._min_blocks_per_mp = min_blocks_per_mp

    @cute.experimental.jit
    def __call__(
        self,
        gH0: cute.Tensor,
        gSrc: cute.Tensor,
        gDst: cute.Tensor,
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gBase: cute.Tensor,
        gCount: cute.Tensor,
        gActive: cute.Tensor,
        gNumActive: cute.Tensor,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        ring_slots: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        op = MmaF16BF16Op(io, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op)
        # Same half-K TMA atom as the decode kernel: logical (pool, HV, V, K)
        # reordered to (V, K, HV, pool) so the per-CTA box is (V_DIM_C, K_HALF)
        # and the trailing modes survive tma_partition as outer coords.
        gH0_vkhp = cute.make_tensor(
            gH0.iterator,
            cute.select(gH0.layout, mode=[2, 3, 1, 0]),
        )
        sH_tma_layout = _make_sH_sw128_layout_half()
        tma_atom_h, tma_tensor_h = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gH0_vkhp,
            sH_tma_layout,
            (V_DIM_C, K_HALF),
        )
        self.kernel(
            gH0,
            gSrc,
            gDst,
            gKC,
            gUC,
            gGC,
            gBase,
            gCount,
            gActive,
            gNumActive,
            tiled_mma,
            HV,
            V_DIM,
            H,
            ring_slots,
            tma_atom_h,
            tma_tensor_h,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=[THREADS, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=self._min_blocks_per_mp,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        gH0: cute.Tensor,
        gSrc: cute.Tensor,
        gDst: cute.Tensor,
        gKC: cute.Tensor,
        gUC: cute.Tensor,
        gGC: cute.Tensor,
        gBase: cute.Tensor,
        gCount: cute.Tensor,
        gActive: cute.Tensor,
        gNumActive: cute.Tensor,
        tiled_mma: cute.TiledMma,
        HV: cutlass.Int32,
        V_DIM: cutlass.Int32,
        H: cutlass.Int32,
        ring_slots: cutlass.Int32,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane_id = tidx & 31
        warp_id = tidx // WARP
        # Ring depth is a kernel PARAMETER, not a module constant: the MTP ring
        # is 32 slots and the STP ring 16, and the wrapper compiles one
        # specialization per depth (it must -- the descriptor bakes the ring
        # dimension's stride, so a single kernel cannot serve both). The wrapper
        # derives the depth from k_cache rather than accepting it, so a caller
        # cannot pair a 32 with a 16-deep allocation and have the index run off
        # the end of one request's ring into the next pool slot.
        ring_mask = ring_slots - Int32(1)

        smem = utils.SmemAllocator()

        @cute.struct
        class SS:
            # 8B-aligned mbarrier first, then the 128-aligned tiles.
            h_load_mbar: cute.struct.MemRange[Int64, 1]
            # A operand: the w-scaled u window, [W_RING rows, V cols].
            k_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # B operand: the khist window, [W_RING rows, K cols].
            q_buf: cute.struct.Align[cute.struct.MemRange[io, TK_PAD], 128]
            # State tile: V=128 rows x K_HALF=64 cols, SW128 swizzled, reused
            # across the two TMA half-loads via an mbarrier phase flip.
            h_buf: cute.struct.Align[
                cute.struct.MemRange[state_ty, V_DIM_C * K_HALF], 128
            ]
            # [0:16] = g_j, [16:32] = w_j, [32] = bdec.
            ghist_fp32: cute.struct.Align[cute.struct.MemRange[f32, 48], 128]

        st = smem.allocate(SS)
        sK = st.k_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        sQ = st.q_buf.get_tensor(cute.make_layout((T, K_PADDED), stride=(K_PADDED, 1)))
        sH_layout = _make_sH_sw128_layout_half()
        sH = st.h_buf.get_tensor(sH_layout.outer, swizzle=sH_layout.inner)
        sGhist = st.ghist_fp32.get_tensor(cute.make_layout((48,)))
        _sK_i32 = cute.recast_tensor(sK, cutlass.Int32)
        _sQ_i32 = cute.recast_tensor(sQ, cutlass.Int32)
        _sK_base = sK.iterator.toint()
        _sQ_base = sQ.iterator.toint()
        _sH_base = sH.iterator.toint()

        mbar_h_ptr = st.h_load_mbar.data_ptr()
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_h_ptr, 1)
        cute.arch.mbarrier_init_fence()

        sync_threads()

        # ---- persistent schedule -------------------------------------------
        # A FIXED pool of CTAs -- sized to the GPU, not to the batch -- grid-
        # strides over the (request, head) work list. The batch is padded to a
        # constant for CUDA-graph capture, so a one-CTA-per-item launch spends
        # nearly all of its time creating CTAs that read a sentinel and retire:
        # each one still reserves this kernel's ~25 KB of SMEM before any of its
        # code runs, so an early return does not refund the slot. Measured at
        # B=256/HV=64, the all-skipped launch cost 32 us for zero work, and the
        # cost tracked CTA count linearly across the TP shapes.
        #
        # Requests are addressed through gActive (the compacted active list):
        # entry v gives the PHYSICAL batch row of the v-th live request, and the
        # tail is -1. Everything else here is indexed physically. There is no
        # early break -- cutlass.range needs a trip count up front -- so a CTA
        # past the sentinel still walks its remaining items, but each is a
        # predicate test rather than a CTA launch. CONTRACT NOTE: -1 entries are
        # SKIPPED, not a hard stop, so unlike PR #4815 a malformed list with a
        # live index after a -1 would still process it; for the well-formed
        # live-prefix-then-sentinels list the two behaviors are identical.
        # The loop bound is the LIVE item count, read from a device scalar the
        # caller maintains next to the active list (n_live * HV). cutlass.range
        # has no break, so bounding by B * HV would make every CTA walk its full
        # ~B*HV/grid item share even when nothing is live -- one dependent
        # gActive load per item, ~14 serialized loads deep at B=256. Measured:
        # the all-idle launch cost 21 us with the B-based bound and ~6 us with
        # this one. This is the same early-exit the mamba kernel gets from its
        # break-at-sentinel, expressed as a runtime trip count (the
        # gdn_decode_bf16_state.py idiom). The -1 skip below stays as a
        # belt-and-braces guard; correctness does not depend on the scalar
        # being tight, only on it covering every live entry.
        num_items = gNumActive.iterator[0] * HV
        _cap = Int32(cute.size(gCount.shape[0])) * HV
        if num_items > _cap:
            num_items = _cap
        _n_mine = (num_items - bidx + gdim - Int32(1)) // gdim
        if _n_mine < Int32(0):
            _n_mine = Int32(0)
        for _j in cutlass.range(_n_mine, unroll=1):
            _item = bidx + _j * gdim
            head = _item % HV
            _vr = _item // HV
            i_h = head // (HV // H)

            phys = gActive.iterator[_vr]
            # Skipped: past the active list, a padded slot, a skipped request,
            # or an over-long prefix. Over-long is SKIPPED, not an error --
            # same contract as the mamba materialize kernel's count > MAX_WINDOW.
            # Every term is CTA-uniform, so the barriers inside stay safe.
            _ok = Int32(1)
            if phys < Int32(0):
                _ok = Int32(0)
            count = Int32(0)
            src_idx = Int32(-1)
            dst_idx = Int32(-1)
            ring_base = Int32(0)
            if _ok == Int32(1):
                count = gCount.iterator[phys]
                src_idx = gSrc.iterator[phys]
                dst_idx = gDst.iterator[phys]
                ring_base = gBase.iterator[phys]
                if src_idx < Int32(0):
                    _ok = Int32(0)
                if dst_idx < Int32(0):
                    _ok = Int32(0)
                if count < Int32(0):
                    _ok = Int32(0)
                if count > Int32(W_RING):
                    _ok = Int32(0)

            if _ok == Int32(1):
                # 64-bit per-CTA pool ELEMENT offsets: block-strided paged pools can
                # exceed 2^31 elements, so the pool term stays 64-bit and every
                # remaining per-lane offset is intra-page 32-bit.
                skc_pool = gKC.layout.stride[0]
                skc_h = gKC.layout.stride[1]
                suc_pool = gUC.layout.stride[0]
                suc_hv = gUC.layout.stride[1]
                sgc_pool = gGC.layout.stride[0]
                sgc_hv = gGC.layout.stride[1]
                src64 = Int64(src_idx)
                _kc_pool_e64 = src64 * skc_pool + Int64(i_h) * skc_h
                _uc_pool_e64 = src64 * suc_pool + Int64(head) * suc_hv
                _gc_pool_e64 = src64 * sgc_pool + Int64(head) * sgc_hv

                # g ring LDG on warp 0; lanes >= W_RING keep 0.0. g_cache is f32 so
                # the load is direct. Logical row j lives at (base + j) & ring_mask.
                _g_hist_f32 = f32(0.0)
                if warp_id == 0 and lane_id < Int32(W_RING):
                    _g_hist_f32 = gGC.iterator[
                        _gc_pool_e64 + Int64((ring_base + lane_id) & ring_mask)
                    ]

                # Zero BOTH operand tiles before the ring loads. Rows >= count are
                # never written by the cp.async below, and uninitialized SMEM can hold
                # NaN bit patterns -- a zero A row does not save us there, because
                # 0 * NaN is NaN. The decode kernel avoids this by restoring a full
                # 16-row register snapshot; we zero instead.
                _ZI32 = (T * K_PADDED) // 2
                for _z in cutlass.range_constexpr(9):
                    _z_i = tidx + _z * Int32(THREADS)
                    if _z_i < Int32(_ZI32):
                        _sK_i32.iterator[_z_i] = Int32(0)
                        _sQ_i32.iterator[_z_i] = Int32(0)
                sync_threads()

                # --- w_j and bdec ---------------------------------------------------
                # w_j = e^{G_C - g_j} for j < count else 0. The j >= count mask is
                # MANDATORY: stale ring slots would give inf * 0 = NaN. G_C is lane
                # (count-1)'s value, shuffled -- all 32 lanes execute the shuffle.
                if warp_id == 0:
                    _c_src = count - Int32(1) if count > Int32(0) else Int32(0)
                    _gc_lane = cute.arch.shuffle_sync(
                        _g_hist_f32, _c_src, Int32(0xFFFFFFFF), Int32(0x1F)
                    )
                    _gc = _gc_lane if count > Int32(0) else f32(0.0)
                    if lane_id < Int32(W_RING):
                        _w_j = (
                            _exp_approx_f32(_gc - _g_hist_f32)
                            if lane_id < count
                            else f32(0.0)
                        )
                        sGhist.iterator[W_RING + lane_id] = _w_j
                    if lane_id == Int32(0):
                        # count == 0 degenerates to an exact copy: bdec = 1 and every
                        # w_j = 0, so S_out = 1 * S0 + 0. Exact, not approximate --
                        # the value round-trips through f32 unchanged.
                        _bdec_v = _exp_approx_f32(_gc) if count > Int32(0) else f32(1.0)
                        sGhist.iterator[32] = _bdec_v

                # --- ring loads: u -> sK (A operand), k -> sQ (B operand) ------------
                # 128 threads x 2 iterations = 256 groups; 16 chunks of 8 elements per
                # row covers all W_RING = 16 rows of a 128-wide tile.
                _gUC_base = gUC.iterator.toint() + _uc_pool_e64 * 2
                _gKC_base = gKC.iterator.toint() + _kc_pool_e64 * 2
                for _fr in cutlass.range_constexpr(2):
                    _fr_group = tidx + _fr * Int32(THREADS)
                    _fr_row = _fr_group // Int32(V_DIM_C // 8)
                    _fr_col = (_fr_group % Int32(V_DIM_C // 8)) * Int32(8)
                    if _fr_row < count:
                        _cp_async_bf16x8_cg(
                            _gUC_base,
                            ((ring_base + _fr_row) & ring_mask) * V_DIM + _fr_col,
                            _sK_base
                            + _fr_row * Int32(K_PADDED * 2)
                            + _fr_col * Int32(2),
                        )
                        _cp_async_bf16x8_cg(
                            _gKC_base,
                            ((ring_base + _fr_row) & ring_mask) * Int32(K_DIM)
                            + _fr_col,
                            _sQ_base
                            + _fr_row * Int32(K_PADDED * 2)
                            + _fr_col * Int32(2),
                        )
                _cp_async_commit_group()
                _cp_async_wait_group_0()
                sync_threads()

                # --- b_d = w_j * u_j, in place over the u stage ----------------------
                # W_RING * (V_PADDED/2) i32 pairs; rows >= count keep w = 0, which
                # zeroes both the padding columns and any stale bytes.
                _kpad_i32 = K_PADDED // 2
                for _ws in cutlass.range_constexpr(9):
                    _ws_g = tidx + _ws * Int32(THREADS)
                    _ws_r = _ws_g // Int32(V_PADDED // 2)
                    _ws_c = _ws_g % Int32(V_PADDED // 2)
                    if _ws_r < Int32(W_RING):
                        _w_row = sGhist.iterator[W_RING + _ws_r]
                        _sK_i32.iterator[_ws_r * _kpad_i32 + _ws_c] = _mul_rg2_f32(
                            _sK_i32.iterator[_ws_r * _kpad_i32 + _ws_c], _w_row
                        )

                # --- TMA partition for (src_slot, head) ---------------------------
                gH_tiled = cute.flat_divide(tma_tensor_h, (V_DIM_C, K_HALF))
                gH_slice0 = gH_tiled[None, None, None, 0, head, src_idx]
                gH_slice1 = gH_tiled[None, None, None, 1, head, src_idx]
                sH_grp = cute.group_modes(sH, 0, 2)
                tHsH0, tHgH0 = cpasync.tma_partition(
                    tma_atom_h,
                    0,
                    cute.make_layout(1),
                    sH_grp,
                    cute.group_modes(gH_slice0, 0, 3),
                )
                tHsH1, tHgH1 = cpasync.tma_partition(
                    tma_atom_h,
                    0,
                    cute.make_layout(1),
                    sH_grp,
                    cute.group_modes(gH_slice1, 0, 3),
                )

                # Destination byte base. Pool/head strides come from the descriptor
                # layout, not dense shape products, so paged pools work.
                _gH0_st = (
                    gH0.iterator.toint()
                    + (
                        Int64(dst_idx) * gH0.layout.stride[0]
                        + Int64(head) * gH0.layout.stride[1]
                    )
                    * 2
                )
                _bdec_t = sGhist.iterator[32]
                # Lane->fragment maps for the fold's m16n8k16 MMAs (identical to the
                # decode kernel's fold): _ldm_row = B-operand ldmatrix row for lane;
                # _fa_row/_fa_colb = A-operand ldmatrix.x4.trans row / 16-B column
                # group; _fc_r0/_fc_c0 = accumulator C-fragment (row, col pair) origin.
                _ldm_row = (lane_id % 8) + ((lane_id // 8) % 2) * Int32(8)
                _fa_row = (lane_id & Int32(7)) + (
                    (lane_id >> Int32(4)) & Int32(1)
                ) * Int32(8)
                _fa_colb = ((lane_id >> Int32(3)) & Int32(1)) * Int32(16)
                _fc_r0 = lane_id // Int32(4)
                _fc_c0 = (lane_id & Int32(3)) * Int32(2)

                # --- fold: one K-half at a time -------------------------------------
                # All four warps work each half (2 V-strips per warp x 2 column
                # spans). Epilogue writes S_h back into the SAME swizzled sH bytes,
                # then flushes to the pool in fully-coalesced 16-B chunks.
                for _fh in cutlass.range_constexpr(2):
                    # LOAD-BEARING for _fh == 1: every warp must finish its half-0
                    # coalesced-flush LDS reads of sH before warp 0's TMA overwrites
                    # the tile. Also publishes the w-scaled u stage on _fh == 0.
                    sync_threads()
                    if warp_id == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar_h_ptr, V_DIM_C * K_HALF * 2
                            )
                        if _fh == 0:
                            cute.copy(tma_atom_h, tHgH0, tHsH0, tma_bar_ptr=mbar_h_ptr)
                        else:
                            cute.copy(tma_atom_h, tHgH1, tHsH1, tma_bar_ptr=mbar_h_ptr)
                    cute.arch.mbarrier_wait(mbar_h_ptr, _fh & 1)
                    cute.arch.fence_view_async_shared()
                    sync_threads()
                    for _fs2 in cutlass.range_constexpr(2):
                        _fs = warp_id * Int32(2) + Int32(_fs2)
                        for _fspan in cutlass.range_constexpr(2):
                            _fa0, _fa1, _fa2, _fa3 = _ldmatrix_x4_trans(
                                _sK_base
                                + _fa_row * Int32(K_PADDED * 2)
                                + _fs * Int32(32)
                                + _fa_colb
                            )
                            _frr = _qtv_4mma_rg(
                                _fa0,
                                _fa1,
                                _fa2,
                                _fa3,
                                _sQ_base
                                + _ldm_row * Int32(K_PADDED * 2)
                                + Int32(_fh * 128 + _fspan * 64),
                            )
                            for _ft in cutlass.range_constexpr(4):
                                _f_col = Int32(_fspan * 32 + _ft * 8) + _fc_c0
                                _f_row0 = _fs * Int32(16) + _fc_r0
                                _sh_a0 = _sw128_xor(
                                    _sH_base
                                    + _f_row0 * Int32(K_HALF * 2)
                                    + _f_col * Int32(2)
                                )
                                _s0p = _lds_b32(_sh_a0)
                                _sf0, _sf1 = _fold_fma_bf16x2(
                                    _s0p, _bdec_t, _frr[_ft * 4], _frr[_ft * 4 + 1]
                                )
                                _sts_st2_f32(_sh_a0, _sf0, _sf1)
                                _sh_a1 = _sw128_xor(
                                    _sH_base
                                    + (_f_row0 + Int32(8)) * Int32(K_HALF * 2)
                                    + _f_col * Int32(2)
                                )
                                _s1p = _lds_b32(_sh_a1)
                                _sf2, _sf3 = _fold_fma_bf16x2(
                                    _s1p, _bdec_t, _frr[_ft * 4 + 2], _frr[_ft * 4 + 3]
                                )
                                _sts_st2_f32(_sh_a1, _sf2, _sf3)
                    sync_threads()
                    # Coalesced flush of the updated half: consecutive lanes write
                    # consecutive 16-B chunks (8 chunks = one 128-B swizzle period).
                    for _fc in cutlass.range_constexpr(8):
                        _f_chunk = tidx + _fc * Int32(THREADS)
                        _f_row = _f_chunk >> Int32(3)
                        _f_pos = _f_chunk & Int32(7)
                        _cv0, _cv1, _cv2, _cv3 = _lds_v4_b32(
                            _sw128_xor(
                                _sH_base
                                + _f_row * Int32(K_HALF * 2)
                                + _f_pos * Int32(16)
                            )
                        )
                        _st_global_v4_b32(
                            _gH0_st,
                            _f_row * Int32(K_DIM)
                            + Int32(_fh * K_HALF)
                            + _f_pos * Int32(8),
                            _cv0,
                            _cv1,
                            _cv2,
                            _cv3,
                        )

            # End-of-item barrier. The next item re-tenants sK/sQ/sH, and the
            # fold's stores are partitioned across warps -- without this a fast
            # warp could zero the operand tiles while a slow one is still
            # ldmatrix-ing them. A one-item-per-CTA kernel gets this for free by
            # exiting; a persistent one does not.
            sync_threads()


_CACHE: dict = {}


def gdn_prefix_materialize(
    state: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    k_cache: torch.Tensor,
    u_cache: torch.Tensor,
    g_cache: torch.Tensor,
    cache_base: torch.Tensor,
    count: torch.Tensor,
    active_request_indices: Optional[torch.Tensor] = None,
    num_active: Optional[torch.Tensor] = None,
    *,
    min_blocks_per_mp: int = 8,
) -> torch.Tensor:
    """Materialize prefix states into ``state[dst_slots]``.

    See the module docstring for the contract. ``state`` is mutated in place at
    the destination slots and returned for convenience; sources and rings are
    read-only.

    The two easy-to-get-wrong optional arguments (no exception fires if they
    are wrong -- results are just silently incomplete):

    active_request_indices : [B] CUDA int32
        The PHYSICAL batch rows to process, compacted to the front, ``-1``
        filling the tail. Order among live entries is irrelevant. Derived
        from ``count >= 0`` on the host when omitted; a CUDA-graph caller
        must build it on-device and pass it, since the derivation here syncs.
    num_active : [1] CUDA int32
        Number of LIVE ENTRIES in the list above (requests, not
        request-times-head items). Gates the kernel's early exit only; the
        per-item sentinel guard keeps a loose value correct, but a value
        SMALLER than the true live count silently drops the excess requests.
    """
    if state.dim() != 4:
        raise ValueError("state must be [pool, HV, V, K]")
    if k_cache.dim() != 4 or u_cache.dim() != 4 or g_cache.dim() != 3:
        raise ValueError(
            "k_cache [pool,H,RING,K], u_cache [pool,HV,RING,V], g_cache [pool,HV,RING]"
        )
    _pool, HV, V_dim, K_dim = state.shape
    H = k_cache.shape[1]
    if K_dim != K_DIM or V_dim != V_DIM_C:
        raise ValueError(f"this kernel is fixed at K == V == {K_DIM}")
    if HV % H:
        raise ValueError("HV must be a multiple of H")
    if k_cache.shape[3] != K_dim or u_cache.shape[3] != V_dim:
        raise ValueError("ring feature dims must match the state")
    # Ring depth is DERIVED from the buffer, never taken as an argument: a
    # caller-supplied depth that disagreed with the allocation would make the
    # kernel's (base + j) & (depth - 1) index run off the end of one request's
    # ring and into the next pool slot -- wrong data, no error. Deriving it
    # makes that mismatch unrepresentable. 32 is the MTP ring, 16 the STP ring.
    ring_slots = k_cache.shape[2]
    if u_cache.shape[2] != ring_slots or g_cache.shape[2] != ring_slots:
        raise ValueError(
            f"ring depths disagree: k_cache={ring_slots}, "
            f"u_cache={u_cache.shape[2]}, g_cache={g_cache.shape[2]}"
        )
    if ring_slots & (ring_slots - 1) or ring_slots < W_RING:
        raise ValueError(
            f"ring depth must be a power of two and at least the replay window "
            f"({W_RING}); got {ring_slots}"
        )
    if g_cache.dtype != torch.float32:
        raise ValueError("g_cache must be float32")
    for name, t in (
        ("src_slots", src_slots),
        ("dst_slots", dst_slots),
        ("cache_base", cache_base),
        ("count", count),
    ):
        if t.dtype != torch.int32 or t.dim() != 1:
            raise ValueError(f"{name} must be a 1-D int32 tensor")
    B = src_slots.shape[0]
    if not (dst_slots.shape[0] == cache_base.shape[0] == count.shape[0] == B):
        raise ValueError("per-request tensors must all have length B")
    if active_request_indices is None:
        # Derive the compacted list from count on the host. Fine for eager
        # calls; a CUDA-graph caller should build it on-device (it is data the
        # scheduler already has) and pass it in, since this sync-free path is
        # the reason the kernel takes the tensor at all.
        active_request_indices = torch.full_like(count, -1)
        live = torch.nonzero(count >= 0, as_tuple=False).flatten().to(torch.int32)
        active_request_indices[: live.numel()] = live
        if num_active is None:
            num_active = torch.tensor(
                [live.numel()], dtype=torch.int32, device=count.device
            )
    if num_active is None:
        # Caller gave a list but no count: cover the whole list. Correct (the
        # per-item -1 guard skips the tail) but forfeits the early exit; pass
        # the device scalar to get it back.
        num_active = torch.tensor([B], dtype=torch.int32, device=count.device)
    if (
        num_active.dtype != torch.int32
        or num_active.numel() != 1
        or not num_active.is_cuda
    ):
        raise ValueError("num_active must be a one-element CUDA int32 tensor")
    elif (
        active_request_indices.dtype != torch.int32
        or active_request_indices.dim() != 1
        or active_request_indices.shape[0] != B
    ):
        raise ValueError(
            "active_request_indices must be a 1-D int32 tensor of length B"
        )
    if state.dtype != state_ty_torch():
        raise ValueError(
            f"state dtype {state.dtype} does not match the module's STATE dtype; "
            "set GDN_UCACHE_STATE_DTYPE / GDN_UCACHE_IO_DTYPE to match"
        )

    device = state.device
    stream = cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)
    # Persistent grid: one resident wave, sized to the GPU rather than the
    # batch. 8 blocks/SM is the SMEM bound for this kernel's ~25 KB CTA
    # footprint on SM90/SM100-class parts (228 KB/SM); the same hardcoded-
    # blocks/SM convention as packed_kda_decode_cute's persistent schedule.
    # Capped by the item count so tiny batches do not over-launch.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    grid_ctas = min(B * HV, sms * 8)

    # Paged serving pools (vLLM-style) are strided VIEWS: inner dims dense but
    # the dim-0 slot stride padded up to a page. The kernel supports exactly
    # that shape of non-contiguity -- pool/head strides are read from the
    # descriptor layout -- and nothing more: ring-row and feature strides are
    # hardcoded dense in the kernel, so a tensor strided in an INNER dim would
    # be described faithfully and then read wrongly, silently. Validate the
    # boundary here so unsupported layouts stay a loud error.
    def _check_pool_layout(t, name):
        if t.is_contiguous():
            return
        dense = 1
        for i in range(t.dim() - 1, 0, -1):
            if t.stride(i) != dense:
                raise ValueError(
                    f"{name}: only the slot (dim-0) stride may be padded; "
                    f"dim {i} has stride {t.stride(i)}, expected dense {dense}"
                )
            dense *= t.shape[i]
        if t.stride(0) < dense:
            raise ValueError(f"{name}: overlapping slot stride {t.stride(0)}")
        # 16-byte alignment: the state TMA box and the u/k 16 B cp.async loads
        # index from slot-relative offsets, so every slot must start aligned.
        if t.dtype != torch.float32 and (t.stride(0) * t.element_size()) % 16:
            raise ValueError(
                f"{name}: padded slot stride must keep slots 16-byte aligned "
                f"(stride {t.stride(0)} elements x {t.element_size()} B)"
            )

    for _n, _t in (
        ("state", state),
        ("k_cache", k_cache),
        ("u_cache", u_cache),
        ("g_cache", g_cache),
    ):
        _check_pool_layout(_t, _n)
    for _n, _t in (
        ("src_slots", src_slots),
        ("dst_slots", dst_slots),
        ("cache_base", cache_base),
        ("count", count),
        ("active_request_indices", active_request_indices),
        ("num_active", num_active),
    ):
        # Metadata takes the same descriptor path but is NOT in the compile
        # cache key; a strided view here would bake B and its stride into a
        # cubin that a later call could silently reuse. Cheap to require.
        if not _t.is_contiguous():
            raise ValueError(f"{_n} must be contiguous (got a strided view)")

    def mk_dyn(t):
        cute_tensor = from_dlpack(t, assumed_align=16)
        if t.is_contiguous():
            return cute_tensor.mark_compact_shape_dynamic(
                mode=0, stride_order=tuple(range(t.dim())), divisibility=1
            )
        # Padded slot stride (validated above): keep the exact layout static
        # and specialize on it via the cache key below. mode-0 stays static
        # too, so each distinct paged layout compiles once.
        return cute_tensor

    args = [
        mk_dyn(state),
        mk_dyn(src_slots),
        mk_dyn(dst_slots),
        mk_dyn(k_cache),
        mk_dyn(u_cache),
        mk_dyn(g_cache),
        mk_dyn(cache_base),
        mk_dyn(count),
        mk_dyn(active_request_indices),
        mk_dyn(num_active),
        HV,
        V_dim,
        H,
        ring_slots,
        grid_ctas,
        stream,
    ]
    # ring_slots IS part of the key. A dynamic scalar is not enough on its own:
    # mk_dyn marks only mode 0 (the pool dimension) dynamic, so the ring
    # dimension's STRIDE is baked into the compiled descriptor -- a kernel built
    # against a 32-deep cache carries stride = 32*K and misreads a 16-deep one
    # even with a correct mask. Keying on the depth gives one specialization per
    # ring, chosen automatically; MTP and STP callers each get the right one.
    cache_key = (
        str(device),
        int(HV),
        int(H),
        int(V_dim),
        int(ring_slots),
        int(min_blocks_per_mp),
        str(state.dtype),
        str(k_cache.dtype),
        # Non-contiguous (paged) tensors use STATIC descriptors, so their
        # exact layout is baked into the cubin -- key on it. Empty for the
        # all-contiguous case, preserving the single shared specialization.
        tuple(
            (str(n), tuple(t.shape), tuple(t.stride()))
            for n, t in (
                ("state", state),
                ("k_cache", k_cache),
                ("u_cache", u_cache),
                ("g_cache", g_cache),
            )
            if not t.is_contiguous()
        ),
    )
    if cache_key not in _CACHE:
        _CACHE[cache_key] = cute.compile(
            GdnPrefixMaterializeKernel(min_blocks_per_mp=min_blocks_per_mp),
            *args,
        )
    _CACHE[cache_key](*args)
    return state


def state_ty_torch() -> torch.dtype:
    """The torch dtype matching the module's compile-time STATE element type."""
    return torch.float16 if state_ty is cutlass.Float16 else torch.bfloat16
