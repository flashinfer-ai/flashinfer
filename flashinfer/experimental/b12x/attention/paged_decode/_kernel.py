"""Warp-specialized split-KV paged decode/verify attention kernels (SM120).

One CTA owns every query row of a request for one KV head (rows = q_len x
GQA group, up to 128), so K/V stream once per request instead of once per
query token.  A producer warp issues paged TMA stages gated per stage by
full/empty mbarriers; consumer warps split at run time into row groups and key
groups (key group ``kg`` owns tiles ``kg, kg + KS, ...``), then merge key
groups through shared memory.

* Q is loaded straight into MMA fragments; all shared memory is KV pipeline.
* Split-KV ranges are derived on device from ``cache_seqlens``: the grid
  depends only on the batch capacity, so a captured graph is valid for any
  context length and any mix of per-request query lengths (ragged verify).
* Split partials are FP32 (normalized output + base-2 LSE); the merge kernel
  folds them, and the attention sink, into the BF16 output with one rounding.
* Variants (constexpr): QK/VO head dims 192/128 or 128/128; causal, or
  non-causal (every row sees the whole sequence, bounded on the left by the
  window); BF16 or FP8-E4M3 KV (FP8 fragments are widened exactly to BF16 in
  registers after ``ldmatrix``, so the math is the BF16 math and the FP8 stage
  ring needs no widening scratch).

Softmax statistics are FP32; P is BF16 for the PV MMA, as in FA2-style
kernels.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import Int64
from cutlass.cute.nvgpu import cpasync

from b12x.attention._shared.cute import copy as cute_copy
from b12x.attention._shared.cute import ops as attention_ops
from b12x._lib.intrinsics.mma import bf16_mma_m16n16k16_f32
from b12x._lib.intrinsics.packed import (
    bf16_rowsum_m16k16_f32,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
)
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_v4_f32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m16n16x1_trans_b8,
    shared_ptr_to_u32,
    st_global_v2_f32,
)
from b12x.attention.paged.forward_paged import (
    _assume_tensor_aligned,
    _exit_thread,
    _get_memrange_tensor,
    _literal_pv_mma_into_ofrag_plane_bf16_packed,
    _literal_qk_mma_into_sfrag_plane_bf16,
    _literal_update_mdo_states_fp32_pack_p,
    _make_paged_kv_tma_source_tensor,
    _make_payload_memrange,
    _paged_kv_tma_plane_layout,
    _paged_kv_tma_plane_stage_layout,
)

LOG2_E = 1.4426950408889634
PLANE_DIM = 64
SMEM_LIMIT = 99 * 1024


@cute.jit
def _mask_subtile(
    frag_S: cute.Tensor,
    key_base,
    tile_base,
    tile_tokens,
    lane_pair_base,
    causal_hi: cute.Tensor,
    window_lo: cute.Tensor,
    use_window: cutlass.Constexpr,
):
    for reg_id in cutlass.range_constexpr(8):
        row_slot = (reg_id % 4) // 2
        key_local = key_base + lane_pair_base + 8 * (reg_id // 4) + (reg_id % 2)
        key_abs = tile_base + key_local
        valid = key_local < tile_tokens
        valid = valid and key_abs <= causal_hi[row_slot]
        if const_expr(use_window):
            valid = valid and key_abs >= window_lo[row_slot]
        if not valid:
            frag_S[0, 0, reg_id] = Float32(-Float32.inf)


@cute.jit
def _issue_kv_stage_at(
    load0,
    load1,
    load2,
    num_planes: cutlass.Constexpr,
    stage_rows: cutlass.Constexpr,
    buf_idx,
    full_mbar_ptr,
    expected_bytes,
    page_id,
    tile_token_base,
    page_size,
    tiles_per_entry,
):
    page_idx = tile_token_base // page_size
    page_tile_idx = (tile_token_base - page_idx * page_size) // stage_rows
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(full_mbar_ptr, expected_bytes)
    src_idx = Int64(page_id) * Int64(tiles_per_entry) + Int64(page_tile_idx)
    load0(src_idx=src_idx, dst_idx=buf_idx, tma_bar_ptr=full_mbar_ptr)
    if const_expr(num_planes >= 2):
        load1(src_idx=src_idx, dst_idx=buf_idx, tma_bar_ptr=full_mbar_ptr)
    if const_expr(num_planes == 3):
        load2(src_idx=src_idx, dst_idx=buf_idx, tma_bar_ptr=full_mbar_ptr)


@cute.jit
def _qk_fp8_into_sfrag(
    frag_S, q_regs, k_plane0, k_plane1, lane, key_base, num_mma_d_qk
):
    """S += Q K^T over one 16-key sub-tile of 128-byte FP8 K rows.

    ``ldmatrix`` reads the swizzled FP8 rows as b16 pairs: lane (g, t) of an
    8x8 matrix receives key g's bytes 4t..4t+3 of a 16-column step, widened
    exactly to two BF16 pairs that serve as the B fragment's k 2t..2t+1 and
    2t+8..2t+9.  Q's registers hold the same columns in the same order, so
    every step multiplies matching dims.  One ``x4`` load covers two steps of
    both 8-key halves; a 192-wide head's steps 8..11 read the second plane.
    """
    mat = lane // Int32(8)
    key_row = key_base + (mat % Int32(2)) * Int32(8) + lane % Int32(8)
    row_swizzle = key_row % Int32(8)
    for pair in cutlass.range_constexpr(num_mma_d_qk // 2):
        step = 2 * pair
        plane = k_plane0 if step < 8 else k_plane1
        chunk = Int32(step % 8) + mat // Int32(2)
        r0, r1, r2, r3 = ldmatrix_m8n8x4_b16(
            plane + (key_row * Int32(8) + (chunk ^ row_swizzle)) * Int32(16)
        )
        for half in cutlass.range_constexpr(2):
            b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r0 if half == 0 else r2)
            b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(r1 if half == 0 else r3)
            s_idx = step + half
            d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                frag_S[0, 0, 0],
                frag_S[0, 0, 1],
                frag_S[0, 0, 2],
                frag_S[0, 0, 3],
                frag_S[0, 0, 4],
                frag_S[0, 0, 5],
                frag_S[0, 0, 6],
                frag_S[0, 0, 7],
                q_regs[s_idx, 0, 0],
                q_regs[s_idx, 0, 1],
                q_regs[s_idx, 0, 2],
                q_regs[s_idx, 0, 3],
                b0,
                b1,
                b2,
                b3,
            )
            frag_S[0, 0, 0] = d0
            frag_S[0, 0, 1] = d1
            frag_S[0, 0, 2] = d2
            frag_S[0, 0, 3] = d3
            frag_S[0, 0, 4] = d4
            frag_S[0, 0, 5] = d5
            frag_S[0, 0, 6] = d6
            frag_S[0, 0, 7] = d7


@cute.jit
def _pv_fp8_into_ofrag(o_frag, p_frag, v_plane, lane, key_base, num_mma_d_vo):
    """O += P V over one 16-key sub-tile of 128-byte FP8 V rows.

    SM120's transposed byte ``ldmatrix`` loads a 16-key x 16-column block per
    output step; the lane-to-row order yields the BF16 B fragments of both
    8-column halves after an exact widening.  P stays BF16 and the V dequant
    scale is applied to the FP32 output.
    """
    lane16 = lane % Int32(16)
    lane_row = (
        (lane16 & Int32(0x1))
        | ((lane16 & Int32(0x4)) >> Int32(1))
        | ((lane16 & Int32(0x8)) >> Int32(1))
        | ((lane16 & Int32(0x2)) << Int32(2))
    )
    v_row = key_base + lane_row
    row_swizzle = v_row % Int32(8)
    for mma_d in cutlass.range_constexpr(num_mma_d_vo):
        f0, f1 = ldmatrix_m16n16x1_trans_b8(
            v_plane + (v_row * Int32(8) + (Int32(mma_d) ^ row_swizzle)) * Int32(16)
        )
        b0, b1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f0)
        b2, b3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(f1)
        d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
            o_frag[0, mma_d, 0],
            o_frag[0, mma_d, 1],
            o_frag[0, mma_d, 2],
            o_frag[0, mma_d, 3],
            o_frag[0, mma_d, 4],
            o_frag[0, mma_d, 5],
            o_frag[0, mma_d, 6],
            o_frag[0, mma_d, 7],
            p_frag[0, 0, 0],
            p_frag[0, 0, 1],
            p_frag[0, 0, 2],
            p_frag[0, 0, 3],
            b0,
            b1,
            b2,
            b3,
        )
        o_frag[0, mma_d, 0] = d0
        o_frag[0, mma_d, 1] = d1
        o_frag[0, mma_d, 2] = d2
        o_frag[0, mma_d, 3] = d3
        o_frag[0, mma_d, 4] = d4
        o_frag[0, mma_d, 5] = d5
        o_frag[0, mma_d, 6] = d6
        o_frag[0, mma_d, 7] = d7


class PagedDecodeKernel:
    """Warp-specialized split-KV decode/verify attention over paged K/V (see module doc)."""

    def __init__(
        self,
        *,
        group_size: int,
        max_q_per_req: int = 8,
        window_left: int = -1,
        has_sinks: bool = False,
        split_kv: bool = True,
        stage_rows: int = 16,
        num_stages: int = 9,
        num_consumer_warps: int = 8,
        min_tiles_per_split: int = 4,
        head_dim_qk: int = 192,
        head_dim_vo: int = 128,
        page_size: int = 64,
        head_splits: int = 1,
        causal: bool = True,
        kv_fp8: bool = False,
        descale_layout: str = "tensor",
    ):
        if (head_dim_qk, head_dim_vo) not in ((192, 128), (128, 128)):
            raise ValueError("decode kernel supports QK192/V128 and QK128/V128")
        if stage_rows not in (16, 32) or page_size % stage_rows != 0:
            raise ValueError(f"unsupported stage rows {stage_rows}")
        # The grid's z index also selects one of ``head_splits`` slices of the
        # KV head's query heads (fewer packed rows per CTA, more CTAs); with
        # split-KV, z = split * head_splits + slice.
        if head_splits < 1 or group_size % head_splits != 0:
            raise ValueError(f"unsupported head split {head_splits}")
        # Rows are packed (query token, head) pairs; a row group is 16 rows and
        # a partial last group is masked at run time.
        rows_cap = (group_size // head_splits) * max_q_per_req
        if (rows_cap + 15) // 16 > num_consumer_warps:
            raise ValueError(f"unsupported row capacity {rows_cap}")
        self.head_splits = int(head_splits)
        self.group_size = int(group_size)
        self.max_q_per_req = int(max_q_per_req)
        self.window_left = int(window_left)
        self.has_sinks = bool(has_sinks)
        self.split_kv = bool(split_kv)
        # Split partials exclude the sink logit; the merge adds it once.
        self.sinks_in_kernel = self.has_sinks and not self.split_kv
        self.causal = bool(causal)
        self.kv_fp8 = bool(kv_fp8)
        if descale_layout not in ("tensor", "request", "head"):
            raise ValueError(f"unsupported descale layout {descale_layout!r}")
        self.descale_layout = descale_layout
        self.stage_rows = int(stage_rows)
        self.num_stages = int(num_stages)
        self.ncw = int(num_consumer_warps)
        self.num_warps = self.ncw + 1
        self.min_tiles_per_split = int(min_tiles_per_split)
        self.head_dim_qk = int(head_dim_qk)
        self.head_dim_vo = int(head_dim_vo)
        self.page_size = int(page_size)
        self.num_mma_d_qk = self.head_dim_qk // 16
        self.num_mma_d_vo = self.head_dim_vo // 16
        # A TMA plane row is 128 bytes: 64 BF16 or 128 FP8 columns.  FP8 K
        # takes one plane per 128 columns (a 192-wide head's second plane is
        # half used; TMA zero-fills the columns past the head) and FP8 V one.
        self.num_k_planes = (
            (self.head_dim_qk + 127) // 128
            if self.kv_fp8
            else self.head_dim_qk // PLANE_DIM
        )
        self.num_v_planes = 1 if self.kv_fp8 else self.head_dim_vo // PLANE_DIM
        self.kv_plane_stage_bytes = self.stage_rows * 128
        self.kv_plane_total_bytes = self.num_stages * self.kv_plane_stage_bytes
        self.k_bytes = self.num_k_planes * self.kv_plane_total_bytes
        self.v_bytes = self.num_v_planes * self.kv_plane_total_bytes
        self.kv_copy_bytes_k = self.num_k_planes * self.kv_plane_stage_bytes
        self.kv_copy_bytes_v = self.num_v_planes * self.kv_plane_stage_bytes
        # The key-group combine reuses the payload after the stage loop:
        # o fragments + (m, d).
        self.comb_o_floats = self.ncw * self.num_mma_d_vo * 8 * 32
        self.comb_md_floats = self.ncw * 2 * 32
        comb_bytes = (self.comb_o_floats + 2 * self.comb_md_floats) * 4
        self.shared_storage_bytes = max(self.k_bytes + self.v_bytes, comb_bytes)
        if self.shared_storage_bytes + 2048 > SMEM_LIMIT:
            raise ValueError(
                f"shared storage {self.shared_storage_bytes} exceeds SM120 limit"
            )

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "full_K": cute.struct.MemRange[cutlass.Int64, self.num_stages],
            "full_V": cute.struct.MemRange[cutlass.Int64, self.num_stages],
            "empty": cute.struct.MemRange[cutlass.Int64, self.num_stages],
            "payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, int(self.shared_storage_bytes)],
                1024,
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mVCache: cute.Tensor,
        mPageTable: cute.Tensor,
        mSeqLens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mOPart: cute.Tensor,
        mLsePart: cute.Tensor,
        mSinks: cute.Tensor,
        mKDescale: cute.Tensor,
        mVDescale: cute.Tensor,
        sm_scale: Float32,
        stream: cuda.CUstream,
    ):
        """FP8 caches arrive as uint8 views with fp32 dequant scales: one value
        (``[1]``), per request (``[B]``) or per request and KV head
        (``[B, Hkv]``); the K scale joins the softmax scale and the V scale
        multiplies the output.  BF16 caches ignore both scale tensors."""
        mQ = _assume_tensor_aligned(mQ)
        mKCache = _assume_tensor_aligned(mKCache)
        mVCache = _assume_tensor_aligned(mVCache)
        mKCacheT = _make_paged_kv_tma_source_tensor(mKCache, self.stage_rows)
        mVCacheT = _make_paged_kv_tma_source_tensor(mVCache, self.stage_rows)
        # A TMA plane is 128 bytes per row: 64 BF16 or 128 FP8 columns.
        tma_cols = 2 * PLANE_DIM if const_expr(self.kv_fp8) else PLANE_DIM
        plane_layout = _paged_kv_tma_plane_layout(self.stage_rows, tma_cols)
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            tma_op, mKCacheT, plane_layout, (self.stage_rows, tma_cols), 1
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            tma_op, mVCacheT, plane_layout, (self.stage_rows, tma_cols), 1
        )
        k_tiles_per_entry = mKCache.stride[0] // (self.stage_rows * mKCache.stride[1])
        v_tiles_per_entry = mVCache.stride[0] // (self.stage_rows * mVCache.stride[1])
        num_splits = (
            mLsePart.shape[2] * self.head_splits
            if const_expr(self.split_kv)
            else self.head_splits
        )
        self.kernel(
            mQ,
            tma_tensor_K,
            tma_tensor_V,
            mPageTable,
            mSeqLens,
            mCuSeqlensQ,
            mO,
            mOPart,
            mLsePart,
            mSinks,
            mKDescale,
            mVDescale,
            tma_atom_K,
            tma_atom_V,
            k_tiles_per_entry,
            v_tiles_per_entry,
            Float32(sm_scale * LOG2_E),
            Float32(1.0 / sm_scale),
        ).launch(
            grid=(mSeqLens.shape[0], mKCache.shape[2], num_splits),
            block=[32, self.num_warps, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCacheT: cute.Tensor,
        mVCacheT: cute.Tensor,
        mPageTable: cute.Tensor,
        mSeqLens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mOPart: cute.Tensor,
        mLsePart: cute.Tensor,
        mSinks: cute.Tensor,
        mKDescale: cute.Tensor,
        mVDescale: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        k_tiles_per_entry: Int32,
        v_tiles_per_entry: Int32,
        sm_scale_log2_base: Float32,
        inv_sm_scale_base: Float32,
    ):
        STAGE = self.stage_rows
        S = self.num_stages
        NCW = self.ncw
        lane, warp, _ = cute.arch.thread_idx()
        request_idx, kv_head_idx, split_idx = cute.arch.block_idx()
        head_slice = split_idx
        num_splits = 1
        if const_expr(self.split_kv):
            _, _, num_splits = cute.arch.grid_dim()
            if const_expr(self.head_splits > 1):
                num_splits = num_splits // Int32(self.head_splits)
                head_slice = split_idx % Int32(self.head_splits)
                split_idx = split_idx // Int32(self.head_splits)

        q_start = mCuSeqlensQ[request_idx]
        qo_len = mCuSeqlensQ[request_idx + 1] - q_start
        seq_len = mSeqLens[request_idx]
        if qo_len <= Int32(0) or seq_len <= Int32(0):
            _exit_thread()
        context_len = seq_len - qo_len
        # The m/d states hold raw q.k products; FP8 K dequant scales them into
        # logits, V dequant scales the finished output.
        sm_scale_log2 = sm_scale_log2_base
        inv_sm_scale = inv_sm_scale_base
        v_scale = Float32(1.0)
        if const_expr(self.kv_fp8):
            k_descale = Float32(0.0)
            if const_expr(self.descale_layout == "head"):
                k_descale = mKDescale[request_idx, kv_head_idx]
                v_scale = mVDescale[request_idx, kv_head_idx]
            elif const_expr(self.descale_layout == "request"):
                k_descale = mKDescale[request_idx]
                v_scale = mVDescale[request_idx]
            else:
                k_descale = mKDescale[0]
                v_scale = mVDescale[0]
            sm_scale_log2 = sm_scale_log2_base * k_descale
            inv_sm_scale = inv_sm_scale_base / k_descale
        group_size = Int32(self.group_size)
        sub_group = Int32(self.group_size // self.head_splits)
        head_base = kv_head_idx * group_size
        if const_expr(self.head_splits > 1):
            head_base = head_base + head_slice * sub_group
        packed_rows = qo_len * sub_group

        first_key = Int32(0)
        if const_expr(self.window_left >= 0):
            first_key = cutlass.select_(
                context_len - Int32(self.window_left) > Int32(0),
                context_len - Int32(self.window_left),
                Int32(0),
            )
        tile_begin = first_key // STAGE
        tile_end = (seq_len + (STAGE - 1)) // STAGE
        my_tile_lo = tile_begin
        my_tile_hi = tile_end
        if const_expr(self.split_kv):
            num_tiles = tile_end - tile_begin
            tiles_per_split = (num_tiles + num_splits - 1) // num_splits
            tiles_per_split = cutlass.select_(
                tiles_per_split < Int32(self.min_tiles_per_split),
                Int32(self.min_tiles_per_split),
                tiles_per_split,
            )
            my_tile_lo = tile_begin + split_idx * tiles_per_split
            my_tile_hi = cutlass.select_(
                my_tile_lo + tiles_per_split < tile_end,
                my_tile_lo + tiles_per_split,
                tile_end,
            )
            if my_tile_lo >= tile_end:
                _exit_thread()
        chunk_start = my_tile_lo * STAGE
        chunk_end = cutlass.select_(
            my_tile_hi * STAGE < seq_len, my_tile_hi * STAGE, seq_len
        )
        ntiles = my_tile_hi - my_tile_lo
        num_row_groups = (packed_rows + 15) // 16
        num_key_groups = Int32(NCW) // num_row_groups
        if num_key_groups > Int32(S):
            num_key_groups = Int32(S)
        stages_per_group = Int32(S) // num_key_groups
        tidx = lane + warp * Int32(32)

        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        full_K = storage.full_K.data_ptr()
        full_V = storage.full_V.data_ptr()
        empty = storage.empty.data_ptr()
        if tidx < Int32(S):
            cute.arch.mbarrier_init(full_K + tidx, Int32(1))
            cute.arch.mbarrier_init(full_V + tidx, Int32(1))
            cute.arch.mbarrier_init(empty + tidx, num_row_groups)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((self.shared_storage_bytes,), stride=(1,))
        )
        sKStageBytes = cute.make_tensor(
            payload_u8.iterator, cute.make_layout((self.k_bytes,), stride=(1,))
        )
        sVStageBytes = cute.make_tensor(
            payload_u8.iterator + Int32(self.k_bytes),
            cute.make_layout((self.v_bytes,), stride=(1,)),
        )

        is_producer = warp == Int32(NCW)
        is_consumer = warp < num_row_groups * num_key_groups
        row_group = warp % num_row_groups
        key_group = warp // num_row_groups

        o_frag = cute.make_rmem_tensor(
            cute.make_layout(
                (1, self.num_mma_d_vo, 8), stride=(self.num_mma_d_vo * 8, 8, 1)
            ),
            Float32,
        )
        m_frag = cute.make_rmem_tensor(cute.make_layout((1, 2), stride=(2, 1)), Float32)
        d_frag = cute.make_rmem_tensor(cute.make_layout((1, 2), stride=(2, 1)), Float32)
        row_valid = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        q_token = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        q_head = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        lane_group = lane // 4
        lane_quad = lane % 4
        lane_pair_base = Int32(2 * lane_quad)
        warp_row_base = row_group * Int32(16)

        if is_producer:
            # 128-byte TMA planes: BF16 caches use 64-column planes (2 or 3 for
            # K, 2 for V); FP8 caches one 128-column plane each.
            tma_cols = PLANE_DIM
            plane_dtype = cutlass.BFloat16
            if const_expr(self.kv_fp8):
                tma_cols = 2 * PLANE_DIM
                plane_dtype = cutlass.Uint8
            plane_stage_layout = _paged_kv_tma_plane_stage_layout(STAGE, tma_cols, S)
            plane_elems = S * STAGE * tma_cols
            mKHead = mKCacheT[None, None, kv_head_idx, None]
            mVHead = mVCacheT[None, None, kv_head_idx, None]
            sK0 = _get_memrange_tensor(
                _make_payload_memrange(payload_u8, plane_dtype, 0, plane_elems),
                plane_stage_layout,
            )
            sV0 = _get_memrange_tensor(
                _make_payload_memrange(
                    payload_u8, plane_dtype, self.k_bytes, plane_elems
                ),
                plane_stage_layout,
            )
            gK0 = cute.local_tile(mKHead, (STAGE, tma_cols), (0, 0, None))
            gV0 = cute.local_tile(mVHead, (STAGE, tma_cols), (0, 0, None))
            load_K0, _, _ = cute_copy.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), gK0, sK0
            )
            load_V0, _, _ = cute_copy.tma_get_copy_fn(
                tma_atom_V, 0, cute.make_layout(1), gV0, sV0
            )
            load_K1 = load_K0
            load_K2 = load_K0
            load_V1 = load_V0
            if const_expr(self.kv_fp8 and self.num_k_planes == 2):
                sK1 = _get_memrange_tensor(
                    _make_payload_memrange(
                        payload_u8,
                        cutlass.Uint8,
                        self.kv_plane_total_bytes,
                        plane_elems,
                    ),
                    plane_stage_layout,
                )
                gK1 = cute.local_tile(mKHead, (STAGE, tma_cols), (0, 1, None))
                load_K1, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK1, sK1
                )
            if const_expr(not self.kv_fp8):
                sK1 = _get_memrange_tensor(
                    _make_payload_memrange(
                        payload_u8,
                        cutlass.BFloat16,
                        self.kv_plane_total_bytes,
                        plane_elems,
                    ),
                    plane_stage_layout,
                )
                sV1 = _get_memrange_tensor(
                    _make_payload_memrange(
                        payload_u8,
                        cutlass.BFloat16,
                        self.k_bytes + self.kv_plane_total_bytes,
                        plane_elems,
                    ),
                    plane_stage_layout,
                )
                gK1 = cute.local_tile(mKHead, (STAGE, PLANE_DIM), (0, 1, None))
                gV1 = cute.local_tile(mVHead, (STAGE, PLANE_DIM), (0, 1, None))
                load_K1, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK1, sK1
                )
                load_V1, _, _ = cute_copy.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV1, sV1
                )
                if const_expr(self.num_k_planes == 3):
                    sK2 = _get_memrange_tensor(
                        _make_payload_memrange(
                            payload_u8,
                            cutlass.BFloat16,
                            2 * self.kv_plane_total_bytes,
                            plane_elems,
                        ),
                        plane_stage_layout,
                    )
                    gK2 = cute.local_tile(mKHead, (STAGE, PLANE_DIM), (0, 2, None))
                    load_K2, _, _ = cute_copy.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK2, sK2
                    )
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            # Fetch page ids 32 at a time, one per lane, so the TMA issue loop
            # does not serialize a dependent global load per tile.
            page_table_width = mPageTable.shape[1]
            pt_base = chunk_start // Int32(self.page_size)
            pt_idx = pt_base + lane
            pid_lane = Int32(0)
            if pt_idx < page_table_width:
                pid_lane = mPageTable[request_idx, pt_idx]
            t = Int32(0)
            while t < ntiles:
                # Buffers are partitioned per key group so each buffer is only
                # ever consumed by one group, in order; parity waits cannot
                # then confuse phase k with phase k - 2.
                kg_t = t % num_key_groups
                j = t // num_key_groups
                buf = kg_t * stages_per_group + j % stages_per_group
                use = j // stages_per_group
                if use > Int32(0):
                    cute.arch.mbarrier_wait(
                        empty + buf, phase=(use - Int32(1)) & Int32(1)
                    )
                token_base = chunk_start + t * Int32(STAGE)
                page_idx = token_base // Int32(self.page_size)
                if page_idx - pt_base >= Int32(32):
                    pt_base = pt_base + Int32(32)
                    pt_idx = pt_base + lane
                    pid_lane = Int32(0)
                    if pt_idx < page_table_width:
                        pid_lane = mPageTable[request_idx, pt_idx]
                page_id = cute.arch.shuffle_sync(pid_lane, page_idx - pt_base)
                _issue_kv_stage_at(
                    load_K0,
                    load_K1,
                    load_K2,
                    self.num_k_planes,
                    STAGE,
                    buf,
                    full_K + buf,
                    self.kv_copy_bytes_k,
                    page_id,
                    token_base,
                    Int32(self.page_size),
                    k_tiles_per_entry,
                )
                _issue_kv_stage_at(
                    load_V0,
                    load_V1,
                    load_V1,
                    self.num_v_planes,
                    STAGE,
                    buf,
                    full_V + buf,
                    self.kv_copy_bytes_v,
                    page_id,
                    token_base,
                    Int32(self.page_size),
                    v_tiles_per_entry,
                )
                t += Int32(1)
        elif is_consumer:
            causal_hi = cute.make_rmem_tensor(
                cute.make_layout((2,), stride=(1,)), Int32
            )
            window_lo = cute.make_rmem_tensor(
                cute.make_layout((2,), stride=(1,)), Int32
            )
            for row_slot in cutlass.range_constexpr(2):
                packed_row = warp_row_base + lane_group + 8 * row_slot
                valid_row = packed_row < packed_rows
                row_valid[row_slot] = Int32(valid_row)
                token_local = packed_row // sub_group
                q_token[row_slot] = token_local
                q_head[row_slot] = head_base + (packed_row - token_local * sub_group)
                # Last visible key: the row itself (causal) or the sequence end
                # (non-causal); a window bounds keys on the left only.
                row_hi = context_len + token_local
                if const_expr(not self.causal):
                    row_hi = seq_len - Int32(1)
                causal_hi[row_slot] = cutlass.select_(valid_row, row_hi, Int32(-1))
                window_lo[row_slot] = (
                    context_len + token_local - Int32(self.window_left)
                )

            q_regs = cute.make_rmem_tensor(
                cute.make_layout((self.num_mma_d_qk, 1, 4), stride=(4, 4, 1)), Uint32
            )
            mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
            q_token_stride_bytes = Int64(mQ.stride[0]) * Int64(2)
            q_head_stride_bytes = Int64(mQ.stride[1]) * Int64(2)
            for row_slot in cutlass.range_constexpr(2):
                row_byte = (
                    Int64(q_start + q_token[row_slot]) * q_token_stride_bytes
                    + Int64(q_head[row_slot]) * q_head_stride_bytes
                )
                for mma_d in cutlass.range_constexpr(self.num_mma_d_qk):
                    for half in cutlass.range_constexpr(2):
                        # FP8 K fragments hold columns 4t..4t+3 of each
                        # 16-column step (see _qk_fp8_into_sfrag); Q follows.
                        col = mma_d * 16 + half * 8 + 2 * lane_quad
                        if const_expr(self.kv_fp8):
                            col = mma_d * 16 + 4 * lane_quad + 2 * half
                        value = Uint32(0)
                        if row_valid[row_slot] != Int32(0):
                            value = ld_global_nc_u32(
                                get_ptr_as_int64(mQBytes, row_byte + Int64(col * 2))
                            )
                        q_regs[mma_d, 0, row_slot + 2 * half] = value

            for mma_d in cutlass.range_constexpr(self.num_mma_d_vo):
                for reg_id in cutlass.range_constexpr(8):
                    o_frag[0, mma_d, reg_id] = Float32(0.0)
            for row_slot in cutlass.range_constexpr(2):
                m_frag[0, row_slot] = Float32(-Float32.inf)
                d_frag[0, row_slot] = Float32(1.0)
                if const_expr(self.sinks_in_kernel):
                    # The sink logit is counted once, by key group 0.
                    if row_valid[row_slot] != Int32(0) and key_group == Int32(0):
                        m_frag[0, row_slot] = Float32(
                            mSinks[q_head[row_slot]] * inv_sm_scale
                        )

            frag_s_layout = cute.make_layout((1, 1, 8), stride=(8, 8, 1))
            p_frag = cute.make_rmem_tensor(
                cute.make_layout((1, 1, 4), stride=(4, 4, 1)), Uint32
            )
            # A tile needs no per-element mask when every valid row sees all
            # of it: row 0 bounds the causal limit, the last row the window.
            causal_hi_min = context_len
            if const_expr(not self.causal):
                causal_hi_min = seq_len - Int32(1)
            window_lo_max = context_len + qo_len - Int32(1) - Int32(self.window_left)
            t = key_group
            j = Int32(0)
            while t < ntiles:
                buf = key_group * stages_per_group + j % stages_per_group
                use = j // stages_per_group
                cute.arch.mbarrier_wait(full_K + buf, phase=use & Int32(1))
                cute.arch.mbarrier_wait(full_V + buf, phase=use & Int32(1))
                tile_base = chunk_start + t * Int32(STAGE)
                tile_tokens = cutlass.select_(
                    tile_base + STAGE < chunk_end, Int32(STAGE), chunk_end - tile_base
                )
                stage_off = buf * Int32(self.kv_plane_stage_bytes)
                k0 = shared_ptr_to_u32(sKStageBytes.iterator + stage_off)
                k1 = shared_ptr_to_u32(
                    sKStageBytes.iterator + stage_off + Int32(self.kv_plane_total_bytes)
                )
                k2 = shared_ptr_to_u32(
                    sKStageBytes.iterator
                    + stage_off
                    + Int32(2 * self.kv_plane_total_bytes)
                )
                v0 = shared_ptr_to_u32(sVStageBytes.iterator + stage_off)
                v1 = shared_ptr_to_u32(
                    sVStageBytes.iterator + stage_off + Int32(self.kv_plane_total_bytes)
                )
                if const_expr(not self.kv_fp8 and self.num_k_planes == 2):
                    k2 = k1
                for sub in cutlass.range_constexpr(STAGE // 16):
                    if Int32(sub * 16) < tile_tokens:
                        frag_S = cute.make_rmem_tensor(frag_s_layout, Float32)
                        frag_S.fill(0.0)
                        if const_expr(self.kv_fp8):
                            _qk_fp8_into_sfrag(
                                frag_S,
                                q_regs,
                                k0,
                                k1,
                                lane,
                                Int32(sub * 16),
                                self.num_mma_d_qk,
                            )
                        else:
                            _literal_qk_mma_into_sfrag_plane_bf16(
                                frag_S,
                                Int32(0),
                                k0,
                                k1,
                                k2,
                                k2,
                                lane,
                                Int32(0),
                                Int32(0),
                                Int32(sub * 16),
                                1,
                                1,
                                self.num_mma_d_qk,
                                Int32(self.head_dim_qk // 8),
                                Int32(PLANE_DIM // 8),
                                q_regs=q_regs,
                                q_in_regs=True,
                            )
                        sub_base = tile_base + Int32(sub * 16)
                        interior = (tile_tokens >= Int32(sub * 16 + 16)) and (
                            sub_base + Int32(15) <= causal_hi_min
                        )
                        if const_expr(self.window_left >= 0):
                            interior = interior and sub_base >= window_lo_max
                        if not interior:
                            _mask_subtile(
                                frag_S,
                                Int32(sub * 16),
                                tile_base,
                                tile_tokens,
                                lane_pair_base,
                                causal_hi,
                                window_lo,
                                self.window_left >= 0,
                            )
                        _literal_update_mdo_states_fp32_pack_p(
                            frag_S,
                            o_frag,
                            m_frag,
                            d_frag,
                            p_frag,
                            sm_scale_log2,
                            1,
                            1,
                            self.num_mma_d_vo,
                        )
                        d0, d1 = bf16_rowsum_m16k16_f32(
                            d_frag[0, 0],
                            d_frag[0, 1],
                            p_frag[0, 0, 0],
                            p_frag[0, 0, 1],
                            p_frag[0, 0, 2],
                            p_frag[0, 0, 3],
                        )
                        d_frag[0, 0] = d0
                        d_frag[0, 1] = d1
                        if const_expr(self.kv_fp8):
                            _pv_fp8_into_ofrag(
                                o_frag,
                                p_frag,
                                v0,
                                lane,
                                Int32(sub * 16),
                                self.num_mma_d_vo,
                            )
                        else:
                            _literal_pv_mma_into_ofrag_plane_bf16_packed(
                                o_frag,
                                p_frag,
                                v0,
                                v1,
                                v1,
                                v1,
                                lane,
                                Int32(0),
                                Int32(sub * 16),
                                1,
                                1,
                                self.num_mma_d_vo,
                                Int32(PLANE_DIM // 8),
                                Float32(1.0),
                            )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(empty + buf)
                t += num_key_groups
                j += Int32(1)

        # Every warp reaches here; all TMA writes have been consumed, so the
        # stage buffers can hold the key-group combine scratch.
        cute.arch.sync_threads()
        sComb = cute.make_tensor(
            cute.recast_ptr(payload_u8.iterator.align(16), dtype=Float32),
            cute.make_layout(
                (self.comb_o_floats + 2 * self.comb_md_floats,), stride=(1,)
            ),
        )
        o_words = self.num_mma_d_vo * 8 * 32
        md_base = self.comb_o_floats
        if is_consumer:
            for mma_d in cutlass.range_constexpr(self.num_mma_d_vo):
                for reg_id in cutlass.range_constexpr(8):
                    sComb[warp * o_words + (mma_d * 8 + reg_id) * 32 + lane] = o_frag[
                        0, mma_d, reg_id
                    ]
            for row_slot in cutlass.range_constexpr(2):
                sComb[md_base + (warp * 2 + row_slot) * 32 + lane] = m_frag[0, row_slot]
                sComb[
                    md_base + self.comb_md_floats + (warp * 2 + row_slot) * 32 + lane
                ] = d_frag[0, row_slot]
        cute.arch.sync_threads()

        scales = cute.make_rmem_tensor(cute.make_layout((NCW,), stride=(1,)), Float32)
        if is_consumer:
            mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
            for row_slot in cutlass.range_constexpr(2):
                if row_valid[row_slot] != Int32(0):
                    # Every warp of the row group derives the same (m*, d*) and
                    # then finalizes only its share of the value columns.
                    m_star = Float32(-Float32.inf)
                    for k in cutlass.range_constexpr(NCW):
                        if Int32(k) < num_key_groups:
                            w = row_group + Int32(k) * num_row_groups
                            m_star = attention_ops.fmax(
                                m_star, sComb[md_base + (w * 2 + row_slot) * 32 + lane]
                            )
                    has_mass = m_star != -Float32.inf
                    d_star = Float32(0.0)
                    for k in cutlass.range_constexpr(NCW):
                        scales[k] = Float32(0.0)
                        if Int32(k) < num_key_groups and has_mass:
                            w = row_group + Int32(k) * num_row_groups
                            m_k = sComb[md_base + (w * 2 + row_slot) * 32 + lane]
                            if m_k != -Float32.inf:
                                scales[k] = cute.math.exp2(
                                    (m_k - m_star) * sm_scale_log2, fastmath=True
                                )
                            d_star += (
                                scales[k]
                                * sComb[
                                    md_base
                                    + self.comb_md_floats
                                    + (w * 2 + row_slot) * 32
                                    + lane
                                ]
                            )
                    inv_d = Float32(0.0)
                    if has_mass:
                        inv_d = cute.arch.rcp_approx(d_star)
                    if const_expr(self.kv_fp8):
                        inv_d = inv_d * v_scale
                    q_row = q_start + q_token[row_slot]
                    head = q_head[row_slot]
                    part_base = Int64(0)
                    if const_expr(self.split_kv):
                        part_base = (
                            Int64(q_row) * Int64(mOPart.stride[0])
                            + Int64(head) * Int64(mOPart.stride[1])
                            + Int64(split_idx) * Int64(mOPart.stride[2])
                        ) * Int64(4)
                    for mma_d in cutlass.range_constexpr(self.num_mma_d_vo):
                        if Int32(mma_d) % num_key_groups == key_group:
                            a0 = Float32(0.0)
                            a1 = Float32(0.0)
                            a2 = Float32(0.0)
                            a3 = Float32(0.0)
                            for k in cutlass.range_constexpr(NCW):
                                if Int32(k) < num_key_groups:
                                    w = row_group + Int32(k) * num_row_groups
                                    base = w * o_words + (mma_d * 8) * 32 + lane
                                    a0 += (
                                        scales[k]
                                        * sComb[base + (row_slot * 2 + 0) * 32]
                                    )
                                    a1 += (
                                        scales[k]
                                        * sComb[base + (row_slot * 2 + 1) * 32]
                                    )
                                    a2 += (
                                        scales[k]
                                        * sComb[base + (row_slot * 2 + 4) * 32]
                                    )
                                    a3 += (
                                        scales[k]
                                        * sComb[base + (row_slot * 2 + 5) * 32]
                                    )
                            dim_low = mma_d * 16 + lane_pair_base
                            o0 = a0 * inv_d
                            o1 = a1 * inv_d
                            o2 = a2 * inv_d
                            o3 = a3 * inv_d
                            if const_expr(self.split_kv):
                                st_global_v2_f32(
                                    get_ptr_as_int64(
                                        mOPartBytes, part_base + Int64(dim_low * 4)
                                    ),
                                    o0,
                                    o1,
                                )
                                st_global_v2_f32(
                                    get_ptr_as_int64(
                                        mOPartBytes,
                                        part_base + Int64((dim_low + 8) * 4),
                                    ),
                                    o2,
                                    o3,
                                )
                            else:
                                mO[q_row, head, dim_low + 0] = o0.to(cutlass.BFloat16)
                                mO[q_row, head, dim_low + 1] = o1.to(cutlass.BFloat16)
                                mO[q_row, head, dim_low + 8] = o2.to(cutlass.BFloat16)
                                mO[q_row, head, dim_low + 9] = o3.to(cutlass.BFloat16)
                    if const_expr(self.split_kv):
                        if lane_quad == Int32(0) and key_group == Int32(0):
                            lse = Float32(-Float32.inf)
                            if has_mass:
                                lse = Float32(
                                    m_star * sm_scale_log2
                                    + cute.math.log2(d_star, fastmath=True)
                                )
                            mLsePart[q_row, head, split_idx] = lse


class PagedDecodeMergeKernel:
    """Combine fp32 split partials into the BF16 attention output.

    One CTA owns ``heads_per_cta`` (token, head) rows with ``warps_per_row``
    warps each.  Every warp scans all split LSEs (lanes stride the splits) to
    get the row max and normalizer, then accumulates its share of the splits
    for four value columns per lane; warps combine through shared memory.
    """

    def __init__(
        self,
        *,
        num_q_heads: int = 16,
        max_q_per_req: int = 8,
        min_tiles_per_split: int = 4,
        stage_rows: int = 32,
        head_dim_vo: int = 128,
        window_left: int = -1,
        heads_per_cta: int = 4,
        warps_per_row: int = 4,
        has_sinks: bool = False,
    ):
        if head_dim_vo != 128:
            raise ValueError("merge kernel is specialized for V128")
        if num_q_heads % heads_per_cta != 0:
            raise ValueError("heads_per_cta must divide the head count")
        self.num_q_heads = int(num_q_heads)
        self.max_q_per_req = int(max_q_per_req)
        self.min_tiles_per_split = int(min_tiles_per_split)
        self.stage_rows = int(stage_rows)
        self.head_dim_vo = int(head_dim_vo)
        self.window_left = int(window_left)
        self.heads_per_cta = int(heads_per_cta)
        self.warps_per_row = int(warps_per_row)
        # Split partials exclude the attention sink; it joins the normalizer here.
        self.has_sinks = bool(has_sinks)

    def _get_shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "acc": cute.struct.MemRange[
                Float32, self.heads_per_cta * self.warps_per_row * self.head_dim_vo
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mOPart: cute.Tensor,  # [T, Hq, S, 128] fp32
        mLsePart: cute.Tensor,  # [T, Hq, S] fp32
        mSeqLens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,  # [T, Hq, 128] bf16
        mSinks: cute.Tensor,  # [Hq] fp32 sink logits (read only with has_sinks)
        stream: cuda.CUstream,
    ):
        self.kernel(mOPart, mLsePart, mSeqLens, mCuSeqlensQ, mO, mSinks).launch(
            grid=(
                mSeqLens.shape[0],
                self.max_q_per_req,
                self.num_q_heads // self.heads_per_cta,
            ),
            block=[32, self.heads_per_cta * self.warps_per_row, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOPart: cute.Tensor,
        mLsePart: cute.Tensor,
        mSeqLens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mSinks: cute.Tensor,
    ):
        STAGE = self.stage_rows
        WPR = self.warps_per_row
        lane, warp_y, _ = cute.arch.thread_idx()
        request_idx, token_local, head_group = cute.arch.block_idx()
        head_local = warp_y // WPR
        warp_in_row = warp_y - head_local * WPR
        head = head_group * self.heads_per_cta + head_local
        q_start = mCuSeqlensQ[request_idx]
        qo_len = mCuSeqlensQ[request_idx + 1] - q_start
        seq_len = mSeqLens[request_idx]
        if token_local >= qo_len or seq_len <= Int32(0):
            _exit_thread()
        num_splits = mLsePart.shape[2]
        context_len = seq_len - qo_len
        first_key = Int32(0)
        if const_expr(self.window_left >= 0):
            first_key = cutlass.select_(
                context_len - Int32(self.window_left) > Int32(0),
                context_len - Int32(self.window_left),
                Int32(0),
            )
        tile_begin = first_key // STAGE
        tile_end = (seq_len + (STAGE - 1)) // STAGE
        num_tiles = tile_end - tile_begin
        tiles_per_split = (num_tiles + num_splits - 1) // num_splits
        tiles_per_split = cutlass.select_(
            tiles_per_split < Int32(self.min_tiles_per_split),
            Int32(self.min_tiles_per_split),
            tiles_per_split,
        )
        active = (num_tiles + tiles_per_split - 1) // tiles_per_split
        q_row = q_start + token_local

        # Row max and normalizer over all active splits (lanes stride splits).
        m_local = Float32(-Float32.inf)
        sp = lane
        while sp < active:
            m_local = attention_ops.fmax(m_local, mLsePart[q_row, head, sp])
            sp += Int32(32)
        m_row = cute.arch.warp_reduction_max(m_local)
        sink_log2 = Float32(-Float32.inf)
        if const_expr(self.has_sinks):
            sink_log2 = Float32(mSinks[head] * LOG2_E)
            m_row = attention_ops.fmax(m_row, sink_log2)
        s_local = Float32(0.0)
        if m_row != -Float32.inf:
            sp = lane
            while sp < active:
                lse = mLsePart[q_row, head, sp]
                if lse != -Float32.inf:
                    s_local += cute.math.exp2(lse - m_row, fastmath=True)
                sp += Int32(32)
        s_row = cute.arch.warp_reduction_sum(s_local)
        if const_expr(self.has_sinks):
            s_row = s_row + cute.math.exp2(sink_log2 - m_row, fastmath=True)

        # Weighted sum of this warp's share of the splits.
        mOPartBytes = cute.flatten(cute.recast_tensor(mOPart, cutlass.Uint8))
        row_base = (
            Int64(q_row) * Int64(mOPart.stride[0])
            + Int64(head) * Int64(mOPart.stride[1])
        ) * Int64(4) + Int64(lane * 16)
        split_stride_bytes = Int64(mOPart.stride[2]) * Int64(4)
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        if m_row != -Float32.inf:
            # Every active split stores finite partials (zero with LSE -inf when
            # it saw no keys), so weights need no branch; four independent
            # vector loads per step keep the latency hidden.
            sp = warp_in_row
            while sp + Int32(3 * WPR) < active:
                w0 = cute.math.exp2(mLsePart[q_row, head, sp] - m_row, fastmath=True)
                w1 = cute.math.exp2(
                    mLsePart[q_row, head, sp + Int32(WPR)] - m_row, fastmath=True
                )
                w2 = cute.math.exp2(
                    mLsePart[q_row, head, sp + Int32(2 * WPR)] - m_row, fastmath=True
                )
                w3 = cute.math.exp2(
                    mLsePart[q_row, head, sp + Int32(3 * WPR)] - m_row, fastmath=True
                )
                x0, x1, x2, x3 = ld_global_v4_f32(
                    get_ptr_as_int64(
                        mOPartBytes, row_base + Int64(sp) * split_stride_bytes
                    )
                )
                y0, y1, y2, y3 = ld_global_v4_f32(
                    get_ptr_as_int64(
                        mOPartBytes,
                        row_base + Int64(sp + Int32(WPR)) * split_stride_bytes,
                    )
                )
                z0, z1, z2, z3 = ld_global_v4_f32(
                    get_ptr_as_int64(
                        mOPartBytes,
                        row_base + Int64(sp + Int32(2 * WPR)) * split_stride_bytes,
                    )
                )
                u0, u1, u2, u3 = ld_global_v4_f32(
                    get_ptr_as_int64(
                        mOPartBytes,
                        row_base + Int64(sp + Int32(3 * WPR)) * split_stride_bytes,
                    )
                )
                acc0 += w0 * x0 + w1 * y0 + w2 * z0 + w3 * u0
                acc1 += w0 * x1 + w1 * y1 + w2 * z1 + w3 * u1
                acc2 += w0 * x2 + w1 * y2 + w2 * z2 + w3 * u2
                acc3 += w0 * x3 + w1 * y3 + w2 * z3 + w3 * u3
                sp += Int32(4 * WPR)
            while sp < active:
                w = cute.math.exp2(mLsePart[q_row, head, sp] - m_row, fastmath=True)
                o0, o1, o2, o3 = ld_global_v4_f32(
                    get_ptr_as_int64(
                        mOPartBytes, row_base + Int64(sp) * split_stride_bytes
                    )
                )
                acc0 += w * o0
                acc1 += w * o1
                acc2 += w * o2
                acc3 += w * o3
                sp += Int32(WPR)

        SharedStorage = self._get_shared_storage_cls()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sAcc = storage.acc.get_tensor(
            cute.make_layout(
                (self.heads_per_cta, WPR, self.head_dim_vo),
                stride=(WPR * self.head_dim_vo, self.head_dim_vo, 1),
            )
        )
        sAcc[head_local, warp_in_row, lane * 4 + 0] = acc0
        sAcc[head_local, warp_in_row, lane * 4 + 1] = acc1
        sAcc[head_local, warp_in_row, lane * 4 + 2] = acc2
        sAcc[head_local, warp_in_row, lane * 4 + 3] = acc3
        cute.arch.sync_threads()
        if warp_in_row == Int32(0):
            inv = Float32(0.0)
            if s_row > Float32(0.0):
                inv = Float32(1.0) / s_row
            for c in cutlass.range_constexpr(4):
                total = Float32(0.0)
                for wr in cutlass.range_constexpr(WPR):
                    total += sAcc[head_local, wr, lane * 4 + c]
                mO[q_row, head, lane * 4 + c] = (total * inv).to(cutlass.BFloat16)


def default_max_splits(
    num_seqs: int, num_kv_heads: int, num_sms: int = 188, cap: int = 64
) -> int:
    """Split budget that fills the GPU in one wave for this batch size.

    A decode CTA's stage ring and registers leave room for one CTA per SM, so
    rounding the split count up (``ceil(188 / 32) = 6`` gives 192 CTAs) sends
    the last few CTAs to a second wave and doubles a long-context call.
    """
    ctas = max(num_seqs * num_kv_heads, 1)
    return int(max(1, min(cap, num_sms // ctas)))
