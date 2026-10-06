"""V4.1 paged DSA specialization with a staged TP reduction boundary.

Unlike the FP8 fused route, the published V4.1 score rounds the dot, weighted
product, and head sum to BF16 (model.py:550-559). Selection must follow the TP
sum. Reuse the exact native row selector, not the MSA page/head-max reduction
or the streaming top-k's unrelated candidate folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ..._lib.compiler import KernelCompileSpec, compile as b12x_compile
from ..._lib.program_cache import program_cache
from ..._lib.intrinsics import (
    cvt_fp32x2_to_e2m1x2,
    f16x2_to_f32x2,
    fabs_f32,
    fmax_f32,
    fp4_decode_2,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_shared_u32,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    shared_ptr_to_u32,
    pow2_ceil_ue8m0,
    u32_as_f32,
    ue8m0_to_output_scale,
)
from ..._lib.runtime_control import raise_if_kernel_resolution_frozen
from ..._lib.scratch import scratch_buffer_spec, scratch_tensor
from ..._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from ..._lib.utils import current_cuda_stream, make_ptr
from .tiled_topk import run_row_topk

MXFP4_INDEX_PAGE_SIZE = 64
MXFP4_INDEX_PAGE_BYTES = 64 * 68


def index_mxfp4_page_bytes(page_size: int = MXFP4_INDEX_PAGE_SIZE) -> int:
    """Page storage: ``[page_size,64]`` data, then ``[page_size,4]`` UE8M0.

    Adjacent E2M1 values occupy low/high nibbles. Every data row is 16-byte
    aligned; scales are not interleaved into a 68-byte token stride.
    """
    if page_size <= 0 or page_size % 8:
        raise ValueError("MXFP4 page_size must be a positive multiple of eight")
    return page_size * 68


def _ptr(tensor, dtype):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=1)


@cute.jit
def _flat(ptr: cute.Pointer):
    # All actual accesses are bounded by runtime extents. The large logical
    # extent avoids narrowing pointer arithmetic for high recycled pool pages.
    return cute.make_tensor(ptr, cute.make_layout((Int64(1) << Int64(40),)))


class _Quantize:
    def __init__(self, paged: bool, page_size: int):
        self.paged = paged
        self.page_size = page_size

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        q: cute.Pointer,
        scales: cute.Pointer,
        slots: cute.Pointer,
        rows: Int32,
        pool_pages: Int64,
        page_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(q),
            _flat(scales),
            _flat(slots),
            rows,
            pool_pages,
            page_stride,
        ).launch(grid=((rows * 4 + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        q: cute.Tensor,
        scales: cute.Tensor,
        slots: cute.Tensor,
        rows: Int32,
        pool_pages: Int64,
        page_stride: Int64,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        group = Int32(bx) * Int32(8) + Int32(tx) // Int32(32)
        row = group // Int32(4)
        lane = Int32(tx) % Int32(32)
        if row < rows:
            v = Float32(x[Int64(group) * Int64(32) + Int64(lane)])
            amax = fabs_f32(v)
            for shift in cutlass.range_constexpr(5):
                amax = fmax_f32(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
                )
            amax = fmax_f32(amax, Float32(6.0 * 2.0**-126))
            _, sf = pow2_ceil_ue8m0(amax * Float32(1.0 / 6.0))
            scaled = v * ue8m0_to_output_scale(sf)
            other = cute.arch.shuffle_sync_bfly(scaled, offset=1)
            packed = cvt_fp32x2_to_e2m1x2(scaled, other)
            data_base = Int64(row) * Int64(64)
            scale_base = Int64(row) * Int64(4)
            valid = True
            if cutlass.const_expr(self.paged):
                slot = Int64(slots[row])
                page = slot // Int64(self.page_size)
                token = slot % Int64(self.page_size)
                valid = slot >= Int64(0) and page < pool_pages
                data_base = page * page_stride + token * Int64(64)
                scale_base = (
                    page * page_stride + Int64(self.page_size * 64) + token * Int64(4)
                )
            if valid:
                if lane % Int32(2) == Int32(0):
                    q[
                        data_base
                        + Int64(group % Int32(4)) * Int64(16)
                        + Int64(lane // Int32(2))
                    ] = Uint8(packed)
                if lane == Int32(0):
                    scales[scale_base + Int64(group % Int32(4))] = Uint8(sf)


class _PagedScore:
    """Paged BF16 tensor-core score with the published MXFP4 rounding stages.

    Each warp scores eight candidates against a sixteen-head tile. Q/K are
    dequantized to BF16 before MMA; dot, weighted product and final head sum
    retain their distinct BF16 boundaries before the caller's TP reduction.
    """

    def __init__(self, heads: int, candidates: bool, page_size: int):
        self.heads = heads
        self.candidates = candidates
        self.page_size = page_size

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        qs: cute.Pointer,
        weights: cute.Pointer,
        pool: cute.Pointer,
        pages: cute.Pointer,
        lengths: cute.Pointer,
        active: cute.Pointer,
        candidates: cute.Pointer,
        candidate_lengths: cute.Pointer,
        scores: cute.Pointer,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(q),
            _flat(qs),
            _flat(weights),
            _flat(pool),
            _flat(pages),
            _flat(lengths),
            _flat(active),
            _flat(candidates),
            _flat(candidate_lengths),
            _flat(scores),
            rows,
            width,
            page_width,
            page_row_stride,
            pool_stride,
            pool_pages,
        ).launch(
            grid=(
                cutlass.min((width - Int32(1)) // Int32(64) + Int32(1), Int32(256)),
                rows,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        candidate_lengths: cute.Tensor,
        scores: cute.Tensor,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, row, _ = cute.arch.block_idx()
        grid_cols, _, _ = cute.arch.grid_dim()
        grid_stride = Int32(grid_cols) * Int32(64)
        lane = Int32(tx) & Int32(31)
        warp = Int32(tx) // Int32(32)
        gid, tid = lane >> Int32(2), lane & Int32(3)
        block_start = Int32(bx) * Int32(64)
        # Clearing and scoring own the same strided 64-column tiles. This
        # bounds idle CTA dispatch without needing an inter-CTA barrier.
        clear_col = block_start + Int32(tx)
        if Int32(tx) < Int32(64):
            while clear_col < width:
                scores[Int64(row) * Int64(width) + Int64(clear_col)] = BFloat16(
                    -float("inf")
                )
                clear_col += cutlass.min(grid_stride, width - clear_col)
        end = width
        if cutlass.const_expr(self.candidates):
            end = cutlass.min(end, cutlass.max(candidate_lengths[row], Int32(0)))
        else:
            end = cutlass.min(
                end, cutlass.max(Int32(0), cutlass.min(lengths[row], active[0]))
            )
        smem = cutlass.utils.SmemAllocator()
        # 136 BF16 elements per row gives conflict-free 8x8 matrix rows and
        # packed K fragments while keeping every ldmatrix address 16B aligned.
        sk = smem.allocate_tensor(
            BFloat16, cute.make_layout((64, 136), stride=(136, 1)), byte_alignment=16
        )
        sq = smem.allocate_tensor(
            BFloat16, cute.make_layout((16, 136), stride=(136, 1)), byte_alignment=16
        )
        valid_keys = smem.allocate_tensor(
            Int32, cute.make_layout((64,)), byte_alignment=16
        )
        k_addr = shared_ptr_to_u32(sk.iterator)
        q_addr = shared_ptr_to_u32(sq.iterator)
        tile_start = block_start
        while tile_start < end:
            for part in cutlass.range_constexpr(16):
                pair_index = Int32(tx) + Int32(part * 256)
                key_row = pair_index // Int32(64)
                byte_col = pair_index % Int32(64)
                col = tile_start + key_row
                pos = col
                valid = col < end
                if cutlass.const_expr(self.candidates):
                    if valid:
                        pos = candidates[Int64(row) * Int64(width) + Int64(col)]
                valid = (
                    valid and pos >= Int32(0) and pos < lengths[row] and pos < active[0]
                )
                valid = valid and pos // Int32(self.page_size) < page_width
                page = Int64(-1)
                if valid:
                    page = Int64(
                        pages[
                            Int64(row) * page_row_stride
                            + Int64(pos // Int32(self.page_size))
                        ]
                    )
                valid = valid and page >= Int64(0) and page < pool_pages
                k0, k1 = Float32(0.0), Float32(0.0)
                if valid:
                    token = Int64(pos % Int32(self.page_size))
                    kb = page * pool_stride + token * Int64(64)
                    sb = (
                        page * pool_stride
                        + Int64(self.page_size * 64)
                        + token * Int64(4)
                    )
                    k0, k1 = f16x2_to_f32x2(
                        fp4_decode_2(Uint32(pool[kb + Int64(byte_col)]))
                    )
                    scale = u32_as_f32(
                        Uint32(pool[sb + Int64(byte_col // Int32(16))]) << Uint32(23)
                    )
                    k0, k1 = k0 * scale, k1 * scale
                sk[key_row, byte_col * Int32(2)] = BFloat16(k0)
                sk[key_row, byte_col * Int32(2) + Int32(1)] = BFloat16(k1)
                if byte_col == Int32(0):
                    valid_keys[key_row] = Int32(valid)

            total0, total1 = Float32(0.0), Float32(0.0)
            for head_group in cutlass.range_constexpr((self.heads + 15) // 16):
                for part in cutlass.range_constexpr(4):
                    pair_index = Int32(tx) + Int32(part * 256)
                    local_head = pair_index // Int32(64)
                    byte_col = pair_index % Int32(64)
                    head = Int32(head_group * 16) + local_head
                    q0, q1 = Float32(0.0), Float32(0.0)
                    if head < Int32(self.heads):
                        qb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(64)
                        qsb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(4)
                        q0, q1 = f16x2_to_f32x2(
                            fp4_decode_2(Uint32(q[qb + Int64(byte_col)]))
                        )
                        scale = u32_as_f32(
                            Uint32(qs[qsb + Int64(byte_col // Int32(16))]) << Uint32(23)
                        )
                        q0, q1 = q0 * scale, q1 * scale
                    sq[local_head, byte_col * Int32(2)] = BFloat16(q0)
                    sq[local_head, byte_col * Int32(2) + Int32(1)] = BFloat16(q1)
                cute.arch.sync_threads()
                d0, d1, d2, d3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
                for step in cutlass.range_constexpr(8):
                    a_offset = (
                        (lane & Int32(15)) * Int32(136)
                        + Int32(step * 16)
                        + (lane >> Int32(4)) * Int32(8)
                    ) * Int32(2)
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(q_addr + a_offset)
                    b_offset = (
                        (warp * Int32(8) + gid) * Int32(136)
                        + Int32(step * 16)
                        + tid * Int32(2)
                    ) * Int32(2)
                    b0 = ld_shared_u32(k_addr + b_offset)
                    b1 = ld_shared_u32(k_addr + b_offset + Int32(16))
                    d0, d1, d2, d3 = mma_m16n8k16_f32_bf16(
                        d0, d1, d2, d3, a0, a1, a2, a3, b0, b1
                    )
                head0 = Int32(head_group * 16) + gid
                head1 = head0 + Int32(8)
                w0, w1 = Float32(0.0), Float32(0.0)
                if head0 < Int32(self.heads):
                    w0 = Float32(weights[Int64(row) * Int64(self.heads) + Int64(head0)])
                if head1 < Int32(self.heads):
                    w1 = Float32(weights[Int64(row) * Int64(self.heads) + Int64(head1)])
                p0 = Float32(
                    BFloat16(fmax_f32(Float32(BFloat16(d0)), Float32(0.0)) * w0)
                )
                p1 = Float32(
                    BFloat16(fmax_f32(Float32(BFloat16(d1)), Float32(0.0)) * w0)
                )
                p2 = Float32(
                    BFloat16(fmax_f32(Float32(BFloat16(d2)), Float32(0.0)) * w1)
                )
                p3 = Float32(
                    BFloat16(fmax_f32(Float32(BFloat16(d3)), Float32(0.0)) * w1)
                )
                # Preserve the FP32 head-sum order, including signed weights
                # and cancellation, rather than introducing a new tree fold.
                for head in cutlass.range_constexpr(
                    min(16, self.heads - head_group * 16)
                ):
                    source_lane = Int32((head % 8) * 4) + tid
                    if cutlass.const_expr(head < 8):
                        total0 += cute.arch.shuffle_sync(p0, source_lane)
                        total1 += cute.arch.shuffle_sync(p1, source_lane)
                    else:
                        total0 += cute.arch.shuffle_sync(p2, source_lane)
                        total1 += cute.arch.shuffle_sync(p3, source_lane)
                cute.arch.sync_threads()
            if gid == Int32(0):
                key_row = warp * Int32(8) + tid * Int32(2)
                col = tile_start + key_row
                if col < end and valid_keys[key_row] != Int32(0):
                    scores[Int64(row) * Int64(width) + Int64(col)] = BFloat16(total0)
                if col + Int32(1) < end and valid_keys[key_row + Int32(1)] != Int32(0):
                    scores[Int64(row) * Int64(width) + Int64(col + Int32(1))] = (
                        BFloat16(total1)
                    )
            # No thread may overwrite K/validity for the next tile before all
            # warps have consumed this tile's validity flags.
            cute.arch.sync_threads()
            tile_start += cutlass.min(grid_stride, end - tile_start)


@dsl_user_op
def _mxfp4_mma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb, *, loc=None, ip=None):
    """Packed E2M1 m16n8k64 with two UE8M0 group scales per operand row."""
    operands = [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
    for scale in (sfa, sfb):
        operands.extend(
            (
                Uint32(scale).ir_value(loc=loc, ip=ip),
                Uint16(0).ir_value(loc=loc, ip=ip),
                Uint16(0).ir_value(loc=loc, ip=ip),
            )
        )
    operands.extend(Float32(v).ir_value(loc=loc, ip=ip) for v in (d0, d1, d2, d3))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        operands,
        """
        mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
        {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$0,$1,$2,$3},
        {$10}, {$11,$12}, {$13}, {$14,$15};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,h,h,r,h,h,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cute.jit
def _packed_word(data: cute.Tensor, offset: Int64):
    return ld_global_nc_u32(get_ptr_as_int64(data, offset))


@cute.jit
def _scale_pair(data: cute.Tensor, offset: Int64):
    return Uint32(data[offset]) | (Uint32(data[offset + Int64(1)]) << Uint32(8))


class _TensorCorePagedScore(_PagedScore):
    """Sixteen keys by eight heads per warp, with the staged BF16 score contract.

    Packed MXFP4 operands enter block-scaled FP32-accumulating MMA directly.
    ReLU, weighted-product rounding and ordered head summation remain separate;
    tensor cores do not fuse away the model's BF16 stage boundaries.
    """

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        qs: cute.Pointer,
        weights: cute.Pointer,
        pool: cute.Pointer,
        pages: cute.Pointer,
        lengths: cute.Pointer,
        active: cute.Pointer,
        candidates: cute.Pointer,
        candidate_lengths: cute.Pointer,
        scores: cute.Pointer,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(q),
            _flat(qs),
            _flat(weights),
            _flat(pool),
            _flat(pages),
            _flat(lengths),
            _flat(active),
            _flat(candidates),
            _flat(candidate_lengths),
            _flat(scores),
            rows,
            width,
            page_width,
            page_row_stride,
            pool_stride,
            pool_pages,
        ).launch(
            grid=(cutlass.min((width - Int32(1)) // Int32(64) + Int32(1), Int32(256)), rows, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        candidate_lengths: cute.Tensor,
        scores: cute.Tensor,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, row, _ = cute.arch.block_idx()
        grid_cols, _, _ = cute.arch.grid_dim()
        extent = cutlass.min(width, cutlass.min(lengths[row], active[0]))
        if cutlass.const_expr(self.candidates):
            extent = cutlass.min(width, candidate_lengths[row])
            if lengths[row] <= Int32(0) or active[0] <= Int32(0):
                extent = Int32(0)
        extent = cutlass.max(extent, Int32(0))
        tile_start = Int32(bx) * Int32(64)
        while tile_start < extent:
            self._score_tile(
                q,
                qs,
                weights,
                pool,
                pages,
                lengths,
                active,
                candidates,
                candidate_lengths,
                scores,
                rows,
                width,
                page_width,
                page_row_stride,
                pool_stride,
                pool_pages,
                tile_start,
            )
            tile_start += cutlass.min(Int32(grid_cols) * Int32(64), extent - tile_start)
        # Live tiles own their partial tails; the remaining columns are disjoint.
        tail_start = (Int64(extent) + Int64(63)) // Int64(64) * Int64(64)
        column = tail_start + Int64(bx) * Int64(128) + Int64(tx)
        while column < Int64(width):
            scores[Int64(row) * Int64(width) + column] = BFloat16(-float("inf"))
            column += Int64(grid_cols) * Int64(128)

    @cute.jit
    def _score_tile(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        candidate_lengths: cute.Tensor,
        scores: cute.Tensor,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        tile_start: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        _, row, _ = cute.arch.block_idx()
        lane = Int32(tx) % Int32(32)
        col = (
            tile_start
            + (Int32(tx) // Int32(32)) * Int32(16)
            + lane // Int32(4)
        )
        kb = cute.make_rmem_tensor((2,), Int64)
        sb = cute.make_rmem_tensor((2,), Int64)
        valid_key = cute.make_rmem_tensor((2,), cutlass.Boolean)
        totals = cute.make_rmem_tensor((2,), Float32)
        totals.fill(0.0)
        for part in cutlass.range_constexpr(2):
            column = col + Int32(part * 8)
            pos = column
            valid = column < width
            if cutlass.const_expr(self.candidates):
                valid = valid and column < candidate_lengths[row]
                if valid:
                    pos = candidates[Int64(row) * Int64(width) + Int64(column)]
            valid = valid and pos >= Int32(0) and pos < lengths[row] and pos < active[0]
            valid = valid and pos // Int32(self.page_size) < page_width
            page = Int64(-1)
            if valid:
                page = Int64(
                    pages[
                        Int64(row) * page_row_stride
                        + Int64(pos // Int32(self.page_size))
                    ]
                )
            valid = valid and page >= Int64(0) and page < pool_pages
            token = Int64(pos % Int32(self.page_size))
            valid_key[part] = valid
            kb[part] = page * pool_stride + token * Int64(64)
            sb[part] = (
                page * pool_stride + Int64(self.page_size * 64) + token * Int64(4)
            )
        for head_tile in cutlass.range_constexpr((self.heads + 7) // 8):
            head = Int32(head_tile * 8) + lane // Int32(4)
            qb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(64)
            qsb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(4)
            d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for kstep in cutlass.range_constexpr(2):
                byte_col = Int64(kstep * 32) + Int64(lane % Int32(4)) * Int64(4)
                a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if valid_key[0]:
                    a0 = _packed_word(pool, kb[0] + byte_col)
                    a2 = _packed_word(pool, kb[0] + byte_col + Int64(16))
                if valid_key[1]:
                    a1 = _packed_word(pool, kb[1] + byte_col)
                    a3 = _packed_word(pool, kb[1] + byte_col + Int64(16))
                b0, b1 = Uint32(0), Uint32(0)
                sfa, sfb = Uint32(0x7F7F), Uint32(0x7F7F)
                scale_row = lane % Int32(2)
                if valid_key[scale_row]:
                    sfa = _scale_pair(pool, sb[scale_row] + Int64(kstep * 2))
                if head < Int32(self.heads):
                    b0 = _packed_word(q, qb + byte_col)
                    b1 = _packed_word(q, qb + byte_col + Int64(16))
                    sfb = _scale_pair(qs, qsb + Int64(kstep * 2))
                d0, d1, d2, d3 = _mxfp4_mma(
                    d0,
                    d1,
                    d2,
                    d3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    sfa,
                    sfb,
                )
            contributions = cute.make_rmem_tensor((4,), Float32)
            dots = cute.make_rmem_tensor((4,), Float32)
            dots[0], dots[1], dots[2], dots[3] = d0, d1, d2, d3
            for part in cutlass.range_constexpr(4):
                output_head = (
                    Int32(head_tile * 8)
                    + (lane % Int32(4)) * Int32(2)
                    + Int32(part % 2)
                )
                weight = Float32(0)
                if output_head < Int32(self.heads):
                    weight = Float32(
                        weights[Int64(row) * Int64(self.heads) + Int64(output_head)]
                    )
                dot = fmax_f32(Float32(BFloat16(dots[part])), Float32(0))
                contributions[part] = Float32(BFloat16(dot * weight))
            # Preserve the scalar kernel's increasing-head FP32 sum order.
            for head_pair in cutlass.range_constexpr(4):
                source_lane = lane // Int32(4) * Int32(4) + Int32(head_pair)
                for part in cutlass.range_constexpr(4):
                    contribution = cute.arch.shuffle_sync(
                        contributions[part], source_lane
                    )
                    totals[part // 2] = totals[part // 2] + contribution
        if lane % Int32(4) == Int32(0):
            for part in cutlass.range_constexpr(2):
                column = col + Int32(part * 8)
                if column < width:
                    value = BFloat16(-float("inf"))
                    if valid_key[part]:
                        value = BFloat16(totals[part])
                    scores[Int64(row) * Int64(width) + Int64(column)] = value


class _SelectPrepare:
    def __init__(self, candidates: bool, blocks: bool):
        self.candidates = candidates
        self.blocks = blocks

    @cute.jit
    def __call__(
        self,
        scores: cute.Pointer,
        logits: cute.Pointer,
        lengths: cute.Pointer,
        active: cute.Pointer,
        candidate_lengths: cute.Pointer,
        select_lengths: cute.Pointer,
        rows: Int32,
        width: Int32,
        output_width: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(scores),
            _flat(logits),
            _flat(lengths),
            _flat(active),
            _flat(candidate_lengths),
            _flat(select_lengths),
            rows,
            width,
            output_width,
        ).launch(
            grid=(
                cutlass.min(
                    (output_width - Int32(1)) // Int32(256) + Int32(1), Int32(256)
                ),
                rows,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        logits: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidate_lengths: cute.Tensor,
        select_lengths: cute.Tensor,
        rows: Int32,
        width: Int32,
        output_width: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, row, _ = cute.arch.block_idx()
        grid_cols, _, _ = cute.arch.grid_dim()
        grid_stride = Int32(grid_cols) * Int32(256)
        col = Int32(bx) * Int32(256) + Int32(tx)
        visible = cutlass.min(
            cutlass.max(lengths[row], Int32(0)), cutlass.max(active[0], Int32(0))
        )
        extent = cutlass.min(visible, width)
        if cutlass.const_expr(self.candidates):
            extent = cutlass.min(cutlass.max(candidate_lengths[row], Int32(0)), width)
        if cutlass.const_expr(self.blocks):
            extent = (extent + Int32(7)) // Int32(8)
        if col == Int32(0):
            select_lengths[row] = extent
        # The selector consumes only extent entries. The private logit tail
        # need not be initialized; the public BF16 score tail remains -inf.
        while col < extent:
            value = Float32(-float("inf"))
            if cutlass.const_expr(self.blocks):
                for offset in cutlass.range_constexpr(8):
                    pos = col * Int32(8) + Int32(offset)
                    if pos < width and pos < visible:
                        value = fmax_f32(
                            value,
                            Float32(scores[Int64(row) * Int64(width) + Int64(pos)]),
                        )
                if visible > Int32(0) and col == (visible - Int32(1)) // Int32(8):
                    value = Float32(float("inf"))
            else:
                value = Float32(scores[Int64(row) * Int64(width) + Int64(col)])
            logits[Int64(row) * Int64(output_width) + Int64(col)] = value
            col += cutlass.min(grid_stride, extent - col)


class _SortPositions:
    def __init__(self, topk: int, expand_blocks: bool):
        self.topk = topk
        self.expand_blocks = expand_blocks

    @cute.jit
    def __call__(
        self,
        indices: cute.Pointer,
        values: cute.Pointer,
        out: cute.Pointer,
        out_values: cute.Pointer,
        lengths: cute.Pointer,
        active: cute.Pointer,
        out_lengths: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(indices),
            _flat(values),
            _flat(out),
            _flat(out_values),
            _flat(lengths),
            _flat(active),
            _flat(out_lengths),
        ).launch(grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        indices: cute.Tensor,
        values: cute.Tensor,
        out: cute.Tensor,
        out_values: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        out_lengths: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        # Each of the eight warps owns one element from every 256-slot stripe.
        # The first five bitonic levels are therefore warp-local; retain them in
        # registers and publish only the completed 32-element runs.  Higher
        # levels retain shared storage only for compare-exchanges that actually
        # cross a warp boundary.
        si = smem.allocate_tensor(
            Int32, cute.make_layout((self.topk,)), byte_alignment=16
        )
        if cutlass.const_expr(not self.expand_blocks):
            sv = smem.allocate_tensor(
                Float32, cute.make_layout((self.topk,)), byte_alignment=16
            )
        visible = cutlass.min(
            cutlass.max(lengths[row], Int32(0)), cutlass.max(active[0], Int32(0))
        )
        for slot in cutlass.range(Int32(tx), self.topk, 256):
            base = Int64(row) * Int64(self.topk) + Int64(slot)
            idx = indices[base]
            value = values[base]
            valid = idx >= Int32(0) and value > Float32(-float("inf"))
            if cutlass.const_expr(self.expand_blocks):
                valid = valid and idx * Int32(8) < visible
            else:
                valid = valid and idx < visible
            if not valid:
                idx = Int32(2147483647)

            for level in cutlass.range_constexpr(1, 6):
                for step in cutlass.range_constexpr(level - 1, -1, -1):
                    other_idx = cute.arch.shuffle_sync_bfly(idx, offset=1 << step)
                    ascending = ((slot & Int32(1 << level)) == Int32(0)) == (
                        (slot & Int32(1 << step)) == Int32(0)
                    )
                    swap = (ascending and idx > other_idx) or (
                        not ascending and idx < other_idx
                    )
                    if cutlass.const_expr(not self.expand_blocks):
                        other_value = cute.arch.shuffle_sync_bfly(
                            value, offset=1 << step
                        )
                    if swap:
                        idx = other_idx
                        if cutlass.const_expr(not self.expand_blocks):
                            value = other_value
            si[slot] = idx
            if cutlass.const_expr(not self.expand_blocks):
                sv[slot] = value
        cute.arch.sync_threads()

        for level in cutlass.range_constexpr(6, self.topk.bit_length()):
            # Steps >= 5 pair lanes from distinct warps, so shared storage and a
            # CTA fence are required only for these exchanges.
            for step in cutlass.range_constexpr(level - 1, 4, -1):
                for slot in cutlass.range(Int32(tx), self.topk, 256):
                    other = slot ^ Int32(1 << step)
                    if other > slot:
                        idx, other_idx = si[slot], si[other]
                        ascending = (slot & Int32(1 << level)) == Int32(0)
                        if (ascending and idx > other_idx) or (
                            not ascending and idx < other_idx
                        ):
                            si[slot], si[other] = other_idx, idx
                            if cutlass.const_expr(not self.expand_blocks):
                                value, other_value = sv[slot], sv[other]
                                sv[slot], sv[other] = other_value, value
                cute.arch.sync_threads()

            # The remaining low-bit stages are again confined to each warp.
            for slot in cutlass.range(Int32(tx), self.topk, 256):
                idx = si[slot]
                if cutlass.const_expr(not self.expand_blocks):
                    value = sv[slot]
                for step in cutlass.range_constexpr(4, -1, -1):
                    other_idx = cute.arch.shuffle_sync_bfly(idx, offset=1 << step)
                    ascending = ((slot & Int32(1 << level)) == Int32(0)) == (
                        (slot & Int32(1 << step)) == Int32(0)
                    )
                    swap = (ascending and idx > other_idx) or (
                        not ascending and idx < other_idx
                    )
                    if cutlass.const_expr(not self.expand_blocks):
                        other_value = cute.arch.shuffle_sync_bfly(
                            value, offset=1 << step
                        )
                    if swap:
                        idx = other_idx
                        if cutlass.const_expr(not self.expand_blocks):
                            value = other_value
                si[slot] = idx
                if cutlass.const_expr(not self.expand_blocks):
                    sv[slot] = value
            cute.arch.sync_threads()

        for slot in cutlass.range(Int32(tx), self.topk, 256):
            idx = si[slot]
            valid = idx != Int32(2147483647)
            if cutlass.const_expr(self.expand_blocks):
                for offset in cutlass.range_constexpr(8):
                    pos = idx * Int32(8) + Int32(offset)
                    out_pos = (Int64(row) * Int64(self.topk) + Int64(slot)) * Int64(
                        8
                    ) + Int64(offset)
                    out[out_pos] = Int32(-1)
                    if valid and pos < visible:
                        out[out_pos] = pos
            else:
                out_pos = Int64(row) * Int64(self.topk) + Int64(slot)
                out[out_pos] = Int32(-1)
                out_values[out_pos] = Float32(-float("inf"))
                if valid:
                    out[out_pos] = idx
                    out_values[out_pos] = sv[slot]

        if cutlass.const_expr(self.expand_blocks):
            # Reuse the now-dead first eight index slots for a two-stage CTA
            # reduction; this counts only selected, visible (including partial)
            # blocks and avoids a serial thread-0 scan.
            count = Int32(0)
            for slot in cutlass.range(Int32(tx), self.topk, 256):
                idx = si[slot]
                if idx != Int32(2147483647):
                    count += cutlass.min(Int32(8), visible - idx * Int32(8))
            lane = Int32(tx) & Int32(31)
            warp = Int32(tx) >> Int32(5)
            for offset in cutlass.range_constexpr(5):
                count += cute.arch.shuffle_sync_bfly(count, offset=1 << offset)
            cute.arch.sync_threads()
            if lane == Int32(0):
                si[warp] = count
            cute.arch.sync_threads()
            if warp == Int32(0):
                count = Int32(0)
                if lane < Int32(8):
                    count = si[lane]
                for offset in cutlass.range_constexpr(5):
                    count += cute.arch.shuffle_sync_bfly(count, offset=1 << offset)
                if lane == Int32(0):
                    out_lengths[row] = count


@dataclass(frozen=True)
class _CompiledLauncher:
    raw: object
    dtypes: tuple

    @property
    def __b12x_dependencies__(self):
        return (self.raw,)

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys
        return program_keys(self.raw)


@program_cache
def _compile(kind: str, recipe: tuple, device_index: int):
    # Pointer-only launch ABIs: all live row/page/width/stride quantities are
    # runtime scalars and do not contribute to compile identity.
    if kind == "quantize":
        obj = _Quantize(*recipe)
        dtypes = (BFloat16, Uint8, Uint8, Int64)
        scalars = (Int32(1), Int64(1), Int64(4352))
    elif kind in ("score", "score_tensorcore"):
        obj = (_TensorCorePagedScore if kind == "score_tensorcore" else _PagedScore)(
            *recipe
        )
        dtypes = (
            Uint8,
            Uint8,
            BFloat16,
            Uint8,
            Int32,
            Int32,
            Int32,
            Int32,
            Int32,
            BFloat16,
        )
        scalars = (Int32(1), Int32(64), Int32(1), Int64(1), Int64(4352), Int64(1))
    elif kind == "prepare":
        obj = _SelectPrepare(*recipe)
        dtypes = (BFloat16, Float32, Int32, Int32, Int32, Int32)
        scalars = (Int32(1), Int32(64), Int32(64))
    else:
        obj = _SortPositions(*recipe)
        dtypes = (Int32, Float32, Int32, Float32, Int32, Int32, Int32)
        scalars = (Int32(1),)
    key = (kind, recipe, device_index)
    raise_if_kernel_resolution_frozen("cute.compile", target=obj, cache_key=key)
    pointers = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1) for t in dtypes
    )
    raw = b12x_compile(
        obj,
        *pointers,
        *scalars,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("attention.indexer.mxfp4", 9, key),
    )
    return _CompiledLauncher(raw, dtypes)


def _launch(kind, recipe, tensors, scalars, *, launcher=None):
    if tensors[0].device.type != "cuda":
        raise ValueError("native MXFP4 indexer requires CUDA tensors")
    with torch.cuda.device(tensors[0].device):
        if launcher is None:
            launcher = _compile(kind, recipe, tensors[0].device.index)
        launcher.raw(
            *(_ptr(t, dtype) for t, dtype in zip(tensors, launcher.dtypes, strict=True)),
            *scalars,
            current_cuda_stream(),
        )


def _check(tensor, name, shape, dtype, device):
    if tensor is None or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _quantize_q_mxfp4(
    query: torch.Tensor, *, q_mxfp4: torch.Tensor, q_scales: torch.Tensor,
    launcher,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the prepared query-quantizer launcher into caller-owned buffers."""
    if query.ndim < 2 or query.shape[-1] != 128:
        raise ValueError("query must have shape (...,128)")
    _check(query, "query", query.shape, torch.bfloat16, query.device)
    _check(q_mxfp4, "q_mxfp4", (*query.shape[:-1], 64), torch.uint8, query.device)
    _check(q_scales, "q_scales", (*query.shape[:-1], 4), torch.uint8, query.device)
    rows = query.numel() // 128
    if rows:
        _launch(
            "quantize", (False, 64), (query, q_mxfp4, q_scales, query),
            (rows, 0, 0), launcher=launcher,
        )
    return q_mxfp4, q_scales


def _quantize_write_index_k_mxfp4(
    keys: torch.Tensor,
    *,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int = 64,
    launcher,
) -> torch.Tensor:
    """Run the prepared paged-key writer against caller-owned physical slots."""
    page_bytes = index_mxfp4_page_bytes(page_size)
    if keys.ndim != 2 or keys.shape[1] != 128:
        raise ValueError("keys must have shape (rows,128)")
    _check(keys, "keys", keys.shape, torch.bfloat16, keys.device)
    _check(slot_mapping, "slot_mapping", (keys.shape[0],), torch.int64, keys.device)
    _check_pool(index_k_cache, page_bytes, keys.device)
    if keys.shape[0]:
        _launch(
            "quantize",
            (True, page_size),
            (keys, index_k_cache, index_k_cache, slot_mapping),
            (keys.shape[0], index_k_cache.shape[0], index_k_cache.stride(0)),
            launcher=launcher,
        )
    return index_k_cache


def _check_pool(pool, page_bytes, device):
    if pool.ndim != 2 or pool.shape[1] != page_bytes or pool.dtype != torch.uint8:
        raise ValueError(f"index_k_cache must be uint8 (pages,{page_bytes})")
    if (
        pool.device != device
        or pool.stride(1) != 1
        or pool.stride(0) < page_bytes
        or pool.stride(0) % 16
        or pool.data_ptr() % 16
    ):
        raise ValueError(
            "index_k_cache must have aligned, nonoverlapping page storage on the plan device"
        )


@dataclass(frozen=True)
class MXFP4PagedPlan:
    caps: object
    views: tuple
    nbytes: int
    score_kind: str

    @property
    def layout(self):
        return self

    def scratch_specs(self):
        return (
            scratch_buffer_spec(
                "paged_indexer.mxfp4", nbytes=self.nbytes, device=self.caps.device
            ),
        )

    def shapes_and_dtypes(self):
        return tuple((spec.shape, spec.dtype) for spec in self.scratch_specs())

    def bind(self, *, scratch, score_width=None, **kwargs):
        storage = scratch_tensor(scratch, self.scratch_specs(), owner="MXFP4 DSA")
        views = {}
        for name, shape, dtype, offset, _ in self.views:
            if score_width is not None and name in ("scores", "logits"):
                shape = (shape[0], score_width)
            elif score_width is not None and name == "block_logits":
                shape = (shape[0], (score_width + 7) // 8)
            views[name], _ = materialize_scratch_view(
                storage,
                offset_bytes=offset,
                shape=shape,
                dtype=dtype,
            )
        return MXFP4Runtime(scratch=views, **kwargs)


def plan_mxfp4(caps, *, score_kind=None):
    width = caps.max_candidates or caps.max_page_table_width * caps.page_size
    rows = caps.max_q_rows
    index_mxfp4_page_bytes(caps.page_size)
    if caps.num_q_heads > 32 or 32 % caps.num_q_heads:
        raise ValueError("MXFP4 indexer requires a divisor of 32 global heads")
    if rows * width > 2**31 - 1:
        raise ValueError("MXFP4 logits exceed the row selector's kernel tensor limit")
    shapes = [
        ("scores", (rows, width), torch.bfloat16),
        ("logits", (rows, width), torch.float32),
        ("lengths", (rows,), torch.int32),
        ("indices", (rows, caps.topk), torch.int32),
        ("values", (rows, caps.topk), torch.float32),
        ("sorted_values", (rows, caps.topk), torch.float32),
    ]
    if caps.candidate_topk_blocks:
        shapes += [
            ("block_logits", (rows, (width + 7) // 8), torch.float32),
            ("block_lengths", (rows,), torch.int32),
            ("block_indices", (rows, caps.candidate_topk_blocks), torch.int32),
            ("block_values", (rows, caps.candidate_topk_blocks), torch.float32),
        ]
    offset = 0
    views = []
    for name, shape, dtype in shapes:
        offset = align_up(offset, SCRATCH_ALIGN_BYTES)
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * dtype_nbytes(dtype)
        views.append((name, shape, dtype, offset, nbytes))
        offset += nbytes
    if score_kind is None:
        score_kind = "score_tensorcore" if caps.mode == "prefill" or caps.num_q_heads == 32 else "score"
    if score_kind not in ("score", "score_tensorcore"):
        raise ValueError("unsupported MXFP4 score kind")
    return MXFP4PagedPlan(caps, tuple(views), offset, score_kind)


@dataclass(frozen=True, kw_only=True)
class MXFP4Runtime:
    scratch: dict
    page_table: torch.Tensor
    cache_lengths: torch.Tensor
    active_width: torch.Tensor
    candidate_indices: torch.Tensor | None
    candidate_lengths: torch.Tensor | None
    candidate_output: torch.Tensor | None
    candidate_output_lengths: torch.Tensor | None


def bind_mxfp4(
    plan,
    *,
    scratch,
    q_mxfp4,
    q_scales,
    query_weights,
    index_k_cache,
    page_table,
    cache_lengths,
    active_width,
    output_indices,
    output_scores,
    candidate_indices,
    candidate_lengths,
    candidate_output,
    candidate_output_lengths,
    score_width=None,
):
    from .api import Binding

    caps = plan.caps
    width_capacity = caps.max_candidates or caps.max_page_table_width * caps.page_size
    if score_width is not None and (
        isinstance(score_width, bool) or not isinstance(score_width, int)
    ):
        raise TypeError("score_width must be an integer")
    score_width = width_capacity if score_width is None else score_width
    if not 0 < score_width <= width_capacity:
        raise ValueError("score_width must be positive and within planned capacity")
    if caps.max_candidates and score_width != caps.max_candidates:
        raise ValueError("candidate scoring retains its planned candidate width")
    if q_mxfp4 is None or q_mxfp4.ndim != 3:
        raise ValueError("MXFP4 recipe requires explicit q_mxfp4 and q_scales")
    rows = q_mxfp4.shape[0]
    if rows <= 0 or rows > caps.max_q_rows:
        raise ValueError("query rows must be positive and within planned capacity")
    _check(q_mxfp4, "q_mxfp4", (rows, caps.num_q_heads, 64), torch.uint8, caps.device)
    _check(q_scales, "q_scales", (rows, caps.num_q_heads, 4), torch.uint8, caps.device)
    if query_weights.ndim == 3 and query_weights.shape[-1] == 1:
        query_weights = query_weights.view(rows, caps.num_q_heads)
    _check(
        query_weights,
        "query_weights",
        (rows, caps.num_q_heads),
        torch.bfloat16,
        caps.device,
    )
    _check_pool(index_k_cache, index_mxfp4_page_bytes(caps.page_size), caps.device)
    if page_table.ndim != 2 or page_table.shape[1] > caps.max_page_table_width:
        raise ValueError("page_table width exceeds planned capacity")
    if (
        page_table.dtype != torch.int32
        or page_table.device != caps.device
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            "page_table must be int32 with unit inner stride on plan device"
        )
    if page_table.shape[0] not in (1, rows):
        raise ValueError("page_table must have one shared row or one row per query")
    _check(cache_lengths, "cache_lengths", (rows,), torch.int32, caps.device)
    _check(active_width, "active_width", (1,), torch.int32, caps.device)
    _check(
        output_indices, "output_indices", (rows, caps.topk), torch.int32, caps.device
    )
    if output_scores is not None:
        _check(
            output_scores,
            "output_scores",
            (rows, caps.topk),
            torch.float32,
            caps.device,
        )
    if caps.max_candidates:
        _check(
            candidate_indices,
            "candidate_indices",
            (rows, caps.max_candidates),
            torch.int32,
            caps.device,
        )
        _check(
            candidate_lengths, "candidate_lengths", (rows,), torch.int32, caps.device
        )
    elif candidate_indices is not None or candidate_lengths is not None:
        raise ValueError("candidate inputs require max_candidates in Caps")
    if caps.candidate_topk_blocks:
        _check(
            candidate_output,
            "candidate_output",
            (rows, caps.candidate_topk_blocks * 8),
            torch.int32,
            caps.device,
        )
        _check(
            candidate_output_lengths,
            "candidate_output_lengths",
            (rows,),
            torch.int32,
            caps.device,
        )
    elif candidate_output is not None or candidate_output_lengths is not None:
        raise ValueError("candidate output requires candidate_topk_blocks in Caps")
    runtime = plan.inner.bind(
        scratch=scratch,
        score_width=score_width,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        candidate_indices=candidate_indices,
        candidate_lengths=candidate_lengths,
        candidate_output=candidate_output,
        candidate_output_lengths=candidate_output_lengths,
    )
    return runtime


def score_mxfp4(binding, *, launchers=None):
    caps, rt = binding.plan.caps, binding.runtime
    rows = binding.q_mxfp4.shape[0]
    scores = rt.scratch["scores"][:rows]
    candidates = rt.candidate_indices if caps.max_candidates else rt.cache_lengths
    candidate_lengths = rt.candidate_lengths if caps.max_candidates else rt.cache_lengths
    page_stride = 0 if rt.page_table.shape[0] == 1 else rt.page_table.stride(0)
    kind = binding.plan.inner.score_kind
    recipe = (caps.num_q_heads, bool(caps.max_candidates), caps.page_size)
    _launch(
        kind, recipe,
        (binding.q_mxfp4, binding.q_scales, binding.query_weights,
         binding.index_k_cache, rt.page_table, rt.cache_lengths, rt.active_width,
         candidates, candidate_lengths, scores),
        (rows, scores.shape[1], rt.page_table.shape[1], page_stride,
         binding.index_k_cache.stride(0), binding.index_k_cache.shape[0]),
        launcher=None if launchers is None else launchers[(kind, recipe)],
    )
    return scores


def select_mxfp4(binding, *, launchers=None):
    caps, rt = binding.plan.caps, binding.runtime
    rows = binding.q_mxfp4.shape[0]
    s = {name: view[:rows] for name, view in rt.scratch.items()}
    candidate_lengths = rt.candidate_lengths if caps.max_candidates else rt.cache_lengths
    def launch(kind, recipe, tensors, scalars):
        _launch(kind, recipe, tensors, scalars,
                launcher=None if launchers is None else launchers[(kind, recipe)])
    launch("prepare", (bool(caps.max_candidates), False),
           (s["scores"], s["logits"], rt.cache_lengths, rt.active_width,
            candidate_lengths, s["lengths"]),
           (rows, s["scores"].shape[1], s["logits"].shape[1]))
    run_row_topk(row_logits=s["logits"], lengths=s["lengths"], topk=caps.topk,
                 output_values=s["values"], output_indices=s["indices"],
                 output_gather_table=rt.candidate_indices,
                 launcher=None if launchers is None else launchers[("topk", caps.topk)][0])
    out_values = binding.output_scores if binding.output_scores is not None else s["sorted_values"]
    launch("sort", (caps.topk, False),
           (s["indices"], s["values"], binding.output_indices, out_values,
            rt.cache_lengths, rt.active_width, s["lengths"]), (rows,))
    if caps.candidate_topk_blocks:
        launch("prepare", (False, True),
               (s["scores"], s["block_logits"], rt.cache_lengths, rt.active_width,
                rt.cache_lengths, s["block_lengths"]),
               (rows, s["scores"].shape[1], s["block_logits"].shape[1]))
        run_row_topk(row_logits=s["block_logits"], lengths=s["block_lengths"],
                     topk=caps.candidate_topk_blocks,
                     output_values=s["block_values"], output_indices=s["block_indices"],
                     launcher=None if launchers is None else launchers[("topk", caps.candidate_topk_blocks)][0])
        launch("sort", (caps.candidate_topk_blocks, True),
               (s["block_indices"], s["block_values"], rt.candidate_output,
                s["block_values"], rt.cache_lengths, rt.active_width,
                rt.candidate_output_lengths), (rows,))
    return binding.output_indices


@dataclass(frozen=True, kw_only=True)
class MXFP4Binding:
    """Private binding carrier; public callers receive dsa_indexer.Binding."""
    plan: object
    runtime: MXFP4Runtime
    q_mxfp4: torch.Tensor
    q_scales: torch.Tensor
    query_weights: torch.Tensor
    index_k_cache: torch.Tensor
    output_indices: torch.Tensor
    output_scores: torch.Tensor | None


class MXFP4PreparedState:
    """Materialized MXFP4 indexer retaining every callable used at runtime."""
    def __init__(self, layout, launchers):
        from types import MappingProxyType
        self.layout = layout
        self.caps = layout.caps
        self._launchers = MappingProxyType(dict(launchers))

    @property
    def __b12x_programs__(self):
        """Expose the retained concrete carriers to compile-pool accounting."""
        from b12x._lib.compile_plan import program_keys
        return program_keys(self.__b12x_dependencies__)

    @property
    def __b12x_dependencies__(self):
        return tuple(self._launchers.values())

    def bind(self, *, scratch, q_mxfp4, q_scales, query_weights, index_k_cache,
             page_table, cache_lengths, active_width, output_indices,
             output_scores=None, candidate_indices=None, candidate_lengths=None,
             candidate_output=None, candidate_output_lengths=None, score_width=None,
             **_ignored):
        runtime = bind_mxfp4(
            self, scratch=scratch, q_mxfp4=q_mxfp4, q_scales=q_scales,
            query_weights=query_weights, index_k_cache=index_k_cache,
            page_table=page_table, cache_lengths=cache_lengths,
            active_width=active_width, output_indices=output_indices,
            output_scores=output_scores, candidate_indices=candidate_indices,
            candidate_lengths=candidate_lengths, candidate_output=candidate_output,
            candidate_output_lengths=candidate_output_lengths, score_width=score_width,
        )
        return MXFP4Binding(plan=self, runtime=runtime, q_mxfp4=q_mxfp4,
                            q_scales=q_scales, query_weights=query_weights,
                            index_k_cache=index_k_cache, output_indices=output_indices,
                            output_scores=output_scores)

    @property
    def inner(self):
        return self.layout

    def quantize_query(self, query, *, q_mxfp4, q_scales):
        return _quantize_q_mxfp4(
            query, q_mxfp4=q_mxfp4, q_scales=q_scales,
            launcher=self._launchers[("quantize", (False, 64))],
        )

    def write_index_keys(self, keys, *, index_k_cache, slot_mapping):
        return _quantize_write_index_k_mxfp4(
            keys, index_k_cache=index_k_cache, slot_mapping=slot_mapping,
            page_size=self.caps.page_size,
            launcher=self._launchers[("quantize", (True, self.caps.page_size))],
        )

    def run(self, binding):
        score_mxfp4(binding, launchers=self._launchers)
        return select_mxfp4(binding, launchers=self._launchers)


def materialize_mxfp4(caps, *, device_index, score_kind=None):
    """Create durable layout and launchers; execution never consults compiler caches."""
    layout = plan_mxfp4(caps, score_kind=score_kind)
    recipes = [
        ("quantize", (False, 64)),
        ("quantize", (True, caps.page_size)),
        (layout.score_kind,
         (caps.num_q_heads, bool(caps.max_candidates), caps.page_size)),
        ("prepare", (bool(caps.max_candidates), False)),
        ("sort", (caps.topk, False)),
    ]
    if caps.candidate_topk_blocks:
        recipes.extend((
            ("prepare", (False, True)),
            ("sort", (caps.candidate_topk_blocks, True)),
        ))
    launchers = {
        (kind, recipe): _compile(kind, recipe, device_index)
        for kind, recipe in recipes
    }
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x._lib.compile_plan import compile_only_launches

    width = caps.max_candidates or caps.max_page_table_width * caps.page_size
    selectors = [(caps.topk, width, bool(caps.max_candidates))]
    if caps.candidate_topk_blocks:
        selectors.append((caps.candidate_topk_blocks, (width + 7) // 8, False))
    with FakeTensorMode(), compile_only_launches():
        for topk, columns, gather in selectors:
            def empty(shape, dtype):
                return torch.empty(shape, dtype=dtype, device=caps.device)
            resolved = {}
            run_row_topk(
                row_logits=empty((caps.max_q_rows, columns), torch.float32),
                lengths=empty((caps.max_q_rows,), torch.int32),
                topk=topk,
                output_values=empty((caps.max_q_rows, topk), torch.float32),
                output_indices=empty((caps.max_q_rows, topk), torch.int32),
                output_gather_table=empty((caps.max_q_rows, columns), torch.int32) if gather else None,
                launcher_sink=resolved,
            )
            launchers[("topk", topk)] = (resolved[("row", gather, True)], ())
    return MXFP4PreparedState(layout, launchers)
