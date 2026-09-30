"""Segment metadata, affine boundary propagation, and pooled-state commit."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import register_program_cache
from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32, cp_async_bulk_g2s_mbar,
    ldmatrix_m8n8x4_trans_b16, shared_ptr_to_u32, st_global_v2_f32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream
from .._shared.delta_prefill._cute_kernels import (
    _RecurrenceKernel, _fake_pointer, _numeric_type, _pointer, _pointer_address,
)


class _PartitionKernel:
    def __init__(self, *, heads: int, max_seqs: int, max_segments: int,
                 segment_tokens: int, reuse_outputs: bool) -> None:
        self.heads = heads
        self.max_seqs = max_seqs
        self.max_segments = max_segments
        self.segment_tokens = segment_tokens
        self.reuse_outputs = reuse_outputs
        self.checkpoint_base = 2 + (4 if reuse_outputs else 3) * max_segments

    @cute.jit
    def __call__(self, original: tuple, segments: tuple, pool: cute.Pointer,
                 stream: cuda.CUstream):
        self.kernel(original, segments, pool).launch(
            grid=(self.heads * 16, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, original: tuple, segments: tuple, pool: cute.Pointer):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread, block = Int32(thread), Int32(block)
        cu, _, _, _, checkpoint_offsets, num_seqs, num_tokens = original
        seq_segments, seg_cu, zero, identity, transfer, local, output, checkpoint, offsets, no_checkpoint, seg_count, token_count = segments
        item = thread
        if block == Int32(0):
            while item < Int32(self.max_segments):
                zero[item] = Int32(0)
                identity[item] = Int32(1)
                transfer[item] = Int32(2) + item
                local[item] = Int32(2 + self.max_segments) + item
                output[item] = Int32(2 + 2 * self.max_segments) + item
                checkpoint[item] = Int32(0)
                offsets[item] = Int32(0)
                no_checkpoint[item] = Int32(0)
                seg_cu[item] = Int32(0)
                item += Int32(256)
        for part in cutlass.range_constexpr(4):
            element = block * Int32(1024) + thread + Int32(part * 256)
            column = element % Int32(128)
            row = (element // Int32(128)) % Int32(128)
            value = Float32(0.0)
            if row == column:
                value = Float32(1.0)
            pool[Int64(self.heads * 128 * 128) + element.to(Int64)] = value
        cute.arch.sync_threads()
        if (thread == Int32(0)) & (block == Int32(0)):
            live = num_seqs[Int32(0)].to(Int32)
            tokens = num_tokens[Int32(0)].to(Int32)
            count = Int32(0)
            for seq in cutlass.range(self.max_seqs, unroll=1):
                seq_segments[seq] = count
                if seq < live:
                    start = cu[seq].to(Int32)
                    end = cu[seq + Int32(1)].to(Int32)
                    checkpoint_offset = checkpoint_offsets[seq].to(Int32)
                    relative = Int32(0)
                    while start + relative < end:
                        seg_cu[count] = start + relative
                        length = cutlass.min(Int32(self.segment_tokens), end - start - relative)
                        if (checkpoint_offset > relative) & (checkpoint_offset <= relative + length):
                            checkpoint[count] = Int32(self.checkpoint_base) + seq
                            offsets[count] = checkpoint_offset - relative
                        relative += length
                        if start + relative == end:
                            transfer[count] = Int32(0)
                            if cutlass.const_expr(not self.reuse_outputs):
                                local[count] = Int32(0)
                        count += Int32(1)
            seq_segments[Int32(self.max_seqs)] = count
            item = count
            while item <= Int32(self.max_segments):
                seg_cu[item] = tokens
                item += Int32(1)
            seg_count[Int32(0)] = count
            token_count[Int32(0)] = tokens


class _PackTransferKernel:
    """Pack centered BF16 transfers; flag exact-zero transfers and finite local states.

    Bit 0 certifies every original FP32 transfer entry is zero. Bit 1
    certifies every FP32 local-state entry is finite.
    """

    def __init__(self, *, heads: int, max_seqs: int, max_segments: int) -> None:
        self.heads = heads
        self.max_segments = max_segments

    @cute.jit
    def __call__(self, original: tuple, segments: tuple, pool: cute.Pointer,
                 packed: cute.Pointer, transfer_flags: cute.Pointer, stream: cuda.CUstream):
        self.kernel(original, segments, pool, packed, transfer_flags).launch(
            grid=(self.heads, self.max_segments, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, original: tuple, segments: tuple, pool: cute.Pointer,
               packed: cute.Pointer, transfer_flags: cute.Pointer):
        head, segment, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        head, segment, thread = Int32(head), Int32(segment), Int32(thread)
        if (segment < segments[10][Int32(0)]) & (segments[4][segment] != Int32(0)):
            stride = Int64(self.heads * 128 * 128)
            source_base = (Int64(2) + segment.to(Int64)) * stride + head.to(Int64) * Int64(128 * 128)
            target_base = segment.to(Int64) * stride + head.to(Int64) * Int64(128 * 128)
            local_base = (Int64(2 + self.max_segments) + segment.to(Int64)) * stride + head.to(Int64) * Int64(128 * 128)
            center = Float32(pool[source_base])
            all_zero = cutlass.Boolean(True)
            all_finite = cutlass.Boolean(True)
            element = thread
            while element < Int32(128 * 128):
                row, col = element // Int32(128), element % Int32(128)
                value = Float32(pool[source_base + element.to(Int64)])
                all_zero = all_zero & (value == Float32(0.0))
                local_value = Float32(pool[local_base + element.to(Int64)])
                all_finite = (all_finite & (local_value > Float32(float("-inf")))
                              & (local_value < Float32(float("inf"))))
                if row == col:
                    value -= center
                physical = ((col >> Int32(3)) ^ (row & Int32(7))) * Int32(8) + (col & Int32(7))
                packed[target_base + (row * Int32(128) + physical).to(Int64)] = value.to(BFloat16)
                element += Int32(256)
            warp_zero = cute.arch.vote_all_sync(all_zero)
            warp_finite = cute.arch.vote_all_sync(all_finite)
            allocator = cutlass.utils.SmemAllocator()
            votes = allocator.allocate_tensor(element_type=Int32,
                                              layout=cute.make_layout((8,), stride=(1,)))
            if thread % Int32(32) == Int32(0):
                votes[thread // Int32(32)] = warp_zero.to(Int32) | (warp_finite.to(Int32) << Int32(1))
            cute.arch.sync_threads()
            if thread == Int32(0):
                result = Int32(3)
                for warp in cutlass.range_constexpr(8):
                    result = result & votes[Int32(warp)]
                transfer_flags[segment.to(Int64) * Int64(self.heads) + head.to(Int64)] = result


class _BoundaryKernel(_RecurrenceKernel):
    """Propagate FP32 boundary states through centered affine summaries."""

    def __init__(self, *, heads: int, max_seqs: int, max_segments: int,
                 v_split: int, null_state_index: int | None) -> None:
        super().__init__(heads=heads, tiles_capacity=1, window_tiles=1,
                         rows=max_seqs, v_split=v_split, k_split=1, stages=2,
                         checkpoint_export=False, null_state_index=null_state_index,
                         index_type=Int32)
        self.max_segments = max_segments

    @cute.jit
    def _store_state(self, target: cute.Pointer, acc: cute.Tensor, base: Int64,
                     row0: Int32, row1: Int32, col_base: Int32, tid: Int32):
        for nb in cutlass.range_constexpr(16):
            col = col_base + Int32(nb * 8) + tid * Int32(2)
            offset0 = base + row0.to(Int64) * Int64(128) + col.to(Int64)
            offset1 = base + row1.to(Int64) * Int64(128) + col.to(Int64)
            st_global_v2_f32(_pointer_address(target, offset0), acc[nb, 0], acc[nb, 1])
            st_global_v2_f32(_pointer_address(target, offset1), acc[nb, 2], acc[nb, 3])

    @cute.jit
    def __call__(self, original: tuple, segments: tuple, pool: cute.Pointer,
                 packed: cute.Pointer, transfer_flags: cute.Pointer,
                 source: cute.Pointer, source_stride: Int64,
                 stream: cuda.CUstream):
        self.kernel(original, segments, pool, packed, transfer_flags, source, source_stride).launch(
            grid=(self.heads * self.splits, self.max_segments, 1),
            block=(self.mma_threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, original: tuple, segments: tuple, pool: cute.Pointer,
               packed: cute.Pointer, transfer_flags: cute.Pointer,
               source: cute.Pointer, source_stride: Int64):
        block, target_segment, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread, target_segment = Int32(thread), Int32(target_segment)
        head = Int32(block) // Int32(self.splits)
        split = Int32(block) % Int32(self.splits)
        lane = Int32(cute.arch.lane_idx())
        warp = thread // Int32(32)
        gid, tid = lane >> Int32(2), lane & Int32(3)
        row0 = split * Int32(self.v_split) + warp * Int32(16) + gid
        row1 = row0 + Int32(8)
        matrix, matrix_row = lane >> Int32(3), lane & Int32(7)
        _, initial, _, _, _, num_seqs, _ = original
        seq_segments = segments[0]
        if target_segment < segments[10][Int32(0)]:
            seq = Int32(0)
            high = num_seqs[Int32(0)].to(Int32)
            while high > seq + Int32(1):
                mid = (seq + high) >> Int32(1)
                if seq_segments[mid] <= target_segment:
                    seq = mid
                else:
                    high = mid
            begin = seq_segments[seq].to(Int32)
            end = seq_segments[seq + Int32(1)].to(Int32)
            if begin < end:
                allocator = cutlass.utils.SmemAllocator()
                smem = allocator.allocate_tensor(
                    element_type=BFloat16,
                    layout=cute.make_layout((128 * 128,), stride=(1,)), byte_alignment=128)
                smem_addr = shared_ptr_to_u32(smem.iterator)
                local_smem = allocator.allocate_tensor(
                    element_type=Float32,
                    layout=cute.make_layout((self.v_split * 128,), stride=(1,)), byte_alignment=128)
                local_addr = shared_ptr_to_u32(local_smem.iterator)
                barrier = allocator.allocate_tensor(
                    element_type=Int64, layout=cute.make_layout((1,), stride=(1,)), byte_alignment=8)
                finite_votes = allocator.allocate_tensor(
                    element_type=Int32, layout=cute.make_layout((self.mma_warps,), stride=(1,)))
                if thread == Int32(0):
                    cute.arch.mbarrier_init(barrier.iterator, Int32(1))
                cute.arch.mbarrier_init_fence()
                cute.arch.sync_threads()
                phase = Int32(0)
                acc = cute.make_rmem_tensor((16, 4), Float32)
                shadow = cute.make_rmem_tensor((8, 4), Uint32)
                for nb in cutlass.range_constexpr(16):
                    for item in cutlass.range_constexpr(4):
                        acc[nb, item] = Float32(0.0)
                source_slot = Int64(initial[seq])
                head_base = head.to(Int64) * Int64(128 * 128)
                pool_stride = Int64(self.heads * 128 * 128)
                if not self._is_null(source_slot):
                    self._load_state(source, acc, source_slot * source_stride + head_base,
                                     row0, row1, Int32(0), tid)
                fast_chain = cutlass.Boolean(True)
                for segment in cutlass.range(begin, end - Int32(1), unroll=1):
                    flag = transfer_flags[segment.to(Int64) * Int64(self.heads) + head.to(Int64)]
                    fast_chain = fast_chain & (flag == Int32(3))
                if fast_chain:
                    finite = cutlass.Boolean(True)
                    for nb in cutlass.range_constexpr(16):
                        for item in cutlass.range_constexpr(4):
                            finite = (finite & (acc[nb, item] > Float32(float("-inf")))
                                      & (acc[nb, item] < Float32(float("inf"))))
                    warp_finite = cute.arch.vote_all_sync(finite)
                    if lane == Int32(0):
                        finite_votes[warp] = warp_finite.to(Int32)
                    cute.arch.sync_threads()
                    for w in cutlass.range_constexpr(self.mma_warps):
                        fast_chain = fast_chain & (finite_votes[Int32(w)] != Int32(0))
                if fast_chain:
                    if target_segment > begin:
                        local_slot = Int64(1 + self.max_segments) + target_segment.to(Int64)
                        self._load_state(pool, acc, local_slot * pool_stride + head_base,
                                         row0, row1, Int32(0), tid)
                    output_slot = Int64(2 + 2 * self.max_segments) + target_segment.to(Int64)
                    self._store_state(pool, acc, output_slot * pool_stride + head_base,
                                      row0, row1, Int32(0), tid)
                elif target_segment == begin:
                    for segment in cutlass.range(begin, end, unroll=1):
                        transfer_zero = cutlass.Boolean(False)
                        if segment + Int32(1) < end:
                            transfer_zero = (transfer_flags[segment.to(Int64) * Int64(self.heads) + head.to(Int64)] & Int32(1)) != Int32(0)
                            if thread == Int32(0):
                                packed_base = segment.to(Int64) * pool_stride + head_base
                                local_base = (Int64(2 + self.max_segments) + segment.to(Int64)) * pool_stride + head_base
                                state_base = local_base + split.to(Int64) * Int64(self.v_split * 128)
                                bar_addr = shared_ptr_to_u32(barrier.iterator)
                                copy_bytes = Int32(self.v_split * 128 * 4)
                                if not transfer_zero:
                                    copy_bytes += Int32(128 * 128 * 2)
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    barrier.iterator, copy_bytes)
                                cute.arch.fence_proxy("async.global")
                                if not transfer_zero:
                                    cp_async_bulk_g2s_mbar(smem_addr, _pointer_address(packed, packed_base),
                                                          Int32(128 * 128 * 2), bar_addr)
                                cp_async_bulk_g2s_mbar(local_addr, _pointer_address(pool, state_base),
                                                      Int32(self.v_split * 128 * 4), bar_addr)
                        output_slot = Int64(2 + 2 * self.max_segments) + segment.to(Int64)
                        self._store_state(pool, acc, output_slot * pool_stride + head_base,
                                          row0, row1, Int32(0), tid)
                        if segment + Int32(1) < end:
                            self._refresh_shadow(acc, shadow)
                            skip_product = cutlass.Boolean(False)
                            if transfer_zero:
                                finite = cutlass.Boolean(True)
                                for kb in cutlass.range_constexpr(8):
                                    for item in cutlass.range_constexpr(4):
                                        bits = shadow[kb, item] & Uint32(0x7F807F80)
                                        finite = finite & ((bits & Uint32(0xFFFF)) != Uint32(0x7F80))
                                        finite = finite & ((bits >> Uint32(16)) != Uint32(0x7F80))
                                skip_product = cute.arch.vote_all_sync(finite)
                            transfer_base = (Int64(2) + segment.to(Int64)) * pool_stride + head_base
                            center = Float32(pool[transfer_base])
                            cute.arch.mbarrier_wait(barrier.iterator, phase=phase)
                            phase = phase ^ Int32(1)
                            for nb in cutlass.range_constexpr(16):
                                col = Int32(nb * 8) + tid * Int32(2)
                                base0 = (row0 - split * Int32(self.v_split)) * Int32(128) + col
                                base1 = (row1 - split * Int32(self.v_split)) * Int32(128) + col
                                acc[nb, 0] = acc[nb, 0] * center + Float32(local_smem[base0])
                                acc[nb, 1] = acc[nb, 1] * center + Float32(local_smem[base0 + Int32(1)])
                                acc[nb, 2] = acc[nb, 2] * center + Float32(local_smem[base1])
                                acc[nb, 3] = acc[nb, 3] * center + Float32(local_smem[base1 + Int32(1)])
                            if not skip_product:
                                for kb in cutlass.range_constexpr(8):
                                    krow = Int32(kb * 16) + (matrix & Int32(1)) * Int32(8) + matrix_row
                                    for pair in cutlass.range_constexpr(8):
                                        b0, b1, b2, b3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                                        if not transfer_zero:
                                            chunk = Int32(pair * 2) + (matrix >> Int32(1))
                                            physical = chunk ^ (krow & Int32(7))
                                            b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                                                smem_addr + krow * Int32(256) + physical * Int32(16))
                                        acc[2 * pair, 0], acc[2 * pair, 1], acc[2 * pair, 2], acc[2 * pair, 3] = bf16_mma_m16n8k16_f32(
                                            acc[2 * pair, 0], acc[2 * pair, 1], acc[2 * pair, 2], acc[2 * pair, 3],
                                            shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3], b0, b1)
                                        acc[2 * pair + 1, 0], acc[2 * pair + 1, 1], acc[2 * pair + 1, 2], acc[2 * pair + 1, 3] = bf16_mma_m16n8k16_f32(
                                            acc[2 * pair + 1, 0], acc[2 * pair + 1, 1], acc[2 * pair + 1, 2], acc[2 * pair + 1, 3],
                                            shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3], b2, b3)
                            cute.arch.sync_threads()


class _CommitKernel:
    def __init__(self, *, heads: int, max_seqs: int, max_segments: int,
                 null_state_index: int | None, reuse_outputs: bool) -> None:
        self.heads = heads
        self.max_seqs = max_seqs
        self.max_segments = max_segments
        self.checkpoint_base = 2 + (4 if reuse_outputs else 3) * max_segments
        self.has_null = null_state_index is not None
        self.null = 0 if null_state_index is None else null_state_index

    @cute.jit
    def _is_null(self, slot: Int64):
        result = slot != slot
        if cutlass.const_expr(self.has_null):
            result = slot == Int64(self.null)
        return result

    @cute.jit
    def __call__(self, original: tuple, segments: tuple, pool: cute.Pointer,
                 state: cute.Pointer, output: cute.Pointer, state_stride: Int64,
                 final_stride: Int64, output_stride: Int64, stream: cuda.CUstream):
        self.kernel(original, segments, pool, state, output, state_stride,
                    final_stride, output_stride).launch(
            grid=(self.heads, self.max_seqs, 16), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, original: tuple, segments: tuple, pool: cute.Pointer,
               state: cute.Pointer, output: cute.Pointer, state_stride: Int64,
               final_stride: Int64, output_stride: Int64):
        head, seq, tile = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        head, seq, tile, thread = Int32(head), Int32(seq), Int32(tile), Int32(thread)
        _, initial, final, checkpoint, offsets, num_seqs, _ = original
        if seq < num_seqs[Int32(0)]:
            seq_segments = segments[0]
            begin, end = seq_segments[seq].to(Int32), seq_segments[seq + Int32(1)].to(Int32)
            target = Int64(final[seq.to(Int64) * final_stride])
            source = Int64(initial[seq])
            cp = Int64(checkpoint[seq])
            export = offsets[seq] > Int32(0)
            pool_stride = Int64(self.heads * 128 * 128)
            head_base = head.to(Int64) * Int64(128 * 128)
            for step in cutlass.range_constexpr(4):
                element = tile.to(Int64) * Int64(1024) + thread.to(Int64) + Int64(step * 256)
                if not self._is_null(target):
                    value = Float32(0.0)
                    if begin < end:
                        slot = Int64(2 + 2 * self.max_segments) + (end - Int32(1)).to(Int64)
                        value = Float32(pool[slot * pool_stride + head_base + element])
                    elif not self._is_null(source):
                        value = Float32(state[source * state_stride + head_base + element])
                    state[target * state_stride + head_base + element] = value
                if export:
                    if not self._is_null(cp):
                        slot = Int64(self.checkpoint_base) + seq.to(Int64)
                        state[cp * state_stride + head_base + element] = Float32(pool[slot * pool_stride + head_base + element])


@dataclass(frozen=True)
class Auxiliary:
    partition: Callable
    pack_transfer: Callable
    boundaries: Callable
    commit: Callable


_CACHE: dict[tuple, Auxiliary] = {}
register_program_cache(_CACHE)


def _metadata(binding, *, fake=False):
    parallel = binding.parallel
    assert parallel is not None
    inner = parallel.output
    original = (binding.cu_seqlens, binding.initial_state_indices, binding.final_state_indices,
                binding.checkpoint_state_indices, binding.checkpoint_offsets,
                binding.num_seqs, binding.num_tokens)
    segments = (parallel.seq_segments, inner.cu_seqlens, parallel.local_state.initial_state_indices,
                parallel.transfer.initial_state_indices, parallel.transfer.final_state_indices,
                parallel.local_state.final_state_indices, inner.final_state_indices,
                inner.checkpoint_state_indices, inner.checkpoint_offsets,
                parallel.transfer.checkpoint_offsets, inner.num_seqs, inner.num_tokens)

    def pointers(tensors):
        return tuple(_fake_pointer(_numeric_type(t.dtype)) if fake
                     else _pointer(t, _numeric_type(t.dtype)) for t in tensors)

    return pointers(original), pointers(segments)


def compile_auxiliary(binding) -> Auxiliary:
    parallel = binding.parallel
    assert parallel is not None
    caps = binding._state.caps
    key = (binding.output.device.index, caps.heads, caps.max_seqs,
           parallel._state.max_segments, parallel._state.segment_tokens,
           binding._state.v_split, caps.null_state_index, binding.initial_state_indices.dtype,
           parallel._state.reuse_outputs)
    if key in _CACHE:
        return _CACHE[key]
    geometry = dict(heads=caps.heads, max_seqs=caps.max_seqs,
                    max_segments=parallel._state.max_segments)
    kernels = (
        _PartitionKernel(**geometry, segment_tokens=parallel._state.segment_tokens,
                         reuse_outputs=parallel._state.reuse_outputs),
        _PackTransferKernel(**geometry),
        _BoundaryKernel(**geometry, v_split=binding._state.v_split, null_state_index=caps.null_state_index),
        _CommitKernel(**geometry, null_state_index=caps.null_state_index,
                      reuse_outputs=parallel._state.reuse_outputs),
    )
    original, segments = _metadata(binding, fake=True)
    common = (original, segments, _fake_pointer(Float32))
    arguments = (
        common,
        (*common, _fake_pointer(BFloat16), _fake_pointer(Int32)),
        (*common, _fake_pointer(BFloat16), _fake_pointer(Int32), _fake_pointer(Float32), Int64(1)),
        (*common, _fake_pointer(Float32), _fake_pointer(BFloat16), Int64(1), Int64(1), Int64(1)),
    )
    compiled = []
    for name, kernel, args in zip(("partition", "pack_transfer", "boundaries", "commit"), kernels, arguments):
        raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
        compiled.append(b12x_compile(kernel, *args, current_cuda_stream(),
                        compile_spec=KernelCompileSpec.from_key(f"sequence.gdn_prefill.{name}", 2, key)))

    def args(active):
        metadata = _metadata(active)
        return (*metadata, _pointer(active.parallel.pool, Float32))

    def partition(active):
        compiled[0](*args(active), current_cuda_stream())

    def pack_transfer(active):
        compiled[1](*args(active), _pointer(active.parallel.packed_transfer, BFloat16),
                    _pointer(active.parallel.transfer_flags, Int32), current_cuda_stream())

    def boundaries(active):
        compiled[2](*args(active), _pointer(active.parallel.packed_transfer, BFloat16),
                    _pointer(active.parallel.transfer_flags, Int32),
                    _pointer(active.recurrent_state, Float32),
                    int(active.recurrent_state.stride(0)), current_cuda_stream())

    def commit(active):
        compiled[3](*args(active), _pointer(active.recurrent_state, Float32),
                    _pointer(active.output, BFloat16), int(active.recurrent_state.stride(0)),
                    int(active.final_state_indices.stride(0)), int(active.output.stride(0)),
                    current_cuda_stream())

    result = Auxiliary(partition, pack_transfer, boundaries, commit)
    attach_programs(result, *compiled)
    _CACHE[key] = result
    return result
