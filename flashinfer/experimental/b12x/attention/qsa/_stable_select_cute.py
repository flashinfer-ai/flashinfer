"""Construct exact QSA winners from a radix threshold in one CUDA block."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import compile_only_launches_enabled, program_keys, record_program
from b12x._lib.program_cache import register_program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

_CACHE = {}
register_program_cache(_CACHE)
_TYPES = (Float32, Int32, Int32, Int32, Float32, Float32, Int32)


class StableSelectionKernel:
    def __init__(self, budget):
        self.budget = int(budget)

    @cute.jit
    def __call__(self, pointers: tuple, stride: Int64, rows: Int32,
                 group_offset: Int32, stream: cuda.CUstream):
        self.kernel(pointers, stride, group_offset).launch(
            grid=(rows, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, pointers: tuple, stride: Int64, group_offset: Int32):
        scores, lengths, prior_ids, eligible, top_values, values, ids = pointers
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        base = row.to(Int64) * Int64(self.budget)
        score_base = row.to(Int64) * stride
        length = lengths[row].to(Int32)
        selected = cutlass.min(length, Int32(self.budget))
        allocator = cutlass.utils.SmemAllocator()
        minima = allocator.allocate_tensor(Float32, cute.make_layout((8,)), byte_alignment=16)
        greater_counts = allocator.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        tie_counts = allocator.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        threshold = allocator.allocate_tensor(Float32, cute.make_layout((1,)), byte_alignment=4)
        minimum = Float32(float('inf'))
        for part in cutlass.range_constexpr(self.budget // 256):
            column = tid + Int32(part * 256)
            if column < selected:
                minimum = cutlass.min(minimum, Float32(top_values[base + column.to(Int64)]))
            values[base + column.to(Int64)] = Float32(-float('inf'))
            ids[base + column.to(Int64)] = Int32(-1)
        for shift in cutlass.range_constexpr(5):
            minimum = cutlass.min(minimum, cute.arch.shuffle_sync_bfly(minimum, offset=16 >> shift))
        if lane == Int32(0):
            minima[warp] = minimum
        cute.arch.sync_threads()
        if tid == Int32(0):
            minimum = Float32(float('inf'))
            for w in cutlass.range_constexpr(8):
                minimum = cutlass.min(minimum, Float32(minima[w]))
            threshold[0] = minimum
        cute.arch.sync_threads()
        cutoff = Float32(threshold[0])

        # Each warp owns a contiguous span so threshold ties retain earlier
        # input positions while both score scans use coalesced memory loads.
        tiles = (length + Int32(255)) // Int32(256)
        span = tiles * Int32(32)
        start = warp * span
        ng, nt = Int32(0), Int32(0)
        for tile in cutlass.range(tiles):
            column = start + tile * Int32(32) + lane
            if column < length:
                value = Float32(scores[score_base + column.to(Int64)])
                if value > cutoff:
                    ng += Int32(1)
                if value == cutoff:
                    nt += Int32(1)
        for shift in cutlass.range_constexpr(5):
            ng += cute.arch.shuffle_sync_bfly(ng, offset=16 >> shift)
            nt += cute.arch.shuffle_sync_bfly(nt, offset=16 >> shift)
        if lane == Int32(0):
            greater_counts[warp], tie_counts[warp] = ng, nt
        cute.arch.sync_threads()
        greater_before, ties_before, total_greater = Int32(0), Int32(0), Int32(0)
        for w in cutlass.range_constexpr(8):
            count = Int32(greater_counts[w])
            total_greater += count
            if Int32(w) < warp:
                greater_before += count
                ties_before += Int32(tie_counts[w])
        need = selected - total_greater
        carry = cutlass.min(cutlass.min(eligible[row].to(Int32), group_offset), Int32(self.budget))
        for tile in cutlass.range(tiles):
            column = start + tile * Int32(32) + lane
            value = Float32(-float('inf'))
            is_greater, is_tie = Int32(0), Int32(0)
            if column < length:
                value = Float32(scores[score_base + column.to(Int64)])
                if value > cutoff:
                    is_greater = Int32(1)
                if value == cutoff:
                    is_tie = Int32(1)
            gp, tp = is_greater, is_tie
            for shift in cutlass.range_constexpr(5):
                distance = Int32(1 << shift)
                source_lane = cutlass.max(lane - distance, Int32(0))
                g = cute.arch.shuffle_sync(gp, source_lane)
                t = cute.arch.shuffle_sync(tp, source_lane)
                if lane >= distance:
                    gp += g
                    tp += t
            tie_rank = ties_before + tp - Int32(1)
            chosen = (is_greater != Int32(0)) | ((is_tie != Int32(0)) & (tie_rank < need))
            if chosen:
                destination = greater_before + gp - Int32(1)
                if is_tie != Int32(0):
                    destination = total_greater + tie_rank
                global_id = group_offset + column - carry
                if column < carry:
                    global_id = prior_ids[base + column.to(Int64)].to(Int32)
                values[base + destination.to(Int64)] = value
                ids[base + destination.to(Int64)] = global_id
            greater_before += cute.arch.shuffle_sync(gp, Int32(31))
            ties_before += cute.arch.shuffle_sync(tp, Int32(31))


def compile_stable_selection(budget, device):
    key = (device, int(budget))
    raw = _CACHE.get(key)
    if raw is None:
        kernel = StableSelectionKernel(budget)
        raise_if_kernel_resolution_frozen('cute.compile', target=kernel, cache_key=key)
        pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8) for t in _TYPES)
        raw = b12x_compile(kernel, pointers, Int64(1), Int32(1), Int32(0), current_cuda_stream(),
                          compile_spec=KernelCompileSpec.from_key('attention.qsa.stable_selection', 1, key))
        _CACHE[key] = raw
    return raw


def launch_stable_selection(*, scores, merge_lengths, prior_ids, eligible_counts,
                            topk_values, stable_values, stable_ids, group_offset,
                            group_budget, prepared=None):
    device = scores.device.index
    if device is None:
        device = torch.cuda.current_device()
    with torch.cuda.device(device):
        raw = prepared if prepared is not None else compile_stable_selection(group_budget, device)
        if compile_only_launches_enabled():
            for key in program_keys(raw):
                record_program(key)
            return raw
        tensors = (scores, merge_lengths, prior_ids, eligible_counts, topk_values, stable_values, stable_ids)
        pointers = tuple(make_ptr(t, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=t.width // 8)
                         for t, tensor in zip(_TYPES, tensors, strict=True))
        raw(pointers, Int64(scores.stride(0)), Int32(scores.shape[0]), Int32(group_offset), current_cuda_stream())
        return raw
