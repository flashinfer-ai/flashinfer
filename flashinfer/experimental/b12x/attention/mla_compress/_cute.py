"""Native projected CSA pair pooling, BF16-rounding and ordinary RMSNorm.

Math follows DeepSeek-V4.1-Flash inference/model.py:429-485 (snapshot
fb2764a5cf321eaa5070ca8f9e892818f477c16d). This is not the vLLM V4 C4
(overlap/APE) or C128 (paged ring-state) compressor: those fuse RoPE/cache
packing, whereas CSA2 must expose the unrotated BF16 latent for indexer K.
Reduction and pointer/compiler handling reuse b12x helpers.
"""
from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute
from b12x._lib.compiler import run_compiled
from b12x._lib.intrinsics import block_reduce, fmax_f32, warp_reduce
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


def _add(a, b):
    return a + b


class _Compress:
    def __init__(self, ratio: int, max_tokens: int, max_requests: int, max_states: int):
        self.ratio = ratio
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.max_states = max_states

    @cute.jit
    def __call__(self, values: cute.Pointer, gates: cute.Pointer,
                 weight: cute.Pointer, starts: cute.Pointer,
                 positions: cute.Pointer, state_ids: cute.Pointer,
                 slots: cute.Pointer, counts: cute.Pointer,
                 pending_values: cute.Pointer, pending_gates: cute.Pointer,
                 pending_position: cute.Pointer, out: cute.Pointer,
                 emitted: cute.Pointer, emitted_slots: cute.Pointer,
                 stream: cuda.CUstream):
        self.kernel(values, gates, weight, starts, positions, state_ids,
                    slots, counts, pending_values, pending_gates, pending_position,
                    out, emitted, emitted_slots).launch(
                        grid=(self.max_tokens, 1, 1), block=(128, 1, 1), stream=stream)
        if cutlass.const_expr(self.ratio == 2):
            # A launch boundary is required: the first token of a request may
            # read its old carry while its last token prepares the next carry.
            self.commit(values, gates, starts, positions, state_ids, counts,
                        pending_values, pending_gates, pending_position).launch(
                            grid=(self.max_requests, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, values: cute.Pointer, gates: cute.Pointer,
               weight: cute.Pointer, starts: cute.Pointer,
               positions: cute.Pointer, state_ids: cute.Pointer,
               slots: cute.Pointer, counts: cute.Pointer,
               pending_values: cute.Pointer, pending_gates: cute.Pointer,
               pending_position: cute.Pointer, out: cute.Pointer,
               emitted: cute.Pointer, emitted_slots: cute.Pointer):
        token, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        valid = Boolean(False)
        use_pending = Boolean(False)
        state_id = Int64(0)
        first = Int32(0)
        position = Int64(-1)
        nt = Int32(counts[0])
        nr = Int32(counts[1])
        if nt >= 0 and nt <= self.max_tokens and nr > 0 and nr <= self.max_requests:
            if token < nt:
                # upper_bound handles empty request rows without selecting one.
                low = Int32(0)
                high = nr
                while low < high:
                    mid = (low + high) // 2
                    if Int32(starts[mid + 1]) <= token:
                        low = mid + 1
                    else:
                        high = mid
                if low < nr:
                    first = Int32(starts[low])
                    end = Int32(starts[low + 1])
                    state_id = Int64(state_ids[low])
                    start_position = Int64(positions[low])
                    if first >= 0 and first <= token and end <= nt and token < end:
                        if state_id >= 0 and state_id < Int64(self.max_states) and start_position >= 0:
                            position = start_position + Int64(token - first)
                            valid = Boolean(True)
        if cutlass.const_expr(self.ratio == 2):
            valid = valid and position % Int64(2) == Int64(1)
            if valid and token == first:
                use_pending = Boolean(True)
                valid = Int64(pending_position[state_id]) == position - Int64(1)

        rounded = cute.make_rmem_tensor((4,), Float32)
        square_sum = Float32(0.0)
        for item in cutlass.range_constexpr(4):
            col = Int64(tid + item * 128)
            value = Float32(0.0)
            if valid:
                offset = Int64(token) * Int64(512) + col
                value = Float32(values[offset])
                if cutlass.const_expr(self.ratio == 2):
                    gate1 = Float32(gates[offset])
                    value0 = Float32(0.0)
                    gate0 = Float32(0.0)
                    if use_pending:
                        old_offset = state_id * Int64(512) + col
                        value0 = Float32(pending_values[old_offset])
                        gate0 = Float32(pending_gates[old_offset])
                    else:
                        value0 = Float32(values[offset - Int64(512)])
                        gate0 = Float32(gates[offset - Int64(512)])
                    maximum = fmax_f32(gate0, gate1)
                    e0 = cute.math.exp(gate0 - maximum, fastmath=True)
                    e1 = cute.math.exp(gate1 - maximum, fastmath=True)
                    denominator = e0 + e1
                    value = value0 * (e0 / denominator) + value * (e1 / denominator)
                # The reference rounds the pooled vector BEFORE computing RMS.
                value = Float32(BFloat16(value))
            rounded[item] = value
            square_sum += value * value
        allocator = cutlass.utils.SmemAllocator()
        reduction = allocator.allocate_tensor(
            Float32, cute.make_layout((1, 4)), byte_alignment=16)
        total = block_reduce(warp_reduce(square_sum, _add), _add, reduction, Float32(0.0))
        inverse = cute.math.rsqrt(total / Float32(512.0) + Float32(1e-20), fastmath=True)
        for item in cutlass.range_constexpr(4):
            col = Int64(tid + item * 128)
            normalized = Float32(0.0)
            if valid:
                normalized = rounded[item] * inverse * Float32(weight[col])
            out[Int64(token) * Int64(512) + col] = BFloat16(normalized)
        if tid == 0:
            emitted[token] = valid
            destination = Int64(-1)
            if valid:
                destination = Int64(slots[token])
            emitted_slots[token] = destination

    @cute.kernel
    def commit(self, values: cute.Pointer, gates: cute.Pointer,
               starts: cute.Pointer, positions: cute.Pointer,
               state_ids: cute.Pointer, counts: cute.Pointer,
               pending_values: cute.Pointer, pending_gates: cute.Pointer,
               pending_position: cute.Pointer):
        request, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        nt = Int32(counts[0])
        nr = Int32(counts[1])
        if nr > 0 and nr <= self.max_requests and nt >= 0 and nt <= self.max_tokens:
            if request < nr:
                first = Int32(starts[request])
                end = Int32(starts[request + 1])
                sid = Int64(state_ids[request])
                start_position = Int64(positions[request])
                if first >= 0 and end > first and end <= nt:
                    if sid >= 0 and sid < Int64(self.max_states) and start_position >= 0:
                        last_position = start_position + Int64(end - first - 1)
                        tag = Int64(-1)
                        if last_position % Int64(2) == 0:
                            tag = last_position
                            for item in cutlass.range_constexpr(4):
                                col = Int64(tid + item * 128)
                                src = Int64(end - 1) * Int64(512) + col
                                dst = sid * Int64(512) + col
                                pending_values[dst] = values[src]
                                pending_gates[dst] = gates[src]
                        if tid == 0:
                            pending_position[sid] = tag


_DTYPES = (Float32, Float32, Float32, Int32, Int64, Int64, Int64,
           Int32, Float32, Float32, Int64, BFloat16, Boolean, Int64)


def pointers(tensors):
    dtypes = {torch.float32: Float32, torch.bfloat16: BFloat16,
              torch.int32: Int32, torch.int64: Int64, torch.bool: Boolean}
    return tuple(make_ptr(dtypes[t.dtype], t.data_ptr(), cute.AddressSpace.gmem,
                          assumed_align=t.element_size()) for t in tensors)


@lru_cache(maxsize=None)
def compile_compress(ratio, max_tokens, max_requests, max_states, device_index):
    key = (ratio, max_tokens, max_requests, max_states, device_index)
    entry = _Compress(ratio, max_tokens, max_requests, max_states)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    types = list(_DTYPES)
    if ratio == 1:
        types[0] = BFloat16
    fake = tuple(make_ptr(dtype, 16, cute.AddressSpace.gmem,
                          assumed_align=max(1, dtype.width // 8)) for dtype in types)
    with torch.cuda.device(device_index):
        return compile_cute(entry, *fake, current_cuda_stream(),
                            compile_spec=KernelCompileSpec.from_key(
                                "attention.mla_compress.cute", 1, key))


def launch(binding):
 with torch.cuda.device(binding._state.caps.device): run_compiled(binding._state.compiled,(*binding._pointers,current_cuda_stream()))
