# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused MXFP4 MoE decode preprocessing; minimum architecture SM100 (B300: SM103).

For T=1..16, unpack global expert IDs, convert BF16 router weights to FP32,
optionally sort assignments into compact expert tiles, and clear the combined
BF16 output in one launch. Planning compiles and binds caller-owned buffers
outside capture; execution uses the supplied CUDA stream.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cuda.bindings.driver as cuda
import torch

from ...cute_dsl.utils import make_ptr
from .moe_utils import get_max_num_tiles


_route_preprocess_kernel_cache = {}

_SORT_BUFFER_NAMES = (
    "out_tile_idx_to_expert_idx",
    "out_tile_idx_to_mn_limit",
    "out_expanded_idx_to_permuted_idx",
    "out_permuted_idx_to_expanded_idx",
    "out_total_num_padded_tokens",
    "out_num_non_exiting_tiles",
)


class _RoutePreprocess:
    def __init__(self, mode, threads):
        self.packed = mode == "packed"
        self.convert_weights = mode != "separate_fp32"
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        ids_src_ptr: cute.Pointer,
        weights_src_ptr: cute.Pointer,
        ids_dst_ptr: cute.Pointer,
        weights_dst_ptr: cute.Pointer,
        output_words_ptr: cute.Pointer,
        tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        hidden_size: cutlass.Int32,
        ids_row_stride: cutlass.Int32,
        ids_col_stride: cutlass.Int32,
        weights_row_stride: cutlass.Int32,
        weights_col_stride: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        ids_src = cute.make_tensor(
            ids_src_ptr,
            cute.make_layout((tokens, top_k), stride=(ids_row_stride, ids_col_stride)),
        )
        weights_src = cute.make_tensor(
            weights_src_ptr,
            cute.make_layout(
                (tokens, top_k), stride=(weights_row_stride, weights_col_stride)
            ),
        )
        ids_dst = cute.make_tensor(ids_dst_ptr, cute.make_layout((tokens * top_k,)))
        weights_dst = cute.make_tensor(
            weights_dst_ptr, cute.make_layout((tokens * top_k,))
        )
        output_words = cute.make_tensor(
            output_words_ptr, cute.make_layout((tokens * (hidden_size // 2),))
        )
        # Size the grid for output clearing. The separate route grid-stride
        # loop remains correct even when its domain is larger than this one.
        tasks = tokens * (hidden_size // 2)
        self.kernel(ids_src, weights_src, ids_dst, weights_dst, output_words).launch(
            grid=(cute.ceil_div(tasks, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ids_src: cute.Tensor,
        weights_src: cute.Tensor,
        ids_dst: cute.Tensor,
        weights_dst: cute.Tensor,
        output_words: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        first = bidx * self.threads + tidx
        task_stride = grid_x * self.threads

        if cutlass.const_expr(self.convert_weights):
            # Route and clear loops have separate limits. Do not use the
            # route-count bound/stride to cover the much larger token output.
            for route in cutlass.range(first, cute.size(weights_dst), task_stride):
                token = route // ids_src.shape[1]
                slot = route % ids_src.shape[1]
                if cutlass.const_expr(self.packed):
                    # Signed int32 shift matches torch.bitwise_right_shift.
                    ids_dst[route] = ids_src[(token, slot)] >> 16
                # Packed mode binds a BF16 view of each int32's low half, with
                # doubled element strides. This is a bit-preserving load, not
                # a numeric conversion of the packed integer itself.
                weights_dst[route] = weights_src[(token, slot)].to(cutlass.Float32)

        for word in cutlass.range(first, cute.size(output_words), task_stride):
            # One aligned u32 store writes two BF16 +0 values.
            output_words[word] = cutlass.Uint32(0)


@dsl_user_op
def _routing_shared_add(
    address: cutlass.Int32, value: cutlass.Int32, *, loc=None, ip=None
):
    # Same narrow primitive used by FlashInfer's existing shared histograms.
    # The address is already in the 32-bit shared-memory address space.
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(address).ir_value(loc=loc, ip=ip),
                cutlass.Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared.add.s32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _routing_warp_inclusive(value: cutlass.Int32, lane: cutlass.Int32):
    for step in cutlass.range_constexpr(5):
        offset = 1 << step
        other = cute.arch.shuffle_sync_up(value, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            value += other
    return value


class _FusedRoutePreprocess:
    def __init__(self, mode, threads, single_tile_per_expert=False):
        self.single_tile_per_expert = single_tile_per_expert
        self.packed = mode == "packed"
        self.convert_weights = mode != "separate_fp32"
        self.threads = threads
        self.warps = threads // 32

    @cute.jit
    def __call__(
        self,
        ids_src_ptr: cute.Pointer,
        weights_src_ptr: cute.Pointer,
        ids_dst_ptr: cute.Pointer,
        weights_dst_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        tile_expert_ptr: cute.Pointer,
        tile_limit_ptr: cute.Pointer,
        expanded_ptr: cute.Pointer,
        permuted_ptr: cute.Pointer,
        padded_total_ptr: cute.Pointer,
        active_total_ptr: cute.Pointer,
        tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        hidden: cutlass.Int32,
        num_experts: cutlass.Int32,
        local_experts: cutlass.Int32,
        local_offset: cutlass.Int32,
        tile_size: cutlass.Int32,
        tile_capacity: cutlass.Int32,
        ids_row_stride: cutlass.Int32,
        ids_col_stride: cutlass.Int32,
        weights_row_stride: cutlass.Int32,
        weights_col_stride: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        ids_src = cute.make_tensor(
            ids_src_ptr,
            cute.make_layout((tokens, top_k), stride=(ids_row_stride, ids_col_stride)),
        )
        weights_src = cute.make_tensor(
            weights_src_ptr,
            cute.make_layout(
                (tokens, top_k), stride=(weights_row_stride, weights_col_stride)
            ),
        )
        ids_dst = cute.make_tensor(ids_dst_ptr, cute.make_layout((tokens * top_k,)))
        weights_dst = cute.make_tensor(
            weights_dst_ptr, cute.make_layout((tokens * top_k,))
        )
        output = cute.make_tensor(
            output_ptr, cute.make_layout((tokens * (hidden // 2),))
        )
        tile_expert = cute.make_tensor(
            tile_expert_ptr, cute.make_layout((tile_capacity,))
        )
        tile_limit = cute.make_tensor(
            tile_limit_ptr, cute.make_layout((tile_capacity,))
        )
        expanded = cute.make_tensor(expanded_ptr, cute.make_layout((tokens * top_k,)))
        permuted = cute.make_tensor(
            permuted_ptr, cute.make_layout((tile_capacity * tile_size,))
        )
        padded_total = cute.make_tensor(padded_total_ptr, cute.make_layout((1,)))
        active_total = cute.make_tensor(active_total_ptr, cute.make_layout((1,)))
        self.kernel(
            ids_src,
            weights_src,
            ids_dst,
            weights_dst,
            output,
            tile_expert,
            tile_limit,
            expanded,
            permuted,
            padded_total,
            active_total,
            num_experts,
            local_experts,
            local_offset,
            tile_size,
        ).launch(
            grid=(cute.ceil_div(cute.size(output), self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ids_src: cute.Tensor,
        weights_src: cute.Tensor,
        ids_dst: cute.Tensor,
        weights_dst: cute.Tensor,
        output: cute.Tensor,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        expanded: cute.Tensor,
        permuted: cute.Tensor,
        padded_total: cute.Tensor,
        active_total: cute.Tensor,
        num_experts: cutlass.Int32,
        local_experts: cutlass.Int32,
        local_offset: cutlass.Int32,
        tile_size: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        smem = utils.SmemAllocator()
        counts = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.threads,)))
        bases = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.threads,)))
        warp_sums = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.warps,)))

        # The branch is uniform across the entire CTA. Each barrier below has
        # all threads of CTA0 participating; no named or cluster barriers.
        if block == 0:
            counts[tid] = cutlass.Int32(0)
            cute.arch.sync_threads()
            local = cutlass.Int32(-1)
            rank = cutlass.Int32(-1)
            if tid < cute.size(expanded):
                token = tid // ids_src.shape[1]
                slot = tid % ids_src.shape[1]
                expert = ids_src[(token, slot)]
                if cutlass.const_expr(self.packed):
                    expert = expert >> 16
                    ids_dst[tid] = expert
                if cutlass.const_expr(self.convert_weights):
                    weights_dst[tid] = weights_src[(token, slot)].to(cutlass.Float32)
                local = expert - local_offset
                expanded[tid] = cutlass.Int32(-1)
                if (
                    (expert >= 0)
                    & (expert < num_experts)
                    & (local >= 0)
                    & (local < local_experts)
                ):
                    address = (counts.iterator + local).toint().to(cutlass.Int32)
                    rank = _routing_shared_add(address, cutlass.Int32(1))
            cute.arch.sync_threads()

            count = counts[tid]
            if cutlass.const_expr(self.single_tile_per_expert):
                # Unique IDs give at most T rows per expert, and planning
                # requires tile_size >= T. Every expert has zero or one tile.
                ntiles = (count > 0).to(cutlass.Int32)
                active_mask = cute.arch.vote_ballot_sync(count > 0)
                inclusive_mask = cutlass.Uint32(0xFFFFFFFF) >> (31 - lane)
                within_warp = cute.arch.popc(active_mask & inclusive_mask)
            else:
                ntiles = (count + tile_size - 1) // tile_size
                within_warp = _routing_warp_inclusive(ntiles, lane)
            if lane == 31:
                warp_sums[warp] = within_warp
            cute.arch.sync_threads()
            if warp == 0:
                subtotal = cutlass.Int32(0)
                if lane < self.warps:
                    subtotal = warp_sums[lane]
                subtotal = _routing_warp_inclusive(subtotal, lane)
                if lane < self.warps:
                    warp_sums[lane] = subtotal
            cute.arch.sync_threads()
            preceding_warps = cutlass.Int32(0)
            if warp > 0:
                preceding_warps = warp_sums[warp - 1]
            tile_base = preceding_warps + within_warp - ntiles
            row_base = tile_base * tile_size
            bases[tid] = row_base
            if tid < local_experts:
                for j in cutlass.range(ntiles):
                    tile = tile_base + j
                    limit = (tile + 1) * tile_size
                    if limit > row_base + count:
                        limit = row_base + count
                    tile_expert[tile] = tid
                    tile_limit[tile] = limit
            if tid == self.threads - 1:
                active = preceding_warps + within_warp
                active_total[0] = active
                padded_total[0] = active * tile_size
            cute.arch.sync_threads()
            if rank >= 0:
                row = bases[local] + rank
                expanded[tid] = row
                permuted[row] = tid

        # Each word is covered once, independently of the assignment count.
        word = block * self.threads + tid
        if word < cute.size(output):
            output[word] = cutlass.Uint32(0)


class _RoutePreprocessPlan:
    """Fixed-address launcher. One plan per concurrent mutable buffer set."""

    def __init__(
        self,
        compiled,
        arguments,
        owners,
        route_ids,
        route_weights,
        output,
        mode,
        sorts_tokens=False,
    ):
        self._compiled = compiled
        self._arguments = arguments
        self._owners = owners
        self.route_ids = route_ids
        self.route_weights = route_weights
        self.output = output
        self.device = output.device
        self.mode = mode
        self.sorts_tokens = sorts_tokens

    def run(self, stream):
        """Enqueue one kernel; no buffer allocation, JIT or host synchronization."""
        self._compiled(*self._arguments, stream=stream)
        return self.output


def _check_tensor(name, tensor, shape, dtype, device, *, contiguous=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must be {dtype} {shape} on {device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} must have nonnegative strides")


def _interval(tensor):
    span = 1 + sum(
        (dim - 1) * stride
        for dim, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.data_ptr(), tensor.data_ptr() + span * tensor.element_size()


def _disjoint(left, right):
    a, b = _interval(left), _interval(right)
    return max(a[0], b[0]) >= min(a[1], b[1])


def _plan_route_preprocess(
    topk_ids,
    topk_weights,
    *,
    route_ids=None,
    route_weights=None,
    output,
    threads=256,
    moe_sort_buffers=None,
    num_experts=None,
    num_local_experts=None,
    local_expert_offset=0,
    tile_size=128,
    _single_tile_per_expert=False,
):
    """Compile, bind and enqueue one warmup, outside CUDA Graph capture.

    ``topk_weights=None`` means packed int32 IDs/high16 + BF16 weights/low16.
    Packed input requires caller-owned contiguous int32 ``route_ids`` and FP32
    ``route_weights`` outputs. Separate BF16 weights require ``route_weights``;
    separate IDs are bound directly. Separate FP32 weights are bound directly,
    and require no conversion. Do not pass different destination buffers for
    either direct-bind input. All tensors must be on one device.

    With ``moe_sort_buffers``, CTA0 additionally builds a local expert histogram
    and writes the six existing ``moe_sort`` outputs. Empty experts have no
    tiles; padding inside nonempty expert tiles is unchanged. Expert IDs must
    be unique within each token, as in standard top-k routing. This mode is
    used only when PDL is disabled; later GEMMs wait for the complete kernel.

    Inputs may have independent nonnegative 2-D strides. Output and conversion
    buffers are contiguous. This kernel supports tokens from 1..16, top-k from
    1..32, and a positive even hidden size. No input data are read on
    the host. Tensor contents must be valid when planning because warmup runs.
    Shapes/strides are dynamic kernel arguments; beta values and tactic choices
    do not enter compilation. Cache entries are per input/sort mode, thread
    count, device and single-tile prefix mode. Sorting uses 256..1024 threads,
    selected during planning. The private single-tile option relies on the
    unique-ID contract and falls back when tile_size < tokens.
    """
    if (
        not isinstance(output, torch.Tensor)
        or output.device.type != "cuda"
        or output.ndim != 2
    ):
        raise ValueError("output must be a 2-D CUDA tensor")
    tokens, hidden_size = output.shape
    if not isinstance(topk_ids, torch.Tensor) or topk_ids.ndim != 2:
        raise ValueError("topk_ids must be a 2-D tensor")
    top_k = topk_ids.shape[1]
    if (
        not 1 <= tokens <= 16
        or not 1 <= top_k <= 32
        or hidden_size <= 0
        or hidden_size % 2
    ):
        raise ValueError("require T=1..16, top_k=1..32 and positive even hidden_size")
    if threads not in (128, 256):
        raise ValueError("route preprocessing supports 128 or 256 threads per block")
    device = output.device
    shape = (tokens, top_k)
    _check_tensor(
        "output", output, (tokens, hidden_size), torch.bfloat16, device, contiguous=True
    )
    if output.data_ptr() % 4:
        raise ValueError(
            "output must be aligned to 4 bytes for paired BF16 zero stores"
        )
    _check_tensor("topk_ids", topk_ids, shape, torch.int32, device)
    writable = [("output", output)]
    inputs = [("topk_ids", topk_ids)]
    sorts_tokens = moe_sort_buffers is not None
    if sorts_tokens:
        if not (
            isinstance(num_experts, int)
            and isinstance(num_local_experts, int)
            and top_k <= num_experts <= 1024
            and num_local_experts > 0
            and local_expert_offset >= 0
            and local_expert_offset + num_local_experts <= num_experts
            and tile_size > 0
        ):
            raise ValueError("invalid fused decode sorting geometry")
        tile_capacity = get_max_num_tiles(tokens, top_k, num_local_experts, tile_size)
        shapes = (
            (tile_capacity,),
            (tile_capacity,),
            shape,
            (tile_capacity * tile_size,),
            (1,),
            (1,),
        )
        for name, expected in zip(_SORT_BUFFER_NAMES, shapes, strict=True):
            tensor = moe_sort_buffers[name]
            _check_tensor(name, tensor, expected, torch.int32, device, contiguous=True)
            writable.append((name, tensor))
        required = max(256, tokens * top_k, num_local_experts)
        threads = 1 << (required - 1).bit_length()

    if topk_weights is None:
        mode = "packed"
        _check_tensor(
            "route_ids", route_ids, shape, torch.int32, device, contiguous=True
        )
        _check_tensor(
            "route_weights",
            route_weights,
            shape,
            torch.float32,
            device,
            contiguous=True,
        )
        writable.extend((("route_ids", route_ids), ("route_weights", route_weights)))
        weight_source = topk_ids
        weights_dtype = cutlass.BFloat16
        weight_strides = tuple(2 * value for value in topk_ids.stride())
    else:
        if not isinstance(topk_weights, torch.Tensor) or topk_weights.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                "topk_weights must be BF16 or FP32, or None for packed input"
            )
        _check_tensor("topk_weights", topk_weights, shape, topk_weights.dtype, device)
        inputs.append(("topk_weights", topk_weights))
        if route_ids is not None and route_ids is not topk_ids:
            raise ValueError(
                "separate IDs are bound directly; pass route_ids=None or topk_ids"
            )
        route_ids = topk_ids
        weight_source = topk_weights
        weight_strides = topk_weights.stride()
        if topk_weights.dtype == torch.bfloat16:
            mode = "separate_bf16"
            weights_dtype = cutlass.BFloat16
            _check_tensor(
                "route_weights",
                route_weights,
                shape,
                torch.float32,
                device,
                contiguous=True,
            )
            writable.append(("route_weights", route_weights))
        else:
            mode = "separate_fp32"
            weights_dtype = cutlass.Float32
            if route_weights is not None and route_weights is not topk_weights:
                raise ValueError(
                    "FP32 weights are bound directly; pass route_weights=None or topk_weights"
                )
            route_weights = topk_weights

    for index, (name, tensor) in enumerate(writable):
        for other_name, other in inputs + writable[:index]:
            if not _disjoint(tensor, other):
                raise ValueError(f"{name} must not overlap {other_name}")

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "route preprocessing must be planned outside graph capture"
            )
        arch = torch.cuda.get_device_capability(device)
        if arch not in ((10, 0), (10, 3)):
            raise RuntimeError("route preprocessing requires SM100/SM103")
        pointers = (
            make_ptr(
                cutlass.Int32,
                topk_ids.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                weights_dtype,
                weight_source.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=2 if weights_dtype == cutlass.BFloat16 else 4,
            ),
            make_ptr(
                cutlass.Int32,
                route_ids.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Float32,
                route_weights.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Uint32,
                output.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
        )
        if sorts_tokens:
            pointers += tuple(
                make_ptr(
                    cutlass.Int32,
                    moe_sort_buffers[name].data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for name in _SORT_BUFFER_NAMES
            )
            arguments = pointers + (
                tokens,
                top_k,
                hidden_size,
                num_experts,
                num_local_experts,
                local_expert_offset,
                tile_size,
                tile_capacity,
                *topk_ids.stride(),
                *weight_strides,
            )
        else:
            arguments = pointers + (
                tokens,
                top_k,
                hidden_size,
                *topk_ids.stride(),
                *weight_strides,
            )
        single_tile_per_expert = (
            _single_tile_per_expert and sorts_tokens and tile_size >= tokens
        )
        cache_key = (
            mode, threads, device.index, arch, sorts_tokens, single_tile_per_expert
        )
        compiled = _route_preprocess_kernel_cache.get(cache_key)
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        if compiled is None:
            kernel = (
                _FusedRoutePreprocess(mode, threads, single_tile_per_expert)
                if sorts_tokens else _RoutePreprocess(mode, threads)
            )
            compiled = cute.compile(kernel, *arguments, stream=stream)
            _route_preprocess_kernel_cache[cache_key] = compiled
        bound = _RoutePreprocessPlan(
            compiled,
            arguments,
            (
                topk_ids,
                topk_weights,
                route_ids,
                route_weights,
                output,
                moe_sort_buffers,
            ),
            route_ids,
            route_weights,
            output,
            mode,
            sorts_tokens,
        )
        bound.run(stream)
        return bound
