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
and clear the combined BF16 output in one launch. Planning compiles and binds
caller-owned buffers outside capture; execution uses the supplied CUDA stream.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch

from ...cute_dsl.utils import make_ptr


_route_preprocess_kernel_cache = {}


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


class _RoutePreprocessPlan:
    """Fixed-address launcher. One plan per concurrent mutable buffer set."""

    def __init__(
        self, compiled, arguments, owners, route_ids, route_weights, output, mode
    ):
        self._compiled = compiled
        self._arguments = arguments
        self._owners = owners
        self.route_ids = route_ids
        self.route_weights = route_weights
        self.output = output
        self.device = output.device
        self.mode = mode

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
):
    """Compile, bind and enqueue one warmup, outside CUDA Graph capture.

    ``topk_weights=None`` means packed int32 IDs/high16 + BF16 weights/low16.
    Packed input requires caller-owned contiguous int32 ``route_ids`` and FP32
    ``route_weights`` outputs. Separate BF16 weights require ``route_weights``;
    separate IDs are bound directly. Separate FP32 weights are bound directly,
    and the only GPU work is output clearing. Do not pass different destination
    buffers for either direct-bind input. All tensors must be on one device.

    Inputs may have independent nonnegative 2-D strides. Output and conversion
    buffers are contiguous. This kernel supports tokens from 1..16, top-k from
    1..32, and a positive even hidden size. No input data are read on
    the host. Tensor contents must be valid when planning because warmup runs.
    Shapes/strides are dynamic kernel arguments; beta values and tactic choices
    do not enter compilation. Cache entries are per mode, threads and device.
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
        arguments = (
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
            tokens,
            top_k,
            hidden_size,
            *topk_ids.stride(),
            *weight_strides,
        )
        cache_key = (mode, threads, device.index, arch)
        compiled = _route_preprocess_kernel_cache.get(cache_key)
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        if compiled is None:
            compiled = cute.compile(
                _RoutePreprocess(mode, threads), *arguments, stream=stream
            )
            _route_preprocess_kernel_cache[cache_key] = compiled
        bound = _RoutePreprocessPlan(
            compiled,
            arguments,
            (topk_ids, topk_weights, route_ids, route_weights, output),
            route_ids,
            route_weights,
            output,
            mode,
        )
        bound.run(stream)
        return bound
