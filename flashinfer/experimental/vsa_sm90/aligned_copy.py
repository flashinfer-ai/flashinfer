"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Bit-preserving CuTe copies for contiguous, naturally aligned BF16 tensors.

Torch is used for allocation, metadata-only views, and stream bookkeeping.
Every device copy is performed by the scalar 16-bit CuTe kernel below.
"""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor


@cute.kernel
def _copy_bits(source: cute.Tensor, destination: cute.Tensor, step: cutlass.Int64):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    index = cutlass.Int64(block) * 256 + thread
    while index < source.shape[0]:
        destination[index] = source[index]
        index += step


@cute.jit
def _launch_copy(
    source: cute.Tensor,
    destination: cute.Tensor,
    blocks: cutlass.Int32,
    stream: cuda.CUstream,
):
    _copy_bits(source, destination, cutlass.Int64(blocks) * 256).launch(
        grid=(blocks, 1, 1), block=(256, 1, 1), stream=stream
    )


_compiled = {}


def _validate_tensor(tensor):
    if tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise TypeError("expected a CUDA BF16 tensor")
    if not tensor.is_contiguous():
        raise ValueError("expected a contiguous tensor")
    if tensor.data_ptr() % 2:
        raise ValueError("expected natural two-byte BF16 alignment")


def _validate_alignment(alignment):
    if not isinstance(alignment, int) or alignment < 2 or alignment & (alignment - 1):
        raise ValueError("alignment must be a power of two of at least two bytes")


def copy_bf16(source, destination, caller_stream):
    """Copy equal element counts on an explicit torch.cuda.Stream.

    Shape may differ, but both tensors must be contiguous CUDA BF16 tensors on
    one device. Empty tensors and identical views require no launch. Partially
    overlapping views are rejected; this operation does not implement memmove.
    Only compiled code is cached; no tensor values or private buffers are cached.
    """
    _validate_tensor(source)
    _validate_tensor(destination)
    if source.device != destination.device or source.numel() != destination.numel():
        raise ValueError("copy requires one device and equal element counts")
    if caller_stream.device != source.device:
        raise ValueError("caller stream must belong to the tensor device")
    count = source.numel()
    src_ptr, dst_ptr = source.data_ptr(), destination.data_ptr()
    if count == 0 or src_ptr == dst_ptr:
        return
    if src_ptr < dst_ptr + 2 * count and dst_ptr < src_ptr + 2 * count:
        raise ValueError("partially overlapping views are not supported")
    # Reinterpretation keeps every BF16 bit, including signed zero and NaNs.
    src_bits = source.view(-1).view(torch.int16)
    dst_bits = destination.view(-1).view(torch.int16)
    blocks = min((count + 255) // 256, 65535)
    stream = cuda.CUstream(caller_stream.cuda_stream)
    key = source.device.index
    with torch.cuda.device(source.device):
        if key not in _compiled:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up the VSA alignment fallback before graph capture"
                )
            # A small first input must not specialize the runtime shape ABI to
            # int32: later contiguous BF16 tensors may exceed 2**31 elements.
            extent = cute.sym_int64()
            views = [
                make_fake_tensor(
                    cutlass.Int16, shape=(extent,), stride=(1,), assumed_align=2
                )
                for _ in range(2)
            ]
            _compiled[key] = cute.compile(
                _launch_copy,
                *views,
                cutlass.Int32(blocks),
                stream,
                options="--enable-tvm-ffi",
            )
        source.record_stream(caller_stream)
        destination.record_stream(caller_stream)
        # CuTe's device tensor-shape lowering can still narrow a huge extent
        # even when the host FFI accepts int64. Keep each launch below 1 GiB;
        # slicing changes metadata only, and ordinary copies use one launch.
        chunk_elements = 1 << 29
        for start in range(0, count, chunk_elements):
            end = min(start + chunk_elements, count)
            chunk_blocks = min((end - start + 255) // 256, 65535)
            _compiled[key](
                src_bits[start:end], dst_bits[start:end], chunk_blocks, stream
            )


def aligned_empty_like(tensor, alignment=16):
    """Allocate an uninitialized contiguous BF16 tensor with requested alignment."""
    _validate_tensor(tensor)
    _validate_alignment(alignment)
    storage = torch.empty(
        tensor.numel() + alignment // 2 - 1, dtype=tensor.dtype, device=tensor.device
    )
    offset = (-storage.data_ptr()) % alignment // 2
    return storage[offset : offset + tensor.numel()].view(tensor.shape)


def ensure_aligned(tensor, caller_stream, alignment=16):
    """Return the original aligned tensor, or an aligned CuTe-copied allocation."""
    _validate_tensor(tensor)
    _validate_alignment(alignment)
    if tensor.data_ptr() % alignment == 0:
        return tensor
    result = aligned_empty_like(tensor, alignment)
    copy_bf16(tensor, result, caller_stream)
    return result
