"""Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Layout primitives for head-chunked Ulysses attention.

These functions deliberately do not create process groups, streams, events,
or an attention scheduler. They only transform Q/K/V and output head bands so
framework integrations can build their own communication/compute pipeline.
"""

from typing import Optional, Tuple

import torch

from ..api_logging import flashinfer_api

_INT32_MAX = 2**31 - 1
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _storage_ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Conservatively test whether two positive-strided tensors share bytes."""
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False

    def storage_end(tensor: torch.Tensor) -> int:
        max_element_offset = sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            if size > 0
        )
        return tensor.data_ptr() + (max_element_offset + 1) * tensor.element_size()

    return left.data_ptr() < storage_end(right) and right.data_ptr() < storage_end(left)


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _nonnegative_int(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative int, got {value!r}")
    return value


def _validate_cuda_tensor(
    tensor, name: str, *, ndim: int, contiguous: bool
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(tensor.shape)}")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"{name} dtype must be float16/bfloat16/float32, got {tensor.dtype}"
        )
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dims must be positive, got {tuple(tensor.shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if not contiguous and any(stride <= 0 for stride in tensor.stride()):
        raise ValueError(f"{name} must have positive strides, got {tensor.stride()}")
    return tensor


def _validate_qkv_geometry(
    query,
    key,
    value,
    *,
    world_size,
    head_offset,
    head_count,
) -> Tuple[int, int, int, int, int]:
    world_size = _positive_int(world_size, "world_size")
    head_count = _positive_int(head_count, "head_count")
    head_offset = _nonnegative_int(head_offset, "head_offset")
    query = _validate_cuda_tensor(query, "query", ndim=4, contiguous=False)
    key = _validate_cuda_tensor(key, "key", ndim=4, contiguous=False)
    value = _validate_cuda_tensor(value, "value", ndim=4, contiguous=False)
    for name, tensor in (("key", key), ("value", value)):
        if tensor.shape != query.shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match query "
                f"shape {tuple(query.shape)}"
            )
        if tensor.device != query.device:
            raise ValueError(
                f"{name} is on {tensor.device}, but query is on {query.device}"
            )
        if tensor.dtype != query.dtype:
            raise ValueError(
                f"{name} dtype {tensor.dtype} does not match query dtype {query.dtype}"
            )
    batch, seq_len, global_heads, head_dim = query.shape
    if global_heads % world_size != 0:
        raise ValueError(
            f"global head count {global_heads} must be divisible by world_size "
            f"{world_size}"
        )
    local_heads = global_heads // world_size
    if head_offset + head_count > local_heads:
        raise ValueError(
            f"head band [{head_offset}, {head_offset + head_count}) exceeds "
            f"local_heads={local_heads}"
        )
    payload_elems = batch * seq_len * world_size * head_count * 3 * head_dim
    if payload_elems > _INT32_MAX:
        raise ValueError(
            f"packed QKV payload has {payload_elems} elements, exceeding the "
            f"int32 index range {_INT32_MAX}"
        )
    return batch, seq_len, local_heads, head_dim, payload_elems


def _prepare_qkv_output(
    query,
    out,
    *,
    world_size,
    head_count,
) -> torch.Tensor:
    batch, seq_len, _, head_dim = query.shape
    expected = (batch, seq_len, world_size * head_count, 3 * head_dim)
    if out is None:
        return torch.empty(expected, dtype=query.dtype, device=query.device)
    _validate_cuda_tensor(out, "out", ndim=4, contiguous=True)
    if tuple(out.shape) != expected:
        raise ValueError(f"out has shape {tuple(out.shape)}, expected {expected}")
    if out.device != query.device or out.dtype != query.dtype:
        raise ValueError(
            f"out device/dtype ({out.device}, {out.dtype}) must match query "
            f"({query.device}, {query.dtype})"
        )
    return out


def _launch_pack_qkv(
    out,
    query,
    key,
    value,
    *,
    world_size,
    local_heads,
    head_offset,
    head_count,
    nccl_layout,
) -> None:
    from ..triton.ulysses import pack_ulysses_qkv_head_chunk

    pack_ulysses_qkv_head_chunk(
        out,
        query,
        key,
        value,
        world_size=world_size,
        local_heads=local_heads,
        head_offset=head_offset,
        head_count=head_count,
        nccl_layout=nccl_layout,
    )


@flashinfer_api
def pack_ulysses_qkv_head_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    world_size: int,
    head_offset: int,
    head_count: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Pack one destination-local head band of Q/K/V.

    The three inputs use ``[B, S_local, H, D]`` and may be independent
    positive-strided views (including views produced by a fused projection).
    ``H`` must be divisible by ``world_size``. The selected local band is
    applied inside every destination rank's head slice, and the result is a
    contiguous fused payload ``[B, S_local, world_size * head_count, 3 * D]``.

    The payload can be passed directly to
    :meth:`flashinfer.comm.UlyssesCommunicator.scatter_heads`; its result has
    shape ``[B, S_global, head_count, 3 * D]``. Slicing the final dimension
    yields Q/K/V views for the attention backend.

    This operation runs on the caller's current CUDA stream and allocates only
    when ``out`` is omitted.
    """
    _, _, local_heads, _, _ = _validate_qkv_geometry(
        query,
        key,
        value,
        world_size=world_size,
        head_offset=head_offset,
        head_count=head_count,
    )
    out = _prepare_qkv_output(query, out, world_size=world_size, head_count=head_count)
    if any(_storage_ranges_overlap(out, tensor) for tensor in (query, key, value)):
        raise ValueError("out must not alias query, key, or value")
    _launch_pack_qkv(
        out,
        query,
        key,
        value,
        world_size=world_size,
        local_heads=local_heads,
        head_offset=head_offset,
        head_count=head_count,
        nccl_layout=False,
    )
    return out


def _validate_merge_geometry(
    received,
    out,
    *,
    world_size,
    local_heads,
    head_offset,
):
    world_size = _positive_int(world_size, "world_size")
    local_heads = _positive_int(local_heads, "local_heads")
    head_offset = _nonnegative_int(head_offset, "head_offset")
    received = _validate_cuda_tensor(received, "received", ndim=4, contiguous=True)
    out = _validate_cuda_tensor(out, "out", ndim=4, contiguous=True)
    batch, local_seq, compact_heads, head_dim = received.shape
    if compact_heads % world_size != 0:
        raise ValueError(
            f"received compact head count {compact_heads} must be divisible by "
            f"world_size {world_size}"
        )
    head_count = compact_heads // world_size
    if head_offset + head_count > local_heads:
        raise ValueError(
            f"head band [{head_offset}, {head_offset + head_count}) exceeds "
            f"local_heads={local_heads}"
        )
    expected = (batch, local_seq, world_size * local_heads, head_dim)
    if tuple(out.shape) != expected:
        raise ValueError(f"out has shape {tuple(out.shape)}, expected {expected}")
    if out.device != received.device or out.dtype != received.dtype:
        raise ValueError(
            f"out device/dtype ({out.device}, {out.dtype}) must match received "
            f"({received.device}, {received.dtype})"
        )
    if _storage_ranges_overlap(out, received):
        raise ValueError("out must not alias received")
    if received.numel() > _INT32_MAX:
        raise ValueError(
            f"received has {received.numel()} elements, exceeding the int32 "
            f"index range {_INT32_MAX}"
        )
    return batch, local_seq, head_count, head_dim


def _launch_merge_rank_major(
    received_rank_major,
    out,
    *,
    world_size,
    local_heads,
    head_offset,
) -> None:
    from ..triton.ulysses import merge_ulysses_output_head_chunk

    merge_ulysses_output_head_chunk(
        received_rank_major,
        out,
        world_size=world_size,
        local_heads=local_heads,
        head_offset=head_offset,
    )


@flashinfer_api
def merge_ulysses_output_head_chunk(
    received: torch.Tensor,
    *,
    world_size: int,
    local_heads: int,
    head_offset: int,
    out: torch.Tensor,
) -> torch.Tensor:
    r"""Merge one compact Ulysses output head band into a full output.

    ``received`` is the result of gathering a chunked attention output and has
    shape ``[B, S_local, world_size * head_count, D]``. ``out`` is the full
    ``[B, S_local, world_size * local_heads, D]`` destination. Only the band
    ``[rank * local_heads + head_offset, + head_count)`` for each source rank
    is written; other output elements remain unchanged.

    Calling this function once for every non-overlapping chunk in a schedule
    whose head counts sum to ``local_heads`` reconstructs the ordinary
    whole-head Ulysses result. It runs on the caller's current CUDA stream and
    performs no allocation.
    """
    batch, local_seq, head_count, head_dim = _validate_merge_geometry(
        received,
        out,
        world_size=world_size,
        local_heads=local_heads,
        head_offset=head_offset,
    )
    rank_major = received.view(
        batch, local_seq, world_size, head_count, head_dim
    ).permute(2, 0, 1, 3, 4)
    _launch_merge_rank_major(
        rank_major,
        out,
        world_size=world_size,
        local_heads=local_heads,
        head_offset=head_offset,
    )
    return out


def _launch_pack_output_sequence(
    out_rank_major,
    source,
    *,
    world_size,
) -> None:
    from ..triton.ulysses import pack_ulysses_output_sequence_chunk

    pack_ulysses_output_sequence_chunk(
        out_rank_major,
        source,
        world_size=world_size,
    )
