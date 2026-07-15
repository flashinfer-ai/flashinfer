# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-scheduled contiguous and paged context attention.

The public surface intentionally exposes attention semantics, not scheduler
choices. Every plan uses persistent scheduling; contiguous K/V uses the CLC
dynamic scheduler and paged K/V uses the static scheduler required by its
page-offset producer. The private policy is query-paired unless a positive
left window requires head-paired GQA. Causal windows are bottom-right aligned:
for row ``q``, the inclusive right position is ``q + (S_kv - S_q)`` and
``window_left`` is measured from that position.
"""

from dataclasses import dataclass
import functools
import math
import numbers
import struct
from typing import Literal, Optional

import torch

from flashinfer.api_logging import flashinfer_api


_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
_SUPPORTED_HEAD_DIMS = (128, 256)
_PAGED_KV_PAGE_SIZE = 32
_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class _ContextGeometry:
    """Validated semantic and storage geometry for one reusable plan."""

    device: torch.device
    device_index: int
    packed: bool
    batch_size: int
    total_q: int
    total_k: int
    max_seq_len_q: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    has_q_offset: bool
    max_q_offset: int
    causal_single_kv_tile: bool
    q_shape: tuple[int, ...]
    kv_shape: tuple[int, ...]


@dataclass(frozen=True)
class _PagedContextGeometry:
    """Validated geometry for one packed-Q, paged-KV reusable plan."""

    device: torch.device
    device_index: int
    batch_size: int
    total_q: int
    max_seq_len_q: int
    max_seq_len_k: int
    max_num_pages_per_seq_kv: int
    num_physical_pages: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    output_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    has_q_offset: bool
    max_q_offset: int
    q_shape: tuple[int, ...]
    kv_shape: tuple[int, ...]


@dataclass(frozen=True)
class _PagedContextMetadata:
    """Host-validated metadata used to build stable device-side plan inputs."""

    kv_indptr: tuple[int, ...]
    seq_lens: tuple[int, ...]
    dense_page_indices: tuple[int, ...]


def _validate_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")


def _compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= int(extent)
    return tuple(reversed(strides))


def _validate_compact(tensor: torch.Tensor, name: str, layout: str) -> None:
    expected = _compact_strides(tuple(tensor.shape))
    if tensor.stride() != expected:
        raise ValueError(
            f"{name} must have compact {layout} strides {expected}, "
            f"but has {tensor.stride()}"
        )


def _validate_alignment(tensor: torch.Tensor, name: str, alignment: int) -> None:
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")


def _dtype_key(dtype: torch.dtype) -> str:
    keys = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }
    try:
        return keys[dtype]
    except KeyError as error:
        raise NotImplementedError(
            "attention-ts context supports torch.float16, torch.bfloat16, "
            f"and torch.float8_e4m3fn; got {dtype}"
        ) from error


def _validate_qkv_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    _dtype_key(q.dtype)
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError(
            "attention-ts context requires Q, K, and V to use the same dtype; "
            f"got Q {q.dtype}, K {k.dtype}, and V {v.dtype}"
        )


def _validate_output_dtype(output_dtype: torch.dtype) -> None:
    if not isinstance(output_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")
    _dtype_key(output_dtype)


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _validate_device(device: torch.device) -> int:
    device_index = _device_index(device)
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
    if capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise NotImplementedError(
            "attention-ts context requires an SM100a/B200 or SM103a/B300 GPU; "
            f"device cuda:{device_index} has compute capability {capability}"
        )
    return device_index


def _validate_mask(mask_type: str) -> None:
    if not isinstance(mask_type, str):
        raise TypeError("mask_type must be a string")
    if mask_type not in ("dense", "causal"):
        raise ValueError(
            f"mask_type must be exactly 'dense' or 'causal', got {mask_type!r}"
        )


def _validate_window_left(window_left: int, mask_type: str) -> int:
    if isinstance(window_left, bool) or not isinstance(window_left, int):
        raise TypeError("window_left must be an integer")
    if window_left == 0:
        raise ValueError("window_left=0 is unsupported; use -1 to disable the window")
    if window_left < -1:
        raise ValueError("window_left must be -1 (disabled) or positive")
    if window_left > _INT32_MAX - 1:
        raise ValueError(f"window_left must be no larger than {_INT32_MAX - 1}")
    if window_left > 0 and mask_type != "causal":
        raise ValueError("a positive window_left requires mask_type='causal'")
    return window_left


def _validate_scale(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive Python scalar")
    try:
        as_float = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive Python scalar") from error
    if not math.isfinite(as_float) or as_float <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    try:
        as_float32 = struct.unpack("=f", struct.pack("=f", as_float))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            f"{name} must be representable as a positive float32"
        ) from error
    if not math.isfinite(as_float32) or as_float32 <= 0.0:
        raise ValueError(f"{name} must be representable as a positive float32")
    return as_float32


def _validate_extent(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _INT32_MAX:
        raise NotImplementedError(f"{name} must fit in a signed int32")
    return value


def _validate_indptr_tensor(
    indptr: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    _validate_tensor(indptr, name)
    if indptr.device != device:
        raise ValueError(f"{name} must be on {device}, got {indptr.device}")
    if indptr.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if indptr.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {indptr.ndim}")
    if indptr.numel() < 2:
        raise ValueError(f"{name} must contain at least start and end offsets")
    _validate_compact(indptr, name, "[B+1]")
    _validate_alignment(indptr, name, 4)


def _read_indptr(
    indptr: torch.Tensor,
    name: str,
    *,
    expected_total: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Copy plan metadata once and validate strictly positive row lengths."""

    values = tuple(int(value) for value in indptr.tolist())
    if values[0] != 0:
        raise ValueError(f"{name} must start at 0")
    if values[-1] != expected_total:
        raise ValueError(
            f"the final {name} offset must equal the packed tensor extent; "
            f"expected {expected_total}, got {values[-1]}"
        )
    lengths = tuple(
        curr - prev for prev, curr in zip(values[:-1], values[1:], strict=True)
    )
    if any(length <= 0 for length in lengths):
        raise ValueError(f"{name} offsets must be strictly increasing")
    return values, lengths


def _read_int32_values(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_count: int,
) -> tuple[int, ...]:
    """Copy one plan-time metadata vector after validating its extent."""

    if tensor.numel() != expected_count:
        raise ValueError(
            f"{name} must contain {expected_count} elements, got {tensor.numel()}"
        )
    return tuple(int(value) for value in tensor.tolist())


def _validate_paged_metadata_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
) -> None:
    _validate_tensor(tensor, name)
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got rank {tensor.ndim}")
    _validate_compact(tensor, name, "one-dimensional")
    _validate_alignment(tensor, name, 4)


def _validate_page_size(page_size: int) -> int:
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if page_size != _PAGED_KV_PAGE_SIZE:
        raise NotImplementedError(
            "attention-ts paged context currently requires page_size="
            f"{_PAGED_KV_PAGE_SIZE}; got {page_size}"
        )
    return page_size


def _validate_kv_layout(kv_layout: str) -> None:
    if not isinstance(kv_layout, str):
        raise TypeError("kv_layout must be a string")
    if kv_layout != "HND":
        raise NotImplementedError(
            f"attention-ts paged context supports only kv_layout='HND'; got {kv_layout!r}"
        )


def _validate_base_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        _validate_tensor(tensor, name)
        _validate_alignment(tensor, name, 16)
    if k.device != q.device or v.device != q.device:
        raise ValueError(
            "Q, K, and V must be on one CUDA device; "
            f"got {q.device}, {k.device}, and {v.device}"
        )
    _validate_qkv_dtype(q, k, v)
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError(
            f"v must have the same shape as k; got {tuple(v.shape)} and {tuple(k.shape)}"
        )


def _validate_head_geometry(num_qo_heads: int, num_kv_heads: int) -> int:
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("Q and KV head counts must be positive")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "the Q head count must be divisible by the KV head count; "
            f"got {num_qo_heads} and {num_kv_heads}"
        )
    return num_qo_heads // num_kv_heads


def _validate_head_dim(q_head_dim: int, kv_head_dim: int) -> int:
    if q_head_dim != kv_head_dim:
        raise ValueError(
            "Q and K/V head dimensions must match; "
            f"got Q {q_head_dim} and K/V {kv_head_dim}"
        )
    if q_head_dim not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            "attention-ts context supports head_dim in "
            f"{_SUPPORTED_HEAD_DIMS}; got {q_head_dim}"
        )
    return q_head_dim


def _derive_q_offset_geometry(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    mask_type: str,
) -> tuple[bool, int]:
    """Return whether causal Q offsets exist and their safe domain maximum."""

    if mask_type != "causal":
        return False, 0
    q_offsets = tuple(
        k_length - q_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    )
    return any(offset != 0 for offset in q_offsets), max(q_offsets)


def _resolve_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
) -> _ContextGeometry:
    """Validate a plan and derive its semantic compile key.

    Packed cumulative offsets are copied to the host only here.  A successful
    plan owns their tensor storage and never synchronizes on the run path.
    """

    _validate_base_tensors(q, k, v)
    _validate_output_dtype(output_dtype)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    device_index = _validate_device(q.device)
    device = torch.device("cuda", device_index)
    if (qo_indptr is None) != (kv_indptr is None):
        raise ValueError("qo_indptr and kv_indptr must be provided together")
    packed = qo_indptr is not None

    if packed:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "packed Q/K/V must use [total_tokens, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        total_q, num_qo_heads, q_head_dim = map(int, q.shape)
        total_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        _validate_extent(total_q, "total_q")
        _validate_extent(total_k, "total_k")
        _validate_compact(q, "q", "[total_q, Hq, D]")
        _validate_compact(k, "k", "[total_k, Hkv, D]")
        _validate_compact(v, "v", "[total_k, Hkv, D]")
        assert qo_indptr is not None and kv_indptr is not None
        _validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
        _validate_indptr_tensor(kv_indptr, "kv_indptr", device=device)
        if qo_indptr.numel() != kv_indptr.numel():
            raise ValueError(
                "qo_indptr and kv_indptr must describe the same batch; "
                f"got {qo_indptr.numel() - 1} and {kv_indptr.numel() - 1} rows"
            )
        _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
        _, k_lengths = _read_indptr(kv_indptr, "kv_indptr", expected_total=total_k)
        batch_size = len(q_lengths)
        _validate_extent(batch_size, "batch_size")
        max_seq_len_q = max(q_lengths)
        max_seq_len_k = max(k_lengths)
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)
    else:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                "fixed Q/K/V must use [B, S, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        batch_size, max_seq_len_q, num_qo_heads, q_head_dim = map(int, q.shape)
        k_batch, max_seq_len_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        if batch_size != k_batch:
            raise ValueError(
                f"q and k batch dimensions must match; got {batch_size} and {k_batch}"
            )
        _validate_extent(batch_size, "batch_size")
        _validate_extent(max_seq_len_q, "seq_len_q")
        _validate_extent(max_seq_len_k, "seq_len_k")
        total_q = batch_size * max_seq_len_q
        total_k = batch_size * max_seq_len_k
        _validate_extent(total_q, "B * seq_len_q")
        _validate_extent(total_k, "B * seq_len_k")
        _validate_compact(q, "q", "[B, Sq, Hq, D]")
        _validate_compact(k, "k", "[B, Sk, Hkv, D]")
        _validate_compact(v, "v", "[B, Sk, Hkv, D]")
        q_lengths = (max_seq_len_q,) * batch_size
        k_lengths = (max_seq_len_k,) * batch_size
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)

    _validate_head_dim(q_head_dim, kv_head_dim)
    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    if mask_type == "causal":
        for batch_idx, (q_length, k_length) in enumerate(
            zip(q_lengths, k_lengths, strict=True)
        ):
            if q_length > k_length:
                raise ValueError(
                    "bottom-right causal context requires Sq <= Sk for each "
                    f"request; got batch {batch_idx}: Sq={q_length}, Sk={k_length}"
                )

    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )
    has_q_offset, max_q_offset = _derive_q_offset_geometry(
        q_lengths, k_lengths, mask_type
    )
    causal_single_kv_tile = (
        mask_type == "causal"
        and not packed
        and not head_paired
        and max_seq_len_q == max_seq_len_k
        and max_seq_len_k <= 128
    )
    return _ContextGeometry(
        device=device,
        device_index=device_index,
        packed=packed,
        batch_size=batch_size,
        total_q=total_q,
        total_k=total_k,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        q_dtype=q.dtype,
        output_dtype=output_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        has_q_offset=has_q_offset,
        max_q_offset=max_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        q_shape=q_shape,
        kv_shape=kv_shape,
    )


def _resolve_paged_geometry(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
    mask_type: str,
    window_left: int,
    output_dtype: torch.dtype,
) -> tuple[_PagedContextGeometry, _PagedContextMetadata]:
    """Validate packed-Q paged-KV inputs and materialize their static ABI."""

    _validate_base_tensors(q, k_cache, v_cache)
    _validate_output_dtype(output_dtype)
    _validate_mask(mask_type)
    window_left = _validate_window_left(window_left, mask_type)
    page_size = _validate_page_size(page_size)
    device_index = _validate_device(q.device)
    device = torch.device("cuda", device_index)

    if q.ndim != 3:
        raise ValueError(
            "paged context Q must use packed [total_q, Hq, D] storage; "
            f"got rank {q.ndim}"
        )
    if k_cache.ndim != 4:
        raise ValueError(
            "paged K/V caches must use [num_pages, Hkv, page_size, D] "
            f"storage; got rank {k_cache.ndim}"
        )
    total_q, num_qo_heads, q_head_dim = map(int, q.shape)
    num_physical_pages, num_kv_heads, cache_page_size, kv_head_dim = map(
        int, k_cache.shape
    )
    _validate_extent(total_q, "total_q")
    _validate_extent(num_physical_pages, "num_physical_pages")
    if cache_page_size != page_size:
        raise ValueError(
            f"K/V cache page extent must equal page_size={page_size}; "
            f"got {cache_page_size}"
        )
    _validate_head_dim(q_head_dim, kv_head_dim)
    _validate_compact(q, "q", "[total_q, Hq, D]")
    _validate_compact(k_cache, "k_cache", "[num_pages, Hkv, page_size, D]")
    _validate_compact(v_cache, "v_cache", "[num_pages, Hkv, page_size, D]")

    _validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
    _validate_indptr_tensor(paged_kv_indptr, "paged_kv_indptr", device=device)
    for tensor, name in (
        (paged_kv_indices, "paged_kv_indices"),
        (paged_kv_last_page_len, "paged_kv_last_page_len"),
    ):
        _validate_paged_metadata_tensor(tensor, name, device=device)

    batch_size = int(qo_indptr.numel()) - 1
    _validate_extent(batch_size, "batch_size")
    if paged_kv_indptr.numel() != batch_size + 1:
        raise ValueError(
            "paged_kv_indptr and qo_indptr must describe the same batch; "
            f"got {paged_kv_indptr.numel() - 1} and {batch_size} rows"
        )
    _, q_lengths = _read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    page_indptr_values, page_counts = _read_indptr(
        paged_kv_indptr,
        "paged_kv_indptr",
        expected_total=int(paged_kv_indices.numel()),
    )
    page_indices = _read_int32_values(
        paged_kv_indices,
        "paged_kv_indices",
        expected_count=page_indptr_values[-1],
    )
    last_page_lens = _read_int32_values(
        paged_kv_last_page_len,
        "paged_kv_last_page_len",
        expected_count=batch_size,
    )
    for offset, page_idx in enumerate(page_indices):
        if page_idx < 0 or page_idx >= num_physical_pages:
            raise ValueError(
                "paged_kv_indices entries must index the physical page pool; "
                f"entry {offset} is {page_idx}, pool has {num_physical_pages} pages"
            )
    for batch_idx, last_page_len in enumerate(last_page_lens):
        if last_page_len < 1 or last_page_len > page_size:
            raise ValueError(
                "paged_kv_last_page_len entries must be in [1, page_size]; "
                f"batch {batch_idx} has {last_page_len}, page_size={page_size}"
            )

    k_lengths = tuple(
        (page_count - 1) * page_size + last_page_len
        for page_count, last_page_len in zip(page_counts, last_page_lens, strict=True)
    )
    max_seq_len_q = max(q_lengths)
    max_seq_len_k = max(k_lengths)
    max_num_pages_per_seq_kv = max(page_counts)
    _validate_extent(max_seq_len_q, "max_seq_len_q")
    _validate_extent(max_seq_len_k, "max_seq_len_k")
    _validate_extent(max_num_pages_per_seq_kv, "max_num_pages_per_seq_kv")

    head_ratio = _validate_head_geometry(num_qo_heads, num_kv_heads)
    if mask_type == "causal":
        for batch_idx, (q_length, k_length) in enumerate(
            zip(q_lengths, k_lengths, strict=True)
        ):
            if q_length > k_length:
                raise ValueError(
                    "bottom-right causal context requires Sq <= Sk for each "
                    f"request; got batch {batch_idx}: Sq={q_length}, Sk={k_length}"
                )
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )
    has_q_offset, max_q_offset = _derive_q_offset_geometry(
        q_lengths, k_lengths, mask_type
    )

    logical_kv_indptr = [0]
    for k_length in k_lengths:
        logical_kv_indptr.append(logical_kv_indptr[-1] + k_length)
    _validate_extent(logical_kv_indptr[-1], "sum_seq_len_k")
    # The upstream ABI is [B, 2, max_pages], so batch is the outer dimension.
    # Preserve arbitrary and repeated physical page indices from the
    # FlashInfer CSR representation.
    dense_page_indices: list[int] = []
    for batch_idx, page_count in enumerate(page_counts):
        page_start = page_indptr_values[batch_idx]
        row = page_indices[page_start : page_start + page_count]
        padded_row = row + (0,) * (max_num_pages_per_seq_kv - page_count)
        dense_page_indices.extend(padded_row)
        dense_page_indices.extend(padded_row)

    geometry = _PagedContextGeometry(
        device=device,
        device_index=device_index,
        batch_size=batch_size,
        total_q=total_q,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
        num_physical_pages=num_physical_pages,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        q_dtype=q.dtype,
        output_dtype=output_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        has_q_offset=has_q_offset,
        max_q_offset=max_q_offset,
        q_shape=tuple(q.shape),
        kv_shape=tuple(k_cache.shape),
    )
    metadata = _PagedContextMetadata(
        kv_indptr=tuple(logical_kv_indptr),
        seq_lens=k_lengths,
        dense_page_indices=tuple(dense_page_indices),
    )
    return geometry, metadata


def _semantic_key(geometry: _ContextGeometry) -> tuple[object, ...]:
    return (
        geometry.device_index,
        geometry.batch_size,
        geometry.max_seq_len_q,
        geometry.max_seq_len_k,
        geometry.num_qo_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        _dtype_key(geometry.q_dtype),
        _dtype_key(geometry.output_dtype),
        geometry.mask_type,
        geometry.window_left,
        geometry.packed,
        geometry.head_paired,
        geometry.has_q_offset,
        geometry.max_q_offset,
        geometry.causal_single_kv_tile,
    )


def _paged_semantic_key(
    geometry: _PagedContextGeometry,
) -> tuple[object, ...]:
    return (
        geometry.device_index,
        geometry.batch_size,
        geometry.max_seq_len_q,
        geometry.max_seq_len_k,
        geometry.max_num_pages_per_seq_kv,
        geometry.num_qo_heads,
        geometry.num_kv_heads,
        geometry.head_dim,
        _dtype_key(geometry.q_dtype),
        _dtype_key(geometry.output_dtype),
        geometry.mask_type,
        geometry.window_left,
        geometry.head_paired,
        geometry.has_q_offset,
        geometry.max_q_offset,
    )


@functools.cache
def _get_compiled_context(
    device_index: int,
    batch_size: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    window_left: int,
    packed: bool,
    head_paired: bool,
    has_q_offset: bool,
    max_q_offset: int,
    causal_single_kv_tile: bool,
):
    """Compile and cache one exact semantic context-attention specialization."""

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv
    from cutlass.base_dsl.dsl import BaseDSL
    import cutlass.utils as utils

    from .kernels.fmha_context.fmha_kernel import FmhaTs

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    input_dtype = dtype_map[q_dtype_key]
    output_dtype = dtype_map[output_dtype_key]
    is_causal = mask_type == "causal"
    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_dtype=input_dtype,
        out_dtype=output_dtype,
        d=head_dim,
        is_persistent=True,
        is_causal=is_causal,
        is_clc_dynamic=True,
        head_paired=head_paired,
        window_size_left=window_left if window_left > 0 else 0,
        h_r=num_qo_heads // num_kv_heads,
        enable_skip_correction=True,
        causal_single_kv_tile=causal_single_kv_tile,
    )
    fmha.cfg.has_varlen = packed
    fmha.cfg.has_q_offset = has_q_offset
    if not is_causal and not packed:
        fmha.cfg.fixed_dense_k_tail = max_seq_len_k % fmha.cfg.kv_tile_n
    with torch.cuda.device(device_index):
        max_active_clusters = int(utils.HardwareInfo().get_max_active_clusters(1))

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        scale_softmax_log2: cute.Tensor,
        output_scale: cute.Tensor,
        qo_indptr: cute.Tensor,
        kv_indptr: cute.Tensor,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        static_packed: cutlass.Constexpr[bool],
        static_has_q_offset: cutlass.Constexpr[bool],
        static_max_seq_len_q: cutlass.Constexpr[int],
        static_max_seq_len_k: cutlass.Constexpr[int],
        static_max_q_offset: cutlass.Constexpr[int],
    ) -> None:
        """Adapt torch TVM-FFI tensors to the FmhaTs host entry point."""

        if cutlass.const_expr(static_packed):
            # Independent Q/K maxima do not determine the largest per-request
            # bottom-right offset, so pass the exact plan-time maximum.
            max_q_offset_arg = None
            if cutlass.const_expr(static_has_q_offset):
                max_q_offset_arg = cutlass.Int32(static_max_q_offset)
            fmha(
                q,
                k,
                v,
                out,
                scale_softmax_log2,
                output_scale,
                static_max_active_clusters,
                stream,
                qo_indptr,
                kv_indptr,
                cutlass.Int32(static_max_seq_len_q),
                cutlass.Int32(static_max_seq_len_k),
                max_q_offset_arg,
            )
        else:
            fmha(
                q,
                k,
                v,
                out,
                scale_softmax_log2,
                output_scale,
                static_max_active_clusters,
                stream,
            )

    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    q_shape: tuple[object, ...]
    kv_shape: tuple[object, ...]
    out_shape: tuple[object, ...]
    if packed:
        runtime_total_q = cute.sym_int()
        runtime_total_k = cute.sym_int()
        q_shape = (runtime_total_q, num_qo_heads, head_dim)
        kv_shape = (runtime_total_k, num_kv_heads, head_dim)
        out_shape = (runtime_total_q, num_qo_heads, head_dim)
        indptr_shape = (batch_size + 1,)
    else:
        q_shape = (batch_size, max_seq_len_q, num_qo_heads, head_dim)
        kv_shape = (batch_size, max_seq_len_k, num_kv_heads, head_dim)
        out_shape = q_shape
        indptr_shape = (1,)
    q_fake = fake_compact(input_dtype, q_shape, 16)
    k_fake = fake_compact(input_dtype, kv_shape, 16)
    v_fake = fake_compact(input_dtype, kv_shape, 16)
    out_fake = fake_compact(output_dtype, out_shape, 16)
    scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    output_scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    qo_indptr_fake = fake_compact(cutlass.Int32, indptr_shape, 4)
    kv_indptr_fake = fake_compact(cutlass.Int32, indptr_shape, 4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # TS task objects carry loop-local state through generated control flow;
    # PyIR reconstructs that structure before lowering to TVM FFI.
    with torch.cuda.device(device_index), BaseDSL.enable_pyir():
        compiled = cute.compile(
            tensor_adapter,
            q_fake,
            k_fake,
            v_fake,
            out_fake,
            scale_fake,
            output_scale_fake,
            qo_indptr_fake,
            kv_indptr_fake,
            stream_fake,
            max_active_clusters,
            packed,
            has_q_offset,
            max_seq_len_q,
            max_seq_len_k,
            max_q_offset,
            options=_COMPILE_OPTIONS,
        )
    policy = (
        ("scheduler", "clc_dynamic_persistent"),
        ("pairing", "head" if head_paired else "query"),
        ("causal_single_kv_tile", causal_single_kv_tile),
    )
    return compiled, policy


@functools.cache
def _get_compiled_paged_context(
    device_index: int,
    batch_size: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_num_pages_per_seq_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype_key: str,
    output_dtype_key: str,
    mask_type: str,
    window_left: int,
    head_paired: bool,
    has_q_offset: bool,
    max_q_offset: int,
):
    """Compile one packed-Q, page-size-32 context specialization."""

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv
    from cutlass.base_dsl.dsl import BaseDSL
    import cutlass.utils as utils

    from .kernels.fmha_context.fmha_kernel import FmhaTs

    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
    }
    input_dtype = dtype_map[q_dtype_key]
    output_dtype = dtype_map[output_dtype_key]
    is_causal = mask_type == "causal"
    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_dtype=input_dtype,
        out_dtype=output_dtype,
        d=head_dim,
        is_persistent=True,
        is_causal=is_causal,
        # CUTLASS DSL 4.7 requires the sole CLC scheduler task to produce only
        # WorkQueue. Paged context also needs the auxiliary warp to produce
        # smem_page_offsets_kv, so retain persistent scheduling but use the
        # static scheduler and its dedicated PageOffsetsTask.
        is_clc_dynamic=False,
        head_paired=head_paired,
        window_size_left=window_left if window_left > 0 else 0,
        h_r=num_qo_heads // num_kv_heads,
        enable_skip_correction=True,
        use_paged_kv=True,
        num_tokens_per_page=_PAGED_KV_PAGE_SIZE,
        max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
        # The fixed one-tile shortcut assumes contiguous fixed-shape K/V and
        # is intentionally disabled for the page-table path.
        causal_single_kv_tile=False,
    )
    fmha.cfg.has_varlen = True
    fmha.cfg.has_q_offset = has_q_offset
    with torch.cuda.device(device_index):
        max_active_clusters = int(utils.HardwareInfo().get_max_active_clusters(1))

    @cute.jit
    def tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        out: cute.Tensor,
        scale_softmax_log2: cute.Tensor,
        output_scale: cute.Tensor,
        qo_indptr: cute.Tensor,
        kv_indptr: cute.Tensor,
        page_idx_kv: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        stream: cuda_drv.CUstream,
        static_max_active_clusters: cutlass.Constexpr[int],
        static_has_q_offset: cutlass.Constexpr[bool],
        static_max_seq_len_q: cutlass.Constexpr[int],
        static_max_seq_len_k: cutlass.Constexpr[int],
        static_max_q_offset: cutlass.Constexpr[int],
    ) -> None:
        max_q_offset_arg = None
        if cutlass.const_expr(static_has_q_offset):
            max_q_offset_arg = cutlass.Int32(static_max_q_offset)
        fmha(
            q,
            k_cache,
            v_cache,
            out,
            scale_softmax_log2,
            output_scale,
            static_max_active_clusters,
            stream,
            qo_indptr,
            kv_indptr,
            cutlass.Int32(static_max_seq_len_q),
            cutlass.Int32(static_max_seq_len_k),
            max_q_offset_arg,
            page_idx_kv,
            seq_lens_kv,
        )

    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    runtime_total_q = cute.sym_int()
    runtime_num_pages = cute.sym_int()
    q_fake = fake_compact(input_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    kv_shape = (
        runtime_num_pages,
        num_kv_heads,
        _PAGED_KV_PAGE_SIZE,
        head_dim,
    )
    k_fake = fake_compact(input_dtype, kv_shape, 16)
    v_fake = fake_compact(input_dtype, kv_shape, 16)
    out_fake = fake_compact(output_dtype, (runtime_total_q, num_qo_heads, head_dim), 16)
    scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    output_scale_fake = fake_compact(cutlass.Float32, (1,), 4)
    qo_indptr_fake = fake_compact(cutlass.Int32, (batch_size + 1,), 4)
    kv_indptr_fake = fake_compact(cutlass.Int32, (batch_size + 1,), 4)
    page_idx_fake = fake_compact(
        cutlass.Int32,
        (batch_size, 2, max_num_pages_per_seq_kv),
        4,
    )
    seq_lens_fake = fake_compact(cutlass.Int32, (batch_size,), 4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    with torch.cuda.device(device_index), BaseDSL.enable_pyir():
        compiled = cute.compile(
            tensor_adapter,
            q_fake,
            k_fake,
            v_fake,
            out_fake,
            scale_fake,
            output_scale_fake,
            qo_indptr_fake,
            kv_indptr_fake,
            page_idx_fake,
            seq_lens_fake,
            stream_fake,
            max_active_clusters,
            has_q_offset,
            max_seq_len_q,
            max_seq_len_k,
            max_q_offset,
            options=_COMPILE_OPTIONS,
        )
    policy = (
        ("scheduler", "static_persistent"),
        ("pairing", "head" if head_paired else "query"),
        ("kv_layout", "paged_hnd"),
        ("page_size", _PAGED_KV_PAGE_SIZE),
        ("causal_single_kv_tile", False),
    )
    return compiled, policy


def _validate_runtime_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    geometry: _ContextGeometry,
) -> None:
    """Validate run tensors without allocation, synchronization, or metadata reads."""

    _validate_base_tensors(q, k, v)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.q_dtype:
        raise ValueError(f"q must have dtype {geometry.q_dtype}, got {q.dtype}")
    if tuple(q.shape) != geometry.q_shape:
        raise ValueError(f"q must have shape {geometry.q_shape}, got {tuple(q.shape)}")
    if tuple(k.shape) != geometry.kv_shape:
        raise ValueError(f"k must have shape {geometry.kv_shape}, got {tuple(k.shape)}")
    q_layout = "[total_q, Hq, D]" if geometry.packed else "[B, Sq, Hq, D]"
    kv_layout = "[total_k, Hkv, D]" if geometry.packed else "[B, Sk, Hkv, D]"
    _validate_compact(q, "q", q_layout)
    _validate_compact(k, "k", kv_layout)
    _validate_compact(v, "v", kv_layout)


def _validate_paged_runtime_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    geometry: _PagedContextGeometry,
) -> None:
    """Validate a paged run without reading metadata or allocating tensors."""

    _validate_base_tensors(q, k_cache, v_cache)
    if q.device != geometry.device:
        raise ValueError(f"q must be on {geometry.device}, got {q.device}")
    if q.dtype != geometry.q_dtype:
        raise ValueError(f"q must have dtype {geometry.q_dtype}, got {q.dtype}")
    if tuple(q.shape) != geometry.q_shape:
        raise ValueError(f"q must have shape {geometry.q_shape}, got {tuple(q.shape)}")
    if tuple(k_cache.shape) != geometry.kv_shape:
        raise ValueError(
            f"k_cache must have shape {geometry.kv_shape}, got {tuple(k_cache.shape)}"
        )
    _validate_compact(q, "q", "[total_q, Hq, D]")
    _validate_compact(k_cache, "k_cache", "[num_pages, Hkv, page_size, D]")
    _validate_compact(v_cache, "v_cache", "[num_pages, Hkv, page_size, D]")


def _prepare_out(
    out: Optional[torch.Tensor],
    *,
    q: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if out is None:
        return torch.empty(tuple(q.shape), dtype=output_dtype, device=q.device)
    _validate_tensor(out, "out")
    if tuple(out.shape) != tuple(q.shape):
        raise ValueError(
            f"out must have shape {tuple(q.shape)}, got {tuple(out.shape)}"
        )
    if out.dtype != output_dtype:
        raise ValueError(f"out must have dtype {output_dtype}, got {out.dtype}")
    if out.device != q.device:
        raise ValueError(f"out must be on {q.device}, got {out.device}")
    layout = "[total_q, Hq, D]" if q.ndim == 3 else "[B, Sq, Hq, D]"
    _validate_compact(out, "out", layout)
    _validate_alignment(out, "out", 16)
    return out


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two compact tensors overlap in their storage byte ranges."""

    lhs_start = lhs.data_ptr()
    lhs_end = lhs_start + lhs.numel() * lhs.element_size()
    rhs_start = rhs.data_ptr()
    rhs_end = rhs_start + rhs.numel() * rhs.element_size()
    return lhs_start < rhs_end and rhs_start < lhs_end


def _validate_out_does_not_overlap_inputs(
    out: torch.Tensor,
    *named_inputs: tuple[str, torch.Tensor],
) -> None:
    """Reject output aliasing with any live kernel input tensor."""

    for name, tensor in named_inputs:
        if _tensors_overlap(out, tensor):
            raise ValueError(f"out must not overlap {name} storage")


class BatchPrefillTSWrapper:
    """Plan and reuse task-scheduled fixed or packed-ragged context attention.

    ``plan`` may compile, allocate two one-element scale tensors, and copy
    packed cumulative offsets to the host for validation.  ``run`` performs no
    metadata read or synchronization.  With caller-provided ``out``, its Python
    path allocates no tensors and is suitable for CUDA graph capture. ``out``
    must not overlap any Q, K, V, packed-offset, or scale input storage.

    Packed metadata storage is retained by the plan and must remain alive and
    at stable addresses.  If its values are changed, callers must preserve the
    plan-time batch, total token counts, positive lengths, static maximum Q/K
    lengths, maximum Q offset, and per-request bottom-right causal contract.
    """

    @flashinfer_api
    def __init__(self) -> None:
        self._planned = False

    @flashinfer_api
    def plan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        mask_type: Literal["dense", "causal"] = "dense",
        window_left: int = -1,
        sm_scale: Optional[float] = None,
        output_scale: float = 1.0,
        out_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Validate semantics, select the private policy, and compile once."""

        if out_dtype is None:
            if not isinstance(q, torch.Tensor):
                raise TypeError("q must be a torch.Tensor")
            resolved_out_dtype = q.dtype
        else:
            resolved_out_dtype = out_dtype
        geometry = _resolve_geometry(
            q,
            k,
            v,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            mask_type=mask_type,
            window_left=window_left,
            output_dtype=resolved_out_dtype,
        )
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(geometry.head_dim)
        sm_scale = _validate_scale(sm_scale, "sm_scale")
        output_scale = _validate_scale(output_scale, "output_scale")
        scale_softmax_log2 = _validate_scale(
            sm_scale * math.log2(math.e), "sm_scale * log2(e)"
        )
        compiled, policy = _get_compiled_context(*_semantic_key(geometry))
        scale_tensor = torch.tensor(
            [scale_softmax_log2], dtype=torch.float32, device=geometry.device
        )
        output_scale_tensor = torch.tensor(
            [output_scale], dtype=torch.float32, device=geometry.device
        )
        if geometry.packed:
            assert qo_indptr is not None and kv_indptr is not None
            planned_qo_indptr = qo_indptr
            planned_kv_indptr = kv_indptr
        else:
            # Uniform TVM-FFI signature; fixed specializations compile these
            # arguments away but still keep stable runtime placeholders.
            planned_qo_indptr = torch.zeros(
                1, dtype=torch.int32, device=geometry.device
            )
            planned_kv_indptr = torch.zeros(
                1, dtype=torch.int32, device=geometry.device
            )

        # Publish only after validation, compilation, and allocation succeed.
        self._geometry = geometry
        self._qo_indptr = planned_qo_indptr
        self._kv_indptr = planned_kv_indptr
        self._scale_softmax_log2 = scale_tensor
        self._output_scale = output_scale_tensor
        self._compiled = compiled
        self._policy = policy
        self._planned = True

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Launch on the current stream into output disjoint from Q, K, and V."""

        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        _validate_runtime_inputs(q, k, v, self._geometry)
        out = _prepare_out(out, q=q, output_dtype=self._geometry.output_dtype)
        _validate_out_does_not_overlap_inputs(
            out,
            ("q", q),
            ("k", k),
            ("v", v),
            ("qo_indptr", self._qo_indptr),
            ("kv_indptr", self._kv_indptr),
            ("scale_softmax_log2", self._scale_softmax_log2),
            ("output_scale", self._output_scale),
        )
        self._compiled(
            q,
            k,
            v,
            out,
            self._scale_softmax_log2,
            self._output_scale,
            self._qo_indptr,
            self._kv_indptr,
        )
        return out


class BatchPrefillPagedTSWrapper:
    """Plan and reuse packed-Q context attention over HND paged K/V caches.

    ``plan`` validates FlashInfer CSR page metadata once, translates it to the
    dense page-table ABI consumed by the context kernel, and retains both the
    original and derived device tensors. Arbitrary, repeated, and nonidentity
    physical page indices are preserved. Metadata values are immutable for the
    life of the plan; call ``plan`` again after changing them.

    ``run`` reads no metadata values and performs no synchronization. With a
    caller-provided ``out``, it allocates no tensors and is suitable for CUDA
    graph capture. K and V are separate compact HND tensors with shape
    ``[num_pages, Hkv, 32, D]`` and Q/output use packed
    ``[total_q, Hq, D]`` storage.
    """

    @flashinfer_api
    def __init__(self, kv_layout: Literal["HND"] = "HND") -> None:
        _validate_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._planned = False

    @flashinfer_api
    def plan(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        *,
        page_size: int = _PAGED_KV_PAGE_SIZE,
        mask_type: Literal["dense", "causal"] = "dense",
        window_left: int = -1,
        sm_scale: Optional[float] = None,
        output_scale: float = 1.0,
        out_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Validate metadata, compile one specialization, and own its ABI."""

        if out_dtype is None:
            if not isinstance(q, torch.Tensor):
                raise TypeError("q must be a torch.Tensor")
            resolved_out_dtype = q.dtype
        else:
            resolved_out_dtype = out_dtype
        geometry, metadata = _resolve_paged_geometry(
            q,
            k_cache,
            v_cache,
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            output_dtype=resolved_out_dtype,
        )
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(geometry.head_dim)
        sm_scale = _validate_scale(sm_scale, "sm_scale")
        output_scale = _validate_scale(output_scale, "output_scale")
        scale_softmax_log2 = _validate_scale(
            sm_scale * math.log2(math.e), "sm_scale * log2(e)"
        )
        compiled, policy = _get_compiled_paged_context(*_paged_semantic_key(geometry))

        scale_tensor = torch.tensor(
            [scale_softmax_log2], dtype=torch.float32, device=geometry.device
        )
        output_scale_tensor = torch.tensor(
            [output_scale], dtype=torch.float32, device=geometry.device
        )
        logical_kv_indptr = torch.tensor(
            metadata.kv_indptr, dtype=torch.int32, device=geometry.device
        )
        seq_lens_kv = torch.tensor(
            metadata.seq_lens, dtype=torch.int32, device=geometry.device
        )
        dense_page_idx_kv = torch.tensor(
            metadata.dense_page_indices,
            dtype=torch.int32,
            device=geometry.device,
        ).view(
            geometry.batch_size,
            2,
            geometry.max_num_pages_per_seq_kv,
        )

        # Publish only after validation, compilation, and allocation succeed.
        self._geometry = geometry
        self._qo_indptr = qo_indptr
        self._paged_kv_indptr = paged_kv_indptr
        self._paged_kv_indices = paged_kv_indices
        self._paged_kv_last_page_len = paged_kv_last_page_len
        self._logical_kv_indptr = logical_kv_indptr
        self._seq_lens_kv = seq_lens_kv
        self._dense_page_idx_kv = dense_page_idx_kv
        self._scale_softmax_log2 = scale_tensor
        self._output_scale = output_scale_tensor
        self._compiled = compiled
        self._policy = policy
        self._planned = True

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Launch the planned page-table specialization on the current stream."""

        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        _validate_paged_runtime_inputs(q, k_cache, v_cache, self._geometry)
        out = _prepare_out(out, q=q, output_dtype=self._geometry.output_dtype)
        _validate_out_does_not_overlap_inputs(
            out,
            ("q", q),
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("qo_indptr", self._qo_indptr),
            ("paged_kv_indptr", self._paged_kv_indptr),
            ("paged_kv_indices", self._paged_kv_indices),
            ("paged_kv_last_page_len", self._paged_kv_last_page_len),
            ("logical_kv_indptr", self._logical_kv_indptr),
            ("seq_lens_kv", self._seq_lens_kv),
            ("dense_page_idx_kv", self._dense_page_idx_kv),
            ("scale_softmax_log2", self._scale_softmax_log2),
            ("output_scale", self._output_scale),
        )
        self._compiled(
            q,
            k_cache,
            v_cache,
            out,
            self._scale_softmax_log2,
            self._output_scale,
            self._qo_indptr,
            self._logical_kv_indptr,
            self._dense_page_idx_kv,
            self._seq_lens_kv,
        )
        return out


@flashinfer_api
def batch_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    output_scale: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one-shot fixed or packed-ragged task-scheduled context attention.

    Fixed tensors use ``[B, S, H, D]`` storage. Providing both cumulative
    int32 offset tensors selects packed ``[total_tokens, H, D]`` storage.
    ``D`` may be 128 or 256.
    Causal masking is bottom-right aligned.  ``window_left=-1`` disables the
    left window; a positive value selects the private head-paired GQA policy
    and retains at most ``window_left + 1`` keys at each causal row, including
    when ``S_q < S_kv``.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        q,
        k,
        v,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=sm_scale,
        output_scale=output_scale,
        out_dtype=resolved_out_dtype,
    )
    return wrapper.run(q, k, v, out=out)


@flashinfer_api
def batch_prefill_with_paged_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    *,
    page_size: int = _PAGED_KV_PAGE_SIZE,
    kv_layout: Literal["HND"] = "HND",
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    sm_scale: Optional[float] = None,
    output_scale: float = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one-shot packed-Q context attention over separate HND page pools.

    Q/output use ``[total_q, Hq, D]`` storage and K/V each use
    ``[num_pages, Hkv, 32, D]``. ``qo_indptr`` describes Q rows while the
    three paged-KV metadata tensors use FlashInfer's CSR representation.
    Physical page indices need not be identity ordered. ``D`` may be 128 or
    256; Q, K, and V must share one supported dtype.
    """

    resolved_out_dtype = (
        out.dtype if out_dtype is None and isinstance(out, torch.Tensor) else out_dtype
    )
    wrapper = BatchPrefillPagedTSWrapper(kv_layout=kv_layout)
    wrapper.plan(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=sm_scale,
        output_scale=output_scale,
        out_dtype=resolved_out_dtype,
    )
    return wrapper.run(q, k_cache, v_cache, out=out)


__all__ = [
    "BatchPrefillTSWrapper",
    "BatchPrefillPagedTSWrapper",
    "batch_prefill_with_kv_cache",
    "batch_prefill_with_paged_kv_cache",
]
