# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host side of the plan-time raw-BSR inspection.

This module validates the tensor ABI, launches the GPU inspector, and decodes
its fixed Int64 summary into Python planning facts. Inspection deliberately
performs one packed device-to-host copy and does not build a remapped route
payload or specialize the launch for the current route/guard morphology.
"""

import threading
from dataclasses import dataclass

import torch

from .common import (
    _SIGNED_INT32_MAX,
    _validate_sparse_block_size,
)


# Device summary ABI; keep this order synchronized with block_sparse_inspect.py:
# error, max row NNZ, max KV128 routes, reachable token hole.
_SUMMARY_FIELDS = 4
_SUMMARY_HOST_STORAGE = threading.local()


@dataclass(frozen=True)
class _BlockSparseInspection:
    """Planning facts derived from caller-owned canonical BSR metadata.

    ``max_row_nnz`` counts semantic BSR blocks. ``max_retained_routes`` counts
    KV128 execution routes after selected blocks are expanded into canonical
    8/16/32/64-token atoms and atoms beyond ``seq_len_kv`` are removed.

    ``token_mask_has_holes`` means a selected, non-padding token has a zero
    validity bit. Runtime kernels derive their route-full decision from the
    current token words on every launch rather than freezing a plan-time
    morphology heuristic.
    """

    max_row_nnz: int
    max_retained_routes: int
    token_mask_has_holes: bool


def _validate_positive_int32(value: object, name: str) -> int:
    """Return a positive Python integer representable by device Int32."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive Python integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name} must fit in signed int32")
    return value


def _validate_cuda_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    dtype: torch.dtype,
    ndim: int,
) -> None:
    """Validate the compact tensor contract used by TVM FFI."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_device_int32_product(name: str, *factors: int) -> None:
    """Reject linearized extents that the kernel cannot address with Int32."""

    product = 1
    for factor in factors:
        product *= factor
        if product > _SIGNED_INT32_MAX:
            raise OverflowError(f"{name} must fit in signed int32")


def _read_summary_values(
    summary_gpu: torch.Tensor,
    stream: torch.cuda.Stream,
) -> tuple[int, ...]:
    """Perform the inspection's sole D2H through reusable pinned storage.

    The returned Python tuple is detached from the thread-local staging buffer,
    so a later inspection may safely reuse that pinned allocation.
    """

    summary_host = getattr(_SUMMARY_HOST_STORAGE, "buffer", None)
    if summary_host is None or summary_host.numel() != _SUMMARY_FIELDS:
        summary_host = torch.empty(
            _SUMMARY_FIELDS,
            dtype=torch.int64,
            pin_memory=True,
        )
        _SUMMARY_HOST_STORAGE.buffer = summary_host
    summary_host.copy_(summary_gpu, non_blocking=True)
    stream.synchronize()
    return tuple(int(value) for value in summary_host.tolist())


def _raise_for_noncanonical_bsr(summary_values: tuple[int, ...]) -> None:
    """Translate the device validation code after the one packed copy."""

    error_code = summary_values[0]
    if error_code == 0:
        return
    reason = {
        1: "indices in each row must be strictly increasing and unique",
        2: "indices must select an in-range KV block",
        3: "each indptr row range must be bounded and monotone",
    }.get(error_code, f"unknown validation error {error_code}")
    raise ValueError(f"block_indptr/block_indices must form canonical BSR: {reason}")


def _inspect_block_sparse_bsr(
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    q_block_size: int,
    kv_block_size: int,
    kv_valid_bits: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> _BlockSparseInspection:
    """Inspect raw BSR on its CUDA device and return host planning facts.

    ``block_indptr`` is contiguous Int32
    ``[B, Hkv, ceil(Sq / q_block_size) + 1]`` and stores absolute ranges into
    contiguous Int32 ``block_indices[nnz]``. Optional ``kv_valid_bits`` is
    contiguous Uint32 ``[B, ceil(Skv / 32)]`` and is shared by all KV heads.

    The GPU validates each referenced range before reading it, reduces all rows
    into one zero-initialized Int64[4], and this function performs the sole D2H
    synchronization before returning an immutable ``_BlockSparseInspection``.
    Fine-block route shape is deliberately absent: run resolves the live BSR
    indices and loads each fine atom directly. Only the coarse KV64 path may
    combine the two halves of one KV128 route.
    """

    q_block_size = _validate_sparse_block_size(q_block_size, "q_block_size")
    kv_block_size = _validate_sparse_block_size(kv_block_size, "kv_block_size")
    batch_size = _validate_positive_int32(batch_size, "batch_size")
    num_kv_heads = _validate_positive_int32(num_kv_heads, "num_kv_heads")
    seq_len_q = _validate_positive_int32(seq_len_q, "seq_len_q")
    seq_len_kv = _validate_positive_int32(seq_len_kv, "seq_len_kv")

    _validate_cuda_tensor(
        block_indptr,
        "block_indptr",
        dtype=torch.int32,
        ndim=3,
    )
    _validate_cuda_tensor(
        block_indices,
        "block_indices",
        dtype=torch.int32,
        ndim=1,
    )
    num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
    expected_indptr_shape = (batch_size, num_kv_heads, num_q_block_rows + 1)
    if tuple(block_indptr.shape) != expected_indptr_shape:
        raise ValueError(
            f"block_indptr must have shape {expected_indptr_shape}, got "
            f"{tuple(block_indptr.shape)}"
        )
    if block_indices.numel() > _SIGNED_INT32_MAX:
        raise OverflowError("block_indices.numel() must fit in signed int32")
    _validate_device_int32_product(
        "number of raw BSR rows",
        batch_size,
        num_kv_heads,
        num_q_block_rows,
    )
    _validate_device_int32_product(
        "block_indptr.numel()",
        batch_size,
        num_kv_heads,
        num_q_block_rows + 1,
    )
    if block_indptr.device != block_indices.device:
        raise ValueError("block_indptr and block_indices must be on the same device")

    if kv_valid_bits is not None:
        _validate_cuda_tensor(
            kv_valid_bits,
            "kv_valid_bits",
            dtype=torch.uint32,
            ndim=2,
        )
        expected_bits_shape = (batch_size, (seq_len_kv + 31) // 32)
        if tuple(kv_valid_bits.shape) != expected_bits_shape:
            raise ValueError(
                f"kv_valid_bits must have shape {expected_bits_shape}, got "
                f"{tuple(kv_valid_bits.shape)}"
            )
        if kv_valid_bits.device != block_indptr.device:
            raise ValueError("kv_valid_bits must share the BSR metadata device")
        _validate_device_int32_product(
            "kv_valid_bits.numel()",
            *expected_bits_shape,
        )

    device = block_indptr.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError("stream must be a torch.cuda.Stream")
    if stream is None:
        stream = torch.cuda.current_stream(device_index)
    if stream.device.index != device_index:
        raise ValueError("inspection stream must be on the BSR metadata device")

    # Checking the caller's current stream first avoids entering another
    # device/stream context while an enclosing graph capture is active.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "block-sparse inspection is unsupported during CUDA Graph capture"
        )

    from ..kernels.fmha_decode.block_sparse_inspect import (
        compile_block_sparse_inspection,
    )

    with torch.cuda.device(device_index), torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "block-sparse inspection is unsupported during CUDA Graph capture"
            )
        summary_gpu = torch.zeros(
            _SUMMARY_FIELDS,
            dtype=torch.int64,
            device=device,
        )
        inspect_bsr = compile_block_sparse_inspection(
            device_index=device_index,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            has_token_bits=kv_valid_bits is not None,
        )
        inspect_bsr(
            block_indptr,
            block_indices,
            kv_valid_bits,
            summary_gpu,
        )
        summary_values = _read_summary_values(summary_gpu, stream)

    _raise_for_noncanonical_bsr(summary_values)
    # This positional decode is the host mirror of the device summary ABI.
    return _BlockSparseInspection(
        max_row_nnz=summary_values[1],
        max_retained_routes=summary_values[2],
        token_mask_has_holes=bool(summary_values[3]),
    )


__all__ = ["_BlockSparseInspection", "_inspect_block_sparse_bsr"]
