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

The public wrapper validates the tensor ABI before this module checks Int32
addressing limits, launches the GPU inspector, and decodes its fixed Int64
summary into Python planning facts. Inspection deliberately performs one
packed device-to-host copy and does not build a remapped route payload or
specialize the launch for the current route/guard morphology.
"""

from dataclasses import dataclass

import torch

from .common import _SIGNED_INT32_MAX


# Device summary ABI; keep this order synchronized with block_sparse_inspect.py:
# error, max row NNZ, max KV128 routes, reachable token hole.
_SUMMARY_FIELDS = 4


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


def _validate_int32_extent(value: int, name: str) -> None:
    """Reject a validated positive extent that device Int32 cannot represent."""

    if value > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name} must fit in signed int32")


def _validate_device_int32_product(name: str, *factors: int) -> None:
    """Reject linearized extents that the kernel cannot address with Int32."""

    product = 1
    for factor in factors:
        product *= factor
        if product > _SIGNED_INT32_MAX:
            raise OverflowError(f"{name} must fit in signed int32")


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
    stream: torch.cuda.Stream,
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

    for value, name in (
        (batch_size, "batch_size"),
        (num_kv_heads, "num_kv_heads"),
        (seq_len_q, "seq_len_q"),
        (seq_len_kv, "seq_len_kv"),
    ):
        _validate_int32_extent(value, name)

    num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
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
    if kv_valid_bits is not None:
        expected_bits_shape = (batch_size, (seq_len_kv + 31) // 32)
        _validate_device_int32_product(
            "kv_valid_bits.numel()",
            *expected_bits_shape,
        )

    device = block_indptr.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

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
        summary_values = tuple(int(value) for value in summary_gpu.tolist())

    _raise_for_noncanonical_bsr(summary_values)
    # This positional decode is the host mirror of the device summary ABI.
    return _BlockSparseInspection(
        max_row_nnz=summary_values[1],
        max_retained_routes=summary_values[2],
        token_mask_has_holes=bool(summary_values[3]),
    )


__all__ = ["_BlockSparseInspection", "_inspect_block_sparse_bsr"]
