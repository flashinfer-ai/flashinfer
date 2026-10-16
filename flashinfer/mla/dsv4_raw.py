# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration helpers for trying DSv4 decode before sparse-index packing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_dsv4_raw_decode_workspace_size(rows: int, heads: int) -> int:
    """Return scratch bytes for raw-index decode, without CUDA work.

    Supported geometries are 16 heads with 1/2/4 query rows, or 32 heads with
    1/2 query rows. Each row is one ordinary-decode query, not a speculative
    verification sequence. The caller allocates a contiguous CUDA uint8 tensor
    of at least this size and serializes its use by executions and graph replays.
    Unsupported geometry raises :class:`ValueError`.
    """
    from ._smallhead import workspace_layout

    return workspace_layout(rows, heads)[1]


def try_batch_decode_sparse_mla_dsv4_raw(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    compressed_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    request_ids: torch.Tensor,
    compressed_block_table: torch.Tensor,
    token_valid: torch.Tensor,
    sinks: torch.Tensor,
    workspace_buffer: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    enable_pdl: bool | None = None,
) -> bool:
    """Try ordinary FP8 DSv4 decode directly from logical sparse metadata.

    This integration helper avoids producing an intermediate packed index
    tensor. It returns ``True`` after enqueueing the attention kernels into the
    current stream and writing the caller's ``out``. It returns ``False`` before
    any GPU work when the tensor metadata is unsupported or when a previously
    unseen geometry is encountered during CUDA Graph capture. Callers must
    retain their original index-packing and attention path on ``False``.
    Compilation and execution errors propagate; they are never fallback signals.

    Eligibility is limited to B300 SM103, CuTe DSL 4.8+, PDL, FP8 E4M3 Q/KV,
    head dimension 512, and H16 rows1/2/4 or H32 rows1/2. ``query`` and ``out``
    have shapes ``[rows, heads, 512]`` and dtypes float8_e4m3fn and bfloat16.
    Normalized HND KV pools have shape ``[pages, 1, page_size, 512]``; page size
    is 32 for SWA and 64 or 128 for compressed KV. Token rows are compact;
    physical pages may be padded and the tensors may have nonzero origins.

    ``swa_indices`` is contiguous int32 ``[rows, 128]`` containing logical
    global SWA token IDs. ``compressed_indices`` is contiguous int32
    ``[rows, 512]`` containing request-local compressed token IDs.
    ``request_ids`` is int32 ``[rows]`` indexing the rows of the compact int32
    ``compressed_block_table`` of shape ``[rows, columns]``. Its nonnegative
    entries identify physical compressed-cache pages. ``seq_lens`` is int32
    ``[rows]`` and gives the original sequence lengths for the SWA window.
    ``token_valid`` is bool ``[rows]``; ``sinks`` is float32 ``[heads]``.

    Negative token IDs and negative page-table entries are masked. The number
    of nonnegative compressed IDs defines an active *prefix*, including any
    interior holes; this operation does not compact indices. A false token
    validity masks the compressed segment only. SWA validity still follows
    its indices and sequence length, matching the packed DSv4 decode contract.

    Metadata values are a caller contract and are not copied to the CPU for
    validation: request IDs must be in range, sequence lengths nonnegative,
    nonnegative SWA IDs must be below ``swa_kv_cache.shape[0] * 32``,
    and nonnegative compressed IDs below ``columns * compressed_page_size``.
    Nonnegative page entries must be valid physical pages. These bounds apply
    even to invalid-token rows, because address resolution precedes masking.

    All tensors must be on the query device and 16-byte aligned. Inputs other
    than KV pools must be contiguous. ``out`` and the required workspace region
    must not overlap any input or each other. Obtain the uint8 scratch size
    with :func:`get_dsv4_raw_decode_workspace_size`. Warm each geometry outside
    capture; runtime tensor pointers are rebound on every call and are not cached.
    The warm path performs metadata checks and creates views, with no GPU
    allocation or host synchronization. BMM scales are finite Python numbers.

    This helper deliberately returns an admission flag rather than allocating
    an output or preparing a fallback. Its caller owns the existing native path.
    """
    from ._smallhead_raw import try_run

    return try_run(
        query=query,
        swa_kv_cache=swa_kv_cache,
        compressed_kv_cache=compressed_kv_cache,
        swa_indices=swa_indices,
        compressed_indices=compressed_indices,
        seq_lens=seq_lens,
        request_ids=request_ids,
        compressed_block_table=compressed_block_table,
        token_valid=token_valid,
        sinks=sinks,
        workspace_buffer=workspace_buffer,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        enable_pdl=enable_pdl,
    )
