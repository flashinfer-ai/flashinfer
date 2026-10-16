# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host adapter for ordinary decode from raw metadata.

Inputs are normalized HND pools, logical global SWA token IDs, request-local
compressed token IDs, and a request-to-physical-page table. No vLLM dependency.
The caller supplies output and workspace and serializes their use/replays.

Metadata values are a caller contract, not read back during execution:
request_ids are in [0, rows), nonnegative compressed IDs are below
block_table.shape[1] * compressed_page, and nonnegative block-table entries
are physical cache pages. Negative index/page entries are masked. These bounds
apply even to invalid-token rows, because address resolution precedes masking.
SWA IDs address the logical SWA pool; seq_lens are nonnegative lengths.
Compressed validity preserves the native prefix-of-count convention, including
interior holes. It does not compact the list. An invalid token suppresses only
the compressed segment; SWA validity still follows its indices and seq_lens.
"""

from functools import cache
import hashlib
import math
from pathlib import Path

import torch

from flashinfer.mla._smallhead import (
    _UnpreparedCapture,
    _dsl_unavailable_reason,
    workspace_layout,
    workspace_views,
)


def _pool_span(pool):
    return (pool.shape[0] - 1) * (pool.stride(0) // 512) + pool.shape[2]


def unsupported_reason(
    *,
    query,
    swa_kv_cache,
    compressed_kv_cache,
    swa_indices,
    compressed_indices,
    seq_lens,
    request_ids,
    compressed_block_table,
    token_valid,
    sinks,
    workspace_buffer,
    out,
    bmm1_scale,
    bmm2_scale,
    enable_pdl=None,
):
    """Inspect tensor metadata only; return a reason before any GPU work."""
    if not isinstance(query, torch.Tensor) or query.ndim != 3 or query.shape[-1] != 512:
        return "query must have shape [rows, heads, 512]"
    rows, heads, _ = query.shape
    if heads not in (16, 32) or rows not in ((1, 2, 4) if heads == 16 else (1, 2)):
        return "only H16 rows1/2/4 and H32 rows1/2 are qualified"
    if query.device.type != "cuda" or torch.cuda.get_device_capability(
        query.device
    ) != (10, 3):
        return "raw-index small-head decode requires B300 SM103"
    if enable_pdl is not None and (type(enable_pdl) is not bool or not enable_pdl):
        return "raw-index small-head decode requires PDL"
    if any(
        type(scale) not in (int, float) or not math.isfinite(scale)
        for scale in (bmm1_scale, bmm2_scale)
    ):
        return "BMM scales must be finite Python numbers"
    specs = (
        (query, (rows, heads, 512), torch.float8_e4m3fn, "query"),
        (swa_indices, (rows, 128), torch.int32, "swa_indices"),
        (compressed_indices, (rows, 512), torch.int32, "compressed_indices"),
        (seq_lens, (rows,), torch.int32, "seq_lens"),
        (request_ids, (rows,), torch.int32, "request_ids"),
        (token_valid, (rows,), torch.bool, "token_valid"),
        (sinks, (heads,), torch.float32, "sinks"),
        (out, (rows, heads, 512), torch.bfloat16, "out"),
    )
    for value, shape, dtype, name in specs:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
        ):
            return f"{name} must have shape {shape} and dtype {dtype}"
        if (
            value.device != query.device
            or not value.is_contiguous()
            or value.data_ptr() % 16
        ):
            return f"{name} must be contiguous, aligned and on the query device"
    table = compressed_block_table
    if (
        not isinstance(table, torch.Tensor)
        or table.ndim != 2
        or table.shape[0] != rows
        or not 0 < table.shape[1] <= (2**31 - 1) // 128
        or table.dtype != torch.int32
        or table.device != query.device
        or not table.is_contiguous()
        or table.data_ptr() % 16
    ):
        return "compressed_block_table must be aligned compact int32 [rows, positive columns]"
    regions = [
        (value.data_ptr(), value.numel() * value.element_size(), name)
        for value, _, _, name in specs
    ]
    regions.append(
        (
            table.data_ptr(),
            table.numel() * table.element_size(),
            "compressed_block_table",
        )
    )
    for pool, pages, name in (
        (swa_kv_cache, (32,), "swa_kv_cache"),
        (compressed_kv_cache, (64, 128), "compressed_kv_cache"),
    ):
        if (
            not isinstance(pool, torch.Tensor)
            or pool.ndim != 4
            or pool.shape[0] <= 0
            or pool.shape[1] != 1
            or pool.shape[2] not in pages
            or pool.shape[3] != 512
        ):
            return f"{name} must be normalized HND with page size in {pages}"
        if (
            pool.dtype != query.dtype
            or pool.device != query.device
            or pool.data_ptr() % 16
        ):
            return f"{name} must be aligned FP8 on the query device"
        if (
            pool.stride(-1) != 1
            or pool.stride(-2) != 512
            or pool.stride(0) % 512
            or pool.stride(0) < pool.shape[2] * 512
        ):
            return f"{name} must have compact token rows and distinct pages"
        span = _pool_span(pool)
        if (
            span > 2**31 - 1
            or (pool.storage_offset() + span * 512) > pool.untyped_storage().nbytes()
        ):
            return f"{name} physical span is out of bounds"
        regions.append((pool.data_ptr(), span * 512, name))
    _, required = workspace_layout(rows, heads)
    if (
        not isinstance(workspace_buffer, torch.Tensor)
        or workspace_buffer.dtype != torch.uint8
        or workspace_buffer.device != query.device
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.data_ptr() % 16
        or workspace_buffer.numel() < required
    ):
        return (
            f"workspace_buffer needs at least {required} aligned contiguous uint8 bytes"
        )
    for writable, size, label in (
        (workspace_buffer, required, "workspace_buffer"),
        (out, out.numel() * out.element_size(), "out"),
    ):
        start = writable.data_ptr()
        for other, other_size, name in regions:
            if name != label and max(start, other) < min(
                start + size, other + other_size
            ):
                return f"{label} overlaps {name}"
    return _dsl_unavailable_reason()


@cache
def _get_kernel(
    rows,
    heads,
    swa_span,
    compressed_span,
    swa_stride,
    compressed_stride,
    compressed_page,
    table_columns,
    scale1,
    scale2,
    device_index,
):
    # Cache immutable compilation facts and code only, never runtime tensors.
    with torch.cuda.device(device_index):
        if torch.cuda.is_current_stream_capturing():
            raise _UnpreparedCapture(
                "Warm this raw-index geometry before CUDA Graph capture"
            )
        from flashinfer.cute_dsl.attention import dsv4_smallhead_raw as raw_kernel
        from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        key = (
            rows,
            heads,
            swa_span,
            compressed_span,
            swa_stride,
            compressed_stride,
            compressed_page,
            table_columns,
            scale1.hex(),
            scale2.hex(),
        )
        name = hashlib.sha256(repr(key).encode()).hexdigest()
        return build_and_load_cute_dsl_kernel(
            "dsv4_smallhead_raw",
            name,
            lambda: raw_kernel.compile_raw(
                rows,
                heads,
                swa_span,
                compressed_span,
                scale1,
                scale2,
                32,
                swa_stride,
                compressed_page,
                compressed_stride,
                table_columns,
                device_index,
            ),
            extra_key_files=(str(Path(__file__).resolve()), raw_kernel.__file__),
        )


def run(**inputs):
    """Execute with caller-owned scratch; views only, no warm GPU allocation."""
    reason = unsupported_reason(**inputs)
    if reason is not None:
        raise NotImplementedError(reason)
    query = inputs["query"]
    rows, heads, _ = query.shape
    pools = (inputs["swa_kv_cache"], inputs["compressed_kv_cache"])
    spans = tuple(_pool_span(p) for p in pools)
    strides = tuple(p.stride(0) // 512 for p in pools)
    with torch.cuda.device(query.device):
        kernel = _get_kernel(
            rows,
            heads,
            *spans,
            *strides,
            pools[1].shape[2],
            inputs["compressed_block_table"].shape[1],
            float(inputs["bmm1_scale"]),
            float(inputs["bmm2_scale"]),
            query.device.index,
        )
        po, pl, lse = workspace_views(inputs["workspace_buffer"], rows, heads)
        physical = tuple(
            p.as_strided((span, 512), (512, 1))
            for p, span in zip(pools, spans, strict=True)
        )
        kernel(
            query,
            *physical,
            inputs["compressed_indices"],
            inputs["swa_indices"],
            inputs["seq_lens"],
            inputs["request_ids"],
            inputs["compressed_block_table"],
            inputs["token_valid"],
            inputs["sinks"],
            po,
            pl,
            inputs["out"].view(rows, 1, heads, 512),
            lse,
        )
    return inputs["out"]


def try_run(**inputs):
    """Decline unsupported metadata or an unwarmed capture before any launch."""
    if unsupported_reason(**inputs) is not None:
        return False
    try:
        run(**inputs)
    except _UnpreparedCapture:
        return False
    return True
