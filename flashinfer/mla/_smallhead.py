# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal small-head FP8 decode adapter with caller-owned scratch storage.

This module implements the bounded SM103 automatic decode route.
It accepts normalized HND pools and physical token-row offsets. Unsupported
metadata must retain the existing backend. Calls sharing a workspace must be
serialized, including captured CUDA Graph replays.
"""

from functools import cache
import hashlib
import importlib.metadata
import math

import torch


class _UnpreparedCapture(RuntimeError):
    """A cache miss that automatic dispatch must route to the native backend."""


def workspace_layout(rows: int, heads: int):
    """Return byte offsets and FP32 shapes; no tensor allocation or CUDA work."""
    if type(rows) is not int or type(heads) is not int:
        raise TypeError("rows and heads must be integers")
    if heads not in (16, 32) or rows not in ((1, 2, 4) if heads == 16 else (1, 2)):
        raise ValueError("small-head decode supports H16 rows1/2/4 or H32 rows1/2")
    offset = 0
    parts = []
    for shape in ((rows, heads, 20, 512), (rows, heads, 20), (rows, 1, heads)):
        offset = (offset + 255) // 256 * 256
        size = math.prod(shape) * 4
        parts.append((offset, size, shape))
        offset += size
    return tuple(parts), (offset + 255) // 256 * 256


@cache
def _dsl_unavailable_reason():
    try:
        from packaging.version import Version

        version = importlib.metadata.version("nvidia-cutlass-dsl")
        if Version(version) < Version("4.8.0"):
            return f"small-head decode requires nvidia-cutlass-dsl>=4.8.0; installed {version}"
    except importlib.metadata.PackageNotFoundError:
        return "small-head decode requires nvidia-cutlass-dsl>=4.8.0"
    return None


def _pool_span(pool):
    return (pool.shape[0] - 1) * pool.stride(0) // 512 + pool.shape[2]


def unsupported_reason(
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    sinks,
    workspace_buffer,
    out,
    bmm1_scale,
    bmm2_scale,
    enable_pdl=None,
):
    """Metadata-only eligibility, shared by explicit calls and automatic dispatch."""
    if query.ndim != 3 or query.shape[-1] != 512:
        return "query must have shape [rows, heads, 512]"
    rows, heads, _ = query.shape
    if heads not in (16, 32) or rows not in ((1, 2, 4) if heads == 16 else (1, 2)):
        return "only H16 rows1/2/4 and H32 rows1/2 are qualified"
    if query.device.type != "cuda" or torch.cuda.get_device_capability(
        query.device
    ) != (10, 3):
        return "small-head decode is qualified on B300 SM103 only"
    if enable_pdl not in (None, True) or (
        enable_pdl is not None and type(enable_pdl) is not bool
    ):
        return "this implementation requires PDL; explicit PDL-off keeps the native backend"
    for scale in (bmm1_scale, bmm2_scale):
        if type(scale) not in (int, float) or not math.isfinite(scale):
            return "BMM scales must be finite Python numbers"
    specifications = (
        (query, (rows, heads, 512), torch.float8_e4m3fn, "query"),
        (sparse_indices, (rows, 640), torch.int32, "sparse_indices"),
        (sparse_topk_lens, (rows,), torch.int32, "sparse_topk_lens"),
        (seq_lens, (rows,), torch.int32, "seq_lens"),
        (sinks, (heads,), torch.float32, "sinks"),
        (out, (rows, heads, 512), torch.bfloat16, "out"),
    )
    for value, shape, dtype, name in specifications:
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
            return f"{name} must be contiguous, 16-byte aligned and on the query device"
    for pool in (swa_kv_cache, compressed_kv_cache):
        if not isinstance(pool, torch.Tensor) or pool.ndim != 4 or pool.shape[0] <= 0:
            return "KV pools must be nonempty normalized HND rank-4 tensors"
        if pool.shape[1] != 1 or pool.shape[2] <= 0 or pool.shape[3] != 512:
            return "KV pools must have one head and head dimension512"
        if (
            pool.device != query.device
            or pool.dtype != query.dtype
            or pool.data_ptr() % 16
        ):
            return "KV pools must be aligned FP8 tensors on the query device"
        if pool.stride(-1) != 1 or pool.stride(-2) != 512:
            return "KV token rows must be compact"
        if pool.stride(0) % 512 or pool.stride(0) < pool.shape[2] * 512:
            return "KV page strides must preserve distinct complete token rows"
        if (
            pool.storage_offset() + _pool_span(pool) * 512
            > pool.untyped_storage().nbytes()
        ):
            return "KV physical span exceeds its allocation"
    _, required = workspace_layout(rows, heads)
    if (
        not isinstance(workspace_buffer, torch.Tensor)
        or workspace_buffer.dtype != torch.uint8
    ):
        return "workspace_buffer must be a uint8 tensor"
    if (
        workspace_buffer.device != query.device
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.data_ptr() % 16
    ):
        return "workspace_buffer must be contiguous, aligned and on the query device"
    if workspace_buffer.numel() < required:
        return f"workspace_buffer needs at least {required} bytes"
    scratch_start = workspace_buffer.data_ptr()
    scratch_end = scratch_start + required
    for value, _, _, name in specifications:
        start = value.data_ptr()
        end = start + value.numel() * value.element_size()
        if max(start, scratch_start) < min(end, scratch_end):
            return f"workspace_buffer overlaps {name}"
    for pool in (swa_kv_cache, compressed_kv_cache):
        start = pool.data_ptr()
        end = start + _pool_span(pool) * 512
        if max(start, scratch_start) < min(end, scratch_end):
            return "workspace_buffer overlaps a KV pool"
    return _dsl_unavailable_reason()


def workspace_views(workspace_buffer, rows, heads):
    parts, required = workspace_layout(rows, heads)
    if workspace_buffer.dtype != torch.uint8 or not workspace_buffer.is_contiguous():
        raise ValueError("workspace must be contiguous uint8")
    if workspace_buffer.numel() < required or workspace_buffer.data_ptr() % 16:
        raise ValueError(f"workspace needs {required} aligned bytes")
    flat = workspace_buffer.view(-1)
    return tuple(
        flat.narrow(0, offset, size).view(torch.float32).view(shape)
        for offset, size, shape in parts
    )


@cache
def _get_kernel(rows, heads, swa_span, compressed_span, scale1, scale2, device_index):
    # The cache owns compiled code and immutable metadata only, never tensors.
    # A first use must prepare before graph capture; a hit does not compile.
    with torch.cuda.device(device_index):
        if torch.cuda.is_current_stream_capturing():
            raise _UnpreparedCapture(
                "Warm small-head decode with this geometry before CUDA Graph capture"
            )
        from flashinfer.cute_dsl.attention import dsv4_smallhead
        from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        key = (rows, heads, swa_span, compressed_span, scale1.hex(), scale2.hex())
        name = hashlib.sha256(repr(key).encode()).hexdigest()
        return build_and_load_cute_dsl_kernel(
            "dsv4_smallhead",
            name,
            lambda: dsv4_smallhead.compile_plan(
                rows,
                heads,
                640,
                swa_span,
                compressed_span,
                scale1,
                scale2,
                512,
                True,
                True,
                False,
                False,
                True,
                False,
                16,
                device_index,
            ),
            extra_key_files=(__file__, dsv4_smallhead.__file__),
        )


def run(
    *,
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    sinks,
    workspace_buffer,
    out,
    bmm1_scale,
    bmm2_scale,
    enable_pdl=None,
):
    """Run normalized physical-offset ordinary decode without GPU allocations.

    Inputs must already represent one query per sequence, with the first128
    sparse columns addressing the SWA pool. This private entry does not resolve
    logical page IDs, ragged requests, HCA metadata or packed-cache formats.
    The public caller must dispatch those contracts before reaching this entry.
    """
    reason = unsupported_reason(
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        sinks,
        workspace_buffer,
        out,
        bmm1_scale,
        bmm2_scale,
        enable_pdl,
    )
    if reason is not None:
        raise NotImplementedError(reason)
    rows, heads, _ = query.shape
    spans = tuple(_pool_span(p) for p in (swa_kv_cache, compressed_kv_cache))
    pools = tuple(
        p.as_strided((span, 512), (512, 1))
        for p, span in zip((swa_kv_cache, compressed_kv_cache), spans, strict=True)
    )
    po, pl, lse = workspace_views(workspace_buffer, rows, heads)
    with torch.cuda.device(query.device):
        kernel = _get_kernel(
            rows,
            heads,
            *spans,
            float(bmm1_scale),
            float(bmm2_scale),
            query.device.index,
        )
        kernel(
            query,
            *pools,
            sparse_indices,
            sparse_topk_lens,
            seq_lens,
            sinks,
            po,
            pl,
            out.view(rows, 1, heads, 512),
            lse,
        )
    return out


def try_run(**kwargs):
    """Try the bounded automatic route; preserve native on unsupported metadata.

    An unprepared capture falls back before any kernel launch or GPU allocation.
    Compilation and kernel failures are not swallowed by automatic dispatch.
    """
    if unsupported_reason(**kwargs) is not None:
        return False
    try:
        run(**kwargs)
    except _UnpreparedCapture:
        return False
    return True
