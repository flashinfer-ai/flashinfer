# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Lightweight declarations and caller-owned scratch for Prims-TS DS4.1 decode."""

from dataclasses import dataclass
import functools
import math
import struct

import torch

from .._utils import _overlaps


_ALIGNMENT = 32


def _check_problem(batch_size, compressed_topk, backend):
    if backend != "prims-ts":
        raise ValueError(f"workspace backend must be 'prims-ts', got {backend!r}")
    if type(batch_size) is not int or not 1 <= batch_size <= 65535:
        raise ValueError("batch_size must be an integer in [1, 65535]")
    if (
        type(compressed_topk) is not int
        or not 0 <= compressed_topk <= 512
        or compressed_topk % 64
    ):
        raise ValueError("compressed_topk must be 0 or a multiple of64 up to512")


def _sections(batch_size, compressed_topk):
    sections = [("lse", (batch_size, 64, 1), torch.float32)]
    if compressed_topk == 0:
        sections.append(("dummy_main", (1, 64, 1, 288), torch.uint8))
    return sections


def workspace_size(batch_size, compressed_topk=0, *, backend="prims-ts"):
    """Required bytes for a 32-byte-aligned workspace; no device queries."""
    _check_problem(batch_size, compressed_topk, backend)
    offset = 0
    for _, shape, dtype in _sections(batch_size, compressed_topk):
        offset += (-offset) % _ALIGNMENT
        offset += math.prod(shape) * dtype.itemsize
    return offset


@dataclass(frozen=True)
class WorkspaceViews:
    lse: torch.Tensor
    dummy_main: object = None


def workspace_views(
    workspace_buffer, batch_size, compressed_topk=0, *, backend="prims-ts"
):
    """Bind private typed views, without allocating storage or querying CUDA."""
    required = workspace_size(batch_size, compressed_topk, backend=backend)
    if workspace_buffer.dtype not in (torch.int8, torch.uint8):
        raise ValueError("workspace_buffer must be an int8 or uint8 byte tensor")
    if not workspace_buffer.is_contiguous() or workspace_buffer.data_ptr() % _ALIGNMENT:
        raise ValueError("workspace_buffer must be contiguous and 32-byte aligned")
    if workspace_buffer.numel() < required:
        raise ValueError(f"workspace_buffer requires at least {required} bytes")
    flat = workspace_buffer.view(-1).view(torch.uint8)
    offset, views = 0, {}
    for name, shape, dtype in _sections(batch_size, compressed_topk):
        # Match the actual pointer, not merely the logical section offset.
        address = workspace_buffer.data_ptr() + offset
        offset += (-address) % _ALIGNMENT
        end = offset + math.prod(shape) * dtype.itemsize
        views[name] = flat[offset:end].view(dtype).view(shape)
        offset = end
    return WorkspaceViews(**views)


def _scalar(value, name, *, positive=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a Python scalar, not a tensor")
    try:
        value = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        raise ValueError(f"{name} must be representable as finite FP32") from None
    if not math.isfinite(value) or (positive and value <= 0.0):
        suffix = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be {suffix} FP32")
    return value


@functools.cache
def device_sm_count(device_index):
    """Cache host-only device facts, never caller buffers or captured addresses."""
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("warm DS4.1 decode before CUDA Graph capture")
    props = torch.cuda.get_device_properties(device_index)
    if (props.major, props.minor) not in ((10, 0), (10, 3), (10, 7)):
        raise ValueError("Prims-TS DS4.1 decode requires SM100, SM103 or SM107")
    return props.multi_processor_count


@dataclass(frozen=True)
class DecodeInputs:
    query: torch.Tensor
    swa_cache: torch.Tensor
    main_cache: torch.Tensor
    indices: torch.Tensor
    topk_lens: torch.Tensor
    seq_lens: torch.Tensor
    out: torch.Tensor
    bmm1_scale: float
    bmm2_scale: float
    sinks: object
    views: WorkspaceViews
    batch: int
    compressed_k: int


def bind_inputs(
    query,
    swa_kv_cache,
    workspace_buffer,
    sparse_indices=None,
    compressed_kv_cache=None,
    sparse_topk_lens=None,
    seq_lens=None,
    out=None,
    bmm1_scale=1.0,
    bmm2_scale=1.0,
    sinks=None,
    kv_layout="HND",
    *,
    backend="prims-ts",
):
    """Validate all public ownership before any preparation or output writes."""
    if backend != "prims-ts":
        raise ValueError(f"workspace backend must be 'prims-ts', got {backend!r}")
    if query.dtype != torch.float8_e4m3fn:
        raise ValueError("Prims-TS decode requires E4M3 Q")
    if kv_layout not in ("HND", "NHD"):
        raise ValueError("kv_layout must be 'HND' or 'NHD'")
    bmm1_scale = _scalar(bmm1_scale, "bmm1_scale", positive=True)
    bmm2_scale = _scalar(bmm2_scale, "bmm2_scale")
    if (
        query.ndim != 4
        or query.shape[1:] != (1, 64, 512)
        or not 1 <= query.shape[0] <= 65535
    ):
        raise ValueError("Q must be E4M3 [B,1,64,512], with 1 <= B <= 65535")
    if query.device.type != "cuda":
        raise ValueError(
            "Prims-TS DS4.1 decode requires a CUDA query on SM100, SM103 or SM107"
        )
    batch = query.shape[0]
    if sparse_indices is None or sparse_topk_lens is None or seq_lens is None:
        raise ValueError("sparse_indices, sparse_topk_lens and seq_lens are required")
    if (
        sparse_indices.ndim != 2
        or sparse_indices.shape[0] != batch
        or sparse_indices.dtype != torch.int32
    ):
        raise ValueError("sparse_indices must be int32 [B,128+K]")
    compressed_k = sparse_indices.shape[1] - 128
    _check_problem(batch, compressed_k, backend)
    for name, value in (("sparse_topk_lens", sparse_topk_lens), ("seq_lens", seq_lens)):
        if value.shape != (batch,) or value.dtype != torch.int32:
            raise ValueError(f"{name} must be int32 [B]")
    if compressed_k and compressed_kv_cache is None:
        raise ValueError("compressed_kv_cache is required when K > 0")
    if sinks is not None and (sinks.shape != (64,) or sinks.dtype != torch.float32):
        raise ValueError("sinks must be FP32 [64]")
    for name, cache, width, cache_dtype in (
        ("swa_kv_cache", swa_kv_cache, 512, torch.float8_e4m3fn),
        ("compressed_kv_cache", compressed_kv_cache, 288, torch.uint8),
    ):
        if cache is None:
            if name == "swa_kv_cache":
                raise ValueError("swa_kv_cache must be a nonempty page pool")
            continue
        shape = (1, 64, width) if kv_layout == "HND" else (64, 1, width)
        if (
            cache.ndim != 4
            or cache.shape[0] < 1
            or cache.shape[1:] != shape
            or cache.dtype != cache_dtype
        ):
            raise ValueError(
                f"{name} must be a nonempty {cache_dtype} {kv_layout} page64 pool of width{width}"
            )
    if out is not None and (out.shape != query.shape or out.dtype != torch.bfloat16):
        raise ValueError("out must be BF16 with the query's shape")
    inputs = tuple(
        x
        for x in (
            query,
            swa_kv_cache,
            compressed_kv_cache,
            sparse_indices,
            sparse_topk_lens,
            seq_lens,
            sinks,
        )
        if x is not None
    )
    public = (*inputs, workspace_buffer) + (() if out is None else (out,))
    if any(x.device != query.device for x in public):
        raise ValueError("inputs, workspace_buffer and out must be on Q's device")
    if any(not x.is_contiguous() or x.data_ptr() % 16 for x in public):
        raise ValueError(
            "inputs, workspace_buffer and out must be contiguous and 16-byte aligned"
        )
    if torch.is_grad_enabled() and any(x.requires_grad for x in public):
        raise ValueError("DS4.1 decode is inference-only")
    views = workspace_views(workspace_buffer, batch, compressed_k, backend=backend)
    if any(_overlaps(workspace_buffer, x) for x in inputs):
        raise ValueError("workspace_buffer must not overlap inputs")
    if out is not None and any(_overlaps(out, x) for x in (*inputs, workspace_buffer)):
        raise ValueError("out must not overlap inputs or workspace_buffer")
    with torch.cuda.device(query.device):
        if out is None and torch.cuda.is_current_stream_capturing():
            raise ValueError("CUDA Graph capture requires caller-supplied out")
        device_sm_count(query.device.index)
        if out is None:
            out = torch.empty(query.shape, device=query.device, dtype=torch.bfloat16)
    # Both layouts have one KV head. This changes metadata, not packed bytes.
    swa_cache = swa_kv_cache.view(swa_kv_cache.shape[0], 64, 1, -1)
    main_cache = (
        compressed_kv_cache.view(compressed_kv_cache.shape[0], 64, 1, 288)
        if compressed_k
        else views.dummy_main
    )
    return DecodeInputs(
        query,
        swa_cache,
        main_cache,
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        out,
        bmm1_scale,
        bmm2_scale,
        sinks,
        views,
        batch,
        compressed_k,
    )
