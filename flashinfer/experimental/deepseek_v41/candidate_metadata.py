# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Explicit publication and consumption of candidate metadata snapshots."""

from dataclasses import dataclass

import torch
import triton as tr

from ._utils import _overlaps
from .indexer_cute import (
    _candidate_metadata,
    _check_candidate_layout,
    _launch_candidate,
    workspace_size,
)
from .indexer_fp32 import _check_tensor, _scores_output, _validate_layer_inputs


@dataclass(frozen=True, eq=False)
class _CandidateMetadata:
    # Host configuration is fixed; prepare(..., out=metadata) explicitly
    # republishes only the owned device buffers. No raw tensor identity cache.
    batch: int
    count: int
    page_size: int
    num_physical_pages: int
    max_context_len: int
    device: torch.device
    _storage: torch.Tensor
    _encoded: torch.Tensor
    _mask: torch.Tensor
    _visible: torch.Tensor


def prepare_candidate_metadata(
    visible,
    block_table,
    candidates,
    *,
    page_size,
    num_physical_pages,
    max_context_len,
    out=None,
):
    if visible.device.type != "cuda" or torch.cuda.get_device_capability(
        visible.device
    ) not in ((10, 0), (10, 3)):
        raise ValueError("candidate metadata currently requires SM100/SM103")
    if visible.ndim != 1 or visible.shape[0] < 1:
        raise ValueError("visible requires a positive [B] shape")
    batch = visible.shape[0]
    device = visible.device
    _check_tensor(visible, (batch,), torch.int32, device, "visible")
    if candidates.ndim != 2 or not 1 <= candidates.shape[1] <= 2048:
        raise ValueError("candidate block8 IDs require [B,1..2048]")
    count = candidates.shape[1]
    _check_tensor(candidates, (batch, count), torch.int32, device, "candidates")
    if block_table.ndim != 2 or block_table.shape[1] < 1:
        raise ValueError("positive [B,pages] block table required")
    _check_tensor(
        block_table, (batch, block_table.shape[1]), torch.int32, device, "block table"
    )
    if not isinstance(page_size, int) or page_size not in (32, 64, 128):
        raise ValueError("candidate page_size must be 32, 64 or 128")
    if (
        not isinstance(num_physical_pages, int)
        or not 0 < num_physical_pages * (page_size // 8) < 2**31
    ):
        raise ValueError("positive encoded physical page count must fit int32")
    if not isinstance(max_context_len, int) or not 1 <= max_context_len <= min(
        2**31 - 1, block_table.shape[1] * page_size
    ):
        raise ValueError("positive int32 logical context must fit the block table")
    if batch * tr.cdiv(count * 8, 512) * 512 >= 2**31:
        raise ValueError("candidate padded output element span must fit int32")
    config = (batch, count, page_size, num_physical_pages, max_context_len, device)
    if out is None:
        encoded_bytes, mask_bytes = workspace_size(batch, count)
        storage = torch.empty(
            encoded_bytes + mask_bytes + batch * 4, device=device, dtype=torch.uint8
        )
        encoded = storage[: batch * count * 4].view(torch.int32).view(batch, count)
        mask_stride = tr.cdiv(count, 4) * 4
        mask = storage[encoded_bytes : encoded_bytes + batch * mask_stride].view(
            batch, mask_stride
        )
        snapshot = storage[encoded_bytes + mask_bytes :].view(torch.int32)
        out = _CandidateMetadata(*config, storage, encoded, mask, snapshot)
    elif not isinstance(out, _CandidateMetadata) or config != (
        out.batch,
        out.count,
        out.page_size,
        out.num_physical_pages,
        out.max_context_len,
        out.device,
    ):
        raise ValueError("metadata output configuration must match preparation")
    if any(_overlaps(out._storage, x) for x in (visible, block_table, candidates)):
        raise ValueError("metadata storage must not overlap publication inputs")
    with torch.cuda.device(device):
        _candidate_metadata[(batch, tr.cdiv(count, 256))](
            candidates,
            block_table,
            visible,
            out._encoded,
            out._mask,
            count,
            block_table.shape[1],
            page_size,
            num_physical_pages,
            max_context_len,
            256,
            SNAPSHOT=out._visible,
            num_warps=4,
        )
    return out


def candidate_scores_fp32(q_data, q_scales, kv_cache, weights, metadata, *, out=None):
    if not isinstance(metadata, _CandidateMetadata):
        raise ValueError(
            "metadata must come from prepare_deepseek_v41_candidate_metadata"
        )
    batch, page = _validate_layer_inputs(q_data, q_scales, kv_cache, weights)
    if (batch, page, kv_cache.shape[0], q_data.device) != (
        metadata.batch,
        metadata.page_size,
        metadata.num_physical_pages,
        metadata.device,
    ):
        raise ValueError("layer configuration must match candidate metadata")
    inputs = (q_data, q_scales, kv_cache, weights)
    if any(_overlaps(metadata._storage, x) for x in inputs):
        raise ValueError("metadata storage must not overlap layer inputs")
    out = _scores_output(
        q_data,
        q_scales,
        kv_cache,
        weights,
        metadata.count * 8,
        out,
        (metadata._storage,),
    )
    _check_candidate_layout(q_data, q_scales, kv_cache, weights, out)
    with torch.cuda.device(q_data.device):
        # Address-width dispatch belongs to each layer's complete strided
        # pool, not to the first layer that consumes this metadata.
        return _launch_candidate(
            q_data,
            q_scales,
            kv_cache,
            weights,
            out,
            metadata._encoded,
            metadata._visible,
            metadata._mask,
        )
