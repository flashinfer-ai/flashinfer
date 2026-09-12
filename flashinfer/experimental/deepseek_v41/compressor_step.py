# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Two-launch CSA2 dual projection, state update, pair pooling and normalization."""

import math

import torch

from .compressor import _decode_partials
from .compressor_projection import _partials
from ._utils import _overlaps


def compressor_step(
    x,
    wkv,
    wgate,
    kv_state,
    score_state,
    norm_weight,
    starts,
    *,
    eps=1e-20,
    workspace=None,
    out=None,
    positions=None,
):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 compressor step currently requires SM100/SM103")
    if x.ndim != 2 or not 1 <= x.shape[0] <= 16 or x.shape[1] != 5120:
        raise ValueError("compressor step requires BF16 [B,5120], B in1..16")
    batch = x.shape[0]
    declarations = (
        (x, (batch, 5120), (torch.bfloat16,)),
        (wkv, (512, 5120), (torch.float32,)),
        (wgate, (512, 5120), (torch.float32,)),
        (kv_state, (batch, 2, 512), (torch.float32,)),
        (score_state, (batch, 2, 512), (torch.float32,)),
        (norm_weight, (512,), (torch.bfloat16, torch.float32)),
        (starts, (batch,), (torch.int32,)),
    )
    for tensor, shape, dtypes in declarations:
        if (
            tensor.shape != shape
            or tensor.dtype not in dtypes
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("compressor step input/state declaration mismatch")
        if tensor.requires_grad and torch.is_grad_enabled():
            raise ValueError("compressor step is inference-only")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive finite")
    outputs = []
    for tensor, shape, dtype in (
        (workspace, (8, 2, batch, 512), torch.float32),
        (out, (batch, 512), torch.bfloat16),
        (positions, (batch,), torch.int32),
    ):
        if tensor is None:
            tensor = torch.empty(shape, device=x.device, dtype=dtype)
        elif (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("compressor step output/workspace declaration mismatch")
        outputs.append(tensor)
    destinations = (kv_state, score_state, *outputs)
    for index, destination in enumerate(destinations):
        if any(
            _overlaps(destination, source)
            for source in (x, wkv, wgate, norm_weight, starts, *destinations[:index])
        ):
            raise ValueError("compressor step buffers may not overlap")
    workspace, out, positions = outputs
    with torch.cuda.device(x.device):
        _partials[(16, 8)](x, wkv, wgate, workspace, batch, num_warps=4, num_stages=3)
        _decode_partials[(batch,)](
            workspace,
            kv_state,
            score_state,
            norm_weight,
            starts,
            out,
            positions,
            batch,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, positions
