"""Ownership and in-place refresh of paired MegaMoE weight tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...weights import MoEWeightPack


WeightPairs = tuple[
    tuple[torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]


def _flatten(weights: WeightPairs) -> list[torch.Tensor | None]:
    if not isinstance(weights, tuple) or len(weights) != 2:
        raise ValueError("prepared weights must be a (fc1, fc2) tuple")
    tensors: list[torch.Tensor | None] = []
    for pair in weights:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("each prepared weight must be a (data, scale) tuple")
        if not isinstance(pair[0], torch.Tensor) or (
            pair[1] is not None and not isinstance(pair[1], torch.Tensor)
        ):
            raise ValueError(
                "prepared weights require tensor data and tensor/None scales"
            )
        tensors.extend(pair)
    return tensors


def _span(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError("weight snapshots require materialized strided tensors")
    start = tensor.data_ptr()
    size = (
        1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True))
        if tensor.numel()
        else 0
    )
    return tensor.device, start, start + size * tensor.element_size()


def _overlaps(left, right) -> bool:
    return left[0] == right[0] and left[1] < right[2] and right[1] < left[2]


def check_weight_refresh(out: WeightPairs | None) -> None:
    """Reject capture before preparation can allocate or enqueue quantization."""
    if out is not None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise ValueError("weight refresh cannot run during CUDA graph capture")
        _flatten(out)


@torch.no_grad()
def finish_weight_snapshot(
    prepared: WeightPairs,
    source: MoEWeightPack,
    *,
    out: WeightPairs | None = None,
) -> WeightPairs:
    """Detach a prepared snapshot from source storage, or refresh an existing one.

    Native layouts are preserved. Validation of every destination precedes
    copying either projection, so a metadata error cannot leave mixed weights.
    Refresh is host-driven, outside capture; callers order it against launches
    on all streams that read the snapshot. No device synchronization is added.
    """
    check_weight_refresh(out)
    tensors = _flatten(prepared)
    sources = [
        tensor
        for tensor in (source.w13, source.w13_scale, source.w2, source.w2_scale)
        if tensor is not None
    ]
    source_spans = [_span(tensor) for tensor in sources]
    if out is None:
        snapshot = []
        for tensor in tensors:
            if tensor is not None:
                tensor = tensor.detach()
                if any(_overlaps(_span(tensor), span) for span in source_spans):
                    if tensor.element_size() == 1:
                        tensor = (
                            tensor.view(torch.uint8)
                            .clone(memory_format=torch.preserve_format)
                            .view(tensor.dtype)
                        )
                    else:
                        tensor = tensor.clone(memory_format=torch.preserve_format)
            snapshot.append(tensor)
        return (snapshot[0], snapshot[1]), (snapshot[2], snapshot[3])

    destinations = _flatten(out)
    destination_spans: list[tuple[torch.device, int, int]] = []
    for src, dst in zip(tensors, destinations, strict=True):
        if src is None or dst is None:
            if src is not dst:
                raise ValueError("out must preserve the prepared weight scale format")
            continue
        if (
            src.shape != dst.shape
            or src.stride() != dst.stride()
            or src.dtype != dst.dtype
            or src.device != dst.device
            or dst.requires_grad
            or dst.data_ptr() % 16
        ):
            raise ValueError(
                "out must match prepared weight shape, stride, dtype and device, "
                "be 16-byte aligned and must not require gradients"
            )
        span = _span(dst)
        if any(_overlaps(span, other) for other in source_spans + destination_spans):
            raise ValueError(
                "out tensors must not overlap source weights or each other"
            )
        destination_spans.append(span)
    # Rollout buffers may have been allocated under inference_mode. They are
    # detached snapshots, so allow their refresh from a training no_grad context.
    with torch.inference_mode():
        for src, dst in zip(tensors, destinations, strict=True):
            if src is not None:
                if src.element_size() == 1:
                    dst.view(torch.uint8).copy_(src.view(torch.uint8))
                else:
                    dst.copy_(src)
    return out
