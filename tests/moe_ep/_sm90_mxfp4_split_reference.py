"""Test-owned reference helpers for SM90 MXFP4 split handoff/routing.

This module imports neither the donor nor the vendored kernel implementation.
It deliberately models the public numerical and metadata ABI described by the
migration specification: E4M3 handoff payload, FP32 per-token/K64 scale, and
packed ``(source_rank, source_topk, source_token)`` metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


E4M3_MAX = 448.0
K64 = 64
SCALE_EPSILON = 1.0e-30

RoutingCase = Literal[
    "balanced",
    "skewed",
    "remote_heavy",
    "masked",
    "edge",
]


@dataclass(frozen=True)
class HandoffValidation:
    valid_rows: int
    experts_with_routes: int
    e4m3_values: int
    fp32_scales: int


def quantize_handoff_reference(
    values: torch.Tensor,
    *,
    block_k: int = K64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` using one exact FP32 scale per K64 block.

    Reciprocal multiplication matches the Hopper epilogue/reference contract;
    this matters for bit-exact E4M3 bytes near rounding boundaries.
    """
    if values.ndim < 2:
        raise ValueError("handoff values must have at least two dimensions")
    cols = int(values.shape[-1])
    if cols % block_k:
        raise ValueError(f"handoff K={cols} must be divisible by {block_k}")
    flat = values.to(torch.float32).reshape(-1, cols)
    blocks = flat.reshape(flat.shape[0], cols // block_k, block_k)
    scale = (blocks.abs().amax(dim=-1) * (1.0 / E4M3_MAX)).clamp_min(SCALE_EPSILON)
    quant = (blocks * torch.reciprocal(scale.unsqueeze(-1))).reshape_as(flat)
    q = quant.to(torch.float8_e4m3fn).reshape(values.shape)
    return q, scale.reshape(*values.shape[:-1], cols // block_k).to(torch.float32)


def dequantize_handoff_reference(
    payload: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_k: int = K64,
) -> torch.Tensor:
    if payload.dtype is not torch.float8_e4m3fn:
        raise ValueError(f"payload must be E4M3, got {payload.dtype}")
    cols = int(payload.shape[-1])
    expected = (*payload.shape[:-1], cols // block_k)
    if tuple(scale.shape) != expected or scale.dtype is not torch.float32:
        raise ValueError(
            f"scale must be FP32 {expected}, got {scale.dtype} {tuple(scale.shape)}"
        )
    return payload.to(torch.float32) * scale.repeat_interleave(block_k, dim=-1)


def pack_route_metadata(
    source_rank: int,
    source_token: int,
    source_topk: int,
) -> int:
    for name, value, limit in (
        ("source_rank", source_rank, 1 << 16),
        ("source_topk", source_topk, 1 << 16),
        ("source_token", source_token, 1 << 32),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < limit
        ):
            raise ValueError(f"{name}={value!r} is outside [0, {limit})")
    return (source_rank << 48) | (source_topk << 32) | source_token


def unpack_route_metadata(metadata: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if metadata.dtype is torch.uint8:
        if metadata.ndim != 2 or metadata.shape[1] != 8:
            raise ValueError("uint8 metadata must have shape [rows,8]")
        packed = metadata.contiguous().view(torch.int64).reshape(-1)
    elif metadata.dtype is torch.int64 and metadata.ndim == 1:
        packed = metadata
    else:
        raise ValueError("metadata must be uint8 [rows,8] or int64 [rows]")
    source_token = torch.bitwise_and(packed, 0xFFFFFFFF)
    source_topk = torch.bitwise_and(torch.bitwise_right_shift(packed, 32), 0xFFFF)
    source_rank = torch.bitwise_and(torch.bitwise_right_shift(packed, 48), 0xFFFF)
    return source_rank, source_token, source_topk


def make_routing_case(
    *,
    case: RoutingCase,
    rank: int,
    world_size: int,
    num_tokens: int,
    top_k: int,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic routes with a documented communication pattern."""
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size")
    if num_tokens < 0 or top_k <= 0 or local_experts <= 0:
        raise ValueError("invalid token/top-k/expert geometry")
    token = torch.arange(num_tokens, dtype=torch.int64).view(-1, 1)
    slot = torch.arange(top_k, dtype=torch.int64).view(1, -1)

    if case == "balanced":
        owner = (rank + token + slot) % world_size
        expert = (token * 3 + slot) % local_experts
    elif case == "skewed":
        owner = torch.zeros((num_tokens, top_k), dtype=torch.int64)
        expert = torch.zeros_like(owner)
        if num_tokens and top_k > 1:
            owner[:, -1] = (rank + 1) % world_size
            expert[:, -1] = 1 % local_experts
    elif case == "remote_heavy":
        owner = torch.full(
            (num_tokens, top_k),
            (rank + 1) % world_size,
            dtype=torch.int64,
        )
        expert = (token + slot * 5) % local_experts
    elif case in ("masked", "edge"):
        owner = (rank + token + slot) % world_size
        expert = (token + 2 * slot) % local_experts
    else:
        raise ValueError(f"unknown routing case {case!r}")

    ids = (owner * local_experts + expert).contiguous()
    weights = (
        ((slot + 1).to(torch.float32) / float(top_k * (top_k + 1) // 2))
        .expand(num_tokens, top_k)
        .contiguous()
    )
    if case == "masked" and num_tokens:
        mask = (token + slot) % 3 == 0
        ids[mask] = -1
        weights[mask] = 0.0
    return ids, weights


def validate_route_indexed_handoff(
    *,
    actual_payload: torch.Tensor,
    actual_scale: torch.Tensor,
    actual_metadata: torch.Tensor,
    valid_counts: list[int] | tuple[int, ...],
    route_payload: torch.Tensor,
    route_scale: torch.Tensor,
    global_topk_idx: torch.Tensor,
    target_rank: int,
    local_experts: int,
    token_padding_block: int,
    block_k: int = K64,
) -> HandoffValidation:
    """Check every valid physical row against route-indexed exact references."""
    if actual_payload.dtype is not torch.float8_e4m3fn:
        raise AssertionError("actual handoff payload is not E4M3")
    if actual_scale.dtype is not torch.float32:
        raise AssertionError("actual handoff scale is not FP32")
    if route_payload.dtype is not torch.float8_e4m3fn:
        raise AssertionError("route payload reference is not E4M3")
    if route_scale.dtype is not torch.float32:
        raise AssertionError("route scale reference is not FP32")
    if route_payload.shape[:-1] != global_topk_idx.shape:
        raise AssertionError("route payload/routing shapes disagree")
    if route_scale.shape[:-1] != global_topk_idx.shape:
        raise AssertionError("route scale/routing shapes disagree")
    if route_payload.shape[-1] != actual_payload.shape[-1]:
        raise AssertionError("handoff payload K mismatch")
    if route_scale.shape[-1] != route_payload.shape[-1] // block_k:
        raise AssertionError("route reference is not K64-scaled")

    source_rank, source_token, source_topk = unpack_route_metadata(actual_metadata)
    world_size, tokens, top_k = global_topk_idx.shape
    target_first = target_rank * local_experts
    target_route_mask = (global_topk_idx >= target_first) & (
        global_topk_idx < target_first + local_experts
    )
    seen_routes = torch.zeros_like(target_route_mask, dtype=torch.bool)
    physical_row = 0
    valid_rows = 0
    experts_with_routes = 0
    for local_expert, count_value in enumerate(valid_counts):
        count = int(count_value)
        if count < 0:
            raise AssertionError("negative handoff route count")
        padded = (count + token_padding_block - 1) // token_padding_block
        padded *= token_padding_block
        if physical_row + padded > actual_payload.shape[0]:
            raise AssertionError("route counts exceed physical handoff pool")
        if count:
            experts_with_routes += 1
            rows = slice(physical_row, physical_row + count)
            sr = source_rank[rows]
            st = source_token[rows]
            sk = source_topk[rows]
            in_bounds = (sr < world_size) & (st < tokens) & (sk < top_k)
            if not bool(in_bounds.all().item()):
                raise AssertionError("handoff metadata is out of bounds")
            expected_expert = target_first + local_expert
            if not bool((global_topk_idx[sr, st, sk] == expected_expert).all().item()):
                raise AssertionError("handoff metadata maps to the wrong expert")
            route_linear = (sr * tokens + st) * top_k + sk
            if int(torch.unique(route_linear).numel()) != count:
                raise AssertionError("duplicate handoff route metadata")
            seen_flat = seen_routes.view(-1)
            if bool(seen_flat[route_linear].any().item()):
                raise AssertionError("duplicate handoff route metadata")
            seen_flat[route_linear] = True
            expected_q_bytes = route_payload.contiguous().view(torch.uint8)[sr, st, sk]
            if not torch.equal(
                actual_payload.contiguous().view(torch.uint8)[rows],
                expected_q_bytes,
            ):
                raise AssertionError("E4M3 handoff byte mismatch")
            actual_block_scale = actual_scale[rows, : route_scale.shape[-1]]
            expected_block_scale = route_scale[sr, st, sk]
            if not bool(
                (
                    torch.isfinite(actual_block_scale)
                    & torch.isfinite(expected_block_scale)
                    & (actual_block_scale > 0)
                    & (expected_block_scale > 0)
                )
                .all()
                .item()
            ):
                raise AssertionError("FP32 K64 scales must be finite and positive")
            actual_bits = actual_block_scale.contiguous().view(torch.int32)
            expected_bits = expected_block_scale.contiguous().view(torch.int32)
            if not torch.equal(actual_bits, expected_bits):
                raise AssertionError("FP32 K64 handoff scale bit mismatch")
            valid_rows += count
        physical_row += padded

    expected_routes = int(target_route_mask.sum().item())
    if valid_rows != expected_routes:
        raise AssertionError(
            f"checked {valid_rows} rows but routing targets {expected_routes}"
        )
    if not torch.equal(seen_routes, target_route_mask):
        missing = int((target_route_mask & ~seen_routes).sum().item())
        unexpected = int((seen_routes & ~target_route_mask).sum().item())
        raise AssertionError(
            "handoff metadata does not exactly cover target routes: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return HandoffValidation(
        valid_rows=valid_rows,
        experts_with_routes=experts_with_routes,
        e4m3_values=valid_rows * route_payload.shape[-1],
        fp32_scales=valid_rows * route_scale.shape[-1],
    )


__all__ = [
    "E4M3_MAX",
    "HandoffValidation",
    "K64",
    "make_routing_case",
    "pack_route_metadata",
    "quantize_handoff_reference",
    "unpack_route_metadata",
    "validate_route_indexed_handoff",
]
