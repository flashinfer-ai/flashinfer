"""Cache identity for W4A16 route-pack launch preparation."""

from __future__ import annotations

from typing import Any


def route_pack_prewarm_key(
    device_type: str,
    device_index: int,
    route_ids_dtype: Any,
    token_count: int,
    top_k: int,
    packed_route_slots: int,
    route_blocks: int,
    block_size: int,
    num_experts: int,
    mapped: bool,
) -> tuple[object, ...]:
    """Identify every input that can change a route-pack specialization.

    ``token_count`` and ``top_k`` remain separate because equal routed-row
    counts can select different ``NUMEL_CAPACITY`` values. Caller-owned slot
    and block capacities are included because a fixed arena can replace the
    power-of-two bucket selected for the live shape.
    """

    values = {
        "token_count": token_count,
        "top_k": top_k,
        "packed_route_slots": packed_route_slots,
        "route_blocks": route_blocks,
        "block_size": block_size,
        "num_experts": num_experts,
    }
    normalized = {name: int(value) for name, value in values.items()}
    invalid = {name: value for name, value in normalized.items() if value < 1}
    if invalid:
        raise ValueError(
            f"route-pack prewarm dimensions must be positive: {invalid}"
        )
    return (
        str(device_type),
        int(device_index),
        str(route_ids_dtype),
        normalized["token_count"],
        normalized["top_k"],
        normalized["packed_route_slots"],
        normalized["route_blocks"],
        normalized["block_size"],
        normalized["num_experts"],
        bool(mapped),
    )


__all__ = ["route_pack_prewarm_key"]
