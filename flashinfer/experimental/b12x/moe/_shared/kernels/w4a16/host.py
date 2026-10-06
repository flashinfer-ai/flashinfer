"""Host-side helpers for the CuTeDSL W4A16 MoE path."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from b12x._lib.env import env_flag
from b12x._lib.quant.block_codec import BLOCK_CODECS
from b12x.moe._shared.kernels.activations import (
    SUPPORTED_MOE_ACTIVATIONS,
    is_gated_moe_activation,
    normalize_moe_activation,
)


_W4A16_ALLOWED_ROUTED_SIZES = (8, 16, 32, 48, 64)
_ROUTED_SIZE_TARGET_FILL = 0.9
_SUPPORTED_ACTIVATIONS = SUPPORTED_MOE_ACTIVATIONS


def trellis_decode_table() -> str:
    """Read the plan-time trellis decode-table selection; replay never reads it."""
    value = os.environ.get("B12X_TRELLIS_DECODE_TABLE", "auto")
    if value not in {"auto", "compact", "full"}:
        raise ValueError("B12X_TRELLIS_DECODE_TABLE must be auto, compact, or full")
    return value


def prefill_fused_sum_enabled() -> bool:
    """Enable direct FP32 route reduction for large-M W4A16 launches.

    The FC2 epilogue uses relaxed FP32 global reductions. Disable this path
    when bitwise-identical route accumulation order is required.
    """
    return env_flag("W4A16_PREFILL_FUSED_SUM")


def prefill_fused_sum_eligible(
    *,
    dtype: torch.dtype | str,
    m: int,
    full_rotation: bool,
    weight_layout: str,
    collect_activation_amax: bool,
    enabled: bool | None = None,
) -> bool:
    """Return whether one W4A16 launch may use direct route reduction.

    Args:
        dtype: Activation element dtype as a PyTorch dtype or kernel dtype name.
        m: Planned token capacity, or exact rows for an unprepared launch.
        full_rotation: Whether the launch uses full-rotation Trellis weights.
        weight_layout: Prepared W4A16 weight layout.
        collect_activation_amax: Whether the launch records activation maxima.
        enabled: Frozen feature selection. ``None`` reads the process setting.

    Returns:
        ``True`` when planning and execution may use the FP32 accumulator.
    """
    if enabled is None:
        enabled = prefill_fused_sum_enabled()
    element_dtype = str(dtype).removeprefix("torch.")
    return bool(
        enabled
        and element_dtype in {"bfloat16", "bf16"}
        and int(m) > 8
        and not full_rotation
        and weight_layout in {"packed", "modelopt"}
        and not collect_activation_amax
    )


@dataclass(frozen=True)
class W4A16PackedShape:
    num_experts: int
    hidden_size: int
    intermediate_size: int
    w13_rows: int
    is_gated: bool


@dataclass(frozen=True)
class W4A16PackedBuffers:
    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    output: torch.Tensor
    prefill_sum_accum: torch.Tensor | None = None
    fc1_c_tmp: torch.Tensor | None = None
    fc2_c_tmp: torch.Tensor | None = None
    packed_route_indices: torch.Tensor | None = None
    block_expert_ids: torch.Tensor | None = None
    packed_route_count: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    expert_counts: torch.Tensor | None = None
    rotation_a_gate: torch.Tensor | None = None
    rotation_a_up: torch.Tensor | None = None


@dataclass(frozen=True)
class W4A16BufferPlan:
    routed_rows: int
    fc1_cols: int
    route_slots: int
    route_blocks: int
    fc1_c_tmp_elements: int
    fc2_c_tmp_elements: int
    intermediate_cache13_elements: int
    intermediate_cache2_elements: int
    block_size_m: int
    rotation_a_elements: int = 0
    prefill_sum_accum_elements: int = 0


def validate_activation(activation: str) -> bool:
    activation = normalize_moe_activation(activation)
    return is_gated_moe_activation(activation)


def validate_w4a16_packed_inputs(
    w13_fp4: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
) -> W4A16PackedShape:
    is_gated = validate_activation(activation)
    if w13_fp4.dtype != torch.uint8 or w2_fp4.dtype != torch.uint8:
        raise TypeError("packed FP4 weights must be torch.uint8")
    if (
        w13_global_scale.dtype != torch.float32
        or w2_global_scale.dtype != torch.float32
    ):
        raise TypeError("global scales must be torch.float32")

    num_experts = int(w13_fp4.shape[0])
    hidden_size = int(w2_fp4.shape[1])
    intermediate_size = int(w2_fp4.shape[2] * 2)
    w13_rows = intermediate_size * (2 if is_gated else 1)
    if tuple(w13_fp4.shape) != (num_experts, w13_rows, hidden_size // 2):
        raise ValueError(
            f"expected w13_fp4 shape {(num_experts, w13_rows, hidden_size // 2)}, "
            f"got {tuple(w13_fp4.shape)}"
        )
    if tuple(w2_fp4.shape) != (num_experts, hidden_size, intermediate_size // 2):
        raise ValueError(
            f"expected w2_fp4 shape {(num_experts, hidden_size, intermediate_size // 2)}, "
            f"got {tuple(w2_fp4.shape)}"
        )
    return W4A16PackedShape(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        w13_rows=w13_rows,
        is_gated=is_gated,
    )


def unswizzle_block_scale(
    swizzled_scale: torch.Tensor, rows: int, cols_blocks: int
) -> torch.Tensor:
    cols_padded = ((cols_blocks + 3) // 4) * 4
    rows_padded = ((rows + 127) // 128) * 128
    unswizzled = swizzled_scale.view(torch.float8_e4m3fn).reshape(
        rows_padded // 128,
        cols_padded // 4,
        32,
        4,
        4,
    )
    unswizzled = unswizzled.permute(0, 3, 2, 1, 4).contiguous()
    unswizzled = unswizzled.reshape(rows_padded, cols_padded)
    return unswizzled[:rows, :cols_blocks].to(torch.float32)


def unswizzle_expert_scales(
    swizzled: torch.Tensor,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    if swizzled.dtype != torch.float8_e4m3fn:
        swizzled = swizzled.view(torch.float8_e4m3fn)
    scales = [
        unswizzle_block_scale(swizzled[e], rows, cols // 16).to(torch.float8_e4m3fn)
        for e in range(swizzled.shape[0])
    ]
    return torch.stack(scales, dim=0).contiguous()


def reorder_w13_to_gate_up(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    *,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = int(intermediate_size)
    return (
        torch.cat([w13[:, half:], w13[:, :half]], dim=1).contiguous(),
        torch.cat([w13_scale[:, half:], w13_scale[:, :half]], dim=1).contiguous(),
    )


def select_route_block_size_m(m: int, topk: int, num_experts: int) -> int:
    avg_routes_per_expert = (int(m) * int(topk)) / int(num_experts)
    for routed_size in _W4A16_ALLOWED_ROUTED_SIZES:
        if avg_routes_per_expert < _ROUTED_SIZE_TARGET_FILL * routed_size:
            return routed_size
    return _W4A16_ALLOWED_ROUTED_SIZES[-1]


def route_block_sizes_for_capacity(
    max_tokens: int,
    topk: int,
    num_experts: int,
) -> tuple[int, ...]:
    """Conservative block-size set reachable within a token capacity."""
    max_tokens = max(int(max_tokens), 1)
    first = select_route_block_size_m(1, topk, num_experts)
    last = select_route_block_size_m(max_tokens, topk, num_experts)
    first_idx = _W4A16_ALLOWED_ROUTED_SIZES.index(first)
    last_idx = _W4A16_ALLOWED_ROUTED_SIZES.index(last)
    return _W4A16_ALLOWED_ROUTED_SIZES[first_idx : last_idx + 1]


def max_packed_route_slots(numel: int, block_size: int, num_experts: int) -> int:
    max_packed_routes = int(numel) + int(num_experts) * (int(block_size) - 1)
    if int(numel) < int(num_experts):
        max_packed_routes = min(
            int(numel) * int(block_size),
            max_packed_routes,
        )
    return max_packed_routes


def route_pack_numel_capacity(numel: int, topk: int = 1) -> int:
    topk = max(int(topk), 1)
    tokens = (max(int(numel), 1) + topk - 1) // topk
    return route_pack_token_capacity(tokens, topk) * topk


def route_pack_token_capacity(tokens: int, topk: int) -> int:
    del topk
    return 1 << (max(int(tokens), 1) - 1).bit_length()


def route_pack_warmup_token_counts(capacity: int) -> tuple[int, ...]:
    """Return live row counts covering capacity and scalar specializations.

    Triton specializes the runtime ``live_numel`` scalar on alignment as well
    as the constexpr route-capacity bucket. For a power-of-two bucket, the
    bucket maximum and its first live member cover both alignment classes
    reached by the GLM top-k route counts (for example 8 and 5 rows in the
    8-row bucket). Warming only bucket maxima leaves the less-aligned variant
    to compile on the first odd-sized request.
    """
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError(f"route-pack warmup capacity must be positive, got {capacity}")
    counts: list[int] = []
    bucket = 1
    while True:
        first_in_bucket = 1 if bucket == 1 else bucket // 2 + 1
        if first_in_bucket > capacity:
            break
        for count in (first_in_bucket, min(bucket, capacity)):
            if not counts or counts[-1] != count:
                counts.append(count)
        bucket *= 2
    return tuple(counts)


def route_pack_capacity(
    numel: int,
    block_size: int,
    num_experts: int,
    *,
    topk: int = 1,
    bucket_tokens: bool = True,
) -> tuple[int, int, int]:
    """Return the canonical routed-row, packed-slot, and block capacities."""
    numel_capacity = (
        route_pack_numel_capacity(numel, topk=topk)
        if bucket_tokens
        else max(int(numel), 1)
    )
    packed_routes = max_packed_route_slots(
        numel_capacity, int(block_size), int(num_experts)
    )
    route_blocks = (packed_routes + int(block_size) - 1) // int(block_size)
    return max(numel_capacity, 1), max(packed_routes, 1), max(route_blocks, 1)


def max_w4a16_route_capacity(routed_rows: int, num_experts: int) -> tuple[int, int]:
    route_slots = 0
    route_blocks = 0
    for block_size in _W4A16_ALLOWED_ROUTED_SIZES:
        slots = max_packed_route_slots(
            int(routed_rows), int(block_size), int(num_experts)
        )
        route_slots = max(route_slots, slots)
        route_blocks = max(
            route_blocks, (slots + int(block_size) - 1) // int(block_size)
        )
    return max(route_slots, 1), max(route_blocks, 1)


def packed_gemm_scratch_elements(
    *,
    size_n: int,
    route_slots: int,
    moe_block_size: int,
    sms: int,
    weight_layout: str = "packed",
) -> int:
    elements = min(
        int(size_n) * int(route_slots),
        int(sms) * 4 * int(moe_block_size) * 256,
    )
    if moe_block_size == 8:
        elements *= 2
        if weight_layout in BLOCK_CODECS:
            # Parallel split-K retains one M8 partial per resident CTA.
            elements = max(elements, int(sms) * 2 * 8 * 256)
    return max(elements, 1)


def plan_w4a16_buffers(
    prepared,
    *,
    m: int,
    topk: int,
    route_num_experts: int | None = None,
    sms: int,
    dtype: torch.dtype | None = None,
    full_rotation: bool = False,
    block_size_m: int | None = None,
    weight_layout: str | None = None,
    collect_activation_amax: bool = False,
    prefill_fused_sum: bool | None = None,
) -> W4A16BufferPlan:
    routed_rows = int(m) * int(topk)
    route_num_experts = (
        int(prepared.num_experts)
        if route_num_experts is None
        else int(route_num_experts)
    )
    intermediate_size = int(prepared.intermediate_size)
    hidden_size = int(prepared.hidden_size)
    fc1_cols = (2 if prepared.is_gated else 1) * intermediate_size
    if block_size_m is None:
        block_size_m = select_route_block_size_m(m, topk, route_num_experts)
    else:
        block_size_m = int(block_size_m)
        if block_size_m not in _W4A16_ALLOWED_ROUTED_SIZES:
            raise ValueError(
                "block_size_m must be one of "
                f"{_W4A16_ALLOWED_ROUTED_SIZES}, got {block_size_m}"
            )
    route_slots = max_packed_route_slots(routed_rows, block_size_m, route_num_experts)
    route_blocks = (route_slots + block_size_m - 1) // block_size_m
    # Small-M tensor-core decode may bypass expert packing and assign one full
    # M block to every top-k route. Its accumulation scratch is therefore
    # sized from ``routed_rows * block_size_m``, which can exceed the packed
    # upper bound once routed_rows > route_num_experts. Keep the generic buffer
    # helper graph-safe for every currently supported TC-decode shape.
    gemm_route_slots = route_slots
    if int(m) <= 8 and (bool(prepared.is_gated) or prepared.weight_layout in BLOCK_CODECS):
        gemm_route_slots = max(gemm_route_slots, routed_rows * block_size_m)
    scratch_sms = int(sms)
    weight_layout = str(
        weight_layout
        if weight_layout is not None
        else getattr(prepared, "weight_layout", "packed")
    )
    use_prefill_fused_sum = prefill_fused_sum_eligible(
        dtype=dtype if dtype is not None else "",
        m=m,
        full_rotation=full_rotation,
        weight_layout=weight_layout,
        collect_activation_amax=collect_activation_amax,
        enabled=prefill_fused_sum,
    )
    return W4A16BufferPlan(
        routed_rows=routed_rows,
        fc1_cols=fc1_cols,
        route_slots=route_slots,
        route_blocks=route_blocks,
        fc1_c_tmp_elements=packed_gemm_scratch_elements(
            size_n=fc1_cols,
            route_slots=gemm_route_slots,
            moe_block_size=block_size_m,
            sms=scratch_sms,
            weight_layout=prepared.weight_layout,
        ),
        fc2_c_tmp_elements=packed_gemm_scratch_elements(
            size_n=hidden_size,
            route_slots=gemm_route_slots,
            moe_block_size=block_size_m,
            sms=scratch_sms,
            weight_layout=prepared.weight_layout,
        ),
        intermediate_cache13_elements=(
            routed_rows * fc1_cols
            if use_prefill_fused_sum
            else routed_rows * max(fc1_cols, hidden_size)
        ),
        intermediate_cache2_elements=routed_rows * intermediate_size,
        block_size_m=block_size_m,
        rotation_a_elements=(routed_rows * hidden_size if full_rotation else 0),
        prefill_sum_accum_elements=(
            int(m) * hidden_size
            if use_prefill_fused_sum
            else 0
        ),
    )


def make_w4a16_packed_buffers(
    prepared,
    *,
    m: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    route_num_experts: int | None = None,
    full_rotation: bool = False,
    block_size_m: int | None = None,
) -> W4A16PackedBuffers:
    route_num_experts = (
        int(prepared.num_experts)
        if route_num_experts is None
        else int(route_num_experts)
    )
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=topk,
        route_num_experts=route_num_experts,
        sms=sms,
        dtype=dtype,
        full_rotation=full_rotation,
        block_size_m=block_size_m,
    )
    fc1_c_tmp = torch.empty(
        (plan.fc1_c_tmp_elements,),
        dtype=torch.float32,
        device=device,
    )
    fc2_c_tmp = torch.empty(
        (plan.fc2_c_tmp_elements,),
        dtype=torch.float32,
        device=device,
    )
    rotation_a_gate = (
        torch.empty(
            (plan.routed_rows, int(prepared.hidden_size)),
            dtype=torch.float16,
            device=device,
        )
        if full_rotation
        else None
    )
    rotation_a_up = (
        rotation_a_gate
        if full_rotation and bool(getattr(prepared, "intermediate_hadamard", False))
        else torch.empty(
            (plan.routed_rows, int(prepared.hidden_size)),
            dtype=torch.float16,
            device=device,
        )
        if full_rotation
        else None
    )
    return W4A16PackedBuffers(
        intermediate_cache13=torch.empty(
            (plan.intermediate_cache13_elements,),
            dtype=dtype,
            device=device,
        ),
        intermediate_cache2=torch.empty(
            (plan.routed_rows, int(prepared.intermediate_size)),
            dtype=dtype,
            device=device,
        ),
        output=torch.empty(
            (m, prepared.hidden_size),
            dtype=torch.float32 if full_rotation else dtype,
            device=device,
        ),
        prefill_sum_accum=(
            torch.empty(
                (plan.prefill_sum_accum_elements,),
                dtype=torch.float32,
                device=device,
            )
            if plan.prefill_sum_accum_elements
            else None
        ),
        fc1_c_tmp=fc1_c_tmp,
        fc2_c_tmp=fc2_c_tmp,
        packed_route_indices=torch.empty(
            (plan.route_slots,), dtype=torch.int32, device=device
        ),
        block_expert_ids=torch.empty(
            (plan.route_blocks,), dtype=torch.int32, device=device
        ),
        packed_route_count=torch.empty((1,), dtype=torch.int32, device=device),
        expert_offsets=torch.empty(
            (route_num_experts + 1,), dtype=torch.int32, device=device
        ),
        expert_counts=torch.empty(
            (route_num_experts,), dtype=torch.int32, device=device
        ),
        rotation_a_gate=rotation_a_gate,
        rotation_a_up=rotation_a_up,
    )


__all__ = [
    "W4A16BufferPlan",
    "W4A16PackedBuffers",
    "W4A16PackedShape",
    "make_w4a16_packed_buffers",
    "max_w4a16_route_capacity",
    "packed_gemm_scratch_elements",
    "max_packed_route_slots",
    "plan_w4a16_buffers",
    "reorder_w13_to_gate_up",
    "route_pack_numel_capacity",
    "route_pack_capacity",
    "route_pack_token_capacity",
    "route_pack_warmup_token_counts",
    "select_route_block_size_m",
    "unswizzle_block_scale",
    "unswizzle_expert_scales",
    "validate_activation",
    "validate_w4a16_packed_inputs",
]
