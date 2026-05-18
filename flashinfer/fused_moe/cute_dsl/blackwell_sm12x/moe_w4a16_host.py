"""Host-side helpers for the CuTeDSL W4A16 MoE path."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_W4A16_ALLOWED_ROUTED_SIZES = (8, 16, 32, 48, 64)
_ROUTED_SIZE_TARGET_FILL = 0.9
_SUPPORTED_ACTIVATIONS = {"silu", "relu2"}


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
    fc1_c_tmp: torch.Tensor | None = None
    fc2_c_tmp: torch.Tensor | None = None
    packed_route_indices: torch.Tensor | None = None
    block_expert_ids: torch.Tensor | None = None
    packed_route_count: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None


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


def validate_activation(activation: str) -> bool:
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    return activation == "silu"


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


def normalize_expert_block_scales(
    blockscale: torch.Tensor,
    *,
    num_experts: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    if cols % 16 != 0:
        raise ValueError(
            f"W4A16 block scales require cols to be a multiple of 16, got {cols}"
        )

    cols_blocks = int(cols) // 16
    cols_padded = ((cols_blocks + 3) // 4) * 4
    rows_padded = ((int(rows) + 127) // 128) * 128
    row_groups = rows_padded // 128
    col_groups = cols_padded // 4
    expected_elements = int(num_experts) * rows_padded * cols_padded

    # FlashInfer's existing public path passes the logical 6D MMA view from
    # convert_sf_to_mma_layout: (32, 4, row_groups, 4, col_groups, E).
    if tuple(blockscale.shape) == (32, 4, row_groups, 4, col_groups, int(num_experts)):
        from flashinfer.cute_dsl.utils import convert_sf_from_mma_layout

        return convert_sf_from_mma_layout(
            blockscale,
            m=int(rows),
            k=int(cols),
            num_groups=int(num_experts),
            sf_vec_size=16,
        ).reshape(int(num_experts), rows_padded, cols_padded)

    if blockscale.numel() != expected_elements:
        raise ValueError(
            "W4A16 block scales must be either FlashInfer's 6D MMA layout "
            f"{(32, 4, row_groups, 4, col_groups, int(num_experts))} or "
            "expert-leading swizzled storage with "
            f"{expected_elements} elements; got shape {tuple(blockscale.shape)}"
        )

    if blockscale.ndim > 0 and int(blockscale.shape[0]) == int(num_experts):
        return blockscale
    return blockscale.reshape(int(num_experts), rows_padded, cols_padded)


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


def max_packed_route_slots(numel: int, block_size: int, num_experts: int) -> int:
    max_packed_routes = int(numel) + int(num_experts) * (int(block_size) - 1)
    if int(numel) < int(num_experts):
        max_packed_routes = min(
            int(numel) * int(block_size),
            max_packed_routes,
        )
    return max_packed_routes


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
) -> int:
    elements = min(
        int(size_n) * int(route_slots),
        int(sms) * 4 * int(moe_block_size) * 256,
    )
    if moe_block_size == 8:
        elements *= 2
    return max(elements, 1)


def plan_w4a16_buffers(
    prepared,
    *,
    m: int,
    topk: int,
    route_num_experts: int | None = None,
    sms: int | None = None,
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
    block_size_m = select_route_block_size_m(m, topk, route_num_experts)
    route_slots = max_packed_route_slots(routed_rows, block_size_m, route_num_experts)
    route_blocks = (route_slots + block_size_m - 1) // block_size_m
    scratch_sms = 1 if sms is None else int(sms)
    return W4A16BufferPlan(
        routed_rows=routed_rows,
        fc1_cols=fc1_cols,
        route_slots=route_slots,
        route_blocks=route_blocks,
        fc1_c_tmp_elements=packed_gemm_scratch_elements(
            size_n=fc1_cols,
            route_slots=route_slots,
            moe_block_size=block_size_m,
            sms=scratch_sms,
        ),
        fc2_c_tmp_elements=packed_gemm_scratch_elements(
            size_n=hidden_size,
            route_slots=route_slots,
            moe_block_size=block_size_m,
            sms=scratch_sms,
        ),
        intermediate_cache13_elements=routed_rows * max(fc1_cols, hidden_size),
        intermediate_cache2_elements=routed_rows * intermediate_size,
    )


def make_w4a16_packed_buffers(
    prepared,
    *,
    m: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    route_num_experts: int | None = None,
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
        output=torch.empty((m, prepared.hidden_size), dtype=dtype, device=device),
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
    )


__all__ = [
    "W4A16BufferPlan",
    "W4A16PackedBuffers",
    "W4A16PackedShape",
    "make_w4a16_packed_buffers",
    "max_w4a16_route_capacity",
    "normalize_expert_block_scales",
    "packed_gemm_scratch_elements",
    "max_packed_route_slots",
    "plan_w4a16_buffers",
    "reorder_w13_to_gate_up",
    "select_route_block_size_m",
    "unswizzle_block_scale",
    "unswizzle_expert_scales",
    "validate_activation",
    "validate_w4a16_packed_inputs",
]
