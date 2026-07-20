# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/w4a16/host.py @ 3663c726 (2026-07-03) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Host-side helpers for the CuTeDSL W4A16 MoE path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flashinfer.experimental.sm12x.moe._shared.kernels.activations import (
    SUPPORTED_MOE_ACTIVATIONS,
    is_gated_moe_activation,
    normalize_moe_activation,
)


_W4A16_ALLOWED_ROUTED_SIZES = (8, 16, 32, 48, 64)
_ROUTED_SIZE_TARGET_FILL = 0.9
_SUPPORTED_ACTIVATIONS = SUPPORTED_MOE_ACTIVATIONS


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


def validate_nf3_moe_inputs(
    w13_codes: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_codes: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    activation: str,
) -> W4A16PackedShape:
    """Validate NF3 ("nf3_2p1") code/scale inputs to the packer.

    Codes are integer tensors of values 0..7 in kernel-native output order:
        w13_codes [E, 2*I, hidden]  w2_codes [E, hidden, I]
    Scales are per-K/32-group ``t_s`` floats:
        w13_scale [E, 2*I, hidden//32]  w2_scale [E, hidden, I//32]
    The NF3 runtime contract requires K % 32 == 0 and N % 64 == 0 for both
    GEMMs (the scale byte encoding and the 64-N tile packing).
    """
    is_gated = validate_activation(activation)
    _int_dtypes = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    if w13_codes.dtype not in _int_dtypes or w2_codes.dtype not in _int_dtypes:
        raise TypeError("NF3 codes must be integer tensors of values 0..7")

    num_experts = int(w2_codes.shape[0])
    hidden_size = int(w2_codes.shape[1])
    intermediate_size = int(w2_codes.shape[2])
    w13_rows = intermediate_size * (2 if is_gated else 1)
    if tuple(w13_codes.shape) != (num_experts, w13_rows, hidden_size):
        raise ValueError(
            f"expected w13_codes shape {(num_experts, w13_rows, hidden_size)}, "
            f"got {tuple(w13_codes.shape)}"
        )
    if tuple(w2_codes.shape) != (num_experts, hidden_size, intermediate_size):
        raise ValueError(
            f"expected w2_codes shape {(num_experts, hidden_size, intermediate_size)}, "
            f"got {tuple(w2_codes.shape)}"
        )
    for name, size_k, size_n in (
        ("w13", hidden_size, w13_rows),
        ("w2", intermediate_size, hidden_size),
    ):
        if int(size_k) % 32 != 0:
            raise ValueError(f"NF3 {name} requires K % 32 == 0, got {size_k}")
        if int(size_n) % 64 != 0:
            raise ValueError(f"NF3 {name} requires N % 64 == 0, got {size_n}")
    if tuple(w13_scale.shape) != (num_experts, w13_rows, hidden_size // 32):
        raise ValueError(
            f"expected w13_scale shape {(num_experts, w13_rows, hidden_size // 32)}, "
            f"got {tuple(w13_scale.shape)}"
        )
    if tuple(w2_scale.shape) != (num_experts, hidden_size, intermediate_size // 32):
        raise ValueError(
            "expected w2_scale shape "
            f"{(num_experts, hidden_size, intermediate_size // 32)}, "
            f"got {tuple(w2_scale.shape)}"
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
    sms: int,
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
    scratch_sms = int(sms)
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
    "packed_gemm_scratch_elements",
    "max_packed_route_slots",
    "plan_w4a16_buffers",
    "reorder_w13_to_gate_up",
    "route_pack_numel_capacity",
    "route_pack_token_capacity",
    "select_route_block_size_m",
    "unswizzle_block_scale",
    "unswizzle_expert_scales",
    "validate_activation",
    "validate_nf3_moe_inputs",
    "validate_w4a16_packed_inputs",
]
