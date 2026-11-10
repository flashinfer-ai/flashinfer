# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both W4A16 planners must cover per-route blocks before graph capture."""

from types import SimpleNamespace

import pytest
import torch

from tests.moe.test_b12x_fused_moe import cute_dsl_available


@cute_dsl_available
@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 256])
@pytest.mark.parametrize("experts", [16, 512])
@pytest.mark.parametrize("gated", [False, True])
def test_w4a16_planners_cover_direct_and_packed_scratch(
    tokens, experts, gated, monkeypatch
):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        max_packed_route_slots,
        packed_gemm_scratch_elements,
        plan_w4a16_buffers,
        select_route_block_size_m,
    )

    h, i, topk, sms = 2048, 512, 10, 142
    prepared = SimpleNamespace(
        num_experts=experts, hidden_size=h, intermediate_size=i, is_gated=gated
    )
    plan = plan_w4a16_buffers(prepared, m=tokens, topk=topk, sms=sms)
    monkeypatch.setattr(moe_dispatch, "get_num_sm", lambda _device: sms)
    slots, blocks, fc1, fc2, columns = moe_dispatch._w4a16_workspace_geometry(
        routed_rows=tokens * topk,
        route_num_experts=experts,
        num_topk=topk,
        k=h,
        n=i,
        is_gated=gated,
        device=torch.device("cuda"),
    )
    block = select_route_block_size_m(tokens, topk, experts)
    packed_slots = max_packed_route_slots(tokens * topk, block, experts)
    # Scratch growth must not inflate the route-pack buffers or block count.
    assert plan.route_slots == packed_slots
    assert plan.route_blocks == (packed_slots + block - 1) // block
    assert slots >= plan.route_slots and blocks >= plan.route_blocks
    assert columns == plan.fc1_cols
    required_slots = [packed_slots]
    if tokens <= 8:
        required_slots.append(tokens * topk * block)
    for route_slots in required_slots:
        need1 = packed_gemm_scratch_elements(
            size_n=plan.fc1_cols, route_slots=route_slots, moe_block_size=block, sms=sms
        )
        need2 = packed_gemm_scratch_elements(
            size_n=h, route_slots=route_slots, moe_block_size=block, sms=sms
        )
        assert plan.fc1_c_tmp_elements >= need1 and plan.fc2_c_tmp_elements >= need2
        assert fc1 >= need1 and fc2 >= need2
