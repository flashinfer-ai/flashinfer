# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode tiles must have a valid resource model for small expert banks."""

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_w4a16_kernel as kernel
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
    select_route_block_size_m,
)
from tests.moe.test_b12x_fused_moe import (
    cute_dsl_available,
    sm120_required,
    cuda_13_required,
)


@cute_dsl_available
@pytest.mark.parametrize("tokens", [1, 4, 7, 8])
@pytest.mark.parametrize("experts,top_k", [(1, 1), (2, 1), (2, 2), (16, 2), (16, 10)])
@pytest.mark.parametrize(
    "hidden,intermediate", [(256, 128), (512, 128), (1024, 128), (2048, 512)]
)
@pytest.mark.parametrize("sms", [48, 120])
def test_decode_tiles_have_resource_models(
    tokens, experts, top_k, hidden, intermediate, sms, monkeypatch
):
    # Execute the real planner and kernel constructors, stopping only after
    # both GEMMs have validated their register/smem requirements. No GPU runs.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    constructed = []
    constructor = kernel.W4A16FusedMoeKernel

    class Planned(Exception):
        pass

    def capture(**kwargs):
        constructed.append(constructor(**kwargs))
        raise Planned

    monkeypatch.setattr(kernel, "W4A16FusedMoeKernel", capture)
    block = select_route_block_size_m(tokens, top_k, experts)
    with pytest.raises(Planned):
        kernel.compile_w4a16_fused_moe(
            size_m=tokens,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=experts,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block,
            max_m_blocks=tokens * top_k,
            element_dtype="bf16",
            sms=sms,
            max_shared_mem=101376,
            direct_topk_routes=True,
            tc_decode_fused_sum=True,
        )
    assert len(constructed) == 1
    assert constructed[0].tc_decode_fused_sum
    assert constructed[0].moe_block_size == block


@cute_dsl_available
@sm120_required
@cuda_13_required
@pytest.mark.parametrize("tokens", [4, 8])
@pytest.mark.parametrize("experts", [1, 2])
def test_small_expert_decode_matches_packed_and_live_graph(
    tokens, experts, monkeypatch
):
    from tests.moe.test_b12x_w4a16_bf16_tc_decode import (
        test_bf16_tc8_matches_packed_routes_and_live_graph,
    )

    test_bf16_tc8_matches_packed_routes_and_live_graph(
        tokens=tokens,
        activation="silu",
        hidden=256,
        intermediate=128,
        top_k=experts,
        workspace_kind="packed",
        monkeypatch=monkeypatch,
        experts=experts,
    )
