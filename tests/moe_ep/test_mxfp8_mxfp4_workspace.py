# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.token_comm import (
    make_token_comm_binding,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.workspace import (
    SF_ATOM_TOKEN_ROWS,
    WorkspaceConfig,
    make_workspace_plan,
    sf_uint32_per_token,
)


def _config(**kwargs) -> WorkspaceConfig:
    values: dict[str, Any] = dict(
        world_size=2,
        num_topk=4,
        num_experts_per_rank=8,
        max_tokens_per_rank=64,
        hidden=7168,
        intermediate=2048,
        token_padding_block=128,
        sf_padding_block=128,
        cluster_tile_tokens=256,
    )
    values.update(kwargs)
    return WorkspaceConfig(**values)


@pytest.mark.parametrize(
    ("hidden", "expected"), [(32, 1), (128, 1), (160, 2), (7168, 56)]
)
def test_sf_uint32_transport_padding(hidden: int, expected: int) -> None:
    assert sf_uint32_per_token(hidden) == expected


def test_pool_and_scale_sideband_sizes_match_dispatch_geometry() -> None:
    config = _config()
    assert config.pool_token_capacity == 1536
    assert config.pool_sf_capacity == 1536
    assert config.pool_sf_capacity % SF_ATOM_TOKEN_ROWS == 0

    plan = make_workspace_plan(config)
    token = plan.local.region("l1_token_buffer").spec
    scales = plan.local.region("l1_sf_buffer").spec
    assert token.nbytes == config.pool_token_capacity * config.hidden
    assert scales.nbytes == (config.pool_sf_capacity * config.sf_uint32_per_token * 4)


@pytest.mark.parametrize("mode,count", [("atomic_counter", 1), ("alternating", 2)])
def test_counter_prefixes_end_at_first_data_region(mode, count) -> None:
    plan = make_workspace_plan(_config(load_balance_mode=mode))
    counter = plan.local.region("load_balance_counter")
    assert counter.spec.shape == (count,)
    assert counter.end <= plan.local.zero_prefix_bytes
    assert plan.local.zero_prefix_bytes == plan.local.region("l1_token_buffer").offset
    assert (
        plan.shared.zero_prefix_bytes == plan.shared.region("src_token_topk_idx").offset
    )
    assert plan.local.zero_prefix_i32_count * 4 == plan.local.zero_prefix_bytes
    assert plan.shared.zero_prefix_i32_count * 4 == plan.shared.zero_prefix_bytes
    for name in ("grid_sync_counter", "nvlink_barrier_counter"):
        assert plan.local.region(name).offset >= plan.local.zero_prefix_bytes

    for layout in (plan.local, plan.shared):
        assert layout.total_bytes % 16 == 0
        for region in layout.regions:
            assert region.offset % region.spec.align == 0


def test_fc1_handoff_is_mxfp8_plus_e8m0_k32() -> None:
    config = _config()
    plan = make_workspace_plan(config)
    data = plan.local.region("fc1_output").spec
    scales = plan.local.region("fc1_output_sf").spec
    assert data.dtype == "float8_e4m3"
    assert data.shape == (config.pool_token_capacity, config.intermediate)
    assert scales.dtype == "float8_e8m0"
    assert scales.shape[1] == config.intermediate // 32


def test_transport_score_is_abi_only_not_fc1_policy() -> None:
    config = _config()
    plan = make_workspace_plan(config)
    # The current shared helper requires this compatibility region.  It must
    # never be consumed by FC1; post-FC2 reduction uses home-rank FP32 scores.
    assert plan.local.region("l1_topk_weights_buffer").spec.dtype == "float32"
    binding = make_token_comm_binding(
        config,
        cluster_shape_mn=(2, 1),
        dispatch_warp_start=8,
        num_other_warps=8,
    )
    assert binding.transports_compat_topk_weights is True
    assert binding.apply_routing_weight_in_fc1 is False
    assert binding.sf_uint32_per_token == 56
    assert binding.sf_atom_swizzled is True
    assert binding.is_swap_ab is True


def test_binding_instantiates_existing_helper_contract() -> None:
    captured = {}

    def fake_helper(**kwargs):
        captured.update(kwargs)
        return "helper"

    binding = make_token_comm_binding(
        _config(),
        cluster_shape_mn=(2, 1),
        dispatch_warp_start=8,
        num_other_warps=8,
    )
    assert (
        binding.instantiate(
            helper_type=fake_helper,
            fc1_token_dtype="Float8E4M3FN",
            combine_format="bf16",
        )
        == "helper"
    )
    assert captured["sf_uint32_per_token"] == 56
    assert captured["sf_padding_block"] == 128
    assert captured["token_back_reduce_topk"] is False


def test_rejects_non_atom_aligned_sf_padding() -> None:
    with pytest.raises(ValueError, match="multiple of 128"):
        _config(sf_padding_block=224)
