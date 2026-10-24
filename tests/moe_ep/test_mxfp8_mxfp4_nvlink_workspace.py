# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side invariants for the NVLink symmetric pointer and expert ABI."""

import math

import pytest
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.dispatch import (
    Mxfp8DispatchPhase,
    SingleRankMxfp8DispatchPhase,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.integration import (
    _symmetric_regions,
    _validate_ep_geometry,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.workspace import (
    WorkspaceConfig,
    make_workspace_plan,
)


@pytest.mark.parametrize("world_size", [1, 2, 8, 16])
def test_global_experts_partition_into_local_workspace(world_size):
    _validate_ep_geometry(32, world_size - 1, world_size)
    config = WorkspaceConfig(
        world_size=world_size,
        num_topk=2,
        num_experts_per_rank=32 // world_size,
        max_tokens_per_rank=7,
        hidden=256,
        intermediate=128,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=64,
    )
    assert config.num_total_experts == 32
    phase = Mxfp8DispatchPhase(
        config,
        rank=world_size - 1,
        cluster_shape_mn=(1, 1),
        dispatch_warp_start=0,
        num_other_warps=8,
    )
    assert phase.rank == world_size - 1
    assert phase.config.num_experts_per_rank == 32 // world_size
    plan = make_workspace_plan(config)
    regions, total = _symmetric_regions(7, 256, 2, plan.shared.total_bytes)
    root = torch.zeros(total, dtype=torch.uint8)
    previous_end = 0
    for _name, (offset, shape, dtype) in regions.items():
        assert offset % 128 == 0
        assert offset >= previous_end
        size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        view = root[offset : offset + size].view(dtype).reshape(shape)
        assert view.data_ptr() == root.data_ptr() + offset
        # A single delta maps every pointer, including interior row pointers.
        peer_delta = 8192
        assert view.data_ptr() + peer_delta == root.data_ptr() + peer_delta + offset
        previous_end = offset + size
    assert total >= previous_end
    assert set(regions) == {
        "x",
        "x_sf",
        "topk_idx",
        "topk_weights",
        "route_terms",
        "shared_workspace_bytes",
    }


@pytest.mark.parametrize(
    "experts,rank,world", [(8, 0, 0), (32, 0, 32), (8, 2, 2), (7, 0, 2), (0, 0, 1)]
)
def test_invalid_ep_geometry_rejected_before_allocation(experts, rank, world):
    with pytest.raises(ValueError):
        _validate_ep_geometry(experts, rank, world)


def test_legacy_standalone_phase_stays_single_rank():
    config = WorkspaceConfig(
        world_size=2,
        num_topk=1,
        num_experts_per_rank=1,
        max_tokens_per_rank=4,
        hidden=128,
        intermediate=128,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=64,
    )
    with pytest.raises(NotImplementedError, match="world_size=1"):
        SingleRankMxfp8DispatchPhase(
            config,
            cluster_shape_mn=(1, 1),
            dispatch_warp_start=0,
            num_other_warps=8,
        )
