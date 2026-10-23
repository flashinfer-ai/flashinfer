# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused host tests for opaque device-workspace partitioning."""

from typing import Any

import pytest
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.device_workspace import (
    SCHEDULER_COUNTER_REGIONS,
    DeviceWorkspacePartition,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.workspace import (
    WorkspaceConfig,
    make_workspace_plan,
)


def _config(**overrides) -> WorkspaceConfig:
    values: dict[str, Any] = dict(
        world_size=1,
        num_topk=2,
        num_experts_per_rank=4,
        max_tokens_per_rank=32,
        hidden=160,
        intermediate=256,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=128,
    )
    values.update(overrides)
    return WorkspaceConfig(**values)


def _aligned_bytes(size: int, alignment: int = 128) -> torch.Tensor:
    allocation = torch.empty(size + alignment - 1, dtype=torch.uint8)
    offset = (-allocation.data_ptr()) % alignment
    return allocation[offset : offset + size]


def _partition(**config_overrides) -> DeviceWorkspacePartition:
    plan = make_workspace_plan(_config(**config_overrides))
    local = _aligned_bytes(plan.local.total_bytes)
    shared = _aligned_bytes(plan.shared.total_bytes)
    route_terms = torch.empty(
        (
            plan.config.max_tokens_per_rank,
            plan.config.num_topk,
            plan.config.hidden,
        ),
        dtype=torch.bfloat16,
    )
    return DeviceWorkspacePartition(plan, local, route_terms, shared_workspace=shared)


def test_partitions_every_region_without_allocating_storage() -> None:
    partition = _partition(load_balance_mode="atomic_counter")

    assert set(partition.local) == {
        region.spec.name for region in partition.plan.local.regions
    }
    assert set(partition.shared) == {
        region.spec.name for region in partition.plan.shared.regions
    }
    for layout, workspace, views in (
        (partition.plan.local, partition.local_workspace, partition.local),
        (partition.plan.shared, partition.shared_workspace, partition.shared),
    ):
        assert workspace is not None
        for region in layout.regions:
            view = views[region.spec.name]
            assert tuple(view.shape) == region.spec.shape
            assert view.data_ptr() == workspace.data_ptr() + region.offset
            assert view.numel() * view.element_size() == region.spec.nbytes


def test_exposes_mxfp8_sidebands_metadata_counts_and_readiness() -> None:
    partition = _partition()
    config = partition.plan.config

    assert partition.local["l1_token_buffer"].dtype is torch.uint8
    assert partition.activation_scales.dtype is torch.float8_e8m0fnu
    assert partition.activation_scales.shape == (
        config.pool_sf_capacity,
        config.sf_uint32_per_token * 4,
    )
    assert partition.local["fc1_output"].dtype is torch.float8_e4m3fn
    assert partition.local["fc1_output_sf"].dtype is torch.float8_e8m0fnu
    assert partition.token_metadata_i64.shape == (config.pool_token_capacity,)
    assert (
        partition.expert_counts.data_ptr()
        == partition.local["expert_send_count"].data_ptr()
    )
    assert (
        partition.expert_recv_count_sum.data_ptr()
        == partition.shared["expert_recv_count_sum"].data_ptr()
    )
    assert (
        partition.fc1_ready_counters.data_ptr()
        == partition.local["l1_arrival_count"].data_ptr()
    )
    assert (
        partition.fc2_ready_counters.data_ptr()
        == partition.local["fc1_done_counter"].data_ptr()
    )
    assert SCHEDULER_COUNTER_REGIONS == {
        "fc1_ready_counters": "l1_arrival_count",
        "fc2_ready_counters": "fc1_done_counter",
    }


def test_zeroes_only_declared_counter_prefixes() -> None:
    partition = _partition()
    partition.local_workspace.fill_(0xA5)
    assert partition.shared_workspace is not None
    partition.shared_workspace.fill_(0xA5)

    with pytest.raises(ValueError, match="counter prefix"):
        partition.validate_zero_prefixes()
    partition.zero_prefixes_()
    partition.validate_zero_prefixes()

    assert torch.all(
        partition.local_workspace[partition.plan.local.zero_prefix_bytes :] == 0xA5
    )
    assert torch.all(
        partition.shared_workspace[partition.plan.shared.zero_prefix_bytes :] == 0xA5
    )


def test_expert_pool_offsets_match_scheduler_padding_contract() -> None:
    partition = _partition()
    offsets = partition.expert_pool_offsets((1, 0, 65, 2))
    assert offsets.data_rows == (0, 64, 64, 192, 256)
    assert offsets.scale_rows == (0, 128, 128, 256, 384)
    assert offsets.token_blocks == (0, 1, 1, 2, 3)

    with pytest.raises(ValueError, match="host-only"):
        # Meta is sufficient to exercise the no-device-read contract on CPU CI.
        partition.expert_pool_offsets(torch.empty(4, device="meta", dtype=torch.int64))


def test_rejects_wrong_extent_shape_and_unaligned_base() -> None:
    plan = make_workspace_plan(_config())
    route_terms = torch.empty((32, 2, 160), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires exactly"):
        DeviceWorkspacePartition(
            plan, _aligned_bytes(plan.local.total_bytes + 1), route_terms
        )

    backing = _aligned_bytes(plan.local.total_bytes + 1)
    with pytest.raises(ValueError, match="aligned"):
        DeviceWorkspacePartition(plan, backing[1:], route_terms)

    with pytest.raises(ValueError, match="route_terms must have shape"):
        DeviceWorkspacePartition(
            plan,
            _aligned_bytes(plan.local.total_bytes),
            torch.empty((31, 2, 160), dtype=torch.bfloat16),
        )


def test_cute_conversion_stays_lazy_on_cpu() -> None:
    partition = _partition()
    with pytest.raises(ValueError, match="CUDA"):
        partition.to_cute()
