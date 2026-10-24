# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused host contracts for native persistent-kernel assembly."""

from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.persistent_kernel import (
    ASSEMBLY_READY,
    DEVICE_ASSEMBLY_PROTOCOL,
    PersistentMxfp8Mxfp4KernelAdapter,
    _missing_protocols,
)


def _workspace(**overrides):
    values = dict(
        world_size=1,
        rank=0,
        hidden=128,
        intermediate=256,
        num_topk=2,
        num_experts=4,
        max_tokens_per_rank=64,
        tactic={"mma_tiler_mnk": (128, 128, 128)},
    )
    values.update(overrides)
    values["num_local_experts"] = values["num_experts"] // values["world_size"]
    return SimpleNamespace(**values)


def _config(**overrides):
    values = dict(hidden=128, intermediate=256, topk=2, num_experts=4, max_tokens=64)
    values.update(overrides)
    return SimpleNamespace(**values)


def _spec(**overrides):
    values = dict(
        workspace_regions={
            "local": object(),
            "shared": object(),
            "route_terms": object(),
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_device_assembly_protocol_freezes_required_products() -> None:
    assert ASSEMBLY_READY is True
    assert DEVICE_ASSEMBLY_PROTOCOL.required_launch_regions == (
        "local",
        "shared",
        "route_terms",
    )
    assert "token_src_metadata" in DEVICE_ASSEMBLY_PROTOCOL.required_local_views
    assert "expert_recv_count_sum" in DEVICE_ASSEMBLY_PROTOCOL.dispatch_products
    assert DEVICE_ASSEMBLY_PROTOCOL.weight_scale_products == (
        "fc1_weight_sfa_logical",
        "fc2_weight_sfa_logical",
    )


def test_adapter_uses_local_expert_partition() -> None:
    adapter = PersistentMxfp8Mxfp4KernelAdapter(
        workspace=_workspace(world_size=2),
        tactic={"mma_tiler_mnk": (128, 128, 128)},
        gate_up_clamp=None,
        fast_math=True,
    )
    assert adapter.workspace.num_local_experts == 2


def test_device_protocols_are_present_and_orchestration_is_ready() -> None:
    adapter = PersistentMxfp8Mxfp4KernelAdapter(
        workspace=_workspace(),
        tactic={"mma_tiler_mnk": (128, 128, 128)},
        gate_up_clamp=None,
        fast_math=True,
    )
    assert _missing_protocols() == ()
    assert adapter.assembly_protocol() is DEVICE_ASSEMBLY_PROTOCOL


def test_compile_validates_launch_regions_before_protocol_probe() -> None:
    adapter = PersistentMxfp8Mxfp4KernelAdapter(
        workspace=_workspace(),
        tactic={"mma_tiler_mnk": (128, 128, 128)},
        gate_up_clamp=None,
        fast_math=True,
    )
    with pytest.raises(ValueError, match="route_terms"):
        adapter.compile(
            _config(),
            _spec(workspace_regions={"local": object(), "shared": object()}),
        )
