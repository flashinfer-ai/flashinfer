# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused host contracts for native persistent-kernel assembly."""

from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.persistent_kernel import (
    PersistentMxfp8Mxfp4KernelAdapter,
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


def test_adapter_uses_local_expert_partition() -> None:
    adapter = PersistentMxfp8Mxfp4KernelAdapter(
        workspace=_workspace(world_size=2),
        tactic={"mma_tiler_mnk": (128, 128, 128)},
        gate_up_clamp=None,
        fast_math=True,
    )
    assert adapter.workspace.num_local_experts == 2


def test_compile_validates_launch_regions() -> None:
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
