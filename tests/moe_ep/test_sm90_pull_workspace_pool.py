# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Workspace sharing uses the resolved tactic for both Hopper weight formats."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig,
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.kernel import workspace_pool
import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as pkg
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    hopper_fp8,
    hopper_mxfp4,
    knob_cache,
    mxfp4_tuner,
)


@pytest.fixture
def pool_setup(monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        mxfp4_tuner, "require_hopper_mxfp4_fused_tuning_device", lambda: None
    )
    group = object()
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=64, token_hidden_size=1024)

    def make(mode, knobs=None):
        mxfp4 = mode == "mxfp4_hybrid"
        config_type = (
            Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
            if mxfp4
            else Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig
        )
        backend = create_mega_kernel(
            config_type(
                intermediate_size=512,
                top_k=4,
                fp8_scale_mode=mode,
                knobs=knobs,
            )
        )
        backend._ep_bootstrap = object()
        backend._ep_rank, backend._ep_world_size = 0, 1
        backend._ep_comm_group = group
        return backend

    return fleet, make


@pytest.fixture(params=["per_tensor", "blockwise", "mxfp4_hybrid"])
def pool_case(request, pool_setup):
    mode = request.param
    fleet, make = pool_setup
    mxfp4 = mode == "mxfp4_hybrid"
    fmt = "mxfp4" if mxfp4 else "fp8"
    return SimpleNamespace(
        mode=mode,
        fleet=fleet,
        make=lambda knobs=None: make(mode, knobs),
        shim=hopper_mxfp4 if mxfp4 else hopper_fp8,
        resolver=("_resolve" if mxfp4 else "resolve")
        + f"_hopper_{fmt}_mega_moe_config",
        allocator=f"_get_symm_buffer_for_hopper_{fmt}_mega_moe_from_resolved_config",
        tactic=(
            mxfp4_tuner._ordered_base_candidates(64, hidden=1024, intermediate=512)[0]
            if mxfp4
            else pkg.default_knobs(64, fp8_scale_mode=mode)
        ),
    )


def test_pool_key_uses_resolved_tactic(pool_case, monkeypatch):
    case = pool_case
    heuristic = case.make()._workspace_pool_key(case.fleet)
    assert heuristic == case.make(case.tactic)._workspace_pool_key(case.fleet)
    first = dict(case.tactic, active_dispatch_warps=2, fold_producer_warps=False)
    second = dict(first, active_dispatch_warps=4)
    monkeypatch.setattr(
        knob_cache, "lookup_knobs", mock.Mock(side_effect=[first, second])
    )
    backend = case.make()
    first_key = backend._workspace_pool_key(case.fleet)
    second_key = backend._workspace_pool_key(case.fleet)
    assert first_key != second_key
    assert first_key[-1].active_dispatch_warps == 2
    assert second_key[-1].active_dispatch_warps == 4


def test_pool_key_and_allocator_share_one_resolution(pool_case, monkeypatch):
    case = pool_case
    backend = case.make()
    first = backend._workspace_pool_key(case.fleet)[-1]
    second = replace(first, active_dispatch_warps=2, fold_producer_warps=False)
    resolver = mock.Mock(side_effect=[first, second])
    allocator = mock.Mock(return_value=object())
    acquired = []

    def acquire(key, factory):
        acquired.append(key)
        return factory()

    monkeypatch.setattr(pkg, case.resolver, resolver)
    monkeypatch.setattr(pkg, case.allocator, allocator)
    monkeypatch.setattr(workspace_pool, "acquire_workspace", acquire)
    result = backend.prepare_workspace(
        BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        case.fleet,
    )
    assert result is allocator.return_value
    resolver.assert_called_once()
    assert acquired[0][-1] is first
    allocator.assert_called_once_with(first)


def test_auto_workspace_is_unpooled(pool_case, monkeypatch):
    case = pool_case
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = case.make("auto")
    assert backend._workspace_pool_request(case.fleet) is None
    allocator = mock.Mock(return_value=object())
    monkeypatch.setattr(case.shim, case.allocator, allocator)
    result = backend.prepare_workspace(
        BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        case.fleet,
    )
    assert result is allocator.return_value
    allocator.assert_called_once()
    assert allocator.call_args.args[0].fp8_scale_mode == case.mode
    assert backend._autotune_pending


def test_workspace_pool_isolates_formats_and_routing(pool_setup):
    fleet, make = pool_setup
    keys = {
        make(mode)._workspace_pool_key(fleet)
        for mode in ("per_tensor", "blockwise", "mxfp4_hybrid")
    }
    assert len(keys) == 3
    backend = make("mxfp4_hybrid")
    base = backend._workspace_pool_key(fleet)
    backend._kernel_config = replace(
        backend._kernel_config,
        routing_profile="published_exact_balanced_v1",
    )
    assert backend._workspace_pool_key(fleet) != base
