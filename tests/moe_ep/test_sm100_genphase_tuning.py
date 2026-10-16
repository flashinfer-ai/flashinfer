# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for geometry-based Blackwell GenPhase candidate generation."""

import dataclasses

import pytest

from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe import (
    Sm100GenphaseMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim.tuning import (
    default_sm100_genphase_knobs,
    is_valid_sm100_genphase_knobs,
    sm100_genphase_candidates,
)


def config(**overrides):
    values = dict(
        num_total_experts=256,
        max_tokens_per_rank=1024,
        num_topk=8,
        hidden=4096,
        intermediate=1024,
        rank=0,
        world_size=2,
    )
    return Sm100GenphaseMoeConfig(**(values | overrides))


@pytest.mark.parametrize(
    "tokens,n,cluster,hint",
    [
        (0, 128, 4, None),
        (1, 64, 2, 296),
        (4, 64, 4, 148),
        (17, 64, 4, 148),
        (65, 64, 4, 148),
        (1024, 128, 4, 148),
    ],
)
def test_estimate_spans_latency_regimes(tokens, n, cluster, hint):
    cfg = config()
    knobs = default_sm100_genphase_knobs(cfg, sm_count=148, live_tokens=tokens)
    assert knobs == dict(
        mma_tiler_mnk=(256, n, 128),
        cluster_shape_mn=(cluster, 1),
        schedule_policy=("grouped", hint),
    )
    assert is_valid_sm100_genphase_knobs(knobs, cfg)


@pytest.mark.parametrize("sms", [132, 148, 152])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize(
    "hidden,intermediate,experts,topk", [(4096, 1024, 256, 8), (7168, 3072, 384, 6)]
)
def test_candidate_domain_and_original_fallback(
    sms, world, hidden, intermediate, experts, topk
):
    cfg = config(
        world_size=world,
        hidden=hidden,
        intermediate=intermediate,
        num_total_experts=experts,
        num_topk=topk,
    )
    candidates = sm100_genphase_candidates(cfg, sm_count=sms, live_tokens=16)
    assert len(candidates) == 12
    assert len({repr(x) for x in candidates}) == 12
    assert all(is_valid_sm100_genphase_knobs(x, cfg) for x in candidates)
    assert (
        dict(
            mma_tiler_mnk=(256, 128, 128),
            cluster_shape_mn=(4, 1),
            schedule_policy=("grouped", None),
        )
        in candidates
    )
    assert candidates[0] == default_sm100_genphase_knobs(
        cfg, sm_count=sms, live_tokens=16
    )
    assert candidates == sm100_genphase_candidates(
        dataclasses.replace(cfg, rank=world - 1), sm_count=sms, live_tokens=16
    )


def test_constraints_and_resource_budget():
    cfg = config()
    rows = sm100_genphase_candidates(
        cfg,
        sm_count=148,
        constraints=dict(
            mma_tiler_mnk=(256, 64, 128), schedule_policy=("grouped", 333)
        ),
    )
    assert len(rows) == 2
    assert all(
        x["schedule_policy"] == ("grouped", 333)
        and x["mma_tiler_mnk"] == (256, 64, 128)
        for x in rows
    )
    limited = dataclasses.replace(
        cfg, max_sm_count=2, mma_tiler_mnk=(256, 64, 128), cluster_shape_mn=(2, 1)
    )
    rows = sm100_genphase_candidates(limited, sm_count=148, live_tokens=0)
    assert len(rows) == 4 and all(x["cluster_shape_mn"] == (2, 1) for x in rows)
    for constraints in [
        dict(mma_tiler_mnk=(256, 128, 128), cluster_shape_mn=(2, 1)),
        dict(schedule_policy=("grouped", 0)),
        dict(schedule_policy=("grouped", True)),
        dict(apply_topk_at_fc1=False),
    ]:
        with pytest.raises(ValueError):
            sm100_genphase_candidates(cfg, sm_count=148, constraints=constraints)


@pytest.mark.parametrize("tokens", [-1, 1025, 1.5, True])
def test_invalid_live_count(tokens):
    with pytest.raises(ValueError):
        sm100_genphase_candidates(config(), sm_count=148, live_tokens=tokens)


@pytest.mark.parametrize("sms", [0, 1, 1.5, True])
def test_invalid_hardware_count(sms):
    with pytest.raises(ValueError):
        sm100_genphase_candidates(config(), sm_count=sms)


def test_cache_exact_identity_constraints_and_invalid_record(tmp_path, monkeypatch):
    import json
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import (
        knob_cache as cache,
    )

    path = tmp_path / "knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    cfg = config()
    winner = sm100_genphase_candidates(cfg, sm_count=148, live_tokens=16)[-1]
    path.write_text(
        json.dumps(dict(version=1, entries=[dict(device="legacy", knobs={})]))
    )
    assert cache.record_knobs(winner, cfg, live_tokens=16, p50_us=200, device="B300")
    assert cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B300") == (
        winner,
        "cache",
    )
    assert cache.live_token_buckets(cfg, device="B300") == [16]
    assert (
        cache.resolve_knobs(
            dataclasses.replace(cfg, rank=1),
            sm_count=148,
            live_tokens=16,
            device="B300",
        )[1]
        == "cache"
    )
    for changes in [
        dict(max_tokens_per_rank=512),
        dict(world_size=4),
        dict(gate_up_clamp=10),
        dict(apply_topk_at_fc1=False),
    ]:
        assert (
            cache.resolve_knobs(
                dataclasses.replace(cfg, **changes),
                sm_count=148,
                live_tokens=16,
                device="B300",
            )[1]
            == "heuristic"
        )
    assert (
        cache.resolve_knobs(cfg, sm_count=148, live_tokens=17, device="B300")[1]
        == "heuristic"
    )
    assert (
        cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B200")[1]
        == "heuristic"
    )
    assert cache.resolve_knobs(
        cfg,
        sm_count=148,
        live_tokens=16,
        device="B300",
        constraints=dict(schedule_policy=("grouped", 333)),
    )[0]["schedule_policy"] == ("grouped", 333)
    data = json.loads(path.read_text())
    assert data["entries"][0]["device"] == "legacy"
    current_revision = data["entries"][-1]["backend_revision"]
    data["entries"][-1]["backend_revision"] = "sm100-genphase-c8c67ae7-auto-v1"
    path.write_text(json.dumps(data))
    assert (
        cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B300")[1]
        == "heuristic"
    )
    assert cache.live_token_buckets(cfg, device="B300") == []
    data["entries"][-1]["backend_revision"] = current_revision
    data["entries"][-1]["knobs"]["apply_topk_at_fc1"] = False
    path.write_text(json.dumps(data))
    assert (
        cache.resolve_knobs(cfg, sm_count=148, live_tokens=16, device="B300")[1]
        == "heuristic"
    )
    assert cache.live_token_buckets(cfg, device="B300") == []
    path.write_text("invalid json")
    with pytest.warns(RuntimeWarning):
        assert not cache.record_knobs(
            winner, cfg, live_tokens=16, p50_us=200, device="B300"
        )
    assert path.read_text() == "invalid json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    assert not cache.record_knobs(
        winner, cfg, live_tokens=16, p50_us=200, device="B300"
    )


def test_public_auto_retunes_once_and_capture_guard(monkeypatch):
    from unittest import mock
    import torch
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpConfigError,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = create_mega_kernel(
            Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                1024, 8, kernel_variant="genphase", knobs="auto"
            )
        )
    backend.bind_ep_bootstrap(
        BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    fleet = FleetParams(
        num_experts=256, max_tokens_per_rank=1024, token_hidden_size=4096
    )
    assert backend._workspace_pool_request(fleet) is None
    assert backend._workspace_pool_key(fleet) is None
    with pytest.raises(MoEEpConfigError, match="warmup"):
        backend.validate_capture_ready(None, None)
    winner = sm100_genphase_candidates(config(), sm_count=148)[-1]
    tune = mock.Mock(return_value=winner)
    launch = mock.Mock(return_value=mock.sentinel.output)
    monkeypatch.setattr(pkg, "autotune_sm100_genphase_mega_moe", tune)
    monkeypatch.setattr(pkg, "sm100_genphase_mega_moe", launch)
    workspace = mock.Mock()
    workspace.staged_tokens.return_value = 16
    for _ in range(2):
        assert backend.compute(workspace, (1, 2), output=None) is mock.sentinel.output
    tune.assert_called_once()
    workspace.retune.assert_called_once_with(winner)
    assert backend.get_tuned_knobs() == winner
    backend.validate_capture_ready(workspace, (1, 2))


def test_offline_cli_dispatch_and_admission(monkeypatch):
    from unittest import mock
    from flashinfer.moe_ep import tune

    module = mock.Mock()
    module.run_tuning.return_value = 0
    loader = mock.Mock(return_value=module)
    monkeypatch.setattr(tune.importlib, "import_module", loader)
    args = [
        "--arch",
        "sm100",
        "--dtype",
        "mxfp8_e4m3",
        "--kernel-variant",
        "genphase",
        "--hidden",
        "4096",
        "--intermediate",
        "1024",
        "--num-experts",
        "256",
        "--topk",
        "8",
        "--max-tokens",
        "1024",
        "--live-tokens",
        "65",
    ]
    assert tune.main(args) == 0
    assert "sm100.mxfp8_mxfp8_bf16_cutedsl.tuner" in loader.call_args.args[0]
    assert module.run_tuning.call_args.args[0].kernel_variant == "genphase"
    loader.reset_mock()
    assert tune.main(args + ["--allow-nondeterministic"]) == 2
    loader.assert_not_called()


def test_backend_selection_memo_separates_devices(monkeypatch):
    from types import SimpleNamespace
    from unittest import mock
    import torch
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100 import next_cutedsl_megamoe as pkg

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            major=10, minor=3, name="B300", multi_processor_count=148
        ),
    )
    selected = sm100_genphase_candidates(config(), sm_count=148)[0]
    resolve = mock.Mock(return_value=(selected, "heuristic"))
    monkeypatch.setattr(pkg, "resolve_genphase_knobs", resolve)
    backend = create_mega_kernel(
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(1024, 8, kernel_variant="genphase")
    )
    backend.bind_ep_bootstrap(
        BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    fleet = FleetParams(
        num_experts=256, max_tokens_per_rank=1024, token_hidden_size=4096
    )
    first = backend._genphase_config(fleet)
    assert backend._genphase_config(fleet) is first
    resolve.assert_called_once()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    backend._genphase_config(fleet)
    assert resolve.call_count == 2


def test_device_identity_separates_compiler_stacks_and_options(monkeypatch):
    from types import SimpleNamespace
    import torch
    from flashinfer.jit import cute_dsl_core
    from flashinfer.moe_ep.kernel_src.sm100.next_cutedsl_megamoe.shim import knob_cache

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="B300", major=10, minor=3, multi_processor_count=148
        ),
    )
    monkeypatch.setattr(
        cute_dsl_core, "_get_cute_dsl_version", lambda: "dsl==4.7.1;libs-cu13==4.7.1"
    )
    monkeypatch.delenv("CUTE_DSL_COMPILER_OPT", raising=False)
    original = knob_cache.device_identity()
    monkeypatch.setattr(
        cute_dsl_core, "_get_cute_dsl_version", lambda: "dsl==4.7.1;libs-cu13==4.7.2"
    )
    updated = knob_cache.device_identity()
    assert updated != original
    monkeypatch.setenv("CUTE_DSL_COMPILER_OPT", "iket")
    assert knob_cache.device_identity() not in (original, updated)
