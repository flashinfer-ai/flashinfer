"""CPU-only production wiring tests for SM90 Humming MXFP4 MegaMoE."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    Sm90PullMxfp4MegaKernelBackend,
)
from flashinfer.moe_ep.core.kernel.registry import (
    create_mega_kernel,
    is_mega_kernel_config,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


@pytest.fixture(autouse=True)
def _allow_unit_test_device(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        mxfp4_tuner,
    )

    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_tuning_device",
        lambda: None,
    )
    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_fused_tuning_device",
        lambda: None,
    )


_SHIM_PACKAGE = "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel"


def _config(**overrides):
    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        **overrides,
    )


def _fused_tactic(**overrides):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
        hopper_mxfp4_ordered_candidates,
    )

    tactic = hopper_mxfp4_ordered_candidates(
        64,
        execution_mode="fused",
        hidden=128,
        intermediate=128,
    )[0]
    tactic.update(overrides)
    return tactic


def _fleet(**overrides):
    values = dict(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=128,
    )
    values.update(overrides)
    return FleetParams(**values)


def _shim_config_kwargs(**overrides):
    values = dict(
        num_total_experts=8,
        num_max_tokens=64,
        num_topk=2,
        hidden=128,
        intermediate=128,
        rank=0,
        world_size=1,
    )
    values.update(overrides)
    return values


def _bound_backend(config=None, *, rank=1, world_size=2, group=None):
    backend = Sm90PullMxfp4MegaKernelBackend(config or _config())
    backend._ep_bootstrap = object()
    backend._ep_rank = rank
    backend._ep_world_size = world_size
    backend._ep_comm_group = group if group is not None else object()
    return backend


def _fake_shim(monkeypatch, **symbols):
    module = ModuleType(_SHIM_PACKAGE)
    for name, value in symbols.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, _SHIM_PACKAGE, module)


def test_mxfp4_backend_registry_and_public_exports():
    import flashinfer.moe_ep as moe_ep

    config = _config()
    assert is_mega_kernel_config(config)
    backend = create_mega_kernel(config)
    assert isinstance(backend, Sm90PullMxfp4MegaKernelBackend)
    assert backend.kernel_name() == "sm90_fp8_mxfp4_bf16_pull_cutedsl"
    assert backend.supports_output_view
    assert (
        moe_ep.Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
        is Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
    )
    assert callable(moe_ep.preprocess_sm90_pull_mxfp4_mega_weights)
    assert "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig" in moe_ep.__all__
    assert "preprocess_sm90_pull_mxfp4_mega_weights" in moe_ep.__all__


def test_mxfp4_backend_accepts_dedicated_collective_auto_tuning():
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = Sm90PullMxfp4MegaKernelBackend(_config(knobs="auto"))
    assert backend._autotune_pending


def test_mxfp4_direct_knob_resolution_rejects_fake_auto():
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
        _resolve_mxfp4_knobs,
    )

    with pytest.raises(ValueError, match="direct MXFP4 knob resolution"):
        _resolve_mxfp4_knobs(
            "auto",
            world_size=1,
            hidden=128,
            intermediate=128,
            num_total_experts=8,
            num_topk=2,
            num_max_tokens=64,
        )


def test_mxfp4_cache_identity_is_versioned_and_entries_fail_closed(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        knob_cache,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
        _resolve_mxfp4_knobs,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
        hopper_mxfp4_default_tactic,
    )

    lookup = mock.Mock(return_value=None)
    monkeypatch.setattr(knob_cache, "lookup_knobs", lookup)
    kwargs = dict(
        world_size=2,
        hidden=4096,
        intermediate=1280,
        num_total_experts=128,
        num_topk=6,
        num_max_tokens=64,
    )
    resolved = _resolve_mxfp4_knobs(None, **kwargs)
    assert resolved == hopper_mxfp4_default_tactic(64, execution_mode="fused")
    identity = lookup.call_args.kwargs["dtype"]
    assert (
        lookup.call_args.kwargs["routing_profile"]
        == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    )
    assert identity != "mxfp4_e2m1"
    for axis in (
        "mxfp4_e2m1",
        "k32",
        "humming_v1",
        "m64_k128",
        "gateup8",
        "packedk2",
        "residual64",
        "swapab",
        "fused",
    ):
        assert axis in identity

    invalid = {
        **resolved,
        "stale_layout_field": "silently dropped before this fix",
    }
    lookup.return_value = invalid
    with pytest.raises(ValueError, match="stale_layout_field"):
        _resolve_mxfp4_knobs(None, **kwargs)

    lookup.return_value = {"swap_ab": True}
    with pytest.raises(ValueError, match="mxfp4_fused tactic fields differ"):
        _resolve_mxfp4_knobs(None, **kwargs)


def test_mxfp4_all_none_selectors_preserve_cache_tactic(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    tactic = _fused_tactic(
        load_balance_mode="atomic_counter",
        dedup_dispatch=True,
        active_dispatch_warps=4,
        fc1_store_offload=False,
        fc1_early_done_publish=True,
        fold_producer_warps=False,
    )
    resolver = mock.Mock(return_value=tactic)
    monkeypatch.setattr(hopper_mxfp4, "_resolve_mxfp4_knobs", resolver)

    resolved = hopper_mxfp4._resolve_hopper_mxfp4_mega_moe_config(
        **_shim_config_kwargs()
    )

    resolver.assert_called_once()
    assert resolved.load_balance_mode == "atomic_counter"
    assert resolved.dedup_dispatch is True
    assert resolved.active_dispatch_warps == 4
    assert resolved.fc1_store_offload is False
    assert resolved.fc1_early_done_publish is True
    assert resolved.fold_producer_warps is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("load_balance_mode", "static"),
        ("dedup_dispatch", False),
        ("active_dispatch_warps", 1),
        ("fc1_store_offload", True),
        ("fc1_early_done_publish", False),
        ("fold_producer_warps", True),
    ],
)
def test_mxfp4_each_explicit_execution_axis_uses_cache_free_manual_defaults(
    monkeypatch, field, value
):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    resolver = mock.Mock(side_effect=AssertionError("manual selector consulted cache"))
    monkeypatch.setattr(hopper_mxfp4, "_resolve_mxfp4_knobs", resolver)

    resolved = hopper_mxfp4._resolve_hopper_mxfp4_mega_moe_config(
        **_shim_config_kwargs(**{field: value})
    )

    resolver.assert_not_called()
    assert resolved.swap_ab is True
    assert resolved.pingpong is False
    assert resolved.mma_tiler_mnk == (128, 32, 128)
    assert resolved.cluster_shape_mnk == (1, 1, 1)
    expected = {
        "load_balance_mode": "static",
        "dedup_dispatch": False,
        "active_dispatch_warps": 1,
        "fc1_store_offload": True,
        "fc1_early_done_publish": False,
        "fold_producer_warps": True,
    }
    expected[field] = value
    for name, expected_value in expected.items():
        assert getattr(resolved, name) == expected_value


def test_mxfp4_manual_active_dispatch_two_fails_closed_with_fold_default():
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    with pytest.raises(
        ValueError,
        match="fold_producer_warps=True requires active_dispatch_warps=1",
    ):
        hopper_mxfp4._resolve_hopper_mxfp4_mega_moe_config(
            **_shim_config_kwargs(active_dispatch_warps=2)
        )


@pytest.mark.parametrize("knobs", ["auto", pytest.param(_fused_tactic(), id="dict")])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("load_balance_mode", "static"),
        ("dedup_dispatch", False),
        ("active_dispatch_warps", 1),
        ("fc1_store_offload", True),
        ("fc1_early_done_publish", False),
        ("fold_producer_warps", True),
    ],
)
def test_mxfp4_direct_shim_rejects_knobs_with_explicit_execution_axis(
    knobs, field, value
):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    with pytest.raises(ValueError, match="either explicit fused geometry/execution"):
        hopper_mxfp4._resolve_hopper_mxfp4_mega_moe_config(
            **_shim_config_kwargs(knobs=knobs, **{field: value})
        )


def test_mxfp4_explicit_token_back_keeps_precedence_over_cache(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    resolver = mock.Mock(return_value=_fused_tactic(token_back_mode="epi_warps"))
    monkeypatch.setattr(hopper_mxfp4, "_resolve_mxfp4_knobs", resolver)
    resolved = hopper_mxfp4._resolve_hopper_mxfp4_mega_moe_config(
        **_shim_config_kwargs(token_back_mode="standalone_warps")
    )

    resolver.assert_called_once()
    assert resolved.resolved_token_back_mode == "standalone_warps"


def test_mxfp4_frontend_apply_knobs_rejects_non_tactic_fields(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
        MegaMoEHopperMxfp4Config,
        MegaMoEHopperMxfp4Frontend,
    )

    config = MegaMoEHopperMxfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
    )
    frontend = MegaMoEHopperMxfp4Frontend(config)
    monkeypatch.setattr(hopper_mxfp4, "ensure_not_capturing", lambda _what: None)
    with pytest.raises(ValueError, match="tactic fields differ.*missing"):
        frontend.apply_knobs({})
    with pytest.raises(ValueError, match="world_size"):
        frontend.apply_knobs({"world_size": 2})
    with pytest.raises(ValueError, match="dedup_dispatch.*bool"):
        frontend.apply_knobs(_fused_tactic(dedup_dispatch=1))
    with pytest.raises(ValueError, match="active_dispatch_warps"):
        frontend.apply_knobs(_fused_tactic(active_dispatch_warps=True))
    with pytest.raises(ValueError, match="grouped_token_back"):
        frontend.apply_knobs(_fused_tactic(grouped_token_back=True))
    with pytest.raises(ValueError, match="combine_format='bf16'"):
        frontend.apply_knobs(_fused_tactic(combine_format="32e4m3xe8m0"))

    frontend.apply_knobs(_fused_tactic(dedup_dispatch=True))
    assert frontend.config.dedup_dispatch is True
    assert frontend.config.world_size == 1


def test_mxfp4_compile_key_cannot_equal_fp8_key():
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_fp8 import (
        MegaMoEHopperFp8Config,
        MegaMoEHopperFp8Frontend,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
        MegaMoEHopperMxfp4Config,
        MegaMoEHopperMxfp4Frontend,
    )

    common = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
    )
    fp8 = MegaMoEHopperFp8Frontend(MegaMoEHopperFp8Config(**common))
    mxfp4 = MegaMoEHopperMxfp4Frontend(MegaMoEHopperMxfp4Config(**common))
    fp8_key = fp8._mega_compile_key()
    mxfp4_key = mxfp4._mega_compile_key()
    assert mxfp4_key != fp8_key
    assert mxfp4_key[0] == "sm90_mxfp4_fp8_megamoe"
    for axis in ("e2m1_k32", "humming_v1", "fold64x128", "residual64"):
        assert axis in mxfp4_key


def test_mxfp4_compile_key_tracks_every_variable_layout_knob():
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
        MegaMoEHopperMxfp4Config,
        MegaMoEHopperMxfp4Frontend,
    )

    baseline = MegaMoEHopperMxfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
    )
    baseline_key = MegaMoEHopperMxfp4Frontend(baseline)._mega_compile_key()
    variants = (
        replace(baseline, dedup_dispatch=True),
        replace(
            baseline,
            active_dispatch_warps=2,
            fold_producer_warps=False,
        ),
        replace(baseline, fc1_store_offload=False),
        replace(baseline, fc1_early_done_publish=True),
        replace(baseline, fold_producer_warps=False),
    )
    assert all(
        MegaMoEHopperMxfp4Frontend(variant)._mega_compile_key() != baseline_key
        for variant in variants
    )


def test_mxfp4_validate_init_requires_humming_k128_geometry():
    backend = Sm90PullMxfp4MegaKernelBackend(
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=64,
            top_k=2,
        )
    )
    bootstrap = BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    with pytest.raises(ValueError, match="multiple of 128"):
        backend.validate_init(
            bootstrap,
            _fleet(token_hidden_size=128),
        )


def test_mxfp4_workspace_allocator_calls_only_independent_shim(monkeypatch):
    sentinel = object()
    allocator = mock.Mock(return_value=sentinel)
    _fake_shim(
        monkeypatch,
        get_symm_buffer_for_hopper_mxfp4_mega_moe=allocator,
    )
    backend = _bound_backend()
    result = backend._allocate_workspace(_fleet())

    assert result is sentinel
    allocator.assert_called_once()
    args, kwargs = allocator.call_args
    assert args == ()
    assert kwargs["num_total_experts"] == 8
    assert kwargs["num_max_tokens"] == 64
    assert kwargs["num_topk"] == 2
    assert kwargs["hidden"] == 128
    assert kwargs["intermediate"] == 128
    assert kwargs["rank"] == 1
    assert kwargs["world_size"] == 2
    assert kwargs["kind"] == "fp8_e4m3"
    assert kwargs["fp8_scale_mode"] == "mxfp4_hybrid"
    assert kwargs["fp8_accum_mode"] == "1xacc"
    assert kwargs["knobs"] is None
    assert kwargs["swap_ab"] is None
    assert kwargs["in_kernel_fc2_reduce"] is False
    assert kwargs["load_balance_mode"] is None
    assert kwargs["dedup_dispatch"] is None
    assert kwargs["grouped_token_back"] is False
    assert kwargs["combine_format"] == "bf16"
    assert kwargs["active_dispatch_warps"] is None
    assert kwargs["fc1_store_offload"] is None
    assert kwargs["fc1_early_done_publish"] is None
    assert kwargs["fold_producer_warps"] is None
    assert kwargs["routing_profile"] == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION


def test_mxfp4_compute_calls_only_independent_launch(monkeypatch):
    launch = mock.Mock(return_value=object())
    _fake_shim(monkeypatch, hopper_mxfp4_mega_moe=launch)
    backend = _bound_backend()
    workspace = object()
    l1 = (object(), object(), object(), object())
    l2 = (object(), object(), object(), object())
    output = torch.empty((3, 128), dtype=torch.bfloat16)

    result = backend.compute(workspace, (l1, l2), output=output)

    assert result is output
    launch.assert_called_once()
    args, kwargs = launch.call_args
    assert args == (output, l1, l2, workspace)
    assert kwargs["num_tokens"] == 3
    assert kwargs["fast_math"] is True


def test_mxfp4_compute_runs_collective_auto_once_before_launch(monkeypatch):
    autotune = mock.Mock(return_value={"swap_ab": True})
    ep_group = object()
    launch = mock.Mock(return_value=object())
    _fake_shim(
        monkeypatch,
        autotune_hopper_mxfp4_mega_moe=autotune,
        hopper_mxfp4_mega_moe=launch,
    )
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = _bound_backend(_config(knobs="auto"), group=ep_group)
    workspace = object()
    l1 = (object(), object(), object(), object())
    l2 = (object(), object(), object(), object())
    output = torch.empty((3, 128), dtype=torch.bfloat16)

    assert backend.compute(workspace, (l1, l2), output=output) is output
    autotune.assert_called_once()
    assert autotune.call_args.args == (output, l1, l2, workspace)
    assert autotune.call_args.kwargs["process_group"] is ep_group
    assert autotune.call_args.kwargs["num_tokens"] == 3
    launch.assert_called_once()
    assert not backend._autotune_pending

    assert backend.compute(workspace, (l1, l2), output=output) is output
    autotune.assert_called_once()
    assert launch.call_count == 2


def test_mxfp4_auto_rejects_zero_copy_and_disables_workspace_pooling():
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = _bound_backend(_config(knobs="auto"))
    with pytest.raises(ValueError, match="caller output buffer"):
        backend.compute(
            object(),
            ((object(),), (object(),)),
            output=None,
        )
    assert backend._autotune_pending
    assert backend._workspace_pool_key(_fleet()) is None
    assert backend._workspace_pool_request(_fleet()) is None

    split = _bound_backend(_config(execution_mode="split"))
    assert split._workspace_pool_request(_fleet()) is None


def test_mxfp4_workspace_key_is_explicitly_format_isolated():
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.backend import (
        Sm90PullFp8MegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.config import (
        Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig,
    )

    group = object()
    backend = _bound_backend(group=group)
    fp8_backend = Sm90PullFp8MegaKernelBackend(
        Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
        )
    )
    fp8_backend._ep_bootstrap = object()
    fp8_backend._ep_rank = 1
    fp8_backend._ep_world_size = 2
    fp8_backend._ep_comm_group = group

    device_name = (
        "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel."
        "shim.knob_cache._current_device_name"
    )
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=3),
        mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        mock.patch(
            "torch.cuda.get_device_properties",
            return_value=mock.Mock(multi_processor_count=132),
        ),
        mock.patch(device_name, return_value="test-hopper"),
    ):
        key = backend._workspace_pool_key(_fleet())
        fp8_key = fp8_backend._workspace_pool_key(_fleet())
        changed = _bound_backend(
            replace(_config(), token_back_mode="standalone_warps"),
            group=group,
        )._workspace_pool_key(_fleet())
        exact = _bound_backend(
            replace(
                _config(swap_ab=True),
                routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
            ),
            group=group,
        )._workspace_pool_key(_fleet())

    assert key != fp8_key
    assert changed != key
    assert exact != key
    assert key[-1].routing_profile == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert exact[-1].routing_profile == SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    assert "fused" in key
    assert "mxfp4_e2m1" in key
    assert "fp8_e4m3_per_token_full_hidden" in key
    assert key[-1].fp8_scale_mode == "mxfp4_hybrid"
    assert "humming_sm90_m64_k128_gateup8_residual_x64_v1" in key


def test_mxfp4_workspace_key_contains_resolved_cache_tactic(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    first = {
        "swap_ab": True,
        "pingpong": False,
        "mma_tiler_mnk": (128, 32, 128),
        "cluster_shape_mnk": (1, 1, 1),
        "fp8_accum_mode": "1xacc",
        "group_hint": 64,
        "load_balance_mode": "static",
        "token_back_mode": "epi_warps",
        "in_kernel_fc2_reduce": False,
        "num_sched_stages": 2,
        "dedup_dispatch": False,
        "grouped_token_back": False,
        "combine_format": "bf16",
        "active_dispatch_warps": 4,
        "fc1_store_offload": False,
        "fc1_early_done_publish": False,
        "fold_producer_warps": False,
    }
    second = {**first, "mma_tiler_mnk": (256, 32, 128)}
    resolver = mock.Mock(side_effect=[first, second])
    monkeypatch.setattr(hopper_mxfp4, "_resolve_mxfp4_knobs", resolver)
    backend = _bound_backend(rank=0, world_size=1)

    with mock.patch("torch.cuda.current_device", return_value=3):
        first_key = backend._workspace_pool_key(_fleet())
        second_key = backend._workspace_pool_key(_fleet())

    assert first_key != second_key
    assert first_key[-1].mma_tiler_mnk == (128, 32, 128)
    assert second_key[-1].mma_tiler_mnk == (256, 32, 128)


def test_mxfp4_workspace_key_uses_resolved_not_selector_identity(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    tactic = {
        "swap_ab": True,
        "pingpong": False,
        "mma_tiler_mnk": (128, 32, 128),
        "cluster_shape_mnk": (1, 1, 1),
        "fp8_accum_mode": "1xacc",
        "group_hint": 64,
        "load_balance_mode": "static",
        "token_back_mode": "epi_warps",
        "in_kernel_fc2_reduce": False,
        "num_sched_stages": 2,
        "dedup_dispatch": False,
        "grouped_token_back": False,
        "combine_format": "bf16",
        "active_dispatch_warps": 4,
        "fc1_store_offload": False,
        "fc1_early_done_publish": False,
        "fold_producer_warps": False,
    }
    monkeypatch.setattr(
        hopper_mxfp4, "_resolve_mxfp4_knobs", lambda _knobs, **_kwargs: tactic
    )
    group = object()
    cached = _bound_backend(_config(), rank=0, world_size=1, group=group)
    explicit = _bound_backend(_config(knobs=tactic), rank=0, world_size=1, group=group)

    with mock.patch("torch.cuda.current_device", return_value=3):
        assert cached._workspace_pool_key(_fleet()) == explicit._workspace_pool_key(
            _fleet()
        )


def test_mxfp4_prepare_workspace_binds_key_and_allocator_to_one_resolution():
    import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as pkg
    from flashinfer.moe_ep.core.kernel import workspace_pool

    fleet = _fleet()
    backend = Sm90PullMxfp4MegaKernelBackend(_config())
    first = pkg.MegaMoEHopperMxfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
        active_dispatch_warps=1,
    )
    second = replace(
        first,
        active_dispatch_warps=2,
        fold_producer_warps=False,
    )
    workspace = object()
    captured = {}

    def acquire(key, factory):
        captured["key"] = key
        return factory()

    with (
        mock.patch.object(
            pkg,
            "_resolve_hopper_mxfp4_mega_moe_config",
            side_effect=(first, second),
        ) as resolver,
        mock.patch.object(
            pkg,
            "_get_symm_buffer_for_hopper_mxfp4_mega_moe_from_resolved_config",
            return_value=workspace,
        ) as allocator,
        mock.patch.object(workspace_pool, "acquire_workspace", side_effect=acquire),
        mock.patch("torch.cuda.current_device", return_value=0),
    ):
        result = backend.prepare_workspace(
            BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            fleet,
        )

    assert result is workspace
    resolver.assert_called_once()
    assert captured["key"][-1] is first
    assert allocator.call_args.args[0] is first


def test_mxfp4_runtime_requirements_reuse_sm90_symmetric_heap(monkeypatch):
    from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST

    monkeypatch.delenv("MEGA_NO_DIST", raising=False)
    backend = Sm90PullMxfp4MegaKernelBackend(_config())
    requirements = backend.runtime_requirements(
        BootstrapConfig(rank=0, world_size=1, auto_bootstrap=False)
    )
    assert requirements == frozenset({TORCH_DIST, NVSHMEM})


@pytest.mark.parametrize("execution_mode", ["fused", "split"])
def test_mxfp4_auto_uses_ep_singleton_group_inside_larger_global_job(
    monkeypatch, execution_mode
):
    ep_singleton = object()
    autotune = mock.Mock(return_value={"winner": True})
    launch = mock.Mock(return_value=object())
    auto_name = (
        "autotune_hopper_mxfp4_split_mega_moe"
        if execution_mode == "split"
        else "autotune_hopper_mxfp4_mega_moe"
    )
    launch_name = (
        "hopper_mxfp4_split_mega_moe"
        if execution_mode == "split"
        else "hopper_mxfp4_mega_moe"
    )
    _fake_shim(
        monkeypatch,
        **{auto_name: autotune, launch_name: launch},
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 4)
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = _bound_backend(
            _config(knobs="auto", execution_mode=execution_mode),
            rank=0,
            world_size=1,
            group=ep_singleton,
        )

    output = torch.empty((3, 128), dtype=torch.bfloat16)
    transformed = ((object(),), (object(),))
    assert backend.compute(object(), transformed, output=output) is output
    autotune.assert_called_once()
    assert autotune.call_args.kwargs["process_group"] is ep_singleton
