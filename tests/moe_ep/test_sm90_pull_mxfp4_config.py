"""CPU-only tests for the SM90 FP8 x MXFP4 host config checkpoint."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest import mock

import pytest

from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
    MegaMoEHopperMxfp4Config,
    MegaMoEHopperMxfp4Frontend,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
    hopper_mxfp4_default_tactic,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


def _config(**overrides):
    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128, top_k=4, **overrides
    )


def _fused_knobs(**overrides):
    tactic = hopper_mxfp4_default_tactic(64, execution_mode="fused")
    tactic.update(overrides)
    return tactic


def _resolve_public_fused_config(config):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
        _resolve_hopper_mxfp4_mega_moe_config,
    )

    return _resolve_hopper_mxfp4_mega_moe_config(
        num_total_experts=8,
        num_max_tokens=64,
        num_topk=config.top_k,
        hidden=128,
        intermediate=config.intermediate_size,
        rank=0,
        world_size=1,
        knobs=config.knobs,
        swap_ab=config.swap_ab,
        pingpong=config.pingpong,
        mma_tiler_mnk=config.mma_tiler_mnk,
        cluster_shape_mnk=config.cluster_shape_mnk,
        load_balance_mode=config.load_balance_mode,
        token_back_mode=config.token_back_mode,
        dedup_dispatch=config.dedup_dispatch,
        grouped_token_back=config.grouped_token_back,
        combine_format=config.combine_format,
        active_dispatch_warps=config.active_dispatch_warps,
        fc1_store_offload=config.fc1_store_offload,
        fc1_early_done_publish=config.fc1_early_done_publish,
        fold_producer_warps=config.fold_producer_warps,
        routing_profile=config.routing_profile,
    )


def _frontend_with_complete_tactic(**overrides):
    layout = dict(
        active_dispatch_warps=1,
        fc1_store_offload=True,
        fc1_early_done_publish=False,
        fold_producer_warps=True,
    )
    layout.update(overrides)
    tactic = _fused_knobs(**layout)
    config = MegaMoEHopperMxfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=6,
        num_total_experts=384,
        hidden=7168,
        intermediate=3072,
        **tactic,
    )
    return MegaMoEHopperMxfp4Frontend(config), tactic


def test_sm90_mxfp4_frontend_reports_requested_and_compiled_effective_tactic():
    frontend, requested = _frontend_with_complete_tactic()
    assert frontend.requested_tactic() == requested

    kernel = SimpleNamespace(
        group_hint=requested["group_hint"],
        num_sched_stages=requested["num_sched_stages"],
        dedup_dispatch=requested["dedup_dispatch"],
        grouped_token_back=requested["grouped_token_back"],
        combine_format="bf16",
        token_comm=SimpleNamespace(active_dispatch_warps=2),
        fc1_store_offload=False,
        fc1_early_done_publish=True,
        fold_producer_warps=False,
    )
    frontend._mega_key = frontend._mega_compile_key()
    frontend._mega = SimpleNamespace(compiled=object(), kernel=kernel)

    effective = frontend.effective_tactic()
    assert set(effective) == set(requested)
    assert effective["active_dispatch_warps"] == 2
    assert effective["fc1_store_offload"] is False
    assert effective["fc1_early_done_publish"] is True
    assert effective["fold_producer_warps"] is False


def test_sm90_mxfp4_requested_tactic_preserves_auto_schedule_values():
    config = MegaMoEHopperMxfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
    )
    requested = MegaMoEHopperMxfp4Frontend(config).requested_tactic()
    assert requested["group_hint"] is None
    assert requested["num_sched_stages"] is None


def test_sm90_mxfp4_effective_tactic_fails_closed_without_current_compile():
    frontend, _ = _frontend_with_complete_tactic()
    with pytest.raises(RuntimeError, match="actual compiled kernel"):
        frontend.effective_tactic()

    frontend._mega_key = frontend._mega_compile_key()
    frontend._mega = SimpleNamespace(compiled=object(), kernel=SimpleNamespace())
    with pytest.raises(RuntimeError, match="lacks effective tactic field"):
        frontend.effective_tactic()

    frontend._mega_key = ("stale",)
    with pytest.raises(RuntimeError, match="actual compiled kernel"):
        frontend.effective_tactic()


def test_sm90_mxfp4_config_defaults_are_format_specific():
    cfg = _config()
    assert cfg.kernel_name == "sm90_fp8_mxfp4_bf16_pull_cutedsl"
    assert cfg.kind == "fp8_e4m3"
    assert cfg.fp8_scale_mode == "mxfp4_hybrid"
    assert cfg.fp8_accum_mode == "1xacc"
    assert cfg.humming_max_range == 11
    assert cfg.preprocess_expert_chunk_size == 4
    assert cfg.gate_up_clamp == 10.0
    assert cfg.swap_ab is None  # MXFP4-specific heuristic, never native fallback.
    assert cfg.in_kernel_fc2_reduce is False
    assert cfg.load_balance_mode is None
    assert cfg.dedup_dispatch is None
    assert cfg.grouped_token_back is False
    assert cfg.combine_format == "bf16"
    assert cfg.active_dispatch_warps is None
    assert cfg.fc1_store_offload is None
    assert cfg.fc1_early_done_publish is None
    assert cfg.fold_producer_warps is None
    assert cfg.routing_profile == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert dataclasses.fields(cfg)[-1].name == "routing_profile"
    assert dataclasses.fields(cfg)[-1].kw_only


def test_sm90_mxfp4_public_config_strictly_validates_routing_profile():
    exact = _config(
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    assert exact.routing_profile == SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    for invalid in (None, True, "published_exact_balanced", "block_permutation"):
        with pytest.raises(ValueError, match="routing_profile"):
            _config(routing_profile=invalid)


def test_sm90_mxfp4_fused_layout_knobs_preserve_requested_values():
    public = _config(
        dedup_dispatch=True,
        active_dispatch_warps=2,
        fc1_store_offload=False,
        fc1_early_done_publish=True,
        fold_producer_warps=False,
    )
    assert public.dedup_dispatch is True
    assert public.active_dispatch_warps == 2
    assert public.fc1_store_offload is False
    assert public.fc1_early_done_publish is True
    assert public.fold_producer_warps is False

    shim = MegaMoEHopperMxfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
        dedup_dispatch=True,
        active_dispatch_warps=2,
        fc1_store_offload=False,
        fc1_early_done_publish=True,
        fold_producer_warps=False,
    )
    assert shim.dedup_dispatch is True
    assert shim.active_dispatch_warps == 2
    assert shim.fc1_store_offload is False
    assert shim.fc1_early_done_publish is True
    assert shim.fold_producer_warps is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dedup_dispatch", 1, "dedup_dispatch must be a bool"),
        ("grouped_token_back", True, "does not support grouped_token_back"),
        ("combine_format", "32e4m3xe8m0", "combine_format='bf16'"),
        ("active_dispatch_warps", True, "active_dispatch_warps"),
        ("active_dispatch_warps", 3, "active_dispatch_warps"),
        ("fc1_store_offload", 1, "fc1_store_offload must be a bool"),
        ("fc1_early_done_publish", 0, "fc1_early_done_publish must be a bool"),
        ("fold_producer_warps", "yes", "fold_producer_warps must be a bool"),
    ],
)
def test_sm90_mxfp4_public_config_strictly_validates_fused_layout_knobs(
    field, value, match
):
    with pytest.raises(ValueError, match=match):
        _config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dedup_dispatch", 1, "dedup_dispatch must be a bool"),
        ("grouped_token_back", True, "does not support grouped_token_back"),
        ("combine_format", "32e5m2xe8m0", "combine_format='bf16'"),
        ("active_dispatch_warps", True, "active_dispatch_warps"),
        ("active_dispatch_warps", 3, "active_dispatch_warps"),
        ("fc1_store_offload", 1, "fc1_store_offload must be a bool"),
        ("fc1_early_done_publish", 0, "fc1_early_done_publish must be a bool"),
        ("fold_producer_warps", None, "fold_producer_warps must be a bool"),
    ],
)
def test_sm90_mxfp4_shim_config_strictly_validates_fused_layout_knobs(
    field, value, match
):
    common = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
    )
    with pytest.raises(ValueError, match=match):
        MegaMoEHopperMxfp4Config(**common, **{field: value})


@pytest.mark.parametrize("active_dispatch_warps", (2, 4))
def test_sm90_mxfp4_rejects_ineffective_producer_fold(
    active_dispatch_warps: int,
) -> None:
    match = "fold_producer_warps=True requires active_dispatch_warps=1"
    config = _config(
        knobs=_fused_knobs(
            active_dispatch_warps=active_dispatch_warps,
            fold_producer_warps=True,
        )
    )
    with pytest.raises(ValueError, match=match):
        _resolve_public_fused_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("swap_ab", 1),
        ("pingpong", 0),
        ("in_kernel_fc2_reduce", 0),
    ],
)
def test_sm90_mxfp4_configs_reject_non_bool_inherited_flags(field, value):
    with pytest.raises(ValueError, match=field + " must be a bool"):
        _config(**{field: value})

    common = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
    )
    with pytest.raises(ValueError, match=field + " must be a bool"):
        MegaMoEHopperMxfp4Config(**common, **{field: value})


def test_sm90_mxfp4_shim_config_has_kw_only_strict_routing_identity():
    common = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
    )
    block = MegaMoEHopperMxfp4Config(**common)
    exact = MegaMoEHopperMxfp4Config(
        **common,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    routing_field = dataclasses.fields(MegaMoEHopperMxfp4Config)[-1]
    assert routing_field.name == "routing_profile"
    assert routing_field.kw_only
    assert block != exact
    with pytest.raises(ValueError, match="routing_profile"):
        MegaMoEHopperMxfp4Config(**common, routing_profile="exact")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kind", "fp8_e5m2", "fp8_e4m3"),
        ("fp8_scale_mode", "per_tensor", "mxfp4_hybrid"),
        ("fp8_accum_mode", "2xacc", "1xacc"),
        ("humming_max_range", 10, "max_range=11"),
        ("preprocess_expert_chunk_size", 0, "must be positive"),
        ("swap_ab", False, "swap-AB"),
        ("in_kernel_fc2_reduce", True, "standalone top-k"),
    ],
)
def test_sm90_mxfp4_config_rejects_format_fallback(field, value, match):
    base = _config()
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(base, **{field: value})


def test_sm90_mxfp4_knobs_conflict_with_explicit_geometry():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(knobs="auto", swap_ab=True)


@pytest.mark.parametrize("knobs", ["auto", pytest.param(_fused_knobs(), id="dict")])
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
def test_sm90_mxfp4_knobs_conflict_with_each_explicit_execution_axis(
    knobs, field, value
):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(knobs=knobs, **{field: value})


def test_sm90_mxfp4_public_config_does_not_import_or_validate_kernel_drop():
    raw_knobs = {"deliberately": "not a complete tactic"}
    real_import = __import__

    def reject_drop_import(name, *args, **kwargs):
        if "kernel_src.sm90" in name or "pull_style_cutedsl_megakernel" in name:
            raise AssertionError(f"public config imported kernel drop: {name}")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=reject_drop_import):
        config = _config(knobs=raw_knobs)
    assert config.knobs is raw_knobs


def test_sm90_mxfp4_explicit_knobs_cannot_select_native_ab():
    config = _config(knobs=_fused_knobs(swap_ab=False))
    with pytest.raises(ValueError, match="swap_ab=true"):
        _resolve_public_fused_config(config)


@pytest.mark.parametrize("removed", ["swap_ab", "dedup_dispatch", "combine_format"])
def test_sm90_mxfp4_explicit_knobs_reject_partial_tactics(removed):
    knobs = _fused_knobs()
    del knobs[removed]
    config = _config(knobs=knobs)
    with pytest.raises(ValueError, match=r"tactic fields differ: missing=.*" + removed):
        _resolve_public_fused_config(config)


@pytest.mark.parametrize("extra", ["world_size", "fp8_scale_mode", "typo_flag_bach"])
def test_sm90_mxfp4_explicit_knobs_reject_non_tactic_fields(extra):
    config = _config(knobs={**_fused_knobs(), extra: "invalid"})
    with pytest.raises(ValueError, match=r"tactic fields differ:.*extra=.*" + extra):
        _resolve_public_fused_config(config)


def test_sm90_mxfp4_explicit_knobs_accept_supported_fused_layout_fields():
    cfg = _config(
        knobs=_fused_knobs(
            mma_tiler_mnk=(256, 32, 128),
            dedup_dispatch=True,
            active_dispatch_warps=2,
            fc1_store_offload=False,
            fc1_early_done_publish=True,
            fold_producer_warps=False,
        )
    )
    assert cfg.knobs["dedup_dispatch"] is True
    assert cfg.knobs["active_dispatch_warps"] == 2
    resolved = _resolve_public_fused_config(cfg)
    assert resolved.dedup_dispatch is True
    assert resolved.active_dispatch_warps == 2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dedup_dispatch", 1, "dedup_dispatch must be bool"),
        ("pingpong", 0, "pingpong must be bool"),
        ("active_dispatch_warps", True, "active_dispatch_warps"),
        ("fc1_store_offload", 1, "fc1_store_offload must be bool"),
        ("fc1_early_done_publish", None, "fc1_early_done_publish must be bool"),
        ("fold_producer_warps", "yes", "fold_producer_warps must be bool"),
        ("grouped_token_back", True, "fix grouped_token_back=false"),
        ("combine_format", "32e4m3xe8m0", "combine_format='bf16'"),
    ],
)
def test_sm90_mxfp4_explicit_knobs_strictly_validate_layout_fields(field, value, match):
    config = _config(knobs=_fused_knobs(**{field: value}))
    with pytest.raises(ValueError, match=match):
        _resolve_public_fused_config(config)


def test_sm90_mxfp4_explicit_knobs_preserve_fixed_numerics():
    config = _config(knobs=_fused_knobs(fp8_accum_mode="2xacc"))
    with pytest.raises(ValueError, match="1xacc"):
        _resolve_public_fused_config(config)
    config = _config(knobs=_fused_knobs(in_kernel_fc2_reduce=True))
    with pytest.raises(ValueError, match="in-kernel reduce false"):
        _resolve_public_fused_config(config)
