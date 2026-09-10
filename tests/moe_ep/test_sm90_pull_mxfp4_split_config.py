"""Production config/cache contracts for SM90 MXFP4 Green split execution."""

from __future__ import annotations

import dataclasses

import pytest

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.backend import (
    Sm90PullMxfp4MegaKernelBackend,
    _resolve_split_tactic,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4_split import (
    MegaMoEHopperMxfp4SplitConfig,
    _make_shape_validation_config,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
    hopper_mxfp4_default_tactic,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


def _split_config(**overrides):
    fields = dict(
        intermediate_size=128,
        top_k=2,
        execution_mode="split",
        split_k1_mma_tiler_mnk=(128, 64, 128),
        split_k2_mma_tiler_mnk=(128, 64, 128),
        split_k1_cluster_shape_mnk=(1, 1, 1),
        split_k2_cluster_shape_mnk=(1, 1, 1),
        split_k1_group_hint=80,
        split_k2_group_hint=52,
        split_k1_num_sched_stages=1,
        split_k2_num_sched_stages=3,
        split_k1_sm_count=80,
        split_k2_sm_count=52,
        split_counter_epoch_banks=1,
        split_graph_variant="steady_k3_reset",
    )
    fields.update(overrides)
    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(**fields)


def _split_knobs(**overrides):
    tactic = hopper_mxfp4_default_tactic(64, execution_mode="split")
    tactic.update(overrides)
    return tactic


def _split_selector_config(knobs):
    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode="split",
        knobs=knobs,
    )


def _fleet() -> FleetParams:
    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=32,
        token_hidden_size=128,
    )


def _bound_backend(config):
    backend = Sm90PullMxfp4MegaKernelBackend(config)
    backend.bind_ep_bootstrap(BootstrapConfig(world_size=2, rank=0, device=0))
    # The cache identity uses group identity but need not initialize dist for
    # this pure host contract.
    backend._ep_comm_group = object()
    return backend


def test_default_execution_mode_remains_phase_a_fused() -> None:
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
    )
    assert config.execution_mode == "fused"


def test_split_public_config_exposes_the_fixed_validated_protocol() -> None:
    config = _split_config()

    assert config.dedup_dispatch is False
    assert config.load_balance_mode == "static"
    assert config.grouped_token_back is False
    assert config.combine_format == "bf16"
    assert config.active_dispatch_warps == 4
    assert config.fc1_store_offload is False
    assert config.fc1_early_done_publish is False
    assert config.fold_producer_warps is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dedup_dispatch", True, "non-deduplicated dispatch ABI"),
        ("grouped_token_back", True, "does not support grouped_token_back"),
        ("combine_format", "32e4m3xe8m0", "combine_format='bf16'"),
        ("active_dispatch_warps", 2, "fixes K1 active_dispatch_warps=4"),
        ("active_dispatch_warps", 1, "fixes K1 active_dispatch_warps=4"),
        ("fc1_store_offload", True, "fc1_store_offload=False"),
        ("fc1_early_done_publish", True, "fc1_early_done_publish=False"),
        ("fold_producer_warps", True, "fold_producer_warps=False"),
        ("load_balance_mode", "atomic_counter", "load_balance_mode='static'"),
    ],
)
def test_split_public_config_rejects_unvalidated_fused_protocols(
    field, value, match
) -> None:
    with pytest.raises(ValueError, match=match):
        _split_config(**{field: value})


def test_split_explicit_knobs_use_shared_exact_tactic_schema() -> None:
    knobs = _split_knobs()
    config = _split_selector_config(knobs)

    assert config.knobs == knobs
    assert isinstance(config.knobs["k1_mma_tiler_mnk"], tuple)
    assert isinstance(config.knobs["k2_cluster_shape_mnk"], tuple)
    assert _resolve_split_tactic(config, _fleet(), world_size=2) == knobs


@pytest.mark.parametrize("removed", ["k1_sm_count", "graph_variant"])
def test_split_explicit_knobs_reject_partial_tactics(removed: str) -> None:
    knobs = _split_knobs()
    del knobs[removed]

    config = _split_selector_config(knobs)
    with pytest.raises(ValueError, match=r"tactic fields differ: missing=.*" + removed):
        _resolve_split_tactic(config, _fleet(), world_size=2)


def test_split_explicit_knobs_reject_extra_tactic_fields() -> None:
    config = _split_selector_config({**_split_knobs(), "world_size": 4})
    with pytest.raises(ValueError, match=r"tactic fields differ:.*extra=.*world_size"):
        _resolve_split_tactic(config, _fleet(), world_size=2)


def test_split_explicit_knobs_validate_values_through_shared_schema() -> None:
    config = _split_selector_config(_split_knobs(enable_iket=True))
    with pytest.raises(ValueError, match="first formal split domain fixes IKET false"):
        _resolve_split_tactic(config, _fleet(), world_size=2)


def test_split_explicit_non_h200_partition_defers_live_sm_check() -> None:
    config = _split_selector_config(_split_knobs(k1_sm_count=48, k2_sm_count=30))

    assert config.knobs["k1_sm_count"] == 48
    assert config.knobs["k2_sm_count"] == 30


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        (
            {
                "split_k1_cluster_shape_mnk": (2, 1, 1),
                "split_k2_cluster_shape_mnk": (2, 1, 1),
            },
            "quarantined correctness failures",
        ),
        (
            {"split_k1_sm_count": 79, "split_k2_sm_count": 53},
            "split partition",
        ),
        ({"split_enable_iket": True}, "fixes IKET false"),
    ),
)
def test_legacy_split_fields_use_shared_exact_tactic_schema(
    overrides, match: str
) -> None:
    config = _split_config(**overrides)

    with pytest.raises(ValueError, match=match):
        _resolve_split_tactic(config, _fleet(), world_size=2)


def test_split_session_has_no_fake_dispatch_or_fold_tuning_axes() -> None:
    config = MegaMoEHopperMxfp4SplitConfig(
        rank=0,
        world_size=2,
        num_tokens_per_rank=32,
        num_topk=2,
        num_total_experts=8,
        hidden=128,
        intermediate=128,
        k1_mma_tiler_mnk=(128, 64, 128),
        k2_mma_tiler_mnk=(128, 64, 128),
        k1_cluster_shape_mnk=(1, 1, 1),
        k2_cluster_shape_mnk=(1, 1, 1),
        k1_sm_count=80,
        k2_sm_count=52,
    )
    split_fields = {item.name for item in dataclasses.fields(config)}
    assert split_fields.isdisjoint(
        {
            "dedup_dispatch",
            "grouped_token_back",
            "combine_format",
            "active_dispatch_warps",
            "k1_active_dispatch_warps",
            "k2_active_dispatch_warps",
            "fc1_store_offload",
            "fc1_early_done_publish",
            "fold_producer_warps",
            "k1_fold_producer_warps",
            "k2_fold_producer_warps",
        }
    )

    proxy = _make_shape_validation_config(config)
    assert proxy.dedup_dispatch is False
    assert proxy.grouped_token_back is False
    assert proxy.combine_format == "bf16"
    assert proxy.active_dispatch_warps == 4
    assert proxy.fc1_store_offload is False
    assert proxy.fc1_early_done_publish is False
    assert proxy.fold_producer_warps is False


@pytest.mark.parametrize("mode", ["sequential", "green", "auto", None])
def test_execution_mode_rejects_non_fused_non_split_values(mode) -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            execution_mode=mode,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("split_k1_sm_count", None),
        ("split_k2_sm_count", None),
        ("split_k1_sm_count", 0),
        ("split_k2_sm_count", -1),
        ("split_k1_sm_count", True),
        ("split_k2_sm_count", 52.0),
    ],
)
def test_split_requires_positive_integral_sm_partitions(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        _split_config(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("split_counter_epoch_banks", 0),
        ("split_counter_epoch_banks", 3),
        ("split_counter_epoch_banks", True),
        ("split_graph_variant", "host_reset"),
        ("split_graph_variant", "sequential"),
    ],
)
def test_split_rejects_unsupported_bank_or_graph_policy(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        _split_config(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("split_k1_mma_tiler_mnk", (64, 64, 128)),
        ("split_k2_mma_tiler_mnk", (128, 48, 128)),
        ("split_k1_mma_tiler_mnk", (128, 64, 64)),
        ("split_k2_cluster_shape_mnk", (1, 1, 2)),
        ("split_k1_cluster_shape_mnk", (3, 1, 1)),
        ("split_k2_cluster_shape_mnk", (1, 3, 1)),
        ("split_k1_num_sched_stages", 0),
        ("split_k2_num_sched_stages", True),
    ],
)
def test_split_tactic_validation_fails_closed(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        _split_config(**{field: value})


def test_fused_mode_preserves_unused_split_cluster_shape_semantics() -> None:
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode="fused",
        split_k1_cluster_shape_mnk=(3, 1, 1),
        split_k2_cluster_shape_mnk=(3, 1, 1),
    )

    assert config.split_k1_cluster_shape_mnk == (3, 1, 1)
    assert config.split_k2_cluster_shape_mnk == (3, 1, 1)


def test_k1_tile_k_is_validated_against_hidden_at_backend_init(monkeypatch) -> None:
    config = _split_config(split_k1_mma_tiler_mnk=(128, 64, 256))
    backend = _bound_backend(config)
    module = __import__(Sm90PullMxfp4MegaKernelBackend.__module__, fromlist=["_"])
    monkeypatch.setattr(module, "validate_mega_arch_sm90", lambda: None)
    monkeypatch.setattr(
        module, "validate_mega_fleet_params", lambda *args, **kwargs: None
    )
    bootstrap = BootstrapConfig(world_size=2, rank=0, device=0)
    with pytest.raises(ValueError, match=r"token_hidden_size.*K=256"):
        backend.validate_init(bootstrap, _fleet())
    backend.validate_init(
        bootstrap,
        FleetParams(num_experts=8, max_tokens_per_rank=32, token_hidden_size=256),
    )


def test_split_config_derives_handoff_counter_and_cluster_limits() -> None:
    config = _split_config(
        split_k1_mma_tiler_mnk=(256, 128, 128),
        split_k2_mma_tiler_mnk=(128, 64, 128),
    )
    assert config.split_handoff_token_n == 128
    assert config.split_workspace_counter_tile_tokens == 64
    assert config.split_k1_max_active_clusters == 80
    assert config.split_k2_max_active_clusters == 52


def test_split_session_pooling_is_disabled_for_mvp(monkeypatch) -> None:
    config = _split_config()
    backend = _bound_backend(config)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    assert backend._workspace_pool_key(_fleet()) is None


def test_fused_pool_identity_is_preserved_and_split_never_collides(monkeypatch) -> None:
    fused = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
    )
    split = _split_config()
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr(
        "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel."
        "shim.mxfp4_tuner.require_hopper_mxfp4_fused_tuning_device",
        lambda: None,
    )

    fused_key = _bound_backend(fused)._workspace_pool_key(_fleet())
    split_key = _bound_backend(split)._workspace_pool_key(_fleet())
    assert fused_key is not None
    assert split_key is None
    assert "fused" in repr(fused_key)


def test_every_split_identity_axis_changes_config_equality() -> None:
    baseline = _split_config()
    variants = (
        dataclasses.replace(baseline, split_k1_mma_tiler_mnk=(256, 64, 128)),
        dataclasses.replace(baseline, split_k2_mma_tiler_mnk=(256, 64, 128)),
        dataclasses.replace(
            baseline,
            split_k1_cluster_shape_mnk=(2, 1, 1),
            split_k2_cluster_shape_mnk=(2, 1, 1),
        ),
        dataclasses.replace(baseline, split_k1_num_sched_stages=2),
        dataclasses.replace(baseline, split_k2_num_sched_stages=2),
        dataclasses.replace(baseline, split_k1_group_hint=72),
        dataclasses.replace(baseline, split_k2_group_hint=44),
        dataclasses.replace(baseline, split_k1_sm_count=72),
        dataclasses.replace(baseline, split_k2_sm_count=60),
        dataclasses.replace(baseline, split_counter_epoch_banks=2),
        dataclasses.replace(baseline, split_graph_variant="cold_k0"),
        dataclasses.replace(baseline, split_enable_iket=True),
        dataclasses.replace(
            baseline,
            routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        ),
    )
    assert all(variant != baseline for variant in variants)
    assert len(set(map(repr, variants))) == len(variants)


@pytest.mark.parametrize(
    "overrides",
    [
        {"split_k1_mma_tiler_mnk": (256, 32, 128)},
        {"split_k2_mma_tiler_mnk": (256, 64, 128)},
        {
            "split_k1_cluster_shape_mnk": (2, 1, 1),
            "split_k2_cluster_shape_mnk": (2, 1, 1),
        },
        {"split_k1_group_hint": 80},
        {"split_k2_group_hint": 52},
        {"split_k1_num_sched_stages": 2},
        {"split_k2_num_sched_stages": 1},
        {"split_counter_epoch_banks": 2},
        {"split_graph_variant": "cold_k0"},
        {"split_enable_iket": True},
    ],
)
def test_split_none_rejects_partial_explicit_tactic_without_sm_partition(
    overrides,
) -> None:
    with pytest.raises(ValueError, match="require both split_k1_sm_count"):
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            execution_mode="split",
            **overrides,
        )
