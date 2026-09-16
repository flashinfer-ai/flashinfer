"""CPU-only tests for the SM90 FP8 x MXFP4 host config checkpoint."""

from __future__ import annotations

import dataclasses
import unittest
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
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_optimization import (
    normalize_mxfp4_optimization_tactic,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.src.moe_hopper_fp8.mxfp4_policy import (
    Mxfp4Optimizations,
    resolve_mxfp4_optimizations,
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
    return MegaMoEHopperMxfp4Frontend(config), normalize_mxfp4_optimization_tactic(
        tactic
    )


def test_sm90_mxfp4_frontend_reports_requested_and_compiled_effective_tactic():
    frontend, requested = _frontend_with_complete_tactic()
    assert frontend.requested_tactic() == requested

    kernel = SimpleNamespace(
        mxfp4_optimizations=frontend.config.optimizations,
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


def test_sm90_mxfp4_legacy_tactic_resets_strategy_and_compile_identity():
    legacy = _fused_knobs(
        mma_tiler_mnk=(256, 64, 256),
        cluster_shape_mnk=(2, 1, 1),
        active_dispatch_warps=1,
        fold_producer_warps=True,
    )
    config = MegaMoEHopperMxfp4Config(
        rank=0,
        world_size=4,
        num_tokens_per_rank=2048,
        num_topk=6,
        num_total_experts=384,
        hidden=7168,
        intermediate=3072,
        **legacy,
    )
    frontend = MegaMoEHopperMxfp4Frontend(config)
    original_key = frontend._mega_compile_key()
    enabled = dict(legacy, fc2_tail_n8=True, fc1_ready_mode="k256")
    module = "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4"
    with (
        mock.patch(module + ".ensure_not_capturing"),
        mock.patch.object(frontend, "_release_workspace") as release,
    ):
        frontend.apply_knobs(enabled)
        assert frontend._mega_compile_key() != original_key
        assert frontend.config.optimizations.fc1_ready_bits == 48
        assert frontend.requested_tactic()["fc2_tail_n8"] is True
        frontend.apply_knobs(enabled)
        assert release.call_count == 1  # Identical identity keeps the workspace.
        frontend.apply_knobs(legacy)
        assert release.call_count == 2  # Protocol change releases old workspace.
    assert frontend._mega_compile_key() == original_key
    assert frontend.requested_tactic()["fc1_ready_mode"] == "tile"
    assert frontend.requested_tactic()["fc2_tail_n8"] is False
    assert frontend.config.optimizations.fc1_ready_bits == 0


def test_sm90_mxfp4_unsupported_strategy_rejected_before_workspace_release():
    frontend, tactic = _frontend_with_complete_tactic()
    old_config = frontend.config
    with mock.patch.object(frontend, "_release_workspace") as release:
        with pytest.raises(ValueError, match="unsupported"):
            frontend.apply_knobs(dict(tactic, fc1_ready_mode="k256"))
        release.assert_not_called()
    assert frontend.config is old_config


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


def _resolve_optimization(**overrides):
    arguments = dict(
        fp8_scale_mode="mxfp4_hybrid",
        mma_tiler_mnk=(256, 64, 256),
        cluster_shape_mnk=(2, 1, 1),
        static_expert_shape=(96, 6144, 7168),
        world_size=4,
        fc1_early_done_publish=True,
    )
    arguments.update(overrides)
    return resolve_mxfp4_optimizations(**arguments)


class TestMxfp4OptimizationPolicy(unittest.TestCase):
    def test_local_defaults_keep_optional_protocol_off(self):
        selected = _resolve_optimization()
        self.assertTrue(selected.peer32)
        self.assertTrue(selected.offset_bulk)
        self.assertTrue(selected.skip_zero_counts)
        self.assertFalse(selected.fc2_tail_n8)
        self.assertEqual(selected.fc1_ready_mode, "tile")

    def test_other_formats_and_split_keep_legacy_paths(self):
        for overrides in (
            {"fp8_scale_mode": "blockwise"},
            {"fp8_scale_mode": "per_tensor"},
            {"split_role": "k1", "execution_phase": "fc1"},
            {"split_role": "k2", "execution_phase": "fc2"},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(
                    _resolve_optimization(**overrides, skip_zero_counts=True),
                    Mxfp4Optimizations(),
                )

    def test_other_legal_tiles_fall_back_without_rejecting(self):
        for m in (128, 256):
            for n in (8, 16, 32, 64, 128):
                for k in (128, 256):
                    selected = _resolve_optimization(mma_tiler_mnk=(m, n, k))
                    self.assertEqual(selected.offset_bulk, k == 256)
                    self.assertEqual(selected.peer32, (m, n, k) == (256, 64, 256))
        # Policy eligibility does not grant frontend legality to a new tactic.

    def test_protocol_is_optional_and_geometry_checked(self):
        selected = _resolve_optimization(fc1_ready_mode="k256")
        self.assertEqual(
            (selected.fc1_ready_segments, selected.fc1_ready_bits), (12, 48)
        )
        for overrides in (
            {"world_size": 8},
            {"cluster_shape_mnk": (1, 1, 1)},
            {"static_expert_shape": (96, 8192, 7168)},
            {"fc1_store_offload": True},
            {"fc1_early_done_publish": False},
            {"pingpong": True},
            {"token_back_by_dispatch": True},
        ):
            with self.subTest(overrides=overrides):
                self.assertEqual(
                    _resolve_optimization(**overrides).fc1_ready_mode, "tile"
                )
                with self.assertRaises(ValueError):
                    _resolve_optimization(**overrides, fc1_ready_mode="k256")

    def test_optional_tail_is_not_a_frontend_n8_tactic(self):
        self.assertTrue(_resolve_optimization(fc2_tail_n8=True).fc2_tail_n8)
        for tile in ((256, 8, 256), (256, 16, 256), (256, 64, 128)):
            with self.assertRaises(ValueError):
                _resolve_optimization(mma_tiler_mnk=tile, fc2_tail_n8=True)

    def test_no_hidden_default_flags_on_other_writeback_paths(self):
        for overrides in (
            {"token_back_by_dispatch": True},
            {"fc2_in_kernel_topk_reduce": True},
        ):
            self.assertFalse(_resolve_optimization(**overrides).peer32)

    def test_zero_count_has_diagnostic_optout_and_excludes_dedup(self):
        self.assertTrue(_resolve_optimization(skip_zero_counts=True).skip_zero_counts)
        self.assertFalse(_resolve_optimization(skip_zero_counts=False).skip_zero_counts)
        self.assertFalse(
            _resolve_optimization(
                skip_zero_counts=True, dedup_dispatch=True
            ).skip_zero_counts
        )

    def test_strict_types_and_every_effective_choice_has_identity(self):
        for name, value in (
            ("fc2_tail_n8", 1),
            ("fc1_ready_mode", "auto"),
            ("local_optimizations", None),
            ("skip_zero_counts", "0"),
        ):
            with self.assertRaises(ValueError):
                _resolve_optimization(**{name: value})
        base = _resolve_optimization()
        for name in ("peer32", "offset_bulk", "skip_zero_counts", "fc2_tail_n8"):
            changed = dataclasses.replace(base, **{name: not getattr(base, name)})
            self.assertNotEqual(base.identity(), changed.identity())
        self.assertNotEqual(
            base.identity(), _resolve_optimization(fc1_ready_mode="k256").identity()
        )

    def test_diagnostic_legacy_and_independence_from_total_tokens(self):
        self.assertEqual(
            _resolve_optimization(local_optimizations=False, skip_zero_counts=False),
            Mxfp4Optimizations(),
        )
        # There is intentionally no num_tokens argument or per-token source table.
        import inspect

        self.assertNotIn(
            "num_tokens",
            inspect.signature(resolve_mxfp4_optimizations).parameters,
        )

    def test_cross_h_local_defaults_keep_protocols_off(self):
        for hidden in (4096, 6144, 7168, 8192):
            with self.subTest(hidden=hidden):
                selected = _resolve_optimization(static_expert_shape=(96, 6144, hidden))
                self.assertTrue(selected.peer32)
                self.assertTrue(selected.offset_bulk)
                self.assertTrue(selected.skip_zero_counts)
                self.assertFalse(selected.fc2_tail_n8)
                self.assertEqual(selected.fc1_ready_mode, "tile")
                self.assertEqual(
                    (selected.fc1_ready_segments, selected.fc1_ready_bits), (1, 0)
                )

    def test_cross_h_requires_complete_channel_clusters(self):
        for hidden in (0, -512, 4352, 7167, 7424):
            self.assertFalse(
                _resolve_optimization(static_expert_shape=(96, 6144, hidden)).peer32
            )
        self.assertFalse(_resolve_optimization(static_expert_shape=None).peer32)
        self.assertFalse(_resolve_optimization(cluster_shape_mnk=(0, 1, 1)).peer32)

    def test_cross_h_does_not_widen_optional_protocols(self):
        for hidden in (4096, 6144, 8192):
            for option in ({"fc2_tail_n8": True}, {"fc1_ready_mode": "k256"}):
                with (
                    self.subTest(hidden=hidden, option=option),
                    self.assertRaises(ValueError),
                ):
                    _resolve_optimization(
                        static_expert_shape=(96, 6144, hidden), **option
                    )
        self.assertTrue(_resolve_optimization(fc2_tail_n8=True).fc2_tail_n8)
        self.assertEqual(
            _resolve_optimization(fc1_ready_mode="k256").fc1_ready_bits, 48
        )

    def test_cross_h_keeps_the_same_mma_tile_domain(self):
        for hidden in (4096, 6144, 7168, 8192):
            for m in (128, 256):
                for n in (8, 16, 32, 64, 128):
                    for k in (128, 256):
                        selected = _resolve_optimization(
                            static_expert_shape=(96, 6144, hidden),
                            mma_tiler_mnk=(m, n, k),
                        )
                        self.assertEqual(selected.peer32, (m, n, k) == (256, 64, 256))
                        self.assertEqual(selected.offset_bulk, k == 256)

    def test_cross_h_keeps_other_formats_and_writeback_guards(self):
        for hidden in (4096, 6144, 7168, 8192):
            for overrides in (
                {"fp8_scale_mode": "blockwise"},
                {"fp8_scale_mode": "per_tensor"},
                {"split_role": "k1", "execution_phase": "fc1"},
                {"split_role": "k2", "execution_phase": "fc2"},
                {"pingpong": True},
                {"token_back_by_dispatch": True},
                {"fc2_in_kernel_topk_reduce": True},
                {"dedup_dispatch": True},
                {"local_optimizations": False},
            ):
                with self.subTest(hidden=hidden, overrides=overrides):
                    self.assertFalse(
                        _resolve_optimization(
                            static_expert_shape=(96, 6144, hidden), **overrides
                        ).peer32
                    )
