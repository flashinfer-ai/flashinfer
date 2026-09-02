"""CPU-only tests for the SM90 FP8 x MXFP4 host config checkpoint."""

from __future__ import annotations

import dataclasses

import pytest

from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.hopper_mxfp4 import (
    MegaMoEHopperMxfp4Config,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


def _config(**overrides):
    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128, top_k=4, **overrides
    )


def test_sm90_mxfp4_config_defaults_are_format_specific():
    cfg = _config()
    assert cfg.kernel_name == "sm90_fp8_mxfp4_bf16_pull_cutedsl"
    assert cfg.kind == "fp8_e4m3"
    assert cfg.fp8_scale_mode == "mxfp4_hybrid"
    assert cfg.fp8_accum_mode == "1xacc"
    assert cfg.humming_max_range == 11
    assert cfg.preprocess_expert_chunk_size == 4
    assert cfg.swap_ab is None  # MXFP4-specific heuristic, never native fallback.
    assert cfg.in_kernel_fc2_reduce is False
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
        _config(knobs={"swap_ab": True, "flag_batch": 4}, swap_ab=True)


def test_sm90_mxfp4_explicit_knobs_cannot_select_native_ab():
    with pytest.raises(ValueError, match="swap_ab=True"):
        _config(knobs={"swap_ab": False, "flag_batch": 4})
    with pytest.raises(ValueError, match="swap_ab=True"):
        _config(knobs={"flag_batch": 4})


@pytest.mark.parametrize(
    "knobs",
    [
        {"swap_ab": True, "world_size": 1},
        {"swap_ab": True, "fp8_scale_mode": "per_tensor"},
        {"swap_ab": True, "typo_flag_bach": 4},
    ],
)
def test_sm90_mxfp4_explicit_knobs_reject_non_tactic_fields(knobs):
    with pytest.raises(ValueError, match="unsupported MXFP4 knob"):
        _config(knobs=knobs)


def test_sm90_mxfp4_explicit_knobs_preserve_fixed_numerics():
    with pytest.raises(ValueError, match="1xacc"):
        _config(knobs={"swap_ab": True, "fp8_accum_mode": "2xacc"})
    with pytest.raises(ValueError, match="standalone top-k"):
        _config(knobs={"swap_ab": True, "in_kernel_fc2_reduce": True})
