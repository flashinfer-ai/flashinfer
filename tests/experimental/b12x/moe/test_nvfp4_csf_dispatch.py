"""Scale reconstruction is a prepared choice without changing expert precision."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import TUNING, MoeDecodeConfig
from b12x.preparation import FrozenMapping
from tests.experimental.b12x.preparation.test_precision_choices import (
    DEVICE,
    _nvfp4_query,
)


@pytest.mark.parametrize("capacity", [4, 32, 128])
def test_prepared_compressed_planes_race_both_scale_decoders(capacity):
    native = replace(_nvfp4_query(), num_tokens=capacity, routed_rows=capacity * 4)
    compressed = replace(native, csf_inline_scales=True)
    native_configs = [c for _, c in TUNING.eligible_plan(native, DEVICE).candidates]
    csf_configs = [c for _, c in TUNING.eligible_plan(compressed, DEVICE).candidates]
    assert all(not c.csf_inline_scales for c in native_configs)
    assert {c.csf_inline_scales for c in csf_configs} == {False, True}
    assert set(native_configs) <= set(csf_configs)
    assert TUNING.encode_query(native) != TUNING.encode_query(compressed)
    for config in csf_configs:
        assert MoeDecodeConfig.from_config(FrozenMapping(config.to_dict())) == config
        if config.csf_inline_scales:
            assert config.backend == "dynamic"
            assert not config.nvfp4_materialize_intermediate
            assert config.dynamic_tile_m in {16, 32}


@pytest.mark.parametrize("mode", ["nvfp4", "w4a8_mx"])
@pytest.mark.parametrize("tile_m", [64, 128])
def test_wide_dynamic_tiles_reject_inline_override(mode, tile_m):
    query = replace(
        _nvfp4_query(),
        quant_mode=mode,
        quant_modes=(mode,),
        source_format="modelopt_nvfp4" if mode == "nvfp4" else "fp4_e8m0_k32",
        num_tokens=512,
        routed_rows=512 * _nvfp4_query().top_k,
        csf_inline_scales=True,
    )
    config = MoeDecodeConfig(
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=tile_m,
        dynamic_route_mode="grouped",
        csf_inline_scales=True,
    )
    with pytest.raises(ValueError, match="prepared compressed planes"):
        TUNING.configure(query, device=DEVICE, override=config)


@pytest.mark.parametrize(
    "changes",
    [
        {"csf_inline_scales": False},
        {"hidden_size": 192},
        {"intermediate_size": 96},
        {"activation": "relu2"},
        {"controls": FrozenMapping({"dynamic_tile_mn": (16, 64)})},
    ],
)
def test_inline_scale_override_rejects_incompatible_storage_or_geometry(changes):
    query = replace(
        _nvfp4_query(), num_tokens=4, routed_rows=16, csf_inline_scales=True
    )
    config = next(
        c
        for _, c in TUNING.eligible_plan(query, DEVICE).candidates
        if c.csf_inline_scales and c.backend == "dynamic"
    )
    with pytest.raises(ValueError, match="prepared compressed planes"):
        TUNING.configure(replace(query, **changes), device=DEVICE, override=config)


@pytest.mark.parametrize("mode", ["nvfp4", "w4a8_mx"])
@pytest.mark.parametrize(
    "tokens,inline_tiles", [(128, {16, 32}), (129, {32}), (256, {32}), (257, set())]
)
def test_inline_tuning_prunes_repeated_expert_tiles(tokens, inline_tiles, mode):
    query = replace(
        _nvfp4_query(),
        quant_mode=mode,
        quant_modes=(mode,),
        source_format="modelopt_nvfp4" if mode == "nvfp4" else "fp4_e8m0_k32",
        num_tokens=tokens,
        routed_rows=tokens * 4,
        csf_inline_scales=True,
    )
    configs = [c for _, c in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {c.dynamic_tile_m for c in configs if c.csf_inline_scales} == inline_tiles
    assert any(not c.csf_inline_scales for c in configs)
    override = MoeDecodeConfig(
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        csf_inline_scales=True,
    )
    assert TUNING.configure(query, device=DEVICE, override=override).pinned == override


@pytest.mark.parametrize(
    "mode,n,backend,materialized",
    [
        ("nvfp4", 512, "micro", False),
        ("w4a8_mx", 512, "micro", False),
        ("w4a8_mx", 576, "micro", False),
        ("nvfp4", 512, "dynamic", True),
        ("w4a16", 512, "w4a16", False),
    ],
)
def test_expansion_only_families_reject_inline_override(mode, n, backend, materialized):
    query = replace(
        _nvfp4_query(),
        quant_mode=mode,
        quant_modes=(mode,),
        source_format="modelopt_nvfp4" if mode == "nvfp4" else "fp4_e8m0_k32",
        intermediate_size=n,
        num_tokens=4,
        routed_rows=16,
        csf_inline_scales=True,
        w4a16_weight_layout="modelopt" if mode == "w4a16" else None,
        w4a16_scale_format="e8m0_k32" if mode == "w4a16" else None,
    )
    configs = [c for _, c in TUNING.eligible_plan(query, DEVICE).candidates]
    expanded = next(
        c
        for c in configs
        if c.backend == backend and c.nvfp4_materialize_intermediate == materialized
    )
    assert not expanded.csf_inline_scales
    with pytest.raises(ValueError, match="prepared compressed planes"):
        TUNING.configure(
            query, device=DEVICE, override=replace(expanded, csf_inline_scales=True)
        )


@pytest.mark.parametrize("clusters", [-1, 0, 188])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("clamp", [False, True])
def test_inline_launch_flag_preserves_warps_clamping_and_cluster_count(
    clusters, inline, clamp
):
    from b12x.moe.fused_moe._impl import (
        _decode_dynamic_launch_policy,
        _encode_dynamic_launch_policy,
    )

    policy = _encode_dynamic_launch_policy(
        volatile_launch_state=True,
        external_route_plan_requested=True,
        planned_tile_m=16,
        planned_direct_routing=False,
        w4a8_n64_repacked=False,
        mma_warps=4,
        swiglu_clamp_after_silu=clamp,
        csf_inline_scales=inline,
        policy_max_active_clusters=clusters,
    )
    assert bool(policy & 256) == clamp
    assert bool(policy & 512) == inline
    assert _decode_dynamic_launch_policy(policy) == (
        True,
        True,
        16,
        False,
        False,
        False,
        4,
        clusters,
    )


@pytest.mark.parametrize(
    "source,mode,n",
    [
        ("modelopt_nvfp4", "w4a16", 512),
        ("fp4_e8m0_k32", "w4a16", 512),
        ("fp4_e8m0_k32", "w4a8_mx", 512),
        ("fp4_e8m0_k32", "w4a8_mx", 576),
    ],
)
@pytest.mark.parametrize("capacity", [1, 8, 32])
def test_fp4_families_expose_inline_and_expansion_choices(source, mode, n, capacity):
    query = replace(
        _nvfp4_query(),
        source_format=source,
        quant_mode=mode,
        quant_modes=(mode,),
        num_tokens=capacity,
        routed_rows=capacity * 4,
        intermediate_size=n,
        csf_inline_scales=True,
        weight_layouts=("mma_packed",) if mode == "w4a16" else ("qmma_repacked",),
        w4a16_weight_layout="packed" if mode == "w4a16" else None,
        w4a16_scale_format=("e4m3_k16" if source == "modelopt_nvfp4" else "e8m0_k32")
        if mode == "w4a16"
        else None,
    )
    configs = [c for _, c in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {c.csf_inline_scales for c in configs} == {False, True}
    assert not TUNING.configure(
        query, device=DEVICE, search=False
    ).default.csf_inline_scales
    for c in configs:
        if c.csf_inline_scales:
            assert replace(c, csf_inline_scales=False) in configs
