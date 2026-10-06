"""Dominance predicates and ladders of the raced parameter spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import pytest

from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)


# MoE cluster ladders depend on SM count; A16 modes depend on compute capability.
IDENTITY = DeviceIdentity("nvidia", (12, 0), 188, "synthetic SM120")


def test_predicates_see_complete_conditional_assignments():
    knobs = (
        Knob(name="backend", values=("native", "tiled")),
        Knob(
            name="tile",
            values=None,
            when=FrozenMapping({"backend": "tiled"}),
            otherwise=0,
        ),
        Knob(
            name="splits",
            values=(1, 2, 4),
            binding=ParameterBinding.RUNTIME,
            when=FrozenMapping({"backend": "tiled"}),
            otherwise=1,
        ),
    )
    space = ParameterSpace.create(
        knobs,
        values={"tile": (8, 16)},
        predicates=(
            lambda p: p["backend"] == "native" or p["tile"] * p["splits"] <= 32,
        ),
    )
    assignments = list(space.configurations())
    assert {(p["backend"], p["tile"], p["splits"]) for p in assignments} == {
        ("native", 0, 1),
        ("tiled", 8, 1),
        ("tiled", 8, 2),
        ("tiled", 8, 4),
        ("tiled", 16, 1),
        ("tiled", 16, 2),
    }
    with pytest.raises(ValueError, match="predicates"):
        space.validate(dict(backend="tiled", tile=16, splits=4))


def test_predicates_filter_before_materialization_and_keep_raw_count():
    @dataclass(frozen=True)
    class Config:
        tile: int
        splits: int

    knobs = (Knob(name="tile", values=(8, 16)), Knob(name="splits", values=(1, 2, 4)))

    def parameters(capacity, device):
        return ParameterSpace.create(
            knobs,
            predicates=(lambda p: p["tile"] * p["splits"] <= capacity,),
        )

    def materialize(capacity, device, choice):
        if choice["tile"] * choice["splits"] > capacity:
            raise AssertionError("filtered assignment reached the constructor")
        return Config(**choice)

    contract = TuningContract(
        component_id="test.predicates",
        query_schema_version=1,
        config_schema_version=1,
        query_fields=frozenset({"capacity"}),
        config_fields=frozenset({"tile", "splits"}),
        encode_query=lambda capacity: {"capacity": capacity},
        encode_config=asdict,
        decode_config=lambda payload: Config(**dict(payload)),
        default_config=lambda capacity, device: Config(8, 1),
        validate_query=lambda capacity, device: None,
        validate_config=lambda query, config, device: None,
        knobs=knobs,
        parameters=parameters,
        materialize=materialize,
    )
    plan = contract.eligible_plan(32, None)
    assert plan.cartesian_count == 6
    assert plan.legal_count == 5
    assert {(config.tile, config.splits) for _, config in plan.candidates} == {
        (8, 1),
        (8, 2),
        (8, 4),
        (16, 1),
        (16, 2),
    }
    with pytest.raises(ValueError, match="predicates"):
        contract.lower(32, None, dict(tile=16, splits=4))

    def broken(assignment):
        raise ValueError("planner defect")

    invalid = replace(
        contract,
        parameters=lambda *_: ParameterSpace(knobs=knobs, predicates=(broken,)),
    )
    with pytest.raises(ValueError, match="planner defect"):
        invalid.eligible_plan(32, None)


def test_mhc_predicates_remove_inactive_work_without_rejecting_useful_tails():
    from b12x.norm.mhc import _tuning as component

    query = component.MhcQuery(
        dtype="bfloat16",
        max_tokens=1,
        hidden_size=4096,
        split_k=64,
        operation="post_pre",
        has_norm_weight=True,
        norm_eps=1e-6,
        smem_limit=100 << 10,
    )
    choice = dict(
        backend="tf32_tma",
        lagged_prepare=False,
        partials_per_cta=4,
        projection_tile_n=8,
        projection_tile_k=64,
        projection_num_stages=2,
        projection_num_m_warps=1,
        projection_num_n_warps=1,
        projection_k_splits=8,
    )
    decode = component.TUNING.parameter_space(query, None)
    with pytest.raises(ValueError, match="predicates"):
        decode.validate({**choice, "projection_num_m_warps": 2})
    with pytest.raises(ValueError, match="predicates"):
        decode.validate(
            {**choice, "projection_tile_n": 32, "projection_num_n_warps": 4}
        )
    with pytest.raises(ValueError, match="predicates"):
        decode.validate({**choice, "projection_tile_n": 64})
    with pytest.raises(ValueError, match="predicates"):
        decode.validate(
            {
                **choice,
                "projection_tile_k": 256,
                "projection_k_splits": 32,
                "projection_num_stages": 3,
            }
        )

    boundary = component.TUNING.parameter_space(replace(query, max_tokens=128), None)
    # Forty-eight-row tiles have a partial final CTA at M128, but every warp
    # still does useful work in earlier CTAs. N32 remains a useful N24 overtile.
    tail = {
        **choice,
        "projection_num_m_warps": 3,
        "projection_tile_n": 32,
        "projection_num_n_warps": 2,
    }
    assignments = set(boundary.configurations())
    assert FrozenMapping(tail) in assignments
    assert (
        FrozenMapping({**choice, "projection_tile_n": 24, "projection_num_n_warps": 3})
        in assignments
    )
    with pytest.raises(ValueError, match="predicates"):
        boundary.validate(
            {
                **choice,
                "projection_num_m_warps": 8,
                "projection_tile_n": 24,
                "projection_num_n_warps": 3,
                "projection_tile_k": 256,
                "projection_num_stages": 4,
            }
        )


@pytest.mark.parametrize("tile_k", [8, 16])
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
def test_mhc_tf32_rejects_short_weight_rows(tile_k, operation):
    from b12x.norm.mhc import _tuning as component
    from b12x.norm.mhc._kernels import MHCPrefillTf32ProjectTmaKernel

    query = component.MhcQuery(
        dtype="bfloat16",
        max_tokens=1,
        hidden_size=5120,
        split_k=80,
        operation=operation,
        has_norm_weight=True,
        lagged_mix=True,
        expanded_residual=operation == "pre",
        norm_eps=1e-20,
        rms_eps=1e-20,
        smem_limit=100 << 10,
    )
    choice = dict(
        backend="tf32_tma",
        lagged_prepare=False,
        partials_per_cta=4,
        projection_tile_n=8,
        projection_tile_k=tile_k,
        projection_num_stages=3,
        projection_num_m_warps=1,
        projection_num_n_warps=1,
        projection_k_splits=2,
    )
    space = component.TUNING.parameter_space(query, None)
    with pytest.raises(ValueError, match="predicates"):
        space.validate(choice)
    config = component.MhcConfig(projection_tile_m=16, **choice)
    with pytest.raises(ValueError, match="at least 32"):
        component.TUNING.validate_config(query, config, None)
    with pytest.raises(ValueError, match="at least 32"):
        MHCPrefillTf32ProjectTmaKernel(
            hidden_size=5120,
            split_k=80,
            tile_m=16,
            tile_n=8,
            tile_k=tile_k,
            num_stages=3,
            num_m_warps=1,
            num_n_warps=1,
            k_splits=2,
            split_fp32_fn=True,
        )
    space.validate({**choice, "projection_tile_k": 32})


@pytest.mark.parametrize("partials", [None, "13", "25", "0"])
def test_mhc_prepared_post_pre_honors_partial_group_control(partials):
    """The fused lagged producer must not silently ignore its grouping control."""
    from types import SimpleNamespace

    from b12x.norm.mhc import _preparation, _tuning

    query = _tuning.MhcQuery(
        dtype="bfloat16",
        max_tokens=8,
        hidden_size=5120,
        split_k=80,
        operation="post_pre",
        has_norm_weight=True,
        lagged_mix=True,
        rms_eps=1e-20,
        norm_eps=1e-20,
        controls=FrozenMapping(
            {}
            if partials is None
            else {
                "B12X_MHC_PARTIALS_PER_CTA": partials,
            }
        ),
    )
    device = SimpleNamespace(
        identity=DeviceIdentity(
            "nvidia",
            (12, 1),
            48,
            "NVIDIA GB10",
        )
    )
    if partials == "0":
        with pytest.raises(ValueError, match="partials_per_cta"):
            _tuning.TUNING.configure(query, device=device.identity, search=False)
    else:
        config = _tuning.TUNING.configure(
            query, device=device.identity, search=False
        ).default
        assert config.lagged_prepare
        launch = _preparation._lower_native(query, config, device)
        assert launch.partials_per_cta == (4 if partials is None else int(partials))


@pytest.mark.parametrize("operation", ["pre", "post_pre"])
def test_mhc_races_grouping_only_for_fused_lagged_producers(operation):
    """Grouping changes the producer geometry without duplicating inactive routes."""
    from types import SimpleNamespace

    from b12x.norm.mhc import _preparation, _tuning

    query = _tuning.MhcQuery(
        dtype="bfloat16",
        max_tokens=8,
        hidden_size=5120,
        split_k=80,
        operation=operation,
        has_norm_weight=True,
        lagged_mix=True,
        expanded_residual=operation == "pre",
        rms_eps=1e-20,
        norm_eps=1e-20,
        controls=FrozenMapping({"B12X_MHC_PREFILL_TF32_MMA": "0"}),
    )
    device = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")
    choices = list(_tuning.TUNING.parameter_space(query, device).configurations())
    assert {(p["lagged_prepare"], p["partials_per_cta"]) for p in choices} == {
        (False, 4),
        (True, 4),
        (True, 9),
        (True, 13),
        (True, 25),
    }
    assert len(choices) == 5
    for choice in choices:
        config = _tuning.TUNING.lower(query, device, choice)
        launch = _preparation._lower_native(
            query, config, SimpleNamespace(identity=device)
        )
        assert launch.partials_per_cta == choice["partials_per_cta"]
    pinned = replace(
        query,
        controls=FrozenMapping(
            {
                **query.controls,
                "B12X_MHC_PARTIALS_PER_CTA": "13",
            }
        ),
    )
    choices = list(_tuning.TUNING.parameter_space(pinned, device).configurations())
    assert {(p["lagged_prepare"], p["partials_per_cta"]) for p in choices} == {
        (False, 4),
        (True, 13),
    }


@pytest.mark.parametrize("tokens,partials", [(2, 4), (4, 9), (8, 25)])
def test_mhc_grouping_preserves_spark_pre_defaults(tokens, partials):
    from b12x.norm.mhc import _tuning

    query = _tuning.MhcQuery(
        dtype="bfloat16",
        max_tokens=tokens,
        hidden_size=4096,
        split_k=64,
        operation="pre",
        lagged_mix=True,
        expanded_residual=True,
    )
    device = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")
    config = _tuning.TUNING.configure(query, device=device, search=False).default
    assert config.lagged_prepare
    assert config.partials_per_cta == partials


def test_gdn_prefill_races_one_segment_plan_and_a_window_ladder():
    from b12x.sequence.gdn_prefill import _tuning as component

    query = component.GdnPrefillQuery(
        key_heads=8,
        value_heads=24,
        head_dim=128,
        model_dtype="bfloat16",
        state_dtype="float32",
        qk_l2norm=True,
        checkpoint_export=True,
        max_tokens=128,
        max_seqs=1,
        null_state_index=0,
        dt_bias_dtype="bfloat16",
    )
    space = component.TUNING.parameter_space(query, IDENTITY)
    sequential = dict(
        backend=component.BACKEND,
        algorithm="sequential",
        v_split=32,
        k_split=1,
        stages=3,
        segment_tokens=256,
        window_tiles=9,
    )
    windows = {
        assignment["window_tiles"]
        for assignment in space.configurations()
        if assignment["algorithm"] == "sequential"
    }
    # cap = ceil(128/16) + 1 = 9 tiles, raced as one, two and four windows.
    assert windows == {9, 5, 3}
    space.validate(sequential)
    with pytest.raises(ValueError, match="efficiency predicates"):
        space.validate({**sequential, "window_tiles": 8})
    exhaustive = component.TUNING.parameter_space(
        replace(query, exhaustive=True), IDENTITY
    )
    exhaustive.validate({**sequential, "window_tiles": 8})
    exhaustive.validate(
        {
            **sequential,
            "algorithm": "chunk_parallel",
            "segment_tokens": 1024,
            "window_tiles": None,
        }
    )
    # A 128-token sequence is one 128-token segment: chunk-parallel has no
    # chunk-level parallelism to exploit and is not raced at all.
    assert not any(
        assignment["algorithm"] == "chunk_parallel"
        for assignment in space.configurations()
    )
    with pytest.raises(ValueError, match="predicates"):
        space.validate(
            {
                **sequential,
                "algorithm": "chunk_parallel",
                "segment_tokens": 128,
                "window_tiles": None,
            }
        )

    longer = component.TUNING.parameter_space(replace(query, max_tokens=512), IDENTITY)
    chunked = {
        assignment["segment_tokens"]
        for assignment in longer.configurations()
        if assignment["algorithm"] == "chunk_parallel"
    }
    # Segments at or above the sequence plan one segment per sequence; 512 is
    # the smallest of those, and 1024 would only widen the shared-memory band.
    assert chunked == {128, 256, 512}


def test_moe_cluster_ladder_stops_at_the_planned_task_queue():
    from b12x.moe.fused_moe import _tuning as component

    query = component.MoeDecodeQuery(
        quant_mode="nvfp4",
        quant_modes=("nvfp4",),
        source_format="modelopt_nvfp4",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=512,
        hidden_size=2560,
        intermediate_size=320,
        top_k=10,
        num_tokens=1,
        routed_rows=10,
        route_num_experts=512,
        route_logits_dtype="float32",
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=False,
        swiglu_limit=None,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        w13_layout="fused",
        weight_layouts=("fused",),
        w4a16_weight_layout=None,
        w4a16_scale_format=None,
        w4a16_block_size_m=None,
        fast_math=True,
        numerical_recipe=None,
        controls=FrozenMapping(),
    )
    space = component.TUNING.parameter_space(query, IDENTITY)
    clusters = {
        assignment["max_active_clusters"]
        for assignment in space.configurations()
        if assignment["route_planner"] == "triton"
    }
    # Ten routed rows over three N128 slices plan thirty grouped tasks; wider
    # grids only add clusters that take no task.
    assert clusters == {None, 1, 2, 4, 8, 16, 30}
    triton = dict(
        backend="dynamic",
        route_planner="triton",
        dynamic_tile_m=16,
        dynamic_route_mode="grouped",
        w4a16_route_mode=None,
        max_active_clusters=30,
        nvfp4_share_input=False,
        nvfp4_materialize_intermediate=False,
        nvfp4_inline_scales=False,
        w4a16_tile_config=None,
        w4a16_block_size_m=None,
        w4a16_pipeline_stages=None,
    )
    space.validate(triton)
    for outside in (31, 64, 188):
        with pytest.raises(ValueError, match="eligible values"):
            space.validate({**triton, "max_active_clusters": outside})


def test_a16_wide_tile_needs_more_than_one_n_tile():
    from b12x.gemm.blockscaled._tuning import TUNING, BlockscaledQuery

    narrow = BlockscaledQuery(
        recipe="mxfp8",
        num_tokens=1,
        in_features=2560,
        padded_in_features=2560,
        out_features=48,
    )
    space = TUNING.parameter_space(narrow, IDENTITY)
    assignment = dict(mode="a16", tile_m=16, tile_n=64, tile_k=64, split_k=1)
    space.validate(assignment)
    with pytest.raises(ValueError, match="predicates"):
        space.validate({**assignment, "tile_n": 128})
    wide = TUNING.parameter_space(replace(narrow, out_features=128), IDENTITY)
    wide.validate({**assignment, "tile_n": 128})


def test_iq2_transposed_tiles_are_raced_without_split_k():
    from b12x.gemm.blockscaled._tuning import (
        TUNING,
        BlockscaledConfig,
        BlockscaledQuery,
    )

    query = BlockscaledQuery(
        recipe="iq2_xs",
        num_tokens=8,
        in_features=768,
        padded_in_features=768,
        out_features=136,
    )
    assignment = dict(mode="a16", tile_m=8, tile_n=128, tile_k=256, split_k=1)
    TUNING.parameter_space(query, IDENTITY).validate(assignment)
    TUNING.validate_config(query, BlockscaledConfig(**assignment), IDENTITY)
    with pytest.raises(ValueError, match="split-K"):
        TUNING.validate_config(
            query, BlockscaledConfig(**{**assignment, "split_k": 2}), IDENTITY
        )
    for recipe in ("nvfp4", "mxfp8"):
        with pytest.raises(ValueError, match="predicates"):
            TUNING.parameter_space(replace(query, recipe=recipe), IDENTITY).validate(
                {**assignment, "tile_k": 128}
            )


def test_nvfp4_a16_races_k256_tiles():
    from b12x.gemm.blockscaled._tuning import (
        TUNING,
        BlockscaledConfig,
        BlockscaledQuery,
    )

    query = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=64,
        in_features=800,
        padded_in_features=800,
        out_features=136,
        activation_mode="a16",
    )
    for tile_m in (16, 32, 64):
        assignment = dict(mode="a16", tile_m=tile_m, tile_n=128, tile_k=256, split_k=4)
        TUNING.parameter_space(query, IDENTITY).validate(assignment)
        TUNING.validate_config(query, BlockscaledConfig(**assignment), IDENTITY)
    with pytest.raises(ValueError, match="predicates"):
        TUNING.parameter_space(
            replace(query, recipe="mxfp8", global_scale_kind="none"), IDENTITY
        ).validate({**assignment, "tile_m": 16})


def test_gate_mean_partitions_span_one_warp_to_the_covering_block():
    from b12x.norm.hyperconnection._tuning import TUNING, HyperConnectionQuery

    query = HyperConnectionQuery(
        dtype="bfloat16",
        max_tokens=128,
        hidden_size=2560,
        streams=4,
        lowrank=4,
        operation="gate_mean",
    )
    space = TUNING.parameter_space(query, IDENTITY)
    blocks = {assignment["pointwise_block"] for assignment in space.configurations()}
    assert blocks == {32, 64, 128, 256, 512, 1024, 2048, 4096}
