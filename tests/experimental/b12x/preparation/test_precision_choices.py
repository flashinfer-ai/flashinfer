"""Precision and materialization choices retain their numerical prerequisites."""

from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")


@pytest.mark.parametrize("mode", ("auto", "quantized", "a16"))
@pytest.mark.parametrize("cutoff", (0, 32, 128, 256))
def test_dense_a16_cutoff_constrains_candidates_and_covers_unlisted_counts(
    mode, cutoff
):
    from types import SimpleNamespace
    import torch
    from b12x.gemm.blockscaled import BlockscaledQuery, plan_regimes

    query = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=128,
        in_features=256,
        padded_in_features=256,
        out_features=128,
        activation_mode=mode,
        activation_scale_available=True,
    )
    plan = plan_regimes(query, exact_m=(4, 64), a16_max_tokens=cutoff)
    states = {}
    for rows, child in plan.variants.items():
        expected = "a16" if rows <= cutoff else mode
        assert child.query.activation_mode == expected
        candidates = child.contract.eligible_plan(child.query, DEVICE).candidates
        assert candidates
        if expected != "auto":
            assert all(config.mode == expected for _, config in candidates)
        states[rows] = SimpleNamespace(query=child.query, required_workspace=rows)
    state = plan._assemble(states, None)
    for rows in (1, 4, 17, 31, 32, 33, 64, 127, 128):
        selected = state.resolve(torch.empty(rows, 256, device="meta"))
        assert selected.query.activation_mode == ("a16" if rows <= cutoff else mode)
        assert selected.query.num_tokens >= rows
    assert state.required_workspace == 128


@pytest.mark.parametrize("cutoff", (-1, 1.5, True))
def test_a16_cutoff_rejects_invalid_values(cutoff):
    import torch
    from b12x.gemm.blockscaled import BlockscaledQuery, plan_regimes
    from b12x.moe.fused_moe import ActivationMode, ActivationSpec

    query = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=128,
        in_features=256,
        padded_in_features=256,
        out_features=128,
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        plan_regimes(query, a16_max_tokens=cutoff)
    with pytest.raises(ValueError, match="nonnegative integer"):
        ActivationSpec(
            mode=ActivationMode.A4,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            a16_max_tokens=cutoff,
        )


@pytest.mark.parametrize("rows", (1, 8, 128))
def test_dense_nvfp4_auto_races_a4_and_a16(rows):
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=rows,
        in_features=2560,
        padded_in_features=2560,
        out_features=512,
        activation_mode="auto",
        activation_scale_available=True,
    )
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.mode for config in configs} == {"a16", "quantized"}
    assert TUNING.configure(query, device=DEVICE).default.mode == (
        "a16" if rows <= 8 else "quantized"
    )
    for mode in ("a16", "quantized"):
        forced = replace(query, activation_mode=mode)
        assert {
            config.mode for _, config in TUNING.eligible_plan(forced, DEVICE).candidates
        } == {mode}
        selected = next(config for config in configs if config.mode == mode)
        assert (
            TUNING.configure(query, device=DEVICE, override=selected).default
            == selected
        )


def test_dense_nvfp4_without_activation_scale_excludes_a4():
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=1,
        in_features=256,
        padded_in_features=256,
        out_features=128,
        activation_mode="auto",
        activation_scale_available=False,
    )
    assert {
        config.mode for _, config in TUNING.eligible_plan(query, DEVICE).candidates
    } == {"a16"}


@pytest.mark.parametrize("recipe", ("nvfp4", "iq2_xs", "iq2_xxs", "q8_0"))
@pytest.mark.parametrize("sm_count", (48, 188))
@pytest.mark.parametrize(
    "rows,k,n", ((1, 512, 768), (2, 2048, 640), (16, 3072, 1536), (256, 2048, 768))
)
def test_dense_heuristic_respects_zero_split_workspace(recipe, sm_count, rows, k, n):
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe=recipe,
        num_tokens=rows,
        in_features=k,
        padded_in_features=k,
        out_features=n,
        activation_mode="a16",
        workspace_form="provided",
        workspace_nbytes=0,
    )
    device = DeviceIdentity("nvidia", (12, 0), sm_count, "test device")
    selected = TUNING.configure(query, device=device, search=False).default
    assert selected.split_k == 1


@pytest.mark.parametrize("k,n", ((4096, 18560), (8192, 4096)))
@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16, 256))
def test_super3_mamba_tuning_can_select_wide_k_without_changing_precision(k, n, rows):
    from itertools import product

    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING

    query = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=rows,
        in_features=k,
        padded_in_features=k,
        out_features=n,
        activation_mode="a16",
        activation_scale_available=False,
        output_mode="functional",
        workspace_form="provided",
        workspace_nbytes=2_000_000_000,
        expected_m=rows if rows <= 16 else None,
    )
    device = DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10")
    configs = [config for _, config in TUNING.eligible_plan(query, device).candidates]
    assert {config.mode for config in configs} == {"a16"}
    # Every production decode tile/K-split combination must survive filtering;
    # prefill additionally admits the larger row tiles supported by this path.
    row_tiles = (16,) if rows <= 16 else (16, 32, 64)
    assert {(c.tile_m, c.tile_n, c.tile_k, c.split_k) for c in configs} == set(
        product(row_tiles, (64, 128), (64, 128, 256), (1, 2, 4, 8))
    )


def _nvfp4_query():
    from b12x.moe.fused_moe._tuning import MoeDecodeQuery

    return MoeDecodeQuery(
        quant_mode="nvfp4",
        quant_modes=("nvfp4",),
        source_format="modelopt_nvfp4",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=32,
        hidden_size=512,
        intermediate_size=512,
        top_k=4,
        num_tokens=1024,
        routed_rows=4096,
        route_num_experts=32,
        route_logits_dtype=None,
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=False,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        w13_layout="w13",
        weight_layouts=("mma_view",),
        w4a16_weight_layout=None,
        w4a16_scale_format=None,
        w4a16_block_size_m=None,
        fast_math=True,
        numerical_recipe=None,
        controls=FrozenMapping(),
        shared_input_scales=True,
    )


def _nvfp4_auto_query(rows=1):
    return replace(
        _nvfp4_query(),
        quant_mode="nvfp4_auto",
        quant_modes=("nvfp4", "w4a16"),
        num_experts=256,
        hidden_size=6144,
        intermediate_size=256,
        top_k=8,
        num_tokens=rows,
        routed_rows=rows * 8,
        route_num_experts=256,
        weight_layouts=("mma_view", "source_native"),
        w4a16_weight_layout="modelopt",
        w4a16_scale_format="e4m3_k16",
    )


def test_moe_a4_prefill_options_are_immutable_numerical_controls(monkeypatch):
    import torch
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe import _preparation
    from b12x.moe.fused_moe._tuning import TUNING

    monkeypatch.setenv("B12X_W4A16_A4_PREFILL", "1")
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_TERMS", "1")
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_WARPS", "8")
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(
            num_experts=32, hidden_size=512, intermediate_size=512
        ),
    )._impl
    query = replace(
        _nvfp4_query(),
        quant_mode="w4a16",
        quant_modes=("w4a16",),
        weight_layouts=tuple(layout.value for layout in weight_plan.weight_layouts),
        w4a16_weight_layout=weight_plan.w4a16_weight_layout,
        w4a16_scale_format=weight_plan.w4a16_scale_format,
        controls=_preparation._control_snapshot(),
    )
    config = TUNING.configure(query, device=DEVICE, search=False).default
    for variable, value in (
        ("B12X_W4A16_A4_PREFILL", "0"),
        ("B12X_W4A16_A4_PREFILL_TERMS", "2"),
        ("B12X_W4A16_A4_PREFILL_WARPS", "16"),
    ):
        with monkeypatch.context() as changed:
            changed.setenv(variable, value)
            different = replace(query, controls=_preparation._control_snapshot())
            assert TUNING.encode_query(query) != TUNING.encode_query(different)
            caps = _preparation._lower_caps(
                query, config, weight_plan, torch.device("cuda", 0)
            )
            assert (
                caps.w4a16_a4_prefill_enabled,
                caps.w4a16_a4_prefill_terms,
                caps.w4a16_a4_prefill_warps,
            ) == (True, 1, 8)


@pytest.mark.parametrize("enabled", (None, "0"))
@pytest.mark.parametrize(
    "variable,value",
    (
        ("B12X_W4A16_A4_PREFILL_TERMS", "3"),
        ("B12X_W4A16_A4_PREFILL_WARPS", "4"),
    ),
)
def test_disabled_moe_a4_prefill_ignores_inactive_controls(
    monkeypatch, enabled, variable, value
):
    from b12x.moe.fused_moe import _preparation

    if enabled is None:
        monkeypatch.delenv("B12X_W4A16_A4_PREFILL", raising=False)
    else:
        monkeypatch.setenv("B12X_W4A16_A4_PREFILL", enabled)
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_TERMS", "1")
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_WARPS", "8")
    baseline = _preparation._control_snapshot()
    monkeypatch.setenv(variable, value)
    assert _preparation._control_snapshot() == baseline
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL", "1")
    with pytest.raises(ValueError, match=variable):
        _preparation._control_snapshot()


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_moe_a4_prefill_ignores_retired_token_threshold(enabled, monkeypatch):
    from b12x.moe.fused_moe import _preparation

    monkeypatch.setenv("B12X_W4A16_A4_PREFILL", enabled)
    monkeypatch.delenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", raising=False)
    baseline = _preparation._control_snapshot()
    for value in ("1", "3072", "invalid"):
        monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", value)
        assert _preparation._control_snapshot() == baseline


@pytest.mark.parametrize("value", ["-1", "2", "true"])
def test_moe_a4_prefill_enable_rejects_non_boolean_controls(value, monkeypatch):
    from b12x.moe.fused_moe import _preparation

    monkeypatch.setenv("B12X_W4A16_A4_PREFILL", value)
    with pytest.raises(ValueError, match="B12X_W4A16_A4_PREFILL"):
        _preparation._control_snapshot()


@pytest.mark.parametrize("capability,sms", (((12, 0), 188), ((12, 1), 48)))
@pytest.mark.parametrize("rows", (1, 2, 4, 8, 9, 16))
def test_moe_auto_promotes_native_decode_and_races_both_precisions(
    capability, sms, rows
):
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_auto_query(rows)
    device = DeviceIdentity("nvidia", capability, sms, "synthetic Blackwell")
    default = TUNING.configure(query, device=device).default
    configs = [config for _, config in TUNING.eligible_plan(query, device).candidates]
    assert (default.backend == "w4a16") is (rows <= 8)
    assert {
        config.w4a16_route_mode for config in configs if config.backend == "w4a16"
    } == ({"direct", "packed"} if rows <= 8 else {"packed"})
    assert {config.backend for config in configs} >= {"dynamic", "w4a16"}
    assert default in configs
    assert configs[0].backend == "w4a16"
    pinned_a4 = replace(query, quant_mode="nvfp4", quant_modes=("nvfp4",))
    assert all(
        config.backend != "w4a16"
        for _, config in TUNING.eligible_plan(pinned_a4, device).candidates
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"hidden_size": 192},
        {"top_k": 33, "routed_rows": 33},
        {"apply_router_weight_on_input": True},
        {"collect_activation_amax": True},
        {"io_dtype": "float16"},
    ),
)
def test_moe_native_direct_rejects_unsupported_query_and_override(changes):
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_auto_query()
    direct = TUNING.configure(query, device=DEVICE).default
    query = replace(query, **changes)
    assert TUNING.configure(query, device=DEVICE).default.backend != "w4a16"
    with pytest.raises(ValueError, match="direct routing"):
        TUNING.configure(query, device=DEVICE, override=direct)
    assert not any(
        config.backend == "w4a16" and config.w4a16_route_mode == "direct"
        for _, config in TUNING.eligible_plan(query, DEVICE).candidates
    )


def test_moe_auto_has_no_native_decode_default_on_other_architectures():
    from b12x.moe.fused_moe._tuning import TUNING

    for device in (None, replace(DEVICE, compute_capability=(10, 0))):
        assert (
            TUNING.configure(
                _nvfp4_auto_query(), device=device, search=False
            ).default.backend
            != "w4a16"
        )


@pytest.mark.parametrize(
    "a4_latency,a16_latency,old_backend,backend",
    (
        (6, 6, "micro", "w4a16"),
        (5, 6, "micro", "micro"),
        (6, 5, "w4a16", "w4a16"),
    ),
)
def test_moe_precision_race_and_revision_invalidate_cached_choice(
    tmp_path,
    monkeypatch,
    a4_latency,
    a16_latency,
    old_backend,
    backend,
):
    from types import SimpleNamespace
    from b12x.preparation import DetectedDevice, MemoryRequirements, Plan, PreparedCall
    from b12x.moe.fused_moe._tuning import TUNING
    from .test_session import _deterministic_timer, session

    _deterministic_timer(monkeypatch)
    old_contract = replace(
        TUNING,
        candidate_contract_version=4,
        knobs=(
            replace(TUNING.knobs[0], values=("micro", "dynamic", "w4a16")),
            *TUNING.knobs[1:],
        ),
    )

    def request(contract):
        plan = Plan(
            contract=contract,
            query=_nvfp4_auto_query(),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(
                config=selection.config
            ),
        )

        def call(state):
            latency = a16_latency if state.config.backend == "w4a16" else a4_latency
            # The shared timer measures abs(output - 6) + 1 microseconds.
            return PreparedCall(run=lambda: latency + 5, produce=lambda: None)

        return plan.request(
            name="moe-precision",
            prepare_call=call,
            benchmark_call=call,
        )

    assert TUNING.candidate_contract_version > old_contract.candidate_contract_version
    for contract, source, expected_backend in (
        (old_contract, "tuned", old_backend),
        (TUNING, "tuned", backend),
        (TUNING, "cached", backend),
    ):
        with session(tmp_path, race_batch=2) as engine:
            engine.device = DetectedDevice(ordinal=None, identity=DEVICE)
            result = engine.prepare((request(contract),))
            selection = result.selections["moe-precision"]
            assert (selection.source, selection.config.backend) == (
                source,
                expected_backend,
            )


def test_nvfp4_split_materialization_is_a_real_candidate_and_default():
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_query()
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.nvfp4_materialize_intermediate for config in configs} == {
        False,
        True,
    }
    for config in configs:
        if config.nvfp4_materialize_intermediate:
            assert config.nvfp4_share_input and config.dynamic_tile_m == 128
    default = TUNING.configure(query, device=DEVICE).default
    assert default.nvfp4_share_input and default.nvfp4_materialize_intermediate
    assert default in configs


@pytest.mark.parametrize(
    "changes",
    (
        {"shared_input_scales": False},
        {"deterministic_output": True},
        {"activation": "relu2"},
        {"intermediate_size": 320},
        {"controls": FrozenMapping({"dynamic_down_scale": True})},
        {"controls": FrozenMapping({"dynamic_swap_ab": "1"})},
        {"controls": FrozenMapping({"dynamic_work_source": "ready_queue"})},
    ),
)
def test_nvfp4_split_materialization_rejects_unsupported_contracts(changes):
    from b12x.moe.fused_moe._tuning import TUNING

    query = _nvfp4_query()
    split = next(
        config
        for _, config in TUNING.eligible_plan(query, DEVICE).candidates
        if config.nvfp4_materialize_intermediate
    )
    invalid = replace(query, **changes)
    with pytest.raises(ValueError):
        TUNING.configure(invalid, device=DEVICE, override=split)
    assert all(
        not config.nvfp4_materialize_intermediate
        for _, config in TUNING.eligible_plan(invalid, DEVICE).candidates
    )


def test_compact_w4a8_has_only_its_supported_dynamic_route():
    from b12x.moe.fused_moe._tuning import TUNING

    query = replace(
        _nvfp4_query(),
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32",
        intermediate_size=576,
    )
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert configs
    assert {
        (
            config.backend,
            config.dynamic_tile_m,
            config.dynamic_route_mode,
            config.route_planner,
        )
        for config in configs
    } == {("dynamic", 16, "grouped", "internal")}


@pytest.mark.parametrize("heads", (1, 12, 16, 20, 32))
@pytest.mark.parametrize("mode", ("decode", "extend"))
def test_v41_precision_candidates_match_native_head_group_contract(heads, mode):
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._tuning import TUNING

    q = torch.empty((3, heads, 512), dtype=torch.bfloat16)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    plan = mla.plan(
        mla.Caps(
            device="cpu",
            num_q_heads=heads,
            max_q_rows=3,
            max_width=128,
            swa_width=128,
            indexed_width=0,
            cache_format="deepseek_v41",
            mode=mode,
        ),
        invocation=mla.invocation_from_tensors(
            q=q, swa_k_cache=cache, out=torch.empty_like(q)
        ),
    )
    configs = [
        config for _, config in TUNING.eligible_plan(plan.query, DEVICE).candidates
    ]
    expected = (
        {("bf16", 16), ("fp8", 16)}
        if mode == "extend"
        else {
            ("bf16", 16),
            ("fp8", 8),
        }
    )
    if mode == "decode" and heads % 16 == 0:
        expected.add(("fp8", 16))
    assert {
        (config.v41_compute_mode, config.v41_heads_per_block) for config in configs
    } == expected
    default = TUNING.configure(plan.query, device=DEVICE).default
    assert default in configs and default.v41_compute_mode == "fp8"
    assert default.v41_heads_per_block == (8 if mode == "decode" and heads % 16 else 16)
    for config in configs:
        assert (
            TUNING.configure(plan.query, device=DEVICE, override=config).default
            == config
        )
    invalid = replace(default, v41_compute_mode="bf16", v41_heads_per_block=8)
    with pytest.raises(ValueError, match="head grouping"):
        TUNING.configure(plan.query, device=DEVICE, override=invalid)
    if mode == "decode" and heads % 16:
        with pytest.raises(ValueError, match="complete 16-head groups"):
            TUNING.configure(
                plan.query,
                device=DEVICE,
                override=replace(default, v41_heads_per_block=16),
            )


@pytest.mark.parametrize("mode", ("decode", "extend"))
def test_compressed_mla_rejects_unaligned_q_before_preparation(mode):
    import torch
    from b12x.attention import compressed_sparse_mla as mla

    storage = torch.empty(3 * 16 * 512 + 1, dtype=torch.bfloat16)
    q = storage[1:].view(3, 16, 512)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Q requires 16-byte alignment"):
        mla.plan(
            mla.Caps(
                device="cpu",
                num_q_heads=16,
                max_q_rows=3,
                max_width=128,
                swa_width=128,
                indexed_width=0,
                cache_format="deepseek_v41",
                mode=mode,
            ),
            invocation=mla.invocation_from_tensors(
                q=q, swa_k_cache=cache, out=torch.empty_like(q)
            ),
        )


def test_mxfp8_a16_races_compute_warps_without_merging_programs():
    from b12x.gemm.blockscaled._tuning import (
        BlockscaledConfig,
        BlockscaledQuery,
        TUNING,
    )

    query = BlockscaledQuery(
        recipe="mxfp8",
        num_tokens=4,
        in_features=2560,
        padded_in_features=2560,
        out_features=512,
        activation_mode="auto",
    )
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    a16 = [config for config in configs if config.mode == "a16"]
    assert {config.mma_warps for config in a16} == {2, 4}
    for config in a16:
        assert replace(config, mma_warps=2) in a16
        assert replace(config, mma_warps=4) in a16
        assert BlockscaledConfig.from_config(FrozenMapping(config.to_dict())) == config
    assert all(
        config.mma_warps is None for config in configs if config.mode == "quantized"
    )
    for recipe in ("nvfp4", "mxfp4"):
        restricted = replace(
            query,
            recipe=recipe,
            global_scale_kind="multiplier" if recipe == "nvfp4" else "none",
        )
        assert all(
            config.mma_warps is None
            for _, config in TUNING.eligible_plan(restricted, DEVICE).candidates
        )


def test_nvfp4_m16_races_compute_warps_with_the_same_route_choices():
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig, TUNING

    query = replace(_nvfp4_query(), num_tokens=4, routed_rows=16)
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    m16 = [
        config
        for config in configs
        if config.backend == "dynamic" and config.dynamic_tile_m == 16
    ]
    assert {config.dynamic_mma_warps for config in m16} == {None, 4}
    for config in m16:
        assert replace(config, dynamic_mma_warps=None) in m16
        assert replace(config, dynamic_mma_warps=4) in m16
        assert MoeDecodeConfig.from_config(FrozenMapping(config.to_dict())) == config
    assert all(
        config.dynamic_mma_warps is None for config in configs if config not in m16
    )


@pytest.mark.parametrize("tile", [16, 32, 64, 128])
@pytest.mark.parametrize("warps", [None, 4])
@pytest.mark.parametrize("clusters", [-1, 0, 1, 126, 188])
def test_dynamic_launch_policy_preserves_compute_warps(tile, warps, clusters):
    from b12x.moe.fused_moe._impl import (
        _encode_dynamic_launch_policy,
        _decode_dynamic_launch_policy,
    )

    for flags in range(32):
        volatile, external, direct, n64, materialize = (
            bool(flags & (1 << bit)) for bit in range(5)
        )
        policy = _encode_dynamic_launch_policy(
            volatile_launch_state=volatile,
            external_route_plan_requested=external,
            planned_tile_m=tile,
            planned_direct_routing=direct,
            w4a8_n64_repacked=n64,
            nvfp4_materialize_intermediate=materialize,
            mma_warps=warps,
            policy_max_active_clusters=clusters,
        )
        assert _decode_dynamic_launch_policy(policy) == (
            volatile,
            external,
            tile,
            direct,
            n64,
            materialize,
            warps,
            clusters,
        )
