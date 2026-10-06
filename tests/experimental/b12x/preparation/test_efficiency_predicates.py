"""Exhaustive search lifts efficiency rules while retaining kernel legality."""
from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping, Knob, ParameterSpace


def test_exhaustive_lifts_only_efficiency_predicates():
    space = ParameterSpace.create(
        (Knob(name="tile", values=(8, 16, 32)),),
        predicates=(lambda p: p["tile"] >= 16,),
        efficiency_predicates=(lambda p: p["tile"] <= 16,),
    )
    exhaustive = replace(space, exhaustive=True)
    assert [p["tile"] for p in space.configurations()] == [16]
    assert [p["tile"] for p in exhaustive.configurations()] == [16, 32]
    with pytest.raises(ValueError, match="efficiency predicates"):
        space.validate({"tile": 32})
    exhaustive.validate({"tile": 32})
    for active in (space, exhaustive):
        with pytest.raises(ValueError, match="correctness predicates"):
            active.validate({"tile": 8})
        with pytest.raises(ValueError, match="eligible values"):
            active.validate({"tile": 64})


def test_efficiency_errors_are_not_silent_rejections():
    def broken(p):
        raise RuntimeError("invalid planner state")

    space = ParameterSpace(knobs=(Knob(name="tile", values=(16,)),),
                           efficiency_predicates=(broken,))
    with pytest.raises(RuntimeError, match="invalid planner state"):
        list(space.configurations())
    assert len(list(replace(space, exhaustive=True).configurations())) == 1
    with pytest.raises(TypeError, match="boolean"):
        replace(space, exhaustive="0")


def _query():
    from b12x.norm.mhc._tuning import MhcQuery
    return MhcQuery(dtype="bfloat16", max_tokens=64, hidden_size=5120,
                    split_k=80, operation="post_pre", has_norm_weight=True,
                    lagged_mix=True, norm_eps=1e-20, rms_eps=1e-20,
                    smem_limit=100 << 10)


def _choice(**changes):
    return dict(backend="tf32_tma", lagged_prepare=False, partials_per_cta=4,
                projection_tile_n=24, projection_tile_k=64,
                projection_num_stages=2, projection_num_m_warps=2,
                projection_num_n_warps=1, projection_k_splits=20) | changes


@pytest.mark.parametrize("changes", [
    {"projection_num_m_warps": 3},
    {"projection_tile_n": 64},
    {"projection_tile_n": 32, "projection_num_n_warps": 4},
    {"projection_tile_k": 256, "projection_k_splits": 40,
     "projection_num_stages": 4, "projection_num_m_warps": 1,
     "projection_tile_n": 8},
    {"projection_k_splits": 1},
])
def test_mhc_efficiency_rules_and_explicit_pin(changes):
    from b12x.norm.mhc._tuning import MhcConfig, TUNING
    device = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")
    query = _query()
    exhaustive = replace(query, controls=FrozenMapping({"B12X_AUTOTUNE_EXHAUSTIVE": "1"}))
    choice = _choice(**changes)
    with pytest.raises(ValueError, match="efficiency predicates"):
        TUNING.parameter_space(query, device).validate(choice)
    TUNING.parameter_space(exhaustive, device).validate(choice)
    config = MhcConfig(projection_tile_m=16 * choice["projection_num_m_warps"], **choice)
    assert TUNING.configure(query, device=device, override=config).pinned == config
    assert TUNING.lower(exhaustive, device, choice) == config


@pytest.mark.parametrize("changes", [
    {"projection_tile_k": 16},
    {"projection_tile_n": 24, "projection_num_n_warps": 2},
    {"projection_tile_k": 256, "projection_k_splits": 32},
    {"projection_num_m_warps": 16, "projection_num_n_warps": 3},
    {"projection_num_m_warps": 16, "projection_tile_k": 256,
     "projection_num_stages": 4},
])
def test_mhc_exhaustive_preserves_correctness_rules(changes):
    from b12x.norm.mhc._tuning import TUNING
    query = replace(_query(), controls=FrozenMapping({"B12X_AUTOTUNE_EXHAUSTIVE": "1"}))
    with pytest.raises(ValueError, match="correctness predicates"):
        TUNING.parameter_space(query, None).validate(_choice(**changes))


def test_mhc_grid_floor_scales_with_device():
    from b12x.norm.mhc._tuning import TUNING
    query = replace(_query(), max_tokens=1)
    choice = _choice(projection_num_m_warps=1, projection_k_splits=8)
    TUNING.parameter_space(query, DeviceIdentity("nvidia", (12, 1), 48, "GB10")).validate(choice)
    with pytest.raises(ValueError, match="efficiency predicates"):
        TUNING.parameter_space(query, DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")).validate(choice)


def test_mhc_declaration_captures_exhaustive_policy(monkeypatch):
    from b12x.norm import mhc
    from b12x.norm.mhc._tuning import TUNING
    caps = mhc.Caps(device="cpu", max_tokens=1, hidden_size=5120)
    invocation = FrozenMapping({"operation": "collapse"})
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    pruned = mhc.plan(caps, invocation=invocation)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "1")
    exhaustive = mhc.plan(caps, invocation=invocation)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    assert pruned.query.controls["B12X_AUTOTUNE_EXHAUSTIVE"] == "0"
    assert exhaustive.query.controls["B12X_AUTOTUNE_EXHAUSTIVE"] == "1"
    assert TUNING.encode_query(pruned.query) != TUNING.encode_query(exhaustive.query)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        TUNING.configure(replace(_query(), controls=FrozenMapping({"B12X_AUTOTUNE_EXHAUSTIVE": "yes"})), device=None)


def test_exhaustive_policy_does_not_change_compiled_program_identity(monkeypatch):
    from b12x._lib.compiler import _compile_environment_key
    try:
        monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "0")
        _compile_environment_key.cache_clear()
        pruned = _compile_environment_key()
        monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "1")
        _compile_environment_key.cache_clear()
        assert _compile_environment_key() == pruned
    finally:
        _compile_environment_key.cache_clear()


@pytest.mark.parametrize("capability,sm", [((12, 0), 188), ((12, 1), 48)])
@pytest.mark.parametrize("boundary,changes", [
    (384, {"projection_num_m_warps": 1, "projection_num_n_warps": 3}),
    (384, {"projection_num_m_warps": 3}),
    (2048, {"projection_tile_n": 8}),
    (384, {"projection_tile_k": 32}),
    (384, {"projection_tile_k": 128, "projection_num_stages": 3}),
    (2048, {"projection_k_splits": 16}),
])
def test_mhc_prefill_correlations_preserve_products_and_pins(capability, sm, boundary, changes):
    from b12x.norm.mhc._tuning import MhcConfig, TUNING
    device = DeviceIdentity("nvidia", capability, sm, "Blackwell")
    query = replace(_query(), max_tokens=boundary)
    choice = _choice(**(dict(projection_num_m_warps=4, projection_k_splits=8) | changes))
    before = TUNING.parameter_space(replace(query, max_tokens=boundary - 1), device)
    after = TUNING.parameter_space(query, device)
    if sm == 48 and changes == {"projection_k_splits": 16}:
        with pytest.raises(ValueError, match="efficiency predicates"):
            before.validate(choice)
    else:
        before.validate(choice)
    with pytest.raises(ValueError, match="efficiency predicates"):
        after.validate(choice)
    replace(after, exhaustive=True).validate(choice)
    config = MhcConfig(projection_tile_m=16 * choice["projection_num_m_warps"], **choice)
    assert TUNING.configure(query, device=device, override=config).pinned == config
    assert before.knobs == after.knobs


@pytest.mark.parametrize("capability,sm", [((12, 0), 188), ((12, 1), 48)])
@pytest.mark.parametrize("hidden,m,tm,tn,tk,nw,stages,splits", [
    (5120, 64, 16, 24, 64, 3, 3, 40),
    (5120, 4096, 128, 24, 128, 1, 2, 5),
    (5120, 4096, 128, 32, 64, 1, 2, 5),
    (4096, 384, 64, 24, 64, 1, 3, 8),
    (4096, 1024, 32, 8, 256, 1, 1, 1),
    (7168, 3584, 64, 24, 64, 1, 3, 8),
    (4096, 8192, 128, 24, 64, 1, 2, 4),
])
def test_mhc_prefill_correlations_retain_measured_winner_geometries(
    capability, sm, hidden, m, tm, tn, tk, nw, stages, splits,
):
    from b12x.norm.mhc._tuning import TUNING
    query = replace(_query(), max_tokens=m, hidden_size=hidden, split_k=hidden // 64)
    choice = _choice(
        projection_num_m_warps=tm // 16, projection_tile_n=tn, projection_tile_k=tk,
        projection_num_n_warps=nw, projection_num_stages=stages,
        projection_k_splits=splits,
    )
    TUNING.parameter_space(query, DeviceIdentity("nvidia", capability, sm, "Blackwell")).validate(choice)


def test_mhc_excess_split_grid_scales_with_sm_count_and_preserves_pins():
    from b12x.norm.mhc._tuning import MhcConfig, TUNING

    query = replace(_query(), max_tokens=384)
    choice = _choice(projection_num_m_warps=4, projection_tile_n=8)
    small = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
    large = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")
    TUNING.parameter_space(query, large).validate(choice)
    with pytest.raises(ValueError, match="efficiency predicates"):
        TUNING.parameter_space(query, small).validate(choice)
    replace(TUNING.parameter_space(query, small), exhaustive=True).validate(choice)
    config = MhcConfig(projection_tile_m=64, **choice)
    assert TUNING.configure(query, device=small, override=config).pinned == config
