"""Per-query validity and defaults remain independent of the search subset."""
from dataclasses import dataclass, replace

import pytest

from b12x.preparation import FrozenMapping, Knob, ParameterSpace, TuningContract


@dataclass(frozen=True)
class Query:
    rows: int


@dataclass(frozen=True)
class Config:
    width: int


def contract(*, default=7, values=(1, 2, 4)):
    def validate_query(query, device):
        if not isinstance(query, Query) or query.rows <= 0:
            raise ValueError("invalid row geometry")

    def validate_config(query, config, device):
        if not isinstance(config, Config) or config.width <= 0:
            raise ValueError("invalid width")

    return TuningContract(
        component_id="test.arithmetic", query_schema_version=1, config_schema_version=1,
        query_fields=frozenset({"rows"}), config_fields=frozenset({"width"}),
        encode_query=lambda query: {"rows": query.rows},
        encode_config=lambda config: {"width": config.width},
        decode_config=lambda payload: Config(payload["width"]),
        validate_query=validate_query, validate_config=validate_config,
        default_config=lambda query, device: Config(default),
        knobs=(Knob(name="width", values=values),),
    )


def test_pin_does_not_evaluate_unused_invalid_default():
    def unavailable(query, device):
        raise RuntimeError("default unavailable")

    tuning = replace(contract(), default_config=unavailable)
    configured = tuning.configure(Query(3), device=None, override=Config(9))
    assert configured.default.width * configured.query.rows == 27
    with pytest.raises(ValueError):
        tuning.configure(Query(3), device=None, override=Config(-1))


def test_default_need_not_be_in_pruned_candidate_subset():
    tuning = contract()
    configured = tuning.configure(Query(3), device=None)
    assert configured.default.width == 7
    assert [config.width for _, config in tuning.iterate(configured)] == [1, 2, 4]


def test_invalid_query_fails_before_parameter_generation():
    def forbidden(query, device):
        raise AssertionError("invalid query reached parameter generation")

    tuning = replace(contract(), parameters=forbidden)
    with pytest.raises(ValueError, match="row geometry"):
        tuning.configure(Query(0), device=None)


def test_singleton_is_determined_after_predicates_and_equivalence():
    tuning = contract()
    tuning = replace(
        tuning,
        parameters=lambda query, device: ParameterSpace(
            knobs=tuning.knobs, predicates=(lambda assignment: assignment["width"] != 4,),
        ),
        equivalence_key=lambda query, device, config: {"group": 0},
    )
    choices = tuning.eligible_plan(Query(3), None)
    assert [config.width for _, config in choices.candidates] == [1]
    assert choices.cartesian_count == 3
    assert choices.legal_count == 2


def test_metadata_requires_explicit_nonfinite_codec():
    with pytest.raises(ValueError):
        FrozenMapping({"limit": float("inf")})
    with pytest.raises(ValueError):
        FrozenMapping({"limit": float("nan")})
    assert FrozenMapping({"limit": "+inf"}).to_dict() == {"limit": "+inf"}


def test_packed_forced_a16_fallback_and_quantized_pin_remain_distinct():
    from b12x.preparation import DeviceIdentity
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, BlockscaledConfig, TUNING
    device = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=16, in_features=256, padded_in_features=256,
        out_features=128, activation_mode="a16",
    )
    assert TUNING.configure(query, device=device).default == BlockscaledConfig(
        mode="a16", tile_n=64, tile_k=64, split_k=1,
    )
    padded = replace(query, num_tokens=1, in_features=192)
    assert TUNING.configure(padded, device=device).default.mode == "a16"
    forced_q = replace(query, num_tokens=1, activation_mode="quantized", activation_scale_available=True)
    assert TUNING.configure(forced_q, device=device).default == BlockscaledConfig(mode="quantized")
    with pytest.raises(ValueError, match="activation scale"):
        TUNING.configure(replace(forced_q, activation_scale_available=False), device=device)
    # A complete A16 pin is valid even when AUTO's unused M16 default would
    # require an absent activation scale.
    auto = replace(query, activation_mode="auto")
    pinned = BlockscaledConfig(mode="a16", tile_n=128, tile_k=64, split_k=4)
    assert TUNING.configure(auto, device=device, override=pinned).default == pinned


def test_packed_search_equivalence_uses_logical_k_split_clamping():
    from b12x.preparation import DeviceIdentity
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery, TUNING, effective_a16_config
    query = BlockscaledQuery(
        recipe="nvfp4", num_tokens=1, in_features=192, padded_in_features=256,
        out_features=128, activation_scale_available=True,
    )
    device = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
    eligible = TUNING.eligible_plan(query, device)
    effective = [
        effective_a16_config(query, config) for _, config in eligible.candidates if config.mode == "a16"
    ]
    assert (64, 64, 3) in effective
    assert (64, 256, 1) in effective
    assert (128, 256, 1) in effective
    assert len(effective) == len(set(effective)) == 12
    assert len(eligible.candidates) == 13


@pytest.mark.parametrize("workspace_form", ["owned", "provided"])
def test_packed_search_excludes_candidates_over_workspace_capacity(workspace_form):
    from b12x.preparation import DeviceIdentity
    from b12x.gemm.blockscaled._tuning import (
        BlockscaledQuery,
        TUNING,
        effective_a16_config,
    )

    query = BlockscaledQuery(
        recipe="nvfp4",
        num_tokens=65_536,
        in_features=4_304,
        padded_in_features=4_320,
        out_features=3_456,
        activation_mode="a16",
        workspace_form=workspace_form,
        workspace_nbytes=2_000_000_000,
    )
    device = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")

    eligible = TUNING.eligible_plan(query, device)
    splits = {
        effective_a16_config(query, config)[2]
        for _, config in eligible.candidates
    }

    assert splits == {1, 2}
    assert max(
        split * query.num_tokens * query.out_features * 4
        for split in splits
        if split > 1
    ) == 1_811_939_328
