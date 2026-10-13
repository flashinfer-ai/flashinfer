"""NVFP4 materialization follows the captured declaration and explicit pins."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.moe.fused_moe._tuning import TUNING
from b12x.preparation import FrozenMapping
from tests.experimental.b12x.preparation.test_precision_choices import (
    DEVICE,
    _nvfp4_query,
)


@pytest.mark.parametrize("tile", (16, 64))
def test_split_rejects_other_source_tiles(tile):
    query = _nvfp4_query()
    split = next(
        config
        for _, config in TUNING.eligible_plan(query, DEVICE).candidates
        if config.nvfp4_materialize_intermediate
    )
    with pytest.raises(ValueError, match="M128 contract"):
        TUNING.configure(
            query, device=DEVICE, override=replace(split, dynamic_tile_m=tile)
        )


@pytest.mark.parametrize("enabled", (False, True))
def test_captured_materialization_default_and_explicit_pin(enabled):
    query = replace(
        _nvfp4_query(), controls=FrozenMapping({"dynamic_nvfp4_materialized": enabled})
    )
    default = TUNING.configure(query, device=DEVICE).default
    assert default.nvfp4_materialize_intermediate == enabled
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert {config.nvfp4_materialize_intermediate for config in configs} == {
        False,
        True,
    }
    opposite = replace(default, nvfp4_materialize_intermediate=not enabled)
    assert TUNING.configure(query, device=DEVICE, override=opposite).default == opposite


@pytest.mark.parametrize("environment", ("0", "1"))
@pytest.mark.parametrize("requested", (None, False, True))
def test_preparation_resolves_determinism_before_selecting_split(
    monkeypatch,
    environment,
    requested,
):
    """The split's atomic scatter cannot serve a deterministic output request."""
    import torch
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._preparation import _query

    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", environment)
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        geometry=fused_moe.MoEGeometry(
            num_experts=32,
            hidden_size=512,
            intermediate_size=512,
        ),
        activation=fused_moe.ActivationSpec(
            mode="a4",
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
    )
    # Query construction only reads immutable weight and device metadata.
    experts = SimpleNamespace(
        plan=weight_plan,
        num_experts=32,
        hidden_size=512,
        intermediate_size=512,
        device=torch.device("cuda:0"),
        _impl=SimpleNamespace(can_share_input=lambda **kwargs: True),
    )
    query = _query(
        experts,
        fused_moe.ExecutionCapacity(max_tokens=1024, top_k=4),
        1024,
        fused_moe.RoutingSpec(deterministic_output=requested),
        FrozenMapping(),
        FrozenMapping(),
    )
    expected = environment == "1" if requested is None else requested
    assert query.deterministic_output is expected
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert configs
    assert (
        any(config.nvfp4_materialize_intermediate for config in configs) is not expected
    )
    # Subsequent environment changes cannot alter the declared candidate set.
    monkeypatch.setenv("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "0" if expected else "1")
    assert [
        config for _, config in TUNING.eligible_plan(query, DEVICE).candidates
    ] == configs
