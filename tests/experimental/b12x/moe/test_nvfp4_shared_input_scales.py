"""Preparation-time equality proofs retain direct expert-scale storage."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import _impl as impl
from b12x.moe.fused_moe._impl import B12XFP4ExpertWeights


def _owner(scales: torch.Tensor, *, immutable=True) -> B12XFP4ExpertWeights:
    experts = 4
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=experts, hidden_size=128, intermediate_size=128),
    )
    return B12XFP4ExpertWeights(
        plan=plan._impl,
        a1_gscale=scales,
        a2_gscale=torch.ones(experts),
        w1_fp4=torch.empty(experts, 256, 64, dtype=torch.uint8),
        w1_blockscale=torch.empty(experts, 256, 8, dtype=torch.uint8),
        w1_alphas=torch.ones(experts),
        w2_fp4=torch.empty(experts, 128, 64, dtype=torch.uint8),
        w2_blockscale=torch.empty(experts, 128, 8, dtype=torch.uint8),
        w2_alphas=torch.ones(experts),
        immutable_input_scales=immutable,
    )


def test_uniform_vector_preserves_storage_and_requires_static_contract():
    scales = torch.full((4,), 731.488)
    owner = _owner(scales)
    assert owner.a1_gscale is scales
    assert owner.a1_gscale.numel() == 4
    assert owner.can_share_input(input_scales_static=True)
    assert not owner.can_share_input(input_scales_static=False)


def test_mutable_prepared_vectors_do_not_share_even_with_static_bindings():
    owner = _owner(torch.ones(4), immutable=False)
    assert not owner.can_share_input(input_scales_static=True)
    owner.a1_gscale[-1] = 2.0
    assert not owner.can_share_input(input_scales_static=True)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0, 1.0, 1.001],
        [float("nan")] * 4,
        [float("inf")] * 4,
        [0.0] * 4,
        [-1.0] * 4,
    ],
)
def test_nonuniform_or_invalid_scales_do_not_share(values):
    owner = _owner(torch.tensor(values))
    assert not owner.can_share_input(input_scales_static=True)


def test_one_ulp_difference_does_not_share():
    scales = torch.ones(4)
    scales[-1] = torch.nextafter(scales[-1], torch.tensor(float("inf")))
    assert not _owner(scales).can_share_input(input_scales_static=True)


def test_scale_mutation_invalidates_proof_until_weights_are_reprepared():
    owner = _owner(torch.ones(4))
    owner.a1_gscale[-1] = 2.0
    assert not owner.can_share_input(input_scales_static=True)
    assert not replace(owner).can_share_input(input_scales_static=True)
    owner.a1_gscale.fill_(2.0)
    assert not owner.can_share_input(input_scales_static=True)
    assert replace(owner).can_share_input(input_scales_static=True)


def test_inference_tensor_requires_static_contract():
    with torch.inference_mode():
        scales = torch.ones(4)
        owner = _owner(scales)
        assert torch.is_inference(owner.a1_gscale)
        assert owner.can_share_input(input_scales_static=True)
        assert not owner.can_share_input(input_scales_static=False)


def test_binding_decision_does_not_read_scale_values(monkeypatch):
    owner = _owner(torch.ones(4))

    def fail(*args, **kwargs):
        raise AssertionError("Preparation-time scale values read during binding")

    monkeypatch.setattr(torch.Tensor, "item", fail)
    monkeypatch.setattr(torch, "isfinite", fail)
    assert owner.can_share_input(input_scales_static=True)


def test_scalar_preserves_existing_shared_input_contract():
    owner = _owner(torch.tensor(2.0))
    assert owner.can_share_input(input_scales_static=False)


def test_dynamic_launch_schema_marks_weights_and_scales_read_only():
    schema = torch.ops.b12x.tp_moe_dynamic_launch.default._schema
    arguments = {argument.name: argument for argument in schema.arguments}
    for name in (
        "input_gs",
        "down_input_scale",
        "w1_alpha",
        "w2_alpha",
        "w13_fp4",
        "w13_sf",
        "down_fp4",
        "down_sf",
        "a",
        "flat_ids",
        "flat_weights",
    ):
        alias = arguments[name].alias_info
        assert alias is None or not alias.is_write, name
    for name in ("scatter_output", "materialized_intermediate", "packed_a_flat"):
        assert arguments[name].alias_info.is_write, name


@pytest.mark.parametrize(
    "operation, launch",
    [
        ("tp_moe_dynamic_launch", "_launch_dynamic_flat"),
        ("tp_moe_compact_micro_launch", "_launch_compact_micro_flat"),
    ],
)
def test_launch_dispatch_preserves_canonical_scale_version(
    monkeypatch, operation, launch
):
    """Exercise PyTorch mutation accounting without needing a CUDA device."""
    owner = _owner(torch.ones(4))
    op = getattr(torch.ops.b12x, operation).default
    kwargs = {}
    primitives = {"int": 0, "bool": False, "float": 1.0, "str": "nvfp4"}
    for argument in op._schema.arguments:
        kind = str(argument.type)
        kwargs[argument.name] = (
            torch.zeros(4) if kind == "Tensor" else primitives.get(kind)
        )
    kwargs["input_gs"] = owner.a1_gscale

    def write_output(**operands):
        operands["scatter_output"].fill_(1)

    monkeypatch.setattr(impl, launch, write_output)
    op(**kwargs)
    assert torch.equal(kwargs["scatter_output"], torch.ones(4))
    assert owner.a1_gscale._version == owner._a1_scale_version
    assert owner.can_share_input(input_scales_static=True)
