"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Regression coverage for the pre-FC2 routing-weighting trace reference."""

import pytest
import torch

from flashinfer.tllm_enums import ActivationType
from flashinfer.trace.templates.moe import (
    cute_dsl_fused_moe_trace,
    cute_dsl_moe_wrapper_run_trace,
)


@pytest.mark.parametrize(
    "template",
    [cute_dsl_fused_moe_trace, cute_dsl_moe_wrapper_run_trace],
    ids=["functional", "wrapper"],
)
@pytest.mark.parametrize("hidden", [128, 768, 2048])
@pytest.mark.parametrize("scale_rank", [2, 3])
def test_pre_fc2_reference_empty_batch(template, hidden, scale_rank):
    scale_shape = (0, hidden // 16) + ((1,) if scale_rank == 3 else ())
    output = template.reference(
        x=torch.empty((0, hidden // 2), dtype=torch.uint8),
        x_sf=torch.empty(scale_shape, dtype=torch.uint8),
        token_selected_experts=torch.empty((0, 1), dtype=torch.int32),
        token_final_scales=torch.empty((0, 1)),
        w1_weight=torch.empty((1, 128, hidden // 2), dtype=torch.uint8),
        w1_weight_sf=torch.empty((1, 128, hidden // 16), dtype=torch.uint8),
        w1_alpha=torch.ones(1),
        fc2_input_scale=torch.ones(1),
        w2_weight=torch.empty((1, hidden, 64), dtype=torch.uint8),
        w2_weight_sf=torch.empty((1, hidden, 8), dtype=torch.uint8),
        w2_alpha=torch.ones(1),
        num_experts=1,
        top_k=1,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([16.0]),
        routing_weights_before_fc2=True,
    )
    assert output.shape == (0, hidden)
    assert output.dtype == torch.bfloat16
    assert output.device == torch.device("cpu")


def _standalone_inputs(scale_dtype=torch.uint8):
    """Build an exactly representable single-expert NVFP4 trace fixture."""
    scales = torch.full((1, 8), 56, dtype=torch.uint8).view(scale_dtype)
    weight_scales = torch.full((32, 4, 1, 4, 2, 1), 56, dtype=torch.uint8).view(
        scale_dtype
    )
    inputs = dict(
        x=torch.full((1, 64), 0x22, dtype=torch.uint8),
        x_sf=scales,
        token_selected_experts=torch.zeros((1, 1), dtype=torch.int32),
        token_final_scales=torch.full((1, 1), 0.5),
        w1_weight=torch.full((1, 128, 64), 0x22, dtype=torch.uint8),
        w1_weight_sf=weight_scales,
        w1_alpha=torch.tensor([1 / 128]),
        fc2_input_scale=torch.tensor([24.0]),
        w2_weight=torch.full((1, 128, 64), 0x22, dtype=torch.uint8),
        w2_weight_sf=weight_scales,
        w2_alpha=torch.tensor([1 / 24]),
        num_experts=1,
        top_k=1,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([16.0]),
        routing_weights_before_fc2=True,
    )
    return inputs


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e4m3fn])
def test_pre_fc2_exported_reference_runs_standalone(scale_dtype):
    from flashinfer import cute_dsl_fused_moe

    inputs = _standalone_inputs(scale_dtype)
    definition = cute_dsl_fused_moe.fi_trace(**inputs)
    namespace = {}
    exec(compile(definition["reference"], "<exported-reference>", "exec"), namespace)
    actual = namespace["_cute_dsl_fused_moe_reference"](**inputs)
    torch.testing.assert_close(
        actual, torch.full((1, 128), 64.0, dtype=torch.bfloat16), rtol=0, atol=0
    )


def test_pre_fc2_reference_rounds_fp4_halfway_values_to_even():
    from flashinfer.trace.templates.moe import _pre_fc2_nvfp4_qdq

    hidden = torch.tensor([[3.5, 6.0] + [0.0] * 14], dtype=torch.bfloat16)
    actual = _pre_fc2_nvfp4_qdq(hidden, torch.ones(1))
    torch.testing.assert_close(actual[0, :2], torch.tensor([4.0, 6.0]), rtol=0, atol=0)


@pytest.mark.parametrize("sign", [1, -1])
def test_pre_fc2_reference_saturates_extreme_normalized_values(sign):
    """Overflow-scale distances must still select the largest FP4 magnitude."""
    from flashinfer.trace.templates.moe import _pre_fc2_nvfp4_qdq

    hidden = torch.full((1, 16), sign * 128.0, dtype=torch.bfloat16)
    actual = _pre_fc2_nvfp4_qdq(hidden, torch.tensor([2.0**30]))
    torch.testing.assert_close(
        actual, torch.full((1, 16), sign * 2688.0), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "template", [cute_dsl_fused_moe_trace, cute_dsl_moe_wrapper_run_trace]
)
@pytest.mark.parametrize("scale_count", [1, 3])
def test_trace_scale_axis_is_independent_of_singleton_clamp(template, scale_count):
    """A local expert scale vector must not overwrite the clamp's one axis."""
    inputs = _standalone_inputs()
    for name in ("w1_weight", "w2_weight"):
        inputs[name] = inputs[name].repeat(3, 1, 1)
    for name in ("w1_weight_sf", "w2_weight_sf"):
        inputs[name] = inputs[name].repeat(1, 1, 1, 1, 1, 3)
    for name in ("w1_alpha", "w2_alpha"):
        inputs[name] = inputs[name].repeat(3)
    inputs["num_experts"] = 3
    inputs["fc2_input_scale"] = torch.ones(scale_count)
    definition = template.build_fi_trace_fn("test.moe")(**inputs)
    scale_axis = definition["inputs"]["fc2_input_scale"]["shape"][0]
    clamp_axis = definition["inputs"]["clamped_relu2_limit"]["shape"][0]
    assert scale_axis != clamp_axis
    extractors = template._build_axis_extractors()
    assert extractors[scale_axis](inputs) == scale_count
    assert extractors[clamp_axis](inputs) == 1


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("container", ["tensor", "list", "dict"])
def test_exported_check_handles_cancellation_without_hiding_errors(fused, container):
    """Fused BF16 reduction gets a contribution-derived bound; FP32 reduction does not."""
    import itertools

    inputs = _standalone_inputs()
    for name in ("w1_weight", "w2_weight"):
        inputs[name] = inputs[name].repeat(3, 1, 1)
    inputs["w2_weight"][2].fill_(0xAA)
    for name in ("w1_weight_sf", "w2_weight_sf"):
        inputs[name] = inputs[name].repeat(1, 1, 1, 1, 1, 3)
    inputs["w1_alpha"] = inputs["w1_alpha"].repeat(3)
    inputs["w2_alpha"] = torch.tensor([256.0, 1.0, 256.0]) / (128 * 24)
    inputs["token_selected_experts"] = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    inputs["token_final_scales"] = torch.ones((1, 3))
    inputs.update(num_experts=3, top_k=3, use_fused_finalize=fused)
    definition = cute_dsl_fused_moe_trace.build_fi_trace_fn("test.moe")(**inputs)
    namespace = {}
    exec(definition["reference"], namespace)
    exec(definition["check"], namespace)
    reference = namespace["_cute_dsl_fused_moe_reference"](**inputs)
    check = namespace["_cute_dsl_moe_check"]
    assert (reference == 1).all()

    def wrap(t):
        return (
            [t] if container == "list" else ({"out": t} if container == "dict" else t)
        )

    for values in itertools.permutations([256.0, 1.0, -256.0]):
        actual = torch.zeros_like(reference)
        for v in values:
            actual += v
        expected_valid = fused or bool((actual == reference).all())
        assert check(wrap(reference), wrap(actual), rtol=0, atol=0) == expected_valid
    assert not check(wrap(reference), wrap(reference + 32), rtol=0, atol=0)


def test_exported_situ_reference_preserves_validation_and_numerics():
    """The emitted shared reference must validate SiTU without package imports."""
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        validate_cute_dsl_moe_situ_config,
    )

    inputs = _standalone_inputs()
    definition = cute_dsl_fused_moe_trace.build_fi_trace_fn("test.moe")(**inputs)
    namespace = {}
    exec(definition["reference"], namespace)
    reference = namespace["_moe_bf16_run_experts"]
    base = dict(
        hidden_states=torch.ones((1, 1)),
        gemm1_weights=torch.ones((1, 2, 1)),
        gemm2_weights=torch.ones((1, 1, 1)),
        weights=torch.ones((1, 1)),
        topk_idx=torch.zeros((1, 1), dtype=torch.long),
        local_expert_offset=0,
        E_global=1,
    )
    actual = reference(**base, situ_beta=1.0, situ_linear_beta=2.0)
    expected = (
        torch.tanh(torch.tensor(1.0))
        * torch.sigmoid(torch.tensor(1.0))
        * (2 * torch.tanh(torch.tensor(0.5)))
    )
    torch.testing.assert_close(
        actual, expected.reshape(1, 1).bfloat16(), rtol=0, atol=0
    )
    cases = [(3, None, 1.0), (6, 1.0, None)]
    for value in (0.0, -1.0, float("inf"), float("nan"), 1e100, 1e-100):
        cases.extend([(3, value, None), (3, 1.0, value)])
    for activation, beta, linear in cases:
        with pytest.raises(ValueError) as runtime_error:
            validate_cute_dsl_moe_situ_config(activation, beta, linear)
        with pytest.raises(ValueError) as exported_error:
            reference(
                **base,
                activation_type=activation,
                situ_beta=beta,
                situ_linear_beta=linear,
            )
        assert str(exported_error.value) == str(runtime_error.value)


@pytest.mark.parametrize("slot_order,expected", [((0, 1, 2), 0.0), ((0, 2, 1), 1.0)])
def test_nonfused_reference_matches_fp32_slot_order(slot_order, expected):
    """Both accumulator precision and routing-slot order affect large cancellation."""
    inputs = _standalone_inputs()
    for name in ("w1_weight", "w2_weight"):
        inputs[name] = inputs[name].repeat(3, 1, 1)
    inputs["w2_weight"][2].fill_(0xAA)
    for name in ("w1_weight_sf", "w2_weight_sf"):
        inputs[name] = inputs[name].repeat(1, 1, 1, 1, 1, 3)
    inputs["w1_alpha"] = inputs["w1_alpha"].repeat(3)
    inputs["w2_alpha"] = torch.tensor([33554432.0, 1.0, 33554432.0]) / (128 * 24)
    inputs["token_selected_experts"] = torch.tensor([slot_order], dtype=torch.int32)
    inputs["token_final_scales"] = torch.ones((1, 3))
    inputs.update(num_experts=3, top_k=3, use_fused_finalize=False)
    definition = cute_dsl_fused_moe_trace.build_fi_trace_fn("test.moe")(**inputs)
    namespace = {}
    exec(definition["reference"], namespace)
    actual = namespace["_cute_dsl_fused_moe_reference"](**inputs)
    torch.testing.assert_close(
        actual, torch.full_like(actual, expected), rtol=0, atol=0
    )


@pytest.mark.parametrize("dictionary_side", ["reference", "actual"])
@pytest.mark.parametrize("other_container", ["tensor", "list"])
def test_exported_checker_rejects_one_sided_dictionary(
    dictionary_side, other_container
):
    """Reduction metadata must not weaken the standard output-container contract."""
    inputs = _standalone_inputs()
    definition = cute_dsl_fused_moe_trace.build_fi_trace_fn("test.moe")(**inputs)
    namespace = {}
    exec(definition["reference"], namespace)
    exec(definition["check"], namespace)
    reference = namespace["_cute_dsl_fused_moe_reference"](**inputs)
    actual = reference.clone()
    assert hasattr(reference, "_cute_dsl_bf16_reduction_bound")
    if dictionary_side == "reference":
        reference = {"out": reference}
        actual = [actual] if other_container == "list" else actual
    else:
        actual = {"out": actual}
        reference = [reference] if other_container == "list" else reference
    assert not namespace["_cute_dsl_moe_check"](reference, actual)
