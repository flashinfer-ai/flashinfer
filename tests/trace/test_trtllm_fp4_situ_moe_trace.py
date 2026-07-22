"""Trace coverage for TRT-LLM Gen MXFP4 x MXFP8 SiTU MoE."""

import pytest
import torch

from flashinfer import ActivationType


def _mxfp4_mxfp8_identity_inputs():
    seq_len = 4
    hidden_size = 32
    intermediate_size = 32

    hidden_bf16 = torch.zeros(seq_len, hidden_size, dtype=torch.bfloat16)
    hidden_bf16[:, 0] = torch.tensor([-3.0, -1.0, 3.0, -4.0])
    hidden_bf16[:, 1] = torch.tensor([0.5, -2.0, 4.0, 6.0])
    hidden_states = hidden_bf16.to(torch.float8_e4m3fn)
    hidden_states_scale = torch.full((seq_len, 1), 127, dtype=torch.uint8)

    gemm1_weights = torch.zeros(
        1, 2 * intermediate_size, hidden_size // 2, dtype=torch.uint8
    )
    gemm1_weights[0, 0, 0] = 0x02
    gemm1_weights[0, intermediate_size, 0] = 0x20
    gemm1_weights_scale = torch.full(
        (1, 2 * intermediate_size, 1), 127, dtype=torch.uint8
    )

    gemm2_weights = torch.zeros(
        1, hidden_size, intermediate_size // 2, dtype=torch.uint8
    )
    gemm2_weights[0, 0, 0] = 0x02
    gemm2_weights_scale = torch.full((1, hidden_size, 1), 127, dtype=torch.uint8)

    return {
        "routing_logits": torch.zeros(seq_len, 1, dtype=torch.float32),
        "routing_bias": None,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm1_bias": None,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "gemm2_bias": None,
        "top_k": 1,
        "local_expert_offset": 0,
        "routed_scaling_factor": 1.0,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }


def _situ_params():
    return {
        "activation_type": ActivationType.Situ.value,
        "gemm1_alpha": torch.tensor([1.7], dtype=torch.float32),
        "gemm1_beta": torch.tensor([1.0], dtype=torch.float32),
        "gemm1_clamp_limit": torch.tensor([7.0], dtype=torch.float32),
    }


def test_fp4_situ_trace_schema_records_activation():
    from flashinfer.trace.templates.moe import (
        trtllm_fp4_block_scale_moe_default_routing_trace,
        trtllm_fp4_block_scale_routed_moe_trace,
    )

    inputs = _mxfp4_mxfp8_identity_inputs()
    params = _situ_params()
    common = {
        **inputs,
        **params,
        "num_experts": 1,
        "local_num_experts": 1,
    }
    default_defn = trtllm_fp4_block_scale_moe_default_routing_trace.build_fi_trace_fn(
        "flashinfer.fused_moe.core.trtllm_fp4_block_scale_moe"
    )(**common)
    routed_inputs = dict(common)
    routed_inputs.pop("routing_logits")
    routed_inputs["topk_ids"] = torch.zeros(4, 1, dtype=torch.int32)
    routed_defn = trtllm_fp4_block_scale_routed_moe_trace.build_fi_trace_fn(
        "flashinfer.fused_moe.core.trtllm_fp4_block_scale_routed_moe"
    )(**routed_inputs)

    for defn in (default_defn, routed_defn):
        assert defn["axes"]["activation_type"]["value"] == ActivationType.Situ.value
        assert defn["inputs"]["activation_type"]["dtype"] == "int32"
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            assert defn["inputs"][name]["shape"] == ["num_local_experts"]
            assert defn["inputs"][name]["dtype"] == "float32"
            assert defn["inputs"][name]["optional"] is True

    assert "activation_value == 10" in default_defn["reference"]


def test_fp4_situ_trace_reference_mxfp4_mxfp8_and_pre_routed_match():
    from flashinfer.trace.templates.moe import (
        trtllm_fp4_block_scale_moe_default_routing_trace,
        trtllm_fp4_block_scale_routed_moe_trace,
    )

    inputs = _mxfp4_mxfp8_identity_inputs()
    hidden_size = inputs.pop("hidden_size")
    inputs.pop("intermediate_size")
    params = _situ_params()

    default_out = trtllm_fp4_block_scale_moe_default_routing_trace.reference(
        **inputs, **params
    ).to(torch.float32)
    routed_inputs = dict(inputs)
    routed_inputs.pop("routing_logits")
    routed_inputs.pop("routing_bias")
    routed_out = trtllm_fp4_block_scale_routed_moe_trace.reference(
        topk_ids=torch.zeros(4, 1, dtype=torch.int32),
        num_experts=1,
        **routed_inputs,
        **params,
    ).to(torch.float32)

    x0 = inputs["hidden_states"][:, :1].to(torch.float32).clamp(-7.0, 7.0)
    x1 = inputs["hidden_states"][:, 1:2].to(torch.float32).clamp(max=7.0)
    expected_first = torch.tanh(x0) * 1.7 * torch.tanh(x1 / 1.7) * torch.sigmoid(x1)
    expected = torch.zeros(4, hidden_size, dtype=torch.float32)
    expected[:, :1] = expected_first
    expected = expected.to(torch.bfloat16).to(torch.float32)

    swiglu_out = trtllm_fp4_block_scale_moe_default_routing_trace.reference(
        **inputs, activation_type=ActivationType.Swiglu.value
    ).to(torch.float32)

    torch.testing.assert_close(default_out, expected, rtol=0, atol=0)
    torch.testing.assert_close(routed_out, expected, rtol=0, atol=0)
    assert not torch.allclose(default_out, swiglu_out)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        pytest.param("gemm1_alpha", 0.0, id="zero_alpha"),
        pytest.param("gemm1_beta", -1.0, id="negative_beta"),
        pytest.param("gemm1_clamp_limit", float("nan"), id="nan_clamp"),
    ],
)
def test_fp4_situ_trace_init_rejects_invalid_parameters(parameter, value):
    from flashinfer.trace.templates.moe import (
        trtllm_fp4_block_scale_moe_default_routing_trace,
    )

    with pytest.raises(ValueError, match="finite and positive"):
        trtllm_fp4_block_scale_moe_default_routing_trace.init(
            seq_len=1,
            num_local_experts=1,
            activation_type=ActivationType.Situ.value,
            **{parameter: value},
        )
