# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public SiTU backend correctness and caller-owned graph/workspace behavior."""

import pytest
import torch

from flashinfer.fused_moe import (
    QuantVariant,
    SiTU,
    TrtllmFp4Config,
    cutlass_fused_moe,
    cutlass_fused_moe_prepare_workspace,
    cutlass_fused_moe_workspace_size,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.tllm_enums import ActivationType, RoutingMethodType
from flashinfer.utils import device_support_pdl


HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 3584, 384, 896, 16


def _workspace_size(**overrides):
    options = dict(
        max_num_tokens=2048,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_total=EXPERTS,
        top_k=TOP_K,
        x_dtype=torch.bfloat16,
        weight_dtype=torch.uint8,
        activation_type=ActivationType.Situ,
        tp_size=8,
        backend="cake",
    )
    options.update(overrides)
    return cutlass_fused_moe_workspace_size(**options)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"backend": "unknown"}, "unsupported.*backend"),
        ({"activation_type": ActivationType.Swiglu}, "activation_type"),
        ({"top_k": 8}, "top_k=16"),
    ],
)
def test_cake_situ_rejects_unsupported_dispatch(options, message):
    # This public size query must reject unsupported semantics using metadata.
    with pytest.raises(ValueError, match=message):
        _workspace_size(**options)


@pytest.fixture(scope="module")
def cake_situ_device():
    if not torch.cuda.is_available():
        pytest.skip("Cake SiTU requires a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("Cake SiTU requires SM100 or SM103")
    return device


@pytest.fixture(scope="module")
def cake_situ_weights(cake_situ_device):
    # Use the existing public preparation, including the SiTU gate/up shuffle.
    # Keep one full expert fixture for all token counts instead of repeatedly
    # quantizing the same multi-GiB weights.
    with pytest.MonkeyPatch.context() as patch:
        for name, value in {
            "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
            "FLASHINFER_NVFP4_4OVER6": "0",
            "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MAE",
            "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
            "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "0",
        }.items():
            patch.setenv(name, value)
        generator = torch.Generator(device=cake_situ_device).manual_seed(4568)
        w1 = torch.randn(
            EXPERTS, 2 * INTERMEDIATE, HIDDEN,
            device=cake_situ_device, dtype=torch.bfloat16, generator=generator,
        ).mul_(0.125)
        w2 = torch.randn(
            EXPERTS, HIDDEN, INTERMEDIATE,
            device=cake_situ_device, dtype=torch.bfloat16, generator=generator,
        ).mul_(0.125)
        prepared = TrtllmFp4Config.prepare_weights(
            w1, w2,
            variant=QuantVariant.NVFP4,
            num_local_experts=EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            activation=SiTU(),
        )
        del w1, w2
        yield prepared


@pytest.fixture(scope="module")
def cake_situ_workspace(cake_situ_device):
    # A single maximum-size allocation is reused across all four token sizes.
    return torch.empty(_workspace_size(), dtype=torch.uint8, device=cake_situ_device)


def _trtllm_reference(x, ids, route_weights, prepared):
    quantized, scales = TrtllmFp4Config.prepare_activations(
        x, variant=QuantVariant.NVFP4,
    )
    output = torch.empty_like(x)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, route_weights),
        routing_bias=None,
        hidden_states=quantized,
        hidden_states_scale=scales,
        gemm1_weights=prepared["gemm1_weights"],
        gemm1_weights_scale=prepared["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=prepared["gemm1_alpha"],
        gemm1_beta=prepared["gemm1_beta"],
        gemm1_clamp_limit=None,
        gemm2_weights=prepared["gemm2_weights"],
        gemm2_weights_scale=prepared["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=prepared["output1_scale_scalar"],
        output1_scale_gate_scalar=prepared["output1_scale_gate_scalar"],
        output2_scale_scalar=prepared["output2_scale_scalar"],
        num_experts=EXPERTS,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        activation_type=ActivationType.Situ.value,
        do_finalize=True,
        enable_pdl=device_support_pdl(x.device),
        per_token_scale=None,
        output=output,
        tune_max_num_tokens=16384,
    )
    if isinstance(result, (list, tuple)):
        assert len(result) == 1
        result = result[0]
    assert result.data_ptr() == output.data_ptr()
    return output


@pytest.mark.parametrize(
    "num_tokens", [64, 256, 512, 2048], ids=["n8", "n16", "n32", "n128"],
)
def test_cake_situ_output_workspace_and_external_graph(
    num_tokens, cake_situ_device, cake_situ_weights, cake_situ_workspace,
):
    device, prepared, workspace = (
        cake_situ_device, cake_situ_weights, cake_situ_workspace,
    )
    generator = torch.Generator(device=device).manual_seed(9000 + num_tokens)
    x = torch.randn(
        num_tokens, HIDDEN, device=device, dtype=torch.bfloat16, generator=generator,
    )
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    ids = ((tokens[:, None] * TOP_K + slots[None, :]) % EXPERTS).contiguous()
    route_weights = torch.randn(
        num_tokens, TOP_K, device=device, generator=generator,
    ).softmax(dim=-1).to(torch.bfloat16)
    output = torch.full_like(x, float("nan"))
    output_ptr, workspace_ptr = output.data_ptr(), workspace.data_ptr()
    quant_scales = [
        torch.ones(1, device=device, dtype=torch.float32),
        prepared["gemm1_weights_scale"],
        prepared["output1_scale_gate_scalar"],
        prepared["output1_scale_scalar"],
        prepared["gemm2_weights_scale"],
        prepared["output2_scale_scalar"],
    ]

    def submit(*, explicit_parameters=False):
        activation_kwargs = (
            {"situ_beta": prepared["gemm1_alpha"],
             "situ_linear_beta": prepared["gemm1_beta"]}
            if explicit_parameters else {}
        )
        return cutlass_fused_moe(
            x, ids, route_weights,
            prepared["gemm1_weights"], prepared["gemm2_weights"],
            torch.bfloat16, quant_scales,
            activation_type=ActivationType.Situ,
            tp_size=8, backend="cake",
            output=output, workspace_buffer=workspace,
            **activation_kwargs,
        )

    with pytest.raises(ValueError, match="prepar"):
        submit()
    assert torch.isnan(output).all()
    assert cutlass_fused_moe_prepare_workspace(
        workspace, num_tokens, backend="cake",
        weight_layout="trtllm_shuffled_nvfp4_group16",
    ) is workspace

    expected = _trtllm_reference(x, ids, route_weights, prepared)
    # Ensure an all-zero output could not satisfy the FP4 absolute tolerance.
    assert expected.abs().max() > 2.0
    assert submit() is output
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        submit()
        submit()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_output = submit()
    assert captured_output is output
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)

    # Retain every input address while changing values and expert load balance.
    # Replay must consume these values, not stale routing or hidden states.
    x.mul_(-0.75)
    ids.copy_(slots[None, :].expand_as(ids))
    route_weights.copy_(route_weights.flip(-1))
    changed_expected = _trtllm_reference(x, ids, route_weights, prepared)
    assert not torch.equal(expected, changed_expected)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, changed_expected, atol=1.0, rtol=0.1)

    default_parameters_output = output.clone()
    output.fill_(float("nan"))
    assert submit(explicit_parameters=True) is output
    torch.testing.assert_close(output, default_parameters_output, atol=0.0, rtol=0.0)
    assert output.data_ptr() == output_ptr
    assert workspace.data_ptr() == workspace_ptr
