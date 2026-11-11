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

"""CPU trace contracts for logits-routed BF16 and FP4 padding masks."""

import pytest
import torch

pytestmark = pytest.mark.cpu_only


def test_fp8_padding_skips_expert_matmuls(monkeypatch):
    from flashinfer.trace.templates.moe import _fp8_moe_run_experts

    def unexpected_matmul(*args, **kwargs):
        pytest.fail("Padding must not enter expert matmuls")

    monkeypatch.setattr(torch.Tensor, "matmul", unexpected_matmul)
    output = _fp8_moe_run_experts(
        torch.zeros(2, 128),
        torch.ones(1, 2),
        torch.zeros(1, 256, 128),
        torch.ones(1, 2, 1),
        torch.zeros(1, 128, 128),
        torch.ones(1, 1, 1),
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int64),
        0,
        1,
        token_mask=torch.ones(2, dtype=torch.bool),
    )
    assert output.isnan().all()


def test_fp8_saved_checker_rejects_nonfinite_real_rows():
    import json
    from pathlib import Path

    fixture = (
        Path(__file__).parents[2]
        / "trace/fi_trace_out"
        / ("moe_fp8_block_scale_renormalize_routing_topk4_e32_h7168_i2048.json")
    )
    namespace = {"torch": torch}
    exec(json.loads(fixture.read_text())["check"], namespace)
    check = namespace["_masked_fp8_moe_trace_check"]
    reference = torch.ones(3, 128)
    reference[2] = float("nan")
    actual = torch.ones_like(reference)
    assert check(reference, actual, max_mismatch_pct=1.0)
    for value in (float("nan"), float("inf")):
        actual[0, 0] = value
        assert not check(reference, actual, max_mismatch_pct=1.0)


def _kwargs(kind):
    n, h, inter, experts = 3, 32, 32, 2
    args = dict(
        routing_logits=torch.zeros(n, experts),
        routing_bias=None,
        hidden_states=torch.ones(n, h, dtype=torch.bfloat16),
        num_experts=experts,
        top_k=1,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=experts,
        routing_method_type=0,
        routed_scaling_factor=None,
        token_mask=torch.tensor([False, True, False]),
    )
    if kind == "bf16":
        args.update(
            gemm1_weights=torch.ones(experts, 2 * inter, h, dtype=torch.bfloat16)
            * 0.01,
            gemm2_weights=torch.ones(experts, h, inter, dtype=torch.bfloat16) * 0.01,
        )
    else:
        fp8 = torch.float8_e4m3fn
        args.update(
            hidden_states=torch.full((n, h // 2), 0x11, dtype=torch.uint8),
            hidden_states_scale=torch.ones(n, h // 16).to(fp8),
            gemm1_weights=torch.full(
                (experts, 2 * inter, h // 2), 0x11, dtype=torch.uint8
            ),
            gemm1_weights_scale=torch.ones(experts, 2 * inter, h // 16).to(fp8),
            gemm2_weights=torch.full((experts, h, inter // 2), 0x11, dtype=torch.uint8),
            gemm2_weights_scale=torch.ones(experts, h, inter // 16).to(fp8),
            gemm1_bias=None,
            gemm2_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
        )
    return args


def _api_template(kind):
    from flashinfer.fused_moe import trtllm_bf16_moe, trtllm_fp4_block_scale_moe
    from flashinfer.trace.templates.moe import (
        trtllm_bf16_moe_trace,
        trtllm_fp4_block_scale_moe_default_routing_trace,
    )

    if kind == "bf16":
        return trtllm_bf16_moe, trtllm_bf16_moe_trace
    return trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_moe_default_routing_trace


@pytest.mark.parametrize("kind", ["bf16", "fp4"])
def test_optional_mask_serialization(kind, tmp_path):
    api, _ = _api_template(kind)
    kwargs = _kwargs(kind)
    for mask in (None, kwargs["token_mask"]):
        kwargs["token_mask"] = mask
        definition = api.fi_trace(**kwargs, save_dir=tmp_path)
        descriptor = definition["inputs"]["token_mask"]
        assert descriptor["optional"] is True
        assert descriptor["dtype"] == "bool"
        assert descriptor["shape"] == ["seq_len"]
    assert list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("kind", ["bf16", "fp4"])
def test_reference_marks_only_padding_and_checker_rejects_real_nan(kind):
    _, template = _api_template(kind)
    kwargs = _kwargs(kind)
    expected = template.reference(**kwargs)
    mask = kwargs["token_mask"]
    assert torch.isnan(expected[mask]).all()
    assert torch.isfinite(expected[~mask]).all()
    actual = expected.clone()
    actual[mask] = float("inf")
    assert template.check(expected, actual)
    actual[0, 0] = float("nan")
    assert not template.check(expected, actual)


@pytest.mark.parametrize("kind", ["bf16", "fp4"])
def test_reference_rejects_nonfinite_real_token(kind):
    _, template = _api_template(kind)
    kwargs = _kwargs(kind)
    # A nonfinite expert weight makes every selected real token nonfinite.
    key = "gemm1_weights" if kind == "bf16" else "gemm1_weights_scale"
    kwargs[key].fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite.*real token"):
        template.reference(**kwargs)


def test_fp4_rejects_mask_with_precomputed_routing_metadata(monkeypatch):
    import inspect
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from flashinfer.fused_moe import core

    # Exercise the Python operation before any JIT module or CUDA allocation.
    monkeypatch.setattr(core, "register_custom_op", lambda *a, **k: lambda fn: fn)
    monkeypatch.setattr(core, "register_fake_op", lambda *a, **k: lambda fn: fn)
    module = SimpleNamespace(
        build_and_load=lambda: MagicMock(), get_library_paths=lambda: []
    )
    operation = core._build_trtllm_moe_namespace(module).trtllm_fp4_block_scale_moe
    kwargs = {
        name: None
        for name, parameter in inspect.signature(operation).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    kwargs.update(token_mask=torch.zeros(1, dtype=torch.bool), routing_metadata=[])
    with pytest.raises(ValueError, match="token_mask.*routing_metadata"):
        operation(**kwargs)


def test_bf16_clamped_relu2_preserves_limit_and_padding():
    from flashinfer.trace.templates.moe import _moe_bf16_run_experts

    hidden = torch.tensor([[2.0, -2.0], [float("nan"), float("nan")]])
    identity = torch.eye(2).unsqueeze(0)
    output = _moe_bf16_run_experts(
        hidden,
        identity,
        identity,
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int64),
        local_expert_offset=0,
        E_global=1,
        activation_type=12,
        clamped_relu2_limit=torch.tensor(1.0),
        token_mask=torch.tensor([False, True]),
    )
    expected = torch.relu(torch.tanh(hidden[0])).square().bfloat16()
    torch.testing.assert_close(output[0], expected)
    assert torch.isnan(output[1]).all()
    assert output[0, 0] < 1.0
