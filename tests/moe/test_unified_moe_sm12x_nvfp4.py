# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Native SM12x NVFP4 chain correctness and shared route consumers."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from benchmarks.bench_sm12x_nvfp4_ops import Fixture, activate, rne_codes
from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import is_sm120a_supported


@pytest.mark.parametrize(
    "rows,experts,n,sms,bm",
    [
        (16, 512, 2560, 188, 8),
        (17, 512, 2560, 188, 32),
        (20, 512, 2560, 188, 32),
        (20, 512, 2560, 110, 32),
        (21, 512, 2560, 188, 32),
        (20, 1, 128, 188, 8),
    ],
)
def test_nvfp4_fc2_tile_policy(rows, experts, n, sms, bm):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x._moe_utils.heuristic import (
        select_fp4_fc2_finalize_tile,
    )

    assert select_fp4_fc2_finalize_tile(
        total_rows=rows * experts, n=n, k=640, num_experts=experts, num_sms=sms
    ) == (bm, 128, 128)


def test_nvfp4_torch_oracle_rounding_and_limit():
    values = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected = torch.tensor([0, 0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8)
    torch.testing.assert_close(rne_codes(values), expected, rtol=0, atol=0)
    linear = torch.tensor([[-12.0, 12.0, -4.0, 4.0]])
    gate = torch.tensor([[12.0, 12.0, -12.0, -12.0]])
    actual = activate(torch.cat((linear, gate), dim=1), ActivationType.Swiglu, 10.0)
    expected = torch.nn.functional.silu(gate.clamp(max=10.0)) * linear.clamp(
        -10.0, 10.0
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, torch.nn.functional.silu(gate) * linear)


@pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available")
@pytest.mark.parametrize("op", ["fc1_act_q1", "fc2_finalize"])
def test_nvfp4_stream_uses_input_device(monkeypatch, op):
    module = importlib.import_module(
        f"flashinfer.fused_moe.cute_dsl.blackwell_sm12x.kernel_moe_fp4_{op}"
    )
    requested = []

    def current_stream(device=None):
        requested.append(device)
        return SimpleNamespace(cuda_stream=123)

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(module.cuda, "CUstream", lambda handle: handle)
    device = torch.device("cuda:1")
    assert module._stream(device) == 123
    assert requested == [device]


@pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available")
@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("explicit_tactic", [False, True])
def test_nvfp4_fc1_tuning_reuses_public_entry(monkeypatch, gather, explicit_tactic):
    module = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp4_fc1_act_q1"
    )
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(module, "cute_dsl_sm12x_fc1_act_q1_nvfp4", launch)
    runner = module._Fc1ActQ1Runner(
        torch.uint8, ActivationType.Swiglu, True, 7.0, 1.5, 0.5
    )
    inputs = [object() for _ in range(9 if gather else 8)]
    out = (object(), object())
    epi = module.EpiMethod.WG_S2R_QUANT
    tactic = (64, 128, epi) if explicit_tactic else None
    assert runner.launch(inputs, out, tactic) is None
    assert calls == [
        (
            tuple(inputs[:8]),
            dict(
                out_dtype=torch.uint8,
                tile=(64, 128, module.PLAIN_TILE_K) if explicit_tactic else None,
                epi=epi if explicit_tactic else None,
                tune=False,
                enable_pdl=True,
                activation=ActivationType.Swiglu,
                swiglu_limit=7.0,
                situ_beta=1.5,
                situ_linear_beta=0.5,
                out_q=out[0],
                out_sf=out[1],
                token_map=inputs[8] if gather else None,
            ),
        )
    ]


@pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available")
@pytest.mark.parametrize("explicit_tactic", [False, True])
def test_nvfp4_fc2_tuning_reuses_public_entry(monkeypatch, explicit_tactic):
    module = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp4_fc2_finalize"
    )
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(module, "cute_dsl_sm12x_fc2_finalize_nvfp4", launch)
    runner = module._Fc2FinalizeRunner(enable_pdl=True)
    inputs = [object() for _ in range(10)]
    out = object()
    tactic = (64, 128, module.DEFAULT_EPI) if explicit_tactic else None
    assert runner.launch(inputs, out, tactic) is None
    assert calls == [
        (
            tuple(inputs),
            dict(
                tile=(64, 128, module.BK) if explicit_tactic else None,
                epi=module.DEFAULT_EPI,
                tune=False,
                enable_pdl=True,
                out=out,
            ),
        )
    ]


@pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available")
@pytest.mark.parametrize("tokens", [8, 257])
def test_sm12x_nvfp4_native_chain_on_noncurrent_device(monkeypatch, tokens):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    device = torch.device("cuda:1")
    if not is_sm120a_supported(device):
        pytest.skip("requires an SM120a input device")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    with torch.cuda.device(device):
        data = Fixture("qwen30b", tokens, shape_override=(512, 512, 4, 2))
        monkeypatch.setattr(data.fc1, "_COMPILED", {})
        monkeypatch.setattr(data.fc2, "_COMPILED", {})
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
    other_stream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(stream), torch.cuda.stream(other_stream):
        for _ in range(2):
            assert torch.cuda.current_device() == 0
            output = data.native().clone()
            assert torch.cuda.current_device() == 0
            stream.synchronize()
            data.check(output)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.cuda.stream(other_stream):
        output = data.native()
        assert torch.cuda.current_device() == 0
    with torch.cuda.stream(stream), torch.cuda.stream(other_stream):
        graph.replay()
        assert torch.cuda.current_device() == 0
    stream.synchronize()
    data.check(output)


@pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available")
@pytest.mark.parametrize(
    "model,activation,tokens,enable_pdl,input_scale,shape_override,per_expert_scales",
    [
        ("qwen30b", ActivationType.Swiglu, 8, False, 1.0, (512, 512, 4, 2), False),
        ("qwen30b", ActivationType.Swiglu, 129, True, 1.0, (512, 512, 4, 2), False),
        ("qwen30b", ActivationType.Swiglu, 257, False, 1.0, (512, 512, 4, 2), False),
        ("dsv4", ActivationType.Swiglu, 16, False, 16.0, (512, 512, 4, 2), False),
        ("dsv4", ActivationType.Swiglu, 129, True, 16.0, (512, 512, 4, 2), False),
        ("qwen30b", ActivationType.Situ, 16, True, 1.0, (512, 512, 4, 2), False),
        ("nemotron", ActivationType.Relu2, 129, False, 1.0, (512, 512, 4, 2), False),
        ("nemotron", ActivationType.Relu2, 129, False, 1.0, (512, 1856, 4, 2), False),
        ("qwen30b", ActivationType.Swiglu, 8, False, 1.0, (512, 512, 4, 2), True),
        ("dsv4", ActivationType.Swiglu, 129, True, 16.0, (512, 512, 4, 2), True),
    ],
)
def test_sm12x_nvfp4_native_chain(
    model,
    activation,
    tokens,
    enable_pdl,
    input_scale,
    shape_override,
    per_expert_scales,
):
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    torch.backends.cuda.matmul.allow_tf32 = False
    data = Fixture(
        model,
        tokens,
        shape_override=shape_override,
        activation=activation,
        padded_routes=True,
        input_scale=input_scale,
        per_expert_scales=per_expert_scales,
    )
    immutable = data.input_hashes()
    data.check(data.native(enable_pdl).clone())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = data.native(enable_pdl)
    for _ in range(2):
        output.fill_(float("nan"))
        data.q1.fill_(255)
        data.q1_sf.fill_(255)
        graph.replay()
        torch.cuda.synchronize()
        data.check(output.clone())
        assert torch.count_nonzero(output[-1]).item() == 0
    assert data.input_hashes() == immutable


@pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available")
@pytest.mark.parametrize("precision", ["fp8", "mxfp8"])
@pytest.mark.parametrize("tokens", [17, 129])
def test_sm12x_shared_route_consumers(precision, tokens):
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    module = importlib.import_module(
        f"flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_{precision}_q0_route_triton"
    )
    experts, topk, hidden = 3, 2, 512
    token = torch.arange(tokens, device="cuda")
    ids = torch.stack((token % experts, (token + 1) % experts), dim=1).int()
    ids[-1] = -1
    ids[0, -1] = -1
    weights = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.float32).reshape(
            tokens, topk
        )
        + 1
    ) / (tokens * topk)
    x = torch.arange(tokens * hidden, device="cuda", dtype=torch.float32)
    x = (torch.sin(x * 0.01).reshape(tokens, hidden)).bfloat16()
    workspace = getattr(module, f"make_{precision}_q0_route_workspace")(x, ids, experts)
    offsets, token_map, routed_weights, _, _ = getattr(
        module, f"{precision}_q0_route_triton"
    )(x, ids, weights, experts, workspace=workspace)
    expected_counts = torch.stack(
        [(ids == expert).sum() for expert in range(experts)]
    ).int()
    expected_offsets = torch.cat(
        (expected_counts.new_zeros(1), expected_counts.cumsum(0))
    )
    torch.testing.assert_close(
        offsets, expected_offsets, check_dtype=False, rtol=0, atol=0
    )
    torch.testing.assert_close(offsets.diff(), expected_counts, rtol=0, atol=0)
    bounds = offsets.cpu().tolist()
    for expert, (begin, end) in enumerate(zip(bounds, bounds[1:], strict=False)):
        expected_token, expected_slot = (ids == expert).nonzero(as_tuple=True)
        permutation = token_map[begin:end].argsort()
        torch.testing.assert_close(
            token_map[begin:end][permutation],
            expected_token,
            check_dtype=False,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            routed_weights[begin:end][permutation],
            weights[expected_token, expected_slot],
            rtol=0,
            atol=0,
        )
    assert bounds[-1] == (ids >= 0).sum().item()
    assert not torch.any(token_map[: bounds[-1]] == tokens - 1)
    assert torch.all(workspace.dst_rows[ids < 0] == -1)
