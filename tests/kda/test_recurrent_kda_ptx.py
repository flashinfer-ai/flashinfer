# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness and serving contracts for PTX KDA."""

import math

import pytest
import torch

from flashinfer import RecurrentKDAPrefillWorkspace, recurrent_kda
from flashinfer.utils import get_compute_capability
from tests.test_helpers.kda_prefill import _make_inputs

pytest.importorskip("tvm_ffi")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) != (10, 3),
    reason="Requires SM103 and ptxas >= 13.2",
)


@pytest.fixture(scope="module", autouse=True)
def _require_ptxas():
    from flashinfer.kda_kernels.ptx.static_runtime import _ptxas

    try:
        _ptxas()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _inputs(lengths=(37,), heads=64, packed=False, state=True, seed=0):
    return _make_inputs(
        seq_lens=lengths,
        num_heads=heads,
        packed=packed,
        initial_state=state,
        state_dtype=torch.float32,
        seed=seed,
    )


def _call(inputs, **kwargs):
    options = dict(
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        lower_bound=-5.0,
        backend="ptx",
    )
    options.update(kwargs)
    return recurrent_kda(**inputs, **options)


def _reference(inputs, scale=None, lower_bound=-5.0):
    """FP64 token recurrence; no BF16 rounding of the recurrent state."""
    B, T, H, D = inputs["q"].shape
    scale = 1 / math.sqrt(D) if scale is None else scale
    q, k = (inputs[name].double().reshape(-1, H, D) for name in ("q", "k"))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    v = inputs["v"].double().reshape(-1, H, D)
    gate = lower_bound * torch.sigmoid(
        inputs["A_log"].double().exp().view(1, H, 1)
        * (
            inputs["g"].double().reshape(-1, H, D)
            + inputs["dt_bias"].double().view(1, H, D)
        )
    )
    beta = inputs["beta"].double().sigmoid().reshape(-1, H)
    offsets = (
        list(range(0, B * T + 1, T))
        if inputs["cu_seqlens"] is None
        else inputs["cu_seqlens"].tolist()
    )
    state = (
        torch.zeros(len(offsets) - 1, H, D, D, dtype=torch.float64, device=q.device)
        if inputs["initial_state"] is None
        else inputs["initial_state"].double()
    )
    out = torch.empty_like(q)
    for i, (start, end) in enumerate(zip(offsets, offsets[1:], strict=False)):
        s = state[i]
        for t in range(start, end):
            s = s * gate[t].exp().unsqueeze(-2)
            delta = beta[t].unsqueeze(-1) * (v[t] - (s * k[t].unsqueeze(-2)).sum(-1))
            s = s + delta.unsqueeze(-1) * k[t].unsqueeze(-2)
            out[t] = scale * (s * q[t].unsqueeze(-2)).sum(-1)
        state[i] = s
    return out.reshape(B, T, H, D).float(), state.float()


def _assert_close(result, reference):
    for got, want in zip(result, reference, strict=True):
        assert torch.isfinite(got).all()
        # BF16 MMA operands round once, while the oracle keeps FP64 throughout.
        torch.testing.assert_close(got.float(), want.float(), atol=1e-2, rtol=1e-2)
        error = torch.linalg.vector_norm(got.float() - want.float())
        norm = torch.linalg.vector_norm(want.float())
        assert error <= 0.01 * norm + 1e-6


def _assert_int21_close(result, reference):
    # Imported kernel's accuracy contract for long chains with weak decay.
    for got, want in zip(result, reference, strict=True):
        x, y = got.float(), want.float()
        error = (x - y).abs()
        assert torch.isfinite(x).all()
        assert torch.linalg.vector_norm(error) <= 0.03 * torch.linalg.vector_norm(y)
        assert torch.all(
            error <= torch.maximum(0.5 * y.square().mean().sqrt(), 0.05 * y.abs())
        )


@pytest.mark.parametrize(
    "lengths,packed,heads",
    [
        ((32,), False, 64),
        ((37,), False, 96),
        ((97,), False, 64),
        ((33, 65), True, 96),
        ((1, 17, 32), True, 64),
    ],
)
@pytest.mark.parametrize("state", [False, True])
def test_ptx_kda_reference(lengths, packed, heads, state):
    inputs = _inputs(lengths, heads, packed, state)
    reference = _reference(inputs, scale=0.17)
    output = torch.empty_like(inputs["v"])
    result = _call(inputs, output=output, output_final_state=True, scale=0.17)
    assert result[0] is output
    if state:
        assert result[1] is inputs["initial_state"]
    _assert_close(result, reference)


@pytest.mark.parametrize("heads", [64, 96])
@pytest.mark.parametrize(
    "lengths", [(8192,), (1300, 547, 2048, 963, 271, 3063), (1024,) * 8]
)
def test_ptx_kda_int21(heads, lengths, monkeypatch):
    monkeypatch.setenv("FLA_FLASH_KDA", "0")
    monkeypatch.setenv("FLA_TILELANG", "0")
    chunk_kda = pytest.importorskip("fla.ops.kda").chunk_kda
    inputs = _inputs(lengths, heads, len(lengths) > 1)
    # Realistic K3 biases preserve substantial long-range recurrent state.
    generator = torch.Generator(device="cuda").manual_seed(42)
    inputs["A_log"] = (
        torch.empty(heads, device="cuda").uniform_(1, 16, generator=generator).log()
    )
    dt = (
        torch.empty(heads * 128, device="cuda")
        .uniform_(math.log(0.001), math.log(0.1), generator=generator)
        .exp()
    )
    inputs["dt_bias"] = dt + torch.log(-torch.expm1(-dt))
    reference = chunk_kda(
        **inputs,
        scale=128**-0.5,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    result = _call(inputs, output_final_state=True)
    _assert_int21_close(result, reference)


def test_ptx_kda_updates_state_without_returning_it():
    inputs = _inputs()
    reference = _reference(inputs)
    out, state = _call(inputs)
    assert state is None
    _assert_close((out, inputs["initial_state"]), reference)


def test_ptx_kda_small_norms():
    inputs = _inputs()
    inputs["q"].mul_(1e-5)
    inputs["k"].mul_(1e-5)
    reference = _reference(inputs)
    _assert_close(_call(inputs, output_final_state=True), reference)


@pytest.mark.parametrize(
    "heads,lengths",
    [(64, (37, 65)), (64, (1024,) * 8), (96, (1024,) * 8), (64, (256,)), (96, (256,))],
)
def test_ptx_kda_graph_and_stream(heads, lengths):
    inputs = _inputs(lengths, heads, len(lengths) > 1)
    output = torch.empty_like(inputs["q"])
    initial = inputs["initial_state"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    workspace = RecurrentKDAPrefillWorkspace(device=inputs["q"].device)
    with torch.cuda.stream(stream):
        _call(
            inputs, output=output, output_final_state=True, prefill_workspace=workspace
        )
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _call(
            inputs, output=output, output_final_state=True, prefill_workspace=workspace
        )
    for seed in (10, 11):
        # Change all activations and state, then interleave eager and replay.
        other = _inputs(lengths, heads, len(lengths) > 1, seed=seed)
        for key in ("q", "k", "v", "g", "beta"):
            inputs[key].copy_(other[key])
        inputs["initial_state"].copy_(initial)
        expected = _call(
            {**inputs, "initial_state": initial.clone()}, output_final_state=True
        )
        graph.replay()
        torch.cuda.synchronize()
        _assert_close((output, inputs["initial_state"]), expected)
    with torch.cuda.stream(stream), pytest.raises(RuntimeError, match="participated"):
        _call(inputs, output=output, prefill_workspace=workspace)


def test_ptx_kda_packed_offsets_mutation():
    with torch.inference_mode():
        inputs = _inputs((33, 65), 64, True)
        state = inputs["initial_state"].clone()
        _call(inputs)
        inputs["cu_seqlens"][1] = 17
        inputs["initial_state"].copy_(state)
        _assert_close(
            _call(inputs, output_final_state=True),
            _reference({**inputs, "initial_state": state}),
        )


@pytest.mark.parametrize(
    "kind",
    [
        "bf16_state",
        "strided_beta",
        "alias_output",
        "empty_sequence",
        "bad_offsets",
        "unsupported",
        "cold_capture",
        "gate_bound",
    ],
)
def test_ptx_kda_rejects_invalid_contract(kind):
    inputs = _inputs((37, 65), 64, True)
    if kind == "bf16_state":
        inputs["initial_state"] = inputs["initial_state"].bfloat16()
    elif kind == "strided_beta":
        inputs["beta"] = torch.empty(1, 102, 128, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif kind == "empty_sequence":
        inputs["cu_seqlens"][1] = 0
    elif kind == "bad_offsets":
        inputs["cu_seqlens"][-1] = 101
    if kind == "gate_bound":
        with pytest.raises(ValueError, match="lower_bound=-5.0"):
            _call(inputs, lower_bound=-1.0)
    elif kind == "unsupported":
        with pytest.raises(NotImplementedError, match="ssm_state_indices"):
            _call(
                inputs,
                ssm_state_indices=torch.arange(2, device="cuda", dtype=torch.int32),
            )
    elif kind == "cold_capture":
        with (
            pytest.raises(RuntimeError, match="warmed workspace"),
            torch.cuda.graph(torch.cuda.CUDAGraph()),
        ):
            _call(inputs)
    else:
        with pytest.raises(ValueError):
            _call(inputs, **({"output": inputs["v"]} if kind == "alias_output" else {}))


@pytest.mark.parametrize("lower_bound", [-5.0])
def test_ptx_kda_weak_decay_and_zero_norm(lower_bound):
    inputs = _inputs((257,), 64, False)
    inputs["q"][:, ::5] = 0
    inputs["k"][:, ::7] = 0
    inputs["g"].fill_(-12)
    inputs["A_log"].zero_()
    inputs["dt_bias"].zero_()
    reference = _reference(inputs, lower_bound=lower_bound)
    result = _call(inputs, output_final_state=True, lower_bound=lower_bound)
    # Near-unit decay retains BF16 operand-rounding error for the whole chain.
    _assert_int21_close(result, reference)


def test_ptx_kda_rejects_misaligned_state():
    inputs = _inputs()
    state = inputs["initial_state"]
    storage = torch.empty(state.numel() + 4, device=state.device, dtype=state.dtype)
    inputs["initial_state"] = storage[4:].view_as(state)
    with pytest.raises(ValueError, match="32-byte"):
        _call(inputs)


def test_ptx_kda_capture_requires_exact_warm_buffers():
    inputs = _inputs()
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(inputs, output=output, prefill_workspace=workspace)
    stream.synchronize()
    replacement = inputs["q"].clone()
    with (
        pytest.raises(RuntimeError, match="exact capture tensors"),
        torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream),
    ):
        _call({**inputs, "q": replacement}, output=output, prefill_workspace=workspace)


@pytest.mark.parametrize("heads", [64, 96])
def test_ptx_kda_warm_fixed_int21(heads):
    inputs = _inputs((8192,), heads)
    inputs["g"].fill_(-12)
    inputs["A_log"].zero_()
    inputs["dt_bias"].zero_()
    initial = inputs["initial_state"].clone()
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(device="cuda")
    result = _call(
        inputs, output=output, output_final_state=True, prefill_workspace=workspace
    )
    expected = tuple(tensor.clone() for tensor in result)
    # Warm repeated launches must preserve the complete long recurrence.
    for _ in range(3):
        inputs["initial_state"].copy_(initial)
        result = _call(
            inputs, output=output, output_final_state=True, prefill_workspace=workspace
        )
        for got, want in zip(result, expected, strict=True):
            torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("value", [4097.0, float("inf"), float("nan")])
def test_ptx_rejects_state_outside_numeric_domain(value):
    inputs = _inputs()
    inputs["initial_state"].flatten()[0] = value
    with pytest.raises(ValueError, match="finite initial_state"):
        _call(inputs)


def test_ptx_zero_state_outputs_are_owned():
    inputs = _inputs(state=False)
    first = _call(inputs, output_final_state=True)
    saved = tuple(x.clone() for x in first)
    inputs["v"].zero_()
    _call(inputs, output_final_state=True)
    for got, want in zip(first, saved, strict=False):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("heads", [64, 96])
def test_ptx_raw_launch_eager_capture_replay(heads):
    from flashinfer.kda_kernels.ptx.kda_fwd_fused import prepare_kda_forward

    inputs = _inputs((1024,) * 8, heads, True)
    out = torch.empty_like(inputs["q"])
    state = torch.empty_like(inputs["initial_state"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan = prepare_kda_forward(
            **inputs, scale=128**-0.5, out=out, final_state=state
        )
        plan.launch()
    stream.synchronize()
    expected = (out.clone(), state.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.launch()
    for _ in range(4):
        # Scaling both sources scales the linear recurrence; stale handoff
        # state from an earlier invocation must not be reused.
        inputs["v"].mul_(0.5)
        inputs["initial_state"].mul_(0.5)
        for tensor in expected:
            tensor.mul_(0.5)
        graph.replay()
        plan.launch()
        torch.cuda.synchronize()
        _assert_close((out, state), expected)


def test_ptx_rejects_too_few_total_tokens():
    with pytest.raises(ValueError, match="T>=32"):
        _call(_inputs((2,)))
