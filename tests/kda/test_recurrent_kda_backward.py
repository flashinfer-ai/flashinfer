# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import pytest
import torch

from flashinfer import kda_backward as kda_backward_api
from flashinfer.kda_backward import (
    KDA_BACKWARD_GRADIENT_NAMES,
    RecurrentKDABackwardWorkspace,
    recurrent_kda_backward,
)


SUPPORTED_CASES = (
    ("fixed_t17_h1", (17,), 1, False),
    ("packed_17_33_65_h4", (17, 33, 65), 4, True),
    ("fixed_t17_h16", (17,), 16, False),
    ("fixed_t1024_h4", (1024,), 4, False),
    ("fixed_t4096_h32", (4096,), 32, False),
    ("fixed_t8192_h96", (8192,), 96, False),
    (
        "packed_1300_547_2048_963_271_3063_h96",
        (1300, 547, 2048, 963, 271, 3063),
        96,
        True,
    ),
    ("packed_1024x8_h96", (1024,) * 8, 96, True),
)


def _offsets(seq_lens):
    result = [0]
    for length in seq_lens:
        result.append(result[-1] + length)
    return tuple(result)


def _make_inputs(seq_lens, num_heads, packed, *, seed=0):
    total_tokens = sum(seq_lens)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    token_shape = (1, total_tokens, num_heads, 128)
    state_shape = (len(seq_lens), num_heads, 128, 128)

    def bf16(shape, multiplier=1.0):
        return (
            torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            * multiplier
        ).to(torch.bfloat16)

    inputs = {
        "q": bf16(token_shape),
        "k": bf16(token_shape),
        "v": bf16(token_shape),
        "g": bf16(token_shape, 0.1),
        "beta": bf16(token_shape[:-1]),
        "A_log": torch.log(
            torch.rand(
                (num_heads,), generator=generator, device=device, dtype=torch.float32
            )
            + 1.0
        ),
        "dt_bias": torch.randn(
            (num_heads, 128),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1,
        "initial_state": torch.randn(
            state_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.02,
        "do": bf16(token_shape, 0.1),
        "dfinal_state": torch.randn(
            state_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.1,
        "cu_seqlens": (
            torch.tensor(_offsets(seq_lens), dtype=torch.int64, device=device)
            if packed
            else None
        ),
    }
    return inputs


def _make_outputs(inputs):
    return (
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["q"]),
        torch.empty_like(inputs["beta"]),
        torch.empty_like(inputs["A_log"]),
        torch.empty_like(inputs["dt_bias"]),
        torch.empty_like(inputs["initial_state"]),
    )


def _reference(inputs, seq_lens):
    names = (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "A_log",
        "dt_bias",
        "initial_state",
    )
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True) for name in names
    }
    q_raw = leaves["q"].float()
    k_raw = leaves["k"].float()
    q = q_raw * torch.rsqrt((q_raw * q_raw).sum(-1, keepdim=True) + 1e-6)
    k = k_raw * torch.rsqrt((k_raw * k_raw).sum(-1, keepdim=True) + 1e-6)
    heads = inputs["q"].shape[2]
    q = q.reshape(-1, heads, 128)
    k = k.reshape_as(q)
    v = leaves["v"].float().reshape_as(q)
    g = leaves["g"].float().reshape_as(q)
    beta = torch.sigmoid(leaves["beta"].float().reshape(-1, heads))
    decay = torch.exp(
        -5.0
        * torch.sigmoid(
            leaves["A_log"].exp().view(1, heads, 1)
            * (g + leaves["dt_bias"].view(1, heads, 128))
        )
    )
    states = list(leaves["initial_state"].unbind(0))
    token = 0
    outputs = []
    for sequence, length in enumerate(seq_lens):
        state = states[sequence]
        for _ in range(length):
            decayed = state * decay[token].unsqueeze(1)
            prediction = torch.einsum("hk,hvk->hv", k[token], decayed)
            residual = beta[token].unsqueeze(-1) * (v[token] - prediction)
            state = decayed + residual.unsqueeze(-1) * k[token].unsqueeze(1)
            outputs.append(
                (1.0 / math.sqrt(128)) * torch.einsum("hk,hvk->hv", q[token], state)
            )
            token += 1
        states[sequence] = state
    output = torch.stack(outputs).reshape_as(inputs["q"]).to(torch.bfloat16)
    final_state = torch.stack(states)
    loss = output.float().mul(inputs["do"].float()).sum()
    loss = loss + final_state.mul(inputs["dfinal_state"]).sum()
    return torch.autograd.grad(loss, tuple(leaves[name] for name in names))


def _fla_reference(inputs):
    os.environ.setdefault("FLA_FLASH_KDA", "0")
    kda_ops = pytest.importorskip("fla.ops.kda")
    names = (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "A_log",
        "dt_bias",
        "initial_state",
    )
    leaves = {
        name: inputs[name].detach().clone().requires_grad_(True) for name in names
    }
    leaves["dt_bias"] = (
        inputs["dt_bias"].detach().reshape(-1).clone().requires_grad_(True)
    )
    cu_seqlens = inputs["cu_seqlens"]
    output, final_state = kda_ops.chunk_kda(
        leaves["q"],
        leaves["k"],
        leaves["v"],
        leaves["g"],
        leaves["beta"],
        scale=1.0 / math.sqrt(128),
        initial_state=leaves["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=(None if cu_seqlens is None else cu_seqlens.detach().cpu()),
        A_log=leaves["A_log"],
        dt_bias=leaves["dt_bias"],
        chunk_size=32,
    )
    gradients = torch.autograd.grad(
        (output, final_state),
        tuple(leaves[name] for name in names),
        grad_outputs=(inputs["do"], inputs["dfinal_state"]),
        allow_unused=False,
    )
    return (
        *gradients[:-2],
        gradients[-2].reshape_as(inputs["dt_bias"]),
        gradients[-1],
    )


class _RecorderModule:
    def __init__(self):
        self.low_calls = []
        self.high_calls = []
        self.c16_calls = []

    def run_low(self, *args):
        self.low_calls.append(args)

    def run_high(self, *args):
        self.high_calls.append(args)

    def run_c16_backward(self, *args):
        self.c16_calls.append(args)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _require_blackwell():
    _require_cuda()
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("the frozen KDA backward requires compute capability 10.0 or 10.3")


@pytest.mark.parametrize(("name", "seq_lens", "num_heads", "packed"), SUPPORTED_CASES)
def test_exact_supported_shape_table(name, seq_lens, num_heads, packed):
    spec = kda_backward_api._select_shape(
        (1, sum(seq_lens), num_heads, 128),
        len(seq_lens) + 1 if packed else None,
    )
    assert spec.name == name
    assert spec.seq_lens == seq_lens


@pytest.mark.parametrize(
    ("shape", "cu_numel"),
    [
        ((1, 18, 1, 128), None),
        ((2, 17, 1, 128), None),
        ((1, 115, 4, 128), None),
        ((1, 8192, 96, 128), 5),
        ((1, 8192, 64, 128), None),
        ((1, 17, 16, 64), None),
    ],
)
def test_unsupported_shapes_are_rejected(shape, cu_numel):
    with pytest.raises(ValueError, match="eight documented"):
        kda_backward_api._select_shape(shape, cu_numel)


def test_high_metadata_matches_chunk_schedule():
    spec = kda_backward_api._select_shape((1, 115, 4, 128), 4)
    metadata = kda_backward_api._metadata_values(spec)
    assert metadata["fixed_cu_seqlens"] == (0, 17, 50, 115)
    assert metadata["cu_chunk_offsets"] == (0, 1, 3, 6)
    assert metadata["seq_order"] == (2, 1, 0)
    assert metadata["chunk_sequence"] == (0, 1, 1, 2, 2, 2)
    assert metadata["chunk_index"] == (0, 0, 1, 0, 1, 2)
    assert metadata["consumer_chunk_order"] == (5, 2, 0, 4, 1, 3)
    assert metadata["chunk_pair_start"] == (0, 1, 3, 5)


def test_c16_metadata_matches_uniform_packed_schedule():
    spec = kda_backward_api._select_shape((1, 8192, 96, 128), 9)
    assert spec.c16_route
    metadata = kda_backward_api._metadata_values(spec)
    assert metadata["c16_checkpoint_cu_starts"] == tuple(range(0, 513, 64))
    assert len(metadata["c16_forward_work_items"]) == 8 * 96 * 8
    assert len(metadata["c16_backward_work_items"]) == 8 * 96 * 5
    assert metadata["c16_forward_work_items"][:8] == (0, 0, 0, 64, 0, 64, 0, 1024)
    assert metadata["c16_backward_work_items"][-5:] == (7, 95, 0, 64, 64)


@pytest.mark.parametrize(
    ("q_shape", "cu_numel"),
    [
        ((1, 4096, 32, 128), None),
        ((1, 8192, 96, 128), None),
        ((1, 8192, 96, 128), 7),
    ],
)
def test_c16_route_rejects_other_supported_shapes(q_shape, cu_numel):
    spec = kda_backward_api._select_shape(q_shape, cu_numel)
    assert not spec.c16_route


def test_packed_offsets_with_same_total_and_count_are_rejected(monkeypatch):
    _require_cuda()
    monkeypatch.setattr(kda_backward_api, "get_compute_capability", lambda _: (10, 3))
    recorder = _RecorderModule()
    monkeypatch.setattr(
        kda_backward_api, "_get_flash_kda_backward_module", lambda _: recorder
    )
    inputs = _make_inputs((17, 33, 65), 4, True)
    inputs["cu_seqlens"] = torch.tensor(
        (0, 18, 50, 115), dtype=torch.int64, device="cuda"
    )
    with pytest.raises(ValueError, match="requires cu_seqlens"):
        recurrent_kda_backward(**inputs)
    assert not recorder.low_calls
    assert not recorder.high_calls


@pytest.mark.parametrize(
    ("scale", "lower_bound", "match"),
    [
        (0.1, -5.0, "scale=1/sqrt"),
        (1.0 / math.sqrt(128), -4.0, "lower_bound=-5.0"),
    ],
)
def test_scale_and_lower_bound_are_fixed(monkeypatch, scale, lower_bound, match):
    _require_cuda()
    monkeypatch.setattr(kda_backward_api, "get_compute_capability", lambda _: (10, 3))
    inputs = _make_inputs((17,), 1, False)
    with pytest.raises(ValueError, match=match):
        recurrent_kda_backward(
            **inputs,
            scale=scale,
            lower_bound=lower_bound,
        )


def test_low_ffi_abi(monkeypatch):
    _require_cuda()
    monkeypatch.setattr(kda_backward_api, "get_compute_capability", lambda _: (10, 3))
    recorder = _RecorderModule()
    monkeypatch.setattr(
        kda_backward_api, "_get_flash_kda_backward_module", lambda _: recorder
    )
    inputs = _make_inputs((17,), 1, False)
    outputs = _make_outputs(inputs)
    actual = recurrent_kda_backward(**inputs, out=outputs)
    assert tuple(tensor.data_ptr() for tensor in actual) == tuple(
        tensor.data_ptr() for tensor in outputs
    )
    assert len(recorder.low_calls) == 1
    assert not recorder.high_calls
    args = recorder.low_calls[0]
    assert len(args) == 33
    assert args[0].data_ptr() == inputs["q"].data_ptr()
    assert args[10].dtype == torch.int64
    assert tuple(args[10].cpu().tolist()) == (0, 17)
    assert args[20].data_ptr() == outputs[0].data_ptr()
    assert args[27].data_ptr() == outputs[7].data_ptr()
    assert args[28:30] == (1, 1)


def test_high_ffi_abi_and_capture_prepare_flag(monkeypatch):
    _require_cuda()
    monkeypatch.setattr(kda_backward_api, "get_compute_capability", lambda _: (10, 3))
    recorder = _RecorderModule()
    monkeypatch.setattr(
        kda_backward_api, "_get_flash_kda_backward_module", lambda _: recorder
    )
    capturing = iter((False, True))
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: next(capturing)
    )
    inputs = _make_inputs((17,), 16, False)
    outputs = _make_outputs(inputs)
    workspace = RecurrentKDABackwardWorkspace("cuda")
    recurrent_kda_backward(**inputs, workspace=workspace, out=outputs)
    recurrent_kda_backward(**inputs, workspace=workspace, out=outputs)

    assert len(recorder.high_calls) == 2
    assert not recorder.low_calls
    warm_args, capture_args = recorder.high_calls
    assert len(warm_args) == 57
    assert warm_args[-6] == 1
    assert capture_args[-6] == 0
    descriptor = warm_args[18]
    assert descriptor.dtype == torch.uint8
    assert descriptor.numel() >= 768
    assert descriptor.data_ptr() % 64 == 0
    assert warm_args[5].shape == (32, 16)
    assert warm_args[34].shape == (16,)
    assert tuple(warm_args[12].cpu().tolist()) == (0,)
    assert tuple(warm_args[13].cpu().tolist()) == (0, 1)


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "packed"),
    [
        ((17,), 1, False),
        ((17, 33, 65), 4, True),
        ((17,), 16, False),
    ],
)
def test_backward_matches_pytorch_reference(seq_lens, num_heads, packed):
    _require_blackwell()
    inputs = _make_inputs(seq_lens, num_heads, packed, seed=1234)
    expected = _reference(inputs, seq_lens)
    actual = recurrent_kda_backward(**inputs)
    assert tuple(KDA_BACKWARD_GRADIENT_NAMES) == (
        "dq",
        "dk",
        "dv",
        "dg",
        "dbeta",
        "dA_log",
        "ddt_bias",
        "dinitial_state",
    )
    for name, actual_tensor, expected_tensor in zip(
        KDA_BACKWARD_GRADIENT_NAMES, actual, expected, strict=True
    ):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize(("name", "seq_lens", "num_heads", "packed"), SUPPORTED_CASES)
def test_backward_matches_fla_reference_all_supported_shapes(
    name, seq_lens, num_heads, packed
):
    _require_blackwell()
    inputs = _make_inputs(seq_lens, num_heads, packed, seed=1234)
    expected = _fla_reference(inputs)
    actual = recurrent_kda_backward(**inputs)
    for gradient_name, actual_tensor, expected_tensor in zip(
        KDA_BACKWARD_GRADIENT_NAMES, actual, expected, strict=True
    ):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda message: f"{name}/{gradient_name}: {message}",
        )


def test_cuda_graph_capture_after_exact_warmup():
    _require_blackwell()
    inputs = _make_inputs((17,), 16, False, seed=4321)
    outputs = _make_outputs(inputs)
    workspace = RecurrentKDABackwardWorkspace("cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        recurrent_kda_backward(**inputs, workspace=workspace, out=outputs)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        recurrent_kda_backward(**inputs, workspace=workspace, out=outputs)
    graph.replay()
    torch.cuda.synchronize()
    for tensor in outputs:
        assert torch.isfinite(tensor).all()
