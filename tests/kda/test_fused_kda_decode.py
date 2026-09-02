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

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported

try:
    from flashinfer.kda_decode import (
        _FUSED_KDA_DECODE_AVAILABLE,
        fused_kda_decode,
    )

    _has_fused_kda_decode = _FUSED_KDA_DECODE_AVAILABLE
except ImportError:
    fused_kda_decode = None
    _has_fused_kda_decode = False


_KIMI_K3_CASES = (
    (96, 1),
    (96, 4),
    (96, 8),
    (96, 32),
    (96, 128),
    (48, 1),
    (48, 4),
    (48, 32),
    (48, 128),
    (32, 1),
    (32, 4),
    (32, 32),
    (32, 128),
    (24, 1),
    (24, 4),
    (24, 32),
    (24, 64),
    (12, 1),
    (12, 4),
    (12, 32),
    (12, 256),
)


@pytest.fixture(autouse=True)
def _require_fused_kda_decode():
    if not torch.cuda.is_available():
        pytest.skip("Fused KDA decode requires CUDA")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Fused KDA decode requires SM100a (Blackwell)")
    if not _has_fused_kda_decode:
        pytest.skip("fused_kda_decode is unavailable (missing CuTe DSL dependencies)")


def _page_strides(num_heads, state_dtype):
    conv_slot_bytes = 3 * num_heads * 128 * 3 * 2
    state_element_bytes = torch.empty((), dtype=state_dtype).element_size()
    state_slot_bytes = num_heads * 128 * 128 * state_element_bytes
    page_bytes = conv_slot_bytes + state_slot_bytes
    return page_bytes // 2, page_bytes // state_element_bytes


def _make_inputs(num_heads, num_rows, seed=42, state_dtype=torch.float32):
    device = torch.device("cuda")
    num_slots = num_rows + 1
    hidden_size = num_heads * 128
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype=torch.float32):
        return torch.randn(
            shape, device=device, dtype=torch.float32, generator=generator
        ).to(dtype)

    x_storage = randn((num_rows, 3 * hidden_size + 17), torch.bfloat16)
    x = x_storage[:, : 3 * hidden_size]
    weight = 0.1 * randn((3, 4, hidden_size))

    conv_slot_stride, state_slot_stride = _page_strides(num_heads, state_dtype)
    conv_state = torch.empty_strided(
        (num_slots, 3 * hidden_size, 3),
        (conv_slot_stride, 1, 3 * hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state.copy_(0.1 * randn((num_slots, 3 * hidden_size, 3), torch.bfloat16))
    state = torch.empty_strided(
        (num_slots, num_heads, 128, 128),
        (state_slot_stride, 128 * 128, 128, 1),
        dtype=state_dtype,
        device=device,
    )
    state.copy_(0.01 * randn((num_slots, num_heads, 128, 128), state_dtype))

    beta_storage = randn((1, num_rows, num_heads + 1), torch.bfloat16)
    gate_storage = randn((num_rows, hidden_size + 7), torch.bfloat16)
    return {
        "x": x,
        "weight": weight,
        "conv_state": conv_state,
        "raw_gate": randn((1, num_rows, num_heads, 128), torch.bfloat16),
        "raw_beta": beta_storage[:, :, :num_heads],
        "A_log": 0.5 * randn((num_heads,)),
        "dt_bias": 0.1 * randn((hidden_size,)),
        "state_indices": torch.arange(
            num_rows, 0, -1, dtype=torch.int32, device=device
        ),
        "state": state,
        "output_gate": gate_storage.as_strided(
            (num_rows, num_heads, 128), (hidden_size + 7, 128, 1)
        ),
        "norm_weight": randn((128,)),
    }


def _clone_strided(tensor):
    clone = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    clone.copy_(tensor)
    return clone


@torch.no_grad()
def _reference(inputs, conv_state, state, lower_bound=-5.0):
    x = inputs["x"]
    num_rows = x.shape[0]
    num_heads = inputs["A_log"].numel()
    hidden_size = num_heads * 128
    slots = inputs["state_indices"].long()

    taps = inputs["weight"].permute(0, 2, 1).reshape(3 * hidden_size, 4)
    history = conv_state.index_select(0, slots).float()
    window = torch.cat((history, x.float().unsqueeze(-1)), dim=-1)
    conv_state.index_copy_(0, slots, window[:, :, 1:].to(torch.bfloat16))
    mixed = (window * taps).sum(-1)
    mixed = F.silu(mixed).to(torch.bfloat16).float()
    query, key, value = mixed.view(num_rows, 3, num_heads, 128).unbind(1)

    query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    query *= 128**-0.5
    key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    gate = inputs["raw_gate"][0].float() + inputs["dt_bias"].view(num_heads, 128)
    if lower_bound is None:
        gate = -inputs["A_log"].exp()[None, :, None] * F.softplus(gate)
    else:
        gate = lower_bound * torch.sigmoid(inputs["A_log"].exp()[None, :, None] * gate)

    selected_state = state.index_select(0, slots)
    selected_state = selected_state * gate.exp().unsqueeze(-2)
    state_key = torch.einsum("nhvk,nhk->nhv", selected_state, key)
    delta = value - state_key
    delta *= inputs["raw_beta"][0].float().sigmoid().unsqueeze(-1)
    selected_state += delta.unsqueeze(-1) * key.unsqueeze(-2)
    state.index_copy_(0, slots, selected_state.to(state.dtype))

    output = torch.einsum("nhvk,nhk->nhv", selected_state, query)
    output = output.to(torch.bfloat16).float()
    inverse_rms = torch.rsqrt(output.square().mean(-1, keepdim=True) + 1e-5)
    output = (
        output
        * inverse_rms
        * inputs["norm_weight"]
        * inputs["output_gate"].float().sigmoid()
    )
    return output.unsqueeze(0).to(torch.bfloat16)


@pytest.mark.parametrize(
    ("num_heads", "num_rows"),
    [
        pytest.param(num_heads, num_rows, id=f"h{num_heads}-n{num_rows}")
        for num_heads, num_rows in _KIMI_K3_CASES
    ],
)
def test_fused_kda_decode_kimi_k3_shapes(num_heads, num_rows):
    inputs = _make_inputs(num_heads, num_rows)
    reference_conv_state = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    actual_conv_state = _clone_strided(inputs["conv_state"])
    actual_state = _clone_strided(inputs["state"])

    expected = _reference(inputs, reference_conv_state, reference_state)
    actual = fused_kda_decode(
        x=inputs["x"],
        weight=inputs["weight"],
        conv_state=actual_conv_state,
        raw_gate=inputs["raw_gate"],
        raw_beta=inputs["raw_beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        state_indices=inputs["state_indices"],
        state=actual_state,
        output_gate=inputs["output_gate"],
        norm_weight=inputs["norm_weight"],
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(actual_conv_state, reference_conv_state, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_rows", [1, 4, 32, 128])
def test_fused_kda_decode_kimi_linear_softplus(num_rows, state_dtype):
    inputs = _make_inputs(num_heads=32, num_rows=num_rows, state_dtype=state_dtype)
    reference_conv_state = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    actual_conv_state = _clone_strided(inputs["conv_state"])
    actual_state = _clone_strided(inputs["state"])

    expected = _reference(
        inputs, reference_conv_state, reference_state, lower_bound=None
    )
    kwargs = {
        **inputs,
        "conv_state": actual_conv_state,
        "state": actual_state,
        "lower_bound": None,
    }
    actual = fused_kda_decode(**kwargs)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(actual_conv_state, reference_conv_state, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)


@pytest.mark.parametrize(
    ("num_heads", "lower_bound", "state_dtype"),
    [
        (12, -5.0, torch.float32),
        (12, -5.0, torch.bfloat16),
        (32, -5.0, torch.float32),
        (32, None, torch.float32),
        (32, None, torch.bfloat16),
    ],
)
def test_fused_kda_decode_cuda_graph(num_heads, lower_bound, state_dtype):
    inputs = _make_inputs(num_heads=num_heads, num_rows=4, state_dtype=state_dtype)
    output = torch.empty(
        (1, 4, num_heads, 128), dtype=torch.bfloat16, device=torch.device("cuda")
    )
    kwargs = {**inputs, "lower_bound": lower_bound, "output": output}

    # Compile and allocate before capture. Replays then contain only the fused
    # kernel and preserve the in-place cache update semantics.
    fused_kda_decode(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = fused_kda_decode(**kwargs)
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_fused_kda_decode_null_slots(state_dtype):
    inputs = _make_inputs(num_heads=12, num_rows=3, state_dtype=state_dtype)
    inputs["state_indices"].copy_(
        torch.tensor([0, -1, 1], dtype=torch.int32, device=torch.device("cuda"))
    )
    conv_state_before = _clone_strided(inputs["conv_state"])
    state_before = _clone_strided(inputs["state"])

    output = fused_kda_decode(**inputs)

    torch.testing.assert_close(output[:, :2], torch.zeros_like(output[:, :2]))
    torch.testing.assert_close(inputs["conv_state"][0], conv_state_before[0])
    torch.testing.assert_close(inputs["state"][0], state_before[0])
    torch.testing.assert_close(inputs["conv_state"][2:], conv_state_before[2:])
    torch.testing.assert_close(inputs["state"][2:], state_before[2:])
    assert not torch.equal(inputs["conv_state"][1], conv_state_before[1])
    assert not torch.equal(inputs["state"][1], state_before[1])


def test_fused_kda_decode_requires_matching_cache_slots():
    inputs = _make_inputs(num_heads=12, num_rows=3)
    inputs["state"] = inputs["state"][:-1]

    with pytest.raises(
        ValueError,
        match="conv_state and state must have the same number of cache slots",
    ):
        fused_kda_decode(**inputs)


def test_fused_kda_decode_cache_key_distinguishes_state_and_gate(monkeypatch):
    import importlib

    kernel_module = importlib.import_module("flashinfer.kda_kernels.fused_kda_decode")

    kernel_names = []

    def record_kernel_name(module_name, kernel_name, compile_fn, extra_key_files):
        del module_name, compile_fn, extra_key_files
        kernel_names.append(kernel_name)
        return kernel_name

    monkeypatch.setattr(
        kernel_module, "build_and_load_cute_dsl_kernel", record_kernel_name
    )
    kernel_module._get_compiled_kernel.cache_clear()
    try:
        for state_dtype in (torch.float32, torch.bfloat16):
            for lower_bound in (-5.0, None):
                kernel_module._get_compiled_kernel(state_dtype, lower_bound, 1e-5)
    finally:
        kernel_module._get_compiled_kernel.cache_clear()

    assert len(kernel_names) == len(set(kernel_names)) == 4
