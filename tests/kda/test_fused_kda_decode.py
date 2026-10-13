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
        packed_fused_kda_decode,
    )

    _has_fused_kda_decode = _FUSED_KDA_DECODE_AVAILABLE
except ImportError:
    fused_kda_decode = None
    packed_fused_kda_decode = None
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


def _page_strides(num_heads, state_dtype, conv_history=3):
    conv_slot_bytes = 3 * num_heads * 128 * conv_history * 2
    state_element_bytes = torch.empty((), dtype=state_dtype).element_size()
    state_slot_bytes = num_heads * 128 * 128 * state_element_bytes
    page_bytes = conv_slot_bytes + state_slot_bytes
    return page_bytes // 2, page_bytes // state_element_bytes


def _make_inputs(
    num_heads, num_rows, seed=42, state_dtype=torch.float32, conv_history=3
):
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

    conv_slot_stride, state_slot_stride = _page_strides(
        num_heads, state_dtype, conv_history
    )
    conv_state = torch.empty_strided(
        (num_slots, 3 * hidden_size, conv_history),
        (conv_slot_stride, 1, 3 * hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state.copy_(
        0.1 * randn((num_slots, 3 * hidden_size, conv_history), torch.bfloat16)
    )
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


def _fused_kda_decode(**kwargs):
    assert fused_kda_decode is not None
    return fused_kda_decode(**kwargs)


def _packed_fused_kda_decode(**kwargs):
    assert packed_fused_kda_decode is not None
    return packed_fused_kda_decode(**kwargs)


def _clone_strided(tensor):
    clone = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    clone.copy_(tensor)
    return clone


@torch.no_grad()
def _reference(inputs, conv_state, state, lower_bound: float | None = -5.0):
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


@torch.no_grad()
def _packed_reference(inputs, conv_state, state):
    indices = inputs["state_indices"]
    query_start_loc = inputs["query_start_loc"].cpu().tolist()
    accepted_tokens = inputs["num_accepted_tokens"].cpu().tolist()
    num_sequences, max_tokens = indices.shape
    num_heads = inputs["A_log"].numel()
    hidden_size = num_heads * 128
    taps = inputs["weight"].permute(0, 2, 1).reshape(3 * hidden_size, 4).float()
    output = torch.zeros(
        (1, inputs["x"].shape[0], num_heads, 128),
        dtype=torch.bfloat16,
        device=state.device,
    )
    decay_a = inputs["A_log"].exp().view(num_heads, 1)

    for sequence in range(num_sequences):
        begin, end = query_start_loc[sequence : sequence + 2]
        sequence_length = end - begin
        conv_slot = int(indices[sequence, 0])
        if sequence_length <= 0 or conv_slot <= 0:
            continue
        accepted = accepted_tokens[sequence] - 1
        state_slot = int(indices[sequence, accepted])
        history = conv_state[conv_slot, :, accepted : accepted + 3].float()
        current_state = state[state_slot].float().clone()
        conv_state[conv_slot, :, 0] = history[:, 1].to(torch.bfloat16)
        conv_state[conv_slot, :, 1] = history[:, 2].to(torch.bfloat16)

        for token in range(sequence_length):
            row = begin + token
            current_x = inputs["x"][row].float()
            window = torch.cat((history, current_x.unsqueeze(-1)), dim=-1)
            history = window[:, 1:]
            conv_state[conv_slot, :, token + 2] = current_x.to(torch.bfloat16)

            mixed = F.silu((window * taps).sum(-1)).to(torch.bfloat16)
            query, key, value = mixed.float().view(3, num_heads, 128).unbind(0)
            query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
            query *= 128**-0.5
            key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            raw_gate = inputs["raw_gate"][0, row].float()
            raw_gate += inputs["dt_bias"].view(num_heads, 128)
            decay = torch.exp(-5.0 * torch.sigmoid(decay_a * raw_gate))
            current_state *= decay.unsqueeze(-2)
            state_key = torch.einsum("hvk,hk->hv", current_state, key)
            delta = value - state_key
            beta = inputs["raw_beta"][0, row].float().sigmoid()
            current_state += (delta * beta.unsqueeze(-1)).unsqueeze(-1) * key.unsqueeze(
                -2
            )
            destination = int(indices[sequence, token])
            if destination > 0:
                state[destination].copy_(current_state)
            recurrent = torch.einsum("hvk,hk->hv", current_state, query)
            output[0, row] = recurrent.to(torch.bfloat16)

    output_float = output.float()
    inverse_rms = torch.rsqrt(output_float.square().mean(-1, keepdim=True) + 1e-5)
    return (
        output_float
        * inverse_rms
        * inputs["norm_weight"]
        * inputs["output_gate"].float().sigmoid().unsqueeze(0)
    ).to(torch.bfloat16)


def _make_packed_inputs(
    num_heads, num_sequences, num_tokens, seed=42, state_dtype=torch.float32
):
    inputs = _make_inputs(
        num_heads,
        num_sequences * num_tokens,
        seed=seed,
        state_dtype=state_dtype,
        conv_history=num_tokens + 2,
    )
    inputs["state_indices"] = inputs["state_indices"].reshape(num_sequences, num_tokens)
    inputs["query_start_loc"] = torch.arange(
        0,
        num_sequences * num_tokens + 1,
        num_tokens,
        dtype=torch.int32,
        device=torch.device("cuda"),
    )
    inputs["num_accepted_tokens"] = torch.ones(
        num_sequences, dtype=torch.int32, device=torch.device("cuda")
    )
    return inputs


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
    actual = _fused_kda_decode(
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
    actual = _fused_kda_decode(**kwargs)

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
    _fused_kda_decode(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _fused_kda_decode(**kwargs)
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

    output = _fused_kda_decode(**inputs)

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
        _fused_kda_decode(**inputs)


def test_fused_kda_decode_compile_abi_orders_flags_and_metadata(monkeypatch):
    import importlib

    module = importlib.import_module("flashinfer.kda_kernels.fused_kda_decode")
    compiled = []

    monkeypatch.setattr(module, "_make_compile_inputs", lambda dtype: (dtype,))

    def fake_compile(function, *args, **kwargs):
        del function, kwargs
        compiled.append(args)
        return "compiled"

    def fake_build(module_name, kernel_name, compile_fn, extra_key_files):
        del module_name, kernel_name, extra_key_files
        compile_fn()
        return "loaded"

    monkeypatch.setattr(module.cute, "compile", fake_compile)
    monkeypatch.setattr(module, "build_and_load_cute_dsl_kernel", fake_build)
    module._get_compiled_kernel.cache_clear()
    try:
        for state_dtype in (torch.float32, torch.bfloat16):
            for packed_t1 in (False, True):
                module._get_compiled_kernel(
                    state_dtype, -5.0, 1e-5, packed_t1=packed_t1
                )
    finally:
        module._get_compiled_kernel.cache_clear()

    assert len(compiled) == 4
    # _make_compile_inputs contributes one sentinel; constexpr flags follow
    # the launch signature: state_is_bf16, packed_t1.
    assert {(args[-5], args[-4]) for args in compiled} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


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


@pytest.mark.parametrize(
    ("num_heads", "num_sequences", "num_tokens"),
    [
        pytest.param(12, 1, 1, id="h12-n1-t1"),
        pytest.param(12, 2, 1, id="h12-n2-t1-dynamic-batch"),
        pytest.param(12, 2, 2, id="h12-n2-t2"),
        pytest.param(24, 4, 3, id="h24-n4-t3"),
        pytest.param(32, 2, 4, id="h32-n2-t4"),
        pytest.param(48, 2, 8, id="h48-n2-t8"),
        pytest.param(96, 1, 3, id="h96-n1-t3"),
        pytest.param(12, 32, 5, id="h12-n32-t5-wide"),
    ],
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_packed_fused_kda_decode(num_heads, num_sequences, num_tokens, state_dtype):
    inputs = _make_packed_inputs(
        num_heads, num_sequences, num_tokens, state_dtype=state_dtype
    )
    reference_conv = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    actual_conv = _clone_strided(inputs["conv_state"])
    actual_state = _clone_strided(inputs["state"])

    expected = _packed_reference(inputs, reference_conv, reference_state)
    actual = _packed_fused_kda_decode(
        **{**inputs, "conv_state": actual_conv, "state": actual_state}
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(actual_conv, reference_conv, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)


def test_packed_fused_kda_decode_t1_pre_resolved_matches_direct_and_graph_replay():
    inputs = _make_inputs(12, 4, state_dtype=torch.float32)
    inputs["state_indices"] = torch.tensor(
        [[1], [2], [3], [4]], dtype=torch.int32, device="cuda"
    )
    inputs["query_start_loc"] = torch.tensor(
        [0, 1, 2, 2, 2], dtype=torch.int32, device="cuda"
    )
    inputs["num_accepted_tokens"] = torch.ones(4, dtype=torch.int32, device="cuda")
    packed_slots = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device="cuda")

    def clone_inputs():
        return {
            key: _clone_strided(value) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    direct_inputs = clone_inputs()
    direct_inputs.pop("query_start_loc")
    direct_inputs.pop("num_accepted_tokens")
    direct_inputs["state_indices"] = packed_slots.clone()
    direct = _fused_kda_decode(**direct_inputs)
    packed_inputs = clone_inputs()
    packed_output = torch.empty_like(direct)
    packed = _packed_fused_kda_decode(
        **packed_inputs,
        t1_state_indices=packed_slots,
        output=packed_output,
    )
    assert torch.equal(packed, direct)
    assert torch.equal(packed_inputs["conv_state"], direct_inputs["conv_state"])
    assert torch.equal(packed_inputs["state"], direct_inputs["state"])
    assert torch.equal(packed[0, 2:], torch.zeros_like(packed[0, 2:]))

    packed_conv_seed = _clone_strided(inputs["conv_state"])
    packed_state_seed = _clone_strided(inputs["state"])
    graph_output = torch.empty_like(direct)
    graph_inputs = clone_inputs()
    graph_inputs["conv_state"].copy_(packed_conv_seed)
    graph_inputs["state"].copy_(packed_state_seed)
    # Capture through a closure so the caller-owned slot tensor is read at
    # replay time, exactly as it is in the serving CUDA-graph path.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_fused_kda_decode(
            **graph_inputs,
            t1_state_indices=packed_slots,
            output=graph_output,
        )
    packed_slots.copy_(torch.tensor([2, 1, 0, 0], device="cuda"))
    graph_inputs["conv_state"].copy_(packed_conv_seed)
    graph_inputs["state"].copy_(packed_state_seed)
    graph.replay()
    torch.cuda.synchronize()

    expected_inputs = clone_inputs()
    expected_inputs.pop("query_start_loc")
    expected_inputs.pop("num_accepted_tokens")
    expected_inputs["state_indices"] = packed_slots.clone()
    expected = _fused_kda_decode(**expected_inputs)
    assert torch.equal(graph_output, expected)
    assert torch.equal(graph_inputs["conv_state"], expected_inputs["conv_state"])
    assert torch.equal(graph_inputs["state"], expected_inputs["state"])


def test_packed_fused_kda_decode_t1_uses_legacy_modes():
    inputs = _make_inputs(12, 2, state_dtype=torch.bfloat16)
    inputs["state_indices"] = inputs["state_indices"].reshape(2, 1)
    inputs["query_start_loc"] = torch.arange(
        3, dtype=torch.int32, device=torch.device("cuda")
    )
    inputs["num_accepted_tokens"] = torch.ones(
        2, dtype=torch.int32, device=torch.device("cuda")
    )
    legacy_conv = _clone_strided(inputs["conv_state"])
    legacy_state = _clone_strided(inputs["state"])
    packed_conv = _clone_strided(inputs["conv_state"])
    packed_state = _clone_strided(inputs["state"])

    legacy_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in ("query_start_loc", "num_accepted_tokens")
    }
    legacy = _fused_kda_decode(
        **{
            **legacy_inputs,
            "conv_state": legacy_conv,
            "state": legacy_state,
            "state_indices": inputs["state_indices"][:, 0],
            "lower_bound": None,
        }
    )
    packed = _packed_fused_kda_decode(
        **{
            **inputs,
            "conv_state": packed_conv,
            "state": packed_state,
            "lower_bound": None,
        }
    )

    assert torch.equal(packed, legacy)
    assert torch.equal(packed_conv, legacy_conv)
    assert torch.equal(packed_state, legacy_state)


@pytest.mark.parametrize(
    "query_start_loc",
    [
        pytest.param([0, 0, 1, 2], id="empty-first"),
        pytest.param([0, 1, 1, 2], id="empty-middle"),
        pytest.param([0, 1, 2, 2], id="empty-last"),
    ],
)
def test_packed_fused_kda_decode_t1_empty_sequence(query_start_loc):
    inputs = _make_inputs(12, 3, state_dtype=torch.bfloat16)
    for key in ("x", "raw_gate", "raw_beta", "output_gate"):
        row_dimension = 1 if key in ("raw_gate", "raw_beta") else 0
        inputs[key] = inputs[key].narrow(row_dimension, 0, 2)
    inputs["state_indices"] = torch.arange(
        1, 4, dtype=torch.int32, device=torch.device("cuda")
    ).reshape(3, 1)
    inputs["query_start_loc"] = torch.tensor(
        query_start_loc, dtype=torch.int32, device=torch.device("cuda")
    )
    inputs["num_accepted_tokens"] = torch.ones(
        3, dtype=torch.int32, device=torch.device("cuda")
    )
    live_sequences = [
        sequence
        for sequence in range(3)
        if query_start_loc[sequence] != query_start_loc[sequence + 1]
    ]
    legacy_indices = inputs["state_indices"][live_sequences, 0].contiguous()
    legacy_conv = _clone_strided(inputs["conv_state"])
    legacy_state = _clone_strided(inputs["state"])
    packed_conv = _clone_strided(inputs["conv_state"])
    packed_state = _clone_strided(inputs["state"])

    legacy = _fused_kda_decode(
        **{
            **{
                key: value
                for key, value in inputs.items()
                if key
                not in (
                    "query_start_loc",
                    "num_accepted_tokens",
                    "state_indices",
                )
            },
            "conv_state": legacy_conv,
            "state": legacy_state,
            "state_indices": legacy_indices,
            "lower_bound": None,
        }
    )
    packed_output = torch.empty_like(legacy)
    packed_kwargs = {
        **inputs,
        "conv_state": packed_conv,
        "state": packed_state,
        "lower_bound": None,
        "output": packed_output,
    }
    _packed_fused_kda_decode(**packed_kwargs)
    packed_conv.copy_(inputs["conv_state"])
    packed_state.copy_(inputs["state"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_fused_kda_decode(**packed_kwargs)
    packed_conv.copy_(inputs["conv_state"])
    packed_state.copy_(inputs["state"])
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(packed_output, legacy)
    assert torch.equal(packed_conv, legacy_conv)
    assert torch.equal(packed_state, legacy_state)


@pytest.mark.parametrize(
    ("state_dtype", "lower_bound", "error", "match"),
    [
        pytest.param(
            torch.float16,
            -5.0,
            TypeError,
            "state must have dtype torch.float32 or torch.bfloat16",
            id="fp16-state",
        ),
        pytest.param(
            torch.float32,
            None,
            ValueError,
            "lower_bound must be a finite negative float",
            id="softplus",
        ),
    ],
)
def test_packed_fused_kda_decode_t2_rejects_legacy_only_modes(
    state_dtype, lower_bound, error, match
):
    inputs = _make_inputs(12, 2, state_dtype=state_dtype, conv_history=4)
    inputs["state_indices"] = inputs["state_indices"].reshape(1, 2)
    inputs["query_start_loc"] = torch.tensor(
        [0, 2], dtype=torch.int32, device=torch.device("cuda")
    )
    inputs["num_accepted_tokens"] = torch.ones(
        1, dtype=torch.int32, device=torch.device("cuda")
    )

    with pytest.raises(error, match=match):
        _packed_fused_kda_decode(**inputs, lower_bound=lower_bound)


def test_packed_fused_kda_decode_rejects_unsupported_arch(monkeypatch):
    import importlib

    module = importlib.import_module(
        "flashinfer.kda_kernels.fused_kda_decode_multitoken"
    )
    inputs = _make_packed_inputs(12, 1, 2)
    monkeypatch.setattr(
        module.torch.cuda, "get_device_capability", lambda _device: (9, 0)
    )

    with pytest.raises(NotImplementedError, match="requires SM10x"):
        _packed_fused_kda_decode(**inputs)


def test_packed_fused_kda_decode_shared_memory_limit():
    import importlib

    module = importlib.import_module(
        "flashinfer.kda_kernels.fused_kda_decode_multitoken"
    )
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    supported = 1
    while (
        module._required_smem_bytes(
            supported + 1, *module._TILE_TALL[:2], module._TILE_TALL[3]
        )
        <= limit
    ):
        supported += 1

    assert supported >= 96
    assert (
        module._required_smem_bytes(
            supported, *module._TILE_TALL[:2], module._TILE_TALL[3]
        )
        <= limit
    )
    assert (
        module._required_smem_bytes(
            supported + 1, *module._TILE_TALL[:2], module._TILE_TALL[3]
        )
        > limit
    )


def test_packed_fused_kda_decode_split_tiles_load_frontend_in_one_pass():
    import importlib

    module = importlib.import_module(
        "flashinfer.kda_kernels.fused_kda_decode_multitoken"
    )
    for split, (rows, k_lanes, _, chunks) in module._TILE_SPLIT.items():
        threads = (((128 // split) // chunks) // rows) * k_lanes
        assert threads == 4 * 128


def test_packed_fused_kda_decode_null_row():
    inputs = _make_packed_inputs(num_heads=12, num_sequences=2, num_tokens=3)
    inputs["state_indices"][0].zero_()
    reference_conv = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    actual_conv = _clone_strided(inputs["conv_state"])
    actual_state = _clone_strided(inputs["state"])

    expected = _packed_reference(inputs, reference_conv, reference_state)
    actual = _packed_fused_kda_decode(
        **{**inputs, "conv_state": actual_conv, "state": actual_state}
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    assert torch.count_nonzero(actual[:, :3]) == 0
    torch.testing.assert_close(actual_conv, reference_conv, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)


def test_packed_fused_kda_decode_ragged_rows_and_padding():
    inputs = _make_packed_inputs(num_heads=12, num_sequences=3, num_tokens=4)
    inputs["query_start_loc"].copy_(
        torch.tensor([0, 4, 6, 6], dtype=torch.int32, device=torch.device("cuda"))
    )
    inputs["num_accepted_tokens"].copy_(
        torch.tensor([1, 2, 1], dtype=torch.int32, device=torch.device("cuda"))
    )
    reference_conv = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    actual_conv = _clone_strided(inputs["conv_state"])
    actual_state = _clone_strided(inputs["state"])
    output = torch.full(
        (1, inputs["x"].shape[0], 12, 128),
        torch.nan,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    expected = _packed_reference(inputs, reference_conv, reference_state)
    actual = _packed_fused_kda_decode(
        **{
            **inputs,
            "conv_state": actual_conv,
            "state": actual_state,
            "output": output,
        }
    )

    torch.testing.assert_close(actual[:, :6], expected[:, :6], rtol=3e-2, atol=2e-2)
    assert torch.isnan(actual[:, 6:]).all()
    torch.testing.assert_close(actual_conv, reference_conv, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)


@pytest.mark.parametrize("num_tokens", [3, 7, 8])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_packed_fused_kda_decode_two_verification_windows(num_tokens, state_dtype):
    first = _make_packed_inputs(12, 2, num_tokens, seed=41, state_dtype=state_dtype)
    reference_conv = _clone_strided(first["conv_state"])
    reference_state = _clone_strided(first["state"])
    actual_conv = _clone_strided(first["conv_state"])
    actual_state = _clone_strided(first["state"])

    expected_first = _packed_reference(first, reference_conv, reference_state)
    actual_first = _packed_fused_kda_decode(
        **{**first, "conv_state": actual_conv, "state": actual_state}
    )
    torch.testing.assert_close(actual_conv, reference_conv, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)
    torch.testing.assert_close(actual_first, expected_first, rtol=3e-2, atol=2e-2)

    second = _make_packed_inputs(12, 2, num_tokens, seed=83, state_dtype=state_dtype)
    second["state_indices"].copy_(first["state_indices"])
    second["num_accepted_tokens"].copy_(
        torch.tensor([2, num_tokens], dtype=torch.int32, device=torch.device("cuda"))
    )
    expected_second = _packed_reference(second, reference_conv, reference_state)
    actual_second = _packed_fused_kda_decode(
        **{**second, "conv_state": actual_conv, "state": actual_state}
    )

    torch.testing.assert_close(actual_second, expected_second, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(actual_conv, reference_conv, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)


def test_packed_fused_kda_decode_all_accepted_offsets():
    num_tokens = 8
    num_sequences = 32
    first = _make_packed_inputs(12, num_sequences, num_tokens, seed=107)
    reference_conv = _clone_strided(first["conv_state"])
    reference_state = _clone_strided(first["state"])
    actual_conv = _clone_strided(first["conv_state"])
    actual_state = _clone_strided(first["state"])

    _packed_reference(first, reference_conv, reference_state)
    _packed_fused_kda_decode(
        **{**first, "conv_state": actual_conv, "state": actual_state}
    )

    second = _make_packed_inputs(12, num_sequences, num_tokens, seed=109)
    second["state_indices"].copy_(first["state_indices"])
    second["num_accepted_tokens"].copy_(
        torch.arange(num_sequences, dtype=torch.int32, device=torch.device("cuda"))
        % num_tokens
        + 1
    )
    expected = _packed_reference(second, reference_conv, reference_state)
    actual = _packed_fused_kda_decode(
        **{**second, "conv_state": actual_conv, "state": actual_state}
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(actual_conv, reference_conv, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)


@pytest.mark.parametrize(("num_sequences", "num_tokens"), [(1, 7), (2, 8)])
def test_packed_fused_kda_decode_cluster_determinism(num_sequences, num_tokens):
    inputs = _make_packed_inputs(12, num_sequences, num_tokens, seed=127)
    reference_conv = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    expected = _packed_reference(inputs, reference_conv, reference_state)
    first = None

    for _ in range(8):
        actual_conv = _clone_strided(inputs["conv_state"])
        actual_state = _clone_strided(inputs["state"])
        actual = _packed_fused_kda_decode(
            **{**inputs, "conv_state": actual_conv, "state": actual_state}
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
        torch.testing.assert_close(actual_conv, reference_conv, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=2e-3)
        if first is None:
            first = (actual.clone(), actual_conv.clone(), actual_state.clone())
        else:
            assert torch.equal(actual, first[0])
            assert torch.equal(actual_conv, first[1])
            assert torch.equal(actual_state, first[2])


def test_packed_fused_kda_decode_cuda_graph():
    inputs = _make_packed_inputs(num_heads=12, num_sequences=2, num_tokens=3)
    expected_conv = _clone_strided(inputs["conv_state"])
    expected_state = _clone_strided(inputs["state"])
    expected = _packed_reference(inputs, expected_conv, expected_state)
    output = torch.empty_like(expected)
    kwargs = {**inputs, "output": output}

    _packed_fused_kda_decode(**kwargs)
    # Restore the exact seeded inputs rather than relying on capture-time state.
    fresh = _make_packed_inputs(12, 2, 3)
    inputs["conv_state"].copy_(fresh["conv_state"])
    inputs["state"].copy_(fresh["state"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_fused_kda_decode(**kwargs)
    inputs["conv_state"].copy_(fresh["conv_state"])
    inputs["state"].copy_(fresh["state"])
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(inputs["state"], expected_state, rtol=3e-2, atol=2e-3)


def _t1_inputs(num_heads, state_dtype):
    inputs = _make_inputs(num_heads, 4, state_dtype=state_dtype)
    # Capture a fixed three-row graph while retaining five total cache slots 0..4; four are positive.
    for key in ("x", "raw_gate", "raw_beta", "output_gate"):
        dimension = 1 if key in ("raw_gate", "raw_beta") else 0
        inputs[key] = inputs[key].narrow(dimension, 0, 3)
    inputs["state_indices"] = torch.empty((3, 1), dtype=torch.int32, device="cuda")
    inputs["query_start_loc"] = torch.empty(4, dtype=torch.int32, device="cuda")
    inputs["num_accepted_tokens"] = torch.ones(3, dtype=torch.int32, device="cuda")
    return inputs


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_packed_fused_kda_decode_t1_graph_replay_remaps_rows(state_dtype, lower_bound):
    inputs = _t1_inputs(12, state_dtype)
    original_conv = _clone_strided(inputs["conv_state"])
    original_state = _clone_strided(inputs["state"])
    inputs["query_start_loc"].copy_(torch.tensor([0, 1, 2, 3], device="cuda"))
    inputs["state_indices"].copy_(torch.tensor([[1], [2], [3]], device="cuda"))
    output = torch.empty((1, 3, 12, 128), dtype=torch.bfloat16, device="cuda")
    kwargs = {**inputs, "output": output, "lower_bound": lower_bound}

    _packed_fused_kda_decode(**kwargs)
    inputs["conv_state"].copy_(original_conv)
    inputs["state"].copy_(original_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_fused_kda_decode(**kwargs)

    trials = [
        ([0, 1, 2, 3], [1, 2, 3], [1, 2, 3]),
        ([0, 1, 1, 2], [4, 2, 1], [4, 1]),
        ([0, 1, 2, 3], [3, 4, 2], [3, 4, 2]),
    ]
    for qsl_values, slots, expected_slots in trials:
        assert len(qsl_values) == 4 and max(qsl_values) <= 3
        assert all(0 < slot < inputs["state"].shape[0] for slot in slots)
        inputs["conv_state"].copy_(original_conv)
        inputs["state"].copy_(original_state)
        inputs["query_start_loc"].copy_(torch.tensor(qsl_values, device="cuda"))
        inputs["state_indices"].copy_(torch.tensor(slots, device="cuda").view(3, 1))
        output.fill_(torch.nan)

        live = len(expected_slots)
        oracle = {
            **inputs,
            "x": inputs["x"][:live],
            "raw_gate": inputs["raw_gate"][:, :live],
            "raw_beta": inputs["raw_beta"][:, :live],
            "output_gate": inputs["output_gate"][:live],
            "state_indices": torch.tensor(expected_slots, device="cuda"),
        }
        expected_conv = _clone_strided(original_conv)
        expected_state = _clone_strided(original_state)
        expected = _reference(oracle, expected_conv, expected_state, lower_bound)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output[:, :live], expected, rtol=3e-2, atol=2e-2)
        assert torch.equal(output[:, live:], torch.zeros_like(output[:, live:]))
        torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(
            inputs["state"], expected_state, rtol=3e-2, atol=2e-3
        )
        for slot in range(inputs["state"].shape[0]):
            if slot not in expected_slots:
                assert torch.equal(inputs["state"][slot], original_state[slot])
                assert torch.equal(inputs["conv_state"][slot], original_conv[slot])


@pytest.mark.parametrize("num_rows", [129, 257])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_packed_fused_kda_decode_t1_tail_bound_and_dual_role(
    num_rows, state_dtype, lower_bound
):
    # role 2 is both the live sequence CTA for row 0 and a tail CTA for
    # output row 2.  The independent literal oracle catches an unconditional
    # tail return; the large capacities catch using x.shape[0] (the vector
    # lane extent) instead of the transformed output row extent.
    inputs = _make_inputs(12, num_rows, state_dtype=state_dtype)
    inputs["state_indices"] = torch.tensor(
        [[0], [0], [7]], dtype=torch.int32, device="cuda"
    )
    inputs["query_start_loc"] = torch.tensor(
        [0, 0, 0, 1], dtype=torch.int32, device="cuda"
    )
    inputs["num_accepted_tokens"] = torch.ones(3, dtype=torch.int32, device="cuda")
    before_conv = _clone_strided(inputs["conv_state"])
    before_state = _clone_strided(inputs["state"])
    output = torch.full(
        (1, num_rows, 12, 128), torch.nan, dtype=torch.bfloat16, device="cuda"
    )
    actual = _packed_fused_kda_decode(
        **{**inputs, "output": output, "lower_bound": lower_bound}
    )

    # Only sequence 2 is live, and it owns packed row 0.  Every other output
    # row is padding and must be exactly zero, including the rows beyond 4.
    oracle = {
        **inputs,
        "x": inputs["x"][:1],
        "raw_gate": inputs["raw_gate"][:, :1],
        "raw_beta": inputs["raw_beta"][:, :1],
        "output_gate": inputs["output_gate"][:1],
        "state_indices": torch.tensor([7], dtype=torch.int32, device="cuda"),
    }
    expected_conv = _clone_strided(before_conv)
    expected_state = _clone_strided(before_state)
    expected = _reference(oracle, expected_conv, expected_state, lower_bound)
    torch.testing.assert_close(actual[:, :1], expected, rtol=3e-2, atol=2e-2)
    assert torch.equal(actual[:, 1:], torch.zeros_like(actual[:, 1:]))
    torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(inputs["state"], expected_state, rtol=3e-2, atol=2e-3)
    for slot in range(inputs["state"].shape[0]):
        if slot != 7:
            assert torch.equal(inputs["state"][slot], before_state[slot])
            assert torch.equal(inputs["conv_state"][slot], before_conv[slot])


def test_packed_fused_kda_decode_t1_all_empty_preserves_positive_slots():
    inputs = _t1_inputs(12, torch.bfloat16)
    before_conv = _clone_strided(inputs["conv_state"])
    before_state = _clone_strided(inputs["state"])
    inputs["query_start_loc"].copy_(torch.tensor([0, 0, 0, 0], device="cuda"))
    inputs["state_indices"].copy_(torch.tensor([[1], [2], [3]], device="cuda"))
    output = _packed_fused_kda_decode(**inputs)
    assert torch.equal(output, torch.zeros_like(output))
    torch.testing.assert_close(inputs["conv_state"], before_conv, rtol=0, atol=0)
    torch.testing.assert_close(inputs["state"], before_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("num_sequences", "qsl_values", "slots", "expected_slots"),
    [
        (4, [0, 1, 2, 2, 2], [1, 2, 3, 4], [1, 2]),
        (3, [0, 1, 1, 2], [4, 2, 1], [4, 1]),
        (4, [0, 0, 0, 0, 0], [1, 2, 3, 4], []),
    ],
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_packed_fused_kda_decode_t1_eager_valid_padding(
    num_sequences, qsl_values, slots, expected_slots, state_dtype, lower_bound
):
    inputs = _make_inputs(12, 4, state_dtype=state_dtype)
    inputs["state_indices"] = torch.tensor(
        slots, dtype=torch.int32, device="cuda"
    ).view(num_sequences, 1)
    inputs["query_start_loc"] = torch.tensor(
        qsl_values, dtype=torch.int32, device="cuda"
    )
    inputs["num_accepted_tokens"] = torch.ones(
        num_sequences, dtype=torch.int32, device="cuda"
    )
    before_conv = _clone_strided(inputs["conv_state"])
    before_state = _clone_strided(inputs["state"])
    output = torch.full((1, 4, 12, 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    output = _packed_fused_kda_decode(
        **{**inputs, "output": output, "lower_bound": lower_bound}
    )
    assert all(0 < slot < inputs["state"].shape[0] for slot in slots)
    assert len(expected_slots) == inputs["query_start_loc"][-1].item()
    if expected_slots:
        oracle = {
            **inputs,
            "x": inputs["x"][: len(expected_slots)],
            "raw_gate": inputs["raw_gate"][:, : len(expected_slots)],
            "raw_beta": inputs["raw_beta"][:, : len(expected_slots)],
            "output_gate": inputs["output_gate"][: len(expected_slots)],
            "state_indices": torch.tensor(expected_slots, device="cuda"),
        }
        expected_conv = _clone_strided(before_conv)
        expected_state = _clone_strided(before_state)
        expected = _reference(oracle, expected_conv, expected_state, lower_bound)
        torch.testing.assert_close(
            output[:, : len(expected_slots)], expected, rtol=3e-2, atol=2e-2
        )
        torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(
            inputs["state"], expected_state, rtol=3e-2, atol=2e-3
        )
    else:
        assert torch.equal(output, torch.zeros_like(output))
        assert torch.equal(inputs["conv_state"], before_conv)
        assert torch.equal(inputs["state"], before_state)
    assert torch.equal(
        output[:, len(expected_slots) :],
        torch.zeros_like(output[:, len(expected_slots) :]),
    )
    for slot in range(inputs["state"].shape[0]):
        if slot not in expected_slots:
            assert torch.equal(inputs["state"][slot], before_state[slot])
            assert torch.equal(inputs["conv_state"][slot], before_conv[slot])


def test_packed_fused_kda_decode_cuda_graph_dynamic_metadata():
    inputs = _make_packed_inputs(num_heads=12, num_sequences=3, num_tokens=4)
    seeded_conv = _clone_strided(inputs["conv_state"])
    seeded_state = _clone_strided(inputs["state"])
    output = torch.full(
        (1, inputs["x"].shape[0], 12, 128),
        torch.nan,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    kwargs = {**inputs, "output": output}

    _packed_fused_kda_decode(**kwargs)
    inputs["conv_state"].copy_(seeded_conv)
    inputs["state"].copy_(seeded_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_fused_kda_decode(**kwargs)

    inputs["conv_state"].copy_(seeded_conv)
    inputs["state"].copy_(seeded_state)
    output.fill_(torch.nan)
    inputs["query_start_loc"].copy_(
        torch.tensor([0, 4, 6, 6], dtype=torch.int32, device=torch.device("cuda"))
    )
    inputs["num_accepted_tokens"].copy_(
        torch.tensor([1, 2, 1], dtype=torch.int32, device=torch.device("cuda"))
    )
    expected_conv = _clone_strided(seeded_conv)
    expected_state = _clone_strided(seeded_state)
    expected = _packed_reference(inputs, expected_conv, expected_state)

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output[:, :6], expected[:, :6], rtol=3e-2, atol=2e-2)
    assert torch.isnan(output[:, 6:]).all()
    torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(inputs["state"], expected_state, rtol=3e-2, atol=2e-3)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_packed_fused_kda_decode_t1_trailing_empty_sequences_beyond_rows(
    state_dtype, lower_bound
):
    # The live row is valid, while three trailing sequences have qsl==rows.
    # This specifically guards the old sequence-CTA input load at row_idx=qsl[s]
    # when N > num_rows and total_tokens == num_rows.
    inputs = _make_inputs(12, 4, state_dtype=state_dtype)
    for key in ("x", "raw_gate", "raw_beta", "output_gate"):
        dimension = 1 if key in ("raw_gate", "raw_beta") else 0
        inputs[key] = inputs[key].narrow(dimension, 0, 1)
    inputs["query_start_loc"] = torch.tensor(
        [0, 1, 1, 1, 1], dtype=torch.int32, device="cuda"
    )
    inputs["num_accepted_tokens"] = torch.ones(4, dtype=torch.int32, device="cuda")
    inputs["state_indices"] = torch.tensor(
        [[1], [2], [3], [4]], dtype=torch.int32, device="cuda"
    )
    before_conv = _clone_strided(inputs["conv_state"])
    before_state = _clone_strided(inputs["state"])
    output = torch.full((1, 1, 12, 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    actual = _packed_fused_kda_decode(
        **{**inputs, "output": output, "lower_bound": lower_bound}
    )

    oracle = {
        **inputs,
        "state_indices": torch.tensor([1], dtype=torch.int32, device="cuda"),
    }
    expected_conv = _clone_strided(before_conv)
    expected_state = _clone_strided(before_state)
    expected = _reference(oracle, expected_conv, expected_state, lower_bound)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(inputs["state"], expected_state, rtol=3e-2, atol=2e-3)
    for slot in (0, 2, 3, 4):
        assert torch.equal(inputs["state"][slot], before_state[slot])
        assert torch.equal(inputs["conv_state"][slot], before_conv[slot])


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("live_slot", [0, -1, 4])
def test_packed_fused_kda_decode_t1_late_active_dual_role_null_slots(
    state_dtype, lower_bound, live_slot
):
    # Empty sequences precede the live sequence. Null slots must not touch
    # caches; a positive live slot still owns packed row zero.
    inputs = _make_inputs(12, 4, state_dtype=state_dtype)
    for key in ("x", "raw_gate", "raw_beta", "output_gate"):
        dimension = 1 if key in ("raw_gate", "raw_beta") else 0
        inputs[key] = inputs[key].narrow(dimension, 0, 3)
    inputs["query_start_loc"] = torch.tensor(
        [0, 0, 0, 1], dtype=torch.int32, device="cuda"
    )
    inputs["num_accepted_tokens"] = torch.ones(3, dtype=torch.int32, device="cuda")
    inputs["state_indices"] = torch.tensor(
        [[0], [-1], [live_slot]], dtype=torch.int32, device="cuda"
    )
    assert inputs["x"].shape[0] == 3
    assert inputs["query_start_loc"].tolist() == [0, 0, 0, 1]
    assert -1 <= live_slot < inputs["state"].shape[0]
    before_conv = _clone_strided(inputs["conv_state"])
    before_state = _clone_strided(inputs["state"])
    output = torch.full((1, 3, 12, 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    actual = _packed_fused_kda_decode(
        **{**inputs, "output": output, "lower_bound": lower_bound}
    )

    if live_slot > 0:
        oracle = {
            **inputs,
            "x": inputs["x"][:1],
            "raw_gate": inputs["raw_gate"][:, :1],
            "raw_beta": inputs["raw_beta"][:, :1],
            "output_gate": inputs["output_gate"][:1],
            "state_indices": torch.tensor(
                [live_slot], dtype=torch.int32, device="cuda"
            ),
        }
        expected_conv = _clone_strided(before_conv)
        expected_state = _clone_strided(before_state)
        expected = _reference(oracle, expected_conv, expected_state, lower_bound)
        torch.testing.assert_close(actual[:, :1], expected, rtol=3e-2, atol=2e-2)
        torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(
            inputs["state"], expected_state, rtol=3e-2, atol=2e-3
        )
    else:
        assert torch.equal(actual, torch.zeros_like(actual))
        assert torch.equal(inputs["conv_state"], before_conv)
        assert torch.equal(inputs["state"], before_state)
    assert torch.equal(actual[:, 1:], torch.zeros_like(actual[:, 1:]))
    for slot in range(5):
        if slot != live_slot:
            assert torch.equal(inputs["state"][slot], before_state[slot])
            assert torch.equal(inputs["conv_state"][slot], before_conv[slot])


def test_packed_fused_kda_decode_t1_graph_changed_metadata_null_to_active():
    inputs = _make_inputs(12, 4, state_dtype=torch.float32)
    for key in ("x", "raw_gate", "raw_beta", "output_gate"):
        dimension = 1 if key in ("raw_gate", "raw_beta") else 0
        inputs[key] = inputs[key].narrow(dimension, 0, 3)
    inputs["query_start_loc"] = torch.tensor(
        [0, 1, 2, 3, 3], dtype=torch.int32, device="cuda"
    )
    inputs["num_accepted_tokens"] = torch.ones(4, dtype=torch.int32, device="cuda")
    inputs["state_indices"] = torch.tensor(
        [[1], [2], [3], [4]], dtype=torch.int32, device="cuda"
    )
    assert inputs["x"].shape[0] == 3
    assert inputs["query_start_loc"].tolist() == [0, 1, 2, 3, 3]
    assert all(0 < slot < inputs["state"].shape[0] for slot in [1, 2, 3, 4])
    output = torch.full((1, 3, 12, 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    kwargs = {**inputs, "output": output, "lower_bound": -5.0}
    _packed_fused_kda_decode(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _packed_fused_kda_decode(**kwargs)
    seeded_conv = _clone_strided(inputs["conv_state"])
    seeded_state = _clone_strided(inputs["state"])

    # all-active -> trailing-empty -> late-active, including null/negative
    # slots.  Each replay uses a fresh literal oracle and must clear padding.
    trials = [
        ([0, 1, 2, 3, 3], [1, 2, 3, 4], [1, 2, 3]),
        ([0, 1, 1, 1, 1], [4, 2, 3, 1], [4]),
        ([0, 0, 0, 1, 1], [1, 2, 4, 3], [4]),
        ([0, 0, 0, 1, 1], [1, 2, 0, 3], []),
        ([0, 1, 2, 3, 3], [1, 2, 3, 4], [1, 2, 3]),
    ]
    for qsl, slots, expected_slots in trials:
        assert len(qsl) == 5 and len(slots) == 4
        assert qsl[-1] <= inputs["x"].shape[0]
        assert all(-1 <= slot < inputs["state"].shape[0] for slot in slots)
        inputs["conv_state"].copy_(seeded_conv)
        inputs["state"].copy_(seeded_state)
        inputs["query_start_loc"].copy_(torch.tensor(qsl, device="cuda"))
        inputs["state_indices"].copy_(torch.tensor(slots, device="cuda").view(4, 1))
        output.fill_(torch.nan)
        expected_conv = _clone_strided(seeded_conv)
        expected_state = _clone_strided(seeded_state)
        if expected_slots:
            oracle = {
                **inputs,
                "x": inputs["x"][: len(expected_slots)],
                "raw_gate": inputs["raw_gate"][:, : len(expected_slots)],
                "raw_beta": inputs["raw_beta"][:, : len(expected_slots)],
                "output_gate": inputs["output_gate"][: len(expected_slots)],
                "state_indices": torch.tensor(expected_slots, device="cuda"),
            }
            expected = _reference(oracle, expected_conv, expected_state, -5.0)
        else:
            expected = None
        graph.replay()
        torch.cuda.synchronize()
        if expected is not None:
            torch.testing.assert_close(
                output[:, : len(expected_slots)], expected, rtol=3e-2, atol=2e-2
            )
        else:
            assert torch.equal(output, torch.zeros_like(output))
        assert torch.equal(
            output[:, len(expected_slots) :],
            torch.zeros_like(output[:, len(expected_slots) :]),
        )
        torch.testing.assert_close(inputs["conv_state"], expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(
            inputs["state"], expected_state, rtol=3e-2, atol=2e-3
        )
        for slot in range(5):
            if slot not in expected_slots:
                assert torch.equal(inputs["state"][slot], seeded_state[slot])
                assert torch.equal(inputs["conv_state"][slot], seeded_conv[slot])


def test_packed_fused_kda_decode_cache_key(monkeypatch):
    import importlib

    module = importlib.import_module(
        "flashinfer.kda_kernels.fused_kda_decode_multitoken"
    )
    kernel_names = []

    def record_kernel_name(module_name, kernel_name, compile_fn, extra_key_files):
        del module_name, compile_fn, extra_key_files
        kernel_names.append(kernel_name)
        return kernel_name

    monkeypatch.setattr(module, "build_and_load_cute_dsl_kernel", record_kernel_name)
    module._get_compiled_kernel.cache_clear()
    base = [3, 12, -5.0, 1e-5, 2, 16, 1, 1, 1, False]
    alternatives = [4, 24, -4.0, 2e-5, 1, 32, 2, 2, 4, True]
    try:
        module._get_compiled_kernel(*base)
        for index, alternative in enumerate(alternatives):
            args = list(base)
            args[index] = alternative
            module._get_compiled_kernel(*args)
    finally:
        module._get_compiled_kernel.cache_clear()

    assert len(kernel_names) == len(set(kernel_names)) == 1 + len(alternatives)
