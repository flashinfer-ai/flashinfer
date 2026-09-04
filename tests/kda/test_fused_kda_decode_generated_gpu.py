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

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer.jit.fused_kda_decode_generated import (
    fused_kda_decode_generated_is_available,
)
from flashinfer.utils import get_compute_capability

try:
    from flashinfer.kda_decode import (
        _FUSED_KDA_DECODE_AVAILABLE,
        fused_kda_decode,
    )

    _impl = importlib.import_module("flashinfer.kda_kernels.fused_kda_decode")
except ImportError:
    fused_kda_decode = None
    _FUSED_KDA_DECODE_AVAILABLE = False
    _impl = None


_HEAD_DIM = 128
_OFFICIAL_SHAPES = (
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


def _boundary_shapes():
    shapes = set(_OFFICIAL_SHAPES)
    for num_heads in (12, 24, 32, 48, 96):
        wide_max = 148 // num_heads
        compact_min = 296 // num_heads + 1
        compact_max = 444 // num_heads
        high_min = (
            32 if num_heads == 32 else math.ceil(1184 / num_heads)
        )
        for num_rows in (
            wide_max,
            wide_max + 1,
            compact_min - 1,
            compact_min,
            compact_max,
            compact_max + 1,
            high_min - 1,
            high_min,
        ):
            if num_rows > 0:
                shapes.add((num_heads, num_rows))
    return tuple(sorted(shapes))


_FULL_DOMAIN_SHAPES = _boundary_shapes()
_VARIANT_CASES = (
    ("repeated_safe_f32", 12, 3, torch.float32, "page", "repeated"),
    ("repeated_safe_bf16", 12, 3, torch.bfloat16, "page", "repeated"),
    ("wide512_positive_f32", 12, 1, torch.float32, "page", "positive"),
    ("wide512_f32", 12, 1, torch.float32, "page", "null"),
    ("wide512_bf16", 12, 1, torch.bfloat16, "page", "positive"),
    (
        "compact_async_pr_eval_h96_f32",
        96,
        4,
        torch.float32,
        "page",
        "positive",
    ),
    ("compact_async_f32", 12, 25, torch.float32, "page", "positive"),
    ("compact_async_bf16", 12, 25, torch.bfloat16, "page", "positive"),
    (
        "high_work_positive_h96_pr_strides_f32",
        96,
        13,
        torch.float32,
        "page",
        "positive",
    ),
    (
        "high_work_positive_h96_f32",
        96,
        13,
        torch.float32,
        "padded",
        "positive",
    ),
    ("high_work_positive_f32", 12, 99, torch.float32, "page", "positive"),
    ("high_work_f32", 12, 99, torch.float32, "page", "null"),
    ("high_work_bf16", 12, 99, torch.bfloat16, "page", "positive"),
    ("pr_eval_h32_f32", 32, 5, torch.float32, "page", "positive"),
    ("direct_f32", 24, 7, torch.float32, "page", "positive"),
    ("direct_bf16", 24, 7, torch.bfloat16, "page", "positive"),
)


@pytest.fixture(autouse=True)
def _require_generated_fused_kda_decode():
    if not torch.cuda.is_available():
        pytest.skip("generated fused KDA decode requires CUDA")
    if get_compute_capability(torch.device("cuda")) != (10, 0):
        pytest.skip("generated fused KDA decode requires an SM100a GPU")
    if not _FUSED_KDA_DECODE_AVAILABLE:
        pytest.skip("fused KDA decode dependencies are unavailable")
    assert fused_kda_decode_generated_is_available(), (
        "the checked-in generated fused KDA manifest must be complete"
    )


def _page_strides(num_heads, state_dtype):
    hidden_size = num_heads * _HEAD_DIM
    conv_slot_bytes = 3 * hidden_size * 3 * torch.bfloat16.itemsize
    state_element_bytes = torch.empty((), dtype=state_dtype).element_size()
    state_slot_bytes = (
        num_heads * _HEAD_DIM * _HEAD_DIM * state_element_bytes
    )
    page_bytes = conv_slot_bytes + state_slot_bytes
    return page_bytes // torch.bfloat16.itemsize, page_bytes // state_element_bytes


def _make_inputs(
    num_heads,
    num_rows,
    *,
    state_dtype=torch.float32,
    layout="page",
    slot_class="positive",
    rank4_output_gate=False,
    seed=42,
):
    device = torch.device("cuda")
    hidden_size = num_heads * _HEAD_DIM
    num_slots = max(num_rows + 1, 4)
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype=torch.float32):
        return torch.randn(
            shape, device=device, dtype=torch.float32, generator=generator
        ).to(dtype)

    if layout == "page":
        x_padding, beta_padding, output_gate_padding = 17, 1, 7
        conv_slot_stride, state_slot_stride = _page_strides(
            num_heads, state_dtype
        )
    elif layout == "padded":
        x_padding, beta_padding, output_gate_padding = 29, 3, 11
        conv_slot_stride = 9 * hidden_size + 12
        state_slot_stride = num_heads * _HEAD_DIM * _HEAD_DIM + 8
    else:
        raise ValueError(f"unknown layout: {layout}")

    x_storage = randn(
        (num_rows, 3 * hidden_size + x_padding), torch.bfloat16
    )
    conv_state = torch.empty_strided(
        (num_slots, 3 * hidden_size, 3),
        (conv_slot_stride, 1, 3 * hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state.copy_(
        0.1 * randn((num_slots, 3 * hidden_size, 3), torch.bfloat16)
    )
    state = torch.empty_strided(
        (num_slots, num_heads, _HEAD_DIM, _HEAD_DIM),
        (state_slot_stride, _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1),
        dtype=state_dtype,
        device=device,
    )
    state.copy_(
        0.01
        * randn(
            (num_slots, num_heads, _HEAD_DIM, _HEAD_DIM), state_dtype
        )
    )
    beta_storage = randn(
        (1, num_rows, num_heads + beta_padding), torch.bfloat16
    )
    output_gate_storage = randn(
        (num_rows, hidden_size + output_gate_padding), torch.bfloat16
    )
    output_gate = output_gate_storage.as_strided(
        (num_rows, num_heads, _HEAD_DIM),
        (hidden_size + output_gate_padding, _HEAD_DIM, 1),
    )
    if rank4_output_gate:
        output_gate = output_gate.unsqueeze(0)

    if slot_class == "positive":
        state_indices = torch.arange(
            num_rows, 0, -1, dtype=torch.int32, device=device
        )
    elif slot_class == "null":
        state_indices = torch.arange(
            num_rows, 0, -1, dtype=torch.int32, device=device
        )
        state_indices[: min(2, num_rows)] = torch.tensor(
            (0, -1)[: min(2, num_rows)], dtype=torch.int32, device=device
        )
    elif slot_class == "repeated":
        state_indices = torch.tensor(
            [1 + (row % 2) for row in range(num_rows)],
            dtype=torch.int32,
            device=device,
        )
    else:
        raise ValueError(f"unknown slot class: {slot_class}")

    return {
        "x": x_storage[:, : 3 * hidden_size],
        "weight": 0.1 * randn((3, 4, hidden_size)),
        "conv_state": conv_state,
        "raw_gate": randn(
            (1, num_rows, num_heads, _HEAD_DIM), torch.bfloat16
        ),
        "raw_beta": beta_storage[:, :, :num_heads],
        "A_log": 0.5 * randn((num_heads,)),
        "dt_bias": 0.1 * randn((hidden_size,)),
        "state_indices": state_indices,
        "state": state,
        "output_gate": output_gate,
        "norm_weight": randn((_HEAD_DIM,)),
    }


def _clone_strided(tensor):
    clone = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    clone.copy_(tensor)
    return clone


def _guarded_strided_clone(tensor):
    prefix_elements = 17
    suffix_elements = 31
    span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride())
    )
    storage = torch.full(
        (prefix_elements + span + suffix_elements,),
        -37.5,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    guarded = torch.as_strided(
        storage,
        tensor.shape,
        tensor.stride(),
        storage_offset=prefix_elements,
    )
    guarded.copy_(tensor)
    guard_mask = torch.ones(
        storage.shape, dtype=torch.bool, device=tensor.device
    )
    torch.as_strided(
        guard_mask,
        tensor.shape,
        tensor.stride(),
        storage_offset=prefix_elements,
    ).fill_(False)
    expected_guard = storage.masked_select(guard_mask).clone()
    return guarded, storage, guard_mask, expected_guard


@torch.no_grad()
def _reference_unique_rows(
    inputs, conv_state, state, row_indices, *, lower_bound, norm_eps
):
    num_rows = inputs["x"].shape[0]
    num_heads = inputs["A_log"].numel()
    hidden_size = num_heads * _HEAD_DIM
    output = torch.zeros(
        (1, num_rows, num_heads, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=inputs["x"].device,
    )
    slots = inputs["state_indices"].index_select(0, row_indices).long()
    live_mask = slots > 0
    if not torch.any(live_mask):
        return output
    live_rows = row_indices[live_mask]
    live_slots = slots[live_mask]
    assert torch.unique(live_slots).numel() == live_slots.numel()

    taps = inputs["weight"].permute(0, 2, 1).reshape(3 * hidden_size, 4)
    history = conv_state.index_select(0, live_slots).float()
    x = inputs["x"].index_select(0, live_rows).float()
    window = torch.cat((history, x.unsqueeze(-1)), dim=-1)
    conv_state.index_copy_(0, live_slots, window[:, :, 1:].to(torch.bfloat16))
    mixed = F.silu((window * taps).sum(-1)).to(torch.bfloat16).float()
    query, key, value = mixed.view(-1, 3, num_heads, _HEAD_DIM).unbind(1)
    query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    query *= _HEAD_DIM**-0.5
    key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)

    gate = inputs["raw_gate"][0].index_select(0, live_rows).float()
    gate += inputs["dt_bias"].view(num_heads, _HEAD_DIM)
    A = inputs["A_log"].exp()[None, :, None]
    if lower_bound is None:
        gate = -A * F.softplus(gate)
    else:
        gate = lower_bound * torch.sigmoid(A * gate)

    selected_state = state.index_select(0, live_slots).float()
    selected_state *= gate.exp().unsqueeze(-2)
    state_key = torch.einsum("nhvk,nhk->nhv", selected_state, key)
    delta = value - state_key
    beta = inputs["raw_beta"][0].index_select(0, live_rows).float()
    delta *= beta.sigmoid().unsqueeze(-1)
    selected_state += delta.unsqueeze(-1) * key.unsqueeze(-2)
    state.index_copy_(0, live_slots, selected_state.to(state.dtype))

    rows_output = torch.einsum("nhvk,nhk->nhv", selected_state, query)
    rows_output = rows_output.to(torch.bfloat16).float()
    rows_output *= torch.rsqrt(
        rows_output.square().mean(-1, keepdim=True) + norm_eps
    )
    output_gate = inputs["output_gate"]
    if output_gate.ndim == 4:
        output_gate = output_gate[0]
    rows_output *= inputs["norm_weight"]
    rows_output *= output_gate.index_select(0, live_rows).float().sigmoid()
    output[0].index_copy_(0, live_rows, rows_output.to(torch.bfloat16))
    return output


@torch.no_grad()
def _reference(inputs, conv_state, state, *, lower_bound=-5.0, norm_eps=1e-5):
    num_rows = inputs["x"].shape[0]
    slots = inputs["state_indices"]
    positive_slots = slots[slots > 0]
    if torch.unique(positive_slots).numel() == positive_slots.numel():
        rows = torch.arange(num_rows, device=slots.device)
        return _reference_unique_rows(
            inputs,
            conv_state,
            state,
            rows,
            lower_bound=lower_bound,
            norm_eps=norm_eps,
        )

    output = torch.zeros(
        (1, num_rows, inputs["A_log"].numel(), _HEAD_DIM),
        dtype=torch.bfloat16,
        device=inputs["x"].device,
    )
    for row in range(num_rows):
        row_output = _reference_unique_rows(
            inputs,
            conv_state,
            state,
            torch.tensor((row,), dtype=torch.long, device=slots.device),
            lower_bound=lower_bound,
            norm_eps=norm_eps,
        )
        output[:, row : row + 1].copy_(row_output[:, row : row + 1])
    return output


def _run_and_check_generated(
    monkeypatch,
    inputs,
    *,
    expected_variant=None,
    lower_bound=-5.0,
    norm_eps=1e-5,
    preallocate_output=False,
):
    reference_conv_state = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    actual_conv_state = _clone_strided(inputs["conv_state"])
    actual_state = _clone_strided(inputs["state"])
    expected = _reference(
        inputs,
        reference_conv_state,
        reference_state,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
    )

    routed_variants = []
    original_run = _impl._run_generated_variant

    def record_generated_route(variant, **kwargs):
        routed_variants.append(variant.name)
        return original_run(variant, **kwargs)

    monkeypatch.setattr(_impl, "_run_generated_variant", record_generated_route)
    kwargs = {
        **inputs,
        "conv_state": actual_conv_state,
        "state": actual_state,
        "lower_bound": lower_bound,
        "norm_eps": norm_eps,
    }
    if preallocate_output:
        kwargs["output"] = torch.empty_like(expected)
    actual = fused_kda_decode(**kwargs)

    assert len(routed_variants) == 1
    if expected_variant is not None:
        assert routed_variants == [expected_variant]
    if preallocate_output:
        assert actual is kwargs["output"]
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        actual_conv_state, reference_conv_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_state, reference_state, rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize(
    (
        "variant_name",
        "num_heads",
        "num_rows",
        "state_dtype",
        "layout",
        "slot_class",
    ),
    [pytest.param(*case, id=case[0]) for case in _VARIANT_CASES],
)
def test_generated_fused_kda_decode_all_manifest_routes(
    monkeypatch,
    variant_name,
    num_heads,
    num_rows,
    state_dtype,
    layout,
    slot_class,
):
    inputs = _make_inputs(
        num_heads,
        num_rows,
        state_dtype=state_dtype,
        layout=layout,
        slot_class=slot_class,
    )
    _run_and_check_generated(
        monkeypatch, inputs, expected_variant=variant_name
    )


@pytest.mark.parametrize(
    ("num_heads", "num_rows"),
    [
        pytest.param(num_heads, num_rows, id=f"h{num_heads}-n{num_rows}")
        for num_heads, num_rows in _FULL_DOMAIN_SHAPES
    ],
)
def test_generated_fused_kda_decode_official_and_boundary_shapes(
    monkeypatch, num_heads, num_rows
):
    inputs = _make_inputs(num_heads, num_rows)
    _run_and_check_generated(monkeypatch, inputs)


@pytest.mark.parametrize(
    (
        "state_dtype",
        "lower_bound",
        "norm_eps",
        "layout",
        "slot_class",
        "rank4_output_gate",
        "preallocate_output",
    ),
    (
        (torch.float32, None, 0.0, "page", "positive", False, False),
        (torch.bfloat16, None, 3e-4, "padded", "positive", True, False),
        (torch.float32, -2.75, 3e-4, "padded", "null", True, True),
    ),
)
def test_generated_fused_kda_decode_runtime_configuration_domain(
    monkeypatch,
    state_dtype,
    lower_bound,
    norm_eps,
    layout,
    slot_class,
    rank4_output_gate,
    preallocate_output,
):
    inputs = _make_inputs(
        48,
        5,
        state_dtype=state_dtype,
        layout=layout,
        slot_class=slot_class,
        rank4_output_gate=rank4_output_gate,
    )
    _run_and_check_generated(
        monkeypatch,
        inputs,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        preallocate_output=preallocate_output,
    )


@pytest.mark.parametrize(
    ("state_dtype", "num_rows", "slot_class", "lower_bound"),
    (
        (torch.bfloat16, 13, "null", -5.0),
        (torch.float32, 25, "positive", None),
        (torch.bfloat16, 25, "repeated", -5.0),
    ),
)
def test_generated_fused_kda_decode_cross_surface_boundaries(
    monkeypatch, state_dtype, num_rows, slot_class, lower_bound
):
    inputs = _make_inputs(
        12,
        num_rows,
        state_dtype=state_dtype,
        slot_class=slot_class,
    )
    _run_and_check_generated(
        monkeypatch, inputs, lower_bound=lower_bound, preallocate_output=True
    )


def test_generated_fused_kda_decode_cuda_graph_replay_correctness(monkeypatch):
    inputs = _make_inputs(12, 25, state_dtype=torch.float32)
    reference_conv_state = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])

    routed_variants = []
    original_run = _impl._run_generated_variant

    def record_generated_route(variant, **kwargs):
        routed_variants.append(variant.name)
        return original_run(variant, **kwargs)

    monkeypatch.setattr(_impl, "_run_generated_variant", record_generated_route)

    warmup_kwargs = {
        **inputs,
        "conv_state": _clone_strided(inputs["conv_state"]),
        "state": _clone_strided(inputs["state"]),
        "lower_bound": None,
        "output": torch.empty(
            (1, 25, 12, _HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        ),
    }
    fused_kda_decode(**warmup_kwargs)
    torch.cuda.synchronize()

    actual_conv_state = _clone_strided(inputs["conv_state"])
    actual_state = _clone_strided(inputs["state"])
    output = torch.empty(
        (1, 25, 12, _HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    graph_kwargs = {
        **inputs,
        "conv_state": actual_conv_state,
        "state": actual_state,
        "lower_bound": None,
        "output": output,
    }
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = fused_kda_decode(**graph_kwargs)
    assert captured_output is output
    assert routed_variants == ["compact_async_f32", "compact_async_f32"]

    for _ in range(2):
        expected = _reference(
            inputs,
            reference_conv_state,
            reference_state,
            lower_bound=None,
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            actual_conv_state, reference_conv_state, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual_state, reference_state, rtol=1e-2, atol=1e-2
        )


def test_generated_fused_kda_decode_preserves_write_guards(monkeypatch):
    inputs = _make_inputs(12, 25, layout="padded")
    reference_conv_state = _clone_strided(inputs["conv_state"])
    reference_state = _clone_strided(inputs["state"])
    expected = _reference(inputs, reference_conv_state, reference_state)

    actual_conv_state, conv_storage, conv_mask, expected_conv_guard = (
        _guarded_strided_clone(inputs["conv_state"])
    )
    actual_state, state_storage, state_mask, expected_state_guard = (
        _guarded_strided_clone(inputs["state"])
    )
    output, output_storage, output_mask, expected_output_guard = (
        _guarded_strided_clone(torch.zeros_like(expected))
    )
    routed_variants = []
    original_run = _impl._run_generated_variant

    def record_generated_route(variant, **kwargs):
        routed_variants.append(variant.name)
        return original_run(variant, **kwargs)

    monkeypatch.setattr(_impl, "_run_generated_variant", record_generated_route)
    kwargs = {
        **inputs,
        "conv_state": actual_conv_state,
        "state": actual_state,
        "output": output,
    }
    actual = fused_kda_decode(**kwargs)
    assert actual is output
    assert routed_variants == ["compact_async_f32"]
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        actual_conv_state, reference_conv_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_state, reference_state, rtol=1e-2, atol=1e-2
    )
    for storage, mask, expected_guard in (
        (conv_storage, conv_mask, expected_conv_guard),
        (state_storage, state_mask, expected_state_guard),
        (output_storage, output_mask, expected_output_guard),
    ):
        assert torch.equal(storage.masked_select(mask), expected_guard)
