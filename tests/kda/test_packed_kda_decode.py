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

"""Correctness gates for serving-native packed Kimi K3 T=1 decode.

The large GPU cases intentionally reproduce the exported Cake acceptance
contract. They must run in a Slurm allocation on exact CC 10.0 or 10.3, never
on a login node.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from flashinfer.kda_decode import packed_kda_decode


_HEADS = 12
_HEAD_DIM = 128
_MIXED_WIDTH = 3 * _HEADS * _HEAD_DIM
_GATE_WIDTH = _HEADS * _HEAD_DIM
_LOGICAL_STATE_SLOT = _HEADS * _HEAD_DIM * _HEAD_DIM
_PRODUCTION_MIXED_STRIDE = 6144
_PRODUCTION_STATE_PADDING = 256
_ATOL = 1.0e-2
_RTOL = 1.0e-2
_SCALE = _HEAD_DIM**-0.5
_EPSILON = 1.0e-6
_LOWER_BOUND = -5.0


@pytest.fixture
def packed_kda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip(
            "packed KDA T=1 requires exact CC 10.0 (SM100a) or CC 10.3 (SM103a)"
        )
    return device


def _state_view(storage, slots, slot_stride):
    return storage.as_strided(
        (slots, _HEADS, _HEAD_DIM, _HEAD_DIM),
        (slot_stride, _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1),
    )


def _make_case(
    batch,
    device,
    *,
    seed,
    inactive=True,
    state_padding=_PRODUCTION_STATE_PADDING,
    gate_padding=0,
    beta_padding=0,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    mixed_storage = torch.randn(
        (batch, _PRODUCTION_MIXED_STRIDE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.5)
    gate_storage = torch.randn(
        (batch, _GATE_WIDTH + gate_padding),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.5)
    beta_storage = torch.randn(
        (batch, _HEADS + beta_padding),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.5)

    slots = batch + 9
    state_slot_stride = _LOGICAL_STATE_SLOT + state_padding
    state_storage = torch.randn(
        slots * state_slot_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.05)
    state = _state_view(state_storage, slots, state_slot_stride)

    # Shifted slots prove that state_indices, rather than the batch row, owns
    # the recurrent state. The final row is graph padding for B > 1.
    indices_host = [batch + 2 - row for row in range(batch)]
    if inactive and batch > 1:
        indices_host[-1] = -1
    state_indices = torch.tensor(indices_host, dtype=torch.int32, device=device)

    return {
        "mixed_storage": mixed_storage,
        "mixed_qkv": mixed_storage[:, :_MIXED_WIDTH],
        "gate_storage": gate_storage,
        "raw_gate": gate_storage[:, :_GATE_WIDTH],
        "beta_storage": beta_storage,
        "raw_beta": beta_storage[:, :_HEADS],
        "A_log": torch.randn(
            _HEADS,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(0.2)
        .sub_(2.0),
        "dt_bias": torch.randn(
            _GATE_WIDTH,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ).mul_(0.25),
        "state_storage": state_storage,
        "state": state,
        "state_slot_stride": state_slot_stride,
        "slots": slots,
        "indices_host": indices_host,
        "state_indices": state_indices,
        "output": torch.full(
            (batch, 1, _HEADS, _HEAD_DIM),
            123.0,
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def _call(case, *, state=None, output=None):
    return packed_kda_decode(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        case["state"] if state is None else state,
        case["state_indices"],
        output=case["output"] if output is None else output,
    )


def _reference_step(
    mixed_qkv,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    *,
    work_dtype=torch.float32,
    output_dtype=torch.bfloat16,
):
    batch = mixed_qkv.shape[0]
    packed = mixed_qkv.to(work_dtype).reshape(batch, 3, _HEADS, _HEAD_DIM)
    q_raw = packed[:, 0]
    k_raw = packed[:, 1]
    q = (
        q_raw
        * torch.rsqrt(torch.sum(q_raw * q_raw, dim=-1, keepdim=True) + _EPSILON)
        * _SCALE
    )
    k = k_raw * torch.rsqrt(torch.sum(k_raw * k_raw, dim=-1, keepdim=True) + _EPSILON)
    value = packed[:, 2]
    gate_x = raw_gate.to(work_dtype).reshape(batch, _HEADS, _HEAD_DIM)
    gate_x = gate_x + dt_bias.to(work_dtype).reshape(_HEADS, _HEAD_DIM)
    decay = torch.exp(
        _LOWER_BOUND
        * torch.sigmoid(torch.exp(A_log.to(work_dtype))[None, :, None] * gate_x)
    )
    beta = torch.sigmoid(raw_beta.to(work_dtype))

    active = state_indices >= 0
    safe_indices = state_indices.clamp_min(0).to(torch.long)
    selected = state.index_select(0, safe_indices).to(work_dtype)
    decayed = selected * decay[:, :, None, :]
    prediction = torch.einsum("bhvk,bhk->bhv", decayed, k)
    delta = (value - prediction) * beta[:, :, None]
    updated = decayed + delta[:, :, :, None] * k[:, :, None, :]
    projected = torch.einsum("bhvk,bhk->bhv", updated, q)

    active_slots = state_indices[active].to(torch.long)
    state.index_copy_(0, active_slots, updated[active].to(state.dtype))
    output = torch.where(active[:, None, None], projected, 0.0).to(output_dtype)
    return output.unsqueeze(1)


def _clone_padded_state(case):
    storage = case["state_storage"].clone()
    return storage, _state_view(storage, case["slots"], case["state_slot_stride"])


def _assert_close(actual, expected):
    torch.testing.assert_close(
        actual,
        expected,
        atol=_ATOL,
        rtol=_RTOL,
        check_dtype=False,
    )


def _assert_mutation_contract(case, before_storage):
    selected = {slot for slot in case["indices_host"] if slot >= 0}
    before_state = _state_view(before_storage, case["slots"], case["state_slot_stride"])
    state_bits = case["state"].contiguous().view(torch.int16)
    before_bits = before_state.contiguous().view(torch.int16)
    changed_by_slot = (state_bits != before_bits).reshape(case["slots"], -1).any(dim=1)
    for slot, changed in enumerate(changed_by_slot.cpu().tolist()):
        unchanged = not changed
        if slot in selected:
            assert not unchanged, f"selected state slot {slot} was not updated"
        else:
            assert unchanged, f"unselected state slot {slot} changed"

    storage_rows = case["state_storage"].as_strided(
        (case["slots"], case["state_slot_stride"]),
        (case["state_slot_stride"], 1),
    )
    before_rows = before_storage.as_strided(
        (case["slots"], case["state_slot_stride"]),
        (case["state_slot_stride"], 1),
    )
    assert torch.equal(
        storage_rows[:, _LOGICAL_STATE_SLOT:].contiguous().view(torch.int16),
        before_rows[:, _LOGICAL_STATE_SLOT:].contiguous().view(torch.int16),
    )

    inactive_rows = [row for row, slot in enumerate(case["indices_host"]) if slot < 0]
    if inactive_rows:
        inactive = case["output"][inactive_rows]
        assert torch.equal(
            inactive.contiguous().view(torch.int16),
            torch.zeros_like(inactive).view(torch.int16),
        )


_BATCH_CASES = [
    1,
    8,
    16,
    31,
    32,
    64,
    128,
    pytest.param(256, marks=pytest.mark.long_running),
    pytest.param(512, marks=pytest.mark.long_running),
]


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", _BATCH_CASES)
def test_packed_kda_decode_matches_reference_and_preserves_pool(
    packed_kda_device, batch
):
    case = _make_case(batch, packed_kda_device, seed=20260805 + batch)
    before_storage = case["state_storage"].clone()
    _, reference_state = _clone_padded_state(case)
    reference_output = _reference_step(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        reference_state,
        case["state_indices"],
    )

    result = _call(case)
    torch.cuda.synchronize(packed_kda_device)

    assert result is case["output"]
    assert result.shape == (batch, 1, _HEADS, _HEAD_DIM)
    assert case["mixed_qkv"].stride() == (_PRODUCTION_MIXED_STRIDE, 1)
    assert case["state"].stride() == (
        _LOGICAL_STATE_SLOT + _PRODUCTION_STATE_PADDING,
        _HEAD_DIM * _HEAD_DIM,
        _HEAD_DIM,
        1,
    )
    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)
    _assert_mutation_contract(case, before_storage)


@pytest.mark.arch_blackwell
def test_packed_kda_decode_allocates_canonical_output(packed_kda_device):
    case = _make_case(8, packed_kda_device, seed=20260820)
    _, reference_state = _clone_padded_state(case)
    reference_output = _reference_step(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        reference_state,
        case["state_indices"],
    )

    result = packed_kda_decode(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        case["state"],
        case["state_indices"],
    )
    torch.cuda.synchronize(packed_kda_device)

    assert result.shape == (8, 1, _HEADS, _HEAD_DIM)
    assert result.dtype == torch.bfloat16
    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)


@pytest.mark.arch_blackwell
def test_packed_kda_decode_all_inactive_is_bitwise_noop(packed_kda_device):
    case = _make_case(1, packed_kda_device, seed=20260821, inactive=False)
    case["indices_host"] = [-1]
    case["state_indices"].fill_(-1)
    case["output"].fill_(float("nan"))
    before_storage = case["state_storage"].clone()

    result = _call(case)
    torch.cuda.synchronize(packed_kda_device)

    assert torch.equal(
        case["state_storage"].view(torch.int16),
        before_storage.view(torch.int16),
    )
    assert torch.equal(
        result.view(torch.int16),
        torch.zeros_like(result).view(torch.int16),
    )


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", [8, 64])
def test_packed_kda_decode_sanitizer_schedules(packed_kda_device, batch):
    """Small named entry points for tile8/tile16 sanitizer invocations."""
    case = _make_case(
        batch,
        packed_kda_device,
        seed=20260824 + batch,
        inactive=False,
        state_padding=17,
    )
    _call(case)
    torch.cuda.synchronize(packed_kda_device)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", [8, 64])
def test_packed_kda_decode_uses_current_stream(packed_kda_device, batch):
    case = _make_case(batch, packed_kda_device, seed=20260830 + batch)
    _, reference_state = _clone_padded_state(case)
    reference_output = _reference_step(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        reference_state,
        case["state_indices"],
    )

    stream = torch.cuda.Stream(device=packed_kda_device)
    stream.wait_stream(torch.cuda.current_stream(packed_kda_device))
    with torch.cuda.stream(stream):
        result = _call(case)
    torch.cuda.current_stream(packed_kda_device).wait_stream(stream)

    assert result is case["output"]
    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", [8, 64])
def test_packed_kda_decode_cuda_graph_replays_changed_inputs_and_indices(
    packed_kda_device, batch
):
    case = _make_case(batch, packed_kda_device, seed=20260840 + batch)
    initial_storage = case["state_storage"].clone()

    # Compile and initialize all lazy runtime state before capture.
    _call(case)
    torch.cuda.synchronize(packed_kda_device)
    case["state_storage"].copy_(initial_storage)
    case["output"].fill_(123.0)

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(device=packed_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(packed_kda_device))
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_result = _call(case)
    torch.cuda.synchronize(packed_kda_device)
    assert captured_result is case["output"]

    generator = torch.Generator(device=packed_kda_device).manual_seed(20260900 + batch)
    changed_mixed = torch.randn(
        case["mixed_qkv"].shape,
        dtype=torch.bfloat16,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.25)
    changed_gate = torch.randn(
        case["raw_gate"].shape,
        dtype=torch.bfloat16,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.25)
    changed_beta = torch.randn(
        case["raw_beta"].shape,
        dtype=torch.bfloat16,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.25)
    changed_indices_host = [row + 4 for row in range(batch)]
    changed_indices_host[0] = -1

    case["mixed_qkv"].copy_(changed_mixed)
    case["raw_gate"].copy_(changed_gate)
    case["raw_beta"].copy_(changed_beta)
    case["state_indices"].copy_(
        torch.tensor(
            changed_indices_host,
            dtype=torch.int32,
            device=packed_kda_device,
        )
    )
    case["indices_host"] = changed_indices_host
    case["state_storage"].copy_(initial_storage)
    case["output"].fill_(123.0)

    reference_storage = initial_storage.clone()
    reference_state = _state_view(
        reference_storage, case["slots"], case["state_slot_stride"]
    )
    reference_output = _reference_step(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        reference_state,
        case["state_indices"],
    )

    graph.replay()
    torch.cuda.synchronize(packed_kda_device)

    _assert_close(case["output"], reference_output)
    _assert_close(case["state"], reference_state)
    _assert_mutation_contract(case, initial_storage)


@pytest.mark.arch_blackwell
@pytest.mark.long_running
def test_packed_kda_decode_512_step_fp64_diagnostic(packed_kda_device):
    steps = 512
    batch = 8
    generator = torch.Generator(device=packed_kda_device).manual_seed(20260818)
    mixed_storage = torch.randn(
        (steps, batch, _PRODUCTION_MIXED_STRIDE),
        dtype=torch.bfloat16,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.5)
    gate_storage = torch.randn(
        (steps, batch, _GATE_WIDTH + 17),
        dtype=torch.bfloat16,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.5)
    beta_storage = torch.randn(
        (steps, batch, _HEADS + 5),
        dtype=torch.bfloat16,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.5)
    slots = batch + 9
    state_slot_stride = _LOGICAL_STATE_SLOT + 17
    state_storage = torch.randn(
        slots * state_slot_stride,
        dtype=torch.bfloat16,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.05)
    state = _state_view(state_storage, slots, state_slot_stride)
    oracle_state = state.to(torch.float64)
    state_indices = torch.tensor(
        [3, 4, 5, 6, 7, 8, 9, -1],
        dtype=torch.int32,
        device=packed_kda_device,
    )
    A_log = (
        torch.randn(
            _HEADS,
            dtype=torch.float32,
            device=packed_kda_device,
            generator=generator,
        )
        .mul_(0.2)
        .sub_(2.0)
    )
    dt_bias = torch.randn(
        _GATE_WIDTH,
        dtype=torch.float32,
        device=packed_kda_device,
        generator=generator,
    ).mul_(0.25)
    output = torch.empty(
        (batch, 1, _HEADS, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=packed_kda_device,
    )
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}

    for token in range(steps):
        result = packed_kda_decode(
            mixed_storage[token, :, :_MIXED_WIDTH],
            gate_storage[token, :, :_GATE_WIDTH],
            beta_storage[token, :, :_HEADS],
            A_log,
            dt_bias,
            state,
            state_indices,
            output=output,
        )
        oracle_output = _reference_step(
            mixed_storage[token, :, :_MIXED_WIDTH],
            gate_storage[token, :, :_GATE_WIDTH],
            beta_storage[token, :, :_HEADS],
            A_log,
            dt_bias,
            oracle_state,
            state_indices,
            work_dtype=torch.float64,
            output_dtype=torch.float64,
        )
        if token + 1 in checkpoints:
            torch.cuda.synchronize(packed_kda_device)
            _assert_close(result, oracle_output)
            _assert_close(state, oracle_state)

    output_error = (result.to(torch.float64) - oracle_output).abs()
    state_error = (state.to(torch.float64) - oracle_state).abs()
    assert float(output_error.max()) <= _ATOL
    assert float(state_error.max()) <= _ATOL


def test_public_packed_kda_decode_forwards_canonical_output_cpu(monkeypatch):
    kda_decode_module = importlib.import_module("flashinfer.kda_decode")
    calls = []
    output = torch.empty(2, 1, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)

    def run(**kwargs):
        calls.append(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(kda_decode_module, "_run_packed_kda_decode", run)
    args = [object() for _ in range(7)]

    result = kda_decode_module.packed_kda_decode(*args, output=output)

    assert result is output
    assert len(calls) == 1
    assert calls[0]["output"] is output


class _FakeCudaTensor:
    def __init__(self, shape, *, device="cuda:0"):
        self.shape = tuple(shape)
        self.device = device
        self.is_cuda = True
        self.view_result = None
        self.allocated_shapes = []

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def view(self, *shape):
        self.view_result = _FakeCudaTensor(shape, device=self.device)
        return self.view_result

    def new_empty(self, shape):
        self.allocated_shapes.append(tuple(shape))
        return _FakeCudaTensor(shape, device=self.device)


@pytest.mark.parametrize(
    ("batch", "target", "variant"),
    [(31, "sm100a", "tile8"), (32, "sm103a", "tile16")],
)
def test_kernel_facade_selects_variant_and_caller_stream_cpu(
    monkeypatch, batch, target, variant
):
    packed_module = importlib.import_module("flashinfer.kda_kernels.packed_kda_decode")
    stream_handle = 0x12345678
    fake_torch = SimpleNamespace(
        Tensor=_FakeCudaTensor,
        cuda=SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(cuda_stream=stream_handle)
        ),
    )
    monkeypatch.setattr(packed_module, "torch", fake_torch)
    monkeypatch.setattr(packed_module, "_target_for_device", lambda device: target)

    module_calls = []
    launch_calls = []

    def get_module(selected_variant, selected_target):
        module_calls.append((selected_variant, selected_target))
        return SimpleNamespace(run=lambda *args: launch_calls.append(args))

    monkeypatch.setattr(packed_module, "get_flash_kda_packed_t1_module", get_module)
    mixed_qkv = _FakeCudaTensor((batch, _MIXED_WIDTH))
    inputs = [mixed_qkv, *(_FakeCudaTensor((1,)) for _ in range(6))]
    output = _FakeCudaTensor((batch, 1, _HEADS, _HEAD_DIM))

    result = packed_module.run_packed_kda_decode(*inputs, output=output)

    assert result is output
    assert module_calls == [(variant, target)]
    assert len(launch_calls) == 1
    assert launch_calls[0][-2] is output.view_result
    assert launch_calls[0][-1] == stream_handle


@pytest.mark.parametrize(
    ("capability", "cuda_at_least", "expected"),
    [
        ((10, 0), {"12.8": True}, "sm100a"),
        ((10, 3), {"12.8": True, "12.9": True}, "sm103a"),
        ((10, 0), {"12.8": False}, None),
        ((10, 3), {"12.8": True, "12.9": False}, None),
        ((10, 1), {"12.8": True, "12.9": True}, None),
    ],
)
def test_kernel_facade_selects_exact_physical_target_cpu(
    monkeypatch, capability, cuda_at_least, expected
):
    packed_module = importlib.import_module("flashinfer.kda_kernels.packed_kda_decode")
    monkeypatch.setattr(
        packed_module, "get_compute_capability", lambda device: capability
    )
    monkeypatch.setattr(
        packed_module,
        "is_cuda_version_at_least",
        lambda version: cuda_at_least.get(version, False),
    )
    if expected is None:
        with pytest.raises(RuntimeError):
            packed_module._target_for_device(torch.device("cpu"))
    else:
        assert packed_module._target_for_device(torch.device("cpu")) == expected
