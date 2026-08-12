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

"""Correctness tests for packed-input CuTe KDA T=1 decode."""

import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.kda_kernels.packed_kda_decode_cute import _select_tile_v
from flashinfer.kda_kernels.packed_kda_decode_cute import (
    run_packed_kda_decode_cute,
)


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
def packed_kda_cute_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("packed-input CuTe KDA requires exact CC 10.0")
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


def _call_cute(case, *, state=None, output=None, tile_v=None):
    return run_packed_kda_decode_cute(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        case["state"] if state is None else state,
        case["state_indices"],
        output=case["output"] if output is None else output,
        tile_v=tile_v,
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

    active = (state_indices >= 0) & (state_indices < state.shape[0])
    safe_indices = state_indices.clamp(0, state.shape[0] - 1).to(torch.long)
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
    selected = {slot for slot in case["indices_host"] if 0 <= slot < case["slots"]}
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

    inactive_rows = [
        row
        for row, slot in enumerate(case["indices_host"])
        if slot < 0 or slot >= case["slots"]
    ]
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


def test_packed_kda_cute_tile_selection_cpu():
    assert _select_tile_v(1) == 16
    assert _select_tile_v(11) == 16
    assert _select_tile_v(12) == 8
    assert _select_tile_v(23) == 8
    assert _select_tile_v(24) == 64
    assert _select_tile_v(37) == 64
    assert _select_tile_v(38) == 128
    assert _select_tile_v(512) == 128


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", _BATCH_CASES)
def test_packed_kda_cute_matches_reference_and_preserves_pool(
    packed_kda_cute_device, batch
):
    case = _make_case(batch, packed_kda_cute_device, seed=20261000 + batch)
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

    result = _call_cute(case)
    torch.cuda.synchronize(packed_kda_cute_device)

    assert result is case["output"]
    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)
    _assert_mutation_contract(case, before_storage)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tile_v", [8, 16, 32, 64, 128])
def test_packed_kda_cute_forced_tiles_match_reference(packed_kda_cute_device, tile_v):
    case = _make_case(8, packed_kda_cute_device, seed=20261100 + tile_v)
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

    result = _call_cute(case, tile_v=tile_v)
    torch.cuda.synchronize(packed_kda_cute_device)

    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("batch", "state_padding"),
    [
        (8, _PRODUCTION_STATE_PADDING),
        (64, _PRODUCTION_STATE_PADDING),
        (8, 17),
        (64, 17),
    ],
)
def test_packed_kda_cute_sanitizer_schedules(
    packed_kda_cute_device, batch, state_padding
):
    """Named aligned and unaligned tile8/tile16 sanitizer entry points."""
    case = _make_case(
        batch,
        packed_kda_cute_device,
        seed=20261150 + batch + state_padding,
        inactive=False,
        state_padding=state_padding,
    )
    _call_cute(case)
    torch.cuda.synchronize(packed_kda_cute_device)


def _shifted_contiguous(tensor):
    storage = torch.empty(
        tensor.numel() + 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    shifted = storage[1:].view(tensor.shape)
    shifted.copy_(tensor)
    assert shifted.is_contiguous()
    assert shifted.data_ptr() % 16 != 0
    return shifted


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "target",
    [
        "mixed_qkv",
        "raw_gate",
        "raw_beta",
        "A_log",
        "dt_bias",
        "state",
        "state_indices",
        "output",
    ],
)
def test_packed_kda_cute_shifted_tensors_match_reference(
    packed_kda_cute_device, target
):
    case = _make_case(8, packed_kda_cute_device, seed=20261191, inactive=False)
    case[target] = _shifted_contiguous(case[target])
    reference_state = case["state"].clone()
    reference_output = _reference_step(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        reference_state,
        case["state_indices"],
    )

    result = _call_cute(case, tile_v=8)
    torch.cuda.synchronize(packed_kda_cute_device)

    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tile_v", [8, 64])
def test_packed_kda_cute_out_of_range_slots_are_inactive(
    packed_kda_cute_device, tile_v
):
    case = _make_case(8, packed_kda_cute_device, seed=20261192, inactive=False)
    case["indices_host"][-2:] = [case["slots"], case["slots"] + 7]
    case["state_indices"].copy_(
        torch.tensor(
            case["indices_host"], dtype=torch.int32, device=packed_kda_cute_device
        )
    )
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

    result = _call_cute(case, tile_v=tile_v)
    torch.cuda.synchronize(packed_kda_cute_device)

    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)
    _assert_mutation_contract(case, before_storage)


@pytest.mark.arch_blackwell
def test_packed_kda_cute_all_inactive_is_bitwise_noop(packed_kda_cute_device):
    case = _make_case(1, packed_kda_cute_device, seed=20261200, inactive=False)
    case["indices_host"] = [-1]
    case["state_indices"].fill_(-1)
    before_storage = case["state_storage"].clone()

    result = _call_cute(case)
    torch.cuda.synchronize(packed_kda_cute_device)

    assert torch.equal(
        case["state_storage"].view(torch.int16), before_storage.view(torch.int16)
    )
    assert torch.equal(
        result.view(torch.int16), torch.zeros_like(result).view(torch.int16)
    )


@pytest.mark.arch_blackwell
def test_packed_kda_cute_cuda_graph_replay(packed_kda_cute_device):
    case = _make_case(8, packed_kda_cute_device, seed=20261300)
    initial_storage = case["state_storage"].clone()

    # Materialize the DSL kernel and its persistent cache before capture.
    _call_cute(case)
    torch.cuda.synchronize(packed_kda_cute_device)
    case["state_storage"].copy_(initial_storage)

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(device=packed_kda_cute_device)
    capture_stream.wait_stream(torch.cuda.current_stream(packed_kda_cute_device))
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_result = _call_cute(case)
    torch.cuda.synchronize(packed_kda_cute_device)
    assert captured_result is case["output"]

    case["state_storage"].copy_(initial_storage)
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
    torch.cuda.synchronize(packed_kda_cute_device)
    _assert_close(case["output"], reference_output)
    _assert_close(case["state"], reference_state)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", [8, 64])
def test_packed_kda_cute_uses_current_stream(packed_kda_cute_device, batch):
    case = _make_case(batch, packed_kda_cute_device, seed=20261400 + batch)
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

    stream = torch.cuda.Stream(device=packed_kda_cute_device)
    stream.wait_stream(torch.cuda.current_stream(packed_kda_cute_device))
    with torch.cuda.stream(stream):
        result = _call_cute(case)
    torch.cuda.current_stream(packed_kda_cute_device).wait_stream(stream)

    assert result is case["output"]
    _assert_close(result, reference_output)
    _assert_close(case["state"], reference_state)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", [8, 64])
def test_packed_kda_cute_cuda_graph_replays_changed_inputs_and_indices(
    packed_kda_cute_device, batch
):
    case = _make_case(batch, packed_kda_cute_device, seed=20261500 + batch)
    initial_storage = case["state_storage"].clone()

    # Compile and initialize all lazy runtime state before capture.
    _call_cute(case)
    torch.cuda.synchronize(packed_kda_cute_device)
    case["state_storage"].copy_(initial_storage)
    case["output"].fill_(123.0)

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(device=packed_kda_cute_device)
    capture_stream.wait_stream(torch.cuda.current_stream(packed_kda_cute_device))
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_result = _call_cute(case)
    torch.cuda.synchronize(packed_kda_cute_device)
    assert captured_result is case["output"]

    generator = torch.Generator(device=packed_kda_cute_device).manual_seed(
        20261600 + batch
    )
    case["mixed_qkv"].copy_(
        torch.randn(
            case["mixed_qkv"].shape,
            dtype=torch.bfloat16,
            device=packed_kda_cute_device,
            generator=generator,
        ).mul_(0.25)
    )
    case["raw_gate"].copy_(
        torch.randn(
            case["raw_gate"].shape,
            dtype=torch.bfloat16,
            device=packed_kda_cute_device,
            generator=generator,
        ).mul_(0.25)
    )
    case["raw_beta"].copy_(
        torch.randn(
            case["raw_beta"].shape,
            dtype=torch.bfloat16,
            device=packed_kda_cute_device,
            generator=generator,
        ).mul_(0.25)
    )
    changed_indices_host = [row + 4 for row in range(batch)]
    changed_indices_host[0] = -1
    case["state_indices"].copy_(
        torch.tensor(
            changed_indices_host,
            dtype=torch.int32,
            device=packed_kda_cute_device,
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
    torch.cuda.synchronize(packed_kda_cute_device)

    _assert_close(case["output"], reference_output)
    _assert_close(case["state"], reference_state)
    _assert_mutation_contract(case, initial_storage)


def _run_packed_kda_512_step_fp64_diagnostic(device):
    steps = 512
    batch = 8
    generator = torch.Generator(device=device).manual_seed(20260818)
    mixed_storage = torch.randn(
        (steps, batch, _PRODUCTION_MIXED_STRIDE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.5)
    gate_storage = torch.randn(
        (steps, batch, _GATE_WIDTH + 17),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.5)
    beta_storage = torch.randn(
        (steps, batch, _HEADS + 5),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.5)
    slots = batch + 9
    state_slot_stride = _LOGICAL_STATE_SLOT + 17
    state_storage = torch.randn(
        slots * state_slot_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.05)
    state = _state_view(state_storage, slots, state_slot_stride)
    oracle_state = state.to(torch.float64)
    state_indices = torch.tensor(
        [3, 4, 5, 6, 7, 8, 9, -1],
        dtype=torch.int32,
        device=device,
    )
    A_log = (
        torch.randn(
            _HEADS,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(0.2)
        .sub_(2.0)
    )
    dt_bias = torch.randn(
        _GATE_WIDTH,
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).mul_(0.25)
    output = torch.empty(
        (batch, 1, _HEADS, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    checkpoints = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}

    for token in range(steps):
        arguments = (
            mixed_storage[token, :, :_MIXED_WIDTH],
            gate_storage[token, :, :_GATE_WIDTH],
            beta_storage[token, :, :_HEADS],
            A_log,
            dt_bias,
            state,
            state_indices,
        )
        result = run_packed_kda_decode_cute(*arguments, output=output)
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
            torch.cuda.synchronize(device)
            _assert_close(result, oracle_output)
            _assert_close(state, oracle_state)

    output_error = (result.to(torch.float64) - oracle_output).abs()
    state_error = (state.to(torch.float64) - oracle_state).abs()
    assert float(output_error.max()) <= _ATOL
    assert float(state_error.max()) <= _ATOL


@pytest.mark.arch_blackwell
@pytest.mark.long_running
def test_packed_kda_cute_512_step_fp64_diagnostic(packed_kda_cute_device):
    _run_packed_kda_512_step_fp64_diagnostic(packed_kda_cute_device)


# ---------------------------------------------------------------------------
# recurrent_kda T=1 fast path (FLASHINFER_KDA_T1_FAST_PATH, default on)
# ---------------------------------------------------------------------------


def _recurrent_kda_views(case, batch):
    """Unpacked [B,1,H,K] views over the packed case tensors."""
    mixed = case["mixed_qkv"]
    width = _HEADS * _HEAD_DIM
    q = mixed[:, :width].view(batch, 1, _HEADS, _HEAD_DIM)
    k = mixed[:, width : 2 * width].view(batch, 1, _HEADS, _HEAD_DIM)
    v = mixed[:, 2 * width :].view(batch, 1, _HEADS, _HEAD_DIM)
    g = case["raw_gate"].view(batch, 1, _HEADS, _HEAD_DIM)
    beta = case["raw_beta"].view(batch, 1, _HEADS)
    return q, k, v, g, beta


def _call_recurrent_kda(case, batch, contiguous=False):
    from flashinfer import recurrent_kda

    q, k, v, g, beta = _recurrent_kda_views(case, batch)
    if contiguous:
        # The pre-existing grouped-CTA path silently misreads q/k/v/g views
        # whose row stride exceeds the logical row (observed on main); feed
        # it compact copies. The fast path reads the strided views directly.
        q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    out, _ = recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        initial_state=case["state"],
        ssm_state_indices=case["state_indices"],
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
        use_qk_l2norm_in_kernel=True,
    )
    return out


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", [8, 64])
def test_recurrent_kda_t1_fast_path_matches_reference(
    packed_kda_cute_device, monkeypatch, batch
):
    """Eligible recurrent_kda decode calls route to the packed kernel."""
    # inactive=False: the generic path defines no semantics for -1 rows
    # (the fast path's -1 handling is covered by the packed-kernel tests).
    case = _make_case(
        batch,
        packed_kda_cute_device,
        seed=20261700 + batch,
        state_padding=0,
        inactive=False,
    )
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

    monkeypatch.setenv("FLASHINFER_KDA_T1_FAST_PATH", "1")
    fast_out = _call_recurrent_kda(case, batch)
    torch.cuda.synchronize(packed_kda_cute_device)
    _assert_close(fast_out, reference_output)
    _assert_close(case["state"], reference_state)

    # The generic path must agree on the same inputs.
    case2 = _make_case(
        batch,
        packed_kda_cute_device,
        seed=20261700 + batch,
        state_padding=0,
        inactive=False,
    )
    monkeypatch.setenv("FLASHINFER_KDA_T1_FAST_PATH", "0")
    slow_out = _call_recurrent_kda(case2, batch, contiguous=True)
    torch.cuda.synchronize(packed_kda_cute_device)
    _assert_close(slow_out, reference_output)
    _assert_close(case2["state"], reference_state)


@pytest.mark.arch_blackwell
def test_recurrent_kda_t1_fast_path_toggle(packed_kda_cute_device, monkeypatch):
    """The env toggle switches dispatch: the fast path accepts a padded state
    pool that the generic path rejects, which proves which path ran."""
    case = _make_case(8, packed_kda_cute_device, seed=20261800, inactive=False)

    monkeypatch.setenv("FLASHINFER_KDA_T1_FAST_PATH", "1")
    _call_recurrent_kda(case, 8)
    torch.cuda.synchronize(packed_kda_cute_device)

    monkeypatch.setenv("FLASHINFER_KDA_T1_FAST_PATH", "0")
    with pytest.raises(ValueError, match="non-contiguous initial_state"):
        _call_recurrent_kda(case, 8)


@pytest.mark.arch_blackwell
def test_recurrent_kda_t1_ineligible_calls_fall_back(
    packed_kda_cute_device, monkeypatch
):
    """A pre-sigmoided-beta call is ineligible and must still work."""
    from flashinfer import recurrent_kda

    batch = 8
    case = _make_case(
        batch,
        packed_kda_cute_device,
        seed=20261900,
        state_padding=0,
        inactive=False,
    )
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

    monkeypatch.setenv("FLASHINFER_KDA_T1_FAST_PATH", "1")
    q, k, v, g, beta = _recurrent_kda_views(case, batch)
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    out, _ = recurrent_kda(
        q,
        k,
        v,
        g,
        torch.sigmoid(beta.float()).to(torch.bfloat16),
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        initial_state=case["state"],
        ssm_state_indices=case["state_indices"],
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize(packed_kda_cute_device)
    _assert_close(out, reference_output)
    _assert_close(case["state"], reference_state)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("batch", [8, 64])
def test_recurrent_kda_t1_fast_path_precomputed_gate(
    packed_kda_cute_device, monkeypatch, batch
):
    """The pre-computed convention (log-space g, sigmoided beta) is also
    routed to the fast path and matches an fp32 reference and the generic
    path."""
    from flashinfer import recurrent_kda

    device = packed_kda_cute_device
    generator = torch.Generator(device=device).manual_seed(20262000 + batch)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=device, generator=generator)

    q = randn(batch, 1, _HEADS, _HEAD_DIM).mul_(0.5)
    k = randn(batch, 1, _HEADS, _HEAD_DIM).mul_(0.5)
    v = randn(batch, 1, _HEADS, _HEAD_DIM).mul_(0.5)
    g = torch.nn.functional.logsigmoid(
        randn(batch, 1, _HEADS, _HEAD_DIM, dtype=torch.float32)
    ).to(torch.bfloat16)
    beta = torch.sigmoid(randn(batch, 1, _HEADS).float()).to(torch.bfloat16)
    slots = batch + 3
    state = randn(slots, _HEADS, _HEAD_DIM, _HEAD_DIM).mul_(0.05)
    indices = torch.arange(batch, 0, -1, dtype=torch.int32, device=device)

    # fp32 reference: decay = exp(g), beta used as-is
    qf = q.float().squeeze(1)
    kf = k.float().squeeze(1)
    vf = v.float().squeeze(1)
    decay = torch.exp(g.float().squeeze(1))
    qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * _HEAD_DIM**-0.5
    kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    bt = beta.float().squeeze(1)
    h = state.float().index_select(0, indices.long())
    hd = h * decay[:, :, None, :]
    pred = torch.einsum("bhvk,bhk->bhv", hd, kn)
    delta = (vf - pred) * bt[:, :, None]
    hn = hd + torch.einsum("bhv,bhk->bhvk", delta, kn)
    ref_out = torch.einsum("bhvk,bhk->bhv", hn, qn).unsqueeze(1)

    state_before = state.clone()

    def call():
        out, _ = recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )
        return out

    monkeypatch.setenv("FLASHINFER_KDA_T1_FAST_PATH", "1")
    fast_out = call()
    torch.cuda.synchronize(device)
    _assert_close(fast_out.view(batch, 1, _HEADS, _HEAD_DIM), ref_out)
    _assert_close(state.float().index_select(0, indices.long()), hn)
    fast_state = state.clone()

    state.copy_(state_before)
    monkeypatch.setenv("FLASHINFER_KDA_T1_FAST_PATH", "0")
    slow_out = call()
    torch.cuda.synchronize(device)
    _assert_close(slow_out.view(batch, 1, _HEADS, _HEAD_DIM), ref_out)
    _assert_close(state, fast_state)
