from __future__ import annotations

import gc
from contextlib import ExitStack
from contextvars import ContextVar

import pytest
import torch

from b12x.sequence import gdn_decode as gdn
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x as require_sm120


_case_resources = ContextVar("gdn_case_resources")


@pytest.fixture(autouse=True)
def _prepared_case_lifetime():
    with ExitStack() as resources:
        token = _case_resources.set(resources)
        try:
            yield
        finally:
            _case_resources.reset(token)


def _prepare(caps, tensors, *, restore_state=None):
    original_output = tensors["output"].clone()
    if restore_state is None:
        original_state = tensors["recurrent_state"].clone()
        restore_state = lambda: tensors["recurrent_state"].copy_(original_state)
    declaration = gdn.plan(caps, invocation=gdn.invocation_from_tensors(caps, **tensors))

    def restore():
        restore_state()
        tensors["output"].copy_(original_output)

    def prepare_call(state):
        if "scratch" not in tensors:
            (spec,) = state.layout.scratch_specs()
            tensors["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(**tensors)
        return PreparedCall(run=lambda: state.run(binding), restore=restore)

    resources = _case_resources.get()
    session = resources.enter_context(PreparationSession(device=caps.device, autotune=False, compile_workers=2))
    request = declaration.request(
        name="gdn", prepare_call=prepare_call,
    )
    resources.enter_context(session.prepare((request,)))
    return gdn.bind(declaration, **tensors)


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, device=device)
        .mul_(scale)
        .to(dtype)
        .contiguous()
    )


def _make_case(
    *,
    device: torch.device,
    query_lengths: tuple[int, ...] = (1, 1),
    max_tokens: int | None = None,
    max_seqs: int | None = None,
    state_slots: int | None = None,
    key_heads: int = 8,
    value_heads: int = 24,
    columns: int | None = None,
    accepted: tuple[int, ...] | None = None,
    activation: str = "sigmoid",
    state_dtype: torch.dtype = torch.float32,
    a_log_dtype: torch.dtype = torch.float32,
    dt_bias_dtype: torch.dtype = torch.float32,
    norm_dtype: torch.dtype = torch.bfloat16,
    qk_l2norm: bool = True,
) -> tuple[gdn.Binding, dict[str, torch.Tensor]]:
    live_seqs = len(query_lengths)
    columns = 4 if columns is None else columns
    max_seqs = 4 if max_seqs is None else max_seqs
    live_tokens = sum(query_lengths)
    max_tokens = 16 if max_tokens is None else max_tokens
    state_slots = max_seqs * columns + 1 if state_slots is None else state_slots
    if accepted is None:
        accepted = (1,) * live_seqs
    caps = gdn.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=state_slots,
        key_heads=key_heads,
        value_heads=value_heads,
        state_index_columns=columns,
        state_dtype=state_dtype,
        gate_activation=activation,
        qk_l2norm=qk_l2norm,
    )
    query_start_loc = torch.full(
        (max_seqs + 1,), live_tokens, dtype=torch.int32, device=device
    )
    query_start_loc[0] = 0
    if live_seqs:
        query_start_loc[1 : live_seqs + 1].copy_(
            torch.tensor(query_lengths, dtype=torch.int32, device=device).cumsum(0)
        )
    num_accepted_tokens = torch.ones(max_seqs, dtype=torch.int32, device=device)
    if live_seqs:
        num_accepted_tokens[:live_seqs].copy_(
            torch.tensor(accepted, dtype=torch.int32, device=device)
        )
    state_indices = torch.arange(
        max_seqs * columns, dtype=torch.int32, device=device
    ).view(max_seqs, columns)
    state_indices.remainder_(state_slots)
    tensors = {
        "mixed_qkv": _randn((max_tokens, caps.packed_qkv_width), device=device),
        "a": _randn((max_tokens, value_heads), device=device),
        "b": _randn((max_tokens, value_heads), device=device),
        "z": _randn((max_tokens, value_heads, 128), device=device),
        "A_log": _randn((value_heads,), device=device, dtype=a_log_dtype, scale=0.1),
        "dt_bias": _randn(
            (value_heads,), device=device, dtype=dt_bias_dtype, scale=0.1
        ),
        "norm_weight": (
            1.0 + _randn((128,), device=device, dtype=norm_dtype, scale=0.05)
        ).contiguous(),
        "recurrent_state": _randn(
            (state_slots, value_heads, 128, 128),
            device=device,
            dtype=state_dtype,
            scale=0.1,
        ),
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": num_accepted_tokens,
        "state_indices": state_indices,
        "num_seqs": torch.tensor([live_seqs], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([live_tokens], dtype=torch.int32, device=device),
        "output": torch.full(
            (max_tokens, value_heads, 128),
            7.0,
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    binding = _prepare(caps, tensors)
    return binding, tensors


def test_live_strided_views_are_correct_and_graph_replay_safe() -> None:
    device = require_sm120()
    full_binding, tensors = _make_case(
        device=device,
        query_lengths=(3, 2),
        activation="silu",
    )
    caps = full_binding._state.caps
    token_capacity = 5
    sequence_capacity = 2

    qkvz = torch.full(
        (token_capacity, caps.packed_qkv_width + caps.value_heads * 128),
        31.0,
        dtype=torch.bfloat16,
        device=device,
    )
    mixed_qkv = qkvz[:, : caps.packed_qkv_width]
    z = qkvz[:, caps.packed_qkv_width :].view(token_capacity, caps.value_heads, 128)
    mixed_qkv.copy_(tensors["mixed_qkv"][:token_capacity])
    z.copy_(tensors["z"][:token_capacity])

    ba = torch.full(
        (token_capacity, 2 * caps.value_heads),
        31.0,
        dtype=torch.bfloat16,
        device=device,
    )
    b, a = ba.chunk(2, dim=-1)
    b.copy_(tensors["b"][:token_capacity])
    a.copy_(tensors["a"][:token_capacity])

    output_storage = torch.full(
        (token_capacity, caps.value_heads * 128 + 8),
        31.0,
        dtype=torch.bfloat16,
        device=device,
    )
    output = output_storage[:, : caps.value_heads * 128].view(
        token_capacity, caps.value_heads, 128
    )
    state_index_storage = torch.full(
        (sequence_capacity, caps.state_index_columns + 2),
        -1,
        dtype=torch.int32,
        device=device,
    )
    state_indices = state_index_storage[:, : caps.state_index_columns]
    state_indices.copy_(tensors["state_indices"][:sequence_capacity])

    live_tensors = {
        **tensors,
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "z": z,
        "query_start_loc": tensors["query_start_loc"][: sequence_capacity + 1],
        "num_accepted_tokens": tensors["num_accepted_tokens"][:sequence_capacity],
        "state_indices": state_indices,
        "output": output,
    }
    binding = gdn.bind(full_binding.plan, **live_tensors)
    assert binding.mixed_qkv.stride(0) == qkvz.stride(0)
    assert binding.a.stride(0) == ba.stride(0)
    assert binding.b.stride(0) == ba.stride(0)
    assert binding.output.stride(0) == output_storage.stride(0)
    assert binding.state_indices.stride(0) == state_index_storage.stride(0)

    initial_state = binding.recurrent_state.clone()
    expected_state = initial_state.clone()
    expected = _reference(binding, expected_state)

    gdn.run(full_binding)
    contiguous_output = full_binding.output[:token_capacity].clone()
    contiguous_state = full_binding.recurrent_state.clone()
    binding.recurrent_state.copy_(initial_state)
    gdn.run(binding)
    torch.cuda.synchronize(device)
    binding.recurrent_state.copy_(initial_state)
    output.fill_(31.0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gdn.run(gdn.bind(full_binding.plan, **live_tensors))
    binding.recurrent_state.copy_(initial_state)
    output.fill_(31.0)
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(output, contiguous_output, rtol=0, atol=0)
    torch.testing.assert_close(
        binding.recurrent_state, contiguous_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        binding.recurrent_state, expected_state, rtol=1e-5, atol=2e-5
    )
    assert torch.count_nonzero(output_storage[:, -8:] != 31.0) == 0
    assert torch.count_nonzero(state_index_storage[:, -2:] != -1) == 0


@pytest.mark.parametrize("max_seqs", (4, 16))
def test_bounded_views_reuse_planned_kernels(
    monkeypatch,
    max_seqs: int,
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.sequence.gdn_decode import _kernels

    device = require_sm120()
    full_binding, tensors = _make_case(
        device=device, query_lengths=(4, 4), max_seqs=max_seqs
    )
    initial_state = full_binding.recurrent_state.clone()
    gdn.run(full_binding)

    def reject_resolution(*args, **kwargs):
        pytest.fail("Bound capacities must reuse warmed planned kernels")

    monkeypatch.setattr(_kernels._gated_rmsnorm_kernel, "_do_compile", reject_resolution)

    with kernel_resolution_guard('Qwen GDN bounded-view replay qualification'):
        for rows, requests, columns in ((1, 1, 1), (4, 1, 4), (8, 2, 4), (2, 1, 2)):
            live = dict(tensors)
            for name in ("mixed_qkv", "a", "b", "z", "output"):
                source = tensors[name][:rows]
                width = source[0].numel()
                if rows == 2 and name in ("z", "output"):
                    # Cross the signed 32-bit element-offset boundary.
                    stride = (1 << 31) + 128
                    storage = torch.empty(
                        stride + width, dtype=source.dtype, device=device
                    )
                    live[name] = storage.as_strided(source.shape, (stride, 128, 1))
                else:
                    storage = torch.full(
                        (rows, width + rows + 1),
                        31.0,
                        dtype=source.dtype,
                        device=device,
                    )
                    live[name] = storage[:, 1 : width + 1].view(source.shape)
                live[name].copy_(source)
            if rows == 2:
                stride = (1 << 31) + 8
                index_storage = torch.empty(
                    stride + 1, dtype=torch.int32, device=device
                )
                live["state_indices"] = index_storage.as_strided(
                    (requests, columns), (stride * columns, stride)
                )
            else:
                index_storage = torch.empty(
                    (requests, 2 * columns + 3), dtype=torch.int32, device=device
                )
                live["state_indices"] = index_storage[:, 1 : 2 * columns + 1 : 2]
            live["state_indices"].copy_(tensors["state_indices"][:requests, :columns])
            live["query_start_loc"] = tensors["query_start_loc"][: requests + 1]
            live["num_accepted_tokens"] = tensors["num_accepted_tokens"][:requests]
            live["query_start_loc"].copy_(
                torch.arange(requests + 1, dtype=torch.int32, device=device)
                * (rows // requests)
            )
            live["num_tokens"].fill_(rows)
            live["num_seqs"].fill_(requests)
            live["num_accepted_tokens"].fill_(1)
            binding = gdn.bind(full_binding.plan, **live)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gdn.run(gdn.bind(full_binding.plan, **live))

            live["mixed_qkv"].mul_(0.75)
            binding.recurrent_state.copy_(initial_state)
            expected_state = initial_state.clone()
            expected = _reference(binding, expected_state)
            addresses = tuple(live[name].data_ptr() for name in live)
            graph.replay()
            torch.cuda.synchronize(device)
            assert addresses == tuple(live[name].data_ptr() for name in live)
            torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
            torch.testing.assert_close(
                binding.recurrent_state, expected_state, rtol=1e-5, atol=2e-5
            )


def _reference(
    binding: gdn.Binding, state: torch.Tensor, *, scale: float | None = None
) -> torch.Tensor:
    caps = binding._state.caps
    return gdn.reference.decode(
        binding.mixed_qkv,
        binding.a,
        binding.b,
        binding.z,
        binding.A_log,
        binding.dt_bias,
        binding.norm_weight,
        state,
        binding.query_start_loc,
        binding.num_accepted_tokens,
        binding.state_indices,
        binding.num_seqs,
        binding.num_tokens,
        key_heads=caps.key_heads,
        value_heads=caps.value_heads,
        scale=scale,
        gate_activation=caps.gate_activation,
        qk_l2norm=caps.qk_l2norm,
    )


@pytest.mark.parametrize(
    "state_dtype",
    (
        pytest.param(torch.float32, id="fp32-state"),
        pytest.param(torch.bfloat16, id="bf16-state"),
    ),
)
def test_research_qwen_cute_stages_are_graph_safe_and_correct(
    state_dtype: torch.dtype,
) -> None:
    from b12x.sequence.gdn_decode._cute_kernels import (
        run_gated_rmsnorm,
        run_packed_recurrent_qwen,
    )

    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(1, 1, 1, 1),
        key_heads=8,
        value_heads=24,
        activation="sigmoid",
        state_dtype=state_dtype,
    )
    initial_state = binding.recurrent_state.clone()
    expected_state = initial_state.clone()
    expected = _reference(binding, expected_state)

    def launch() -> None:
        run_packed_recurrent_qwen(binding)
        run_gated_rmsnorm(binding, eps=1.0e-6)

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    binding.recurrent_state.copy_(initial_state)
    binding.output.fill_(float("nan"))
    binding.scratch.fill_(0xFF)
    addresses = (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert allocated_after == allocated_before
    assert addresses == (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
    state_rtol = 1e-2 if state_dtype == torch.bfloat16 else 1e-5
    state_atol = 8e-3 if state_dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        binding.recurrent_state,
        expected_state,
        rtol=state_rtol,
        atol=state_atol,
    )


@pytest.mark.parametrize(
    "query_lengths",
    (
        (1,),
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (4,),
        (4, 2, 1, 3),
        (4, 4, 4, 4),
    ),
)
def test_public_qwen38_planned_capacity_uses_correct_recurrence(
    query_lengths: tuple[int, ...],
) -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=query_lengths,
        max_tokens=16,
        max_seqs=4,
        columns=4,
        state_slots=17,
        key_heads=8,
        value_heads=24,
        activation="sigmoid",
        state_dtype=torch.float32,
    )
    initial_state = binding.recurrent_state.clone()
    expected_state = initial_state.clone()
    expected = _reference(binding, expected_state)

    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, expected_state, rtol=1e-5, atol=2e-5
    )


@pytest.mark.parametrize(
    "key_heads,value_heads",
    (
        pytest.param(16, 48, id="16qk-48v"),
        pytest.param(8, 24, id="8qk-24v"),
        pytest.param(4, 12, id="4qk-12v"),
    ),
)
def test_public_qwen38_sharded_head_geometries_use_cute_recurrence(
    key_heads: int,
    value_heads: int,
) -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(4, 2, 1, 3),
        key_heads=key_heads,
        value_heads=value_heads,
    )
    initial_state = binding.recurrent_state.clone()
    expected_state = initial_state.clone()
    expected = _reference(binding, expected_state)

    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, expected_state, rtol=1e-5, atol=2e-5
    )


def test_public_qwen38_cute_path_is_graph_safe_without_replay_allocation() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(4, 2, 1, 3),
        max_tokens=16,
        max_seqs=4,
        columns=4,
        state_slots=17,
        key_heads=16,
        value_heads=48,
        activation="sigmoid",
        state_dtype=torch.float32,
    )
    initial_state = binding.recurrent_state.clone()
    expected_state = initial_state.clone()
    expected = _reference(binding, expected_state)

    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gdn.run(binding)

    binding.recurrent_state.copy_(initial_state)
    binding.output.fill_(float("nan"))
    binding.scratch.fill_(0xFF)
    addresses = (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert allocated_after == allocated_before
    assert addresses == (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, expected_state, rtol=1e-5, atol=2e-5
    )


def _transformers_kv_state_reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    key_heads: int,
    value_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Decode a mathematical ``[slot, head, key_dim, value_dim]`` state."""
    key_dim = value_dim = 128
    ratio = value_heads // key_heads
    q_width = key_heads * key_dim
    q = mixed_qkv[:, :q_width].view(-1, key_heads, key_dim).float()
    k = mixed_qkv[:, q_width : 2 * q_width].view(-1, key_heads, key_dim).float()
    v = mixed_qkv[:, 2 * q_width :].view(-1, value_heads, value_dim).float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q *= key_dim**-0.5
    output = torch.zeros(
        (mixed_qkv.shape[0], value_heads, value_dim), dtype=torch.bfloat16
    )

    for token in range(int(mixed_qkv.shape[0])):
        slot = int(state_indices[token, 0])
        for value_head in range(value_heads):
            key_head = value_head // ratio
            state = recurrent_state[slot, value_head].float()
            softplus_input = a[token, value_head].float() + dt_bias[value_head].float()
            softplus = torch.where(
                softplus_input <= 20.0,
                torch.log1p(torch.exp(softplus_input)),
                softplus_input,
            )
            decay = torch.exp(-torch.exp(A_log[value_head].float()) * softplus)
            beta = (
                torch.sigmoid(b[token, value_head].float()).to(torch.bfloat16).float()
            )
            state *= decay
            delta = v[token, value_head] - (
                state * k[token, key_head].unsqueeze(-1)
            ).sum(dim=-2)
            state += k[token, key_head].unsqueeze(-1) * (delta * beta).unsqueeze(-2)
            decoded = (state * q[token, key_head].unsqueeze(-1)).sum(dim=-2)
            output[token, value_head].copy_(decoded.to(torch.bfloat16))
            recurrent_state[slot, value_head].copy_(state.to(recurrent_state.dtype))

    values = output.float()
    values *= torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    values = values.to(torch.bfloat16) * norm_weight
    return (values * torch.sigmoid(z.float())).to(torch.bfloat16)


def test_v_by_k_padded_slot_layout_matches_transposed_mathematical_oracle() -> None:
    torch.manual_seed(17)
    tokens, slots, key_heads, value_heads = 2, 2, 1, 3
    width = 2 * key_heads * 128 + value_heads * 128
    cpu = torch.device("cpu")
    mixed_qkv = _randn((tokens, width), device=cpu)
    a = _randn((tokens, value_heads), device=cpu)
    b = _randn((tokens, value_heads), device=cpu)
    z = _randn((tokens, value_heads, 128), device=cpu)
    A_log = _randn((value_heads,), device=cpu, dtype=torch.float32, scale=0.1)
    dt_bias = _randn((value_heads,), device=cpu, dtype=torch.float32, scale=0.1)
    norm_weight = 1.0 + _randn((128,), device=cpu, scale=0.05)
    slot_elements = value_heads * 128 * 128
    slot_stride = slot_elements + 37
    storage_offset = 19
    state_storage = torch.full(
        (storage_offset + (slots - 1) * slot_stride + slot_elements,),
        91.0,
        dtype=torch.float32,
        device=cpu,
    )
    state_vk = torch.as_strided(
        state_storage,
        size=(slots, value_heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
        storage_offset=storage_offset,
    )
    state_vk.copy_(
        _randn(
            (slots, value_heads, 128, 128),
            device=cpu,
            dtype=torch.float32,
            scale=0.1,
        )
    )
    state_kv = state_vk.transpose(-1, -2).contiguous()
    state_indices = torch.tensor([[0], [1]], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    accepted = torch.ones(2, dtype=torch.int32)

    expected = _transformers_kv_state_reference(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state_kv,
        state_indices,
        key_heads=key_heads,
        value_heads=value_heads,
    )
    actual = gdn.reference.decode(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state_vk,
        query_start_loc,
        accepted,
        state_indices,
        2,
        2,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_activation="sigmoid",
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        state_vk.transpose(-1, -2), state_kv, rtol=1e-5, atol=2e-8
    )
    assert torch.count_nonzero(state_storage[:storage_offset] != 91.0) == 0
    gap_start = storage_offset + slot_elements
    gap_end = storage_offset + slot_stride
    assert torch.count_nonzero(state_storage[gap_start:gap_end] != 91.0) == 0


@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [128**-0.5, 1.0])
def test_qwen3_8_flash_next_output_norm_preserves_parameter_dtype_rounding(
    norm_dtype: torch.dtype,
    scale: float,
) -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(1,),
        norm_dtype=norm_dtype,
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference, scale=scale)

    actual = gdn.run(binding, scale=scale)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_qwen_binding_rejects_equal_head_kda_execution() -> None:
    device = require_sm120()
    with pytest.raises(ValueError):
        _make_case(device=device, query_lengths=(1,), key_heads=1, value_heads=1)


@pytest.mark.parametrize("query_lengths", [(4,), (4, 4)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_small_softplus_with_large_rate_preserves_decay(query_lengths, state_dtype) -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=query_lengths,
                           key_heads=2, value_heads=6, state_dtype=state_dtype)
    binding.a.fill_(-20)
    binding.A_log.fill_(20)
    binding.dt_bias.zero_()
    initial = binding.recurrent_state.clone()
    state_reference = initial.clone()
    expected = _reference(binding, state_reference)
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gdn.run(binding)
    binding.recurrent_state.copy_(initial)
    binding.output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
    state_rtol = 1e-2 if state_dtype == torch.bfloat16 else 1e-5
    state_atol = 8e-3 if state_dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(binding.recurrent_state, state_reference, rtol=state_rtol, atol=state_atol)


def test_qwen_bf16_state_uses_cute_recurrence() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(4, 2, 1, 3),
        state_dtype=torch.bfloat16,
        dt_bias_dtype=torch.bfloat16,
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-2, atol=8e-3
    )


def test_padded_state_stride_matches_reference_without_copy() -> None:
    device = require_sm120()
    state_slots = 17
    key_heads = 8
    value_heads = 24
    binding, tensors = _make_case(
        device=device,
        query_lengths=(2, 1),
        state_slots=state_slots,
        key_heads=key_heads,
        value_heads=value_heads,
        state_dtype=torch.float32,
    )

    storage_offset = 25_600 // torch.float32.itemsize
    slot_elements = value_heads * 128 * 128
    slot_stride = slot_elements + 2_048
    state_storage = torch.full(
        (storage_offset + (state_slots - 1) * slot_stride + slot_elements,),
        91.0,
        dtype=torch.float32,
        device=device,
    )
    recurrent_state = torch.as_strided(
        state_storage,
        size=(state_slots, value_heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
        storage_offset=storage_offset,
    )
    recurrent_state.copy_(
        _randn(
            tuple(recurrent_state.shape),
            device=device,
            dtype=torch.float32,
            scale=0.1,
        )
    )
    tensors["recurrent_state"] = recurrent_state
    binding = gdn.bind(binding.plan, **tensors)

    state_reference = recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    assert binding.recurrent_state.data_ptr() == recurrent_state.data_ptr()
    assert binding.recurrent_state.stride(0) == slot_stride
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(recurrent_state, state_reference, rtol=1e-5, atol=2e-5)
    assert torch.count_nonzero(state_storage[:storage_offset] != 91.0) == 0
    for slot in range(state_slots - 1):
        gap_start = storage_offset + slot * slot_stride + slot_elements
        gap_end = storage_offset + (slot + 1) * slot_stride
        assert torch.count_nonzero(state_storage[gap_start:gap_end] != 91.0) == 0


def test_bind_rejects_noncontiguous_state_slot_contents() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device)
    tensors["recurrent_state"] = tensors["recurrent_state"].transpose(-1, -2)

    with pytest.raises(ValueError, match="contiguous within each state slot"):
        gdn.bind(binding.plan, **tensors)


def test_bind_rejects_insufficient_packed_metadata_capacity() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device, max_tokens=4, max_seqs=2, columns=2)
    tensors["state_indices"] = tensors["state_indices"][:1]
    with pytest.raises(ValueError, match=r"4 > 1 \* 2"):
        gdn.bind(binding.plan, **tensors)


def test_bind_rejects_overlapping_state_slots() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device)
    shape = tuple(tensors["recurrent_state"].shape)
    slots, value_heads, value_dim, key_dim = shape
    slot_elements = value_heads * value_dim * key_dim
    state_storage = torch.empty(
        slots * slot_elements,
        dtype=binding.recurrent_state.dtype,
        device=device,
    )
    tensors["recurrent_state"] = torch.as_strided(
        state_storage,
        size=shape,
        stride=(slot_elements - 1, value_dim * key_dim, key_dim, 1),
    )

    with pytest.raises(ValueError, match="slots must not overlap"):
        gdn.bind(binding.plan, **tensors)


def test_accepted_column_selects_rollback_checkpoint() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(3,),
        accepted=(3,),
    )
    binding.recurrent_state[0].fill_(0.25)
    binding.recurrent_state[1].fill_(-0.5)
    binding.recurrent_state[2].fill_(0.75)
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_rejected_draft_restarts_next_iteration_from_accepted_checkpoint() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(3,),
        accepted=(1,),
    )
    gdn.run(binding)
    torch.cuda.synchronize(device)
    accepted_checkpoint = binding.recurrent_state[1].clone()

    binding.num_accepted_tokens.fill_(2)
    binding.recurrent_state[2].fill_(73.0)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.a.copy_(torch.randn_like(binding.a).mul_(0.2))
    binding.b.copy_(torch.randn_like(binding.b).mul_(0.2))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.2))
    state_reference = binding.recurrent_state.clone()
    torch.testing.assert_close(state_reference[1], accepted_checkpoint, rtol=0, atol=0)
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_bind_rejects_scratch_alias_with_mutable_output() -> None:
    device = require_sm120()
    binding, tensors = _make_case(device=device)
    output_nbytes = binding.output.numel() * binding.output.element_size()
    scratch_nbytes = binding._state.scratch_specs()[0].nbytes
    shared = torch.empty(
        max(output_nbytes, scratch_nbytes), dtype=torch.uint8, device=device
    )
    aliased_output = shared[:output_nbytes].view(torch.bfloat16).view_as(binding.output)
    tensors.update(scratch=shared, output=aliased_output)

    with pytest.raises(ValueError, match="scratch and output"):
        gdn.bind(binding.plan, **tensors)


def test_distant_state_slots_across_requests_match_reference() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device, query_lengths=(1, 1), max_seqs=16,
        state_slots=129, key_heads=1, value_heads=3,
    )
    binding.state_indices[:2, 0].copy_(
        torch.tensor([0, 128], dtype=torch.int32, device=device)
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_zero_length_request_and_capacity_tail_are_zero() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(2, 0),
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    assert torch.count_nonzero(actual[2:]) == 0


def test_cuda_graph_replay_uses_device_counts_and_fixed_addresses() -> None:
    device = require_sm120()
    binding, _ = _make_case(
        device=device,
        query_lengths=(2, 1),
        state_slots=17,
        accepted=(2, 1),
    )
    gdn.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run(binding)

    binding.query_start_loc.copy_(
        torch.tensor([0, 1, 3, 3, 3], dtype=torch.int32, device=device)
    )
    binding.num_seqs.fill_(2)
    binding.num_tokens.fill_(3)
    binding.num_accepted_tokens[:2].copy_(
        torch.tensor([1, 2], dtype=torch.int32, device=device)
    )
    binding.state_indices[:2].copy_(
        torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7]], dtype=torch.int32, device=device)
    )
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.25))
    binding.a.copy_(torch.randn_like(binding.a).mul_(0.25))
    binding.b.copy_(torch.randn_like(binding.b).mul_(0.25))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.25))
    binding.recurrent_state.copy_(torch.randn_like(binding.recurrent_state).mul_(0.1))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert captured_output.data_ptr() == binding.output.data_ptr()
    assert allocated_after == allocated_before
    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_torch_compile_fullgraph_keeps_outer_op_opaque() -> None:
    device = require_sm120()
    binding, _ = _make_case(device=device, query_lengths=(2,), accepted=(2,))

    def launch() -> torch.Tensor:
        return gdn.run(binding)

    launch()
    compiled = torch.compile(launch, fullgraph=True)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.25))
    binding.recurrent_state.copy_(torch.randn_like(binding.recurrent_state).mul_(0.1))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    actual = compiled()
    torch.cuda.synchronize(device)
    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_qwen_grouped_state_slot_offset_past_int32_boundary() -> None:
    device = require_sm120()
    key_heads, value_heads = 8, 24
    slot_elements = value_heads * 128 * 128
    slot_stride = slot_elements + 2_048
    tail_slot = (1 << 31) // slot_stride + 1
    assert tail_slot * slot_stride > 1 << 31
    caps = gdn.Caps(
        device=device,
        max_tokens=16,
        max_seqs=4,
        max_state_slots=tail_slot + 1,
        key_heads=key_heads,
        value_heads=value_heads,
        state_index_columns=4,
        state_dtype=torch.float32,
        gate_activation="sigmoid",
    )
    mixed_qkv = _randn((16, caps.packed_qkv_width), device=device)
    a = _randn((16, value_heads), device=device)
    b = _randn((16, value_heads), device=device)
    z = _randn((16, value_heads, 128), device=device)
    A_log = _randn((value_heads,), device=device, dtype=torch.float32, scale=0.1)
    dt_bias = _randn((value_heads,), device=device, dtype=torch.float32, scale=0.1)
    norm_weight = (1.0 + _randn((128,), device=device, scale=0.05)).contiguous()
    state_storage = torch.empty(
        tail_slot * slot_stride + slot_elements,
        dtype=torch.float32,
        device=device,
    )
    recurrent_state = torch.as_strided(
        state_storage,
        size=(tail_slot + 1, value_heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
    )
    recurrent_state[tail_slot].copy_(
        _randn(
            (value_heads, 128, 128),
            device=device,
            dtype=torch.float32,
            scale=0.1,
        )
    )
    compact_reference_state = recurrent_state[tail_slot : tail_slot + 1].clone()
    indices = torch.zeros((4, 4), dtype=torch.int32, device=device)
    indices[0, 0] = tail_slot
    compact_indices = torch.zeros((4, 4), dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32, device=device)
    accepted = torch.ones(4, dtype=torch.int32, device=device)
    num_seqs = torch.ones(1, dtype=torch.int32, device=device)
    num_tokens = torch.ones(1, dtype=torch.int32, device=device)
    output = torch.empty((16, value_heads, 128), dtype=torch.bfloat16, device=device)
    tensors = dict(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        z=z,
        A_log=A_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        recurrent_state=recurrent_state,
        query_start_loc=query_start_loc,
        num_accepted_tokens=accepted,
        state_indices=indices,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        output=output,
    )
    binding = _prepare(
        caps, tensors,
        restore_state=lambda: recurrent_state[tail_slot : tail_slot + 1].copy_(compact_reference_state),
    )
    expected = gdn.reference.decode(
        mixed_qkv,
        a,
        b,
        z,
        A_log,
        dt_bias,
        norm_weight,
        compact_reference_state,
        query_start_loc,
        accepted,
        compact_indices,
        num_seqs,
        num_tokens,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_activation="sigmoid",
    )

    actual = gdn.run(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        recurrent_state[tail_slot],
        compact_reference_state[0],
        rtol=1e-5,
        atol=2e-5,
    )

    del binding, recurrent_state, state_storage
    gc.collect()
    torch.cuda.empty_cache()


def test_caps_accept_divisible_head_ratios_and_reject_invalid_capacity() -> None:
    device = require_sm120()
    caps = gdn.Caps(
        device=device,
        max_tokens=1,
        max_seqs=1,
        max_state_slots=1,
        key_heads=1,
        value_heads=5,
    )
    assert caps.value_heads_per_key_head == 5
    with pytest.raises(ValueError, match="at most 8"):
        gdn.Caps(
            device=device,
            max_tokens=1,
            max_seqs=1,
            max_state_slots=1,
            key_heads=1,
            value_heads=1,
            state_index_columns=9,
        )
    with pytest.raises(ValueError, match="must fit"):
        gdn.Caps(
            device=device,
            max_tokens=3,
            max_seqs=2,
            max_state_slots=2,
            key_heads=1,
            value_heads=1,
            state_index_columns=1,
        )
