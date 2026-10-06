from __future__ import annotations

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import replace
import pytest
import torch

from b12x.sequence import ple, ple_hash
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence.ple.reference import (
    ple_projected_packed_reference,
    ple_projected_sequence_reference,
    ple_projected_u_reference,
)
from b12x.sequence.ple_hash.reference import (
    eos_bounded_windows,
    ple_hash_ids_reference,
    ple_hash_packed_reference,
    ple_multipliers,
    ple_table_geometry,
)

from ..conftest import require_b12x


_case_resources = ContextVar("ple_case_resources")


@pytest.fixture(autouse=True)
def _prepared_case_lifetime():
    with ExitStack() as resources:
        token = _case_resources.set(resources)
        try:
            yield
        finally:
            _case_resources.reset(token)


def _hash_caps(*, device) -> ple_hash.Caps:
    return ple_hash.Caps(
        device=device,
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )


def test_ple_hash_geometry_is_distinct_deterministic_and_aligned() -> None:
    caps = ple_hash.Caps(
        device="cpu",
        max_tokens=8,
        max_seqs=3,
        vocab_size=1000,
        eos_token_id=999,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=100,
        table_alignment=128,
    )
    geometry = ple_hash.compute_geometry(caps)

    assert geometry.prime_sizes == (101, 103, 107, 109)
    assert geometry.table_offsets == (0, 101, 204, 311)
    assert geometry.padded_vocab_size == 512
    expected_multipliers = (
        5159850018220775,
        4902196785138501,
        6891410296393783,
    )
    assert geometry.multipliers == expected_multipliers
    assert (
        ple_multipliers(
            vocab_size=1000,
            max_order=3,
            dense_layer_ordinal=0,
        ).tolist()
        == list(expected_multipliers)
    )
    assert all(value & 1 for value in geometry.multipliers)
    assert max(geometry.multipliers) <= ((1 << 63) - 1) // 1000

    next_sizes, _ = ple_table_geometry(
        base_size=100,
        dense_layer_ordinal=1,
        total_heads=4,
    )
    assert next_sizes.tolist() == [113, 127, 131, 137]
    minimum_sizes, minimum_offsets = ple_table_geometry(
        base_size=1,
        dense_layer_ordinal=0,
        total_heads=4,
    )
    assert minimum_sizes.tolist() == [2, 3, 5, 7]
    assert minimum_offsets.tolist() == [0, 2, 5, 10]


def test_ple_hash_geometry_rejects_cumulative_table_extent_beyond_int64() -> None:
    caps = replace(_hash_caps(device="cpu"), max_order=2, heads_per_order=2)
    with pytest.raises(ValueError):
        ple_hash.compute_geometry(
            caps,
            prime_sizes=torch.tensor([101, 9223372036854775783], dtype=torch.int64),
            table_offsets=torch.tensor([0, 101], dtype=torch.int64),
            multipliers=torch.tensor([1, 3], dtype=torch.int64),
        )


def test_ple_hash_geometry_rejects_padded_table_extent_beyond_int64() -> None:
    caps = replace(_hash_caps(device="cpu"), max_order=2, heads_per_order=1)
    with pytest.raises(ValueError):
        ple_hash.compute_geometry(
            caps,
            prime_sizes=torch.tensor([9223372036854775783], dtype=torch.int64),
            table_offsets=torch.tensor([0], dtype=torch.int64),
            multipliers=torch.tensor([1, 3], dtype=torch.int64),
        )


def test_ple_eos_bounded_windows_reset_left_context() -> None:
    windows = eos_bounded_windows(
        torch.tensor([4, 5, 99, 6], dtype=torch.int64),
        eos_token_id=99,
        max_order=3,
    )
    assert windows[2].tolist() == [[99, 4], [4, 5], [5, 99], [99, 6]]
    assert windows[3].tolist() == [
        [99, 99, 4],
        [99, 4, 5],
        [4, 5, 99],
        [99, 99, 6],
    ]


def test_ple_packed_hash_matches_per_request_complete_history() -> None:
    eos = 99
    token_ids = torch.tensor([9, eos, 10, 3, 4], dtype=torch.int64)
    starts = torch.tensor([0, 3, 5], dtype=torch.int32)
    history = torch.tensor([[7, 8], [eos, eos]], dtype=torch.int64)
    multipliers = torch.tensor([11, 13, 17], dtype=torch.int64)
    sizes = torch.tensor([101, 103, 107, 109], dtype=torch.int64)
    offsets = torch.tensor([0, 101, 204, 311], dtype=torch.int64)

    actual = ple_hash_packed_reference(
        token_ids,
        starts,
        history,
        eos_token_id=eos,
        multipliers=multipliers,
        prime_sizes=sizes,
        table_offsets=offsets,
        heads_per_order=2,
    )
    expected_parts = []
    for committed, query in ((history[0], token_ids[:3]), (history[1], token_ids[3:])):
        windows = eos_bounded_windows(
            torch.cat((committed, query)),
            eos_token_id=eos,
            max_order=3,
        )
        expected_parts.append(
            ple_hash_ids_reference(
                {order: rows[2:] for order, rows in windows.items()},
                multipliers=multipliers,
                prime_sizes=sizes,
                table_offsets=offsets,
                heads_per_order=2,
            )
        )
    torch.testing.assert_close(actual, torch.cat(expected_parts))




@pytest.mark.parametrize("alias_kind", ["out_input", "scratch_input"])
@torch.inference_mode()
def test_ple_hash_prepared_bind_rejects_read_only_aliases(alias_kind: str) -> None:
    device = require_b12x()
    caps = _hash_caps(device=device)
    inputs = dict(
        token_ids=torch.tensor([1, 2], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
        committed_history=torch.tensor([[99, 99]], dtype=torch.int64, device=device),
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([2], dtype=torch.int32, device=device),
        out=torch.empty((2, 4), dtype=torch.int64, device=device),
    )
    _, binding = _bind_cuda_hash(caps, **inputs)
    if alias_kind == "out_input":
        shared = torch.empty(8, dtype=torch.int64, device=device)
        inputs["token_ids"] = shared[:2]
        inputs["out"] = shared.view(2, 4)
        match = "out.*read-only tensor token_ids"
    else:
        # The hash scratch may be smaller than two int64 token IDs; a larger
        # scratch buffer is accepted, so size it to hold the aliased input.
        shared = torch.empty(
            max(binding.scratch.numel(), 16), dtype=torch.uint8, device=device
        )
        inputs["scratch"] = shared
        inputs["token_ids"] = shared[:16].view(torch.int64)
        match = "scratch.*read-only tensor token_ids"

    inputs.setdefault("scratch", binding.scratch)
    with pytest.raises(ValueError, match=match):
        ple_hash.bind(binding.plan, **inputs)


@torch.inference_mode()
def test_ple_hash_prepared_bind_rejects_geometry_alias() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=1,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    inputs = dict(
        token_ids=torch.tensor([1], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        committed_history=torch.tensor([[99, 99]], dtype=torch.int64, device=device),
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        out=torch.empty((1, 4), dtype=torch.int64, device=device),
    )
    _, binding = _bind_cuda_hash(caps, **inputs)
    inputs["out"] = binding.geometry.prime_sizes.view(1, 4)

    inputs["scratch"] = binding.scratch
    with pytest.raises(ValueError, match="out.*read-only tensor prime_sizes"):
        ple_hash.bind(binding.plan, **inputs)

def _projected_inputs(tokens: int, streams: int, hidden: int):
    generator = torch.Generator().manual_seed(4107)
    residual = torch.randn(tokens, streams, hidden, generator=generator).to(
        torch.bfloat16
    )
    key = torch.randn(tokens, streams, hidden, generator=generator).to(torch.bfloat16)
    value = torch.randn(tokens, hidden, generator=generator).to(torch.bfloat16)
    weights = [
        (torch.randn(streams, hidden, generator=generator) / 10).to(torch.bfloat16)
        for _ in range(3)
    ]
    return residual, key, value, weights


def _cuda_projected_inputs(
    tokens: int,
    streams: int,
    hidden: int,
    *,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    residual = torch.randn(
        (tokens, streams, hidden),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(residual)
    value = torch.randn(
        (tokens, hidden),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = [
        (
            torch.randn(
                (streams * hidden,),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            / 32
        ).contiguous()
        for _ in range(3)
    ]
    return residual, key, value, weights, generator


def _bind_cuda_hash(
    caps: ple_hash.Caps, **tensors: torch.Tensor
) -> tuple[ple_hash.GeometryTensors, ple_hash.Binding]:
    geometry = ple_hash.compute_geometry(caps)
    geometry_tensors = ple_hash.allocate_geometry(geometry, device=caps.device)
    declaration = ple_hash.plan(
        caps,
        geometry=geometry,
        prime_sizes=geometry_tensors.prime_sizes,
        table_offsets=geometry_tensors.table_offsets,
        multipliers=geometry_tensors.multipliers,
        invocation=ple_hash.invocation_from_tensors(**tensors),
    )

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        tensors["scratch"] = torch.empty(
            spec.shape, dtype=spec.dtype, device=spec.device
        )
        trial = state.bind(**tensors)
        return PreparedCall(run=lambda: state.run(trial))

    resources = _case_resources.get()
    session = resources.enter_context(
        PreparationSession(device=caps.device, autotune=False, compile_workers=2)
    )
    request = declaration.request(
        name="ple_hash",
        prepare_call=prepare_call,
    )
    session.prepare((request,))
    return geometry_tensors, ple_hash.bind(declaration, **tensors)


def _bind_cuda_layer(
    *,
    mode: str,
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weights: list[torch.Tensor],
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_seqs: int,
    num_tokens: int,
    conv_state: torch.Tensor,
    max_speculative_tokens: int,
    dilation: int,
    request_is_prefill: torch.Tensor | None = None,
):
    max_tokens, streams, hidden = residual.shape
    max_seqs = int(state_slot_ids.numel())
    caps = ple.Caps(
        device=residual.device, mode=mode, max_tokens=max_tokens, max_seqs=max_seqs,
        max_state_slots=conv_state.shape[0], max_speculative_tokens=max_speculative_tokens,
        streams=streams, hidden_size=hidden, kernel_size=conv_weight.shape[-1], dilation=dilation,
    )
    out = torch.full_like(residual, 91)
    tensors = dict(
        residual=residual,
        key=key,
        value=value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=torch.tensor([num_seqs], dtype=torch.int32, device=residual.device),
        num_tokens=torch.tensor(
            [num_tokens], dtype=torch.int32, device=residual.device
        ),
        conv_state=conv_state,
        out=out,
        request_is_prefill=request_is_prefill,
    )
    declaration = ple.plan(caps, invocation=ple.invocation_from_tensors(**tensors))
    slots = sorted({int(slot) for slot in state_slot_ids.tolist() if 0 <= int(slot) < conv_state.shape[0]})
    indices = torch.tensor(slots, dtype=torch.int64, device=residual.device)
    original_state = conv_state.index_select(0, indices)
    original_output = out.clone()

    def restore():
        conv_state.index_copy_(0, indices, original_state)
        out.copy_(original_output)

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        tensors["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        trial = state.bind(**tensors)
        return PreparedCall(run=lambda: state.run(trial, eps=1e-6), restore=restore)

    resources = _case_resources.get()
    session = resources.enter_context(PreparationSession(device=residual.device, autotune=False, compile_workers=2))
    request = declaration.request(name="ple", prepare_call=prepare_call)
    session.prepare((request,))
    binding = ple.bind(declaration, **tensors)
    return binding._state, binding


def _mixed_layer_reference(
    *,
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weights: list[torch.Tensor],
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    request_is_prefill: torch.Tensor,
    num_seqs: int,
    num_tokens: int,
    conv_state: torch.Tensor,
    dilation: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_out = torch.zeros_like(residual)
    expected_state = conv_state.clone()
    starts = [
        int(value) for value in query_start_loc[: num_seqs + 1].detach().cpu().tolist()
    ]
    slots = [int(value) for value in state_slot_ids[:num_seqs].cpu().tolist()]
    fresh = [bool(value) for value in state_is_fresh[:num_seqs].cpu().tolist()]
    accepted = [int(value) for value in num_accepted_tokens[:num_seqs].cpu().tolist()]
    prefill = [bool(value) for value in request_is_prefill[:num_seqs].cpu().tolist()]
    state_length = dilation * (int(conv_weight.shape[-1]) - 1)
    max_speculative = int(conv_state.shape[-1]) - state_length

    for request in range(num_seqs):
        start, end = starts[request : request + 2]
        if start == end or slots[request] < 0:
            continue
        if fresh[request]:
            prior = torch.zeros_like(conv_state[slots[request], :, :state_length])
        else:
            rollback = 0 if prefill[request] else accepted[request] - 1
            prior = conv_state[slots[request], :, rollback : rollback + state_length]
        contribution, newest = ple_projected_sequence_reference(
            residual[start:end],
            key[start:end],
            value[start:end],
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            conv_weight=conv_weight,
            eps=eps,
            dilation=dilation,
            prior_state=prior,
        )
        expected_out[start:end].copy_(contribution)
        if prefill[request]:
            committed = newest
        else:
            _, committed = ple_projected_sequence_reference(
                residual[start : start + 1],
                key[start : start + 1],
                value[start : start + 1],
                k_norm_weight=weights[0],
                q_norm_weight=weights[1],
                u_norm_weight=weights[2],
                conv_weight=conv_weight,
                eps=eps,
                dilation=dilation,
                prior_state=prior,
            )
        state = expected_state[slots[request]]
        state[:, :state_length].copy_(committed)
        state[:, state_length:].zero_()
        if not prefill[request] and end - start > 1:
            _, normalized_u = ple_projected_u_reference(
                residual[start:end],
                key[start:end],
                value[start:end],
                k_norm_weight=weights[0],
                q_norm_weight=weights[1],
                u_norm_weight=weights[2],
                eps=eps,
            )
            candidates = min(end - start - 1, max_speculative)
            state[:, state_length : state_length + candidates].copy_(
                normalized_u[1 : candidates + 1].transpose(0, 1)
            )

    expected_out[num_tokens:].zero_()
    return expected_out, expected_state


def test_ple_stateful_decode_chunks_match_full_prefill_oracle() -> None:
    tokens, streams, hidden = 7, 2, 4
    kernel_size, dilation = 3, 2
    channels = streams * hidden
    state_length = dilation * (kernel_size - 1)
    residual, key, value, weights = _projected_inputs(tokens, streams, hidden)
    generator = torch.Generator().manual_seed(919)
    conv_weight = (torch.randn(channels, kernel_size, generator=generator) / 8).to(
        torch.bfloat16
    )
    zeros = torch.zeros(channels, state_length, dtype=torch.bfloat16)

    expected, expected_state = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=zeros,
    )
    first, state = ple_projected_sequence_reference(
        residual[:2],
        key[:2],
        value[:2],
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=zeros,
    )
    second, state = ple_projected_sequence_reference(
        residual[2:],
        key[2:],
        value[2:],
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=state,
    )
    torch.testing.assert_close(torch.cat((first, second)), expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_ple_packed_oracle_is_request_local() -> None:
    streams, hidden = 2, 4
    residual, key, value, weights = _projected_inputs(5, streams, hidden)
    generator = torch.Generator().manual_seed(721)
    conv_weight = (torch.randn(8, 3, generator=generator) / 8).to(torch.bfloat16)
    starts = torch.tensor([0, 3, 5], dtype=torch.int32)

    packed, states = ple_projected_packed_reference(
        residual,
        key,
        value,
        starts,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=2,
    )
    for request, (start, end) in enumerate(((0, 3), (3, 5))):
        expected, expected_state = ple_projected_sequence_reference(
            residual[start:end],
            key[start:end],
            value[start:end],
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            conv_weight=conv_weight,
            eps=1e-6,
            dilation=2,
        )
        torch.testing.assert_close(packed[start:end], expected, rtol=0, atol=0)
        torch.testing.assert_close(states[request], expected_state, rtol=0, atol=0)




@torch.inference_mode()
def test_ple_padded_state_stride_reaches_past_int32_offset_boundary() -> None:
    device = require_b12x()
    tokens, streams, hidden = 1, 1, 16
    dilation, max_speculative = 1, 0
    residual, key, value, weights, _ = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1401
    )
    conv_weight = torch.tensor(
        [[0.25, 0.5]] * hidden,
        dtype=torch.bfloat16,
        device=device,
    )

    # A representative 818176-byte aligned hybrid-cache page has this BF16
    # stride. The live tail slot starts beyond the signed-32-bit element range.
    state_slot_stride = 818176 // torch.tensor([], dtype=torch.bfloat16).element_size()
    tail_slot = (2**31) // state_slot_stride + 1
    state_elements = tail_slot * state_slot_stride + hidden
    state_storage = torch.empty(state_elements, dtype=torch.bfloat16, device=device)
    conv_state = torch.as_strided(
        state_storage,
        (tail_slot + 1, hidden, 1),
        (state_slot_stride, 1, 1),
    )
    assert tail_slot * conv_state.stride(0) > 2**31

    generator = torch.Generator(device=device).manual_seed(1402)
    prior = torch.randn(
        (hidden, 1), generator=generator, dtype=torch.bfloat16, device=device
    )
    conv_state[tail_slot].copy_(prior)
    conv_state[0].fill_(7)
    low_slot_before = conv_state[0].clone()
    expected, expected_state = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=prior,
    )
    _, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([tail_slot], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(binding.out, expected, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(
        conv_state[tail_slot], expected_state, rtol=0.02, atol=0.0078125
    )
    torch.testing.assert_close(conv_state[0], low_slot_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_hash_target_shape_matches_oracle_and_pads_output() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=7,
        max_seqs=3,
        vocab_size=248320,
        eos_token_id=248044,
        max_order=3,
        heads_per_order=8,
        dense_layer_ordinal=0,
        base_table_size=20000000,
    )
    token_ids = torch.tensor(
        [7, caps.eos_token_id, 8, 9, 10, 0, 0],
        dtype=torch.int64,
        device=device,
    )
    query_start_loc = torch.tensor([0, 3, 5, 5], dtype=torch.int32, device=device)
    committed_history = torch.tensor(
        [
            [3, 4],
            [5, caps.eos_token_id],
            [caps.eos_token_id, caps.eos_token_id],
        ],
        dtype=torch.int64,
        device=device,
    )
    out = torch.empty(
        (caps.max_tokens, caps.head_count), dtype=torch.int64, device=device
    )
    geometry, binding = _bind_cuda_hash(
        caps,
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=torch.tensor([2], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([5], dtype=torch.int32, device=device),
        out=out,
    )

    ple_hash.run(binding)
    torch.cuda.synchronize()
    expected = ple_hash_packed_reference(
        token_ids[:5],
        query_start_loc[:3],
        committed_history[:2],
        eos_token_id=caps.eos_token_id,
        multipliers=geometry.multipliers,
        prime_sizes=geometry.prime_sizes,
        table_offsets=geometry.table_offsets,
        heads_per_order=caps.heads_per_order,
    )

    torch.testing.assert_close(out[:5], expected, rtol=0, atol=0)
    assert bool((out[5:] == -1).all().item())


@torch.inference_mode()
def test_ple_hash_cuda_graph_replay_is_allocation_free() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=2,
        max_seqs=1,
        vocab_size=100,
        eos_token_id=99,
        max_order=3,
        heads_per_order=2,
        dense_layer_ordinal=0,
        base_table_size=101,
    )
    token_ids = torch.tensor([1, 2], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=device)
    committed_history = torch.tensor([[99, 99]], dtype=torch.int64, device=device)
    geometry, binding = _bind_cuda_hash(
        caps,
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([2], dtype=torch.int32, device=device),
        out=torch.empty(
            (caps.max_tokens, caps.head_count), dtype=torch.int64, device=device
        ),
    )
    ple_hash.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple_hash.run(binding)
    output_address = captured_out.data_ptr()

    token_ids.copy_(torch.tensor([3, 4], dtype=torch.int64, device=device))
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)
    expected = ple_hash_packed_reference(
        token_ids,
        query_start_loc,
        committed_history,
        eos_token_id=caps.eos_token_id,
        multipliers=geometry.multipliers,
        prime_sizes=geometry.prime_sizes,
        table_offsets=geometry.table_offsets,
        heads_per_order=caps.heads_per_order,
    )

    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_hash_target_cuda_graph_replays_dynamic_packed_metadata() -> None:
    device = require_b12x()
    caps = ple_hash.Caps(
        device=device,
        max_tokens=8,
        max_seqs=3,
        vocab_size=248320,
        eos_token_id=248044,
        max_order=3,
        heads_per_order=8,
        dense_layer_ordinal=0,
        base_table_size=20000000,
    )
    token_ids = torch.tensor(
        [1, 2, 0, 0, 0, 0, 0, 0], dtype=torch.int64, device=device
    )
    query_start_loc = torch.tensor([0, 2, 2, 2], dtype=torch.int32, device=device)
    committed_history = torch.full(
        (caps.max_seqs, caps.max_order - 1),
        caps.eos_token_id,
        dtype=torch.int64,
        device=device,
    )
    num_seqs = torch.tensor([1], dtype=torch.int32, device=device)
    num_tokens = torch.tensor([2], dtype=torch.int32, device=device)
    geometry, binding = _bind_cuda_hash(
        caps,
        token_ids=token_ids,
        query_start_loc=query_start_loc,
        committed_history=committed_history,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        out=torch.empty(
            (caps.max_tokens, caps.head_count), dtype=torch.int64, device=device
        ),
    )

    ple_hash.run(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple_hash.run(binding)
    output_address = captured_out.data_ptr()

    token_ids.copy_(
        torch.tensor(
            [7, caps.eos_token_id, 8, 9, 10, 11, 0, 0],
            dtype=torch.int64,
            device=device,
        )
    )
    query_start_loc.copy_(torch.tensor([0, 3, 4, 6], dtype=torch.int32, device=device))
    committed_history.copy_(
        torch.tensor(
            [
                [3, 4],
                [5, caps.eos_token_id],
                [6, 7],
            ],
            dtype=torch.int64,
            device=device,
        )
    )
    num_seqs.fill_(3)
    num_tokens.fill_(6)
    expected = ple_hash_packed_reference(
        token_ids[:6],
        query_start_loc[:4],
        committed_history[:3],
        eos_token_id=caps.eos_token_id,
        multipliers=geometry.multipliers,
        prime_sizes=geometry.prime_sizes,
        table_offsets=geometry.table_offsets,
        heads_per_order=caps.heads_per_order,
    )
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out[:6], expected, rtol=0, atol=0)
    assert bool((captured_out[6:] == -1).all().item())


@torch.inference_mode()
def test_ple_target_shape_prefill_matches_oracle_and_recycles_dirty_slots() -> None:
    device = require_b12x()
    tokens, streams, hidden = 3, 4, 2560
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=991
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.full(
        (3, streams * hidden, state_length + max_speculative),
        7,
        dtype=torch.bfloat16,
        device=device,
    )
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    plan, binding = _bind_cuda_layer(
        mode="prefill",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=torch.tensor([0, 1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True, True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.zeros(2, dtype=torch.int32, device=device),
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_prefill(binding, eps=1e-6)
    torch.cuda.synchronize()
    expected, expected_state = ple_projected_packed_reference(
        residual,
        key,
        value,
        query_start_loc,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
    )

    torch.testing.assert_close(binding.out, expected, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(
        conv_state[:2, :, : plan.state_length],
        expected_state,
        rtol=0.02,
        atol=0.0078125,
    )
    assert bool((conv_state[:2, :, plan.state_length :] == 0).all().item())
    assert bool((conv_state[2] == 7).all().item())


@torch.inference_mode()
def test_ple_mixed_packed_order_matches_request_local_oracles() -> None:
    device = require_b12x()
    max_tokens, num_tokens, streams, hidden = 16, 12, 2, 64
    num_seqs = 4
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        max_tokens, streams, hidden, device=device, seed=1210
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (4, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_start_loc = torch.tensor(
        [0, 6, 9, 10, 12, 12], dtype=torch.int32, device=device
    )
    state_slot_ids = torch.tensor([2, 0, -1, 1, -1], device=device)
    state_is_fresh = torch.tensor(
        [True, False, False, False, False], dtype=torch.bool, device=device
    )
    num_accepted_tokens = torch.tensor(
        [0, 3, 1, 99, -7], dtype=torch.int32, device=device
    )
    request_is_prefill = torch.tensor(
        [True, False, False, True, False], dtype=torch.bool, device=device
    )
    expected_out, expected_state = _mixed_layer_reference(
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        request_is_prefill=request_is_prefill,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        conv_state=conv_state,
        dilation=dilation,
        eps=1e-6,
    )
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=num_seqs,
        num_tokens=num_tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=request_is_prefill,
    )

    ple.run_mixed(binding, eps=1e-6)
    torch.cuda.synchronize()

    torch.testing.assert_close(binding.out, expected_out, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(conv_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("accepted", [1, 3, 5])
@torch.inference_mode()
def test_ple_decode_rolls_back_and_retains_candidates(accepted: int) -> None:
    device = require_b12x()
    tokens, streams, hidden = 5, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1200
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    state_capacity = state_length + max_speculative
    prior = torch.randn(
        (streams * hidden, state_capacity),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state = torch.full(
        (2, streams * hidden, state_capacity),
        11,
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state[0].copy_(prior)
    untouched = conv_state[1].clone()
    plan, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor(
            [0, tokens, tokens], dtype=torch.int32, device=device
        ),
        state_slot_ids=torch.tensor([0, -1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False, False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor(
            [accepted, 0], dtype=torch.int32, device=device
        ),
        num_seqs=1,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    rollback = accepted - 1
    effective_history = prior[:, rollback : rollback + state_length].contiguous()
    expected, _ = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=effective_history,
    )
    _, normalized_u = ple_projected_u_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        eps=1e-6,
    )
    expected_base = torch.cat(
        (effective_history[:, 1:], normalized_u[0].unsqueeze(1)), dim=1
    )
    expected_tail = torch.zeros(
        (streams * hidden, max_speculative),
        dtype=torch.bfloat16,
        device=device,
    )
    expected_tail[:, : tokens - 1].copy_(normalized_u[1:].transpose(0, 1))

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()

    torch.testing.assert_close(binding.out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        conv_state[0, :, : plan.state_length], expected_base, rtol=0, atol=0
    )
    torch.testing.assert_close(
        conv_state[0, :, plan.state_length :], expected_tail, rtol=0, atol=0
    )
    torch.testing.assert_close(conv_state[1], untouched, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_decode_fresh_flag_ignores_dirty_recycled_state() -> None:
    device = require_b12x()
    tokens, streams, hidden = 2, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1201
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.full(
        (1, streams * hidden, state_length + max_speculative),
        7,
        dtype=torch.bfloat16,
        device=device,
    )
    plan, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([0], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )
    zero_history = torch.zeros(
        (streams * hidden, state_length), dtype=torch.bfloat16, device=device
    )
    expected, _ = ple_projected_sequence_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
        prior_state=zero_history,
    )
    _, normalized_u = ple_projected_u_reference(
        residual,
        key,
        value,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        eps=1e-6,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()

    expected_base = torch.cat(
        (zero_history[:, 1:], normalized_u[0].unsqueeze(1)), dim=1
    )
    torch.testing.assert_close(binding.out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        conv_state[0, :, : plan.state_length], expected_base, rtol=0, atol=0
    )
    torch.testing.assert_close(
        conv_state[0, :, plan.state_length], normalized_u[1], rtol=0, atol=0
    )
    assert bool((conv_state[0, :, plan.state_length + 1 :] == 0).all().item())


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@torch.inference_mode()
def test_ple_zero_token_live_request_preserves_entire_state(mode: str) -> None:
    device = require_b12x()
    tokens, streams, hidden = 1, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1207
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (2, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    empty_state_before = conv_state[0].clone()
    _, binding = _bind_cuda_layer(
        mode=mode,
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([0, 1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True, True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32, device=device),
        num_seqs=2,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    if mode == "decode":
        ple.run_decode(binding, eps=1e-6)
    else:
        ple.run_prefill(binding, eps=1e-6)
    torch.cuda.synchronize()

    torch.testing.assert_close(conv_state[0], empty_state_before, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_dummy_slots_replay_under_cuda_graph_without_state_mutation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 3, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1203
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (2, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    _, binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([-1, -1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False, False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32, device=device),
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_decode(binding, eps=1e-6)
    torch.cuda.synchronize()
    state_before = conv_state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ple.run_decode(binding, eps=1e-6)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    for _ in range(3):
        binding.out.fill_(91)
        graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert bool((binding.out == 0).all().item())
    torch.testing.assert_close(conv_state, state_before, rtol=0, atol=0)
    assert allocated_after_replay == allocated_before_replay


@torch.inference_mode()
def test_ple_prefill_replays_under_cuda_graph_without_allocation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 3, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1209
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    channels = streams * hidden
    state_capacity = state_length + max_speculative
    state_payload = channels * state_capacity
    state_slot_stride = state_payload + 19
    state_storage = torch.randn(
        state_slot_stride + state_payload,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state = torch.as_strided(
        state_storage,
        (2, channels, state_capacity),
        (state_slot_stride, state_capacity, 1),
    )
    untouched_state = conv_state[0].clone()
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    plan, binding = _bind_cuda_layer(
        mode="prefill",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=torch.tensor([1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([True], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([0], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )

    ple.run_prefill(binding, eps=1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple.run_prefill(binding, eps=1e-6)
    output_address = captured_out.data_ptr()

    residual.copy_(torch.randn_like(residual))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)
    expected, expected_state = ple_projected_packed_reference(
        residual,
        key,
        value,
        query_start_loc,
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=conv_weight,
        eps=1e-6,
        dilation=dilation,
    )

    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out, expected, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(
        conv_state[1, :, : plan.state_length],
        expected_state[0],
        rtol=0.02,
        atol=0.0078125,
    )
    assert bool((conv_state[1, :, plan.state_length :] == 0).all().item())
    torch.testing.assert_close(conv_state[0], untouched_state, rtol=0, atol=0)


@torch.inference_mode()
def test_ple_mixed_replays_runtime_request_modes_without_allocation() -> None:
    device = require_b12x()
    tokens, streams, hidden = 4, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1213
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (2, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    state_slot_ids = torch.tensor([0, 1], dtype=torch.int64, device=device)
    state_is_fresh = torch.tensor([True, False], dtype=torch.bool, device=device)
    num_accepted_tokens = torch.tensor([99, 1], dtype=torch.int32, device=device)
    request_is_prefill = torch.tensor([True, False], dtype=torch.bool, device=device)
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=request_is_prefill,
    )

    ple.run_mixed(binding, eps=1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple.run_mixed(binding, eps=1e-6)
    torch.cuda.synchronize()
    output_address = captured_out.data_ptr()

    residual.copy_(torch.randn_like(residual))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    state_is_fresh.zero_()
    num_accepted_tokens.copy_(torch.tensor([2, 99], dtype=torch.int32, device=device))
    request_is_prefill.logical_not_()
    expected_out, expected_state = _mixed_layer_reference(
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        request_is_prefill=request_is_prefill,
        num_seqs=2,
        num_tokens=tokens,
        conv_state=conv_state,
        dilation=dilation,
        eps=1e-6,
    )
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize()
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured_out, expected_out, rtol=0.02, atol=0.0078125)
    torch.testing.assert_close(conv_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("token_count", "high_state_slots"), [(None, False), (8, False), (8, True)]
)
@torch.inference_mode()
def test_ple_target_mixed_graph_replays_dynamic_packed_metadata(
    token_count: int | None, high_state_slots: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from triton.runtime import JITFunction

    from b12x.sequence.ple import _kernels

    device = require_b12x()
    max_tokens, streams, hidden = 16, 4, 2560
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        max_tokens, streams, hidden, device=device, seed=1214
    )
    conv_weight = (
        torch.randn(
            (streams * hidden, kernel_size),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 32
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    slot_stride = streams * hidden * (state_length + max_speculative)
    slot_offset = (1 << 31) // slot_stride + 1 if high_state_slots else 0
    conv_state = torch.empty(
        (slot_offset + 3, streams * hidden, state_length + max_speculative),
        dtype=torch.bfloat16,
        device=device,
    )
    live_state = conv_state[slot_offset:]
    live_state.normal_(generator=generator)
    query_start_loc = torch.tensor([0, 2, 5, 5], dtype=torch.int32, device=device)
    state_slot_ids = torch.arange(3, dtype=torch.int64, device=device) + slot_offset
    state_is_fresh = torch.tensor([True, False, False], dtype=torch.bool, device=device)
    num_accepted_tokens = torch.tensor([99, 2, 1], dtype=torch.int32, device=device)
    request_is_prefill = torch.tensor(
        [True, False, False], dtype=torch.bool, device=device
    )
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        num_seqs=2,
        num_tokens=5,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=request_is_prefill,
    )

    ple.run_mixed(binding, eps=1e-6)

    def reject_resolution(*args, **kwargs):
        pytest.fail("Live launch bounds must reuse the warmed PLE kernels")

    for kernel in vars(_kernels).values():
        if isinstance(kernel, JITFunction):
            monkeypatch.setattr(kernel, "_do_compile", reject_resolution)

    state_before = live_state.clone()
    binding.num_seqs.zero_()
    binding.num_tokens.zero_()
    for launch_bound in (0, 1, 3, 8):
        binding.out.fill_(91)
        out = ple.run_mixed(binding, eps=1e-6, token_count=launch_bound)
        assert bool((out == 0).all().item())
        assert bool((binding.out[launch_bound:] == 91).all().item())
    torch.testing.assert_close(live_state, state_before, rtol=0, atol=0)
    binding.num_seqs.fill_(2)
    binding.num_tokens.fill_(5)
    binding.out.fill_(91)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = ple.run_mixed(binding, eps=1e-6, token_count=token_count)
    output_address = captured_out.data_ptr()

    query_start_loc.copy_(torch.tensor([0, 1, 5, 8], dtype=torch.int32, device=device))
    binding.num_seqs.fill_(3)
    binding.num_tokens.fill_(8)
    request_is_prefill.copy_(
        torch.tensor([False, True, False], dtype=torch.bool, device=device)
    )
    num_accepted_tokens.copy_(
        torch.tensor([2, 99, 5], dtype=torch.int32, device=device)
    )
    state_is_fresh.copy_(
        torch.tensor([False, True, False], dtype=torch.bool, device=device)
    )
    residual.copy_(torch.randn_like(residual).mul_(0.2))
    key.copy_(torch.randn_like(key).mul_(0.2))
    value.copy_(torch.randn_like(value).mul_(0.2))
    live_state.copy_(torch.randn_like(live_state).mul_(0.2))
    expected_out, expected_state = _mixed_layer_reference(
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=query_start_loc,
        state_slot_ids=state_slot_ids - slot_offset,
        state_is_fresh=state_is_fresh,
        num_accepted_tokens=num_accepted_tokens,
        request_is_prefill=request_is_prefill,
        num_seqs=3,
        num_tokens=8,
        conv_state=live_state,
        dilation=dilation,
        eps=1e-6,
    )
    initial_state = live_state.clone()
    full_out = ple.run_mixed(binding, eps=1e-6).clone()
    full_state = live_state.clone()
    live_state.copy_(initial_state)
    binding.out.fill_(91)
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert captured_out.data_ptr() == output_address == binding.out.data_ptr()
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(
        captured_out, expected_out[: captured_out.shape[0]], rtol=0.02, atol=0.0078125
    )
    torch.testing.assert_close(
        captured_out, full_out[: captured_out.shape[0]], rtol=0, atol=0
    )
    assert bool((binding.out[captured_out.shape[0] :] == 91).all().item())
    torch.testing.assert_close(live_state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(live_state, full_state, rtol=0, atol=0)
@torch.inference_mode()
def test_ple_prepared_runs_compile_and_capture_mutating_outputs() -> None:
    device = require_b12x()
    hash_caps = _hash_caps(device=device)
    hash_tokens = torch.tensor([1, 2], dtype=torch.int64, device=device)
    hash_starts = torch.tensor([0, 2], dtype=torch.int32, device=device)
    hash_history = torch.tensor([[99, 99]], dtype=torch.int64, device=device)
    hash_geometry, hash_binding = _bind_cuda_hash(
        hash_caps,
        token_ids=hash_tokens,
        query_start_loc=hash_starts,
        committed_history=hash_history,
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([2], dtype=torch.int32, device=device),
        out=torch.empty((2, 4), dtype=torch.int64, device=device),
    )

    tokens, streams, hidden = 1, 2, 32
    kernel_size, dilation, max_speculative = 4, 3, 4
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=1205
    )
    conv_weight = torch.randn(
        (streams * hidden, kernel_size),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_length = dilation * (kernel_size - 1)
    conv_state = torch.randn(
        (1, streams * hidden, state_length + max_speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    _, layer_binding = _bind_cuda_layer(
        mode="decode",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([-1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
    )
    _, mixed_binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor([-1], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([99], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=1,
        conv_state=conv_state,
        max_speculative_tokens=max_speculative,
        dilation=dilation,
        request_is_prefill=torch.tensor([True], dtype=torch.bool, device=device),
    )

    expected_hash = ple_hash_packed_reference(
        hash_tokens,
        hash_starts,
        hash_history,
        eos_token_id=hash_caps.eos_token_id,
        multipliers=hash_geometry.multipliers,
        prime_sizes=hash_geometry.prime_sizes,
        table_offsets=hash_geometry.table_offsets,
        heads_per_order=hash_caps.heads_per_order,
    )

    compiled_hash = torch.compile(
        lambda: ple_hash.run(hash_binding), backend="eager", fullgraph=True
    )
    compiled_layer = torch.compile(
        lambda: ple.run_decode(layer_binding, eps=1e-6),
        backend="eager",
        fullgraph=True,
    )
    compiled_mixed = torch.compile(
        lambda: ple.run_mixed(mixed_binding, eps=1e-6),
        backend="eager",
        fullgraph=True,
    )
    compiled_hash()
    compiled_layer()
    compiled_mixed()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_hash = compiled_hash()
        captured_layer = compiled_layer()
        captured_mixed = compiled_mixed()
    hash_address = captured_hash.data_ptr()
    layer_address = captured_layer.data_ptr()
    mixed_address = captured_mixed.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    assert captured_hash.data_ptr() == hash_address == hash_binding.out.data_ptr()
    assert captured_layer.data_ptr() == layer_address == layer_binding.out.data_ptr()
    assert captured_mixed.data_ptr() == mixed_address == mixed_binding.out.data_ptr()
    torch.testing.assert_close(captured_hash, expected_hash, rtol=0, atol=0)
    assert bool((captured_layer == 0).all().item())
    assert bool((captured_mixed == 0).all().item())


@torch.inference_mode()
def test_ple_state_slot_past_int32_element_offset_matches_oracle() -> None:
    device = require_b12x()
    tokens, streams, hidden = 1, 4, 2560
    kernel_size, dilation, max_speculative = 4, 3, 4
    state_length = dilation * (kernel_size - 1)
    state_capacity = state_length + max_speculative
    channels = streams * hidden
    slot_stride_elements = channels * state_capacity
    int32_max = torch.iinfo(torch.int32).max
    high_slot = int32_max // slot_stride_elements + 2
    assert high_slot * slot_stride_elements > int32_max

    binding = None
    conv_state = None
    try:
        # The pool is intentionally mostly uninitialized. Only slot zero and
        # the live tail slot are touched, reproducing a high recycled slot ID.
        conv_state = torch.empty(
            (high_slot + 1, channels, state_capacity),
            dtype=torch.bfloat16,
            device=device,
        )
        assert conv_state.stride(0) == slot_stride_elements
        assert high_slot * conv_state.stride(0) > int32_max
        conv_state[0].fill_(7)
        low_slot_before = conv_state[0].clone()

        residual, key, value, weights, generator = _cuda_projected_inputs(
            tokens, streams, hidden, device=device, seed=1204
        )
        conv_weight = (
            torch.randn(
                (channels, kernel_size),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            / 32
        ).contiguous()
        prior = torch.randn(
            (channels, state_capacity),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        conv_state[high_slot].copy_(prior)
        plan, binding = _bind_cuda_layer(
            mode="decode",
            residual=residual,
            key=key,
            value=value,
            weights=weights,
            conv_weight=conv_weight,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            state_slot_ids=torch.tensor([high_slot], dtype=torch.int64, device=device),
            state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
            num_accepted_tokens=torch.tensor([3], dtype=torch.int32, device=device),
            num_seqs=1,
            num_tokens=1,
            conv_state=conv_state,
            max_speculative_tokens=max_speculative,
            dilation=dilation,
        )
        effective_history = prior[:, 2 : 2 + state_length].contiguous()
        expected, _ = ple_projected_sequence_reference(
            residual,
            key,
            value,
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            conv_weight=conv_weight,
            eps=1e-6,
            dilation=dilation,
            prior_state=effective_history,
        )
        _, normalized_u = ple_projected_u_reference(
            residual,
            key,
            value,
            k_norm_weight=weights[0],
            q_norm_weight=weights[1],
            u_norm_weight=weights[2],
            eps=1e-6,
        )
        expected_base = torch.cat(
            (effective_history[:, 1:], normalized_u[0].unsqueeze(1)), dim=1
        )

        ple.run_decode(binding, eps=1e-6)
        torch.cuda.synchronize()

        torch.testing.assert_close(binding.out, expected, rtol=0.02, atol=0.0078125)
        torch.testing.assert_close(
            conv_state[high_slot, :, : plan.state_length],
            expected_base,
            rtol=0.02,
            atol=0.0078125,
        )
        assert bool((conv_state[high_slot, :, plan.state_length :] == 0).all().item())
        torch.testing.assert_close(conv_state[0], low_slot_before, rtol=0, atol=0)
    finally:
        del binding
        del conv_state
        torch.cuda.empty_cache()
