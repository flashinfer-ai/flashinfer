"""Correctness coverage for ReplaySSM prefix-state materialization."""

import pytest
import torch

from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
from flashinfer.mamba.replayssm_materialize import replayssm_materialize
from flashinfer.jit.mamba.replayssm_materialize import (
    gen_replayssm_materialize_module,
)
from flashinfer.utils import is_cvt_rs_supported


def _ptr_table(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors], dtype=torch.int64, device="cuda"
    )


def _stride_table(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [tensor.stride(0) for tensor in tensors], dtype=torch.int64, device="cuda"
    )


def _active_request_indices(replay_prefix_len: torch.Tensor) -> torch.Tensor:
    indices = torch.full_like(replay_prefix_len, -1)
    active = torch.nonzero(replay_prefix_len >= 0, as_tuple=False).flatten()
    indices[: active.numel()] = active
    return indices


def _materialize(
    state: list[torch.Tensor],
    x_cache: list[torch.Tensor],
    b_cache: list[torch.Tensor],
    dt_cache: list[torch.Tensor],
    a: list[torch.Tensor],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    replay_prefix_len: torch.Tensor,
    ring_buffer_len: int,
    heads_per_group: int = 1,
    active_request_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
    rand_seed: torch.Tensor | None = None,
    philox_rounds: int = 0,
    dependency_inputs: list[torch.Tensor] | None = None,
    dependency_outputs: list[torch.Tensor] | None = None,
    state_scales: list[torch.Tensor] | None = None,
) -> None:
    layers = len(state)
    zero_table = torch.zeros(layers, dtype=torch.int64, device="cuda")
    if (state[0].element_size() == 1) != (state_scales is not None):
        raise ValueError("state_scales are required exactly for one-byte state storage")
    if state_scales is not None and len(state_scales) != layers:
        raise ValueError("state_scales must have one tensor per layer")
    replayssm_materialize(
        _ptr_table(state),
        _stride_table(state),
        _ptr_table(x_cache),
        _stride_table(x_cache),
        _ptr_table(b_cache),
        _stride_table(b_cache),
        _ptr_table(dt_cache),
        _stride_table(dt_cache),
        _ptr_table(a),
        _ptr_table(state_scales) if state_scales is not None else zero_table,
        _stride_table(state_scales) if state_scales is not None else zero_table,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        active_request_indices
        if active_request_indices is not None
        else _active_request_indices(replay_prefix_len),
        state_dtype=state[0].dtype,
        input_dtype=x_cache[0].dtype,
        matrixA_dtype=a[0].dtype,
        dim=state[0].size(-2),
        dstate=state[0].size(-1),
        num_heads=state[0].size(1),
        heads_per_group=heads_per_group,
        max_window=8,
        ring_buffer_len=ring_buffer_len,
        pad_slot_id=pad_slot_id,
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
        dependency_inputs=dependency_inputs,
        dependency_outputs=dependency_outputs,
    )


@pytest.mark.parametrize(
    "input_dtype",
    [torch.float16, torch.float32, torch.int8, torch.float8_e4m3fn],
    ids=["fp16", "fp32", "int8", "fp8"],
)
def test_replayssm_materialize_rejects_non_bf16_input_dtype(
    input_dtype: torch.dtype,
) -> None:
    """Replay operands are BF16, not merely arbitrary two-byte storage."""
    with pytest.raises(ValueError, match="input_dtype=torch.bfloat16"):
        gen_replayssm_materialize_module(
            torch.bfloat16,
            input_dtype,
            torch.float32,
            64,
            64,
            1,
            8,
        )


def test_replayssm_materialize_bf16_replay_and_copy() -> None:
    """Positive replay, zero exact-copy, and source immutability in one launch."""
    torch.manual_seed(0)
    layers, slots, ring_buffer_len = 2, 8, 12
    state = [
        torch.randn(slots, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    x_cache = [
        torch.randn(slots, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    b_cache = [
        torch.randn(slots, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    dt_cache = [
        torch.rand(slots, 1, ring_buffer_len, device="cuda") for _ in range(layers)
    ]
    a = [-torch.rand(1, device="cuda") for _ in range(layers)]
    src_slots = torch.tensor(
        [[0, 1, 2, 3], [2, 0, 4, 5]], dtype=torch.int32, device="cuda"
    )
    dst_slots = torch.tensor(
        [[4, 5, 6, 7], [1, 3, 6, 7]], dtype=torch.int32, device="cuda"
    )
    ring_start = torch.tensor([10, 3, 0, 0], dtype=torch.int32, device="cuda")
    replay_prefix_len = torch.tensor([3, 0, -1, -1], dtype=torch.int32, device="cuda")

    before = [tensor.clone() for tensor in state]
    # Deliberately reverse the active prefix and follow it with sentinels.
    _materialize(
        state,
        x_cache,
        b_cache,
        dt_cache,
        a,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        ring_buffer_len,
        active_request_indices=torch.tensor(
            [1, 0, -1, -1], dtype=torch.int32, device="cuda"
        ),
        dependency_inputs=[*x_cache, *b_cache, *dt_cache, *a],
        dependency_outputs=state,
    )
    torch.cuda.synchronize()

    # The zero-count request is a raw byte-for-byte state copy.
    for layer in range(layers):
        assert torch.equal(
            state[layer][dst_slots[layer, 1]], before[layer][src_slots[layer, 1]]
        )

    # Positive replay follows the selective-state recurrence.  The production
    # path uses bf16 MMA and bf16 state storage, so compare after each stored
    # bf16 step rather than against an fp32-only recurrence.
    for layer in range(layers):
        expected = before[layer][src_slots[layer, 0]].float()
        source_slot = int(src_slots[layer, 0].cpu())
        start = int(ring_start[0].cpu())
        for token in range(int(replay_prefix_len[0].cpu())):
            row = (start + token) % ring_buffer_len
            expected = expected * torch.exp(
                a[layer][0] * dt_cache[layer][source_slot, 0, row]
            )
            expected = (
                expected
                + torch.outer(
                    x_cache[layer][source_slot, 0, row].float(),
                    b_cache[layer][source_slot, 0, row].float(),
                )
                * dt_cache[layer][source_slot, 0, row]
            )
            expected = expected.to(torch.bfloat16).float()
        actual = state[layer][dst_slots[layer, 0]].float()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)
        assert torch.equal(
            state[layer][src_slots[layer, 0]], before[layer][src_slots[layer, 0]]
        )
        for request in (2, 3):
            assert torch.equal(
                state[layer][dst_slots[layer, request]],
                before[layer][dst_slots[layer, request]],
            )


def test_replayssm_materialize_persistent_grid_stride_zero_copy() -> None:
    """A large active batch exercises a second persistent grid-stride item."""
    batch = 8192
    state = torch.zeros(batch + 1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    state[0].normal_()
    source = state[0].clone()
    x_cache = [torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="cuda")]
    b_cache = [torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="cuda")]
    dt_cache = [torch.empty(1, 1, 1, device="cuda")]
    a = [torch.empty(1, device="cuda")]
    src_slots = torch.zeros((1, batch), dtype=torch.int32, device="cuda")
    dst_slots = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda").unsqueeze(
        0
    )
    ring_start = torch.zeros(batch, dtype=torch.int32, device="cuda")
    replay_prefix_len = torch.zeros(batch, dtype=torch.int32, device="cuda")

    _materialize(
        [state],
        x_cache,
        b_cache,
        dt_cache,
        a,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        ring_buffer_len=1,
        active_request_indices=torch.arange(batch, dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()
    assert torch.equal(state[1:], source.expand_as(state[1:]))


def test_replayssm_materialize_no_active_requests_is_noop() -> None:
    """An empty active-request prefix does not touch source or destination state."""
    torch.manual_seed(1)
    ring_buffer_len = 12
    state = [torch.randn(2, 1, 64, 64, dtype=torch.bfloat16, device="cuda")]
    x_cache = [
        torch.randn(2, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda")
    ]
    b_cache = [
        torch.randn(2, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda")
    ]
    dt_cache = [torch.rand(2, 1, ring_buffer_len, device="cuda")]
    a = [-torch.rand(1, device="cuda")]
    src_slots = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    dst_slots = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    ring_start = torch.tensor([0], dtype=torch.int32, device="cuda")
    replay_prefix_len = torch.zeros(1, dtype=torch.int32, device="cuda")
    before = state[0].clone()

    _materialize(
        state,
        x_cache,
        b_cache,
        dt_cache,
        a,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        ring_buffer_len,
        active_request_indices=torch.full((1,), -1, dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()
    assert torch.equal(state[0], before)


def test_replayssm_materialize_multilayer_multhead_grouped_b() -> None:
    """Layer tables and per-group B addressing work with multiple heads."""
    torch.manual_seed(11)
    layers, slots, heads, ring_buffer_len = 2, 2, 2, 12
    state = [
        torch.randn(slots, heads, 64, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    x_cache = [
        torch.randn(
            slots, heads, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda"
        )
        for _ in range(layers)
    ]
    b_cache = [
        torch.randn(slots, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(layers)
    ]
    dt_cache = [
        torch.rand(slots, heads, ring_buffer_len, device="cuda") for _ in range(layers)
    ]
    a = [-torch.rand(heads, device="cuda") for _ in range(layers)]
    src_slots = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    dst_slots = torch.tensor([[1], [0]], dtype=torch.int32, device="cuda")
    ring_start = torch.tensor([10], dtype=torch.int32, device="cuda")
    replay_prefix_len = torch.tensor([3], dtype=torch.int32, device="cuda")
    before = [tensor.clone() for tensor in state]

    _materialize(
        state,
        x_cache,
        b_cache,
        dt_cache,
        a,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        ring_buffer_len,
        heads_per_group=2,
    )
    torch.cuda.synchronize()

    for layer in range(layers):
        for head in range(heads):
            expected = before[layer][src_slots[layer, 0], head].float()
            for token in range(int(replay_prefix_len[0].cpu())):
                row = (int(ring_start[0].cpu()) + token) % ring_buffer_len
                expected = expected * torch.exp(
                    a[layer][head] * dt_cache[layer][src_slots[layer, 0], head, row]
                )
                expected = (
                    expected
                    + torch.outer(
                        x_cache[layer][src_slots[layer, 0], head, row].float(),
                        b_cache[layer][src_slots[layer, 0], 0, row].float(),
                    )
                    * dt_cache[layer][src_slots[layer, 0], head, row]
                )
                expected = expected.to(torch.bfloat16).float()
            torch.testing.assert_close(
                state[layer][dst_slots[layer, 0], head].float(),
                expected,
                rtol=2e-2,
                atol=5e-2,
            )


@pytest.mark.parametrize(
    ("state_dtype", "philox_rounds"),
    [(torch.bfloat16, 0), (torch.float16, 5)],
    ids=["bf16_rn", "fp16_philox5"],
)
def test_replayssm_materialize_matches_checkpointing_ssu_replay(
    state_dtype: torch.dtype, philox_rounds: int
) -> None:
    """Replay matches checkpointing_ssu for deterministic and Philox stores."""
    torch.manual_seed(2)
    cache_size, ring_buffer_len, predicted = 2, 12, 4
    if philox_rounds and not is_cvt_rs_supported():
        pytest.skip("FP16 Philox stochastic rounding requires SM100a/SM103a")
    state = torch.randn(cache_size, 1, 64, 64, dtype=state_dtype, device="cuda")
    x_cache = torch.randn(
        cache_size, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda"
    )
    b_cache = torch.randn(
        cache_size, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda"
    )
    dt_cache = torch.rand(cache_size, 1, ring_buffer_len, device="cuda")
    ring_start = torch.tensor([8, 0], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([5, 0], dtype=torch.int32, device="cuda")

    x = torch.randn(1, predicted, 1, 64, dtype=torch.bfloat16, device="cuda")
    dt = torch.rand(1, predicted, 1, dtype=torch.bfloat16, device="cuda")
    dt = dt.unsqueeze(-1).expand(-1, -1, -1, 64)
    a_values = -torch.rand(1, device="cuda")
    # checkpointing_ssu represents tie_hdim A as an H-long physical tensor.
    a = a_values.as_strided((1, 64, 64), (1, 0, 0))
    b = torch.randn(1, predicted, 1, 64, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros_like(b)
    out = torch.empty_like(x)

    expected = state.clone()
    rand_seed = (
        torch.tensor([12345], device="cuda", dtype=torch.int64)
        if philox_rounds
        else None
    )
    checkpointing_ssu(
        expected,
        x_cache.clone(),
        b_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
        accepted.clone(),
        x,
        dt,
        a,
        b,
        c,
        out,
        state_batch_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        algorithm="monolith",
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
    )

    actual = state.clone()
    _materialize(
        [actual],
        [x_cache],
        [b_cache],
        [dt_cache],
        [a_values],
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        ring_start[:1],
        accepted[:1],
        ring_buffer_len,
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
    )
    torch.cuda.synchronize()
    if philox_rounds:
        expected_value = expected[0]
        actual_value = actual[1]
        assert torch.all(
            (actual_value == expected_value)
            | (
                actual_value
                == torch.nextafter(
                    expected_value, torch.full_like(expected_value, float("inf"))
                )
            )
            | (
                actual_value
                == torch.nextafter(
                    expected_value, torch.full_like(expected_value, float("-inf"))
                )
            )
        )
    else:
        assert torch.equal(actual[1], expected[0])


@pytest.mark.parametrize(
    "state_dtype", [torch.int8, torch.float8_e4m3fn], ids=["int8", "fp8_e4m3fn"]
)
def test_replayssm_materialize_8bit_matches_checkpointing_ssu_replay(
    state_dtype: torch.dtype,
) -> None:
    """One-byte state bytes and block scales follow the existing two-pass path."""
    torch.manual_seed(3)
    cache_size, ring_buffer_len, predicted = 2, 12, 4
    if state_dtype == torch.int8:
        state = torch.randint(
            -50, 50, (cache_size, 1, 64, 128), dtype=state_dtype, device="cuda"
        )
    else:
        state = torch.randn(cache_size, 1, 64, 128, device="cuda").to(state_dtype)
    scales = torch.rand(cache_size, 1, 64, device="cuda") + 0.01
    x_cache = torch.randn(
        cache_size, 1, ring_buffer_len, 64, dtype=torch.bfloat16, device="cuda"
    )
    b_cache = torch.randn(
        cache_size, 1, ring_buffer_len, 128, dtype=torch.bfloat16, device="cuda"
    )
    dt_cache = torch.rand(cache_size, 1, ring_buffer_len, device="cuda")
    ring_start = torch.tensor([8, 0], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([5, 0], dtype=torch.int32, device="cuda")
    x = torch.randn(1, predicted, 1, 64, dtype=torch.bfloat16, device="cuda")
    dt = torch.rand(1, predicted, 1, dtype=torch.bfloat16, device="cuda")
    dt = dt.unsqueeze(-1).expand(-1, -1, -1, 64)
    a_values = -torch.rand(1, device="cuda")
    a = a_values.as_strided((1, 64, 128), (1, 0, 0))
    b = torch.randn(1, predicted, 1, 128, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros_like(b)
    out = torch.empty_like(x)

    expected_state, expected_scales = state.clone(), scales.clone()
    checkpointing_ssu(
        expected_state,
        x_cache.clone(),
        b_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
        accepted.clone(),
        x,
        dt,
        a,
        b,
        c,
        out,
        state_batch_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        state_scale=expected_scales,
        algorithm="monolith",
    )
    actual_state, actual_scales = state.clone(), scales.clone()
    _materialize(
        [actual_state],
        [x_cache],
        [b_cache],
        [dt_cache],
        [a_values],
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        ring_start[:1],
        accepted[:1],
        ring_buffer_len,
        state_scales=[actual_scales],
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_state[1], expected_state[0])
    assert torch.equal(actual_scales[1], expected_scales[0])


@pytest.mark.parametrize(
    "state_dtype", [torch.int8, torch.float8_e4m3fn], ids=["int8", "fp8_e4m3fn"]
)
def test_replayssm_materialize_8bit_zero_count_copies_state_and_scale(
    state_dtype: torch.dtype,
) -> None:
    """Zero count preserves the raw one-byte state and its per-row scales."""
    if state_dtype == torch.int8:
        state = torch.randint(
            -50, 50, (2, 1, 64, 128), dtype=state_dtype, device="cuda"
        )
    else:
        state = torch.randn(2, 1, 64, 128, device="cuda").to(state_dtype)
    scales = torch.rand(2, 1, 64, device="cuda") + 0.01
    state_before, scales_before = state.clone(), scales.clone()
    _materialize(
        [state],
        [torch.empty(1, 1, 1, 64, dtype=torch.bfloat16, device="cuda")],
        [torch.empty(1, 1, 1, 128, dtype=torch.bfloat16, device="cuda")],
        [torch.empty(1, 1, 1, device="cuda")],
        [torch.empty(1, device="cuda")],
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        ring_buffer_len=1,
        state_scales=[scales],
    )
    torch.cuda.synchronize()
    assert torch.equal(state[0], state_before[0])
    assert torch.equal(state[1], state_before[0])
    assert torch.equal(scales[0], scales_before[0])
    assert torch.equal(scales[1], scales_before[0])
