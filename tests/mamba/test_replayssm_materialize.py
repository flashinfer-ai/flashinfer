"""Correctness coverage for ReplaySSM prefix-state materialization."""

import pytest
import torch

from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
from flashinfer.mamba.replayssm_materialize import replayssm_materialize


def _ptr_table(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors], dtype=torch.int64, device="cuda"
    )


def _stride_table(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [tensor.stride(0) for tensor in tensors], dtype=torch.int64, device="cuda"
    )


def _materialize(
    state: list[torch.Tensor],
    x_cache: list[torch.Tensor],
    b_cache: list[torch.Tensor],
    dt_cache: list[torch.Tensor],
    a: list[torch.Tensor],
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    flush_count: torch.Tensor,
    ring_buffer_len: int,
    heads_per_group: int = 1,
) -> None:
    layers = len(state)
    zero_table = torch.zeros(layers, dtype=torch.int64, device="cuda")
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
        zero_table,
        zero_table,
        src_slots,
        dst_slots,
        ring_start,
        flush_count,
        state_dtype=torch.bfloat16,
        input_dtype=torch.bfloat16,
        matrixA_dtype=torch.float32,
        dim=64,
        dstate=64,
        num_heads=state[0].size(1),
        heads_per_group=heads_per_group,
        max_window=8,
        ring_buffer_len=ring_buffer_len,
    )


def test_replayssm_materialize_bf16_replay_and_copy() -> None:
    """Positive replay, zero exact-copy, and source immutability in one launch."""
    torch.manual_seed(0)
    layers, slots, ring_buffer_len = 2, 4, 12
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
    src_slots = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32, device="cuda")
    dst_slots = torch.tensor([[2, 3], [1, 3]], dtype=torch.int32, device="cuda")
    ring_start = torch.tensor([10, 3], dtype=torch.int32, device="cuda")
    flush_count = torch.tensor([3, 0], dtype=torch.int32, device="cuda")

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
        flush_count,
        ring_buffer_len,
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
        for token in range(int(flush_count[0].cpu())):
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


def test_replayssm_materialize_negative_count_is_noop() -> None:
    """A negative count does not touch either source or destination state."""
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
    flush_count = torch.tensor([-1], dtype=torch.int32, device="cuda")
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
        flush_count,
        ring_buffer_len,
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
    flush_count = torch.tensor([3], dtype=torch.int32, device="cuda")
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
        flush_count,
        ring_buffer_len,
        heads_per_group=2,
    )
    torch.cuda.synchronize()

    for layer in range(layers):
        for head in range(heads):
            expected = before[layer][src_slots[layer, 0], head].float()
            for token in range(int(flush_count[0].cpu())):
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


def test_replayssm_materialize_matches_checkpointing_ssu_replay() -> None:
    """The shared replay helper produces bitwise-identical BF16 state."""
    torch.manual_seed(2)
    cache_size, ring_buffer_len, predicted = 2, 12, 4
    state = torch.randn(cache_size, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
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
    )
    torch.cuda.synchronize()
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
    _materialize_int8(
        actual_state,
        actual_scales,
        x_cache,
        b_cache,
        dt_cache,
        a_values,
        ring_start[:1],
        accepted[:1],
        ring_buffer_len,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_state[1], expected_state[0])
    assert torch.equal(actual_scales[1], expected_scales[0])


def _materialize_int8(
    state: torch.Tensor,
    scales: torch.Tensor,
    x_cache: torch.Tensor,
    b_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    a: torch.Tensor,
    ring_start: torch.Tensor,
    flush_count: torch.Tensor,
    ring_buffer_len: int,
) -> None:
    replayssm_materialize(
        _ptr_table([state]),
        _stride_table([state]),
        _ptr_table([x_cache]),
        _stride_table([x_cache]),
        _ptr_table([b_cache]),
        _stride_table([b_cache]),
        _ptr_table([dt_cache]),
        _stride_table([dt_cache]),
        _ptr_table([a]),
        _ptr_table([scales]),
        _stride_table([scales]),
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        ring_start,
        flush_count,
        state_dtype=state.dtype,
        input_dtype=torch.bfloat16,
        matrixA_dtype=torch.float32,
        dim=64,
        dstate=128,
        num_heads=1,
        heads_per_group=1,
        max_window=8,
        ring_buffer_len=ring_buffer_len,
    )
