"""Correctness tests for the temporary FlashInfer-MUSA SSU provider."""

import os

import pytest

torch = pytest.importorskip("torch")
TEST_DEVICE = torch.device(os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE", "cpu"))

from flashinfer.mamba.musa_reference import (  # noqa: E402
    ssd_combined_fwd_musa_reference,
    ssd_combined_fwd_varlen_musa_reference,
    selective_state_update_musa_reference,
)


def _inputs(*, batch: int = 2, steps: int | None = None):
    torch.manual_seed(0)
    heads, dim, dstate, groups = 2, 3, 4, 1
    state = torch.randn(8, heads, dim, dstate, dtype=torch.float32, device=TEST_DEVICE)
    x_shape = (batch, heads, dim) if steps is None else (batch, steps, heads, dim)
    b_shape = (batch, groups, dstate) if steps is None else (batch, steps, groups, dstate)
    x = torch.randn(*x_shape, dtype=torch.bfloat16, device=TEST_DEVICE)
    dt = torch.randn(*x_shape, dtype=torch.float32, device=TEST_DEVICE)
    A = -torch.rand(heads, dim, dstate, dtype=torch.float32, device=TEST_DEVICE) - 1
    B = torch.randn(*b_shape, dtype=torch.bfloat16, device=TEST_DEVICE)
    C = torch.randn(*b_shape, dtype=torch.bfloat16, device=TEST_DEVICE)
    D = torch.randn(heads, dim, dtype=torch.float32, device=TEST_DEVICE)
    bias = torch.randn(heads, dim, dtype=torch.float32, device=TEST_DEVICE)
    return state, x, dt, A, B, C, D, bias


def test_musa_reference_single_token_updates_selected_slot():
    state, x, dt, A, B, C, D, bias = _inputs()
    original = state.clone()
    out = torch.empty_like(x)
    selected = torch.tensor([3, 5], dtype=torch.int32, device=TEST_DEVICE)

    result = selective_state_update_musa_reference(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        None,
        bias,
        True,
        selected,
        None,
        -1,
        out,
        False,
        intermediate_states_buffer=None,
        intermediate_state_indices=None,
        state_scale=None,
        intermediate_state_scales=None,
        rand_seed=None,
        philox_rounds=0,
        cache_steps=0,
        cu_seqlens=None,
        num_accepted_tokens=None,
    )

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert not torch.equal(state[selected], original[selected])
    untouched = torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=TEST_DEVICE)
    assert torch.equal(state[untouched], original[untouched])


def test_musa_reference_int16_state_and_stochastic_rounding():
    _, x, dt, A, B, C, D, bias = _inputs(batch=1)
    state = torch.zeros(8, 2, 3, 4, dtype=torch.int16, device=TEST_DEVICE)
    scales = torch.ones(8, 2, 3, dtype=torch.float32, device=TEST_DEVICE)
    out = torch.empty_like(x)
    selected = torch.tensor([3], dtype=torch.int32, device=TEST_DEVICE)
    seed = torch.tensor([123], dtype=torch.int64, device=TEST_DEVICE)

    result = selective_state_update_musa_reference(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        None,
        bias,
        True,
        selected,
        None,
        -1,
        out,
        False,
        intermediate_states_buffer=None,
        intermediate_state_indices=None,
        state_scale=scales,
        intermediate_state_scales=None,
        rand_seed=seed,
        philox_rounds=5,
        cache_steps=0,
        cu_seqlens=None,
        num_accepted_tokens=None,
    )

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert torch.isfinite(scales[selected]).all()
    assert torch.any(state[selected] != 0)


def test_musa_reference_mtp_writes_destination_slots_and_intermediates():
    state, x, dt, A, B, C, D, bias = _inputs(batch=1, steps=3)
    out = torch.empty_like(x)
    read = torch.tensor([[2, 3, 4]], dtype=torch.int32, device=TEST_DEVICE)
    destinations = torch.tensor([[4, 5, 6]], dtype=torch.int32, device=TEST_DEVICE)
    accepted = torch.tensor([1], dtype=torch.int32, device=TEST_DEVICE)
    intermediate = torch.empty(
        1, 3, *state.shape[1:], dtype=state.dtype, device=TEST_DEVICE
    )

    result = selective_state_update_musa_reference(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        None,
        bias,
        True,
        read,
        destinations,
        -1,
        out,
        False,
        intermediate,
        intermediate_state_indices=None,
        state_scale=None,
        intermediate_state_scales=None,
        rand_seed=None,
        philox_rounds=3,
        cache_steps=3,
        cu_seqlens=None,
        num_accepted_tokens=accepted,
    )

    assert result.shape == x.shape
    assert torch.isfinite(result).all()
    assert torch.isfinite(intermediate).all()
    assert not torch.equal(state[4], state[5])
    assert not torch.equal(state[5], state[6])


def test_musa_reference_ssd_returns_final_state_and_respects_gate():
    torch.manual_seed(1)
    batch, seqlen, heads, dim, dstate, groups = 2, 4, 2, 3, 4, 1
    x = torch.randn(batch, seqlen, heads, dim, dtype=torch.bfloat16, device=TEST_DEVICE)
    dt = torch.randn(batch, seqlen, heads, dtype=torch.float32, device=TEST_DEVICE)
    A = -torch.rand(heads, dtype=torch.float32, device=TEST_DEVICE) - 1
    B = torch.randn(batch, seqlen, groups, dstate, dtype=torch.bfloat16, device=TEST_DEVICE)
    C = torch.randn_like(B)
    D = torch.randn(heads, dim, dtype=torch.bfloat16, device=TEST_DEVICE)
    z = torch.randn_like(x)
    initial = torch.randn(batch, heads, dim, dstate, dtype=torch.float16, device=TEST_DEVICE)

    output, final = ssd_combined_fwd_musa_reference(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_softplus=True,
        initial_states=initial,
    )

    assert output.shape == x.shape
    assert final is not None
    assert final.shape == initial.shape
    assert final.dtype == initial.dtype
    assert torch.isfinite(output).all()
    assert torch.isfinite(final).all()


def test_musa_reference_ssd_varlen_returns_chunk_or_final_states():
    torch.manual_seed(2)
    heads, dim, dstate, groups = 2, 3, 4, 1
    lengths = [3, 2]
    tokens = sum(lengths)
    x = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device=TEST_DEVICE)
    dt = torch.randn(tokens, heads, dtype=torch.float32, device=TEST_DEVICE)
    A = -torch.rand(heads, dtype=torch.float32, device=TEST_DEVICE) - 1
    B = torch.randn(tokens, groups, dstate, dtype=torch.bfloat16, device=TEST_DEVICE)
    C = torch.randn_like(B)
    initial = torch.randn(2, heads, dim, dstate, dtype=torch.float16, device=TEST_DEVICE)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device=TEST_DEVICE)
    cu_chunk_seqlens = torch.tensor([0, 2, 3, 5], dtype=torch.int32, device=TEST_DEVICE)
    last_chunk_indices = torch.tensor([1, 2], dtype=torch.int32, device=TEST_DEVICE)
    seq_idx = torch.tensor([0, 0, 1], dtype=torch.int32, device=TEST_DEVICE)
    out = torch.empty_like(x)

    intermediate = ssd_combined_fwd_varlen_musa_reference(
        x,
        dt,
        A,
        B,
        C,
        3,
        cu_seqlens,
        cu_chunk_seqlens,
        last_chunk_indices,
        seq_idx,
        out=out,
        initial_states=initial,
        return_intermediate_states=True,
    )
    final = ssd_combined_fwd_varlen_musa_reference(
        x,
        dt,
        A,
        B,
        C,
        3,
        cu_seqlens,
        cu_chunk_seqlens,
        last_chunk_indices,
        seq_idx,
        out=torch.empty_like(x),
        initial_states=initial,
        return_intermediate_states=False,
    )
    assert intermediate.shape == (3, heads, dim, dstate)
    assert final.shape == (2, heads, dim, dstate)
    assert torch.isfinite(intermediate).all()
    assert torch.isfinite(final).all()
