"""Public Mamba API tests executed on a MUSA device."""

import math
import os

import pytest

torch = pytest.importorskip("torch")
if os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa":
    pytest.skip("set FLASHINFER_MAMBA_TEST_DEVICE=musa", allow_module_level=True)

from flashinfer.mamba import (  # noqa: E402
    CakeSSDCombined,
    cake_selective_state_update,
    mamba_chunk_scan_combined_varlen,
    selective_state_update,
    ssd_combined_fwd,
)


DEVICE = torch.device("musa")
H, D, N, G = 2, 3, 4, 1


def _ssu_inputs(batch=1, steps=None):
    x_shape = (batch, H, D) if steps is None else (batch, steps, H, D)
    b_shape = (batch, G, N) if steps is None else (batch, steps, G, N)
    x = torch.randn(*x_shape, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(*x_shape, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, D, N, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(*b_shape, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    D_skip = torch.randn(H, D, device=DEVICE, dtype=torch.float32)
    return x, dt, A, B, C, D_skip


@pytest.mark.parametrize(
    "algorithm", ["auto", "simple", "vertical", "horizontal", "async_horizontal"]
)
def test_selective_state_update_stochastic_rounding(algorithm):
    x, dt, A, B, C, D_skip = _ssu_inputs()
    state = torch.zeros(8, H, D, N, device=DEVICE, dtype=torch.float16)
    slot = torch.tensor([3], device=DEVICE, dtype=torch.int32)
    seed = torch.tensor([123], device=DEVICE, dtype=torch.int64)

    y = selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D_skip,
        state_batch_indices=slot,
        rand_seed=seed,
        philox_rounds=5,
        algorithm=algorithm,
    )

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.any(state[slot] != 0)


def test_selective_state_update_philox_reproducibility():
    x, dt, A, B, C, D_skip = _ssu_inputs()
    seed = torch.tensor([987], device=DEVICE, dtype=torch.int64)
    slot = torch.tensor([2], device=DEVICE, dtype=torch.int32)
    state_a = torch.zeros(8, H, D, N, device=DEVICE, dtype=torch.float16)
    state_b = torch.zeros_like(state_a)
    selective_state_update(
        state_a, x, dt, A, B, C, D=D_skip, state_batch_indices=slot,
        rand_seed=seed, philox_rounds=5,
    )
    selective_state_update(
        state_b, x, dt, A, B, C, D=D_skip, state_batch_indices=slot,
        rand_seed=seed, philox_rounds=5,
    )
    torch.testing.assert_close(state_a, state_b, rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.int8, torch.float8_e4m3fn])
def test_selective_state_update_quantized_state(state_dtype):
    x, dt, A, B, C, D_skip = _ssu_inputs()
    state = torch.zeros(8, H, D, N, device=DEVICE, dtype=state_dtype)
    scales = torch.ones(8, H, D, device=DEVICE, dtype=torch.float32)
    y = selective_state_update(
        state, x, dt, A, B, C, D=D_skip,
        state_batch_indices=torch.tensor([2], device=DEVICE, dtype=torch.int32),
        state_scale=scales,
    )
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(scales).all()


def test_selective_state_update_philox_slot_offset_changes_rounding():
    x, dt, A, B, C, D_skip = _ssu_inputs()
    seed = torch.tensor([987], device=DEVICE, dtype=torch.int64)
    state_a = torch.zeros(8, H, D, N, device=DEVICE, dtype=torch.float16)
    state_b = torch.zeros_like(state_a)
    selective_state_update(
        state_a, x, dt, A, B, C, D=D_skip,
        state_batch_indices=torch.tensor([1], device=DEVICE, dtype=torch.int32),
        rand_seed=seed, philox_rounds=5,
    )
    selective_state_update(
        state_b, x, dt, A, B, C, D=D_skip,
        state_batch_indices=torch.tensor([2], device=DEVICE, dtype=torch.int32),
        rand_seed=seed, philox_rounds=5,
    )
    assert not torch.equal(state_a[1], state_b[2])


def test_cake_selective_state_update_public_api_on_musa():
    x, dt, A, B, C, D_skip = _ssu_inputs()
    state = torch.zeros(8, H, D, N, device=DEVICE, dtype=torch.float16)
    y = cake_selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D_skip,
        state_batch_indices=torch.tensor([2], device=DEVICE, dtype=torch.int32),
    )
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_selective_state_update_mtp_replay_with_intermediate_states():
    steps = 3
    x, dt, A, B, C, D_skip = _ssu_inputs(steps=steps)
    state = torch.zeros(8, H, D, N, device=DEVICE, dtype=torch.float32)
    read = torch.tensor([[2, 3, 4]], device=DEVICE, dtype=torch.int32)
    intermediate = torch.empty(1, steps, H, D, N, device=DEVICE, dtype=torch.float32)

    y = selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D_skip,
        state_batch_indices=read,
        intermediate_states_buffer=intermediate,
        num_accepted_tokens=torch.tensor([1], device=DEVICE, dtype=torch.int32),
        cache_steps=steps,
    )

    assert y.shape == x.shape
    assert intermediate.shape == (1, steps, H, D, N)
    assert torch.isfinite(y).all()
    assert torch.isfinite(intermediate).all()


def test_mamba_chunk_scan_combined_varlen_public_api():
    tokens = 5
    x = torch.randn(tokens, H, D, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(tokens, H, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(tokens, G, N, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    initial = torch.randn(2, H, D, N, device=DEVICE, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 3, 5], device=DEVICE, dtype=torch.int32)
    cu_chunk_seqlens = torch.tensor([0, 2, 3, 5], device=DEVICE, dtype=torch.int32)
    last = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)
    seq_idx = torch.tensor([0, 0, 1], device=DEVICE, dtype=torch.int32)

    states = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        3,
        cu_seqlens,
        cu_chunk_seqlens,
        last,
        seq_idx,
        torch.empty_like(x),
        initial_states=initial,
        return_intermediate_states=True,
        state_dtype=torch.float16,
    )

    assert states.shape == (3, H, D, N)
    assert torch.isfinite(states).all()


def test_cake_ssd_padded_varlen_metadata_on_musa():
    batch, seqlen = 1, 4
    x = torch.randn(batch, seqlen, H, D, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(batch, seqlen, H, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(batch, seqlen, G, N, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    initial = torch.randn(2, H, D, N, device=DEVICE, dtype=torch.float16)
    seq_idx = torch.tensor([[0, 0, 1, 1]], device=DEVICE, dtype=torch.int32)
    runner = CakeSSDCombined(
        4,
        H,
        D,
        N,
        G,
        io_dtype=torch.bfloat16,
        state_dtype=torch.float16,
        has_d=False,
        d_has_hdim=False,
        has_initial_states=True,
        has_varlen=True,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )
    out, final = runner.run(
        x,
        dt,
        A,
        B,
        C,
        initial_states=initial,
        seq_idx=seq_idx,
        chunk_indices=torch.tensor([0], device=DEVICE, dtype=torch.int32),
        chunk_offsets=torch.tensor([0], device=DEVICE, dtype=torch.int32),
        return_final_states=True,
    )
    assert out.shape == x.shape
    assert final.shape == initial.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(final).all()


def test_cake_ssd_packed_varlen_metadata_on_musa():
    tokens = 5
    x = torch.randn(tokens, H, D, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(tokens, H, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(tokens, G, N, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    runner = CakeSSDCombined(
        3,
        H,
        D,
        N,
        G,
        io_dtype=torch.bfloat16,
        state_dtype=torch.float16,
        has_d=False,
        d_has_hdim=False,
        has_initial_states=False,
        has_varlen=True,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )
    out, final = runner.run(
        x,
        dt,
        A,
        B,
        C,
        seq_idx=torch.tensor([0, 0, 0, 1, 1], device=DEVICE, dtype=torch.int32),
        chunk_offsets=torch.tensor([0, 3, 5], device=DEVICE, dtype=torch.int32),
        return_final_states=True,
    )
    assert out.shape == x.shape
    assert final.shape == (2, H, D, N)
    assert torch.isfinite(out).all()
    assert torch.isfinite(final).all()


def test_ssd_checkpoint_token_and_slot_on_musa():
    batch, seqlen = 1, 4
    x = torch.ones(batch, seqlen, H, D, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.full((batch, seqlen, H), 0.25, device=DEVICE, dtype=torch.float32)
    A = -torch.ones(H, device=DEVICE, dtype=torch.float32)
    B = torch.ones(batch, seqlen, G, N, device=DEVICE, dtype=torch.bfloat16)
    C = torch.ones_like(B)
    checkpoint = torch.full(
        (1, H, D, N), float("nan"), device=DEVICE, dtype=torch.float16
    )
    out, final = ssd_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        checkpoint_token_indices=torch.tensor([2], device=DEVICE, dtype=torch.int32),
        checkpoint_state_slots=torch.tensor([0], device=DEVICE, dtype=torch.int32),
        checkpoint_states=checkpoint,
    )
    assert out.shape == x.shape
    assert final.shape == (batch, H, D, N)
    # After two tokens, s_2 = 0.25 * exp(-0.25) + 0.25. Random negative
    # dt values can legitimately clamp to zero and leave a zero checkpoint.
    expected = torch.full_like(checkpoint, 0.25 * (1 + math.exp(-0.25)))
    torch.testing.assert_close(checkpoint, expected, rtol=0, atol=0)


def test_ssd_matches_independent_eager_recurrence_on_musa():
    batch, seqlen = 1, 3
    x = torch.randn(batch, seqlen, H, D, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(batch, seqlen, H, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(batch, seqlen, G, N, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    out, final = ssd_combined_fwd(x, dt, A, B, C)
    state = torch.zeros(batch, H, D, N, device=DEVICE, dtype=torch.float32)
    expected = torch.empty(batch, seqlen, H, D, device=DEVICE, dtype=torch.float32)
    for token in range(seqlen):
        delta = dt[:, token].clamp_min(0)
        state = state * torch.exp(A[None, :, None, None] * delta[:, :, None, None])
        for head in range(H):
            state[:, head] += (
                delta[:, head, None, None]
                * x[:, token, head].to(torch.float32)[:, :, None]
                * B[:, token, 0].to(torch.float32)[:, None, :]
            )
            expected[:, token, head] = torch.sum(
                C[:, token, 0].to(torch.float32)[:, None, :] * state[:, head], dim=-1
            )
    torch.testing.assert_close(out.to(torch.float32), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(final.to(torch.float32), state, rtol=2e-2, atol=2e-2)


def test_ssd_update_seq_chunk_cumsum_on_musa():
    x = torch.randn(1, 4, H, D, device=DEVICE, dtype=torch.bfloat16)
    dt = torch.randn(1, 4, H, device=DEVICE, dtype=torch.float32)
    A = -torch.rand(H, device=DEVICE, dtype=torch.float32) - 1
    B = torch.randn(1, 4, G, N, device=DEVICE, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    seq_idx = torch.tensor([[0, 0, 1, 1]], device=DEVICE, dtype=torch.int32)
    chunk_indices = torch.tensor([0, 1], device=DEVICE, dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 2], device=DEVICE, dtype=torch.int32)
    cumsum = torch.empty(3, device=DEVICE, dtype=torch.int32)
    out, _ = ssd_combined_fwd(
        x, dt, A, B, C, seq_idx=seq_idx,
        chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=cumsum, update_seq_chunk_cumsum=True,
    )
    assert out.shape == x.shape
    assert torch.equal(cumsum.cpu(), torch.tensor([0, 1, 2], dtype=torch.int32))
