# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the CUDA ssu_incremental kernel.
Validates against the Triton replay_selective_state_update reference.
"""

import pytest
import torch
from einops import repeat

from flashinfer.mamba.ssu_incremental import ssu_incremental

# Import Triton reference (upstream "replay" variant)
from .triton_reference.replay_selective_state_update import (
    replay_selective_state_update,
)

# Configs from Nemotron Mamba2 (nheads=128, headdim=64, d_state=128, ngroups=8):
# TP=8: nheads=16, ngroups=1
# TP=4: nheads=32, ngroups=2
_CONFIGS = [
    # (nheads, head_dim, d_state, ngroups)
    (16, 64, 128, 1),
    (32, 64, 128, 2),
]


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["contiguous_cache", "paged_cache"]
)
@pytest.mark.parametrize("T", [2, 6, 16], ids=["mtp2", "mtp6", "mtp16"])
def test_ssu_incremental(
    nheads, head_dim, d_state, ngroups, state_dtype, paged_cache, T
):
    """
    For each k in 0..T, run both CUDA and Triton incremental kernels with
    identical inputs and verify output and state match.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    # A: (nheads, dim, dstate) with tie_hdim
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)

    # dt_bias: (nheads, dim) with tie_hdim
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

    # D: (nheads, dim) with tie_hdim
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Initial state
    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )

    # Step 1 inputs (will become the "old" cached data)
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)  # noqa: F841
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)  # noqa: F841

    # Compute processed dt and cumAdt for step 1 (to build cache)
    dt1_proc = dt1_base.float() + dt_bias_base.float()[None, None, :]
    dt1_proc = torch.where(dt1_proc > 20.0, dt1_proc, torch.log1p(torch.exp(dt1_proc)))
    cumAdt1 = torch.cumsum(A_base.float()[None, None, :] * dt1_proc, dim=1)

    # Build cache tensors
    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt_proc = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    old_cumAdt = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    # Fill caches
    slots = state_batch_indices if paged_cache else slice(None)
    old_x[slots] = x1

    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        old_B[slot, buf] = B1[i]
        old_dt_proc[slot, buf] = dt1_proc[i].permute(1, 0)  # (T, nheads) → (nheads, T)
        old_cumAdt[slot, buf] = cumAdt1[i].permute(1, 0)

    # Test each replay count k
    for k in range(T + 1):
        torch.manual_seed(k + 100)

        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

        # --- Triton reference ---
        ref_state = state0.clone()
        ref_prev = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        replay_selective_state_update(
            ref_state,
            old_x.clone(),
            old_B.clone(),
            old_dt_proc.clone(),
            old_cumAdt.clone(),
            cache_buf_idx.clone(),
            ref_prev,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=ref_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
        )

        # --- CUDA kernel ---
        test_state = state0.clone()
        test_prev = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        ssu_incremental(
            test_state,
            old_x.clone(),
            old_B.clone(),
            old_dt_proc.clone(),
            old_cumAdt.clone(),
            cache_buf_idx.clone(),
            test_prev,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=test_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
        )

        # Compare output
        print(f"\n--- k={k} ---")
        print(f"test_out abs max: {test_out.abs().max().item():.6f}")
        print(f"ref_out abs max: {ref_out.abs().max().item():.6f}")
        print(f"diff abs max: {(test_out - ref_out).abs().max().item():.6f}")
        print(f"test_out[0,0,0,:8]: {test_out[0, 0, 0, :8]}")
        print(f"ref_out[0,0,0,:8]:  {ref_out[0, 0, 0, :8]}")
        print(f"test_state abs max: {test_state[slots].abs().max().item():.6f}")
        print(f"ref_state abs max: {ref_state[slots].abs().max().item():.6f}")
        print(
            f"state diff max: {(test_state[slots] - ref_state[slots]).abs().max().item():.6f}"
        )
        torch.testing.assert_close(
            test_state[slots],
            ref_state[slots],
            rtol=2e-2,
            atol=5e-1,
            msg=f"State mismatch at k={k}",
        )

        torch.testing.assert_close(
            test_out,
            ref_out,
            rtol=2e-2,
            atol=5e-1,
            msg=f"Output mismatch at k={k}",
        )


@pytest.mark.parametrize("T", [27, 32, 55], ids=["T27", "T32", "T55"])
def test_ssu_incremental_rejects_large_T(T):
    """CUDA kernel only supports T <= 16.  Verify it raises for larger T."""
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    A = repeat(
        -torch.rand(nheads, device=device) - 0.5, "h -> h p n", p=head_dim, n=d_state
    )
    dt_bias = repeat(
        torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim
    )
    D = repeat(torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim)

    state = torch.randn(
        batch, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    old_x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt_proc = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(batch, device=device, dtype=torch.int32)
    prev_tokens = torch.full((batch,), T // 2, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt = repeat(
        torch.randn(batch, T, nheads, device=device, dtype=dtype),
        "b t h -> b t h p",
        p=head_dim,
    )
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)

    with pytest.raises(AssertionError, match="at most 16 MTP tokens"):
        ssu_incremental(
            state,
            old_x,
            old_B,
            old_dt_proc,
            old_cumAdt,
            cache_buf_idx,
            prev_tokens,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            out=out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )


@pytest.mark.skip(reason="CUDA kernel does not support Philox stochastic rounding yet")
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
def test_ssu_incremental_philox(nheads, head_dim, d_state, ngroups, paged_cache, T):
    """
    Verify that Philox stochastic rounding produces correct results.

    Runs our CUDA kernel twice with identical inputs: once without rounding
    (fp16 state, deterministic), once with rounding (fp16 state, Philox).
    Compares both against the Triton reference.
    Also verifies the state dtype remains fp16.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float16
    assert nheads % ngroups == 0

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )

    # Cache tensors
    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt_proc = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    old_cumAdt = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    # New token inputs
    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((cache_size,), T // 2, device=device, dtype=torch.int32)

    common_kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
    )

    # --- Run without rounding (deterministic fp16 state store) ---
    state_nornd = state0.clone()
    out_nornd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    ssu_incremental(
        state_nornd,
        old_x.clone(),
        old_B.clone(),
        old_dt_proc.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_nornd,
        **common_kwargs,
    )

    # --- Run with Philox rounding ---
    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rnd = state0.clone()
    out_rnd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    ssu_incremental(
        state_rnd,
        old_x.clone(),
        old_B.clone(),
        old_dt_proc.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rnd,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Outputs should be nearly identical — rounding only perturbs the
    # post-replay state by ±1 ULP before the output phase reads it.
    torch.testing.assert_close(
        out_rnd,
        out_nornd,
        rtol=2e-2,
        atol=5e-1,
        msg="Output diverged with Philox rounding",
    )

    # State should remain fp16
    assert state_rnd.dtype == torch.float16

    # States should differ by at most a few fp16 ULPs per element.
    slots = state_batch_indices if paged_cache else slice(None)
    torch.testing.assert_close(
        state_rnd[slots],
        state_nornd[slots],
        rtol=2e-3,
        atol=0.2,
        msg="State diverged with Philox rounding",
    )


@pytest.mark.skip(reason="CUDA kernel does not support Philox stochastic rounding yet")
def test_philox_rounding_unbiased():
    """
    Verify that Philox stochastic rounding is unbiased.

    Runs the CUDA kernel with fp32 state (capturing the true fp32
    post-replay state) and with fp16 state + Philox rounding.  Compares the
    rounding residual (fp16_state.float() - fp32_state) against deterministic
    rounding (fp32_state.to(fp16).float() - fp32_state).

    Deterministic round-to-nearest-even has a systematic positive bias on
    the residual.  Philox stochastic rounding should be unbiased: the mean
    residual should be near zero.

    Uses a large batch (16) for ~2M state elements — plenty of statistics.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch, T = 16, 6
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Use fp32 initial state so replay produces non-fp16-representable values
    state0 = torch.randn(
        batch, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )

    old_x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt_proc = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    cache_buf_idx = torch.zeros(batch, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt_val = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.full((batch,), T, device=device, dtype=torch.int32)

    common_kwargs = dict(
        x=x,
        dt=dt_val,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    # 1. fp32 state — captures true post-replay state
    state_fp32 = state0.clone()
    out_fp32 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    ssu_incremental(
        state_fp32,
        old_x.clone(),
        old_B.clone(),
        old_dt_proc.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_fp32,
        **common_kwargs,
    )

    # 2. fp16 state with Philox rounding
    rand_seed = torch.tensor([99999], device=device, dtype=torch.int64)
    state_rnd = state0.to(torch.float16).clone()
    out_rnd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    ssu_incremental(
        state_rnd,
        old_x.clone(),
        old_B.clone(),
        old_dt_proc.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rnd,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Compute rounding residuals where fp32 state has non-zero values
    fp32_vals = state_fp32.flatten()
    stochastic_residual = state_rnd.float().flatten() - fp32_vals
    deterministic_residual = fp32_vals.to(torch.float16).float() - fp32_vals

    # Only consider elements where rounding matters (non-zero residual possible)
    nonzero_mask = deterministic_residual.abs() > 0
    n_nonzero = nonzero_mask.sum().item()
    assert n_nonzero > 1000, f"Too few roundable elements: {n_nonzero}"

    stoch_mean = stochastic_residual[nonzero_mask].mean().item()
    determ_mean = deterministic_residual[nonzero_mask].mean().item()

    # Stochastic rounding should be less biased than deterministic.
    # With ~millions of elements, the stochastic mean should be very close to 0.
    assert abs(stoch_mean) < abs(determ_mean) or abs(stoch_mean) < 1e-5, (
        f"Stochastic rounding appears biased: stoch_mean={stoch_mean:.6f}, "
        f"determ_mean={determ_mean:.6f}, n_elements={n_nonzero}"
    )
