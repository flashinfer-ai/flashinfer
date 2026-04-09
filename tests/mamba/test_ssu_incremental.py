# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the CUDA ssu_incremental kernel.
Validates against the Triton reference implementation.
"""

import pytest
import torch
from einops import repeat

from flashinfer.mamba.ssu_incremental import ssu_incremental

# Import Triton reference
from .triton_reference.incremental_selective_state_update import (
    incremental_selective_state_update,
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
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["contiguous_cache", "paged_cache"]
)
def test_ssu_incremental(nheads, head_dim, d_state, ngroups, state_dtype, paged_cache):
    """
    For each k in 0..T, run both CUDA and Triton incremental kernels with
    identical inputs and verify output and state match.
    """
    T = 6
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
        incremental_selective_state_update(
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
