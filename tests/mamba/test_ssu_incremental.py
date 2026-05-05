# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the CUDA ssu_incremental kernel.
Validates against the Triton replay_selective_state_update reference.
"""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import repeat

from flashinfer.mamba.ssu_incremental import ssu_incremental

# Import Triton reference (upstream "replay" / new "checkpointing" variant —
# same file, alias kept for backward compat).
from .triton_reference.replay_selective_state_update import (
    _get_sm_version,
    checkpointing_state_update,
    replay_selective_state_update,
)
from .triton_reference.selective_state_update import (
    selective_state_update_triton as selective_state_update,
)

# Configs from Nemotron Mamba2 (nheads=128, headdim=64, d_state=128, ngroups=8):
# TP=8: nheads=16, ngroups=1
# TP=4: nheads=32, ngroups=2
_CONFIGS = [
    # (nheads, head_dim, d_state, ngroups)
    (16, 64, 128, 1),
    (32, 64, 128, 2),
]


def _run_ssu_incremental_case(
    nheads, head_dim, d_state, ngroups, state_dtype, paged_cache, T, d_split=None
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
        print(
            f"PYTEST k={k}: x2[0,0,0,0]={x2[0, 0, 0, 0].item()}, B2[0,0,0,0]={B2[0, 0, 0, 0].item()}, C2[0,0,0,0]={C2[0, 0, 0, 0].item()}"
        )

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
        kernel_kwargs = {}
        if d_split is not None:
            kernel_kwargs["d_split"] = d_split
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
            **kernel_kwargs,
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
        diff = (test_state[slots] - ref_state[slots]).abs()
        max_diff = diff.max().item()
        print(f"state diff max: {max_diff:.6f}")
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


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [2, 6, 16], ids=["mtp2", "mtp6", "mtp16"])
def test_ssu_incremental(nheads, head_dim, d_state, ngroups, state_dtype, T):
    """Paged-cache path: exercises configs × state dtypes × T."""
    _run_ssu_incremental_case(
        nheads, head_dim, d_state, ngroups, state_dtype, paged_cache=True, T=T
    )


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [2, 6, 16], ids=["mtp2", "mtp6", "mtp16"])
@pytest.mark.parametrize(
    "paged_cache", [True, False], ids=["paged_cache", "no_cache_indices"]
)
def test_ssu_incremental_d_split2(
    nheads, head_dim, d_state, ngroups, state_dtype, T, paged_cache
):
    """v12 §59 — exercise the D_SPLIT=2 (D-output split) kernel path.

    Forces ``d_split=2`` via the public Python API so the JIT compiles the
    ``D_SPLIT=2`` kernel specialization and the grid runs with two CTAs per
    head (``D_PER_CTA = head_dim / 2 = 32``).  Compares against the Triton
    reference using the existing tolerance — must match every parameter
    combination (paged + contiguous, all state dtypes, mtp ∈ {2, 6, 16}).
    """
    _run_ssu_incremental_case(
        nheads,
        head_dim,
        d_state,
        ngroups,
        state_dtype,
        paged_cache=paged_cache,
        T=T,
        d_split=2,
    )


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
# Pairs chosen so prev_k can reach the must_checkpoint boundary while staying
# within Triton's supported range (prev_k <= NPREDICTED — the Triton main
# kernel masks cache reads at offs_t < T, so prev_k > T is unsupported).
@pytest.mark.parametrize(
    "npredicted,max_window",
    # (4, 8) exercises the K_SMALL replay path (MAX_WINDOW_PAD_MMA_K=8 → m16n8k8 atom);
    # (10/12/14, 16) exercise K_BIG (MAX_WINDOW_PAD_MMA_K=16 → m16n8k16 atom).
    [(4, 8), (10, 16), (12, 16), (14, 16)],
    ids=["np4w8", "np10w16", "np12w16", "np14w16"],
)
def test_ssu_incremental_max_window_gt_npredicted(
    nheads, head_dim, d_state, ngroups, state_dtype, paged_cache, npredicted, max_window
):
    """Verify CUDA matches Triton when MAX_WINDOW > NPREDICTED.

    Iterates prev_k over [0, npredicted] (Triton's supported range — its main
    kernel masks cache reads at offs_t < T_arg, so it can only handle replay
    of up to NPREDICTED old tokens).  The kernel implicitly derives
    must_checkpoint = (prev_k + npredicted > max_window), and Triton's
    write_checkpoint flag is set to match.  By choosing npredicted close to
    max_window (e.g. (10, 16)), prev_k crosses the must_checkpoint boundary
    while staying within Triton's range — so we exercise both the
    no-checkpoint and checkpoint paths.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    assert nheads % ngroups == 0
    assert npredicted < max_window, "this test exercises the strict-less-than case"
    assert max_window <= 16, "kernel cap"

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

    # Step-1 inputs (x1/B1/dt1) populate the cache at slots [0, npredicted)
    # of the active buffer.  old_dt_proc / old_cumAdt MUST be consistent
    # with these (real softplus / cumsum), not arbitrary random — otherwise
    # the replay's `coeff = exp(total - old_cumAdt) * old_dt` produces
    # nonsensical scaling that drives the post-replay state into magnitudes
    # where fp16's ULP can't keep up with Triton's fp32-register precision.
    # Slots [npredicted, max_window) keep their `torch.randn` init (never
    # read by replay since prev_k <= npredicted in this test).
    x1 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_proc, dim=1)

    old_x = torch.zeros(
        cache_size, max_window, nheads, head_dim, device=device, dtype=dtype
    )
    old_B = torch.randn(
        cache_size, 2, max_window, ngroups, d_state, device=device, dtype=dtype
    )
    old_dt = torch.randn(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    old_dA_cumsum = torch.randn(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    # Populate the active buffer's [0:npredicted) with step-1 consistent data.
    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    slots = state_batch_indices if paged_cache else slice(None)
    old_x[slots, :npredicted] = x1
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        old_B[slot, buf, :npredicted] = B1[i]
        old_dt[slot, buf, :, :npredicted] = dt1_proc[i].T  # (T, nheads) → (nheads, T)
        old_dA_cumsum[slot, buf, :, :npredicted] = dA_cumsum1[i].T

    for prev_k in range(0, npredicted + 1):
        torch.manual_seed(prev_k + 200)

        x2 = torch.randn(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        dt2_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
        dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
        B2 = torch.randn(
            batch, npredicted, ngroups, d_state, device=device, dtype=dtype
        )
        C2 = torch.randn(
            batch, npredicted, ngroups, d_state, device=device, dtype=dtype
        )

        must_checkpoint = (prev_k + npredicted) > max_window

        # ── Triton reference ──
        ref_state = state0.clone()
        ref_prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
        ref_out = torch.zeros(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        old_x_ref = old_x.clone()
        old_B_ref = old_B.clone()
        old_dt_ref = old_dt.clone()
        old_dA_ref = old_dA_cumsum.clone()
        replay_selective_state_update(
            ref_state,
            old_x_ref,
            old_B_ref,
            old_dt_ref,
            old_dA_ref,
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
            write_checkpoint=must_checkpoint,
        )

        # ── CUDA kernel ──
        test_state = state0.clone()
        test_prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
        test_out = torch.zeros(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        old_x_test = old_x.clone()
        old_B_test = old_B.clone()
        old_dt_test = old_dt.clone()
        old_dA_test = old_dA_cumsum.clone()
        ssu_incremental(
            test_state,
            old_x_test,
            old_B_test,
            old_dt_test,
            old_dA_test,
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

        torch.testing.assert_close(
            test_out,
            ref_out,
            rtol=2e-2,
            atol=5e-1,
            msg=f"out mismatch at prev_k={prev_k} (must_checkpoint={must_checkpoint})",
        )
        slots = state_batch_indices if paged_cache else slice(None)
        torch.testing.assert_close(
            test_state[slots],
            ref_state[slots],
            rtol=2e-2,
            atol=5e-1,
            msg=f"state mismatch at prev_k={prev_k} (must_checkpoint={must_checkpoint})",
        )
        # Cache writes — both kernels should land at the same slot/offset.
        torch.testing.assert_close(
            old_x_test,
            old_x_ref,
            rtol=0,
            atol=0,
            msg=f"old_x mismatch at prev_k={prev_k}",
        )
        torch.testing.assert_close(
            old_B_test,
            old_B_ref,
            rtol=0,
            atol=0,
            msg=f"old_B mismatch at prev_k={prev_k}",
        )
        torch.testing.assert_close(
            old_dt_test,
            old_dt_ref,
            rtol=1e-4,
            atol=1e-4,
            msg=f"old_dt mismatch at prev_k={prev_k}",
        )
        torch.testing.assert_close(
            old_dA_test,
            old_dA_ref,
            rtol=1e-4,
            atol=1e-4,
            msg=f"old_dA mismatch at prev_k={prev_k}",
        )


def test_ssu_incremental_contiguous():
    """Smoke test for the contiguous-cache path (TP=8, bf16 state, mtp=16)."""
    _run_ssu_incremental_case(
        nheads=16,
        head_dim=64,
        d_state=128,
        ngroups=1,
        state_dtype=torch.bfloat16,
        paged_cache=False,
        T=16,
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

    with pytest.raises(AssertionError, match="at most 16 cache tokens"):
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


# ============================================================================
# Tests ported from upstream `test_checkpointing_state_update.py`.
#
# These exercise the merged Triton reference (`checkpointing_state_update`)
# directly — they DO NOT yet target the FlashInfer CUDA kernel.  Step 3 of the
# porting plan wires the CUDA kernel for the new functionality (per-batch
# `write_checkpoint`, `max_window > T`, quantized state, fp8 SR); after that,
# we extend / replace these tests to compare CUDA vs Triton.  Marked `xfail`
# in the meantime per `.plans/e4m3_stochastic_rounding.md` Step 2.
#
# The upstream tests use `selective_state_update_triton` as the fp32 reference;
# kept as-is.  We dropped the trtllm imports in favour of the FlashInfer-local
# Triton reference + `_get_sm_version`.
# ============================================================================


# Philox stochastic rounding uses PTX cvt.rs.f16x2.f32 / cvt.rs.satfinite.e4m3x4.f32
# which require sm >= 100.
_skip_pre_sm100 = pytest.mark.skipif(
    _get_sm_version() < 100, reason="Philox stochastic rounding needs sm >= 100"
)

# Quantized state dtypes and their representable-magnitude limits (== QUANT_MAX
# in the kernel).  fp8_e4m3fn cells require SM 89+ for the fp32↔fp8 cvt PTX
# instructions; SR variants of fp16/fp8 additionally need SM 100+.
_QUANT_MAX_BY_DTYPE = {
    torch.int8: 127.0,
    torch.int16: 32767.0,
    torch.float8_e4m3fn: 448.0,
}


def _quantize_state(
    state_fp32: torch.Tensor, state_dtype: torch.dtype, quant_max: float
):
    """Quantize fp32 state to (state_quant, decode_scale) using the same
    per-(head, dim) channel scheme the kernel does on store.  decode_scale =
    max_abs_per_channel / quant_max (= 1/encode_scale).
    """
    amax = state_fp32.abs().amax(dim=-1)  # (cache, nheads, head_dim)
    encode_scale = quant_max / amax.clamp(min=1e-30)
    decode_scale = 1.0 / encode_scale
    scaled = state_fp32 * encode_scale.unsqueeze(-1)
    if state_dtype == torch.float8_e4m3fn:
        # Native cast does RN at the fp8 grid; explicit round() would destroy
        # sub-integer precision (matches the kernel's fp8 RN path).
        state_quant = scaled.clamp(-quant_max, quant_max).to(state_dtype)
    else:
        state_quant = scaled.round().clamp(-quant_max, quant_max).to(state_dtype)
    return state_quant, decode_scale


def _dequantize_state(state_quant: torch.Tensor, decode_scale: torch.Tensor):
    return state_quant.to(torch.float32) * decode_scale.unsqueeze(-1)


def _maybe_skip_dtype(state_dtype, use_sr):
    """Skip on insufficient SM.  fp8 e4m3fn (any) needs SM 89+; fp16/fp8 SR
    needs SM 100+; int8/int16 (RN or SR) runs anywhere."""
    if state_dtype == torch.float8_e4m3fn and _get_sm_version() < 89:
        pytest.skip("fp8_e4m3fn requires SM 89+ (Ada Lovelace / Hopper / Blackwell)")
    if (
        use_sr
        and state_dtype in (torch.float16, torch.float8_e4m3fn)
        and _get_sm_version() < 100
    ):
        pytest.skip(
            f"{state_dtype} stochastic rounding requires SM 100+ (Blackwell B200+)"
        )


_XFAIL_REASON = (
    "Step 3 of the porting plan: CUDA kernel wiring for the new checkpointing "
    "functionality (per-batch write_checkpoint, max_window > T, quantized state, "
    "fp8 SR) is not yet landed.  See .plans/e4m3_stochastic_rounding.md."
)


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "state_dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.int16,
        torch.float8_e4m3fn,
    ],
    ids=["fp16", "bf16", "fp32", "int8", "int16", "fp8"],
)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize(
    "T", [6, 10, 16, 27, 32, 55], ids=["T6", "T10", "T16", "T27", "T32", "T55"]
)
@pytest.mark.parametrize("write_checkpoint", [True, False], ids=["write", "no_write"])
def test_checkpointing_state_update(
    nheads, head_dim, d_state, ngroups, state_dtype, paged_cache, T, write_checkpoint
):
    """
    Verify that:
      checkpointing_state_update(state0, old_caches, k, new_x, ...)
    produces the same output as:
      selective_state_update(state_after_k_old_tokens, new_x, ...)
    and writes state_after_k_old_tokens back to the state tensor.

    Quantized state dtypes (int8/int16/fp8) follow the same flow with
    a per-(head, dim) channel decode-scale tensor; comparison is done
    via dequant(state, scales) against the fp32 reference.
    """
    _maybe_skip_dtype(state_dtype, use_sr=False)

    quant_max = _QUANT_MAX_BY_DTYPE.get(state_dtype, 0.0)
    is_quantized = quant_max > 0.0

    batch = 2
    device = "cuda"
    dtype = torch.bfloat16  # input activations are bf16
    assert nheads % ngroups == 0

    # Cache T-axis size (max_window).  Use the kernel's BLOCK_SIZE_T as the
    # ceiling — this is what the wrapper allows and enables PNAT-aware writes
    # at [PNAT, PNAT+T) for no-checkpoint mode.  For T=6 that's 16 (production
    # max_window); for larger T it scales with np2(T).
    max_window = max(triton.next_power_of_2(T), 16)

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    else:
        cache_size = batch
        state_batch_indices = None

    torch.manual_seed(42)

    # A: (nheads, head_dim, d_state) with stride(-2)=0, stride(-1)=0  [tie_hdim]
    A_base = -torch.rand(nheads, device=device) - 0.5  # float32, negative
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)

    # dt_bias: (nheads, head_dim) with stride(-1)=0  [tie_hdim]
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)

    # D: (nheads, head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # Initial SSM state (cache_size slots).  Quantized dtypes need a separate
    # init: derive scales from a fp32 source so the quantized state isn't
    # garbage on dequant.  ref_input_state is what the fp32 reference run
    # sees — for non-quant it's state0 (cast to fp32 inside reference); for
    # quant it's the lossy dequant of state0 (matches what the kernel sees
    # internally on load).
    if is_quantized:
        state0_fp32 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
        )
        state0, state0_scales = _quantize_state(state0_fp32, state_dtype, quant_max)
        ref_input_state = _dequantize_state(state0, state0_scales)
    else:
        state0 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
        )
        state0_scales = None
        ref_input_state = state0.float()

    # Old inputs: T tokens per batch request
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)  # stride(-1)=0
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # Capture intermediate SSM states using selective_state_update.
    states_buffer_f32 = torch.zeros(
        cache_size, T, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = (
        state_batch_indices
        if paged_cache
        else torch.arange(batch, device=device, dtype=torch.int32)
    )
    out1 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_input_state.clone(),
        x1,
        dt1,
        A,
        B1,
        C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=T,
        out=out1,
        disable_state_update=True,
    )

    # Build cache tensors for the replay kernel.
    # old_x: (cache, max_window, nheads, dim) bf16 — single-buffered
    # old_B: (cache, 2, max_window, ngroups, dstate) bf16 — double-buffered
    # old_dt: (cache, 2, nheads, max_window) fp32 — double-buffered, T contiguous
    # old_dA_cumsum: (cache, 2, nheads, max_window) fp32 — double-buffered, T contiguous
    # cache_buf_idx: random 0s and 1s to verify indexing correctness
    old_x = torch.zeros(
        cache_size, max_window, nheads, head_dim, device=device, dtype=dtype
    )
    old_B = torch.randn(
        cache_size, 2, max_window, ngroups, d_state, device=device, dtype=dtype
    )
    old_dt = torch.randn(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    old_dA_cumsum = torch.randn(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    # Fill each slot's active buffer (= cache_buf_idx) with step 1's data at
    # positions [0:T).  Positions [T:max_window) stay as torch.randn garbage —
    # they're outside the test's PNAT range, kernel doesn't read them.
    # The OTHER (inactive) buffer has random garbage to catch indexing bugs.
    slots = state_batch_indices if paged_cache else slice(None)
    old_x[slots, :T] = x1

    # Compute processed dt and dA_cumsum for step 1
    dt1 = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1, dim=1)

    # Write to each slot's active buffer based on its cache_buf_idx
    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        batch_idx = i  # maps slot back to the batch index
        old_B[slot, buf, :T] = B1[batch_idx]
        old_dt[slot, buf, :, :T] = dt1[batch_idx].T  # (T, nheads) → (nheads, T)
        old_dA_cumsum[slot, buf, :, :T] = dA_cumsum1[
            batch_idx
        ].T  # (T, nheads) → (nheads, T)

    # Main loop: test each k (number of old tokens replayed).  For
    # write_checkpoint=False, the kernel writes new tokens at [k:k+T) of the
    # active buffer — skip k where this would exceed max_window.
    for k in range(T + 1):
        if not write_checkpoint and k + T > max_window:
            continue
        torch.manual_seed(k + 100)

        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

        # Reference (fp32, starting from the same lossy-or-not state the
        # kernel sees).
        ref_state_f32 = ref_input_state.clone()
        if k > 0:
            ref_state_f32[slots] = states_buffer_f32[slots, k - 1]

        ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        selective_state_update(
            ref_state_f32,
            x2,
            dt2,
            A,
            B2,
            C2,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=(state_batch_indices if paged_cache else None),
            out=ref_out,
        )

        # Replay kernel — clone caches into mutable working copies that we
        # can inspect AFTER the call to verify cache postconditions.
        test_state = state0.clone()
        test_scales = state0_scales.clone() if is_quantized else None
        prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        old_x_w = old_x.clone()
        old_B_w = old_B.clone()
        old_dt_w = old_dt.clone()
        old_dA_cumsum_w = old_dA_cumsum.clone()
        # cache_buf_idx stays at its random values — each slot reads from its own buffer

        checkpointing_state_update(
            test_state,
            old_x_w,
            old_B_w,
            old_dt_w,
            old_dA_cumsum_w,
            cache_buf_idx.clone(),
            prev_tokens,
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
            state_scales=test_scales,
            write_checkpoint=write_checkpoint,
        )

        out_atol = (
            {torch.int8: 1.6, torch.int16: 1.05, torch.float8_e4m3fn: 4.0}[state_dtype]
            if is_quantized
            else 1.0
        )
        out_rtol = (
            {torch.int8: 2e-2, torch.int16: 2e-2, torch.float8_e4m3fn: 5e-2}[
                state_dtype
            ]
            if is_quantized
            else 2e-2
        )
        torch.testing.assert_close(
            test_out,
            ref_out,
            rtol=out_rtol,
            atol=out_atol,
            msg=f"Output mismatch at k={k}",
        )

        # State expectation depends on write_checkpoint:
        #   True  → kernel writes the post-replay state; expect the
        #           selective_state_update reference's state at step k-1.
        #   False → kernel skips the HBM store; state must be UNCHANGED
        #           from the input (state0; for quant, scales also unchanged).
        if is_quantized:
            if write_checkpoint:
                expected_fp32 = (
                    ref_input_state[slots]
                    if k == 0
                    else states_buffer_f32[slots, k - 1]
                )
                actual_fp32 = _dequantize_state(test_state[slots], test_scales[slots])
                state_atol = {
                    torch.int8: 1.1,
                    torch.int16: 1.0,
                    torch.float8_e4m3fn: 2.5,
                }[state_dtype]
                state_rtol = {
                    torch.int8: 5e-2,
                    torch.int16: 2e-2,
                    torch.float8_e4m3fn: 1e-1,
                }[state_dtype]
                torch.testing.assert_close(
                    actual_fp32,
                    expected_fp32,
                    rtol=state_rtol,
                    atol=state_atol,
                    msg=f"State mismatch at k={k} dtype={state_dtype}",
                )
                assert test_scales.dtype == torch.float32
                assert torch.isfinite(test_scales[slots]).all()
                assert (test_scales[slots] > 0).all()
            else:
                assert torch.equal(test_state[slots], state0[slots])
                assert torch.equal(test_scales[slots], state0_scales[slots])
        else:
            if write_checkpoint:
                expected_state = (
                    state0[slots]
                    if k == 0
                    else states_buffer_f32[slots, k - 1].to(state_dtype)
                )
            else:
                expected_state = state0[slots]
            torch.testing.assert_close(
                test_state[slots],
                expected_state,
                rtol=2e-2,
                atol=1.0 if write_checkpoint else 0.0,
                msg=f"State mismatch at k={k} (write_checkpoint={write_checkpoint})",
            )

        # --- Cache postconditions ---
        dt2_proc = F.softplus(dt2_base.float() + dt_bias_base.float()[None, None, :])
        dA_cumsum2 = torch.cumsum(A_base.float()[None, None, :] * dt2_proc, dim=1)
        write_offset = 0 if write_checkpoint else k

        for batch_idx, slot in enumerate(slot_indices):
            active = cache_buf_idx[slot].item()
            wb = (1 - active) if write_checkpoint else active

            # --- old_x (single-buffered): write at [write_offset : +T) of slot ---
            torch.testing.assert_close(
                old_x_w[slot, write_offset : write_offset + T],
                x2[batch_idx],
                rtol=0,
                atol=0,
            )
            if write_offset > 0:
                torch.testing.assert_close(
                    old_x_w[slot, :write_offset],
                    old_x[slot, :write_offset],
                    rtol=0,
                    atol=0,
                )
            if write_offset + T < max_window:
                torch.testing.assert_close(
                    old_x_w[slot, write_offset + T :],
                    old_x[slot, write_offset + T :],
                    rtol=0,
                    atol=0,
                )

            # --- old_B (double-buffered): write at write_buf, [write_offset:+T) ---
            torch.testing.assert_close(
                old_B_w[slot, wb, write_offset : write_offset + T],
                B2[batch_idx],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                old_B_w[slot, 1 - wb],
                old_B[slot, 1 - wb],
                rtol=0,
                atol=0,
            )

            # --- old_dt (double-buffered, fp32, layout (heads, T)): ---
            torch.testing.assert_close(
                old_dt_w[slot, wb, :, write_offset : write_offset + T],
                dt2_proc[batch_idx].T,
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                old_dt_w[slot, 1 - wb],
                old_dt[slot, 1 - wb],
                rtol=0,
                atol=0,
            )

            # --- old_dA_cumsum (double-buffered, fp32, layout (heads, T)): ---
            torch.testing.assert_close(
                old_dA_cumsum_w[slot, wb, :, write_offset : write_offset + T],
                dA_cumsum2[batch_idx].T,
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                old_dA_cumsum_w[slot, 1 - wb],
                old_dA_cumsum[slot, 1 - wb],
                rtol=0,
                atol=0,
            )


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "state_dtype",
    [torch.float16, torch.int8, torch.int16, torch.float8_e4m3fn],
    ids=["fp16", "int8", "int16", "fp8"],
)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize("T", [6, 16, 32], ids=["T6", "T16", "T32"])
def test_checkpointing_state_update_philox(
    state_dtype, nheads, head_dim, d_state, ngroups, paged_cache, T
):
    """
    Verify that Philox stochastic rounding produces correct results across
    all SR-supported state dtypes (fp16, int8, int16, fp8_e4m3fn).
    """
    _maybe_skip_dtype(state_dtype, use_sr=True)

    quant_max = _QUANT_MAX_BY_DTYPE.get(state_dtype, 0.0)
    is_quantized = quant_max > 0.0

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

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    if is_quantized:
        state0_fp32 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
        )
        state0, state0_scales = _quantize_state(state0_fp32, state_dtype, quant_max)
    else:
        state0 = torch.randn(
            cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
        )
        state0_scales = None

    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

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

    state_no_round = state0.clone()
    scales_no_round = state0_scales.clone() if is_quantized else None
    out_no_round = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_state_update(
        state_no_round,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_no_round,
        state_scales=scales_no_round,
        **common_kwargs,
    )

    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rounded = state0.clone()
    scales_rounded = state0_scales.clone() if is_quantized else None
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_state_update(
        state_rounded,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rounded,
        rand_seed=rand_seed,
        philox_rounds=10,
        state_scales=scales_rounded,
        **common_kwargs,
    )

    out_atol = (
        {torch.int8: 1.5, torch.int16: 1.0, torch.float8_e4m3fn: 6.0}[state_dtype]
        if is_quantized
        else 1.0
    )
    out_rtol = (
        {torch.int8: 2e-2, torch.int16: 2e-2, torch.float8_e4m3fn: 5e-2}[state_dtype]
        if is_quantized
        else 2e-2
    )
    torch.testing.assert_close(
        out_rounded,
        out_no_round,
        rtol=out_rtol,
        atol=out_atol,
        msg=f"Output diverged with Philox rounding ({state_dtype})",
    )

    assert state_rounded.dtype == state_dtype

    slots = state_batch_indices if paged_cache else slice(None)
    if is_quantized:
        rounded_fp32 = _dequantize_state(state_rounded[slots], scales_rounded[slots])
        no_round_fp32 = _dequantize_state(state_no_round[slots], scales_no_round[slots])
        diff = (rounded_fp32 - no_round_fp32).abs()
        scale_bound = torch.maximum(
            scales_no_round[slots], scales_rounded[slots]
        ).unsqueeze(-1)
        cell_pad = 32.0 if state_dtype == torch.float8_e4m3fn else 1.0
        bound = scale_bound * (cell_pad * 1.5)
        assert (diff <= bound).all(), (
            f"State RN-SR diff exceeds 1 cell per element ({state_dtype})"
        )
    else:
        torch.testing.assert_close(
            state_rounded[slots],
            state_no_round[slots],
            rtol=2e-3,
            atol=0.2,
            msg=f"State diverged with Philox rounding ({state_dtype})",
        )


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
@pytest.mark.parametrize(
    "state_dtype",
    [torch.float16, torch.int8, torch.int16, torch.float8_e4m3fn],
    ids=["fp16", "int8", "int16", "fp8"],
)
def test_checkpointing_philox_rounding_unbiased(state_dtype):
    """Verify Philox SR is statistically unbiased (mean residual ≈ 0).

    Renamed from the upstream `test_philox_rounding_unbiased` to disambiguate
    from the existing CUDA-kernel-specific `test_philox_rounding_unbiased`
    above (which tests `ssu_incremental` directly, not the merged Triton)."""
    _maybe_skip_dtype(state_dtype, use_sr=True)

    quant_max = _QUANT_MAX_BY_DTYPE.get(state_dtype, 0.0)
    is_quantized = quant_max > 0.0

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

    state0_fp32 = torch.randn(
        batch, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )

    old_x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
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

    state_fp32 = state0_fp32.clone()
    out_fp32 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_state_update(
        state_fp32,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_fp32,
        **common_kwargs,
    )

    rand_seed = torch.tensor([99999], device=device, dtype=torch.int64)
    if is_quantized:
        state_rounded, scales_rounded = _quantize_state(
            state0_fp32, state_dtype, quant_max
        )
    else:
        state_rounded = state0_fp32.to(state_dtype)
        scales_rounded = None
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_state_update(
        state_rounded,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rounded,
        rand_seed=rand_seed,
        philox_rounds=10,
        state_scales=scales_rounded,
        **common_kwargs,
    )

    if is_quantized:
        fp32_vals = state_fp32.flatten()
        stochastic_residual = (
            _dequantize_state(state_rounded, scales_rounded).flatten() - fp32_vals
        )
        det_quant, det_scales = _quantize_state(state_fp32, state_dtype, quant_max)
        deterministic_residual = (
            _dequantize_state(det_quant, det_scales).flatten() - fp32_vals
        )
    else:
        fp32_vals = state_fp32.flatten()
        stochastic_residual = state_rounded.float().flatten() - fp32_vals
        deterministic_residual = fp32_vals.to(state_dtype).float() - fp32_vals

    nonzero_mask = deterministic_residual.abs() > 0
    num_nonzero = nonzero_mask.sum().item()
    assert num_nonzero > 1000

    stochastic_mean = stochastic_residual[nonzero_mask].mean().item()
    stochastic_std = stochastic_residual[nonzero_mask].std().item()

    se_sr = stochastic_std / (num_nonzero**0.5)
    K = 4
    assert abs(stochastic_mean) < K * se_sr, (
        f"SR mean exceeds {K}*SE (likely biased) ({state_dtype}): "
        f"stochastic_mean={stochastic_mean:.3e}, SE={se_sr:.3e}"
    )


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [6, 16, 32], ids=["T6", "T16", "T32"])
@pytest.mark.parametrize("heads_per_block", [2, 4], ids=["HPB2", "HPB4"])
def test_checkpointing_heads_per_block(
    nheads,
    head_dim,
    d_state,
    ngroups,
    state_dtype,
    T,
    heads_per_block,
):
    """Single-step HPB > 1 test — exercises the precompute kernel's two-loop
    structure (store per-head dt/dA_cumsum in loop 1, reload in loop 2)."""
    batch = 8
    device = "cuda"
    dtype = torch.bfloat16

    if nheads % heads_per_block != 0:
        pytest.skip(
            f"nheads ({nheads}) not divisible by heads_per_block ({heads_per_block})"
        )
    if heads_per_block > nheads // ngroups:
        pytest.skip(
            f"heads_per_block ({heads_per_block}) exceeds heads_per_group ({nheads // ngroups})"
        )

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    cache_size = batch
    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )

    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    states_buffer_f32 = torch.zeros(
        cache_size, T, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    cache_idx_for_capture = torch.arange(batch, device=device, dtype=torch.int32)
    out1 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        state0.clone(),
        x1,
        dt1,
        A,
        B1,
        C1,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=cache_idx_for_capture,
        intermediate_states_buffer=states_buffer_f32,
        cache_steps=T,
        out=out1,
        disable_state_update=True,
    )

    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.randint(0, 2, (cache_size,), device=device, dtype=torch.int32)

    old_x[:] = x1
    dt1 = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1, dim=1)

    for slot in range(cache_size):
        buf = cache_buf_idx[slot].item()
        old_B[slot, buf] = B1[slot]
        old_dt[slot, buf] = dt1[slot].T
        old_dA_cumsum[slot, buf] = dA_cumsum1[slot].T

    k = T
    torch.manual_seed(123)

    x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    ref_state_f32 = state0.float().clone()
    ref_state_f32[:] = states_buffer_f32[:, k - 1]
    ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_state_f32,
        x2,
        dt2,
        A,
        B2,
        C2,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=None,
        out=ref_out,
    )

    test_state = state0.clone()
    prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
    test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)

    checkpointing_state_update(
        test_state,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_dA_cumsum.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        x=x2,
        dt=dt2,
        A=A,
        B=B2,
        C=C2,
        out=test_out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=None,
        _heads_per_block=heads_per_block,
    )

    torch.testing.assert_close(
        test_out,
        ref_out,
        rtol=2e-2,
        atol=1.0,
        msg=f"Output mismatch with HPB={heads_per_block}",
    )

    expected_state = states_buffer_f32[:, k - 1].to(state_dtype)
    torch.testing.assert_close(
        test_state,
        expected_state,
        rtol=2e-2,
        atol=1.0,
        msg=f"State mismatch with HPB={heads_per_block}",
    )


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
@pytest.mark.parametrize("heads_per_block", [2, 4], ids=["HPB2", "HPB4"])
@pytest.mark.parametrize("paged_cache", [False, True], ids=["contig", "paged"])
def test_checkpointing_heads_per_block_multistep(
    nheads,
    head_dim,
    d_state,
    ngroups,
    state_dtype,
    T,
    heads_per_block,
    paged_cache,
):
    """Multi-step HPB > 1 test — bugs accumulate across steps and surface here."""
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    n_steps = 8

    if nheads % heads_per_block != 0:
        pytest.skip(f"nheads ({nheads}) not divisible by HPB ({heads_per_block})")
    if heads_per_block > nheads // ngroups:
        pytest.skip(
            f"HPB ({heads_per_block}) exceeds heads_per_group ({nheads // ngroups})"
        )

    torch.manual_seed(42)

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    if paged_cache:
        cache_size = 4
        state_batch_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
        slots = state_batch_indices
    else:
        cache_size = batch
        state_batch_indices = None
        slots = slice(None)

    all_x = []
    all_dt = []
    all_B = []
    all_C = []
    for step in range(n_steps):
        torch.manual_seed(1000 + step)
        all_x.append(
            torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        )
        dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        all_dt.append(repeat(dt_base, "b t h -> b t h p", p=head_dim))
        all_B.append(
            torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        )
        all_C.append(
            torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        )

    torch.manual_seed(999)
    state_init = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )

    ref_state = state_init.float().clone()
    ref_outs = []
    ref_slots = (
        state_batch_indices
        if paged_cache
        else torch.arange(batch, device=device, dtype=torch.int32)
    )
    for step in range(n_steps):
        out_step = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        selective_state_update(
            ref_state,
            all_x[step],
            all_dt[step],
            A,
            all_B[step],
            all_C[step],
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=ref_slots,
            out=out_step,
        )
        ref_outs.append(out_step)

    test_state = state_init.clone()
    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.zeros(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.zeros(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_dA_cumsum = torch.zeros(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    for step in range(n_steps):
        k = T if step > 0 else 0
        prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)

        checkpointing_state_update(
            test_state,
            old_x,
            old_B,
            old_dt,
            old_dA_cumsum,
            cache_buf_idx,
            prev_tokens,
            x=all_x[step],
            dt=all_dt[step],
            A=A,
            B=all_B[step],
            C=all_C[step],
            out=test_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices,
            _heads_per_block=heads_per_block,
        )

        if paged_cache:
            cache_buf_idx[slots] = 1 - cache_buf_idx[slots]
        else:
            cache_buf_idx[:] = 1 - cache_buf_idx

        torch.testing.assert_close(
            test_out,
            ref_outs[step],
            rtol=2e-2,
            atol=2.0,
            msg=f"Output mismatch at step {step}",
        )


# ----- SR grid-bracket tests (fp8 and fp16) -----
# Verify each PTX SR output lands on the destination grid as a bracket
# neighbour of the fp32 input.  Catches byte-order traps in the inline-asm
# source-register specifier.  Pure inline-asm test — no CUDA kernel involved.


@triton.jit
def _bracket_kernel_fp8(x_ptr, rand_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    rand = tl.load(rand_ptr + offs)
    y = tl.inline_asm_elementwise(
        asm="cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;",
        constraints="=r,r,r,r,r,r,r,r,r",
        args=(x, rand),
        dtype=tl.float8e4nv,
        is_pure=True,
        pack=4,
    )
    tl.store(out_ptr + offs, y)


@triton.jit
def _bracket_kernel_fp16(x_ptr, rand_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    rand = tl.load(rand_ptr + offs)
    y = tl.inline_asm_elementwise(
        asm="""{
        cvt.rs.f16x2.f32 $0, $2, $1, $3;
        }""",
        constraints=("=r,r,r,r,r"),
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )
    tl.store(out_ptr + offs, y)


_BRACKET_KERNEL = {
    torch.float8_e4m3fn: _bracket_kernel_fp8,
    torch.float16: _bracket_kernel_fp16,
}


def _build_finite_grid(dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype == torch.float8_e4m3fn:
        ints = torch.arange(256, dtype=torch.uint8, device=device)
        full = ints.view(torch.float8_e4m3fn).to(torch.float32)
    elif dtype == torch.float16:
        ints = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
        full = ints.view(torch.float16).to(torch.float32)
    else:
        raise ValueError(f"Unsupported bracket-test dtype: {dtype}")
    return full[torch.isfinite(full)].sort()[0].unique()


def _build_bracket_inputs(dtype: torch.dtype, n: int, device: str) -> torch.Tensor:
    grid = _build_finite_grid(dtype, device)
    g_min, g_max = grid[0].item(), grid[-1].item()
    x = torch.empty(n, device=device, dtype=torch.float32)
    if dtype == torch.float8_e4m3fn:
        x.uniform_(g_min * 1.5, g_max * 1.5)
    else:
        x[: n // 4].uniform_(-1.0, 1.0)
        x[n // 4 : n // 2].uniform_(-100, 100)
        x[n // 2 : 3 * n // 4].uniform_(-1000, 1000)
        x[3 * n // 4 :].uniform_(g_min * 0.99, g_max * 0.99)
    return x, grid


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=False)
@_skip_pre_sm100
@pytest.mark.parametrize(
    "state_dtype",
    [torch.float8_e4m3fn, torch.float16],
    ids=["fp8", "fp16"],
)
def test_sr_grid_bracket(state_dtype):
    """Verify SR PTX outputs each lie on the destination grid as a bracket
    neighbour of the fp32 input."""
    device = "cuda"
    n = 1024  # multiple of both pack=4 (fp8) and pack=2 (fp16)

    torch.manual_seed(42)
    x, grid_finite = _build_bracket_inputs(state_dtype, n, device)
    g_min, g_max = grid_finite[0].item(), grid_finite[-1].item()

    x_clamped = x.clamp(g_min, g_max)
    idx = torch.searchsorted(grid_finite, x_clamped, right=False).clamp(
        min=1, max=len(grid_finite) - 1
    )
    lo = grid_finite[idx - 1]
    hi = grid_finite[idx]

    kernel = _BRACKET_KERNEL[state_dtype]

    for seed in range(4):
        torch.manual_seed(seed)
        rand = torch.randint(-(2**31), 2**31, (n,), device=device, dtype=torch.int32)
        out = torch.empty(n, device=device, dtype=state_dtype)
        kernel[(1,)](x, rand, out, BLOCK=n)
        out_fp32 = out.to(torch.float32)

        on_grid = (out_fp32 == lo) | (out_fp32 == hi)
        assert on_grid.all(), (
            f"{state_dtype} SR output not on grid bracket (seed={seed})"
        )
