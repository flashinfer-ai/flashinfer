# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the CUDA checkpointing_ssu kernel.
Validates against the Triton checkpointing_state_update reference.
"""

import pytest
import torch
import torch.nn.functional as F
import triton
from einops import repeat

from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
from flashinfer.utils import is_cvt_rs_supported

# Import Triton reference.  `replay_selective_state_update` is a backwards-compat
# alias for `checkpointing_state_update` kept inside the same module.
from .triton_reference.checkpointing_state_update import (
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


def _run_checkpointing_ssu_case(
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
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
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
        old_dt[slot, buf] = dt1_proc[i].permute(1, 0)  # (T, nheads) → (nheads, T)
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
            old_dt.clone(),
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
        checkpointing_ssu(
            test_state,
            old_x.clone(),
            old_B.clone(),
            old_dt.clone(),
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


# ``test_checkpointing_ssu`` (T ∈ {2, 6, 16}, NPREDICTED == MAX_WINDOW) was
# the original always-write-era test.  After v19 it implicitly covers both
# Branch A (must_checkpoint=True, prev_k > 0) and Branch B (prev_k = 0).
# Both branches are now covered more thoroughly by:
#   - test_checkpointing_ssu_max_window_gt_npredicted (10,16) — Branch A+B
#     crossing the boundary with K_BIG=16, identical kernel template.
#   - test_checkpointing_ssu_max_window_gt_npredicted (4,8) — Branch B with
#     K_SMALL=8, same template as T={2,6} would have used.
# So the original test was retired.  The d_split=2 path is exercised once
# as a smoke test below; both d_split={1,2} compile into the same .so via
# `D_SPLIT` template specialization, so this is a runtime-dispatch check.


def test_checkpointing_ssu_d_split2():
    """v12 §59 smoke — D_SPLIT=2 path dispatches and runs.  Both
    ``d_split={1,2}`` specializations are baked into the same JIT .so via
    the public dispatcher's switch; this test verifies the d_split=2 grid
    + smem footprint + partition_C indexing work end-to-end.  Wider dtype/T
    coverage is provided by the d_split=1 tests sharing the same .so."""
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    _run_checkpointing_ssu_case(
        nheads,
        head_dim,
        d_state,
        ngroups,
        torch.bfloat16,
        paged_cache=True,
        T=16,
        d_split=2,
    )


def test_checkpointing_ssu_heads_per_group():
    """Smoke test the multi-group code path (group_idx != 0 for some heads).

    The two ``_CONFIGS`` entries both have HEADS_PER_GROUP = nheads/ngroups
    = 16 (one group covers every head), so all the other CUDA-kernel tests
    only exercise the degenerate `group_idx = 0` routing.  Since
    HEADS_PER_GROUP is JIT-stamped (one .so per HPG value), a separate
    config does not share JIT cache with the rest of the suite — one case
    here catches a broken `head / HEADS_PER_GROUP` indexing without
    multiplying the .so count.

    HPG=8 chosen because it splits the 16 heads into 2 groups, exercising
    both `group_idx = 0` and `group_idx = 1` paths in the same launch."""
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 2  # HPG = 8
    _run_checkpointing_ssu_case(
        nheads,
        head_dim,
        d_state,
        ngroups,
        torch.bfloat16,
        paged_cache=True,
        T=16,
    )


def test_checkpointing_ssu_pdl_bf16():
    """PDL smoke: bf16 state.  Runs the kernel twice on the same inputs —
    once with enable_pdl=False, once with True — and asserts the outputs and
    states match bit-pattern-equivalent (PDL only changes launch attribute
    + load order; gdc_wait is a no-op without a paired upstream PDL kernel).
    Validates that the JIT-stamped ENABLE_PDL specialization compiles and
    produces identical results.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch, T = 2, 6
    cache_size = batch
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=dtype
    )

    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)
    prev_tokens = torch.full((cache_size,), T // 2, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    common_kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    state_off = state0.clone()
    out_off = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_off,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens.clone(),
        out=out_off,
        enable_pdl=False,
        **common_kwargs,
    )

    state_on = state0.clone()
    out_on = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_on,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens.clone(),
        out=out_on,
        enable_pdl=True,
        **common_kwargs,
    )

    torch.testing.assert_close(
        out_on,
        out_off,
        rtol=0,
        atol=0,
        msg="enable_pdl=True output differs from enable_pdl=False",
    )
    torch.testing.assert_close(
        state_on,
        state_off,
        rtol=0,
        atol=0,
        msg="enable_pdl=True state differs from enable_pdl=False",
    )


@pytest.mark.skipif(
    not is_cvt_rs_supported(),
    reason="fp8 + Philox SR requires HW cvt.rs (sm_100a+)",
)
def test_checkpointing_ssu_pdl_fp8_philox5():
    """PDL smoke: fp8 e4m3 state + Philox-5 SR.  Same as the bf16 PDL test
    but exercises the int8/fp8 kernel binary (`checkpointing_ssu_kernel_8bit`)
    and the SR encode path.  Uses a fixed Philox seed so PDL=on vs PDL=off
    produce bit-identical state writebacks.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch, T = 2, 6
    cache_size = batch
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float8_e4m3fn
    quant_max = 448.0

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    state0, state0_scales = _quantize_state(state0_fp32, state_dtype, quant_max)

    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)
    prev_tokens = torch.full((cache_size,), T // 2, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    common_kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        rand_seed=rand_seed,
        philox_rounds=5,
    )

    state_off = state0.clone()
    scale_off = state0_scales.clone()
    out_off = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_off,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens.clone(),
        out=out_off,
        state_scale=scale_off,
        enable_pdl=False,
        **common_kwargs,
    )

    state_on = state0.clone()
    scale_on = state0_scales.clone()
    out_on = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_on,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens.clone(),
        out=out_on,
        state_scale=scale_on,
        enable_pdl=True,
        **common_kwargs,
    )

    torch.testing.assert_close(
        out_on,
        out_off,
        rtol=0,
        atol=0,
        msg="fp8+philox5: enable_pdl=True output differs from off",
    )
    # int8 reinterpret for bit-identical compare of fp8 quantized state.
    torch.testing.assert_close(
        state_on.view(torch.int8),
        state_off.view(torch.int8),
        rtol=0,
        atol=0,
        msg="fp8+philox5: enable_pdl=True state differs from off",
    )
    torch.testing.assert_close(
        scale_on,
        scale_off,
        rtol=0,
        atol=0,
        msg="fp8+philox5: enable_pdl=True state_scale differs from off",
    )


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
# Pairs chosen so prev_k can reach the must_checkpoint boundary while staying
# within Triton's supported range (prev_k <= NPREDICTED — the Triton main
# kernel masks cache reads at offs_t < T, so prev_k > T is unsupported).
# (4, 8) exercises K_SMALL (m16n8k8 replay atom); (10, 16) exercises K_BIG
# (m16n8k16) and crosses the must_checkpoint boundary within the prev_k
# sweep — covers both Branch A and Branch B with K_BIG.  (12,16)/(14,16)
# were dropped (same K-atom + same kernel template params as (10,16)).
@pytest.mark.parametrize(
    "npredicted,max_window",
    [(4, 8), (10, 16)],
    ids=["np4w8", "np10w16"],
)
def test_checkpointing_ssu_max_window_gt_npredicted(
    state_dtype, paged_cache, npredicted, max_window
):
    # Only one _CONFIGS entry — both share (head_dim, d_state, HPG=16) so
    # the JIT key is identical; the other config doesn't add coverage.
    nheads, head_dim, d_state, ngroups = _CONFIGS[0]
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
    # of the active buffer.  old_dt / old_cumAdt MUST be consistent
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
        checkpointing_ssu(
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


# Philox SR + must_checkpoint=False: verify the kernel correctly skips the
# state HBM write under the Philox path.  The cvt_rs+philox-refresh runs
# internally at the lane-pair level (skipping it would require routing
# must_checkpoint into pair_idx amortization), but the STG.64 must be
# elided.  Strongest signal: state remains byte-identical to state0.
@pytest.mark.skipif(
    not is_cvt_rs_supported(),
    reason="Philox stochastic rounding requires cvt.rs PTX (SM100a/SM110a only — "
    "not SM120a / consumer Blackwell)",
)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
# (4, 8) gives must_checkpoint = (prev_k + 4 > 8) = False for prev_k ∈ [0, 4].
@pytest.mark.parametrize("npredicted,max_window", [(4, 8)], ids=["np4w8"])
def test_checkpointing_ssu_philox_no_checkpoint(
    nheads, head_dim, d_state, ngroups, paged_cache, npredicted, max_window
):
    """Verify Philox SR path skips state HBM write when must_checkpoint=False.

    With NPREDICTED=4, MAX_WINDOW=8 and prev_k in [0, 4], the kernel-derived
    must_checkpoint = (prev_k + 4 > 8) is always False.  Under that condition
    the kernel must leave state HBM byte-identical to the prior checkpoint
    (state0) regardless of the Philox path running internally.  Output and
    cache writes should still match the Triton reference run with
    write_checkpoint=False.
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

    # Step-1 inputs populate the active buffer's [0, npredicted) with
    # consistent (real softplus/cumsum) data — same precision strategy as
    # test_checkpointing_ssu_max_window_gt_npredicted.  Slots
    # [npredicted, max_window) keep their `randn` init (never read by replay
    # since prev_k <= npredicted).
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

    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    slots_idx = state_batch_indices if paged_cache else slice(None)
    old_x[slots_idx, :npredicted] = x1
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        old_B[slot, buf, :npredicted] = B1[i]
        old_dt[slot, buf, :, :npredicted] = dt1_proc[i].T
        old_dA_cumsum[slot, buf, :, :npredicted] = dA_cumsum1[i].T

    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)

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

        # Test invariant: kernel must derive must_checkpoint=False for every
        # iteration.  If anyone tweaks the (np, mw) params and breaks this,
        # the assert fails loud rather than silently passing under True.
        must_checkpoint = (prev_k + npredicted) > max_window
        assert not must_checkpoint, (
            f"test invariant violated: must_checkpoint should be False for "
            f"(prev_k={prev_k}, npredicted={npredicted}, max_window={max_window}), "
            f"got True"
        )

        # ── Triton reference (write_checkpoint=False to mirror kernel) ──
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
            write_checkpoint=False,
        )

        # ── CUDA kernel with Philox enabled ──
        test_state = state0.clone()
        test_prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
        test_out = torch.zeros(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        old_x_test = old_x.clone()
        old_B_test = old_B.clone()
        old_dt_test = old_dt.clone()
        old_dA_test = old_dA_cumsum.clone()
        checkpointing_ssu(
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
            rand_seed=rand_seed,
            philox_rounds=10,
        )

        # KEY assertion: state HBM byte-identical to state0.  Kernel must
        # not have stored to it under must_checkpoint=False, even with the
        # Philox SR cvt_rs path running internally.
        torch.testing.assert_close(
            test_state[slots_idx],
            state0[slots_idx],
            rtol=0,
            atol=0,
            msg=f"state should be byte-identical to state0 at prev_k={prev_k} "
            f"(must_checkpoint=False) — kernel wrote state HBM despite the gate",
        )

        # Output should match the Triton ref with write_checkpoint=False.
        # bf16 + accumulated SR ULPs through the output phase → loose tol,
        # same as the non-philox max_window>npredicted test.
        torch.testing.assert_close(
            test_out,
            ref_out,
            rtol=2e-2,
            atol=5e-1,
            msg=f"out mismatch at prev_k={prev_k}",
        )

        # Cache writes happen unconditionally; only target buffer/offset
        # depends on must_checkpoint.  Should match Triton ref bit-exactly
        # for old_x/old_B and to fp32 tolerance for old_dt/old_dA_cumsum.
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


# Philox SR + must_checkpoint=True under MAX_WINDOW > NPREDICTED: rounds out
# the fp16+Philox validation matrix.  The existing test_checkpointing_ssu_philox
# only exercises the degenerate MAX_WINDOW == NPREDICTED case where every
# call checkpoints; this test verifies the non-degenerate case where the
# checkpoint decision actually depends on prev_k vs the buffer's remaining
# capacity.
@pytest.mark.skipif(
    not is_cvt_rs_supported(),
    reason="Philox stochastic rounding requires cvt.rs PTX (SM100a/SM110a only — "
    "not SM120a / consumer Blackwell)",
)
# 4-tuple subset of the 16-element cartesian (was: 4 (np,mw,pk) × 2 paged ×
# 2 configs, fp16 only).  Keep K_BIG path with shallow + deep replay,
# rotate paged + config so each pair gets one of each.
@pytest.mark.parametrize(
    "npredicted,max_window,prev_k,paged_cache,_cfg_idx",
    [
        # Shallow replay (just past boundary) — paged, config 0
        (10, 16, 7, True, 0),
        # Max replay for np=10 — no-paged, config 1
        (10, 16, 10, False, 1),
        # Deeper template with deeper replay — paged, config 1
        (14, 16, 14, True, 1),
        # Just-over-boundary on deeper template — no-paged, config 0
        (14, 16, 3, False, 0),
    ],
    ids=["np10w16_pk7", "np10w16_pk10", "np14w16_pk14", "np14w16_pk3"],
)
def test_checkpointing_ssu_philox_with_checkpoint(
    npredicted,
    max_window,
    prev_k,
    paged_cache,
    _cfg_idx,
):
    """Verify Philox SR path correctly writes state HBM when must_checkpoint=True
    in the non-degenerate MAX_WINDOW > NPREDICTED regime.

    Asserts:
      1. Output and state HBM match Triton ref with write_checkpoint=True.
      2. State actually differs from state0 in some elements (kernel really
         wrote — guards against the gate being silently inverted).
      3. Cache writes match Triton ref (always written, just at the
         must_checkpoint-dependent buffer/offset).
    """
    nheads, head_dim, d_state, ngroups = _CONFIGS[_cfg_idx]
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float16
    assert nheads % ngroups == 0
    must_checkpoint = (prev_k + npredicted) > max_window
    assert must_checkpoint, (
        f"test invariant violated: must_checkpoint should be True for "
        f"(prev_k={prev_k}, npredicted={npredicted}, max_window={max_window}), "
        f"got False"
    )
    assert prev_k <= npredicted, (
        f"Triton range constraint: prev_k must be <= npredicted, got "
        f"prev_k={prev_k}, npredicted={npredicted}"
    )

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

    # Step-1 inputs populate the active buffer's [0, npredicted) with
    # consistent (real softplus/cumsum) data — same precision strategy as
    # test_checkpointing_ssu_max_window_gt_npredicted.
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

    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    slots_idx = state_batch_indices if paged_cache else slice(None)
    old_x[slots_idx, :npredicted] = x1
    for i, slot in enumerate(slot_indices):
        buf = cache_buf_idx[slot].item()
        old_B[slot, buf, :npredicted] = B1[i]
        old_dt[slot, buf, :, :npredicted] = dt1_proc[i].T
        old_dA_cumsum[slot, buf, :, :npredicted] = dA_cumsum1[i].T

    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    torch.manual_seed(prev_k + 200)

    x2 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)

    # ── Triton reference (write_checkpoint=True to mirror kernel) ──
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
        write_checkpoint=True,
    )

    # ── CUDA kernel with Philox enabled ──
    test_state = state0.clone()
    test_prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
    test_out = torch.zeros(
        batch, npredicted, nheads, head_dim, device=device, dtype=dtype
    )
    old_x_test = old_x.clone()
    old_B_test = old_B.clone()
    old_dt_test = old_dt.clone()
    old_dA_test = old_dA_cumsum.clone()
    checkpointing_ssu(
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
        rand_seed=rand_seed,
        philox_rounds=10,
    )

    # Sanity guard: the kernel must have written state HBM (gate not inverted).
    # Use any-not-equal across the touched slots — if the gate were silently
    # backwards, state would equal state0 byte-for-byte.
    assert not torch.equal(test_state[slots_idx], state0[slots_idx]), (
        "state HBM unchanged from state0 despite must_checkpoint=True — "
        "checkpoint gate likely inverted"
    )

    # State match Triton ref.  bf16 + replay + Philox SR ULP noise → loose
    # tol matching the non-philox max_window>npredicted test.
    torch.testing.assert_close(
        test_state[slots_idx],
        ref_state[slots_idx],
        rtol=2e-2,
        atol=5e-1,
        msg=f"state mismatch (prev_k={prev_k}, np={npredicted}, mw={max_window})",
    )
    torch.testing.assert_close(
        test_out,
        ref_out,
        rtol=2e-2,
        atol=5e-1,
        msg=f"out mismatch (prev_k={prev_k}, np={npredicted}, mw={max_window})",
    )

    # Cache writes happen unconditionally; only target buffer/offset depends
    # on must_checkpoint.
    torch.testing.assert_close(
        old_x_test, old_x_ref, rtol=0, atol=0, msg=f"old_x mismatch (prev_k={prev_k})"
    )
    torch.testing.assert_close(
        old_B_test, old_B_ref, rtol=0, atol=0, msg=f"old_B mismatch (prev_k={prev_k})"
    )
    torch.testing.assert_close(
        old_dt_test,
        old_dt_ref,
        rtol=1e-4,
        atol=1e-4,
        msg=f"old_dt mismatch (prev_k={prev_k})",
    )
    torch.testing.assert_close(
        old_dA_test,
        old_dA_ref,
        rtol=1e-4,
        atol=1e-4,
        msg=f"old_dA mismatch (prev_k={prev_k})",
    )


def _quantize_state_int8(state_fp32: torch.Tensor):
    """Quantize fp32 state (cache, nheads, dim, dstate) to int8 + decode_scale.

    Per-(cache, head, dim) channel: `amax = max(|state|, axis=dstate)`,
    `decode_scale = amax / 127`, `state_q = round(state / decode_scale)` clamped
    to [-127, 127].  Matches Triton's _quantize_state with QUANT_MAX=127.
    """
    QUANT_MAX = 127.0
    amax = state_fp32.abs().amax(dim=-1)  # (cache, nheads, dim)
    encode_scale = QUANT_MAX / amax.clamp(min=1e-30)
    decode_scale = 1.0 / encode_scale
    scaled = state_fp32 * encode_scale.unsqueeze(-1)
    state_q = scaled.round().clamp(-QUANT_MAX, QUANT_MAX).to(torch.int8)
    return state_q, decode_scale


def _dequantize_state_int8(state_q: torch.Tensor, decode_scale: torch.Tensor):
    return state_q.to(torch.float32) * decode_scale.unsqueeze(-1)


# 3b.1 — int8 RN parity vs Triton.  The CUDA kernel runs through
# replay_state_mma_int8 which dequants on load, runs the replay matmul in
# fp32, and encodes back to int8 with RN.  Triton does the same when called
# without rand_seed.  Tolerance is loose because int8 grid is coarse and
# bf16 matmul ULPs add additional noise (matches Triton's own RN test
# tolerance: ~1.6 atol on output, ~1.1 atol on dequantized state).
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
def test_checkpointing_ssu_int8_rn_parity(
    nheads, head_dim, d_state, ngroups, paged_cache, T
):
    """int8 + RN parity vs fp32 selective_state_update reference.

    Builds physically realistic cache tensors (old_dt, old_cumAdt
    derived from A_base and softplus(dt+bias)), then validates the CUDA
    kernel's output and dequantized state against the fp32 recurrence.
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
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # int8 state init: derive from fp32 source so dequant gives sensible values.
    state0_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    state0, state0_scale = _quantize_state_int8(state0_fp32)
    ref_input_state = _dequantize_state_int8(state0, state0_scale)

    # ── Step 1 inputs (old tokens to be replayed) ──
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # Capture intermediate fp32 states via selective_state_update.
    slots = state_batch_indices if paged_cache else slice(None)
    cache_idx_for_capture = (
        state_batch_indices
        if paged_cache
        else torch.arange(batch, device=device, dtype=torch.int32)
    )
    states_buffer_f32 = torch.zeros(
        cache_size, T, nheads, head_dim, d_state, device=device, dtype=torch.float32
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

    # Build physically realistic cache tensors.
    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.zeros(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.zeros(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    old_x[slots, :T] = x1
    dt1_processed = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_processed, dim=1)

    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot_val in enumerate(slot_indices):
        buf = cache_buf_idx[slot_val].item()
        old_B[slot_val, buf, :T] = B1[i]
        old_dt[slot_val, buf, :, :T] = dt1_processed[i].T
        old_cumAdt[slot_val, buf, :, :T] = dA_cumsum1[i].T

    # prev_k = T//2 with NPREDICTED == MAX_WINDOW == T → must_checkpoint=True
    k = T // 2
    prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)

    # ── Step 2 inputs (new tokens) ──
    torch.manual_seed(100)
    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

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

    # ── fp32 reference: selective_state_update from state after k old tokens ──
    ref_state_f32 = ref_input_state.clone()
    if k > 0:
        ref_state_f32[slots] = states_buffer_f32[slots, k - 1]
    ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_state_f32,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=(state_batch_indices if paged_cache else None),
        out=ref_out,
    )

    # ── CUDA kernel ──
    test_state = state0.clone()
    test_scale = state0_scale.clone()
    test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        test_state,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens.clone(),
        out=test_out,
        state_scale=test_scale,
        **common_kwargs,
    )

    # Compare dequantized state against fp32 reference state after replay.
    expected_state_fp32 = (
        ref_input_state[slots] if k == 0 else states_buffer_f32[slots, k - 1]
    )
    actual_fp32 = _dequantize_state_int8(test_state[slots], test_scale[slots])
    torch.testing.assert_close(
        actual_fp32,
        expected_state_fp32,
        rtol=5e-2,
        atol=1.1,
        msg="int8 RN state mismatch vs fp32 reference (dequantized)",
    )

    # Output parity vs fp32 reference.
    torch.testing.assert_close(
        test_out,
        ref_out,
        rtol=2e-2,
        atol=1.6,
        msg="int8 RN output mismatch vs fp32 reference",
    )

    # Sanity: state_scale must be all positive finite, state_t int8.
    assert test_state.dtype == torch.int8
    assert torch.isfinite(test_scale).all()
    assert (test_scale > 0).all()


# 3b.2 — fp8 e4m3fn RN parity vs fp32 reference.  Mirrors the int8 path
# (per-(cache, head, dim) decode-scale, M-shard replay, RN encode) — the
# CUDA kernel routes both through checkpointing_ssu_kernel_8bit, only the
# encode primitive and QUANT_MAX differ.  Tolerances are looser than int8
# because the fp8 grid is non-uniform (cell size grows by 2x per binade).
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8
    or (
        torch.cuda.get_device_capability()[0] == 8
        and torch.cuda.get_device_capability()[1] < 9
    ),
    reason="fp8_e4m3fn requires SM 89+ (Ada Lovelace / Hopper / Blackwell)",
)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
def test_checkpointing_ssu_fp8_rn_parity(
    nheads, head_dim, d_state, ngroups, paged_cache, T
):
    """fp8 e4m3fn + RN parity vs fp32 selective_state_update reference.

    Mirrors test_checkpointing_ssu_int8_rn_parity; substitutes the int8
    quantize/dequantize helpers with the generic block-scaling ones at
    QUANT_MAX = 448.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float8_e4m3fn
    quant_max = 448.0
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

    # fp8 state init: derive from fp32 source so dequant gives sensible values.
    state0_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    state0, state0_scale = _quantize_state(state0_fp32, state_dtype, quant_max)
    ref_input_state = _dequantize_state(state0, state0_scale)

    # ── Step 1 inputs (old tokens to be replayed) ──
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt1 = repeat(dt1_base, "b t h -> b t h p", p=head_dim)
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # Capture intermediate fp32 states via selective_state_update.
    slots = state_batch_indices if paged_cache else slice(None)
    cache_idx_for_capture = (
        state_batch_indices
        if paged_cache
        else torch.arange(batch, device=device, dtype=torch.int32)
    )
    states_buffer_f32 = torch.zeros(
        cache_size, T, nheads, head_dim, d_state, device=device, dtype=torch.float32
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

    # Build physically realistic cache tensors.
    old_x = torch.zeros(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.zeros(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.zeros(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    old_x[slots, :T] = x1
    dt1_processed = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_processed, dim=1)

    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot_val in enumerate(slot_indices):
        buf = cache_buf_idx[slot_val].item()
        old_B[slot_val, buf, :T] = B1[i]
        old_dt[slot_val, buf, :, :T] = dt1_processed[i].T
        old_cumAdt[slot_val, buf, :, :T] = dA_cumsum1[i].T

    # prev_k = T//2 with NPREDICTED == MAX_WINDOW == T → must_checkpoint=True
    k = T // 2
    prev_tokens = torch.full((cache_size,), k, device=device, dtype=torch.int32)

    # ── Step 2 inputs (new tokens) ──
    torch.manual_seed(100)
    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

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

    # ── fp32 reference: selective_state_update from state after k old tokens ──
    ref_state_f32 = ref_input_state.clone()
    if k > 0:
        ref_state_f32[slots] = states_buffer_f32[slots, k - 1]
    ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    selective_state_update(
        ref_state_f32,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=(state_batch_indices if paged_cache else None),
        out=ref_out,
    )

    # ── CUDA kernel ──
    test_state = state0.clone()
    test_scale = state0_scale.clone()
    test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        test_state,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens.clone(),
        out=test_out,
        state_scale=test_scale,
        **common_kwargs,
    )

    # Compare dequantized state against fp32 reference state after replay.
    expected_state_fp32 = (
        ref_input_state[slots] if k == 0 else states_buffer_f32[slots, k - 1]
    )
    actual_fp32 = _dequantize_state(test_state[slots], test_scale[slots])
    torch.testing.assert_close(
        actual_fp32,
        expected_state_fp32,
        rtol=1e-1,
        atol=2.5,
        msg="fp8 RN state mismatch vs fp32 reference (dequantized)",
    )

    # Output parity vs fp32 reference.
    torch.testing.assert_close(
        test_out,
        ref_out,
        rtol=5e-2,
        atol=4.0,
        msg="fp8 RN output mismatch vs fp32 reference",
    )

    # Sanity: state_scale must be all positive finite, state_t fp8 e4m3fn.
    assert test_state.dtype == torch.float8_e4m3fn
    assert torch.isfinite(test_scale).all()
    assert (test_scale > 0).all()


def test_checkpointing_ssu_int8_smoke():
    """Int8 state path — compile + run smoke test.

    Verifies the JIT compiles for int8 state with state_scale and the
    kernel runs end-to-end without crashing.  Functional parity vs Triton
    is tested separately once the int8 numerics are validated.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch, T = 2, 6
    cache_size = batch
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # int8 state: random in [-127, 127].  state_scale: random positive fp32.
    state = torch.randint(
        -127,
        128,
        (cache_size, nheads, head_dim, d_state),
        device=device,
        dtype=torch.int8,
    )
    state_scale = (
        torch.rand(
            cache_size,
            nheads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
        + 0.01
    )  # positive fp32 in (0.01, 0.11)

    # Cache tensors.
    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    # Inputs.
    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    # prev_k = T//2 with NPREDICTED == MAX_WINDOW == T → must_checkpoint=True
    prev_tokens = torch.full((cache_size,), T // 2, device=device, dtype=torch.int32)

    state_before = state.clone()
    state_scale_before = state_scale.clone()
    out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)

    checkpointing_ssu(
        state,
        old_x,
        old_B,
        old_dt,
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
        state_scale=state_scale,
    )

    # State HBM must have been overwritten (must_checkpoint=True).
    assert not torch.equal(state, state_before), (
        "int8 state HBM unchanged after must_checkpoint=True call — kernel "
        "did not store the post-replay state"
    )
    # state_scale must have been updated.
    assert not torch.equal(state_scale, state_scale_before), (
        "state_scale gmem unchanged after must_checkpoint=True call — "
        "kernel did not write decode_scale"
    )
    # Output should be finite (no NaN/Inf).
    assert torch.isfinite(out).all(), "output contains NaN/Inf"
    # state_scale should be all positive finite.
    assert torch.isfinite(state_scale).all(), "state_scale contains NaN/Inf"
    assert (state_scale > 0).all(), "state_scale must be all positive"


# Mixed-batch: some batches need a checkpoint, others don't, in a single call.
# This is the realistic production scenario (some sequences had a rejection
# this step, others didn't).  The CUDA kernel handles it natively in one
# launch via per-CTA must_checkpoint derived from per-batch prev_k.  Triton
# can't — its WRITE_CHECKPOINT is tl.constexpr, set once per launch — so the
# reference is built from two sub-launches on disjoint sub-batches.
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize("with_philox", [False, True], ids=["no_philox", "philox"])
def test_checkpointing_ssu_mixed_checkpoint_batch(
    nheads, head_dim, d_state, ngroups, paged_cache, with_philox
):
    """Verify CUDA kernel handles a batch with mixed must_checkpoint values.

    batch=4 with prev_k = [0, 6, 7, 10] under (npredicted=10, max_window=16):
      must_checkpoint = (prev_k + 10 > 16) = [False, False, True, True]
    The CUDA kernel runs a single launch; per-CTA must_checkpoint is derived
    from prev_num_accepted_tokens[cache_slot].  The Triton reference is
    assembled from two launches (one per group) with disjoint sub-batches.
    """
    if with_philox and not is_cvt_rs_supported():
        pytest.skip(
            "Philox stochastic rounding requires cvt.rs PTX (SM100a/SM110a only — "
            "not SM120a / consumer Blackwell)"
        )

    batch = 4
    npredicted = 10
    max_window = 16
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float16
    assert nheads % ngroups == 0

    # Per-batch prev_k.  Contiguous False/True groups give clean slicing
    # for the two-launch Triton reference.
    prev_k_per_batch = [0, 6, 7, 10]
    must_checkpoint_per_batch = [
        (pk + npredicted) > max_window for pk in prev_k_per_batch
    ]
    assert must_checkpoint_per_batch == [False, False, True, True], (
        "test invariant violated: prev_k choice must yield [F, F, T, T]"
    )
    false_idx = [i for i, mc in enumerate(must_checkpoint_per_batch) if not mc]
    true_idx = [i for i, mc in enumerate(must_checkpoint_per_batch) if mc]
    # Slicing assumes contiguous groups (so x[false_idx_slice] is a contiguous
    # view, not a fancy-index gather).
    assert false_idx == [0, 1] and true_idx == [2, 3]

    cache_size = 4
    if paged_cache:
        # Non-trivial batch→slot mapping to verify the kernel uses
        # state_batch_indices everywhere (not implicit batch index).
        state_batch_indices_full = torch.tensor(
            [1, 3, 0, 2], device=device, dtype=torch.int32
        )
    else:
        # Identity mapping; explicit form so sub-launches can slice.
        state_batch_indices_full = torch.arange(batch, device=device, dtype=torch.int32)
    slot_per_batch = state_batch_indices_full.tolist()

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

    # Step-1 inputs populate each used slot's active-buffer [0, npredicted).
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

    for i, slot in enumerate(slot_per_batch):
        old_x[slot, :npredicted] = x1[i]
        buf = cache_buf_idx[slot].item()
        old_B[slot, buf, :npredicted] = B1[i]
        old_dt[slot, buf, :, :npredicted] = dt1_proc[i].T
        old_dA_cumsum[slot, buf, :, :npredicted] = dA_cumsum1[i].T

    torch.manual_seed(7)
    x2 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)

    # Per-cache-slot prev_num_accepted_tokens.  prev_k_per_batch[i] applies
    # to whichever cache slot batch i maps to.
    prev_per_slot = torch.zeros(cache_size, device=device, dtype=torch.int32)
    for i, slot in enumerate(slot_per_batch):
        prev_per_slot[slot] = prev_k_per_batch[i]

    rand_seed = (
        torch.tensor([12345], device=device, dtype=torch.int64) if with_philox else None
    )
    philox_kwargs = (
        dict(rand_seed=rand_seed, philox_rounds=10) if with_philox else dict()
    )

    # ── Triton reference: two launches on disjoint sub-batches ──
    #
    # Each sub-launch passes the full state/cache tensors and the full
    # prev_per_slot vector — Triton only touches the slots referenced by
    # its sliced state_batch_indices, so the two launches' writes don't
    # alias.  write_checkpoint is the constexpr that differs per sub-launch.
    ref_state = state0.clone()
    old_x_ref = old_x.clone()
    old_B_ref = old_B.clone()
    old_dt_ref = old_dt.clone()
    old_dA_ref = old_dA_cumsum.clone()
    ref_out = torch.zeros(
        batch, npredicted, nheads, head_dim, device=device, dtype=dtype
    )

    for sub_idx, sub_write in [(false_idx, False), (true_idx, True)]:
        sub_slice = slice(sub_idx[0], sub_idx[-1] + 1)  # contiguous range
        replay_selective_state_update(
            ref_state,
            old_x_ref,
            old_B_ref,
            old_dt_ref,
            old_dA_ref,
            cache_buf_idx.clone(),
            prev_per_slot.clone(),
            x=x2[sub_slice],
            dt=dt2[sub_slice],
            A=A,
            B=B2[sub_slice],
            C=C2[sub_slice],
            out=ref_out[sub_slice],
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_batch_indices=state_batch_indices_full[sub_slice],
            write_checkpoint=sub_write,
        )

    # ── CUDA kernel: single launch with full per-batch prev_k ──
    test_state = state0.clone()
    old_x_test = old_x.clone()
    old_B_test = old_B.clone()
    old_dt_test = old_dt.clone()
    old_dA_test = old_dA_cumsum.clone()
    test_out = torch.zeros(
        batch, npredicted, nheads, head_dim, device=device, dtype=dtype
    )
    checkpointing_ssu(
        test_state,
        old_x_test,
        old_B_test,
        old_dt_test,
        old_dA_test,
        cache_buf_idx.clone(),
        prev_per_slot.clone(),
        x=x2,
        dt=dt2,
        A=A,
        B=B2,
        C=C2,
        out=test_out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=(state_batch_indices_full if paged_cache else None),
        **philox_kwargs,
    )

    # Per-slot state validation, split by gate value:
    #   False-group: kernel must NOT write; state byte-identical to state0
    #     AND Triton ref must match the same byte-identical state0 (sanity:
    #     confirms our two-launch ref correctly skipped these slots too).
    #   True-group:  kernel must write the post-replay state; values must
    #     match Triton ref to bf16/SR ULP tolerance.  The byte-inequality
    #     vs state0 also serves as a gate-not-inverted guard.
    for i, slot in enumerate(slot_per_batch):
        if must_checkpoint_per_batch[i]:
            assert not torch.equal(test_state[slot], state0[slot]), (
                f"slot {slot} (batch {i}, must_checkpoint=True) was not "
                "written — gate likely inverted"
            )
            torch.testing.assert_close(
                test_state[slot],
                ref_state[slot],
                rtol=2e-2,
                atol=5e-1,
                msg=f"True-group slot {slot} (batch {i}) value mismatch vs Triton",
            )
        else:
            assert torch.equal(test_state[slot], state0[slot]), (
                f"slot {slot} (batch {i}, must_checkpoint=False) was written "
                "— gate likely inverted"
            )
            assert torch.equal(ref_state[slot], state0[slot]), (
                f"Triton ref state for False-group slot {slot} (batch {i}) "
                "differs from state0 — two-launch reference is buggy"
            )

    # Output (full batch) matches Triton ref.  Loose tol absorbs bf16 dot
    # ULPs and Philox SR ULPs (when enabled).
    torch.testing.assert_close(
        test_out, ref_out, rtol=2e-2, atol=5e-1, msg="out mismatch"
    )

    # Cache writes happen unconditionally; only target buffer/offset depends
    # on per-batch must_checkpoint.
    torch.testing.assert_close(
        old_x_test, old_x_ref, rtol=0, atol=0, msg="old_x mismatch"
    )
    torch.testing.assert_close(
        old_B_test, old_B_ref, rtol=0, atol=0, msg="old_B mismatch"
    )
    torch.testing.assert_close(
        old_dt_test, old_dt_ref, rtol=1e-4, atol=1e-4, msg="old_dt mismatch"
    )
    torch.testing.assert_close(
        old_dA_test, old_dA_ref, rtol=1e-4, atol=1e-4, msg="old_dA mismatch"
    )


def test_checkpointing_ssu_contiguous():
    """Smoke test for the contiguous-cache path (TP=8, bf16 state, mtp=16)."""
    _run_checkpointing_ssu_case(
        nheads=16,
        head_dim=64,
        d_state=128,
        ngroups=1,
        state_dtype=torch.bfloat16,
        paged_cache=False,
        T=16,
    )


@pytest.mark.parametrize("T", [27, 32, 55], ids=["T27", "T32", "T55"])
def test_checkpointing_ssu_rejects_large_T(T):
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
    old_dt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
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
        checkpointing_ssu(
            state,
            old_x,
            old_B,
            old_dt,
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
def test_checkpointing_ssu_philox(nheads, head_dim, d_state, ngroups, paged_cache, T):
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
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
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
    checkpointing_ssu(
        state_nornd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
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
    checkpointing_ssu(
        state_rnd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
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
    old_dt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
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
    checkpointing_ssu(
        state_fp32,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
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
    checkpointing_ssu(
        state_rnd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
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
# Int8 + Philox stochastic rounding tests (CUDA kernel)
# ============================================================================


@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
def test_checkpointing_ssu_int8_philox(
    nheads, head_dim, d_state, ngroups, paged_cache, T
):
    """Int8 + Philox SR: run the CUDA kernel twice (RN vs SR), check outputs
    are close and state diff is bounded by 1 quantization cell per element.

    Adapted from upstream test_checkpointing_state_update_philox.
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

    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    state0, state0_scales = _quantize_state_int8(state0_fp32)

    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(
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

    # --- Run without rounding (deterministic RN) ---
    state_nornd = state0.clone()
    scale_nornd = state0_scales.clone()
    out_nornd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_nornd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_nornd,
        state_scale=scale_nornd,
        **common_kwargs,
    )

    # --- Run with Philox SR ---
    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rnd = state0.clone()
    scale_rnd = state0_scales.clone()
    out_rnd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_rnd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rnd,
        state_scale=scale_rnd,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Outputs should be nearly identical.
    torch.testing.assert_close(
        out_rnd,
        out_nornd,
        rtol=2e-2,
        atol=1.6,
        msg="int8+SR output diverged from int8+RN",
    )

    assert state_rnd.dtype == torch.int8

    # State diff between RN and SR is bounded by 1 quant cell per element.
    slots = state_batch_indices if paged_cache else slice(None)
    rnd_fp32 = _dequantize_state_int8(state_rnd[slots], scale_rnd[slots])
    nornd_fp32 = _dequantize_state_int8(state_nornd[slots], scale_nornd[slots])
    diff = (rnd_fp32 - nornd_fp32).abs()
    scale_bound = torch.maximum(scale_nornd[slots], scale_rnd[slots]).unsqueeze(-1)
    bound = scale_bound * 1.5
    assert (diff <= bound).all(), (
        f"State RN-SR diff exceeds 1 cell per element (int8). "
        f"max_diff={diff.max().item():.4g}, max_bound={bound.max().item():.4g}"
    )


def test_checkpointing_ssu_int8_philox_unbiased():
    """Verify that int8 + Philox SR is statistically unbiased.

    Runs the CUDA kernel with fp32 state (capturing true post-replay values),
    then with int8 state + Philox SR.  Compares the SR rounding residual
    against deterministic RN: SR should have mean residual closer to zero.

    Uses a large batch (16) for ~2M state elements.

    Adapted from upstream test_philox_rounding_unbiased.
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

    state0_fp32 = torch.randn(
        batch, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )

    old_x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(batch, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(batch, 2, nheads, T, device=device, dtype=torch.float32)
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

    # 1. fp32 state — captures true post-replay state.
    state_fp32 = state0_fp32.clone()
    out_fp32 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_fp32,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_fp32,
        **common_kwargs,
    )

    # 2. int8 state with Philox SR.
    rand_seed = torch.tensor([99999], device=device, dtype=torch.int64)
    state_rounded, scales_rounded = _quantize_state_int8(state0_fp32)
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_rounded,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rounded,
        state_scale=scales_rounded,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Compute residuals.
    fp32_vals = state_fp32.flatten()
    stochastic_residual = (
        _dequantize_state_int8(state_rounded, scales_rounded).flatten() - fp32_vals
    )
    det_quant, det_scales = _quantize_state_int8(state_fp32)
    deterministic_residual = (
        _dequantize_state_int8(det_quant, det_scales).flatten() - fp32_vals
    )

    nonzero_mask = deterministic_residual.abs() > 0
    num_nonzero = nonzero_mask.sum().item()
    assert num_nonzero > 1000, f"Too few roundable elements: {num_nonzero}"

    stochastic_mean = stochastic_residual[nonzero_mask].mean().item()
    stochastic_std = stochastic_residual[nonzero_mask].std().item()

    # SE-based bias check (K=4 ≈ ~3.2e-5 one-sided false-positive rate).
    se_sr = stochastic_std / (num_nonzero**0.5)
    K = 4
    assert abs(stochastic_mean) < K * se_sr, (
        f"SR mean exceeds {K}*SE (likely biased) (int8): "
        f"stochastic_mean={stochastic_mean:.3e}, SE={se_sr:.3e}, "
        f"n_elements={num_nonzero}"
    )


# Single-token-prediction (STP) sanity test at NPREDICTED=1 with int8 + Philox-5.
# Exercises the smallest possible new-token count through the same kernel path —
# the m16n8 matmul atoms still tile to padded M=16 internally, so this also acts
# as a "does the row-predicate STG actually mask 15/16 rows" check.
def test_checkpointing_ssu_int8_philox_npredicted1():
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    T = 1
    batch = 2
    cache_size = batch
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
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    state0, state0_scales = _quantize_state_int8(state0_fp32)

    # max_window = T = 1: cache holds at most 1 old token per slot.  prev_k=0
    # below (no replay) → must_checkpoint = (0 + 1 > 1) = False → no-write path.
    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(
        cache_size, 2, nheads, T, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)

    x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
    B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

    prev_tokens = torch.zeros(cache_size, device=device, dtype=torch.int32)

    common_kwargs = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=None,
    )

    # ── Run without rounding (deterministic RN) ──
    state_nornd = state0.clone()
    scale_nornd = state0_scales.clone()
    out_nornd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_nornd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_nornd,
        state_scale=scale_nornd,
        **common_kwargs,
    )

    # ── Run with Philox-5 SR ──
    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rnd = state0.clone()
    scale_rnd = state0_scales.clone()
    out_rnd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_rnd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rnd,
        state_scale=scale_rnd,
        rand_seed=rand_seed,
        philox_rounds=5,
        **common_kwargs,
    )

    # Outputs nearly identical (SR introduces only 0/1-cell perturbations on
    # the encoded state, and at prev_k=0 the no-write path doesn't even encode).
    torch.testing.assert_close(
        out_rnd,
        out_nornd,
        rtol=2e-2,
        atol=1.6,
        msg="int8+SR output diverged from int8+RN at NPREDICTED=1",
    )

    assert state_rnd.dtype == torch.int8

    # State diff between RN and SR is bounded by 1 quant cell per element.
    rnd_fp32 = _dequantize_state_int8(state_rnd, scale_rnd)
    nornd_fp32 = _dequantize_state_int8(state_nornd, scale_nornd)
    diff = (rnd_fp32 - nornd_fp32).abs()
    scale_bound = torch.maximum(scale_nornd, scale_rnd).unsqueeze(-1)
    bound = scale_bound * 1.5
    assert (diff <= bound).all(), (
        f"State RN-SR diff exceeds 1 cell per element (int8, NPREDICTED=1). "
        f"max_diff={diff.max().item():.4g}, max_bound={bound.max().item():.4g}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8
    or (
        torch.cuda.get_device_capability()[0] == 8
        and torch.cuda.get_device_capability()[1] < 9
    ),
    reason="fp8_e4m3fn requires SM 89+ (Ada Lovelace / Hopper / Blackwell)",
)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
@pytest.mark.parametrize("T", [6, 16], ids=["T6", "T16"])
def test_checkpointing_ssu_fp8_philox(
    nheads, head_dim, d_state, ngroups, paged_cache, T
):
    """fp8 e4m3fn + Philox SR: run the CUDA kernel twice (RN vs SR), check
    outputs are close and the state diff is bounded by ~1 quantization cell
    per element.  fp8's cell size varies by 2x per binade, so the bound is
    looser than int8 (cell_pad = 32x covers the largest binade).

    Adapted from test_checkpointing_ssu_int8_philox.
    """
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float8_e4m3fn
    quant_max = 448.0
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

    state0_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    state0, state0_scales = _quantize_state(state0_fp32, state_dtype, quant_max)

    old_x = torch.randn(cache_size, T, nheads, head_dim, device=device, dtype=dtype)
    old_B = torch.randn(cache_size, 2, T, ngroups, d_state, device=device, dtype=dtype)
    old_dt = torch.randn(cache_size, 2, nheads, T, device=device, dtype=torch.float32)
    old_cumAdt = torch.randn(
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

    # --- Run without rounding (deterministic RN) ---
    state_nornd = state0.clone()
    scale_nornd = state0_scales.clone()
    out_nornd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_nornd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_nornd,
        state_scale=scale_nornd,
        **common_kwargs,
    )

    # --- Run with Philox SR ---
    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rnd = state0.clone()
    scale_rnd = state0_scales.clone()
    out_rnd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_rnd,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rnd,
        state_scale=scale_rnd,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Outputs should be close (looser tolerance than int8 — fp8 grid is coarser).
    torch.testing.assert_close(
        out_rnd,
        out_nornd,
        rtol=5e-2,
        atol=6.0,
        msg="fp8+SR output diverged from fp8+RN",
    )

    assert state_rnd.dtype == torch.float8_e4m3fn

    # State diff between RN and SR is bounded by ~1 quant cell per element.
    # fp8 cells vary by 2x per binade — the largest cell within a channel is
    # ~32x the decode_scale (max binade above channel amax/448).  Use 32 * 1.5
    # slack for floating-point compare quirks at the exact-cell boundary.
    slots = state_batch_indices if paged_cache else slice(None)
    rnd_fp32 = _dequantize_state(state_rnd[slots], scale_rnd[slots])
    nornd_fp32 = _dequantize_state(state_nornd[slots], scale_nornd[slots])
    diff = (rnd_fp32 - nornd_fp32).abs()
    scale_bound = torch.maximum(scale_nornd[slots], scale_rnd[slots]).unsqueeze(-1)
    bound = scale_bound * (32.0 * 1.5)
    assert (diff <= bound).all(), (
        f"State RN-SR diff exceeds 1 cell per element (fp8). "
        f"max_diff={diff.max().item():.4g}, max_bound={bound.max().item():.4g}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8
    or (
        torch.cuda.get_device_capability()[0] == 8
        and torch.cuda.get_device_capability()[1] < 9
    ),
    reason="fp8_e4m3fn requires SM 89+ (Ada Lovelace / Hopper / Blackwell)",
)
def test_checkpointing_ssu_fp8_philox_unbiased():
    """Verify that fp8 + Philox SR is statistically unbiased.

    Runs the CUDA kernel with fp32 state (captures true post-replay values),
    then with fp8 state + Philox SR.  Compares the SR rounding residual
    against the deterministic RN residual: SR should have mean residual
    closer to zero (SE-based bias check).

    Adapted from test_checkpointing_ssu_int8_philox_unbiased.
    """
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    batch, T = 16, 6
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float8_e4m3fn
    quant_max = 448.0

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

    # 1. fp32 state — captures true post-replay state.
    state_fp32 = state0_fp32.clone()
    out_fp32 = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_fp32,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_fp32,
        **common_kwargs,
    )

    # 2. fp8 state with Philox SR.
    rand_seed = torch.tensor([99999], device=device, dtype=torch.int64)
    state_rounded, scales_rounded = _quantize_state(state0_fp32, state_dtype, quant_max)
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_rounded,
        old_x.clone(),
        old_B.clone(),
        old_dt.clone(),
        old_cumAdt.clone(),
        cache_buf_idx.clone(),
        prev_tokens,
        out=out_rounded,
        state_scale=scales_rounded,
        rand_seed=rand_seed,
        philox_rounds=10,
        **common_kwargs,
    )

    # Compute residuals.
    fp32_vals = state_fp32.flatten()
    stochastic_residual = (
        _dequantize_state(state_rounded, scales_rounded).flatten() - fp32_vals
    )
    det_quant, det_scales = _quantize_state(state_fp32, state_dtype, quant_max)
    deterministic_residual = (
        _dequantize_state(det_quant, det_scales).flatten() - fp32_vals
    )

    nonzero_mask = deterministic_residual.abs() > 0
    num_nonzero = nonzero_mask.sum().item()
    assert num_nonzero > 1000, f"Too few roundable elements: {num_nonzero}"

    stochastic_mean = stochastic_residual[nonzero_mask].mean().item()
    stochastic_std = stochastic_residual[nonzero_mask].std().item()

    # SE-based bias check (K=4 ≈ ~3.2e-5 one-sided false-positive rate).
    se_sr = stochastic_std / (num_nonzero**0.5)
    K = 4
    assert abs(stochastic_mean) < K * se_sr, (
        f"SR mean exceeds {K}*SE (likely biased) (fp8): "
        f"stochastic_mean={stochastic_mean:.3e}, SE={se_sr:.3e}, "
        f"n_elements={num_nonzero}"
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
    not is_cvt_rs_supported(),
    reason="Philox stochastic rounding requires cvt.rs PTX (SM100a/SM110a only — "
    "not SM120a / consumer Blackwell)",
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
    uses cvt.rs PTX which only exists on SM100a/SM110a (datacenter Blackwell);
    int8/int16 (RN or SR) runs anywhere."""
    if state_dtype == torch.float8_e4m3fn and _get_sm_version() < 89:
        pytest.skip("fp8_e4m3fn requires SM 89+ (Ada Lovelace / Hopper / Blackwell)")
    if (
        use_sr
        and state_dtype in (torch.float16, torch.float8_e4m3fn)
        and not is_cvt_rs_supported()
    ):
        pytest.skip(
            f"{state_dtype} stochastic rounding requires cvt.rs PTX "
            f"(SM100a/SM110a only — not SM120a / consumer Blackwell)"
        )


# Hand-picked 30-tuple subset of the 288-element cartesian (was: 6 dtype × 2
# paged × 6 T × 2 write_ckpt × 2 configs).  Coverage targets:
#   - every dtype hit at least twice (write + no_write, paged + nopaged)
#   - every T bucket (small=6,10 / mid=16 / huge=27,32,55) hit on at least one dtype
#   - every (write_ckpt × paged) corner exercised
#   - both _CONFIGS entries used
_STATE_UPDATE_CASES = [
    # (state_dtype, paged_cache, T, write_checkpoint, cfg_idx)
    (torch.float16, True, 6, True, 0),
    (torch.float16, False, 16, False, 1),
    (torch.float16, True, 55, True, 0),
    (torch.bfloat16, True, 10, True, 1),
    (torch.bfloat16, False, 16, True, 0),
    (torch.bfloat16, True, 27, False, 1),
    (torch.bfloat16, True, 55, True, 0),
    (torch.float32, False, 6, True, 0),
    (torch.float32, True, 16, False, 1),
    (torch.float32, True, 32, True, 0),
    (torch.int8, True, 6, True, 0),
    (torch.int8, False, 16, True, 1),
    (torch.int8, True, 16, False, 0),
    (torch.int8, True, 32, True, 1),
    (torch.int8, True, 55, False, 0),
    (torch.int16, True, 16, True, 0),
    (torch.int16, False, 16, False, 1),
    (torch.int16, True, 32, True, 1),
    (torch.float8_e4m3fn, True, 6, True, 0),
    (torch.float8_e4m3fn, False, 10, True, 1),
    (torch.float8_e4m3fn, True, 16, False, 0),
    (torch.float8_e4m3fn, True, 27, True, 1),
    (torch.float8_e4m3fn, True, 55, True, 0),
    # Extra mid-T + mixed corners to fill out write_ckpt × paged grid:
    (torch.float16, False, 10, True, 0),
    (torch.bfloat16, False, 6, False, 1),
    (torch.float32, False, 16, True, 0),
    (torch.float32, True, 55, False, 1),
    (torch.int8, False, 10, True, 0),
    (torch.float8_e4m3fn, False, 16, True, 1),
    (torch.bfloat16, True, 32, True, 0),
]


@pytest.mark.parametrize(
    "state_dtype,paged_cache,T,write_checkpoint,_cfg_idx", _STATE_UPDATE_CASES
)
def test_checkpointing_state_update(
    state_dtype, paged_cache, T, write_checkpoint, _cfg_idx
):
    nheads, head_dim, d_state, ngroups = _CONFIGS[_cfg_idx]
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


# 12-tuple subset of the 48-element cartesian (was: 4 dtype × 2 paged × 3 T
# × 2 configs).  Each dtype hit 3 times; T ∈ {6,16,32} each hit; paged on +
# off both exercised per dtype; both _CONFIGS rotated.
_STATE_UPDATE_PHILOX_CASES = [
    # (state_dtype, paged_cache, T, cfg_idx)
    (torch.float16, True, 6, 0),
    (torch.float16, False, 16, 1),
    (torch.float16, True, 32, 0),
    (torch.int8, True, 6, 1),
    (torch.int8, False, 16, 0),
    (torch.int8, True, 32, 1),
    (torch.int16, False, 6, 0),
    (torch.int16, True, 16, 1),
    (torch.int16, False, 32, 0),
    (torch.float8_e4m3fn, True, 6, 1),
    (torch.float8_e4m3fn, False, 16, 0),
    (torch.float8_e4m3fn, True, 32, 1),
]


@pytest.mark.parametrize(
    "state_dtype,paged_cache,T,_cfg_idx", _STATE_UPDATE_PHILOX_CASES
)
def test_checkpointing_state_update_philox(state_dtype, paged_cache, T, _cfg_idx):
    nheads, head_dim, d_state, ngroups = _CONFIGS[_cfg_idx]
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


@pytest.mark.parametrize(
    "state_dtype",
    [torch.float16, torch.int8, torch.int16, torch.float8_e4m3fn],
    ids=["fp16", "int8", "int16", "fp8"],
)
def test_checkpointing_philox_rounding_unbiased(state_dtype):
    """Verify Philox SR is statistically unbiased (mean residual ≈ 0).

    Renamed from the upstream `test_philox_rounding_unbiased` to disambiguate
    from the existing CUDA-kernel-specific `test_philox_rounding_unbiased`
    above (which tests `checkpointing_ssu` directly, not the merged Triton)."""
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


# =============================================================================
# v20.0 — varlen (cu_seqlens) tests
# =============================================================================
# Reference strategy: run the kernel twice on the *same* data — once in varlen
# mode (packed (1, total_tokens, ...) + cu_seqlens), once per-batch in
# non-varlen mode with each sequence zero-padded to NPREDICTED.  The two
# invocations must agree on:
#   - out[bos_i : bos_i + seq_len_i]                    (rows past seq_len_i
#     are not part of the contract — the non-varlen reference produces zeros
#     from the padded zero inputs there, the varlen mode never writes them)
#   - state[cache_slot_i]                               (after replay/checkpoint)
#   - old_x / old_B / old_dt / old_cumAdt          (over [0:seq_len_i]
#     of the written T-range, again ignoring the trailing padding rows)
#
# Constraint for the comparison to be meaningful: `must_checkpoint` must agree
# between the varlen call (uses seq_len) and the per-batch padded call (uses
# NPREDICTED), since their cache write_offset / buf_write depend on it.
# Configs below pick (NPREDICTED, MAX_WINDOW, prev_k) such that the two modes
# agree:
#   (a) both no-checkpoint:  prev_k + NPREDICTED <= MAX_WINDOW.
#   (b) both checkpoint:     prev_k + min(seq_lens) > MAX_WINDOW
#                            (forces varlen to checkpoint for every sequence,
#                             and prev_k + NPREDICTED >= prev_k + seq_len >
#                             MAX_WINDOW for non-varlen).

_VARLEN_DTYPES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.int8, id="int8"),
    pytest.param(
        torch.float8_e4m3fn,
        id="fp8_e4m3fn",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or _get_sm_version() < 89,
            reason="fp8_e4m3fn requires SM 89+",
        ),
    ),
]


def _setup_varlen_inputs(
    *,
    seq_lens,
    prev_ks,
    npredicted,
    max_window,
    nheads,
    head_dim,
    d_state,
    ngroups,
    state_dtype,
    dtype,
    device,
    seed=42,
):
    """Build per-batch inputs (lists of tensors of length batch), plus shared
    static tensors (A, D, dt_bias, cache).  No varlen-specific packing yet —
    callers either pack into 3D or run the inputs through per-batch
    non-varlen calls.
    """
    batch = len(seq_lens)
    cache_size = batch
    assert max(seq_lens) <= npredicted
    assert npredicted <= max_window
    torch.manual_seed(seed)

    # Static per-head tensors (tie_hdim).
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    is_quantized = state_dtype in (torch.int8, torch.float8_e4m3fn)
    quant_max = 127.0 if state_dtype == torch.int8 else 448.0

    # Initial state.
    state0_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    if is_quantized:
        state0, state0_scale = _quantize_state(state0_fp32, state_dtype, quant_max)
    else:
        state0 = state0_fp32.to(state_dtype)
        state0_scale = None

    # Cache tensors.  `npredicted` (cache T-dim) == `max_window` here for
    # simplicity — varlen doesn't change the cache T-dim semantics.
    old_x = torch.randn(
        cache_size, max_window, nheads, head_dim, device=device, dtype=dtype
    )
    old_B = torch.randn(
        cache_size, 2, max_window, ngroups, d_state, device=device, dtype=dtype
    )
    # old_dt / old_cumAdt need physically-realistic values for the
    # replay paths — derive from a step-1 pass.
    dt1_base = torch.randn(batch, max_window, nheads, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_proc, dim=1)
    old_dt = torch.zeros(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    old_cumAdt = torch.zeros(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)
    for i in range(cache_size):
        buf = cache_buf_idx[i].item()
        old_dt[i, buf] = dt1_proc[i].T
        old_cumAdt[i, buf] = dA_cumsum1[i].T

    prev_tokens = torch.tensor(prev_ks, device=device, dtype=torch.int32)

    # Per-batch new-token inputs (truncated to seq_len_i).  dt is stored as
    # the tie_hdim *base* (shape (sl, nheads), stride[-1]=nheads), so that
    # both the packed varlen call and the per-batch padded reference can
    # broadcast the head_dim axis as stride-0 (required by the kernel's
    # `dt.stride(2)=1, dt.stride(3)=0` contract).  A concatenation +
    # `repeat` after packing reconstructs the broadcast for the varlen call;
    # the per-batch reference does the same on the padded base.
    torch.manual_seed(seed + 100)
    x_list, dt_base_list, B_list, C_list, z_list = [], [], [], [], []
    for sl in seq_lens:
        x_i = torch.randn(sl, nheads, head_dim, device=device, dtype=dtype)
        dt_base_i = torch.randn(sl, nheads, device=device, dtype=dtype)
        B_i = torch.randn(sl, ngroups, d_state, device=device, dtype=dtype)
        C_i = torch.randn(sl, ngroups, d_state, device=device, dtype=dtype)
        z_i = torch.randn(sl, nheads, head_dim, device=device, dtype=dtype)
        x_list.append(x_i)
        dt_base_list.append(dt_base_i)
        B_list.append(B_i)
        C_list.append(C_i)
        z_list.append(z_i)

    return dict(
        batch=batch,
        cache_size=cache_size,
        is_quantized=is_quantized,
        A=A,
        D=D,
        dt_bias=dt_bias,
        state0=state0,
        state0_scale=state0_scale,
        old_x=old_x,
        old_B=old_B,
        old_dt=old_dt,
        old_cumAdt=old_cumAdt,
        cache_buf_idx=cache_buf_idx,
        prev_tokens=prev_tokens,
        x_list=x_list,
        dt_base_list=dt_base_list,
        B_list=B_list,
        C_list=C_list,
        z_list=z_list,
        head_dim=head_dim,
    )


def _pack_varlen(x_list, npredicted_dim, dtype, device):
    """Pack a list of per-batch tensors of shape (seq_len_i, *rest) into
    (1, total_tokens, *rest) plus cu_seqlens (batch+1,).  Returns (packed,
    cu_seqlens).
    """
    seq_lens = [x.shape[0] for x in x_list]
    cu_seqlens = torch.tensor([0] + seq_lens, device=device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    rest = x_list[0].shape[1:]
    packed = torch.cat(x_list, dim=0).unsqueeze(0)  # (1, total_tokens, *rest)
    assert packed.shape == (1, sum(seq_lens), *rest)
    assert packed.dtype == dtype
    return packed, cu_seqlens


def _run_varlen_and_compare(
    *,
    seq_lens,
    prev_ks,
    npredicted,
    max_window,
    nheads=16,
    head_dim=64,
    d_state=128,
    ngroups=1,
    state_dtype,
    dtype=torch.bfloat16,
    use_z=False,
):
    """Shared body: run the kernel in varlen mode and per-batch padded
    non-varlen mode, then compare slices."""
    device = "cuda"
    s = _setup_varlen_inputs(
        seq_lens=seq_lens,
        prev_ks=prev_ks,
        npredicted=npredicted,
        max_window=max_window,
        nheads=nheads,
        head_dim=head_dim,
        d_state=d_state,
        ngroups=ngroups,
        state_dtype=state_dtype,
        dtype=dtype,
        device=device,
    )
    batch = s["batch"]

    # ── 1. Varlen call ──
    x_packed, cu_seqlens = _pack_varlen(s["x_list"], head_dim, dtype, device)
    # dt: pack the (sl, nheads) base, then `repeat` to broadcast the head_dim
    # axis with stride 0 (tie_hdim contract — dt.stride(2)=1, dt.stride(3)=0).
    dt_base_packed = torch.cat(s["dt_base_list"], dim=0).unsqueeze(
        0
    )  # (1, total_tokens, nheads)
    dt_packed = repeat(dt_base_packed, "b t h -> b t h p", p=head_dim)
    B_packed, _ = _pack_varlen(s["B_list"], d_state, dtype, device)
    C_packed, _ = _pack_varlen(s["C_list"], d_state, dtype, device)
    z_packed, _ = (
        _pack_varlen(s["z_list"], head_dim, dtype, device)
        if use_z
        else (
            None,
            None,
        )
    )
    out_packed = torch.zeros_like(x_packed)

    state_varlen = s["state0"].clone()
    state_scale_varlen = s["state0_scale"].clone() if s["is_quantized"] else None
    old_x_varlen = s["old_x"].clone()
    old_B_varlen = s["old_B"].clone()
    old_dt_varlen = s["old_dt"].clone()
    old_cumAdt_varlen = s["old_cumAdt"].clone()

    checkpointing_ssu(
        state_varlen,
        old_x_varlen,
        old_B_varlen,
        old_dt_varlen,
        old_cumAdt_varlen,
        s["cache_buf_idx"].clone(),
        s["prev_tokens"].clone(),
        x=x_packed,
        dt=dt_packed,
        A=s["A"],
        B=B_packed,
        C=C_packed,
        out=out_packed,
        D=s["D"],
        z=z_packed,
        dt_bias=s["dt_bias"],
        dt_softplus=True,
        state_batch_indices=None,
        state_scale=state_scale_varlen,
        cu_seqlens=cu_seqlens,
    )

    # ── 2. Per-batch padded reference (non-varlen) ──
    for i in range(batch):
        sl = seq_lens[i]

        # Pad each per-batch input to (1, npredicted, ...) — trailing rows zero.
        def _pad(t, target_T):
            rest = t.shape[1:]
            out = torch.zeros(target_T, *rest, device=device, dtype=t.dtype)
            out[:sl] = t
            return out.unsqueeze(0)

        x_i = _pad(s["x_list"][i], npredicted)
        # dt: pad the (sl, nheads) base then `repeat` so the head_dim axis
        # carries stride 0 (tie_hdim — required by the wrapper validation).
        dt_base_padded = _pad(s["dt_base_list"][i], npredicted)  # (1, T, nheads)
        dt_i = repeat(dt_base_padded, "b t h -> b t h p", p=head_dim)
        B_i = _pad(s["B_list"][i], npredicted)
        C_i = _pad(s["C_list"][i], npredicted)
        z_i = _pad(s["z_list"][i], npredicted) if use_z else None
        out_i = torch.zeros(1, npredicted, nheads, head_dim, device=device, dtype=dtype)

        # Reference cache state: snapshot of pre-call cache, restricted to slot i.
        state_ref = s["state0"].clone()
        state_scale_ref = s["state0_scale"].clone() if s["is_quantized"] else None
        # cache_buf_idx / prev_tokens: only slot i is consumed by this call.
        cbi = s["cache_buf_idx"].clone()
        pt = s["prev_tokens"].clone()
        old_x_ref = s["old_x"].clone()
        old_B_ref = s["old_B"].clone()
        old_dt_ref = s["old_dt"].clone()
        old_cumAdt_ref = s["old_cumAdt"].clone()

        # Run with a single-batch input pointing at slot i.
        state_batch_indices_i = torch.tensor([i], device=device, dtype=torch.int32)

        checkpointing_ssu(
            state_ref,
            old_x_ref,
            old_B_ref,
            old_dt_ref,
            old_cumAdt_ref,
            cbi,
            pt,
            x=x_i,
            dt=dt_i,
            A=s["A"],
            B=B_i,
            C=C_i,
            out=out_i,
            D=s["D"],
            z=z_i,
            dt_bias=s["dt_bias"],
            dt_softplus=True,
            state_batch_indices=state_batch_indices_i,
            state_scale=state_scale_ref,
        )

        # ── Compare slice for sequence i ──
        bos = int(cu_seqlens[i].item())
        # Output: varlen wrote (1, total_tokens, ...) — slice [bos:bos+sl].
        torch.testing.assert_close(
            out_packed[0, bos : bos + sl],
            out_i[0, :sl],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded output mismatch at batch={i}, seq_len={sl}",
        )

        # State for slot i: must match between the two modes.
        if s["is_quantized"]:
            actual = _dequantize_state(state_varlen[i], state_scale_varlen[i])
            expected = _dequantize_state(state_ref[i], state_scale_ref[i])
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0,
                atol=0,
                msg=f"varlen vs padded state mismatch at batch={i}",
            )
        else:
            torch.testing.assert_close(
                state_varlen[i],
                state_ref[i],
                rtol=0,
                atol=0,
                msg=f"varlen vs padded state mismatch at batch={i}",
            )

        # Cache writes: kernel writes T-rows starting at write_offset.  Both
        # modes agree on write_offset because (by construction) must_checkpoint
        # matches.  Compare the [write_offset : write_offset + sl] slice.
        must_ckpt = prev_ks[i] + sl > max_window
        write_offset = 0 if must_ckpt else prev_ks[i]
        buf_read = int(s["cache_buf_idx"][i].item())
        buf_write = 1 - buf_read if must_ckpt else buf_read

        # old_x: (cache, T, nheads, dim) — single-buffered.
        torch.testing.assert_close(
            old_x_varlen[i, write_offset : write_offset + sl],
            old_x_ref[i, write_offset : write_offset + sl],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded old_x mismatch at batch={i}",
        )
        # old_B: (cache, 2, T, ngroups, dstate) — double-buffered.
        torch.testing.assert_close(
            old_B_varlen[i, buf_write, write_offset : write_offset + sl],
            old_B_ref[i, buf_write, write_offset : write_offset + sl],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded old_B mismatch at batch={i}",
        )
        # old_dt: (cache, 2, nheads, T) — double-buffered.
        torch.testing.assert_close(
            old_dt_varlen[i, buf_write, :, write_offset : write_offset + sl],
            old_dt_ref[i, buf_write, :, write_offset : write_offset + sl],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded old_dt mismatch at batch={i}",
        )
        # old_cumAdt: same shape as old_dt.
        torch.testing.assert_close(
            old_cumAdt_varlen[i, buf_write, :, write_offset : write_offset + sl],
            old_cumAdt_ref[i, buf_write, :, write_offset : write_offset + sl],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded old_cumAdt mismatch at batch={i}",
        )


# ``test_checkpointing_ssu_varlen_uniform_seqlen`` was retired — its
# coverage (pure Branch B, varlen indexing) is a subset of
# ``test_checkpointing_ssu_varlen_mixed_no_checkpoint`` below, which
# additionally tests seq_len < NPREDICTED masking.


@pytest.mark.parametrize("state_dtype", _VARLEN_DTYPES)
def test_checkpointing_ssu_varlen_mixed_no_checkpoint(state_dtype):
    """Mixed seq_lens, prev_k chosen so that prev_k + NPREDICTED <= MAX_WINDOW
    for every batch (no-checkpoint regime).  Tests that the varlen masking
    of T-rows >= seq_len matches the non-varlen reference where those rows
    contain zero-padded inputs (which compute_CB_scaled / matmul should
    zero-propagate)."""
    npredicted = 4
    max_window = 16
    seq_lens = [3, 1, 4, 2, 4]  # < NPREDICTED for some batches
    prev_ks = [0, 5, 10, 12, 8]  # all <= MAX_WINDOW - NPREDICTED = 12
    _run_varlen_and_compare(
        seq_lens=seq_lens,
        prev_ks=prev_ks,
        npredicted=npredicted,
        max_window=max_window,
        state_dtype=state_dtype,
    )


@pytest.mark.parametrize("state_dtype", _VARLEN_DTYPES)
def test_checkpointing_ssu_varlen_mixed_checkpoint(state_dtype):
    """Mixed seq_lens, prev_k large enough that prev_k + min(seq_lens) > MAX_WINDOW
    — forces every batch (varlen and non-varlen) to checkpoint.  Exercises
    the replay + state-write path under varlen indexing."""
    npredicted = 8
    max_window = 16
    seq_lens = [3, 5, 8, 6, 7]
    # min(seq_lens) = 3; pick prev_k_i >= 14 so every prev_k + seq_len > 16.
    prev_ks = [14, 15, 16, 14, 15]
    _run_varlen_and_compare(
        seq_lens=seq_lens,
        prev_ks=prev_ks,
        npredicted=npredicted,
        max_window=max_window,
        state_dtype=state_dtype,
    )


def _run_varlen_cuda_vs_triton(
    *,
    seq_lens,
    prev_ks,
    npredicted,
    max_window,
    write_checkpoint,
    state_dtype,
    nheads=16,
    head_dim=64,
    d_state=128,
    ngroups=1,
    dtype=torch.bfloat16,
):
    """Compare CUDA varlen vs Triton varlen reference on the same packed inputs.

    Independent-reference check — catches bugs that the CUDA-self-consistency
    tests (varlen vs per-batch padded non-varlen) would miss because both
    sides go through the same CUDA kernel.

    The Triton reference takes `write_checkpoint` as a uniform per-call
    constexpr (no per-batch branching).  This caller chooses configs where
    every sequence ends up on the same branch:
      - write_checkpoint=False: all (prev_k + seq_len) <= MAX_WINDOW.
      - write_checkpoint=True : all (prev_k + seq_len) > MAX_WINDOW (i.e.
        prev_k + min(seq_lens) > MAX_WINDOW).
    """
    device = "cuda"
    s = _setup_varlen_inputs(
        seq_lens=seq_lens,
        prev_ks=prev_ks,
        npredicted=npredicted,
        max_window=max_window,
        nheads=nheads,
        head_dim=head_dim,
        d_state=d_state,
        ngroups=ngroups,
        state_dtype=state_dtype,
        dtype=dtype,
        device=device,
    )

    # Pack varlen inputs once (used by both kernels).
    x_packed, cu_seqlens = _pack_varlen(s["x_list"], head_dim, dtype, device)
    dt_base_packed = torch.cat(s["dt_base_list"], dim=0).unsqueeze(0)
    dt_packed = repeat(dt_base_packed, "b t h -> b t h p", p=head_dim)
    B_packed, _ = _pack_varlen(s["B_list"], d_state, dtype, device)
    C_packed, _ = _pack_varlen(s["C_list"], d_state, dtype, device)

    # Verify the must_checkpoint precondition the Triton ref needs (uniform
    # across batch).
    for i, (pk, sl) in enumerate(zip(prev_ks, seq_lens, strict=True)):
        actual = (pk + sl) > max_window
        assert actual == write_checkpoint, (
            f"test setup error: batch {i} has must_checkpoint={actual} but "
            f"write_checkpoint={write_checkpoint} requested"
        )

    # --- CUDA varlen ---
    state_cuda = s["state0"].clone()
    state_scale_cuda = s["state0_scale"].clone() if s["is_quantized"] else None
    old_x_cuda = s["old_x"].clone()
    old_B_cuda = s["old_B"].clone()
    old_dt_cuda = s["old_dt"].clone()
    old_cumAdt_cuda = s["old_cumAdt"].clone()
    out_cuda = torch.zeros_like(x_packed)

    checkpointing_ssu(
        state_cuda,
        old_x_cuda,
        old_B_cuda,
        old_dt_cuda,
        old_cumAdt_cuda,
        s["cache_buf_idx"].clone(),
        s["prev_tokens"].clone(),
        x=x_packed,
        dt=dt_packed,
        A=s["A"],
        B=B_packed,
        C=C_packed,
        out=out_cuda,
        D=s["D"],
        dt_bias=s["dt_bias"],
        dt_softplus=True,
        state_scale=state_scale_cuda,
        cu_seqlens=cu_seqlens,
    )

    # --- Triton varlen ---
    state_tri = s["state0"].clone()
    state_scale_tri = s["state0_scale"].clone() if s["is_quantized"] else None
    old_x_tri = s["old_x"].clone()
    old_B_tri = s["old_B"].clone()
    old_dt_tri = s["old_dt"].clone()
    old_cumAdt_tri = s["old_cumAdt"].clone()
    out_tri = torch.zeros_like(x_packed)

    checkpointing_state_update(
        state_tri,
        old_x_tri,
        old_B_tri,
        old_dt_tri,
        old_cumAdt_tri,
        s["cache_buf_idx"].clone(),
        s["prev_tokens"].clone(),
        x=x_packed,
        dt=dt_packed,
        A=s["A"],
        B=B_packed,
        C=C_packed,
        out=out_tri,
        D=s["D"],
        dt_bias=s["dt_bias"],
        dt_softplus=True,
        state_scales=state_scale_tri,
        write_checkpoint=write_checkpoint,
        cu_seqlens=cu_seqlens,
        max_seqlen=npredicted,
    )

    # --- Compare ---
    # Tolerances mirror existing CUDA-vs-Triton tests: bf16 matmul + dtype
    # accumulation differences across the two kernels.
    if s["is_quantized"]:
        out_atol, out_rtol = (1.6 if state_dtype == torch.int8 else 4.0), 5e-2
    else:
        out_atol, out_rtol = 5e-1, 2e-2
    torch.testing.assert_close(
        out_cuda,
        out_tri,
        rtol=out_rtol,
        atol=out_atol,
        msg=f"CUDA vs Triton varlen output mismatch (write_ckpt={write_checkpoint})",
    )
    # State comparison: only checkpoint runs update state HBM.
    if write_checkpoint:
        if s["is_quantized"]:
            cuda_state_fp32 = _dequantize_state(state_cuda, state_scale_cuda)
            tri_state_fp32 = _dequantize_state(state_tri, state_scale_tri)
            state_atol = 1.1 if state_dtype == torch.int8 else 2.5
            torch.testing.assert_close(
                cuda_state_fp32,
                tri_state_fp32,
                rtol=5e-2,
                atol=state_atol,
                msg="CUDA vs Triton varlen state mismatch (dequantized)",
            )
        else:
            torch.testing.assert_close(
                state_cuda,
                state_tri,
                rtol=2e-2,
                atol=5e-1,
                msg="CUDA vs Triton varlen state mismatch",
            )


@pytest.mark.parametrize("state_dtype", _VARLEN_DTYPES)
def test_checkpointing_ssu_varlen_cuda_vs_triton_no_checkpoint(state_dtype):
    """CUDA vs Triton (independent reference) — pure-Branch-B varlen.

    Constraint on test config: the Triton kernel caps prev_k <= NPREDICTED
    by design (its replay loop masks cache reads at `offs_t < T`).  We
    therefore size NPREDICTED = MAX_WINDOW so any prev_k in [0, MAX_WINDOW]
    is acceptable to both kernels.
    """
    _run_varlen_cuda_vs_triton(
        seq_lens=[3, 1, 4, 2, 4],
        prev_ks=[0, 5, 10, 12, 8],  # all prev_k + max(seq_len)=4 <= 16
        npredicted=16,
        max_window=16,
        write_checkpoint=False,
        state_dtype=state_dtype,
    )


@pytest.mark.parametrize("state_dtype", _VARLEN_DTYPES)
def test_checkpointing_ssu_varlen_cuda_vs_triton_checkpoint(state_dtype):
    """CUDA vs Triton (independent reference) — pure-Branch-A varlen."""
    _run_varlen_cuda_vs_triton(
        seq_lens=[3, 5, 8, 6, 7],
        prev_ks=[14, 15, 16, 14, 15],
        npredicted=16,
        max_window=16,
        write_checkpoint=True,
        state_dtype=state_dtype,
    )


# =============================================================================
# v20.0 — non-contiguous stride tests (varlen + non-varlen)
# =============================================================================
# Verify the wrapper extracts the correct per-tensor strides — both
# `stride_batch` and `stride_mtp` — and the kernel honors them via
# `params.X_stride_{batch,mtp}` rather than assuming the contiguous packed
# layout.  Strategy:
#   1. Build a contiguous "golden" batch of inputs; run the kernel on it.
#   2. Build a non-contiguous "padded" batch with the same data but extra
#      slack along one outer dim per tensor; run the kernel on it.
#   3. The two runs must produce byte-identical output / state / cache writes
#      (same computation, same data — only the gmem layout differs).
#
# Tensors covered: x, dt (tie_hdim preserved), B, C, z, output, plus the
# cache-side old_x / old_B / old_dt / old_cumAdt.  Each gets a different
# padding pattern so we exercise a mix of stride changes.


def _pad_outer(t, pad):
    """Return a non-contiguous view of `t` with extra `pad` rows on dim 1
    (slack between the slice and the underlying buffer).  Result has the
    same data as `t` but `stride(0) > size(1) * stride(1)`."""
    shape = list(t.shape)
    pad_shape = list(shape)
    pad_shape[1] = shape[1] + pad
    full = torch.empty(pad_shape, device=t.device, dtype=t.dtype)
    full.narrow(1, 0, shape[1]).copy_(t)
    return full.narrow(1, 0, shape[1])


def _pad_inner(t, pad, dim):
    """Pad `t` along axis `dim` by `pad` rows, slice back to original.
    Result has stride(dim-1) > size(dim) * stride(dim)."""
    shape = list(t.shape)
    pad_shape = list(shape)
    pad_shape[dim] = shape[dim] + pad
    full = torch.empty(pad_shape, device=t.device, dtype=t.dtype)
    full.narrow(dim, 0, shape[dim]).copy_(t)
    return full.narrow(dim, 0, shape[dim])


@pytest.mark.parametrize(
    "state_dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.int8, id="int8"),
    ],
)
@pytest.mark.parametrize("varlen", [False, True], ids=["non_varlen", "varlen"])
def test_checkpointing_ssu_noncontig(state_dtype, varlen):
    """Run the kernel on tensors with non-default (padded) strides and
    compare to a contiguous-clone reference.  Exercises stride plumbing
    for x / dt / B / C / z / out / old_x / old_B / old_dt / old_cumAdt.
    """
    device = "cuda"
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    dtype = torch.bfloat16
    npredicted = 8
    max_window = 16

    if varlen:
        seq_lens = [3, 8, 5, 7]
        batch = len(seq_lens)
        cu_seqlens = torch.tensor(
            [0] + seq_lens, device=device, dtype=torch.int32
        ).cumsum(0, dtype=torch.int32)
    else:
        batch = 4
        seq_lens = [npredicted] * batch
        cu_seqlens = None
    cache_size = batch
    total_tokens = sum(seq_lens)

    is_quantized = state_dtype in (torch.int8, torch.float8_e4m3fn)
    quant_max = 127.0 if state_dtype == torch.int8 else 448.0

    torch.manual_seed(7)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    state0_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    if is_quantized:
        state0, state0_scale = _quantize_state(state0_fp32, state_dtype, quant_max)
    else:
        state0 = state0_fp32.to(state_dtype)
        state0_scale = None

    # Cache tensors: realistic dt_proc / cumAdt from a step-1 pass (so
    # replay sees consistent values).
    dt1_base = torch.randn(batch, max_window, nheads, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    dA_cumsum1 = torch.cumsum(A_base.float()[None, None, :] * dt1_proc, dim=1)

    old_x_contig = torch.randn(
        cache_size, max_window, nheads, head_dim, device=device, dtype=dtype
    )
    old_B_contig = torch.randn(
        cache_size, 2, max_window, ngroups, d_state, device=device, dtype=dtype
    )
    old_dt_contig = torch.zeros(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    old_ca_contig = torch.zeros(
        cache_size, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    cache_buf_idx = torch.zeros(cache_size, device=device, dtype=torch.int32)
    for i in range(cache_size):
        old_dt_contig[i, 0] = dt1_proc[i].T
        old_ca_contig[i, 0] = dA_cumsum1[i].T

    prev_tokens = torch.tensor([2, 4, 6, 3], device=device, dtype=torch.int32)[:batch]

    # New-token inputs.
    torch.manual_seed(13)
    if varlen:
        x_contig = torch.randn(
            1, total_tokens, nheads, head_dim, device=device, dtype=dtype
        )
        dt_base_contig = torch.randn(
            1, total_tokens, nheads, device=device, dtype=dtype
        )
        B_contig = torch.randn(
            1, total_tokens, ngroups, d_state, device=device, dtype=dtype
        )
        C_contig = torch.randn(
            1, total_tokens, ngroups, d_state, device=device, dtype=dtype
        )
        z_contig = torch.randn(
            1, total_tokens, nheads, head_dim, device=device, dtype=dtype
        )
    else:
        x_contig = torch.randn(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        dt_base_contig = torch.randn(
            batch, npredicted, nheads, device=device, dtype=dtype
        )
        B_contig = torch.randn(
            batch, npredicted, ngroups, d_state, device=device, dtype=dtype
        )
        C_contig = torch.randn(
            batch, npredicted, ngroups, d_state, device=device, dtype=dtype
        )
        z_contig = torch.randn(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
    dt_contig = repeat(dt_base_contig, "b t h -> b t h p", p=head_dim)

    common_static = dict(
        A=A,
        dt_bias=dt_bias,
        D=D,
        dt_softplus=True,
        state_batch_indices=None,
    )

    # ── Run 1: contiguous golden ──
    state_ref = state0.clone()
    state_scale_ref = state0_scale.clone() if is_quantized else None
    old_x_ref = old_x_contig.clone()
    old_B_ref = old_B_contig.clone()
    old_dt_ref = old_dt_contig.clone()
    old_ca_ref = old_ca_contig.clone()
    out_ref = torch.zeros_like(x_contig)
    checkpointing_ssu(
        state_ref,
        old_x_ref,
        old_B_ref,
        old_dt_ref,
        old_ca_ref,
        cache_buf_idx.clone(),
        prev_tokens.clone(),
        x=x_contig,
        dt=dt_contig,
        B=B_contig,
        C=C_contig,
        out=out_ref,
        z=z_contig,
        state_scale=state_scale_ref,
        cu_seqlens=cu_seqlens,
        max_seqlen=npredicted if varlen else None,
        **common_static,
    )

    # ── Run 2: same data via non-contiguous tensors ──
    # Build padded views — same data, different strides.  Each tensor gets
    # a different padding count so its stride differs from the natural one.
    x_nc = _pad_outer(x_contig, pad=3)
    B_nc = _pad_outer(B_contig, pad=2)
    C_nc = _pad_outer(C_contig, pad=5)
    z_nc = _pad_outer(z_contig, pad=4)
    out_nc_storage = torch.empty_like(_pad_outer(x_contig, pad=6))
    out_nc = out_nc_storage.narrow(1, 0, x_contig.size(1))
    # dt: pad the (b/1, T, nheads) base then broadcast — broadcast preserves
    # the padded stride on dims 0/1 while keeping stride(3) == 0 (tie_hdim).
    dt_base_nc = _pad_outer(dt_base_contig, pad=7)
    dt_nc = repeat(dt_base_nc, "b t h -> b t h p", p=head_dim)
    # Cache tensors: pad an interior dim to vary the cache-side strides.
    old_x_nc_storage = _pad_inner(old_x_contig, pad=2, dim=1)
    old_x_nc = old_x_nc_storage  # already shaped (cache, max_window, nheads, dim)
    old_B_nc_storage = _pad_inner(old_B_contig, pad=2, dim=2)
    old_B_nc = old_B_nc_storage
    old_dt_nc_storage = _pad_inner(old_dt_contig, pad=4, dim=2)
    old_dt_nc = old_dt_nc_storage
    old_ca_nc_storage = _pad_inner(old_ca_contig, pad=3, dim=2)
    old_ca_nc = old_ca_nc_storage

    # Sanity: assert that the padded views actually have non-default outer
    # strides — `_pad_outer` pads dim 1, which makes stride(0) > the
    # contig size(1)*stride(1).  Cache tensors padded along dim 1/2 are
    # checked similarly on the affected outer stride.
    assert x_nc.stride(0) != x_contig.stride(0), (
        f"x_nc stride didn't change: {x_nc.stride()} == {x_contig.stride()}"
    )
    assert dt_nc.stride(0) != dt_contig.stride(0), (
        f"dt_nc stride(0) didn't change: {dt_nc.stride()} == {dt_contig.stride()}"
    )
    assert dt_nc.stride(3) == 0, f"dt tie_hdim broken: stride(3)={dt_nc.stride(3)}"
    assert old_x_nc.stride(0) != old_x_contig.stride(0), (
        f"old_x stride(0) didn't change: {old_x_nc.stride()} == {old_x_contig.stride()}"
    )
    assert old_B_nc.stride(1) != old_B_contig.stride(1), (
        f"old_B stride(1) didn't change: {old_B_nc.stride()} == {old_B_contig.stride()}"
    )
    assert old_dt_nc.stride(1) != old_dt_contig.stride(1), (
        f"old_dt stride(1) didn't change: {old_dt_nc.stride()} == {old_dt_contig.stride()}"
    )

    state_test = state0.clone()
    state_scale_test = state0_scale.clone() if is_quantized else None
    # Cache writes: we need contiguous storage tensors with the same data so
    # the kernel can write into them and we can compare back.  Use the _nc
    # views directly — they alias the padded storage so written rows persist
    # in the padded buffer.  Compare via .contiguous() at the end.
    cbi_test = cache_buf_idx.clone()
    pt_test = prev_tokens.clone()
    # Copy contig data into the nc views before the call (initial state of
    # the cache).
    old_x_nc.copy_(old_x_contig)
    old_B_nc.copy_(old_B_contig)
    old_dt_nc.copy_(old_dt_contig)
    old_ca_nc.copy_(old_ca_contig)

    checkpointing_ssu(
        state_test,
        old_x_nc,
        old_B_nc,
        old_dt_nc,
        old_ca_nc,
        cbi_test,
        pt_test,
        x=x_nc,
        dt=dt_nc,
        B=B_nc,
        C=C_nc,
        out=out_nc,
        z=z_nc,
        state_scale=state_scale_test,
        cu_seqlens=cu_seqlens,
        max_seqlen=npredicted if varlen else None,
        **common_static,
    )

    # ── Compare ──
    # Output: same data layout (only outer-dim padding differs); .contiguous()
    # the non-contig view to get the same layout as the reference.
    torch.testing.assert_close(
        out_nc.contiguous(),
        out_ref,
        rtol=0,
        atol=0,
        msg=f"output mismatch under non-contig (varlen={varlen})",
    )

    if is_quantized:
        torch.testing.assert_close(
            _dequantize_state(state_test, state_scale_test),
            _dequantize_state(state_ref, state_scale_ref),
            rtol=0,
            atol=0,
            msg=f"state (dequant) mismatch (varlen={varlen})",
        )
    else:
        torch.testing.assert_close(
            state_test,
            state_ref,
            rtol=0,
            atol=0,
            msg=f"state mismatch (varlen={varlen})",
        )

    torch.testing.assert_close(
        old_x_nc.contiguous(),
        old_x_ref,
        rtol=0,
        atol=0,
        msg=f"old_x mismatch (varlen={varlen})",
    )
    torch.testing.assert_close(
        old_B_nc.contiguous(),
        old_B_ref,
        rtol=0,
        atol=0,
        msg=f"old_B mismatch (varlen={varlen})",
    )
    torch.testing.assert_close(
        old_dt_nc.contiguous(),
        old_dt_ref,
        rtol=0,
        atol=0,
        msg=f"old_dt mismatch (varlen={varlen})",
    )
    torch.testing.assert_close(
        old_ca_nc.contiguous(),
        old_ca_ref,
        rtol=0,
        atol=0,
        msg=f"old_cumAdt mismatch (varlen={varlen})",
    )
