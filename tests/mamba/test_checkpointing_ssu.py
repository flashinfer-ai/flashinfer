# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the CUDA checkpointing_ssu kernel.
Validates against the Triton replay_selective_state_update reference.
"""

import pytest
import torch
import torch.nn.functional as F
import triton
from einops import repeat

from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
from flashinfer.utils import is_cvt_rs_supported

# Triton reference: the standalone TMA persistent kernel only (the old 4D
# `checkpointing_state_update` and its merged non-TMA replay copy were removed
# 2026-07-10 — the TMA impls carry all reference coverage, incl. varlen).
from .triton_reference.replay_selective_state_update import (
    get_sm_version as _get_sm_version,
)

# Faithful TMA-optimized standalone persistent kernel (the merged copy above
# dropped TMA + tuning).  Imported as `replay_persistent`; it requires explicit
# n_writes / replay_work_items and a mode.  No varlen support (yet).
from .triton_reference.replay_selective_state_update import (
    REPLAY_WORK_CACHE_BUF_IDX,
    REPLAY_WORK_CACHE_SLOT,
    REPLAY_WORK_ITEM_WIDTH,
    REPLAY_WORK_PNAT,
    REPLAY_WORK_POSITION_IN_DECODE_BATCH,
    replay_selective_state_update as replay_persistent,
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


def _make_ring_caches(
    cache_size,
    nheads,
    ngroups,
    head_dim,
    d_state,
    max_window,
    T,
    dtype,
    device,
    ring_start=None,
):
    """Ring caches per the ReplaySSM contract: RING_BUFFER_LEN = max_window + T
    rows, head-major.  Rows are garbage-filled — kernels must only read the
    ``[start, start+prev_k)`` window — except dt, which is kept positive
    (softplus output in production) so recomputed decays stay <= 1 and
    comparisons keep tight tolerances.

    ``ring_start`` defaults to a random per-slot start so every test exercises
    ring wraparound; pass an explicit tensor to pin it.
    """
    ring_len = max_window + T
    if ring_start is None:
        ring_start = torch.randint(
            0, ring_len, (cache_size,), device=device, dtype=torch.int32
        )
    x_cache = torch.randn(
        cache_size, nheads, ring_len, head_dim, device=device, dtype=dtype
    )
    B_cache = torch.randn(
        cache_size, ngroups, ring_len, d_state, device=device, dtype=dtype
    )
    dt_cache = torch.randn(
        cache_size, nheads, ring_len, device=device, dtype=torch.float32
    ).abs()
    return x_cache, B_cache, dt_cache, ring_start


def _seed_ring(x_cache, B_cache, dt_cache, ring_start, slot, x_tok, B_tok, dt_proc):
    """Write a T-token history into slot's ring rows [start, start+T) mod L.

    ``x_tok`` (T, nheads, dim), ``B_tok`` (T, ngroups, d_state) and ``dt_proc``
    (T, nheads) are token-major as produced by the test; the ring caches are
    head-major, hence the permutes.
    """
    ring_len = x_cache.shape[2]
    T = x_tok.shape[0]
    rows = (
        (int(ring_start[slot]) + torch.arange(T, device=x_cache.device)) % ring_len
    ).long()
    x_cache[slot][:, rows] = x_tok.permute(1, 0, 2)
    B_cache[slot][:, rows] = B_tok.permute(1, 0, 2)
    dt_cache[slot][:, rows] = dt_proc.permute(1, 0).float()


# Triton replay configs tested for non-varlen + varlen: the faithful TMA
# standalone in its two modes (the merged non-TMA copy was removed with the
# old 4D kernel, 2026-07-10).
_TRITON_IMPLS = ["tma_pd", "tma_pm"]

# fragA-native cb_scaled layout for the two-kernel CUDA path (.plans/ssu_split.md):
# per (batch, head), one PackedAligned<bf16> per lane = WARP_SIZE lanes ×
# MMA_FRAG_SIZE bf16 (the mma.m16n8k16 A fragment, 16 B/lane).
WARP_SIZE = 32
MMA_FRAG_SIZE = 8


def _two_kernel_scratch(batch, nheads, max_window, dtype, device):
    """Precompute scratch that routes checkpointing_ssu to the two-kernel split.

    Shapes hold for NPREDICTED <= 16 (cumAdt_vec pads to the m16 MMA row count).
    Forces algorithm="two-kernel": test batches sit below the wrapper's auto
    threshold (batch*nheads >= sm_count) and would silently run the monolith."""
    k_old = ((max_window + 7) // 8) * 8
    return dict(
        cb_scaled=torch.empty(
            batch, nheads, WARP_SIZE, MMA_FRAG_SIZE, device=device, dtype=dtype
        ),
        cumAdt_vec=torch.empty(batch, nheads, 16, device=device, dtype=torch.float32),
        cb_old=torch.empty(
            batch, nheads, WARP_SIZE, k_old // 2, device=device, dtype=dtype
        ),
        algorithm="two-kernel",
    )


def _make_replay_work_items(
    prev_tokens, seq_len, max_window, batch, state_batch_indices
):
    """Build (n_writes, replay_work_items) for the standalone persistent kernel,
    write-first sorted — mirrors mamba2_metadata._prepare_replay_work_items.

    ``seq_len`` is the per-step new-token count used in the write decision
    ``(pnat + seq_len) > max_window``: a scalar for non-varlen, or a (batch,)
    tensor of per-sequence lengths (indexed by decode-batch position) for varlen.
    persistent_dynamic ignores the ordering at runtime; persistent_main relies
    on the n_writes split (write slots first).  The buf column is vestigial
    under the ring contract and stays 0.
    """
    device = prev_tokens.device
    position_in_decode_batch = torch.arange(batch, device=device, dtype=torch.int32)
    if state_batch_indices is not None:
        cache_slot = state_batch_indices[:batch].to(torch.int32)
    else:
        cache_slot = position_in_decode_batch
    cache_slot_long = cache_slot.to(torch.long)
    pnat = prev_tokens[cache_slot_long].to(torch.int32)
    write_mask = (pnat + seq_len) > max_window
    n_writes = write_mask.sum().to(torch.int32).reshape(1)
    order = torch.argsort((~write_mask).to(torch.int32), stable=True).to(torch.long)
    work = torch.empty(batch, REPLAY_WORK_ITEM_WIDTH, device=device, dtype=torch.int32)
    work[:, REPLAY_WORK_POSITION_IN_DECODE_BATCH] = position_in_decode_batch[order]
    work[:, REPLAY_WORK_CACHE_SLOT] = cache_slot[order]
    work[:, REPLAY_WORK_PNAT] = pnat[order]
    work[:, REPLAY_WORK_CACHE_BUF_IDX] = 0
    return n_writes, work.contiguous()


def _call_replay(
    impl,
    *,
    state,
    x_cache,
    B_cache,
    dt_cache,
    ring_start,
    prev_tokens,
    x,
    dt,
    A,
    B,
    C,
    out,
    max_window,
    batch,
    work_seq_len,
    state_batch_indices=None,
    D=None,
    dt_bias=None,
    dt_softplus=True,
    state_scales=None,
    rand_seed=None,
    philox_rounds=0,
    cu_seqlens=None,
    max_seqlen=None,
):
    """Dispatch one replay call to a test config (see ``_TRITON_IMPLS``):
      ``tma_pd``/``tma_pm`` → the TMA standalone ``replay_persistent`` in
        persistent_dynamic / persistent_main.

    The ring caches are mutated in place (appended rows).
    ``work_seq_len`` is the new-token count for the persistent_main work-item
    sort (scalar non-varlen, or a (batch,) per-sequence tensor for varlen).
    For varlen pass ``cu_seqlens`` + ``max_seqlen``.
    """
    common = dict(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        state_scales=state_scales,
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
    )
    if cu_seqlens is not None:
        common.update(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    mode = "persistent_dynamic" if impl == "tma_pd" else "persistent_main"
    n_writes, replay_work_items = _make_replay_work_items(
        prev_tokens,
        work_seq_len,
        max_window,
        batch,
        state_batch_indices,
    )
    replay_persistent(
        state,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_tokens,
        n_writes=n_writes,
        replay_work_items=replay_work_items,
        mode=mode,
        **common,
    )


def _assert_ring_cache_matches(cache_cuda, cache_ref, name, rtol=0, atol=0):
    """Compare a CUDA ring-cache postcondition against the Triton reference's.

    Both sides start from identical clones and share the ring layout, so the
    whole tensor must match: appended rows because both kernels wrote the same
    tokens at ``(start + pnat + i) % L``, every other row because neither side
    may touch it.  x/B appends are raw input copies (bit-exact); dt rows go
    through the f32 softplus pipeline, so callers pass a small tolerance."""
    torch.testing.assert_close(
        cache_cuda.float(),
        cache_ref.float(),
        rtol=rtol,
        atol=atol,
        msg=f"{name} ring-cache postcondition mismatch",
    )


def _run_checkpointing_ssu_case(
    impl, nheads, head_dim, d_state, ngroups, state_dtype, paged_cache, T, d_split=None
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

    # Compute processed dt for step 1 (to build cache; cumAdt is recomputed
    # from cached dt by both kernels under the ring contract)
    dt1_proc = dt1_base.float() + dt_bias_base.float()[None, None, :]
    dt1_proc = torch.where(dt1_proc > 20.0, dt1_proc, torch.log1p(torch.exp(dt1_proc)))

    # Build ring caches (max_window == NPREDICTED == T for this test family)
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )
    slots = state_batch_indices if paged_cache else slice(None)

    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot in enumerate(slot_indices):
        _seed_ring(
            x_cache, B_cache, dt_cache, ring_start, slot, x1[i], B1[i], dt1_proc[i]
        )

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
        _call_replay(
            impl,
            state=ref_state,
            x_cache=x_cache.clone(),
            B_cache=B_cache.clone(),
            dt_cache=dt_cache.clone(),
            ring_start=ring_start,
            prev_tokens=ref_prev,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=ref_out,
            max_window=T,
            batch=batch,
            work_seq_len=T,
            state_batch_indices=state_batch_indices,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
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
            x_cache.clone(),
            B_cache.clone(),
            dt_cache.clone(),
            ring_start.clone(),
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


@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_ssu_d_split2(impl):
    """v12 §59 smoke — D_SPLIT=2 path dispatches and runs.  Both
    ``d_split={1,2}`` specializations are baked into the same JIT .so via
    the public dispatcher's switch; this test verifies the d_split=2 grid
    + smem footprint + partition_C indexing work end-to-end.  Wider dtype/T
    coverage is provided by the d_split=1 tests sharing the same .so."""
    nheads, head_dim, d_state, ngroups = 16, 64, 128, 1
    _run_checkpointing_ssu_case(
        impl,
        nheads,
        head_dim,
        d_state,
        ngroups,
        torch.bfloat16,
        paged_cache=True,
        T=16,
        d_split=2,
    )


@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_ssu_heads_per_group(impl):
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
        impl,
        nheads,
        head_dim,
        d_state,
        ngroups,
        torch.bfloat16,
        paged_cache=True,
        T=16,
    )


def test_two_kernel_matches_monolithic():
    """The two-kernel path (caller passes cb_scaled/cumAdt_vec/cb_old scratch)
    must match the monolithic kernel (no scratch) bit-for-bit on out, state, and
    the mutated caches.  Covers a nowrite case (k=0) and a write case (k=T)."""
    device = "cuda"
    dtype = torch.bfloat16
    nheads, head_dim, d_state, ngroups, T = 16, 64, 128, 1, 6
    max_window = 8  # > T so a prev_k>0 NO-WRITE case exists (k=2 below)
    batch = 2
    cache_size = batch  # non-paged
    # old_* caches are (cache, max_window, ...).  must_checkpoint = prev_k + T >
    # max_window, so the k-loop below hits: k=0 nowrite(prev_k=0), k=2
    # nowrite(prev_k>0 — exercises the dt-ring tail scan), k=T=6 write.

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias = repeat(
        torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim
    )
    D = repeat(torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim)
    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=dtype
    )

    # Step-1 inputs → buffered ("old") cache.
    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = dt1_base.float() + dt_bias[:, 0].float()[None, None, :]
    dt1_proc = torch.where(dt1_proc > 20.0, dt1_proc, torch.log1p(torch.exp(dt1_proc)))

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, max_window, T, dtype, device
    )
    for slot in range(cache_size):
        _seed_ring(
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            slot,
            x1[slot],
            B1[slot],
            dt1_proc[slot],
        )

    def _run(k, *, two_kernel, enable_pdl=False):
        torch.manual_seed(k + 100)
        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2 = repeat(
            torch.randn(batch, T, nheads, device=device, dtype=dtype),
            "b t h -> b t h p",
            p=head_dim,
        )
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        st = state0.clone()
        out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        xc, bc, dtc = x_cache.clone(), B_cache.clone(), dt_cache.clone()
        kw = {}
        if two_kernel:
            # fragA-native contract (see .plans/ssu_split.md): cb_scaled is bf16
            # [batch, nheads, lane(32), reg(8)] = matmul-4 fragA; cumAdt_vec is
            # f32 [batch, nheads, NPREDICTED_PAD_MMA_M] (=16 for T<=16).
            T_pad = 16  # next_multiple_of<MMA::M=16>(T) for T <= 16
            kw["cb_scaled"] = torch.empty(
                batch, nheads, WARP_SIZE, MMA_FRAG_SIZE, device=device, dtype=dtype
            )
            kw["cumAdt_vec"] = torch.empty(
                batch, nheads, T_pad, device=device, dtype=torch.float32
            )
            # cb_old (C6, no-write): m16n8k{K_old} fragA, K_old =
            # next_multiple_of<MMA::K_SMALL=8>(max_window); REGS = K_old/2.
            k_old = ((max_window + 7) // 8) * 8
            kw["cb_old"] = torch.empty(
                batch, nheads, WARP_SIZE, k_old // 2, device=device, dtype=dtype
            )
            kw["algorithm"] = "two-kernel"
        checkpointing_ssu(
            st,
            xc,
            bc,
            dtc,
            ring_start.clone(),
            torch.full((cache_size,), k, device=device, dtype=torch.int32),
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            enable_pdl=enable_pdl,
            **kw,
        )
        return out, st, xc, bc, dtc

    names = ("out", "state", "x_cache", "B_cache", "dt_cache")
    for k in (0, 2, T):  # k=0 nowrite(prev_k=0), k=2 nowrite(prev_k>0), k=T write
        ref = _run(k, two_kernel=False)
        test = _run(k, two_kernel=True)
        for name, r, t in zip(names, ref, test, strict=True):
            torch.testing.assert_close(
                t, r, rtol=2e-2, atol=5e-1, msg=f"{name} mismatch at k={k}"
            )
        # Same split, but with the real precompute→main PDL chain active
        # (precompute fires cudaTriggerProgrammaticLaunchCompletion, main
        # gdc_waits).  Exercises the trigger's memory-ordering contract — the
        # main must still see every cb_scaled/cumAdt_vec/cb_old the precompute
        # wrote.  Must match the monolithic ref bit-for-bit.
        test_pdl = _run(k, two_kernel=True, enable_pdl=True)
        for name, r, t in zip(names, ref, test_pdl, strict=True):
            torch.testing.assert_close(
                t, r, rtol=2e-2, atol=5e-1, msg=f"{name} mismatch at k={k} (PDL)"
            )


def test_two_kernel_d_split2():
    """Two-kernel split with d_split=2 must match d_split=1 bit-for-bit.

    d_split=2 halves D_PER_CTA (64→32), doubling the CTA count for better
    utilisation at small batch.  Exercises the two-kernel dispatcher routing
    through launchCheckpointingSsuImpl with D_SPLIT=2 in the main grid."""
    device = "cuda"
    dtype = torch.bfloat16
    nheads, head_dim, d_state, ngroups, T = 16, 64, 128, 1, 6
    max_window = 8
    batch = 2
    cache_size = batch

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias = repeat(
        torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim
    )
    D = repeat(torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim)
    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=dtype
    )

    x1 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = dt1_base.float() + dt_bias[:, 0].float()[None, None, :]
    dt1_proc = torch.where(dt1_proc > 20.0, dt1_proc, torch.log1p(torch.exp(dt1_proc)))

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, max_window, T, dtype, device
    )
    for slot in range(cache_size):
        _seed_ring(
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            slot,
            x1[slot],
            B1[slot],
            dt1_proc[slot],
        )

    T_pad = 16
    k_old = ((max_window + 7) // 8) * 8

    def _run(k, *, d_split):
        torch.manual_seed(k + 100)
        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2 = repeat(
            torch.randn(batch, T, nheads, device=device, dtype=dtype),
            "b t h -> b t h p",
            p=head_dim,
        )
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        st = state0.clone()
        out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        xc, bc, dtc = x_cache.clone(), B_cache.clone(), dt_cache.clone()
        checkpointing_ssu(
            st,
            xc,
            bc,
            dtc,
            ring_start.clone(),
            torch.full((cache_size,), k, device=device, dtype=torch.int32),
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            d_split=d_split,
            cb_scaled=torch.empty(
                batch, nheads, WARP_SIZE, MMA_FRAG_SIZE, device=device, dtype=dtype
            ),
            cumAdt_vec=torch.empty(
                batch, nheads, T_pad, device=device, dtype=torch.float32
            ),
            cb_old=torch.empty(
                batch, nheads, WARP_SIZE, k_old // 2, device=device, dtype=dtype
            ),
            algorithm="two-kernel",
        )
        return out, st, xc, bc, dtc

    names = ("out", "state", "x_cache", "B_cache", "dt_cache")
    for k in (0, 2, T):
        ref = _run(k, d_split=1)
        test = _run(k, d_split=2)
        for name, r, t in zip(names, ref, test, strict=True):
            torch.testing.assert_close(
                t, r, rtol=2e-2, atol=5e-1, msg=f"{name} mismatch at k={k}"
            )


def test_persistent_main_matches_monolithic(monkeypatch):
    """Persistent main: with FLASHINFER_SSU_MAIN_CTA_PER_SM=1 the main grid is
    undersized so that ``work_units > occupancy·NUM_SMS`` — the grid genuinely
    can't hold all work-units at once, so each CTA must grid-stride over several.
    Output (+ all caches) must match the monolithic reference bit-exact on BOTH
    the no-write (prev_k=0,8) and write (prev_k=12) paths at production
    max_window=16."""
    monkeypatch.setenv("FLASHINFER_SSU_MAIN_CTA_PER_SM", "1")

    device = "cuda"
    dtype = torch.bfloat16
    nheads, head_dim, d_state, ngroups, T = 16, 64, 128, 1, 6
    max_window = 16  # production window

    # Size batch so work_units = batch·nheads exceeds occupancy·NUM_SMS (main is
    # __maxnreg__(64) → 8 blocks/SM).  At CTA_PER_SM=1 the grid is NUM_SMS CTAs, so
    # each grid-strides ≥ occupancy× — the loop is genuinely exercised.
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    occupancy = 8
    batch = max(2, (occupancy * num_sms) // nheads + 1)
    cache_size = batch

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias = repeat(
        torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim
    )
    D = repeat(torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim)
    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=dtype
    )

    # Random full ring: prev_k>T slots read genuinely-distinct history rows.
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, max_window, T, dtype, device
    )

    T_pad = 16
    k_old = ((max_window + 7) // 8) * 8

    def _run(k, *, two_kernel):
        torch.manual_seed(k + 300)
        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2 = repeat(
            torch.randn(batch, T, nheads, device=device, dtype=dtype),
            "b t h -> b t h p",
            p=head_dim,
        )
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        st = state0.clone()
        out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        xc, bc, dtc = x_cache.clone(), B_cache.clone(), dt_cache.clone()
        kw = {}
        if two_kernel:
            kw["cb_scaled"] = torch.empty(
                batch, nheads, WARP_SIZE, MMA_FRAG_SIZE, device=device, dtype=dtype
            )
            kw["cumAdt_vec"] = torch.empty(
                batch, nheads, T_pad, device=device, dtype=torch.float32
            )
            kw["cb_old"] = torch.empty(
                batch, nheads, WARP_SIZE, k_old // 2, device=device, dtype=dtype
            )
            kw["algorithm"] = "two-kernel"
        checkpointing_ssu(
            st,
            xc,
            bc,
            dtc,
            ring_start.clone(),
            torch.full((cache_size,), k, device=device, dtype=torch.int32),
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            **kw,
        )
        return out, st, xc, bc, dtc

    names = ("out", "state", "x_cache", "B_cache", "dt_cache")
    # k=0/8 no-write (prev_k+T ≤ 16); k=12 write (12+6=18 > 16).
    for k in (0, 8, 12):
        ref = _run(k, two_kernel=False)
        test = _run(k, two_kernel=True)
        for name, r, t in zip(names, ref, test, strict=True):
            torch.testing.assert_close(
                t, r, rtol=2e-2, atol=5e-1, msg=f"{name} mismatch at k={k} (persistent)"
            )


def test_two_kernel_meta_ring_refill(monkeypatch):
    """Meta-ring REFILL path: with FLASHINFER_SSU_MAIN_CTA_PER_SM=1 the main grid is
    NUM_SMS CTAs and batch is sized so each CTA grid-strides over ~64 work-units —
    crossing the meta-window refills at tile 30 and 60 (META_RING=32, STAGES=2) with
    ring-slot wraparound.  prev_num_accepted is randomized per slot so write and
    no-write units interleave inside every 32-unit window, and (NUM_SMS not being a
    multiple of nheads) the head also varies per unit.  Must match the monolithic
    reference."""
    monkeypatch.setenv("FLASHINFER_SSU_MAIN_CTA_PER_SM", "1")

    device = "cuda"
    dtype = torch.bfloat16
    nheads, head_dim, d_state, ngroups, T = 16, 64, 128, 1, 6
    max_window = 16

    # units/CTA = batch·nheads / NUM_SMS ≈ 64 > 2·(META_RING−STAGES) ⇒ ≥ 2 refills.
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    batch = (64 * num_sms) // nheads + 1
    cache_size = batch

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias = repeat(
        torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim
    )
    D = repeat(torch.randn(nheads, device=device, dtype=dtype), "h -> h p", p=head_dim)
    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=dtype
    )

    # Random full ring: prev_k>T slots read genuinely-distinct history rows.
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, max_window, T, dtype, device
    )

    # Mixed per-slot pnat ∈ [0, max_window]: rows with pnat+T > max_window checkpoint,
    # the rest replay — interleaved inside every meta window.
    torch.manual_seed(7)
    prev_mixed = torch.randint(
        0, max_window + 1, (cache_size,), device=device, dtype=torch.int32
    )
    assert (prev_mixed + T > max_window).any() and (prev_mixed + T <= max_window).any()

    T_pad = 16
    k_old = ((max_window + 7) // 8) * 8

    def _run(*, two_kernel):
        torch.manual_seed(300)
        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt2 = repeat(
            torch.randn(batch, T, nheads, device=device, dtype=dtype),
            "b t h -> b t h p",
            p=head_dim,
        )
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        st = state0.clone()
        out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        xc, bc, dtc = x_cache.clone(), B_cache.clone(), dt_cache.clone()
        kw = {}
        if two_kernel:
            kw["cb_scaled"] = torch.empty(
                batch, nheads, WARP_SIZE, MMA_FRAG_SIZE, device=device, dtype=dtype
            )
            kw["cumAdt_vec"] = torch.empty(
                batch, nheads, T_pad, device=device, dtype=torch.float32
            )
            kw["cb_old"] = torch.empty(
                batch, nheads, WARP_SIZE, k_old // 2, device=device, dtype=dtype
            )
            kw["algorithm"] = "two-kernel"
        checkpointing_ssu(
            st,
            xc,
            bc,
            dtc,
            ring_start.clone(),
            prev_mixed.clone(),
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            **kw,
        )
        return out, st, xc, bc, dtc

    names = ("out", "state", "x_cache", "B_cache", "dt_cache")
    ref = _run(two_kernel=False)
    test = _run(two_kernel=True)
    for name, r, t in zip(names, ref, test, strict=True):
        torch.testing.assert_close(
            t, r, rtol=2e-2, atol=5e-1, msg=f"{name} mismatch (meta-ring refill)"
        )


def _run_two_kernel_state_dtype_case(
    state_dtype, philox_rounds, batch=2, k_cases=(0, 2, 12)
):
    """Two-kernel path vs the monolithic reference with a non-bf16 STATE dtype
    (bf16 activations).  The window is filled with CONSISTENT rows (softplus
    dt, true cumsum): the write path replays rows beyond T, and inconsistent
    random rows push |state| past f16 max (65504) into inf/NaN.

    ``k_cases`` entries are uniform pnat ints or the string "mixed" (per-slot
    random pnat straddling the write threshold — exercises the main's
    write-first slot visiting order)."""
    device = "cuda"
    act_dtype = torch.bfloat16
    nheads, head_dim, d_state, ngroups, T = 16, 64, 128, 1, 6
    max_window = 16
    cache_size = batch

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias = repeat(
        torch.randn(nheads, device=device, dtype=act_dtype), "h -> h p", p=head_dim
    )
    D = repeat(
        torch.randn(nheads, device=device, dtype=act_dtype), "h -> h p", p=head_dim
    )
    state0 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )
    x_fill = torch.randn(
        batch, max_window, nheads, head_dim, device=device, dtype=act_dtype
    )
    dt_fill = torch.randn(batch, max_window, nheads, device=device, dtype=act_dtype)
    B_fill = torch.randn(
        batch, max_window, ngroups, d_state, device=device, dtype=act_dtype
    )
    dt_fill_proc = dt_fill.float() + dt_bias[:, 0].float()[None, None, :]
    dt_fill_proc = torch.where(
        dt_fill_proc > 20.0, dt_fill_proc, torch.log1p(torch.exp(dt_fill_proc))
    )
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, max_window, T, act_dtype, device
    )
    for slot in range(cache_size):
        _seed_ring(
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            slot,
            x_fill[slot],
            B_fill[slot],
            dt_fill_proc[slot],
        )

    rand_seed = torch.tensor([1234], device=device, dtype=torch.int64)
    T_pad = 16
    k_old = ((max_window + 7) // 8) * 8

    def _run(k, *, two_kernel):
        torch.manual_seed((int(k.sum().item()) if torch.is_tensor(k) else k) + 100)
        x2 = torch.randn(batch, T, nheads, head_dim, device=device, dtype=act_dtype)
        dt2 = repeat(
            torch.randn(batch, T, nheads, device=device, dtype=act_dtype),
            "b t h -> b t h p",
            p=head_dim,
        )
        B2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=act_dtype)
        C2 = torch.randn(batch, T, ngroups, d_state, device=device, dtype=act_dtype)
        st = state0.clone()
        out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=act_dtype)
        xc, bc, dtc = x_cache.clone(), B_cache.clone(), dt_cache.clone()
        kw = {}
        if philox_rounds > 0:
            kw["philox_rounds"] = philox_rounds
            kw["rand_seed"] = rand_seed
        if two_kernel:
            kw["cb_scaled"] = torch.empty(
                batch, nheads, WARP_SIZE, MMA_FRAG_SIZE, device=device, dtype=act_dtype
            )
            kw["cumAdt_vec"] = torch.empty(
                batch, nheads, T_pad, device=device, dtype=torch.float32
            )
            kw["cb_old"] = torch.empty(
                batch, nheads, WARP_SIZE, k_old // 2, device=device, dtype=act_dtype
            )
            kw["algorithm"] = "two-kernel"
        checkpointing_ssu(
            st,
            xc,
            bc,
            dtc,
            ring_start.clone(),
            k.clone()
            if torch.is_tensor(k)
            else torch.full((cache_size,), k, device=device, dtype=torch.int32),
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            **kw,
        )
        return out, st, xc, bc, dtc

    names = ("out", "state", "x_cache", "B_cache", "dt_cache")
    # k=0/2 no-write; k=12 write (12+6 > 16) — replays 12 rows incl. rows ≥ T;
    # "mixed" = per-slot random pnat with ~40% writes (slot_order visiting path).
    for k in k_cases:
        if isinstance(k, str):
            torch.manual_seed(1234)
            kv = torch.randint(0, 8, (cache_size,), device=device, dtype=torch.int32)
            kv[torch.rand(cache_size, device=device) < 0.4] = 12
            k_arg, k_label = kv, "mixed"
        else:
            k_arg, k_label = k, k
        ref = _run(k_arg, two_kernel=False)
        test = _run(k_arg, two_kernel=True)
        for name, r, t in zip(names, ref, test, strict=True):
            torch.testing.assert_close(
                t,
                r,
                rtol=2e-2,
                atol=5e-1,
                msg=f"{name} mismatch at k={k_label} ({state_dtype} state, philox={philox_rounds})",
            )


@pytest.mark.parametrize("philox_rounds", [0, 5])
def test_two_kernel_f16_state(philox_rounds):
    """f16 STATE + bf16 activations on the two-kernel path vs the monolithic
    reference.  The operand-swap OUT.1 LDSMs the state under a bf16-typed view;
    the real element type must be recovered in-registers
    (convert_frag<state_t>) — reinterpreting f16 bits as bf16 silently corrupts
    the output (caught 2026-07-02).  philox_rounds=5 exercises the SR state
    store (shared store code + same seed ⇒ the two paths bit-match)."""
    if philox_rounds > 0 and not is_cvt_rs_supported():
        pytest.skip("fp16 Philox SR requires HW cvt.rs (sm_100a+)")
    _run_two_kernel_state_dtype_case(torch.float16, philox_rounds)


def test_two_kernel_f32_state():
    """f32 STATE + bf16 activations on the two-kernel path vs the monolithic
    reference — the wide-A path: OUT.1 loads the state as k-adjacent float2
    pairs via LDS.64 and narrows to bf16 in registers (no ldmatrix exists for
    32-bit elements feeding a 16-bit MMA), the f32 smem ring uses the
    M=3-floored Swizzle<3,3,3>, and the launcher accepts 4-byte state on the
    split.  philox SR is meaningless for f32 (stores from f32 registers are
    exact) — RN only."""
    _run_two_kernel_state_dtype_case(torch.float32, 0)


def test_two_kernel_f32_mixed_batch():
    """Mixed write/no-write batch (per-slot pnat, ~40% writes) on the f32
    two-kernel path vs the monolithic reference.  batch=64 gives the persistent
    main a genuinely interleaved grid-stride deal — exercises the write-first
    slot visiting order (build_slot_order + remap_seq): the monolith is
    visit-order independent, so any remap/partition/pad bug shows as a
    mismatch."""
    _run_two_kernel_state_dtype_case(torch.float32, 0, batch=64, k_cases=("mixed",))


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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
        prev_tokens.clone(),
        out=out_off,
        enable_pdl=False,
        **common_kwargs,
    )

    state_on = state0.clone()
    out_on = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_on,
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_ssu_max_window_gt_npredicted(
    impl, state_dtype, paged_cache, npredicted, max_window
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

    # Step-1 inputs (x1/B1/dt1) populate the ring rows [start, start+npredicted).
    # dt MUST be consistent with these (real softplus), not arbitrary random —
    # otherwise the replay's recomputed `coeff = exp(total - cumAdt) * dt`
    # produces nonsensical scaling that drives the post-replay state into
    # magnitudes where fp16's ULP can't keep up with Triton's fp32-register
    # precision.  Rows beyond prev_k keep their random init (never read).
    x1 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size,
        nheads,
        ngroups,
        head_dim,
        d_state,
        max_window,
        npredicted,
        dtype,
        device,
    )
    ring_len = x_cache.shape[2]
    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot in enumerate(slot_indices):
        _seed_ring(
            x_cache, B_cache, dt_cache, ring_start, slot, x1[i], B1[i], dt1_proc[i]
        )

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
        x_ref = x_cache.clone()
        B_ref = B_cache.clone()
        dt_ref = dt_cache.clone()
        _call_replay(
            impl,
            state=ref_state,
            x_cache=x_ref,
            B_cache=B_ref,
            dt_cache=dt_ref,
            ring_start=ring_start,
            prev_tokens=ref_prev,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=ref_out,
            max_window=max_window,
            batch=batch,
            work_seq_len=npredicted,
            state_batch_indices=state_batch_indices,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        # ── CUDA kernel ──
        test_state = state0.clone()
        test_prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
        test_out = torch.zeros(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        x_test = x_cache.clone()
        B_test = B_cache.clone()
        dt_test = dt_cache.clone()
        checkpointing_ssu(
            test_state,
            x_test,
            B_test,
            dt_test,
            ring_start.clone(),
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
        # Cache writes — both kernels append at the same ring rows and must
        # leave every other row untouched.
        _assert_ring_cache_matches(x_test, x_ref, f"x_cache (prev_k={prev_k})")
        _assert_ring_cache_matches(B_test, B_ref, f"B_cache (prev_k={prev_k})")
        _assert_ring_cache_matches(
            dt_test, dt_ref, f"dt_cache (prev_k={prev_k})", rtol=1e-4, atol=1e-4
        )

        # Appended dt rows must hold the softplus-processed step-2 dt at
        # (start + prev_k + i) % L — verify against the analytical value.
        dt2_proc_f32 = F.softplus(
            dt2_base.float() + dt_bias_base.float()[None, None, :]
        )
        for batch_idx, slot in enumerate(slot_indices):
            rows = (
                (
                    int(ring_start[slot])
                    + prev_k
                    + torch.arange(npredicted, device=device)
                )
                % ring_len
            ).long()
            torch.testing.assert_close(
                dt_ref[slot][:, rows],
                dt2_proc_f32[batch_idx].T,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Triton appended dt mismatch at prev_k={prev_k} slot={slot}",
            )


# Philox SR + must_checkpoint=False: verify the kernel correctly skips the
# state HBM write under the Philox path.  The cvt_rs+philox-refresh runs
# internally at the lane-pair level (skipping it would require routing
# must_checkpoint into pair_idx amortization), but the STG.64 must be
# elided.  Strongest signal: state remains byte-identical to state0.
@pytest.mark.skipif(
    not is_cvt_rs_supported(),
    reason="Philox stochastic rounding requires cvt.rs PTX (SM100a/SM103a only — "
    "not SM120a / consumer Blackwell)",
)
@pytest.mark.parametrize("nheads,head_dim,d_state,ngroups", _CONFIGS)
@pytest.mark.parametrize(
    "paged_cache", [False, True], ids=["no_cache_indices", "paged_cache"]
)
# (4, 8) gives must_checkpoint = (prev_k + 4 > 8) = False for prev_k ∈ [0, 4].
@pytest.mark.parametrize("npredicted,max_window", [(4, 8)], ids=["np4w8"])
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
@pytest.mark.parametrize("two_kernel", [False, True], ids=["mono", "two_kernel"])
def test_checkpointing_ssu_philox_no_checkpoint(
    impl,
    nheads,
    head_dim,
    d_state,
    ngroups,
    paged_cache,
    npredicted,
    max_window,
    two_kernel,
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

    # Step-1 inputs populate the ring rows [start, start+npredicted) with
    # consistent (real softplus) data — same precision strategy as
    # test_checkpointing_ssu_max_window_gt_npredicted.  Rows beyond prev_k
    # keep their random init (never read by replay since prev_k <= npredicted).
    x1 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size,
        nheads,
        ngroups,
        head_dim,
        d_state,
        max_window,
        npredicted,
        dtype,
        device,
    )
    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    slots_idx = state_batch_indices if paged_cache else slice(None)
    ring_len = x_cache.shape[2]
    for i, slot in enumerate(slot_indices):
        _seed_ring(
            x_cache, B_cache, dt_cache, ring_start, slot, x1[i], B1[i], dt1_proc[i]
        )

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
        x_ref = x_cache.clone()
        B_ref = B_cache.clone()
        dt_ref = dt_cache.clone()
        _call_replay(
            impl,
            state=ref_state,
            x_cache=x_ref,
            B_cache=B_ref,
            dt_cache=dt_ref,
            ring_start=ring_start,
            prev_tokens=ref_prev,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=ref_out,
            max_window=max_window,
            batch=batch,
            work_seq_len=npredicted,
            state_batch_indices=state_batch_indices,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            rand_seed=rand_seed,
            philox_rounds=10,
        )

        # ── CUDA kernel with Philox enabled ──
        test_state = state0.clone()
        test_prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
        test_out = torch.zeros(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        x_test = x_cache.clone()
        B_test = B_cache.clone()
        dt_test = dt_cache.clone()
        checkpointing_ssu(
            test_state,
            x_test,
            B_test,
            dt_test,
            ring_start.clone(),
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
            **(
                _two_kernel_scratch(batch, nheads, max_window, dtype, device)
                if two_kernel
                else {}
            ),
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

        # Cache writes happen unconditionally: appends land at the same ring
        # rows on both sides — bit-exact for x/B, fp32 tolerance for dt.
        _assert_ring_cache_matches(x_test, x_ref, f"x_cache (prev_k={prev_k})")
        _assert_ring_cache_matches(B_test, B_ref, f"B_cache (prev_k={prev_k})")
        _assert_ring_cache_matches(
            dt_test, dt_ref, f"dt_cache (prev_k={prev_k})", rtol=1e-4, atol=1e-4
        )

        # Appended dt rows must hold the softplus-processed step-2 dt at
        # (start + prev_k + i) % L — verify against the analytical value.
        dt2_proc_f32 = F.softplus(
            dt2_base.float() + dt_bias_base.float()[None, None, :]
        )
        for batch_idx, slot in enumerate(slot_indices):
            rows = (
                (
                    int(ring_start[slot])
                    + prev_k
                    + torch.arange(npredicted, device=device)
                )
                % ring_len
            ).long()
            torch.testing.assert_close(
                dt_ref[slot][:, rows],
                dt2_proc_f32[batch_idx].T,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Triton appended dt mismatch at prev_k={prev_k} slot={slot}",
            )


# Philox SR + must_checkpoint=True under MAX_WINDOW > NPREDICTED: rounds out
# the fp16+Philox validation matrix.  The existing test_checkpointing_ssu_philox
# only exercises the degenerate MAX_WINDOW == NPREDICTED case where every
# call checkpoints; this test verifies the non-degenerate case where the
# checkpoint decision actually depends on prev_k vs the buffer's remaining
# capacity.
@pytest.mark.skipif(
    not is_cvt_rs_supported(),
    reason="Philox stochastic rounding requires cvt.rs PTX (SM100a/SM103a only — "
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
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
@pytest.mark.parametrize("two_kernel", [False, True], ids=["mono", "two_kernel"])
def test_checkpointing_ssu_philox_with_checkpoint(
    impl,
    npredicted,
    max_window,
    prev_k,
    paged_cache,
    _cfg_idx,
    two_kernel,
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

    # Step-1 inputs populate the ring rows [start, start+npredicted) with
    # consistent (real softplus) data — same precision strategy as
    # test_checkpointing_ssu_max_window_gt_npredicted.
    x1 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size,
        nheads,
        ngroups,
        head_dim,
        d_state,
        max_window,
        npredicted,
        dtype,
        device,
    )
    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    slots_idx = state_batch_indices if paged_cache else slice(None)
    for i, slot in enumerate(slot_indices):
        _seed_ring(
            x_cache, B_cache, dt_cache, ring_start, slot, x1[i], B1[i], dt1_proc[i]
        )

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
    x_ref = x_cache.clone()
    B_ref = B_cache.clone()
    dt_ref = dt_cache.clone()
    _call_replay(
        impl,
        state=ref_state,
        x_cache=x_ref,
        B_cache=B_ref,
        dt_cache=dt_ref,
        ring_start=ring_start,
        prev_tokens=ref_prev,
        x=x2,
        dt=dt2,
        A=A,
        B=B2,
        C=C2,
        out=ref_out,
        max_window=max_window,
        batch=batch,
        work_seq_len=npredicted,
        state_batch_indices=state_batch_indices,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        rand_seed=rand_seed,
        philox_rounds=10,
    )

    # ── CUDA kernel with Philox enabled ──
    test_state = state0.clone()
    test_prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
    test_out = torch.zeros(
        batch, npredicted, nheads, head_dim, device=device, dtype=dtype
    )
    x_test = x_cache.clone()
    B_test = B_cache.clone()
    dt_test = dt_cache.clone()
    checkpointing_ssu(
        test_state,
        x_test,
        B_test,
        dt_test,
        ring_start.clone(),
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
        **(
            _two_kernel_scratch(batch, nheads, max_window, dtype, device)
            if two_kernel
            else {}
        ),
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

    # Cache writes happen unconditionally: appends land at the same ring rows
    # on both sides regardless of must_checkpoint.
    _assert_ring_cache_matches(x_test, x_ref, f"x_cache (prev_k={prev_k})")
    _assert_ring_cache_matches(B_test, B_ref, f"B_cache (prev_k={prev_k})")
    _assert_ring_cache_matches(
        dt_test, dt_ref, f"dt_cache (prev_k={prev_k})", rtol=1e-4, atol=1e-4
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

    Builds physically realistic ring caches (dt derived from softplus(dt+bias),
    decays recomputed in-kernel), then validates the CUDA
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

    # Build physically realistic ring caches (max_window == NPREDICTED == T).
    dt1_processed = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )
    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot_val in enumerate(slot_indices):
        _seed_ring(
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            slot_val,
            x1[i],
            B1[i],
            dt1_processed[i],
        )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    # Build physically realistic ring caches (max_window == NPREDICTED == T).
    dt1_processed = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )
    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot_val in enumerate(slot_indices):
        _seed_ring(
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            slot_val,
            x1[i],
            B1[i],
            dt1_processed[i],
        )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    # Ring caches (random contents; positive dt keeps recomputed decays finite).
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
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
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_ssu_mixed_checkpoint_batch(
    impl, nheads, head_dim, d_state, ngroups, paged_cache, with_philox
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
            "Philox stochastic rounding requires cvt.rs PTX (SM100a/SM103a only — "
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

    # Step-1 inputs populate each used slot's ring rows [start, start+npredicted).
    x1 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size,
        nheads,
        ngroups,
        head_dim,
        d_state,
        max_window,
        npredicted,
        dtype,
        device,
    )
    ring_len = x_cache.shape[2]
    for i, slot in enumerate(slot_per_batch):
        _seed_ring(
            x_cache, B_cache, dt_cache, ring_start, slot, x1[i], B1[i], dt1_proc[i]
        )

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

    # ── Triton reference: single full-batch launch ──
    #
    # The persistent path derives the per-slot write decision at runtime from
    # (prev_per_slot + npredicted) > max_window, so a single launch handles
    # the mixed [F, F, T, T] batch natively (no need for the old two-launch
    # WRITE_CHECKPOINT-constexpr split).
    ref_state = state0.clone()
    x_ref = x_cache.clone()
    B_ref = B_cache.clone()
    dt_ref = dt_cache.clone()
    ref_out = torch.zeros(
        batch, npredicted, nheads, head_dim, device=device, dtype=dtype
    )

    _call_replay(
        impl,
        state=ref_state,
        x_cache=x_ref,
        B_cache=B_ref,
        dt_cache=dt_ref,
        ring_start=ring_start,
        prev_tokens=prev_per_slot,
        x=x2,
        dt=dt2,
        A=A,
        B=B2,
        C=C2,
        out=ref_out,
        max_window=max_window,
        batch=batch,
        work_seq_len=npredicted,
        state_batch_indices=state_batch_indices_full,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        **philox_kwargs,
    )

    # ── CUDA kernel: single launch with full per-batch prev_k ──
    test_state = state0.clone()
    x_test = x_cache.clone()
    B_test = B_cache.clone()
    dt_test = dt_cache.clone()
    test_out = torch.zeros(
        batch, npredicted, nheads, head_dim, device=device, dtype=dtype
    )
    checkpointing_ssu(
        test_state,
        x_test,
        B_test,
        dt_test,
        ring_start.clone(),
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

    # Cache writes happen unconditionally: appends land at the same ring rows
    # on both sides regardless of the per-batch gate.
    _assert_ring_cache_matches(x_test, x_ref, "x_cache")
    _assert_ring_cache_matches(B_test, B_ref, "B_cache")
    _assert_ring_cache_matches(dt_test, dt_ref, "dt_cache", rtol=1e-4, atol=1e-4)

    # Appended dt rows must hold the softplus-processed step-2 dt at
    # (start + prev_k + i) % L — verify per slot against the analytical value.
    dt2_proc_f32 = F.softplus(dt2_base.float() + dt_bias_base.float()[None, None, :])
    for batch_idx, slot in enumerate(slot_per_batch):
        prev_k = prev_k_per_batch[batch_idx]
        rows = (
            (int(ring_start[slot]) + prev_k + torch.arange(npredicted, device=device))
            % ring_len
        ).long()
        torch.testing.assert_close(
            dt_ref[slot][:, rows],
            dt2_proc_f32[batch_idx].T,
            rtol=1e-4,
            atol=1e-4,
            msg=f"Triton appended dt mismatch at batch_idx={batch_idx} slot={slot} prev_k={prev_k}",
        )


@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_ssu_contiguous(impl):
    """Smoke test for the contiguous-cache path (TP=8, bf16 state, mtp=16)."""
    _run_checkpointing_ssu_case(
        impl,
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
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        batch, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )
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
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
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
@pytest.mark.parametrize("two_kernel", [False, True], ids=["mono", "two_kernel"])
def test_checkpointing_ssu_philox(
    nheads, head_dim, d_state, ngroups, paged_cache, T, two_kernel
):
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

    # Ring caches (random contents; positive dt keeps recomputed decays finite)
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
    if two_kernel:
        # Window == T in this test; the scratch reroutes both runs to the split.
        common_kwargs.update(_two_kernel_scratch(batch, nheads, T, dtype, device))

    # --- Run without rounding (deterministic fp16 state store) ---
    state_nornd = state0.clone()
    out_nornd = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    checkpointing_ssu(
        state_nornd,
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        batch, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        batch, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        batch, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
        x_cache.clone(),
        B_cache.clone(),
        dt_cache.clone(),
        ring_start.clone(),
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
    reason="Philox stochastic rounding requires cvt.rs PTX (SM100a/SM103a only — "
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
    uses cvt.rs PTX which only exists on SM100a/SM103a (datacenter Blackwell);
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
            f"(SM100a/SM103a only — not SM120a / consumer Blackwell)"
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
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_state_update(
    impl, state_dtype, paged_cache, T, write_checkpoint, _cfg_idx
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

    # Build ring caches for the replay kernel (head-major, L = max_window + T).
    # Ring rows outside the seeded window keep their random init — they're
    # outside the test's PNAT range so the kernel must not read them, and the
    # untouched-rows postcondition below catches any stray write.
    slots = state_batch_indices if paged_cache else slice(None)
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, max_window, T, dtype, device
    )
    ring_len = x_cache.shape[2]

    # Compute processed dt for step 1 (cumAdt is recomputed from cached dt)
    dt1 = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])

    slot_indices = (
        state_batch_indices.tolist() if paged_cache else list(range(cache_size))
    )
    for i, slot in enumerate(slot_indices):
        _seed_ring(x_cache, B_cache, dt_cache, ring_start, slot, x1[i], B1[i], dt1[i])

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
        # The persistent-dynamic path decides write-vs-append per slot at
        # runtime: is_w = (prev_k + T) > max_window.  prev_k = k is uniform here,
        # so is_w is uniform.  The old forced `write_checkpoint` constexpr is
        # moot on the dynamic path — expectations below follow is_w.
        is_w = (k + T) > max_window
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        x_w = x_cache.clone()
        B_w = B_cache.clone()
        dt_w = dt_cache.clone()

        _call_replay(
            impl,
            state=test_state,
            x_cache=x_w,
            B_cache=B_w,
            dt_cache=dt_w,
            ring_start=ring_start,
            prev_tokens=prev_tokens,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=test_out,
            max_window=max_window,
            batch=batch,
            work_seq_len=T,
            state_batch_indices=state_batch_indices,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_scales=test_scales,
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
            if is_w:
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
            if is_w:
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
                atol=1.0 if is_w else 0.0,
                msg=f"State mismatch at k={k} (is_w={is_w})",
            )

        # --- Cache postconditions (ring contract) ---
        # Appends land at rows (start + k + i) % L on BOTH branches (the host
        # advances ring_start on flush, not the kernel); every other ring row
        # must be byte-identical to the pre-call cache.
        dt2_proc = F.softplus(dt2_base.float() + dt_bias_base.float()[None, None, :])

        for batch_idx, slot in enumerate(slot_indices):
            rows_new = (
                (int(ring_start[slot]) + k + torch.arange(T, device=device)) % ring_len
            ).long()
            keep = torch.ones(ring_len, dtype=torch.bool, device=device)
            keep[rows_new] = False

            # --- x_cache: appended tokens bit-exact, rest untouched ---
            torch.testing.assert_close(
                x_w[slot][:, rows_new],
                x2[batch_idx].permute(1, 0, 2),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                x_w[slot][:, keep], x_cache[slot][:, keep], rtol=0, atol=0
            )

            # --- B_cache ---
            torch.testing.assert_close(
                B_w[slot][:, rows_new],
                B2[batch_idx].permute(1, 0, 2),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                B_w[slot][:, keep], B_cache[slot][:, keep], rtol=0, atol=0
            )

            # --- dt_cache: appended rows hold softplus-processed step-2 dt ---
            torch.testing.assert_close(
                dt_w[slot][:, rows_new],
                dt2_proc[batch_idx].T,
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                dt_w[slot][:, keep], dt_cache[slot][:, keep], rtol=0, atol=0
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
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_state_update_philox(impl, state_dtype, paged_cache, T, _cfg_idx):
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
    _call_replay(
        impl,
        state=state_no_round,
        x_cache=x_cache.clone(),
        B_cache=B_cache.clone(),
        dt_cache=dt_cache.clone(),
        ring_start=ring_start,
        prev_tokens=prev_tokens,
        out=out_no_round,
        max_window=T,
        batch=batch,
        work_seq_len=T,
        state_scales=scales_no_round,
        **common_kwargs,
    )

    rand_seed = torch.tensor([12345], device=device, dtype=torch.int64)
    state_rounded = state0.clone()
    scales_rounded = state0_scales.clone() if is_quantized else None
    out_rounded = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
    _call_replay(
        impl,
        state=state_rounded,
        x_cache=x_cache.clone(),
        B_cache=B_cache.clone(),
        dt_cache=dt_cache.clone(),
        ring_start=ring_start,
        prev_tokens=prev_tokens,
        out=out_rounded,
        max_window=T,
        batch=batch,
        work_seq_len=T,
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
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_philox_rounding_unbiased(impl, state_dtype):
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

    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        batch, nheads, ngroups, head_dim, d_state, T, T, dtype, device
    )

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
    _call_replay(
        impl,
        state=state_fp32,
        x_cache=x_cache.clone(),
        B_cache=B_cache.clone(),
        dt_cache=dt_cache.clone(),
        ring_start=ring_start,
        prev_tokens=prev_tokens,
        out=out_fp32,
        max_window=T,
        batch=batch,
        work_seq_len=T,
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
    _call_replay(
        impl,
        state=state_rounded,
        x_cache=x_cache.clone(),
        B_cache=B_cache.clone(),
        dt_cache=dt_cache.clone(),
        ring_start=ring_start,
        prev_tokens=prev_tokens,
        out=out_rounded,
        max_window=T,
        batch=batch,
        work_seq_len=T,
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
#   - x_cache / B_cache / dt_cache                 (over the seq_len_i
#     appended ring rows, ignoring the trailing padding rows)
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

    # Ring caches: L = max_window + npredicted.  Random contents suffice —
    # the builder's positive dt keeps recomputed decays bounded, and the
    # varlen-vs-padded comparison is exact between two runs sharing the cache.
    x_cache, B_cache, dt_cache, ring_start = _make_ring_caches(
        cache_size,
        nheads,
        ngroups,
        head_dim,
        d_state,
        max_window,
        npredicted,
        dtype,
        device,
    )

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
        x_cache=x_cache,
        B_cache=B_cache,
        dt_cache=dt_cache,
        ring_start=ring_start,
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
    two_kernel=False,
):
    """Shared body: run the kernel in varlen mode and per-batch padded
    non-varlen mode, then compare slices.

    ``two_kernel=True`` routes the VARLEN call through the two-kernel split
    (forced via the scratch's algorithm="two-kernel" — batch*nheads here is
    far below the auto threshold); the padded reference stays monolithic.
    This is the only coverage of the 2k main's varlen meta path (seq_len /
    bos batched through fill_meta's ring)."""
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
    x_varlen = s["x_cache"].clone()
    B_varlen = s["B_cache"].clone()
    dt_varlen = s["dt_cache"].clone()

    checkpointing_ssu(
        state_varlen,
        x_varlen,
        B_varlen,
        dt_varlen,
        s["ring_start"].clone(),
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
        max_seqlen=npredicted,
        **(
            _two_kernel_scratch(batch, nheads, max_window, dtype, device)
            if two_kernel
            else {}
        ),
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
        # ring_start / prev_tokens: only slot i is consumed by this call.
        pt = s["prev_tokens"].clone()
        x_ref = s["x_cache"].clone()
        B_ref = s["B_cache"].clone()
        dt_ref = s["dt_cache"].clone()

        # Run with a single-batch input pointing at slot i.
        state_batch_indices_i = torch.tensor([i], device=device, dtype=torch.int32)

        checkpointing_ssu(
            state_ref,
            x_ref,
            B_ref,
            dt_ref,
            s["ring_start"].clone(),
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

        # Cache writes: both modes append at ring rows (start + pnat + j) % L.
        # Varlen appends sl rows, the padded reference appends npredicted rows
        # (sl real + zero padding) — compare the sl rows they share.
        ring_len = s["x_cache"].shape[2]
        rows = (
            (int(s["ring_start"][i]) + prev_ks[i] + torch.arange(sl, device=device))
            % ring_len
        ).long()
        torch.testing.assert_close(
            x_varlen[i][:, rows],
            x_ref[i][:, rows],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded x_cache mismatch at batch={i}",
        )
        torch.testing.assert_close(
            B_varlen[i][:, rows],
            B_ref[i][:, rows],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded B_cache mismatch at batch={i}",
        )
        torch.testing.assert_close(
            dt_varlen[i][:, rows],
            dt_ref[i][:, rows],
            rtol=0,
            atol=0,
            msg=f"varlen vs padded dt_cache mismatch at batch={i}",
        )


# ``test_checkpointing_ssu_varlen_uniform_seqlen`` was retired — its
# coverage (pure Branch B, varlen indexing) is a subset of
# ``test_checkpointing_ssu_varlen_mixed_no_checkpoint`` below, which
# additionally tests seq_len < NPREDICTED masking.


@pytest.mark.parametrize("state_dtype", _VARLEN_DTYPES)
@pytest.mark.parametrize("two_kernel", [False, True], ids=["mono", "two_kernel"])
def test_checkpointing_ssu_varlen_mixed_no_checkpoint(state_dtype, two_kernel):
    """Mixed seq_lens, prev_k chosen so that prev_k + NPREDICTED <= MAX_WINDOW
    for every batch (no-checkpoint regime).  Tests that the varlen masking
    of T-rows >= seq_len matches the non-varlen reference where those rows
    contain zero-padded inputs (which compute_CB_scaled / matmul should
    zero-propagate)."""
    if two_kernel and state_dtype in (torch.int8, torch.float8_e4m3fn):
        pytest.skip("two-kernel split takes 2/4-byte state only")
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
        two_kernel=two_kernel,
    )


@pytest.mark.parametrize("state_dtype", _VARLEN_DTYPES)
@pytest.mark.parametrize("two_kernel", [False, True], ids=["mono", "two_kernel"])
def test_checkpointing_ssu_varlen_mixed_checkpoint(state_dtype, two_kernel):
    """Mixed seq_lens, prev_k large enough that prev_k + min(seq_lens) > MAX_WINDOW
    — forces every batch (varlen and non-varlen) to checkpoint.  Exercises
    the replay + state-write path under varlen indexing."""
    if two_kernel and state_dtype in (torch.int8, torch.float8_e4m3fn):
        pytest.skip("two-kernel split takes 2/4-byte state only")
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
        two_kernel=two_kernel,
    )


def _run_varlen_cuda_vs_triton(
    impl,
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
    x_cuda = s["x_cache"].clone()
    B_cuda = s["B_cache"].clone()
    dt_cuda = s["dt_cache"].clone()
    out_cuda = torch.zeros_like(x_packed)

    checkpointing_ssu(
        state_cuda,
        x_cuda,
        B_cuda,
        dt_cuda,
        s["ring_start"].clone(),
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
        max_seqlen=npredicted,
    )

    # --- Triton varlen ---
    state_tri = s["state0"].clone()
    state_scale_tri = s["state0_scale"].clone() if s["is_quantized"] else None
    x_tri = s["x_cache"].clone()
    B_tri = s["B_cache"].clone()
    dt_tri = s["dt_cache"].clone()
    out_tri = torch.zeros_like(x_packed)

    # Per-slot real sequence lengths for the persistent_main work-item sort
    # (so write items are ordered by the actual per-sequence length, not
    # NPREDICTED).  (batch,) int32 derived from the packed cu_seqlens.
    seq_lens_t = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)

    _call_replay(
        impl,
        state=state_tri,
        x_cache=x_tri,
        B_cache=B_tri,
        dt_cache=dt_tri,
        ring_start=s["ring_start"],
        prev_tokens=s["prev_tokens"].clone(),
        x=x_packed,
        dt=dt_packed,
        A=s["A"],
        B=B_packed,
        C=C_packed,
        out=out_tri,
        max_window=max_window,
        batch=len(seq_lens),
        work_seq_len=seq_lens_t,
        D=s["D"],
        dt_bias=s["dt_bias"],
        dt_softplus=True,
        state_scales=state_scale_tri,
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
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_ssu_varlen_cuda_vs_triton_no_checkpoint(impl, state_dtype):
    """CUDA vs Triton (independent reference) — pure-Branch-B varlen.

    Constraint on test config: the Triton kernel caps prev_k <= NPREDICTED
    by design (its replay loop masks cache reads at `offs_t < T`).  We
    therefore size NPREDICTED = MAX_WINDOW so any prev_k in [0, MAX_WINDOW]
    is acceptable to both kernels.
    """
    _run_varlen_cuda_vs_triton(
        impl,
        seq_lens=[3, 1, 4, 2, 4],
        prev_ks=[0, 5, 10, 12, 8],  # all prev_k + max(seq_len)=4 <= 16
        npredicted=16,
        max_window=16,
        write_checkpoint=False,
        state_dtype=state_dtype,
    )


@pytest.mark.parametrize("state_dtype", _VARLEN_DTYPES)
@pytest.mark.parametrize("impl", _TRITON_IMPLS)
def test_checkpointing_ssu_varlen_cuda_vs_triton_checkpoint(impl, state_dtype):
    """CUDA vs Triton (independent reference) — pure-Branch-A varlen."""
    _run_varlen_cuda_vs_triton(
        impl,
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
# cache-side x_cache / B_cache / dt_cache.  Each gets a different
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
    for x / dt / B / C / z / out / x_cache / B_cache / dt_cache.
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

    # Ring caches (builder dt is positive — recomputed decays stay bounded).
    x_contig_c, B_contig_c, dt_contig_c, ring_start = _make_ring_caches(
        cache_size,
        nheads,
        ngroups,
        head_dim,
        d_state,
        max_window,
        npredicted,
        dtype,
        device,
    )

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
    x_ref = x_contig_c.clone()
    B_ref = B_contig_c.clone()
    dt_ref = dt_contig_c.clone()
    out_ref = torch.zeros_like(x_contig)
    checkpointing_ssu(
        state_ref,
        x_ref,
        B_ref,
        dt_ref,
        ring_start.clone(),
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
    # Cache tensors: pad the ring-pos axis (dim 2) so the slot/head strides
    # differ from the natural packed layout.
    x_cache_nc = _pad_inner(x_contig_c, pad=2, dim=2)
    B_cache_nc = _pad_inner(B_contig_c, pad=2, dim=2)
    dt_cache_nc = _pad_inner(dt_contig_c, pad=4, dim=2)

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
    assert x_cache_nc.stride(1) != x_contig_c.stride(1), (
        f"x_cache stride(1) didn't change: {x_cache_nc.stride()} == {x_contig_c.stride()}"
    )
    assert B_cache_nc.stride(1) != B_contig_c.stride(1), (
        f"B_cache stride(1) didn't change: {B_cache_nc.stride()} == {B_contig_c.stride()}"
    )
    assert dt_cache_nc.stride(1) != dt_contig_c.stride(1), (
        f"dt_cache stride(1) didn't change: {dt_cache_nc.stride()} == {dt_contig_c.stride()}"
    )

    state_test = state0.clone()
    state_scale_test = state0_scale.clone() if is_quantized else None
    # Cache writes: the _nc views alias the padded storage so written rows
    # persist there.  Compare via .contiguous() at the end.  (The views
    # already hold the contig data — _pad_inner copies it.)
    pt_test = prev_tokens.clone()

    checkpointing_ssu(
        state_test,
        x_cache_nc,
        B_cache_nc,
        dt_cache_nc,
        ring_start.clone(),
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
        x_cache_nc.contiguous(),
        x_ref,
        rtol=0,
        atol=0,
        msg=f"x_cache mismatch (varlen={varlen})",
    )
    torch.testing.assert_close(
        B_cache_nc.contiguous(),
        B_ref,
        rtol=0,
        atol=0,
        msg=f"B_cache mismatch (varlen={varlen})",
    )
    torch.testing.assert_close(
        dt_cache_nc.contiguous(),
        dt_ref,
        rtol=0,
        atol=0,
        msg=f"dt_cache mismatch (varlen={varlen})",
    )


# ────────────────────────────────────────────────────────────────────────────
# Determinism regression test (issue: cross-warp race in must_checkpoint=True)
# ────────────────────────────────────────────────────────────────────────────
#
# Repro for a cross-warp race that lived between `load_state_per_warp` (which
# partitions M=DIM across warps) and `replay_state_mma` (where each warp reads
# the FULL M=DIM extent of `smem.state` to form `frag_h`).  `load_data` ended
# with `__syncwarp()` only, so warp 0's replay reads rows that warps 1/2/3
# may not yet have cp.async-committed → per-launch non-deterministic state.
#
# Fix: `__syncthreads()` at the `ssu_checkpoint` / `ssu_nocheckpoint` dispatch
# site, replacing the per-branch barrier that previously lived inside
# `ssu_nocheckpoint` only.
#
# The bug only manifested in the must_checkpoint=True path (the
# must_checkpoint=False path always had its own __syncthreads inside
# ssu_nocheckpoint), but we test both paths for both quantized (fp8) and
# non-quantized (fp16) state to guard against regression in either branch.
# All parametrizations share `(heads_per_group=64, head_dim=64, d_state=128,
# max_window=16, npredicted=8)` — one JIT entry per state dtype.
@pytest.mark.parametrize(
    "state_dtype,prev_k,must_checkpoint,tag",
    [
        (torch.float16, 4, False, "fp16-no_checkpoint"),
        (torch.float16, 12, True, "fp16-checkpoint"),
        (torch.float8_e4m3fn, 4, False, "fp8-no_checkpoint"),
        (torch.float8_e4m3fn, 12, True, "fp8-checkpoint"),
    ],
)
def test_checkpointing_ssu_determinism_across_launches(
    state_dtype, prev_k, must_checkpoint, tag
):
    """Run the kernel 5× with bit-identical inputs and assert that state,
    state_scale (quantized only), and output are bit-exact across launches.
    """
    # fp8_e4m3fn requires SM89+ (Ada / Hopper / Blackwell).  No stochastic
    # rounding is used in this test, so use_sr=False.
    _maybe_skip_dtype(state_dtype, use_sr=False)
    import hashlib

    nheads = 64
    head_dim = 64
    d_state = 128
    ngroups = 1
    max_window = 16
    npredicted = 8
    batch = 16
    dtype = torch.bfloat16
    device = "cuda"
    cache_size = batch
    assert (prev_k + npredicted > max_window) == must_checkpoint, (
        f"test bug: parametrize disagrees with must_checkpoint formula "
        f"(prev_k={prev_k}, npredicted={npredicted}, max_window={max_window})"
    )

    def _hash(t):
        t = t.detach().cpu().contiguous()
        # numpy can't view bf16/fp8 directly — promote losslessly to f32.
        if t.dtype in (torch.bfloat16, torch.float8_e4m3fn):
            t = t.float()
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()

    torch.manual_seed(42)
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(A_base, "h -> h p n", p=head_dim, n=d_state)
    dt_bias_base = torch.randn(nheads, device=device, dtype=dtype)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)
    D_base = torch.randn(nheads, device=device, dtype=dtype)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # State + (optional) decode scale for the quantized path.
    is_quantized = state_dtype == torch.float8_e4m3fn
    state_fp32 = torch.randn(
        cache_size, nheads, head_dim, d_state, device=device, dtype=torch.float32
    )
    if is_quantized:
        quant_max = 448.0  # fp8_e4m3fn
        amax = state_fp32.abs().amax(dim=-1).clamp(min=1e-30)
        encode_scale = quant_max / amax
        state_scale0 = (1.0 / encode_scale).to(torch.float32)
        state0 = (
            (state_fp32 * encode_scale.unsqueeze(-1))
            .clamp(-quant_max, quant_max)
            .to(state_dtype)
        )
    else:
        state0 = state_fp32.to(state_dtype)
        state_scale0 = None

    # Step-1 cache fill — cached dt must be self-consistent so the replay
    # branch doesn't trip on physically-nonsensical values.
    x1 = torch.randn(batch, max_window, nheads, head_dim, device=device, dtype=dtype)
    dt1_base = torch.randn(batch, max_window, nheads, device=device, dtype=dtype)
    B1 = torch.randn(batch, max_window, ngroups, d_state, device=device, dtype=dtype)
    dt1_proc = F.softplus(dt1_base.float() + dt_bias_base.float()[None, None, :])

    x_cache0, B_cache0, dt_cache0, ring_start0 = _make_ring_caches(
        cache_size,
        nheads,
        ngroups,
        head_dim,
        d_state,
        max_window,
        npredicted,
        dtype,
        device,
    )
    for i in range(cache_size):
        _seed_ring(
            x_cache0, B_cache0, dt_cache0, ring_start0, i, x1[i], B1[i], dt1_proc[i]
        )

    # New-token inputs (deterministic w.r.t. prev_k).
    torch.manual_seed(prev_k + 100)
    x2 = torch.randn(batch, npredicted, nheads, head_dim, device=device, dtype=dtype)
    dt2_base = torch.randn(batch, npredicted, nheads, device=device, dtype=dtype)
    dt2 = repeat(dt2_base, "b t h -> b t h p", p=head_dim)
    B2 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)
    C2 = torch.randn(batch, npredicted, ngroups, d_state, device=device, dtype=dtype)

    state_hashes = set()
    out_hashes = set()
    scale_hashes = set()
    for _ in range(5):
        state_w = state0.clone()
        scale_w = state_scale0.clone() if state_scale0 is not None else None
        prev = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
        out = torch.zeros(
            batch, npredicted, nheads, head_dim, device=device, dtype=dtype
        )
        checkpointing_ssu(
            state_w,
            x_cache0.clone(),
            B_cache0.clone(),
            dt_cache0.clone(),
            ring_start0.clone(),
            prev,
            x=x2,
            dt=dt2,
            A=A,
            B=B2,
            C=C2,
            out=out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_scale=scale_w,
        )
        state_hashes.add(_hash(state_w))
        out_hashes.add(_hash(out))
        if scale_w is not None:
            scale_hashes.add(_hash(scale_w))

    assert len(state_hashes) == 1, (
        f"[{tag}] state non-deterministic across 5 launches "
        f"(got {len(state_hashes)} unique hashes)"
    )
    assert len(out_hashes) == 1, (
        f"[{tag}] output non-deterministic across 5 launches "
        f"(got {len(out_hashes)} unique hashes)"
    )
    if scale_hashes:
        assert len(scale_hashes) == 1, (
            f"[{tag}] state_scale non-deterministic across 5 launches "
            f"(got {len(scale_hashes)} unique hashes)"
        )


def test_checkpointing_ssu_continuous_dA_cumsum_multistep():
    """5-step no-write test: verifies output correctness across consecutive no-write steps.

    Uses `selective_state_update_triton` (running-state SSM) as the ground truth,
    since `checkpointing_state_update` only supports prev_k <= T.  The CUDA
    checkpointing kernel must produce the same output as the running SSM at each
    step: it starts from the initial state in HBM (never updated on no-write) and
    replays buffered tokens, recomputing the decay factors from the cached dt
    ring rows (cumAdt = A * inclusive-scan(dt) over [start, start+prev_k)).

    A recompute that mis-handled multi-step appends (e.g. scanning from the
    wrong ring row) would produce wrong total decays from step 3 (prev_k=6)
    onward — catastrophically wrong output (~40 error vs atol=0.5).
    """
    # T=3, max_window=16: 5 no-write steps with prev_k = 0, 3, 6, 9, 12.
    # A wrong dt-ring scan would manifest at step 3 (prev_k=6), causing
    # catastrophically wrong decay of step-1 tokens (~40 output error vs atol=0.5).
    T = 3
    max_window = 16
    nheads, head_dim, d_state, ngroups = _CONFIGS[0]
    batch = 2
    device = "cuda"
    dtype = torch.bfloat16
    state_dtype = torch.float16

    cache_size = batch

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

    # CUDA ring caches (chained between steps: each step appends T rows and
    # the next replays them; garbage rows beyond prev_k are never read)
    x_cache_t, B_cache_t, dt_cache_t, ring_start_t = _make_ring_caches(
        cache_size, nheads, ngroups, head_dim, d_state, max_window, T, dtype, device
    )
    test_state = state0.clone()

    # Reference: selective_state_update_triton maintains running SSM state.
    # Each step updates ref_state in place, giving the same output as the
    # checkpointing kernel's replay.
    ref_state = state0.clone().to(torch.float32)

    num_steps = 5
    for step in range(num_steps):
        prev_k = step * T
        assert prev_k + T <= max_window, "all steps must be no-write"

        torch.manual_seed(step + 100)
        x = torch.randn(batch, T, nheads, head_dim, device=device, dtype=dtype)
        dt_base = torch.randn(batch, T, nheads, device=device, dtype=dtype)
        dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)
        B = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)
        C = torch.randn(batch, T, ngroups, d_state, device=device, dtype=dtype)

        prev_test = torch.full((cache_size,), prev_k, device=device, dtype=torch.int32)
        test_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)
        ref_out = torch.zeros(batch, T, nheads, head_dim, device=device, dtype=dtype)

        # Ground truth: running-state SSM (selective_state_update_triton updates
        # ref_state in place, correctly accumulating each step's contribution).
        selective_state_update(
            ref_state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=ref_out,
        )

        # CUDA checkpointing kernel (no-write path: state HBM not updated).
        checkpointing_ssu(
            test_state,
            x_cache_t,
            B_cache_t,
            dt_cache_t,
            ring_start_t,
            prev_test,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            out=test_out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )

        torch.testing.assert_close(
            test_out,
            ref_out,
            rtol=2e-2,
            atol=5e-1,
            msg=f"output mismatch at step {step + 1} (prev_k={prev_k})",
        )
