"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---------------------------------------------------------------------------
Correctness tests for the GDN prefix-cache materialization kernel
(flashinfer/gdn_kernels/gdn_prefix_materialize.py).

HOW THIS TESTS, IN PLAIN WORDS
  The kernel replays a chosen number of ring entries onto a checkpoint and
  writes the answer to a different pool slot. The same answer is computed
  twice:

    1. with the KERNEL (CuTe-DSL, tensor cores, bf16/fp16 storage), and
    2. with `materialize_ref` - a slow, obviously-correct fp32 PyTorch
       function in reference_gdn_materialize.py.

  That reference is not taken on faith either: it is itself cross-checked
  against `_ref_fp32` in test_decode_ucache.py (the trusted token-at-a-time
  GDN oracle) by test_reference_agrees_with_decode_oracle below. With an fp32
  state pool the two agree BITWISE, so the chain is
  decode-oracle -> materialize_ref -> kernel with no unverified link.

  The interesting cases are the ones the decode kernel's fold never exercises:
    - PARTIAL replay (count < hist_len) -- the whole point of the feature
    - count == 0 -- must be an exact byte copy, not merely close
    - a DIFFERENT destination slot, with the source left untouched
    - wrapped ring windows at every base alignment
    - skip semantics (negative slots / count, over-long count)
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush import W_RING

# Both decode rings, exercised by every test: 32 is the MTP/spec-decode ring
# (gdn_decode_bf16_wy_ucache_flush.py), 16 is the STP ring (..._stp.py). They
# differ only in depth, and the kernel takes that at runtime -- so the pair
# below is what proves one compiled kernel really does serve both.
RING_DEPTHS = (32, 16)
from flashinfer.gdn_kernels.gdn_prefix_materialize import (
    gdn_prefix_materialize,
    state_ty_torch,
)

try:
    from .reference_gdn_materialize import materialize_ref
except ImportError:
    # For direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from reference_gdn_materialize import materialize_ref

DEV = "cuda"
H, HV, K, V = 16, 64, 128, 128
STATE_DTYPE = state_ty_torch()
RING_DTYPE = STATE_DTYPE
# The fold accumulates `count` outer products in f32 and rounds once to the
# state dtype; 2e-2 matches STATE_TOL in test_decode_ucache.py.
STATE_TOL = 2e-2


def _skip_if_not_sm90_or_later():
    # Allow-list, matching test_decode_ucache: a hypothetical SM13x should be
    # re-validated, not silently assumed compatible.
    from flashinfer.utils import get_compute_capability

    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip("GDN ucache kernels target SM90+")


def _make_pool(n_slots, hist, bases, seed, ring_slots=32, state_dtype=STATE_DTYPE):
    """Build a state pool plus consistent rings for `len(hist)` requests.

    Requests occupy slots [0, B); the remaining slots are zeroed destinations.
    Ring rows are written at PHYSICAL row (base + j) % ring_slots so the
    wrapped-window addressing is exercised for real.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    B = len(hist)
    state = (torch.randn(n_slots, HV, V, K, generator=g, device=DEV) * 0.5).to(
        state_dtype
    )
    kc = torch.zeros(n_slots, H, ring_slots, K, dtype=RING_DTYPE, device=DEV)
    uc = torch.zeros(n_slots, HV, ring_slots, V, dtype=RING_DTYPE, device=DEV)
    gc = torch.zeros(n_slots, HV, ring_slots, dtype=torch.float32, device=DEV)
    for r in range(B):
        P = hist[r]
        if P == 0:
            continue
        rows = torch.tensor(
            [(bases[r] + j) % ring_slots for j in range(P)],
            dtype=torch.long,
            device=DEV,
        )
        kh = torch.randn(H, P, K, generator=g, device=DEV)
        kc[r, :, rows] = F.normalize(kh, dim=-1).to(RING_DTYPE)
        uc[r, :, rows] = (torch.randn(HV, P, V, generator=g, device=DEV) * 0.3).to(
            RING_DTYPE
        )
        # Strictly decreasing cumulative log-decay, as the decode kernel writes.
        la = -(torch.rand(HV, P, generator=g, device=DEV) * 0.3 + 0.003)
        gc[r, :, rows] = torch.cumsum(la, dim=-1)
    return state, kc, uc, gc


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEV)


def _run_both(state, kc, uc, gc, src, dst, base, count):
    """Run reference and kernel on independent copies; return (ref, got)."""
    ref_state = state.clone()
    materialize_ref(
        [ref_state], [kc], [uc], [gc], src[None, :], dst[None, :], base, count
    )
    got_state = state.clone()
    gdn_prefix_materialize(got_state, src, dst, kc, uc, gc, base, count)
    torch.cuda.synchronize()
    return ref_state, got_state


def _assert_close(ref_state, got_state, slots, tol=STATE_TOL):
    for s in slots:
        err = (got_state[s].float() - ref_state[s].float()).abs().max().item()
        assert err < tol, f"slot {s}: absmax {err:.3e} exceeds {tol}"


# ---------------------------------------------------------------------------
# The reference itself, checked against the trusted decode oracle.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
def test_reference_agrees_with_decode_oracle(ring_slots):
    """materialize_ref == step 1 of test_decode_ucache._ref_fp32, bitwise at fp32.

    This is the anchor for every other test in the file: it shows the oracle
    used below is the same function the decode kernel is already judged
    against, evaluated at a caller-chosen prefix instead of the full window.

    Inputs are built here rather than with that module's _make_case (whose
    rings are fixed at its own depth); _ref_fp32's step-1 state does not
    depend on q/k/v/a/b at all, so those arguments are zeros shaped only to
    satisfy its token loop.
    """
    try:
        from .test_decode_ucache import _ref_fp32
        from .test_decode_ucache import T as ORACLE_T
    except ImportError:
        from test_decode_ucache import _ref_fp32
        from test_decode_ucache import T as ORACLE_T

    def _logical(t, base, n):
        rows = torch.tensor(
            [(base + j) % ring_slots for j in range(n)],
            dtype=torch.long,
            device=DEV,
        )
        return t.index_select(1, rows)

    B = 4
    hist = [13, 12, 15, 7]
    bases = [11, 5, 0, 9]
    counts = [13, 5, 0, 7]  # full, partial, exact-copy, full
    # fp32 state pool: with no storage rounding the two references must agree
    # BITWISE, so any real formula difference fails torch.equal rather than
    # hiding inside a tolerance.
    state, kc, uc, gc = _make_pool(
        2 * B, hist, bases, seed=99, ring_slots=ring_slots, state_dtype=torch.float32
    )
    zq = torch.zeros(ORACLE_T, H, K, dtype=torch.bfloat16, device=DEV)
    zv = torch.zeros(ORACLE_T, HV, V, dtype=torch.bfloat16, device=DEV)
    za = torch.zeros(ORACLE_T, HV, dtype=torch.bfloat16, device=DEV)
    zh = torch.zeros(HV, dtype=torch.bfloat16, device=DEV)

    src = torch.arange(B, dtype=torch.int32, device=DEV)
    got = state.clone()
    materialize_ref(
        [got],
        [kc],
        [uc],
        [gc],
        src[None, :],
        (src + B)[None, :],
        _i32(bases),
        _i32(counts),
    )
    for r in range(B):
        n = max(counts[r], 1)
        _, s_ref = _ref_fp32(
            zq,
            zq,
            zv,
            za,
            za,
            zh,
            zh,
            state[r],
            _logical(kc[r], bases[r], n),
            _logical(uc[r], bases[r], n),
            _logical(gc[r], bases[r], n),
            counts[r],
        )
        assert torch.equal(got[B + r], s_ref), (
            f"request {r}: materialize_ref disagrees with the decode oracle"
        )


# ---------------------------------------------------------------------------
# Kernel vs reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
def test_mixed_batch_matches_reference(ring_slots):
    """Full replay, partial replay, exact copy and a padding row in ONE launch.

    Also checks the source slots are untouched -- the live request is still
    decoding against them.
    """
    _skip_if_not_sm90_or_later()
    hist = [15, 12, 15, 7]
    bases = [11, 5, 0, 9]
    counts = [15, 5, 0, -1]  # full, partial, copy, skipped
    state, kc, uc, gc = _make_pool(8, hist, bases, seed=3, ring_slots=ring_slots)
    src = _i32([0, 1, 2, 3])
    dst = _i32([4, 5, 6, 7])
    before = state.clone()

    ref_state, got_state = _run_both(
        state, kc, uc, gc, src, dst, _i32(bases), _i32(counts)
    )

    assert torch.equal(got_state[:4], before[:4]), "kernel modified source slots"
    _assert_close(ref_state, got_state, [4, 5, 6])
    # count == 0 must be an EXACT copy, not merely close.
    assert torch.equal(got_state[6], before[2])
    # A negative count leaves the destination exactly as it was.
    assert torch.equal(got_state[7], before[7])


@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
@pytest.mark.parametrize("count", list(range(0, W_RING + 1)))
def test_exhaustive_prefix_lengths(count, ring_slots):
    """Every legal replay length, at a wrapped base.

    The prefix length is the one input a prefix cache actually varies, so it
    gets exhaustive rather than sampled coverage.
    """
    _skip_if_not_sm90_or_later()
    hist = [W_RING]
    bases = [7]
    state, kc, uc, gc = _make_pool(
        2, hist, bases, seed=100 + count, ring_slots=ring_slots
    )
    src, dst = _i32([0]), _i32([1])
    before = state.clone()

    ref_state, got_state = _run_both(
        state, kc, uc, gc, src, dst, _i32(bases), _i32([count])
    )
    _assert_close(ref_state, got_state, [1])
    if count == 0:
        assert torch.equal(got_state[1], before[0])


@pytest.mark.parametrize(
    "ring_slots,base",
    [(r, b) for r in RING_DEPTHS for b in range(r)],
)
def test_all_base_alignments(ring_slots, base):
    """Every ring-window origin at BOTH depths, including wrapping windows.

    This is the test that catches a baked-in ring depth: with the mask hard-
    coded to 32, a 16-slot ring at base >= 8 indexes past the end of the
    request's ring and into the next pool slot, silently. Only a non-zero base
    on the SHORT ring exposes it, which is why the sweep is exhaustive rather
    than sampled.
    """
    _skip_if_not_sm90_or_later()
    hist = [12]
    state, kc, uc, gc = _make_pool(
        2, hist, [base], seed=200 + base, ring_slots=ring_slots
    )
    ref_state, got_state = _run_both(
        state, kc, uc, gc, _i32([0]), _i32([1]), _i32([base]), _i32([9])
    )
    _assert_close(ref_state, got_state, [1])


@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
def test_shuffled_slots_leave_untouched_slots_alone(ring_slots):
    """Non-identity src/dst permutation; unindexed slots must not change.

    An identity mapping would hide an addressing bug that happens to compute
    `slot` where it meant `request`.
    """
    _skip_if_not_sm90_or_later()
    hist = [15, 11, 8]
    bases = [3, 14, 0]
    state, kc, uc, gc = _make_pool(10, hist, bases, seed=7, ring_slots=ring_slots)
    # Requests live in slots 0..2; scatter their destinations non-monotonically.
    src = _i32([2, 0, 1])
    dst = _i32([9, 5, 7])
    # Per-request base/count must follow the SOURCE slot's ring, not the
    # request index -- src[i] = 2, 0, 1 means bases 0, 3, 14 and counts 8, 15, 11.
    before = state.clone()

    ref_state, got_state = _run_both(
        state,
        kc,
        uc,
        gc,
        src,
        dst,
        _i32([bases[2], bases[0], bases[1]]),
        _i32([8, 15, 11]),
    )
    _assert_close(ref_state, got_state, [9, 5, 7])
    untouched = [3, 4, 6, 8]
    for s in untouched:
        assert torch.equal(got_state[s], before[s]), f"slot {s} was written"


@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
@pytest.mark.parametrize(
    "src_v,dst_v,count_v,label",
    [
        (-1, 4, 8, "negative src"),
        (0, -1, 8, "negative dst"),
        (0, 4, -1, "negative count"),
        (0, 4, W_RING + 1, "count above the window depth"),
    ],
)
def test_skip_semantics(src_v, dst_v, count_v, label, ring_slots):
    """Skipped rows touch neither source nor destination.

    Note an over-long count is SKIPPED rather than raising, matching the mamba
    materialize kernel's `count > MAX_WINDOW` guard.
    """
    _skip_if_not_sm90_or_later()
    state, kc, uc, gc = _make_pool(8, [15], [0], seed=11, ring_slots=ring_slots)
    before = state.clone()
    got_state = state.clone()
    gdn_prefix_materialize(
        got_state, _i32([src_v]), _i32([dst_v]), kc, uc, gc, _i32([0]), _i32([count_v])
    )
    torch.cuda.synchronize()
    assert torch.equal(got_state, before), f"{label}: kernel wrote something"


@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
def test_grouped_kv_heads(ring_slots):
    """Value head hv must read k head hv // (HV // H).

    With HV = 64 and H = 16 the grouping is 4:1, so a kernel that dropped the
    division would read the wrong k rows for 3 of every 4 value heads.
    """
    _skip_if_not_sm90_or_later()
    assert HV // H == 4, "this test assumes a 4:1 value-to-key head grouping"
    state, kc, uc, gc = _make_pool(2, [10], [5], seed=21, ring_slots=ring_slots)
    ref_state, got_state = _run_both(
        state, kc, uc, gc, _i32([0]), _i32([1]), _i32([5]), _i32([10])
    )
    # Compare per value head so a grouping bug localizes instead of averaging out.
    for hv in range(HV):
        err = (got_state[1, hv].float() - ref_state[1, hv].float()).abs().max().item()
        assert err < STATE_TOL, f"value head {hv}: absmax {err:.3e}"


@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
def test_repeated_launch_is_idempotent(ring_slots):
    """Materializing twice writes the same bytes.

    Guards the shared-memory lifetime inside the kernel: a missing barrier
    between the ring staging and the fold tends to show up as run-to-run
    variation rather than a consistently wrong answer.
    """
    _skip_if_not_sm90_or_later()
    state, kc, uc, gc = _make_pool(4, [15, 13], [6, 2], seed=31, ring_slots=ring_slots)
    src, dst = _i32([0, 1]), _i32([2, 3])
    first = state.clone()
    gdn_prefix_materialize(first, src, dst, kc, uc, gc, _i32([6, 2]), _i32([15, 13]))
    second = state.clone()
    gdn_prefix_materialize(second, src, dst, kc, uc, gc, _i32([6, 2]), _i32([15, 13]))
    torch.cuda.synchronize()
    assert torch.equal(first, second), "repeated launches disagree"


def test_persistent_grid_second_iteration():
    """A batch large enough that every CTA walks MORE than one work item.

    The persistent pool is min(B*HV, SMs*8) CTAs; with B=64 and HV=64 the item
    count (4096) exceeds any current GPU's pool, so CTAs take a second trip
    through the grid-stride loop. That second trip is the only path where the
    end-of-item barrier and the TMA mbarrier's phase re-use can break -- a
    one-item launch gets teardown for free. All requests are live and all bases
    wrap somewhere, so a stale-SMEM bug corrupts a comparable, checked value.
    """
    _skip_if_not_sm90_or_later()
    B = 64
    rng = torch.Generator(device=DEV).manual_seed(77)
    hist = [15] * B
    bases = [int(x) for x in torch.randint(0, 16, (B,), generator=rng, device=DEV)]
    counts = [int(x) for x in torch.randint(0, 16, (B,), generator=rng, device=DEV)]
    state, kc, uc, gc = _make_pool(2 * B, hist, bases, seed=78, ring_slots=32)
    src = torch.arange(B, dtype=torch.int32, device=DEV)
    dst = src + B

    ref_state, got_state = _run_both(
        state, kc, uc, gc, src, dst, _i32(bases), _i32(counts)
    )
    for r in range(B):
        err = (got_state[B + r].float() - ref_state[B + r].float()).abs().max().item()
        assert err < STATE_TOL, f"request {r} (count={counts[r]}): absmax {err:.3e}"


def test_active_list_order_and_sentinel():
    """The compacted active list is a MAP, not a range, and -1 tails are inert.

    Live entries are deliberately NOT in batch order, so a kernel that treated
    the list as arange(B) would read the wrong rows. Request 3 has a live
    count but is ABSENT from the list (the tail is -1), so it must not run --
    membership in the list, not count, decides. Note this does NOT pin
    stop-at-first-sentinel: the kernel's documented contract is that -1
    entries are skipped, so a malformed list like [2, -1, 0] would still
    process row 0; only well-formed live-prefix-then-sentinel lists are
    supported. Reordering mirrors PR #4815's reversed-indices test.
    """
    _skip_if_not_sm90_or_later()
    hist = [15, 12, 9, 7]
    bases = [11, 5, 0, 9]
    state, kc, uc, gc = _make_pool(8, hist, bases, seed=13)
    src = _i32([0, 1, 2, 3])
    dst = _i32([4, 5, 6, 7])
    counts = _i32([15, 12, 9, 7])  # ALL counts >= 0, incl. request 3's
    # Reversed order for the live prefix; request 3 hidden behind the sentinel.
    active = _i32([2, 0, 1, -1])
    before = state.clone()

    ref_state = state.clone()
    materialize_ref(
        [ref_state],
        [kc],
        [uc],
        [gc],
        src[None, :],
        dst[None, :],
        _i32(bases),
        counts,
        active_request_indices=active,
    )
    got_state = state.clone()
    gdn_prefix_materialize(got_state, src, dst, kc, uc, gc, _i32(bases), counts, active)
    torch.cuda.synchronize()

    _assert_close(ref_state, got_state, [4, 5, 6])
    # Request 3 sits after the sentinel: its destination must be untouched
    # despite count[3] == 7 saying "do me".
    assert torch.equal(got_state[7], before[7]), (
        "kernel processed a request hidden behind the -1 sentinel"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
def test_randomized_stress(ring_slots, seed):
    """Everything random at once: fold lengths, window origins, fill levels,
    slot permutations, and the live/skip pattern.

    The directed tests each vary ONE axis; this varies them together, because
    addressing bugs love correlated inputs (e.g. count == hist == base only
    fails when they differ). Four seeds x two depths; any failure reproduces
    exactly from the seed.
    """
    _skip_if_not_sm90_or_later()
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)

    def ri(lo, hi, n):  # inclusive bounds
        return [int(x) for x in torch.randint(lo, hi + 1, (n,), generator=g)]

    B = 24
    hist = ri(0, W_RING, B)
    bases = ri(0, ring_slots - 1, B)
    # count <= hist per request (the caller contract); ~1 in 4 rows skipped.
    counts = [
        c if keep else -1
        for c, keep in zip((ri(0, h, 1)[0] for h in hist), ri(0, 3, B), strict=True)
    ]
    state, kc, uc, gc = _make_pool(
        3 * B, hist, bases, seed=2000 + seed, ring_slots=ring_slots
    )
    # Random src/dst permutations over disjoint slot ranges; rings live with
    # their slots, so base/count follow src through the permutation.
    perm = torch.randperm(B, generator=g)
    src = perm.to(torch.int32).to(DEV)
    dst = (torch.randperm(B, generator=g) + B).to(torch.int32).to(DEV)
    base_t = _i32([bases[i] for i in perm.tolist()])
    cnt_t = _i32([counts[i] for i in perm.tolist()])
    before = state.clone()

    ref_state, got_state = _run_both(state, kc, uc, gc, src, dst, base_t, cnt_t)

    assert torch.equal(got_state[:B], before[:B]), "a source slot was modified"
    for i in range(B):
        d = int(dst[i])
        if int(cnt_t[i]) < 0:
            assert torch.equal(got_state[d], before[d]), (
                f"seed {seed}: skipped row {i} wrote its destination"
            )
        else:
            err = (got_state[d].float() - ref_state[d].float()).abs().max().item()
            assert err < STATE_TOL, (
                f"seed {seed}: row {i} src={int(src[i])} count={int(cnt_t[i])} "
                f"base={int(base_t[i])} absmax={err:.3e}"
            )
    # Slots in [2B, 3B) were never named by src or dst: must be untouched.
    assert torch.equal(got_state[2 * B :], before[2 * B :]), (
        "an unindexed slot was written"
    )


# ---------------------------------------------------------------------------
# Paged (block-strided) serving pools: dense inner dims, padded slot stride.
# ---------------------------------------------------------------------------
def _padded_view(t, pad_elems):
    """Rebuild `t` as a strided view with `pad_elems` of dead space per slot.

    This is the vLLM-style paged layout: slot contents identical, slot-to-slot
    stride larger than the slot. The pad region is poisoned with NaN-ish bytes
    so a kernel that strays into it produces loudly wrong values, and we also
    assert afterwards that the pad was never written.
    """
    per = t[0].numel()
    buf = torch.empty(t.shape[0] * (per + pad_elems), dtype=t.dtype, device=t.device)
    buf.fill_(float("nan") if t.dtype.is_floating_point else -1)
    view = buf.as_strided(t.shape, (per + pad_elems,) + tuple(t.stride()[1:]))
    view.copy_(t)
    return view, buf


@pytest.mark.parametrize("pad", [8, 256])  # multiples of 8 keep 16 B alignment
@pytest.mark.parametrize("ring_slots", RING_DEPTHS)
def test_padded_slot_stride_pools(ring_slots, pad):
    """Paged pools: every big tensor a strided view, results match contiguous.

    Two different pads run in one session, so the compile cache must hold a
    correct specialization PER layout (static descriptors bake the stride);
    a stale-cubin mixup would fail one of the two parametrizations.
    """
    _skip_if_not_sm90_or_later()
    hist = [15, 12, 9, 7]
    bases = [11, 5, 0, 9]
    counts = _i32([15, 5, 0, -1])  # full, partial, copy, skipped
    state, kc, uc, gc = _make_pool(8, hist, bases, seed=41, ring_slots=ring_slots)
    src, dst = _i32([0, 1, 2, 3]), _i32([4, 5, 6, 7])

    # Ground truth on the contiguous tensors.
    ref_state = state.clone()
    materialize_ref(
        [ref_state], [kc], [uc], [gc], src[None, :], dst[None, :], _i32(bases), counts
    )

    state_v, state_buf = _padded_view(state, pad)
    kc_v, _ = _padded_view(kc, pad)
    uc_v, _ = _padded_view(uc, pad)
    gc_v, _ = _padded_view(gc, pad)
    assert not state_v.is_contiguous()
    pad_before = state_buf.clone()

    gdn_prefix_materialize(state_v, src, dst, kc_v, uc_v, gc_v, _i32(bases), counts)
    torch.cuda.synchronize()

    _assert_close(ref_state, state_v, [4, 5, 6])
    assert torch.equal(state_v[7], state[7]), "skipped row's destination changed"
    # The dead space between slots must be untouched (NaN poison intact).
    per = state[0].numel()
    for s in range(8):
        gap = state_buf[s * (per + pad) + per : (s + 1) * (per + pad)]
        ref_gap = pad_before[s * (per + pad) + per : (s + 1) * (per + pad)]
        assert torch.equal(gap.view(torch.int16), ref_gap.view(torch.int16)), (
            f"kernel wrote into the pad after slot {s}"
        )


def test_inner_stride_rejected():
    """Non-density in an INNER dim must be a loud error, never accepted.

    The kernel hardcodes dense ring-row/feature addressing; a faithfully
    described inner-strided view would be read wrongly and silently. The
    wrapper must refuse it.
    """
    _skip_if_not_sm90_or_later()
    state, kc, uc, gc = _make_pool(4, [8], [0], seed=42)
    # state with a padded HEAD stride (inner dim 1) -- dense slots, gapped heads.
    buf = torch.zeros(4 * HV * (V * K + 64), dtype=state.dtype, device=DEV)
    bad = buf.as_strided((4, HV, V, K), (HV * (V * K + 64), V * K + 64, K, 1))
    with pytest.raises(ValueError, match="dim 1 has stride"):
        gdn_prefix_materialize(
            bad, _i32([0]), _i32([1]), kc, uc, gc, _i32([0]), _i32([2])
        )


def test_noncontiguous_metadata_rejected():
    """Strided metadata is refused: it is not covered by the compile cache
    key, so a baked-in view layout could be silently reused by a later call."""
    _skip_if_not_sm90_or_later()
    state, kc, uc, gc = _make_pool(4, [8, 8], [0, 0], seed=43)
    table = torch.zeros(2, 2, dtype=torch.int32, device=DEV)
    strided_count = table[:, 0]  # shape (2,), stride 2 -> non-contiguous
    strided_count.fill_(2)
    assert not strided_count.is_contiguous()
    with pytest.raises(ValueError, match="count must be contiguous"):
        gdn_prefix_materialize(
            state, _i32([0, 1]), _i32([2, 3]), kc, uc, gc, _i32([0, 0]), strided_count
        )


def test_undersized_ring_heads_rejected():
    """u_cache/g_cache with fewer heads than the state's HV must be refused.

    The kernel indexes every value head < HV into both rings; an undersized
    allocation would be read past its end (CodeRabbit PR finding)."""
    _skip_if_not_sm90_or_later()
    state, kc, uc, gc = _make_pool(4, [8], [0], seed=51)
    short_uc = uc[:, : HV - 1].contiguous()
    with pytest.raises(ValueError, match="head dims must equal state HV"):
        gdn_prefix_materialize(
            state, _i32([0]), _i32([1]), kc, short_uc, gc, _i32([0]), _i32([2])
        )
    short_gc = gc[:, :1].contiguous()
    with pytest.raises(ValueError, match="head dims must equal state HV"):
        gdn_prefix_materialize(
            state, _i32([0]), _i32([1]), kc, uc, short_gc, _i32([0]), _i32([2])
        )


def test_wrong_ring_dtype_rejected():
    """Ring dtype must equal the module RING dtype, not merely be 16-bit.

    The 16 B cp.async copies raw bytes and the fold MMA is typed for the ring
    element type, so an fp32 (or otherwise mismatched) ring would be silently
    reinterpreted bytewise. Note the check targets the RING dtype: in the
    mixed modes it legitimately differs from the STATE dtype."""
    _skip_if_not_sm90_or_later()
    from flashinfer.gdn_kernels.gdn_prefix_materialize import ring_ty_torch

    state, kc, uc, gc = _make_pool(4, [8], [0], seed=52)
    bad = torch.float32 if ring_ty_torch() != torch.float32 else torch.float16
    with pytest.raises(ValueError, match="module RING dtype"):
        gdn_prefix_materialize(
            state, _i32([0]), _i32([1]), kc.to(bad), uc, gc, _i32([0]), _i32([2])
        )
    with pytest.raises(ValueError, match="module RING dtype"):
        gdn_prefix_materialize(
            state, _i32([0]), _i32([1]), kc, uc.to(bad), gc, _i32([0]), _i32([2])
        )


def test_misaligned_base_pointer_rejected():
    """A sliced view whose data_ptr is not 16-byte aligned must be refused.

    The descriptors promise assumed_align=16 to the compiler (TMA base, 16 B
    ring cp.async); an 8-byte-aligned slice would make that promise false."""
    _skip_if_not_sm90_or_later()
    state, kc, uc, gc = _make_pool(4, [8], [0], seed=53)
    n = state.numel()
    # Offset by 4 bf16 elements = 8 bytes: contiguous, correct shape, misaligned.
    buf = torch.zeros(n + 4, dtype=state.dtype, device=DEV)
    shifted = buf[4 : 4 + n].view(state.shape)
    shifted.copy_(state)
    assert shifted.data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="16-byte aligned"):
        gdn_prefix_materialize(
            shifted, _i32([0]), _i32([1]), kc, uc, gc, _i32([0]), _i32([2])
        )
