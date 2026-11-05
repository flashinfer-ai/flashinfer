"""
Correctness tests for the UMMA (tcgen05 / TMEM) MTP ReplaySSM kernel
(flashinfer/gdn_kernels/gdn_replay_mtp_umma_ring64.py).

The oracle is FlashInfer's ``_ref_fp32`` (tests/gdn/test_decode_ucache.py), generalised to
any T: an fp32, token-at-a-time PyTorch loop over the ring replay and the delta rule.  Case
builder follows ``_make_case`` (RING-slot physical ring, logical row j at (base + j) % RING) and
additionally NaN-poisons every ring slot outside the live window, so a kernel that lets a
stale slot reach an MMA (0 * NaN = NaN) fails loudly.

Covered: T in 1..8, wrapped windows, mixed verify/flush batches, padded rows, row order,
predictive flush_min, ring appends (k̂, u, g) and the committed state, a multi-step horizon
with cursor commits, and parity with ``gated_delta_rule_mtp_ucache_flush`` (fp16_state arm).

Run:  pytest tests/gdn/test_mtp_umma_ring64.py -q
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

DEV = "cuda"
H, HV, K, V = 16, 64, 128, 128
W = int(
    os.environ.get("GDN_MTP_UMMA_W_RING", "32")
)  # window capacity class of the build under test (32 / 48 / 64)
RING = (
    64  # RING_SLOTS = 64 ring; see flashinfer/gdn_kernels/gdn_replay_mtp_umma_ring64.py
)
RS = RING // 32  # scale of the 32-slot-era literal bases / depths
# physical window starts used by the wrapped-window cases: 8 values spread over the ring (the 32-slot list
# scaled to this ring, each shifted so windows of every depth wrap in some of the cases)
_B8 = [0, 28 * RS, 5 * RS, 30 * RS, 17 * RS, 31 * RS, 20 * RS, 25 * RS]
_B8W = [28 * RS, 5 * RS, 30 * RS, 17 * RS, 31 * RS, 20 * RS, 25 * RS, 9 * RS]


def _fm_max(T):
    """Largest ring-safe flush_min of this W/RING build (the wrapper's default): nu - T + 1 = min(W - T + 1, RING - 2 T + 1).
    A row with P == fm_max flushes; a verify row at fm_max - 1 advances to at most nu, so the next call's window still
    fits the nu-row window; the T appends land at physical slots P..P+T-1 with P + T <= nu + T <= RING, no overlap
    against the P live rows just read."""
    return _p_max(T) - T + 1


def _default_fm(T):
    """The wrapper's default flush_min: the ring-safe maximum, except at T = 1 (one ring row per step), where a window
    longer than one 16-row chunk costs more in window-row reads than it saves in flush writes: F = 16."""
    return min(_fm_max(T), 16) if T == 1 else _fm_max(T)


def _p_max(T):
    """Deepest live window a call may see: nu = min(W, RING - T) (= W at W < RING)."""
    return min(W, RING - T)


def _umma_imports():
    from flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 import W_RING, W_CHUNK

    return W_RING, W_CHUNK


def _skip_if_vf_unsupported():
    w_ring, w_chunk = _umma_imports()
    if w_ring > w_chunk:
        pytest.skip(
            f"unf_vf (vf=True) is not implemented for W_RING ({w_ring}) > W_CHUNK ({w_chunk})"
        )


SCALE = 1.0 / math.sqrt(K)
Y_TOL = 8e-3  # FlashInfer's tolerance for outputs
STATE_TOL = 2e-2  # FlashInfer's tolerance for the committed state
U_TOL = 1.5e-2  # appended corrections (bf16 storage of u)

_ROOT = Path(__file__).resolve().parents[2]
if (
    str(_ROOT) not in sys.path
):  # run against this checkout even without an installed wheel
    sys.path.insert(0, str(_ROOT))


def _skip_if_not_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("UMMA kernel requires SM100 (Blackwell)")


def _umma():
    from flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 import (
        gated_delta_rule_mtp_ucache_flush_umma_ring64,
    )

    return gated_delta_rule_mtp_ucache_flush_umma_ring64


_FI = {}


def _flashinfer_fp16_state():
    """FlashInfer's MTP ucache flush kernel, fp16_state arm (dtype is fixed at import)."""
    if "mod" in _FI:
        return _FI["mod"]
    old = {
        k: os.environ.pop(k, None)
        for k in (
            "GDN_UCACHE_IO_DTYPE",
            "GDN_UCACHE_STATE_DTYPE",
            "GDN_UCACHE_RING_DTYPE",
        )
    }
    os.environ["GDN_UCACHE_STATE_DTYPE"] = "fp16"
    try:
        path = _ROOT / "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py"
        spec = importlib.util.spec_from_file_location(
            "flashinfer.gdn_kernels._uc_flush_fp16_state", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k_, v_ in old.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    _FI["mod"] = mod
    return mod


# ---------------------------------------------------------------------------
# fp32 oracle (FlashInfer's _ref_fp32, any T).  One request; S is [HV, V, K].
# Returns y [T, HV, V], the state a fold commits, and the T corrections u [T, HV, V].
# ---------------------------------------------------------------------------
def _ref_fp32(q, k, v, a, b, A_log, dt_bias, S0, kc, uc, gc, P):
    f = torch.float32
    T = q.shape[0]
    grp = HV // H
    S = S0.to(f).clone()
    if P > 0:
        GP = gc[:, P - 1].to(f)
        w = torch.exp(GP[:, None] - gc[:, :P].to(f))
        kc_hv = kc[:, :P].to(f).repeat_interleave(grp, dim=0)
        S = torch.exp(GP)[:, None, None] * S + torch.einsum(
            "hpv,hpk->hvk", w[:, :, None] * uc[:, :P].to(f), kc_hv
        )
    S_after_history = S.clone()
    khat = F.normalize(k.to(f), dim=-1)
    qhat = F.normalize(q.to(f), dim=-1) * SCALE
    y = torch.zeros(T, HV, V, dtype=f, device=q.device)
    us = torch.zeros(T, HV, V, dtype=f, device=q.device)
    gam = torch.zeros(T, HV, dtype=f, device=q.device)
    acc = torch.zeros(HV, dtype=f, device=q.device)
    for t in range(T):
        la = -torch.exp(A_log.to(f)) * F.softplus(a[t].to(f) + dt_bias.to(f))
        acc = acc + la
        gam[t] = acc
        beta = torch.sigmoid(b[t].to(f))
        k_hv = khat[t].repeat_interleave(grp, dim=0)
        q_hv = qhat[t].repeat_interleave(grp, dim=0)
        S = S * torch.exp(la)[:, None, None]
        pred = torch.einsum("hvk,hk->hv", S, k_hv)
        u_t = (v[t].to(f) - pred) * beta[:, None]
        us[t] = u_t
        S = S + u_t[:, :, None] * k_hv[:, None, :]
        y[t] = torch.einsum("hvk,hk->hv", S, q_hv)
    return y, S_after_history, us, khat, gam


def _make_case(B, T, hist_lens, seed, bases=None, poison=True):
    io = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io)

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, a, b = rn(B, T, HV, V, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (
        torch.full((HV,), -3.0, device=DEV)
        + torch.rand(HV, generator=g, device=DEV) * 0.3
    ).to(io)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(torch.float16)
    fill = float("nan") if poison else 0.0
    kc = torch.full((B, H, RING, K), fill, dtype=io, device=DEV)
    uc = torch.full((B, HV, RING, V), fill, dtype=io, device=DEV)
    gc = torch.full((B, HV, RING), fill, dtype=torch.float32, device=DEV)
    hl = torch.tensor(hist_lens, dtype=torch.int32, device=DEV)
    bases = bases or [0] * B
    cb = torch.tensor(bases, dtype=torch.int32, device=DEV)
    for r in range(B):
        P = int(hl[r])
        if P == 0:
            continue
        rows = torch.tensor(
            [(bases[r] + j) % RING for j in range(P)], dtype=torch.long, device=DEV
        )
        kh = torch.randn(H, P, K, generator=g, device=DEV)
        kc[r, :, rows] = F.normalize(kh, dim=-1).to(io)
        uc[r, :, rows] = (torch.randn(HV, P, V, generator=g, device=DEV) * 0.3).to(io)
        la = -(torch.rand(HV, P, generator=g, device=DEV) * 0.3 + 0.003)
        gc[r, :, rows] = torch.cumsum(la, dim=-1)
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx


def _logical(kc_r, uc_r, gc_r, base):
    rows = torch.tensor(
        [(base + j) % RING for j in range(W)], dtype=torch.long, device=DEV
    )
    return (
        kc_r.index_select(1, rows),
        uc_r.index_select(1, rows),
        gc_r.index_select(1, rows),
    )


def _run(fn, case, flush_min, row_order=None, restart=False, window_rows=None):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    kw = dict(
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=flush_min,
        restart_hist_on_flush=restart,
    )
    if row_order is not None:
        kw["row_order"] = row_order
    T = q.shape[1]
    if (
        window_rows is None
        and flush_min is not None
        and int(hl.max()) > flush_min - 1 + T
    ):
        # histories deeper than the commit rule allows (P <= flush_min - 1 + T: flush_min = 1 sweeps, predictive
        # flush_min with deep rows) need the window passed explicitly; in-contract cases use the derived default
        window_rows = min(int(hl.max()), _p_max(T))
    if window_rows is not None:
        kw["window_rows"] = window_rows
    return fn(A_log, a, dt_bias, **kw)


def _check_case(
    T, hist, bases, flush_min, seed, row_order=None, check_rings=True, hpc=None
):
    fn = _umma()
    if hpc is not None:
        import functools

        fn = functools.partial(fn, heads_per_cta=hpc)
    B = len(hist)
    case = _make_case(B, T, hist, seed, bases)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    pool0, kc0, uc0, gc0 = pool.clone(), kc.clone(), uc.clone(), gc.clone()
    ro = None
    if row_order == "flush_first":
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 import (
            flush_first_row_order,
        )

        ro = flush_first_row_order(hl, flush_min)
    elif row_order == "reverse":
        ro = torch.arange(B - 1, -1, -1, dtype=torch.int32, device=DEV)
    y = _run(fn, case, flush_min, ro)
    torch.cuda.synchronize()
    errs = {"y": 0.0, "S": 0.0, "u": 0.0, "k": 0.0, "g": 0.0}
    for r in range(B):
        P = hist[r]
        kc_l, uc_l, gc_l = _logical(kc0[r], uc0[r], gc0[r], bases[r])
        y_ref, S_ref, u_ref, khat, gam = _ref_fp32(
            q[r], k[r], v[r], a[r], b[r], A_log, dt_bias, pool0[r], kc_l, uc_l, gc_l, P
        )
        errs["y"] = max(errs["y"], (y[r].float() - y_ref).abs().max().item())
        flushed = P >= flush_min
        if flushed:
            errs["S"] = max(errs["S"], (pool[r].float() - S_ref).abs().max().item())
        else:
            assert torch.equal(pool[r], pool0[r]), (
                f"row {r}: verify row modified the state"
            )
        if check_rings:
            app = torch.tensor(
                [(bases[r] + P + s) % RING for s in range(T)],
                dtype=torch.long,
                device=DEV,
            )
            win = torch.tensor(
                [(bases[r] + j) % RING for j in range(P)], dtype=torch.long, device=DEV
            )
            errs["u"] = max(
                errs["u"],
                (uc[r][:, app].float() - u_ref.transpose(0, 1)).abs().max().item(),
            )
            errs["k"] = max(
                errs["k"],
                (kc[r][:, app].float() - khat.transpose(0, 1)).abs().max().item(),
            )
            gp = (
                0.0
                if (flushed or P == 0)
                else gc0[r][:, (bases[r] + P - 1) % RING][:, None]
            )
            errs["g"] = max(
                errs["g"],
                (gc[r][:, app] - (gam.transpose(0, 1) + gp)).abs().max().item(),
            )
            for name, before, after in (("k", kc0, kc), ("u", uc0, uc), ("g", gc0, gc)):
                assert torch.equal(
                    before[r].index_select(1, win), after[r].index_select(1, win)
                ), f"row {r}: live {name} window modified"
    assert errs["y"] < Y_TOL, errs
    assert errs["S"] < STATE_TOL, errs
    if check_rings:
        assert errs["u"] < U_TOL and errs["k"] < 1e-2 and errs["g"] < 1e-4, errs
    return errs


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("bases", ["base0", "wrap"])
def test_matches_fp32_reference_any_T(T, bases):
    _skip_if_not_sm100()
    fm = _fm_max(T)
    hist = [0, fm - 1, fm, _p_max(T), 1, fm - 1, _p_max(T), 3]
    b = [0] * 8 if bases == "base0" else _B8W
    _check_case(T, hist, b, fm, seed=100 + T)


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("order", [None, "flush_first", "reverse"])
def test_row_order_and_predictive_flush(T, order):
    _skip_if_not_sm100()
    fm = (9 - (T == 8) * 4) * RS  # predictive flush_min (< lazy default)
    hist = [fm, 0, fm + 2, fm - 1, 2, _p_max(T), fm, 1]
    _check_case(
        T,
        hist,
        [3 * RS, 30 * RS, 27 * RS, 0, 31 * RS, 16 * RS, 29 * RS, 7 * RS],
        fm,
        seed=7 + T,
        row_order=order,
    )


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("hpc", [1, 2, 4])
def test_heads_per_cta(T, hpc):
    """Every heads-per-CTA split (4: one CTA per key head; 2 / 1: a key head over 2 / 4 CTAs)
    against the fp32 reference: mixed verify / flush rows, wrapped windows, rings."""
    _skip_if_not_sm100()
    fm = _fm_max(T)
    hist = [fm, 0, fm - 1, _p_max(T), 3, fm - 1, 1, fm]
    _check_case(
        T,
        hist,
        [28 * RS, 5 * RS, 30 * RS, 0, 31 * RS, 17 * RS, 9 * RS, 26 * RS],
        fm,
        seed=50 + T + hpc,
        hpc=hpc,
    )


@pytest.mark.parametrize("T", [4, 8])
def test_launch_policy_is_bit_identical(T):
    """heads_per_cta (grid split) and min_blocks_per_mp (register cap) only change how the
    kernel is launched: outputs, committed state and every ring must stay bit-identical to
    the default launch."""
    _skip_if_not_sm100()
    import functools

    fm = _fm_max(T)
    hist = [fm, 0, fm - 1, _p_max(T), 3, fm - 1, 1, fm]
    bases = [28 * RS, 5 * RS, 30 * RS, 0, 31 * RS, 17 * RS, 9 * RS, 26 * RS]
    fn = _umma()

    def run(**kw):
        case = _make_case(len(hist), T, hist, 11 + T, bases)
        y = _run(functools.partial(fn, **kw), case, fm)
        torch.cuda.synchronize()
        _, _, _, _, _, _, _, pool, kc, uc, gc, _, _, _ = case
        return y, pool, kc, uc, gc

    ref = run()
    for kw in (
        dict(min_blocks_per_mp=3),
        dict(min_blocks_per_mp=4),
        dict(heads_per_cta=2, min_blocks_per_mp=4),
        dict(heads_per_cta=1),
        dict(heads_per_cta=4, min_blocks_per_mp=4),
    ):
        for name, x, y_ in zip(
            ("y", "state", "k", "u", "g"), ref, run(**kw), strict=False
        ):
            # bit comparison: the rings hold NaN poison outside the live window (NaN != NaN)
            assert torch.equal(
                x.contiguous().flatten().view(torch.uint8),
                y_.contiguous().flatten().view(torch.uint8),
            ), (kw, name)


@pytest.mark.parametrize("minb", [1, 2, 5])
def test_min_blocks_per_mp_rejects_out_of_range(minb):
    """5: TMEM holds at most 4 CTAs per SM. 1 and 2: the two warpgroups re-split a launch cap of 64 (4 CTAs/SM)
    or 80 (3) registers per thread with setmaxnreg, so a larger cap would have the consumers "increase" to
    96 registers below their launch count."""
    _skip_if_not_sm100()
    import functools

    with pytest.raises(AssertionError):
        _run(
            functools.partial(_umma(), min_blocks_per_mp=minb),
            _make_case(2, 4, [0, 1], 1, [0, 0]),
            _fm_max(4),
        )


def test_auto_heads_per_cta_policy(monkeypatch):
    """The W=16 policy (4 CTAs/SM, 8 waves) is unchanged; the module's own policy follows the
    CTAs/SM its shared memory allows (4 at W_RING = 32 after the smem diet) and its measured
    crossovers."""
    import flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 as m

    monkeypatch.setattr(m, "_CTA_SLOTS_PER_SM", 4)
    monkeypatch.setattr(m, "_HPC2_MAX_WAVES", 8)
    monkeypatch.setattr(m, "_HPC2_MAX_WAVES_T8", 8)
    for T in (1, 4, 8):
        assert [
            m.auto_heads_per_cta(b, 16, 148, T) for b in (1, 9, 10, 148, 149, 512)
        ] == [1, 1, 2, 2, 4, 4]
    monkeypatch.undo()
    assert m._SMEM_CTAS_PER_SM == (4 if m.W_RING <= 32 else 3)
    assert m._MINB == int(os.environ.get("GDN_MTP_UMMA_MINB", m._SMEM_CTAS_PER_SM))
    if m.W_RING == 32:
        for T in (4, 8):
            assert [
                m.auto_heads_per_cta(b, 16, 148, T)
                for b in (1, 9, 10, 55, 56, 64, 74, 128, 512)
            ] == [1, 1, 2, 2, 4, 4, 4, 4, 4]
        assert m.auto_heads_per_cta(100, 16, 148) == m.auto_heads_per_cta(
            100, 16, 148, 4
        )
    if m.W_RING > 32:
        # 3 CTAs/SM (66.8 KB of shared memory at W_RING = 48, 74.8 KB at 64): 444 CTA slots on 148 SMs; hpc = 1 while
        # B * 64 <= 444 (B <= 6), hpc = 2 while B * 32 <= 3 * 444 (B <= 41), else 4 (the W = 32 waves rule, unmeasured here)
        for T in (4, 8):
            assert [
                m.auto_heads_per_cta(b, 16, 148, T) for b in (1, 6, 7, 41, 42, 64, 512)
            ] == [1, 1, 2, 2, 4, 4, 4]


def test_full_window_flush_P16_and_T1():
    _skip_if_not_sm100()
    fm = _fm_max(1)
    _check_case(1, [fm, fm - 1, fm, 0], [20 * RS, 31 * RS, 0, 5], fm, seed=3)


def test_padded_rows_untouched():
    _skip_if_not_sm100()
    fn = _umma()
    T, B = 4, 4
    case = list(_make_case(B, T, [12, 13, 13, 5], seed=11, bases=[0, 28 * RS, 3, 9]))
    idx = torch.tensor([0, -1, 2, -1], dtype=torch.int32, device=DEV)
    case[13] = idx
    pool, kc, uc, gc = case[7], case[8], case[9], case[10]
    snap = [x.clone() for x in (pool, kc, uc, gc)]
    out = torch.full((B, T, HV, V), 7.0, dtype=torch.bfloat16, device=DEV)
    q, k, v, a, b, A_log, dt_bias = case[:7]
    fn(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=case[11],
        cache_base=case[12],
        scale=SCALE,
        flush_min=13,
        output=out,
        restart_hist_on_flush=False,
    )
    torch.cuda.synchronize()
    for r in (1, 3):
        assert torch.all(out[r] == 7.0), "padded row wrote output"
        for before, after in zip(snap, (pool, kc, uc, gc), strict=False):
            assert torch.equal(before[r], after[r]) or torch.equal(
                before[r].nan_to_num(), after[r].nan_to_num()
            ), "padded row touched pool/rings"


@pytest.mark.parametrize("T", [4, 8])
def test_multistep_horizon_with_commits(T):
    """Run 24 steps with the wrapper's cursor commit; every step's outputs must match a
    pure-fp32 recurrent reference that never uses the ring (S evolves token by token and
    the accepted count is always T)."""
    _skip_if_not_sm100()
    fn = _umma()
    B = 4
    fm = _fm_max(T)
    g = torch.Generator(device=DEV).manual_seed(5)
    case = _make_case(B, T, [0] * B, seed=5, bases=[0, 9 * RS, 30 * RS, 17 * RS])
    _, _, _, _, _, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    hl.copy_(torch.tensor([0, 4, 8, 12][:B], dtype=torch.int32, device=DEV) % fm)
    hl.zero_()
    S_true = pool.float().clone()
    worst = 0.0
    for _step in range(24):
        q = (torch.randn(B, T, H, K, generator=g, device=DEV)).bfloat16()
        k = (torch.randn(B, T, H, K, generator=g, device=DEV)).bfloat16()
        v = (torch.randn(B, T, HV, V, generator=g, device=DEV) * 0.5).bfloat16()
        a = (torch.randn(B, T, HV, generator=g, device=DEV) * 0.5).bfloat16()
        b = torch.randn(B, T, HV, generator=g, device=DEV).bfloat16()
        y = fn(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=True,
        )
        # accept all T tokens: verify rows len += T (flush rows were reset to 0 -> T)
        hl.add_(T)
        torch.cuda.synchronize()
        for r in range(B):
            ref_y, _, _, _, _ = _ref_fp32(
                q[r],
                k[r],
                v[r],
                a[r],
                b[r],
                A_log,
                dt_bias,
                S_true[r],
                kc[r],
                uc[r],
                gc[r],
                0,
            )
            worst = max(worst, (y[r].float() - ref_y).abs().max().item())
            # advance the fp32 truth by the T tokens
            khat = F.normalize(k[r].float(), dim=-1)
            for t in range(T):
                la = -torch.exp(A_log.float()) * F.softplus(
                    a[r, t].float() + dt_bias.float()
                )
                beta = torch.sigmoid(b[r, t].float())
                k_hv = khat[t].repeat_interleave(HV // H, dim=0)
                S = S_true[r] * torch.exp(la)[:, None, None]
                u = (v[r, t].float() - torch.einsum("hvk,hk->hv", S, k_hv)) * beta[
                    :, None
                ]
                S_true[r] = S + u[:, :, None] * k_hv[:, None, :]
    assert worst < 2 * Y_TOL, f"multi-step |y - fp32 truth| = {worst:.2e}"


def test_build_constants_match_the_tests():
    """The tests' W / RING literals are the kernel's: a window of the wrong depth would pass vacuously."""
    m = __import__("flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64", fromlist=["x"])
    assert (m.W_RING, m.RING_SLOTS, m.RING_MASK, m.W_CHUNKS) == (
        W,
        RING,
        RING - 1,
        W // 16,
    )
    for T in (1, 4, 8):
        assert m.GdnReplayMtpUmma(T, H, HV).nu == min(W, RING - T) == _p_max(T)
        assert m.GdnReplayMtpUmma(T, H, HV, pmax=16).nu == 16
        assert m.GdnReplayMtpUmma(T, H, HV, pmax=_p_max(T)).nu == _p_max(T)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
def test_window_rows_derivation_is_bit_exact_vs_full_window(T):
    """The wrapper compiles the shared-memory u window for flush_min - 1 + T rows (window_rows): on histories that
    obey the commit rule (P <= flush_min - 1 + T) outputs, committed state and every ring must be bit-identical
    (NaN-poisoned slots included) to the build with the whole window (window_rows = nu). Rows deeper than the
    derived window are truncated, which the same comparison shows: the check is able to fail."""
    _skip_if_not_sm100()
    fn = _umma()
    top = _p_max(T)

    def call(case, fm, wr):
        y = fn(
            case[5],
            case[3],
            case[6],
            q=case[0],
            k=case[1],
            v=case[2],
            b=case[4],
            initial_state_source=case[7],
            initial_state_indices=case[13],
            k_cache=case[8],
            u_cache=case[9],
            g_cache=case[10],
            hist_len=case[11],
            cache_base=case[12],
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            window_rows=wr,
        )
        torch.cuda.synchronize()
        return y, case[7], case[8], case[9], case[10]

    for fm in sorted({max(1, _fm_max(T) - 8), max(1, _fm_max(T) - 3), _fm_max(T)}):
        deepest = min(fm - 1 + T, top)
        hist = sorted(
            {0, 1, 2, 15, 16, 17, 31, 32, fm - 1, fm, deepest} & set(range(deepest + 1))
        )
        bases = [_SWEEP_BASES[j % 8] for j in range(len(hist))]
        res = {}
        for name, wr in (("derived", None), ("full", top)):
            res[name] = call(
                _make_case(len(hist), T, hist, 300 + T + fm, bases), fm, wr
            )
        for key, x, y_ in zip(
            ("out", "pool", "k ring", "u ring", "g ring"),
            res["derived"],
            res["full"],
            strict=False,
        ):
            assert torch.equal(_bits(x), _bits(y_)), (
                f"{key} differs (T={T}, flush_min={fm}, hist={hist})"
            )
    fm = max(1, _fm_max(T) - 8)
    if (
        fm - 1 + T < top
    ):  # a window shallower than P truncates the deepest rows: the comparison can fail
        hist = [min(fm - 1 + T + 1, top)]
        outs = [
            call(_make_case(1, T, hist, 399 + T, [5]), fm, wr)[0]
            for wr in (fm - 1 + T, top)
        ]
        assert not torch.equal(_bits(outs[0]), _bits(outs[1])), (
            "a window shallower than P must change the result"
        )


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
def test_launch_shared_memory_matches_the_budget(T):
    """Launch shared memory as the profiler reports it (kineto) equals the module's formula for the default window and
    for the window derived from a lower flush_min, and stays inside the per-CTA budget of the build's CTAs/SM
    (57,344 B for 4 CTAs/SM at W_RING = 32)."""
    _skip_if_not_sm100()
    import json
    import tempfile

    from torch.profiler import ProfilerActivity, profile

    import flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 as m

    fn = _umma()
    budget = m._SMEM_PER_SM // m._SMEM_CTAS_PER_SM - m._SMEM_RESERVED_PER_CTA
    for fm in sorted({_fm_max(T), max(1, _fm_max(T) - 8)}):
        wr = min(fm - 1 + T, _p_max(T))
        case = _make_case(2, T, [0, 1], 5, [0, 3])

        def launch():
            fn(
                case[5],
                case[3],
                case[6],
                q=case[0],
                k=case[1],
                v=case[2],
                b=case[4],
                initial_state_source=case[7],
                initial_state_indices=case[13],
                k_cache=case[8],
                u_cache=case[9],
                g_cache=case[10],
                hist_len=case[11],
                cache_base=case[12],
                scale=SCALE,
                flush_min=fm,
                restart_hist_on_flush=False,
            )

        launch()  # compile outside the profiled launch
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            launch()
            torch.cuda.synchronize()
        tf = tempfile.mktemp(suffix=".json")
        prof.export_chrome_trace(tf)
        with open(tf) as fh:
            ev = [e for e in json.load(fh)["traceEvents"] if e.get("cat") == "kernel"]
        os.remove(tf)
        assert len(ev) == 1, [e["name"] for e in ev]
        smem = int(ev[0]["args"]["shared memory"])
        assert smem == m._SMEM_FIXED + 256 * max(wr, 16), (T, fm, wr, smem)
        assert smem <= budget, (T, fm, smem, budget)


@pytest.mark.parametrize("T", [1, 3, 4, 8])
def test_multistep_partial_acceptance_wraps_the_ring(T):
    """Random acceptance 1..T per row and step, the ring-safe maximum flush_min (explicit), the wrapper's cursor commit: the
    window start walks around the whole 64-slot ring many times, flushes at every P in [W - T + 1, W] (window depths
    up to the 32-row window, chunk 1), and every output must match the token-by-token fp32 recurrence over the
    ACCEPTED tokens only."""
    _skip_if_not_sm100()
    fn = _umma()
    B = 4
    g = torch.Generator(device=DEV).manual_seed(900 + T)
    rng = torch.Generator().manual_seed(77 + T)
    case = _make_case(B, T, [0] * B, seed=5 + T, bases=[0, 9 * RS, 30 * RS, 17 * RS])
    _, _, _, _, _, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    hl.zero_()
    S_true = pool.float().clone()
    worst = 0.0
    seen_depth = 0
    flushes = 0
    fm_def = _fm_max(
        T
    )  # explicit: the T = 1 default (16) would never reach window chunk 1
    for _step in range(150 if T > 1 else 200):
        q = torch.randn(B, T, H, K, generator=g, device=DEV).bfloat16()
        k = torch.randn(B, T, H, K, generator=g, device=DEV).bfloat16()
        v = (torch.randn(B, T, HV, V, generator=g, device=DEV) * 0.5).bfloat16()
        a = (torch.randn(B, T, HV, generator=g, device=DEV) * 0.5).bfloat16()
        b = torch.randn(B, T, HV, generator=g, device=DEV).bfloat16()
        seen_depth = max(seen_depth, int(hl.max()))
        flushes += int((hl >= fm_def).sum())
        y = fn(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm_def,
            restart_hist_on_flush=True,
        )
        acc = torch.randint(1, T + 1, (B,), generator=rng)
        hl.add_(acc.to(device=DEV, dtype=torch.int32))
        torch.cuda.synchronize()
        for r in range(B):
            ref_y, _, _, _, _ = _ref_fp32(
                q[r],
                k[r],
                v[r],
                a[r],
                b[r],
                A_log,
                dt_bias,
                S_true[r],
                kc[r],
                uc[r],
                gc[r],
                0,
            )
            worst = max(worst, (y[r].float() - ref_y).abs().max().item())
            khat = F.normalize(k[r].float(), dim=-1)
            for t in range(int(acc[r])):
                la = -torch.exp(A_log.float()) * F.softplus(
                    a[r, t].float() + dt_bias.float()
                )
                beta = torch.sigmoid(b[r, t].float())
                k_hv = khat[t].repeat_interleave(HV // H, dim=0)
                S = S_true[r] * torch.exp(la)[:, None, None]
                u = (v[r, t].float() - torch.einsum("hvk,hk->hv", S, k_hv)) * beta[
                    :, None
                ]
                S_true[r] = S + u[:, :, None] * k_hv[:, None, :]
    assert 16 < fm_def <= seen_depth <= W, (
        seen_depth
    )  # the window really got deep (chunk 1), never past W
    assert flushes >= B, flushes  # and every row flushed at least once
    assert worst < 2 * Y_TOL, (
        f"partial-acceptance multi-step |y - fp32 truth| = {worst:.2e}"
    )


@pytest.mark.parametrize("T", [4, 8])
def test_parity_with_flashinfer_fp16_state(T):
    """Same inputs through FlashInfer's gated_delta_rule_mtp_ucache_flush (fp16_state arm)."""
    _skip_if_not_sm100()
    w_ring, _ = _umma_imports()
    from flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush import (
        W_RING as _HMMA_W_RING,
    )

    if w_ring != _HMMA_W_RING:
        pytest.skip(
            f"UMMA W_RING ({w_ring}) != HMMA reference W_RING ({_HMMA_W_RING}); "
            "the two kernels no longer implement the same window contract"
        )
    fi = _flashinfer_fp16_state().gated_delta_rule_mtp_ucache_flush
    fm = _fm_max(T)
    hist = [fm - 1, fm, fm, 3, 0, fm - 1, fm, 1]
    bases = [0, 28, 5, 30, 17, 31, 2, 9]
    c1 = _make_case(8, T, hist, seed=21, bases=bases, poison=False)
    c2 = [x.clone() for x in c1]
    y1 = _run(_umma(), c1, fm)
    y2 = _run(fi, c2, fm)
    torch.cuda.synchronize()
    dy = (y1.float() - y2.float()).abs().max().item()
    dS = (c1[7].float() - c2[7].float()).abs().max().item()
    du = (c1[9].float() - c2[9].float()).abs().max().item()
    dg = (c1[10] - c2[10]).abs().max().item()
    assert dy < Y_TOL and dS < STATE_TOL and du < 2 * U_TOL and dg < 1e-4, (
        dy,
        dS,
        du,
        dg,
    )


# ---------------------------------------------------------------------------
# unf_vf: the flush re-solves the window's corrections from the v / beta rings against S0.
# Its inputs must be CONSISTENT: rings produced by actually running the P history tokens
# through the recurrence from the checkpoint (then stored u == re-solved u up to rounding).
# ---------------------------------------------------------------------------
def _make_consistent_case(B, T, hist_lens, seed, bases):
    f = torch.float32
    g = torch.Generator(device=DEV).manual_seed(seed)
    case = list(_make_case(B, T, [0] * B, seed, bases, poison=True))
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    vc = torch.full((B, HV, RING, V), float("nan"), dtype=torch.bfloat16, device=DEV)
    bc = torch.full((B, HV, RING), float("nan"), dtype=f, device=DEV)
    grp = HV // H
    for r in range(B):
        P = hist_lens[r]
        S = pool[r].to(f).clone()
        gam = torch.zeros(HV, dtype=f, device=DEV)
        for j in range(P):
            slot = (bases[r] + j) % RING
            kj = F.normalize(
                torch.randn(H, K, generator=g, device=DEV), dim=-1
            ).bfloat16()
            vj = (torch.randn(HV, V, generator=g, device=DEV) * 0.5).bfloat16()
            aj = (torch.randn(HV, generator=g, device=DEV) * 0.5).bfloat16()
            bj = torch.randn(HV, generator=g, device=DEV).bfloat16()
            la = -torch.exp(A_log.to(f)) * F.softplus(aj.to(f) + dt_bias.to(f))
            beta = torch.sigmoid(bj.to(f))
            k_hv = kj.to(f).repeat_interleave(grp, dim=0)
            S = S * torch.exp(la)[:, None, None]
            u = (vj.to(f) - torch.einsum("hvk,hk->hv", S, k_hv)) * beta[:, None]
            S = S + u[:, :, None] * k_hv[:, None, :]
            gam = gam + la
            kc[r, :, slot] = kj
            uc[r, :, slot] = u.bfloat16()
            gc[r, :, slot] = gam
            vc[r, :, slot] = vj
            bc[r, :, slot] = beta
        hl[r] = P
    return case, vc, bc


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("poison_u", [False, True])
def test_vf_matches_reference_and_ignores_u_on_flush(T, poison_u):
    """unf_vf against the fp32 oracle on consistent histories; with poison_u the u-ring
    windows of the FLUSH rows are NaN — the re-solve must not read them."""
    _skip_if_not_sm100()
    _skip_if_vf_unsupported()
    fn = _umma()
    fm = _fm_max(T)
    hist = [fm, fm - 1, W, 2, fm, 0, fm - 1, min(W, fm + 1)]
    bases = [28, 5, 30, 17, 0, 9, 31, 20]
    B = len(hist)
    case, vc, bc = _make_consistent_case(B, T, hist, seed=40 + T, bases=bases)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    pool0, kc0, uc0, gc0 = pool.clone(), kc.clone(), uc.clone(), gc.clone()
    if poison_u:
        for r in range(B):
            if hist[r] >= fm:
                for j in range(hist[r]):
                    uc[r, :, (bases[r] + j) % RING] = float("nan")
    y = fn(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=fm,
        restart_hist_on_flush=False,
        v_cache=vc,
        beta_cache=bc,
    )
    torch.cuda.synchronize()
    ey = es = 0.0
    for r in range(B):
        P = hist[r]
        kc_l, uc_l, gc_l = _logical(kc0[r], uc0[r], gc0[r], bases[r])
        y_ref, S_ref, _, _, gam = _ref_fp32(
            q[r], k[r], v[r], a[r], b[r], A_log, dt_bias, pool0[r], kc_l, uc_l, gc_l, P
        )
        ey = max(ey, (y[r].float() - y_ref).abs().max().item())
        if P >= fm:
            es = max(es, (pool[r].float() - S_ref).abs().max().item())
        app = [(bases[r] + P + s) % RING for s in range(T)]
        # v / beta ring appends
        assert torch.equal(vc[r][:, app], v[r].transpose(0, 1)), f"row {r}: v append"
        beta = torch.sigmoid(b[r].float()).transpose(0, 1)
        assert (bc[r][:, app] - beta).abs().max().item() < 1e-5, f"row {r}: beta append"
    assert ey < Y_TOL and es < STATE_TOL, (ey, es)


@pytest.mark.parametrize("T", [4, 8])
def test_uf_vf_agree_on_consistent_histories(T):
    _skip_if_not_sm100()
    _skip_if_vf_unsupported()
    fn = _umma()
    fm = _fm_max(T)
    hist = [fm, fm - 1, W, fm, 3, fm + 1 if fm < W else W]
    bases = [3, 30, 12, 27, 0, 22]
    outs = {}
    for mode in ("uf", "vf"):
        case, vc, bc = _make_consistent_case(len(hist), T, hist, seed=77, bases=bases)
        q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
        extra = dict(v_cache=vc, beta_cache=bc) if mode == "vf" else {}
        y = fn(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            **extra,
        )
        torch.cuda.synchronize()
        outs[mode] = (y.float(), pool.float())
    dy = (outs["uf"][0] - outs["vf"][0]).abs().max().item()
    dS = (outs["uf"][1] - outs["vf"][1]).abs().max().item()
    assert dy < Y_TOL and dS < STATE_TOL, (dy, dS)


@pytest.mark.parametrize("T", [4, 8])
def test_vf_multistep_horizon_with_commits(T):
    """24 steps of unf_vf with cursor commits against a token-by-token fp32 recurrence:
    every flush re-solves from the v / beta rings the kernel itself appended."""
    _skip_if_not_sm100()
    _skip_if_vf_unsupported()
    fn = _umma()
    B = 4
    fm = _fm_max(T)
    g = torch.Generator(device=DEV).manual_seed(6)
    case = _make_case(B, T, [0] * B, seed=6, bases=[0, 9, 30, 17])
    _, _, _, _, _, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    vc = torch.zeros(B, HV, RING, V, dtype=torch.bfloat16, device=DEV)
    bc = torch.zeros(B, HV, RING, dtype=torch.float32, device=DEV)
    S_true = pool.float().clone()
    worst = 0.0
    for _step in range(24):
        q = torch.randn(B, T, H, K, generator=g, device=DEV).bfloat16()
        k = torch.randn(B, T, H, K, generator=g, device=DEV).bfloat16()
        v = (torch.randn(B, T, HV, V, generator=g, device=DEV) * 0.5).bfloat16()
        a = (torch.randn(B, T, HV, generator=g, device=DEV) * 0.5).bfloat16()
        b = torch.randn(B, T, HV, generator=g, device=DEV).bfloat16()
        y = fn(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=True,
            v_cache=vc,
            beta_cache=bc,
        )
        hl.add_(T)
        torch.cuda.synchronize()
        for r in range(B):
            ref_y, _, _, _, _ = _ref_fp32(
                q[r],
                k[r],
                v[r],
                a[r],
                b[r],
                A_log,
                dt_bias,
                S_true[r],
                kc[r],
                uc[r],
                gc[r],
                0,
            )
            worst = max(worst, (y[r].float() - ref_y).abs().max().item())
            khat = F.normalize(k[r].float(), dim=-1)
            for t in range(T):
                la = -torch.exp(A_log.float()) * F.softplus(
                    a[r, t].float() + dt_bias.float()
                )
                beta = torch.sigmoid(b[r, t].float())
                k_hv = khat[t].repeat_interleave(HV // H, dim=0)
                S = S_true[r] * torch.exp(la)[:, None, None]
                u = (v[r, t].float() - torch.einsum("hvk,hk->hv", S, k_hv)) * beta[
                    :, None
                ]
                S_true[r] = S + u[:, :, None] * k_hv[:, None, :]
    assert worst < 2 * Y_TOL, f"vf multi-step |y - fp32 truth| = {worst:.2e}"


# --------------------------------------------------------------------------- stochastic rounding
_PHILOX = (
    0xD2511F53,
    0xCD9E8D57,
    0x9E3779B9,
    0xBB67AE85,
)  # round A, round B, key A, key B


def _philox_noise13(seed, sidx, rounds):
    """13-bit cvt.rs noise of every element of state slot ``sidx`` ([HV, V, K]) under the
    kernel's scheme (FlashInfer Mamba's): Philox4x32 with key = seed and counter = flat
    state-pool offset of the element's 8-element chunk; word i -> pair i, even element
    rbits[12:0], odd element rbits[28:16]. numpy reference of include/flashinfer/mamba."""
    import numpy as np

    u64, m32 = np.uint64, np.uint64(0xFFFFFFFF)
    ra, rb, ka, kb = (u64(x) for x in _PHILOX)
    chunk = np.arange(HV * V * (K // 8), dtype=np.uint64) * u64(8) + u64(
        sidx * HV * V * K
    )
    c0, c1 = chunk & m32, chunk >> u64(32)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    k0, k1 = u64(seed & 0xFFFFFFFF), u64((seed >> 32) & 0xFFFFFFFF)
    for _ in range(rounds):
        pb, pa = c2 * rb, c0 * ra
        c0, c2 = (pb >> u64(32)) ^ c1 ^ k0, (pa >> u64(32)) ^ c3 ^ k1
        c1, c3 = pb & m32, pa & m32
        k0, k1 = (k0 + ka) & m32, (k1 + kb) & m32
    words = np.stack([c0, c1, c2, c3], axis=-1)  # [chunks, 4 pairs]
    pairs = np.stack([words & u64(0x1FFF), (words >> u64(16)) & u64(0x1FFF)], axis=-1)
    return torch.from_numpy(pairs.reshape(HV, V, K).astype(np.int32))


def _lcg_noise13(seed, sidx):
    """13-bit cvt.rs noise of state slot ``sidx`` ([HV, V, K]) under the kernel's "lcg" mode:
    (state row, half) seed = murmur3(id * golden + id_hi * c + key), key = murmur3 of the seed;
    pair n of the half (n = 4 chunk + i) takes LCG state n folded as s ^ (s >> 16). numpy
    reference of flashinfer/gdn_kernels (fmix32, LCG_* in _umma_helpers.py; _rmw)."""
    import numpy as np

    u64, m = np.uint64, np.uint64(0xFFFFFFFF)

    def fmix(x):
        x = x ^ (x >> u64(16))
        x = (x * u64(0x85EBCA6B)) & m
        x = x ^ (x >> u64(13))
        x = (x * u64(0xC2B2AE35)) & m
        return x ^ (x >> u64(16))

    def out(x):
        return x ^ (x >> u64(16))

    key = fmix(
        u64(seed & 0xFFFFFFFF) ^ fmix(u64((seed >> 32) & 0xFFFFFFFF) ^ u64(0x9E3779B9))
    )
    hv, v, g = np.meshgrid(
        np.arange(HV, dtype=np.uint64),
        np.arange(V, dtype=np.uint64),
        np.arange(2, dtype=np.uint64),
        indexing="ij",
    )
    cid = ((u64(sidx * HV) + hv) * u64(V) + v) * u64(2) + g
    x = fmix(
        ((cid & m) * u64(0x9E3779B9) + (cid >> u64(32)) * u64(0x85EBCA6B) + key) & m
    )
    words = []
    for _ in range(32):  # pair n = 4 chunk + i
        words.append(out(x))
        x = (x * u64(1664525) + u64(1013904223)) & m
    w = np.stack(words, axis=-1)  # [HV, V, 2, 32]
    pairs = np.stack(
        [w & u64(0x1FFF), (w >> u64(16)) & u64(0x1FFF)], axis=-1
    )  # [.., 32, 2]
    return torch.from_numpy(pairs.reshape(HV, V, K).astype(np.int32))


def _sr_case(T, vf, seed):
    fm = _fm_max(T)
    hist = [
        fm,
        fm - 1,
        fm,
        fm,
        0,
        fm,
    ]  # fm is the per-call-safe max P (not W: P + T <= RING)
    bases = [28 * RS, 5 * RS, 30 * RS, 17 * RS, 9 * RS, 0]
    if vf:
        case, vc, bc = _make_consistent_case(len(hist), T, hist, seed=seed, bases=bases)
        extra = dict(v_cache=vc, beta_cache=bc)
    else:
        case, extra = _make_case(len(hist), T, hist, seed, bases), {}
    return case, extra, hist, fm


def _sr_call(case, extra, fm, **kw):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() if isinstance(x, torch.Tensor) else x for x in case
    ]
    ex = {n: t.clone() for n, t in extra.items()}
    y = _umma()(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=fm,
        restart_hist_on_flush=False,
        **ex,
        **kw,
    )
    torch.cuda.synchronize()
    return y, pool, (kc, uc, gc, *ex.values())


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("vf", [False, True])
@pytest.mark.parametrize("mode", ["philox10", "philox5", "lcg"])
@pytest.mark.parametrize("hpc", [1, 4])
def test_stochastic_rounding_matches_reference_bits(T, vf, mode, hpc):
    """SR only changes how the flushed fp16 state is rounded: outputs and rings are
    bit-identical to round-to-nearest, verify rows keep their state, every flushed element
    is RN or its neighbour, and each rounding direction agrees with the reference noise
    bits (rounding up where RN went down needs noise >= 4096 of 8192; down where RN went up
    needs noise < 4096 -- cvt.rs's dropped-bits-plus-noise carry is ">= 8192", so the single
    dead-center draw of 4096 belongs to "up", not "down"). Noise from a different seed must
    violate that, which proves the check can fail. hpc 1 and 4 cover both RMW group sizes
    (16 / 32 TMEM columns)."""
    _skip_if_not_sm100()
    if vf:
        _skip_if_vf_unsupported()
    case, extra, hist, fm = _sr_case(T, vf, seed=70 + T)
    seed = 0x1234_5678_9ABC_DEF1 + T
    if mode == "lcg":
        kw = dict(stochastic_rounding="lcg")
        noise_of = _lcg_noise13
    else:
        rounds = int(mode[len("philox") :])
        kw = dict(stochastic_rounding="philox", philox_rounds=rounds)

        def noise_of(sd, sidx):
            return _philox_noise13(sd, sidx, rounds)

    pool0 = case[7].clone()
    y_rn, s_rn, rings_rn = _sr_call(case, extra, fm, heads_per_cta=hpc)
    y_sr, s_sr, rings_sr = _sr_call(
        case,
        extra,
        fm,
        heads_per_cta=hpc,
        **kw,
        rand_seed=torch.tensor([seed], dtype=torch.int64, device=DEV),
    )
    assert torch.equal(y_rn, y_sr)
    for x, z in zip(rings_rn, rings_sr, strict=True):
        assert torch.equal(x.nan_to_num(), z.nan_to_num())
    n_diff = n_bad = n_bad_ctl = 0
    for r, hist_r in enumerate(hist):
        if hist_r < fm:
            assert torch.equal(s_sr[r], pool0[r]) and torch.equal(s_rn[r], pool0[r])
            continue
        rn = s_rn[r].view(torch.int16).int().cpu()
        sr = s_sr[r].view(torch.int16).int().cpu()
        mag_rn, mag_sr = rn & 0x7FFF, sr & 0x7FFF
        normal = (mag_rn >= 0x0400) & (
            mag_rn < 0x7C00
        )  # 13-bit noise model: normal fp16
        assert torch.equal((rn >> 15)[normal], (sr >> 15)[normal]), "sign flipped"
        step = mag_sr - mag_rn
        assert step.abs().max().item() <= 1, "SR result is not RN or its neighbour"
        noise = noise_of(seed, r)
        ctl = noise_of(seed + 1, r)
        up, down = normal & (step == 1), normal & (step == -1)
        n_diff += int(up.sum() + down.sum())
        # cvt.rs.f16x2.f32 (sm_100a/sm_103a) resolves its internal "dropped-bits + noise"
        # carry with an inclusive >= 8192, so a noise draw of *exactly* 4096 (the sole
        # value equidistant from both ends of the 13-bit range, 1/8192 of the space) carries
        # into "round up" whenever the true value is itself an exact dropped-bits tie (RN's
        # own tie-break lands on the even/lower candidate in that case). That makes the
        # up/down split asymmetric at the midpoint: up is valid for noise in [4096, 8192),
        # down only for noise in [0, 4096). Confirmed on hardware (identical index, same
        # rn/sr bits, reproduced under both hpc=1 and hpc=4 grid configs): row 0 of the
        # T=1 case here, state[hv=32, v=111, k=80], draws noise13 == 4096 exactly and the
        # kernel rounds up both times -- not a W_RING-related regression (this cid/noise
        # derivation -- see _lcg_noise13 -- never reads W_RING/RING_SLOTS/base/P, only the
        # state-pool coordinates), just the one dead-center draw the bit-level model must
        # account for the same way the hardware does.
        bad_mask = ((noise < 4096) & up) | ((noise >= 4096) & down)
        n_bad += int(bad_mask.sum())
        n_bad_ctl += int(((ctl < 4096) & up).sum() + ((ctl >= 4096) & down).sum())
    total = sum(HV * V * K for hist_r in hist if hist_r >= fm)
    assert abs(n_diff / total - 0.25) < 0.05, n_diff / total  # E[min(f, 1 - f)] = 1/4
    assert n_bad == 0, (
        f"{n_bad} of {n_diff} rounding directions contradict the {mode} bits"
    )
    assert n_bad_ctl > 0.3 * n_diff, (n_bad_ctl, n_diff)


def test_stochastic_rounding_seed_and_graph():
    """Same seed -> bit-identical state; another seed -> different; the seed is read on
    device, so an in-place update reaches a captured CUDA graph; default is RN."""
    _skip_if_not_sm100()
    T = 4
    case, extra, hist, fm = _sr_case(T, False, seed=91)
    seed_t = torch.tensor([7], dtype=torch.int64, device=DEV)
    _, a1, _ = _sr_call(case, extra, fm, rand_seed=seed_t)
    _, a2, _ = _sr_call(case, extra, fm, rand_seed=seed_t)
    _, b1, _ = _sr_call(
        case, extra, fm, rand_seed=torch.tensor([8], dtype=torch.int64, device=DEV)
    )
    _, rn, _ = _sr_call(case, extra, fm)
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b1) and not torch.equal(a1, rn)

    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    pool_init = pool.clone()
    fn = _umma()
    kw = dict(
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=fm,
        restart_hist_on_flush=False,
        rand_seed=seed_t,
    )
    fn(A_log, a, dt_bias, **kw)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(A_log, a, dt_bias, **kw)
    for want_seed, want in ((7, a1), (8, b1)):
        pool.copy_(pool_init)
        seed_t.fill_(want_seed)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, want), f"graph replay with seed {want_seed}"


@pytest.mark.parametrize("vf", [False, True])
def test_lcg_unseeded_uses_clock(vf):
    """stochastic_rounding="lcg" without a seed: %clock-seeded, so two runs differ, while each
    stays RN or its neighbour with the expected ~1/4 of elements rounded the other way;
    "none" (and the default without rand_seed) is round-to-nearest."""
    _skip_if_not_sm100()
    if vf:
        _skip_if_vf_unsupported()
    case, extra, hist, fm = _sr_case(4, vf, seed=12)
    _, rn, _ = _sr_call(case, extra, fm)
    _, rn2, _ = _sr_call(
        case,
        extra,
        fm,
        stochastic_rounding="none",
        rand_seed=torch.tensor([3], dtype=torch.int64, device=DEV),
    )
    _, c1, _ = _sr_call(case, extra, fm, stochastic_rounding="lcg")
    _, c2, _ = _sr_call(case, extra, fm, stochastic_rounding="lcg")
    assert torch.equal(rn, rn2)
    assert not torch.equal(c1, c2)
    rows = [r for r, h in enumerate(hist) if h >= fm]
    for st in (c1, c2):
        step = (st[rows].view(torch.int16).int() & 0x7FFF) - (
            rn[rows].view(torch.int16).int() & 0x7FFF
        )
        assert step.abs().max().item() <= 1
        assert abs((step != 0).float().mean().item() - 0.25) < 0.05


def test_stochastic_rounding_rejects_bad_seed():
    _skip_if_not_sm100()
    case, extra, _, fm = _sr_case(4, False, seed=5)
    for bad in (
        torch.tensor([1, 2], dtype=torch.int64, device=DEV),
        torch.tensor([1], dtype=torch.int32, device=DEV),
        torch.tensor([1], dtype=torch.int64),
    ):
        with pytest.raises(ValueError):
            _sr_call(case, extra, fm, rand_seed=bad)
    with pytest.raises(ValueError):
        _sr_call(
            case,
            extra,
            fm,
            rand_seed=torch.tensor([1], dtype=torch.int64, device=DEV),
            philox_rounds=0,
        )
    with pytest.raises(ValueError):
        _sr_call(case, extra, fm, stochastic_rounding="philox")  # philox needs a seed
    with pytest.raises(ValueError):
        _sr_call(case, extra, fm, stochastic_rounding="pcg")


# ---------------------------------------------------------------------------
# P-adaptive window skipping (rows j >= P of the W-row window are dead).  The kernel skips every
# 16-row chunk the live window does not reach (chunk c while P <= 16 c), predicates the ring loads of dead rows off and copies
# only the live u-window rows; these tests sweep every live depth P against the fp32 oracle (ring
# slots outside the window are NaN-poisoned, so a dead row reaching an MMA fails loudly) and
# compare the kernel bit-for-bit against a reference build that always processes the full window.
# ---------------------------------------------------------------------------
_SWEEP_BASES = [0, 28 * RS, 5 * RS, 30 * RS, 17 * RS, 31 * RS, 20 * RS, 25 * RS]


def _sweep_P(T, pmax):
    if T in (4, 8):
        return list(range(pmax + 1))
    fm = _fm_max(T)
    # 16-row chunk boundaries of every chunk of the window (15/16/17, 31/32/33, 47/48; 49 is past the window)
    edges = {c * 16 + d for c in range(1, W // 16 + 1) for d in (-1, 0, 1)}
    return sorted(
        {
            p
            for p in (0, 1, 2, 7, 8, 15, 16, 17, fm - 1, fm, pmax, *edges)
            if 0 <= p <= pmax
        }
    )


def _sweep_cases(T):
    """(hist, bases, flush_min, seed) tuples covering every live depth P: at the default flush_min = W - T + 1
    (rows with P >= it flush, P up to W), with every row a verify row (same flush_min, P <= W - T) and with
    flush_min = 1 (every row with P >= 1 flushes: the flush path at every depth, chunk edges included)."""
    fmd = _fm_max(T)
    out = []
    for mode, fm, pmax in (
        ("flush", fmd, _p_max(T)),
        ("verify", fmd, fmd - 1),
        ("allflush", 1, _p_max(T)),
    ):
        Ps = _sweep_P(T, pmax)
        for i in range(0, len(Ps), 8):
            chunk = Ps[i : i + 8]
            bases = [_SWEEP_BASES[(i + j) % 8] for j in range(len(chunk))]
            out.append(
                (
                    chunk,
                    bases,
                    fm,
                    1000 * T + 17 * i + {"flush": 0, "verify": 5, "allflush": 9}[mode],
                )
            )
    return out


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
def test_every_live_depth_matches_fp32_reference(T):
    _skip_if_not_sm100()
    for hist, bases, fm, seed in _sweep_cases(T):
        _check_case(T, hist, bases, fm, seed=seed)


@pytest.mark.parametrize("hpc", [1, 2, 4])
def test_live_depth_sweep_heads_per_cta(hpc):
    _skip_if_not_sm100()
    for T in (4, 8):
        for hist, bases, fm, seed in _sweep_cases(T)[::2]:
            _check_case(T, hist, bases, fm, seed=seed + hpc, hpc=hpc)


@pytest.mark.parametrize("fm_kind", ["default", "low"])
def test_chunk_boundary_depths_with_padded_rows(fm_kind):
    """P around the 16-row chunk boundaries (15/16/17, 31/32) mixed with padded rows (state index < 0):
    padded rows must stay untouched while the live rows around them skip / run chunk 1. flush_min is the
    default (only the deepest rows flush) or 16 (every row with P >= 16 flushes)."""
    _skip_if_not_sm100()
    fn = _umma()
    T, B = 4, 8
    fm = _fm_max(T) if fm_kind == "default" else 16
    edges = [e for e in (15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64) if e <= _p_max(T)]
    hist = (edges + [16, 17, 15, 31])[:8]
    case = list(_make_case(B, T, hist, seed=77, bases=_SWEEP_BASES))
    pads = (1, 6)
    idx = torch.tensor([0, -1, 2, 3, 4, 5, -1, 7], dtype=torch.int32, device=DEV)
    case[13] = idx
    pool, kc, uc, gc = case[7], case[8], case[9], case[10]
    snap = [x.clone() for x in (pool, kc, uc, gc)]
    out = torch.full((B, T, HV, V), 7.0, dtype=torch.bfloat16, device=DEV)
    q, k, v, a, b, A_log, dt_bias = case[:7]
    fn(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=case[11],
        cache_base=case[12],
        scale=SCALE,
        flush_min=fm,
        output=out,
        restart_hist_on_flush=False,
        window_rows=_p_max(T),
    )
    torch.cuda.synchronize()
    for r in pads:
        assert torch.all(out[r] == 7.0), "padded row wrote output"
        for before, after in zip(snap, (pool, kc, uc, gc), strict=False):
            assert torch.equal(
                before[r].view(
                    torch.int16 if before.element_size() == 2 else torch.int32
                ),
                after[r].view(
                    torch.int16 if after.element_size() == 2 else torch.int32
                ),
            ), "padded row touched pool/rings"
    for r in (0, 2, 3, 4, 5, 7):
        assert torch.isfinite(out[r].float()).all(), (
            f"row {r}: non-finite output (dead row leaked)"
        )


def _bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("hpc", [2, 4])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_l2pf_prefetch_is_a_pure_hint(monkeypatch, T, hpc, depth):
    """GDN_MTP_UMMA_L2PF = N only adds cp.async.bulk.prefetch.L2 hints: outputs, state and rings must be bit-identical
    to the default build (its entry loop used to rebind the mbarrier phase parity of the producer warp, which only
    shows with more than one value head per CTA)."""
    _skip_if_not_sm100()
    import functools

    import flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 as m

    fm = _fm_max(T)
    hist = [0, 3, fm - 1, fm, 20, 7, 11, 16]
    bases = [0, 5 * RS, 31 * RS, 17 * RS, 9 * RS, 30 * RS, 2 * RS, 24 * RS]
    fn = functools.partial(
        m.gated_delta_rule_mtp_ucache_flush_umma_ring64, heads_per_cta=hpc
    )
    res = {}
    for pf in (0, depth):
        monkeypatch.setattr(m, "_L2PF", pf)
        case = _make_case(len(hist), T, hist, 11, bases)
        c = tuple(x.clone() for x in case)
        out = _run(fn, c, fm)
        torch.cuda.synchronize()
        res[pf] = (out, c[7], c[8], c[9], c[10])
    for name, x, y in zip(
        ("out", "pool", "k ring", "u ring", "g ring"), res[0], res[depth], strict=False
    ):
        assert torch.equal(_bits(x), _bits(y)), (
            f"{name} differs with L2PF={depth} (T={T}, hpc={hpc})"
        )


@pytest.mark.parametrize("T", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("B,repeats", [(20, 1000), (48, 300)])
def test_repeated_launches_are_bit_stable(T, B, repeats):
    """The single-buffered u window of a multi-head CTA is released by the consumers while the next head's window
    is bulk-copied over it: the release must come after the rows were read. A release that overtook a still-queued
    shared load showed as rows of head h + 1 read in head h (about 1 launch in 700 at T = 5..7, B = 20 / 48), so one
    launch must not be trusted: every repeat of the same inputs has to reproduce the first one bit for bit.
    B = 20 (one latency-bound wave) shows the race far more often than B = 48 or 130."""
    _skip_if_not_sm100()
    import random

    fn = _umma()
    fm = _fm_max(T)
    rng = random.Random(T)
    hist = [fm - 1, fm, _p_max(T), _p_max(T) - 1] + [
        rng.randint(0, _p_max(T)) for _ in range(B - 4)
    ]
    bases = [rng.randint(0, RING - 1) for _ in range(B)]
    case = _make_case(B, T, hist, 21, bases)
    ref = None
    for rep in range(repeats):
        c = tuple(x.clone() for x in case)
        out = _run(fn, c, fm)
        torch.cuda.synchronize()
        got = (out, c[7], c[8], c[9], c[10])
        if ref is None:
            ref = got
            continue
        for name, x, y in zip(
            ("out", "pool", "k ring", "u ring", "g ring"), ref, got, strict=False
        ):
            assert torch.equal(_bits(x), _bits(y)), (
                f"{name} differs from the first launch at repeat {rep} (T={T}, B={B})"
            )


@pytest.mark.parametrize("T", [1, 4, 8])
def test_default_flush_min_is_ring_safe(T):
    """flush_min=None must resolve to min(W_RING - T + 1, RING - 2 T + 1), capped at W_CHUNK = 16 for T = 1 (= W - T + 1 at W = 48 on the 64-slot ring;
    at W_RING == RING == 32 the old default W_RING - T + 1 (29 at T = 4) would let a verify row reach P = RING - T + 1,
    past the window the kernel reads). Rows at P = fm .. nu must flush under the default, exactly as with the
    explicit safe value (bit-identical outputs, state and rings)."""
    _skip_if_not_sm100()
    import flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 as m

    fm = min(m.W_RING - T + 1, RING - 2 * T + 1)
    if T == 1:
        fm = min(fm, m.W_CHUNK)
    hist = list(
        range(0, min(_p_max(T), fm - 1 + T) + 1)
    )  # P <= flush_min - 1 + T: the default window
    B = len(hist)
    case_a = _make_case(B, T, hist, 21 + T, [5] * B)
    case_b = tuple(x.clone() for x in case_a)
    y_a = _run(_umma(), case_a, None)
    y_b = _run(_umma(), case_b, fm)
    torch.cuda.synchronize()
    for x, y_ in (
        (y_a, y_b),
        (case_a[7], case_b[7]),
        (case_a[8], case_b[8]),
        (case_a[9], case_b[9]),
        (case_a[10], case_b[10]),
    ):
        assert torch.equal(
            x.contiguous().flatten().view(torch.uint8),
            y_.contiguous().flatten().view(torch.uint8),
        )
    # the deepest legal rows really did flush (state changed) under the default
    ref = _make_case(B, T, hist, 21 + T, [5] * B)[7]
    assert not torch.equal(case_a[7][B - 1], ref[B - 1])


@pytest.mark.parametrize("T", [1, 4, 8])
def test_flush_min_bounds_and_warning(T):
    """flush_min outside [1, W_RING - T + 1] is rejected; values above the safe default RING - 2 T + 1 are
    accepted (single calls, e.g. every row a verify row) but warn: a verify row can then advance past
    RING - T and its appends overwrite the live window."""
    _skip_if_not_sm100()
    import warnings

    import flashinfer.gdn_kernels.gdn_replay_mtp_umma_ring64 as m

    fn = _umma()
    upper = m.W_RING - T + 1
    safe = min(m.W_RING - T + 1, RING - 2 * T + 1)
    for bad in (0, -1, upper + 1):
        with pytest.raises(AssertionError, match="flush_min"):
            _run(fn, _make_case(2, T, [0, 1], 3, [0, 0]), bad)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run(fn, _make_case(2, T, [0, 1], 3, [0, 0]), upper)
        _run(fn, _make_case(2, T, [0, 1], 3, [0, 0]), safe)
    msgs = [str(w.message) for w in rec if "flush_min" in str(w.message)]
    assert (len(msgs) == 1) == (upper > safe), (T, upper, safe, msgs)


# ---------------------------------------------------------------------------
# backend="umma_ring64" of the public gated_delta_rule_mtp_ucache_flush


def _public_call(case, flush_min=None, **extra):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    return _flashinfer_fp16_state().gated_delta_rule_mtp_ucache_flush(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=flush_min,
        restart_hist_on_flush=False,
        **extra,
    )


@pytest.mark.parametrize("T", [1, 4, 8])
def test_public_backend_umma_ring64_matches_the_direct_call(T):
    """backend='umma_ring64' hands the call to the ring-64 kernel unchanged: outputs, state pool and rings are
    bit-identical to calling gated_delta_rule_mtp_ucache_flush_umma_ring64 directly (verify and flush rows)."""
    _skip_if_not_sm100()
    hist = list(range(0, _default_fm(T) + 1))
    B = len(hist)
    case_a = _make_case(B, T, hist, 40 + T, [3] * B)
    case_b = tuple(x.clone() for x in case_a)
    y_a = _public_call(case_a, backend="umma_ring64")
    y_b = _run(_umma(), case_b, None)
    torch.cuda.synchronize()
    for x, y_ in (
        (y_a, y_b),
        (case_a[7], case_b[7]),
        (case_a[8], case_b[8]),
        (case_a[9], case_b[9]),
        (case_a[10], case_b[10]),
    ):
        assert torch.equal(_bits(x), _bits(y_))


def test_public_backend_umma_ring64_needs_64_slot_rings():
    """32-slot rings (the HMMA / W=16 UMMA contract) are refused with a message naming the ring depth."""
    _skip_if_not_sm100()
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _make_case(
        2, 4, [0, 1], 5, [0, 0]
    )
    case = (
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool,
        kc[:, :, :32].contiguous(),
        uc[:, :, :32].contiguous(),
        gc[:, :, :32].contiguous(),
        hl,
        cb,
        idx,
    )
    with pytest.raises(ValueError, match="64-slot rings"):
        _public_call(case, backend="umma_ring64")


def test_public_backend_umma_ring64_rejects_an_fp16_u_ring():
    """The fp16 u ring (vLLM's u-cache dtype) is served by backend='umma' only."""
    _skip_if_not_sm100()
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _make_case(
        2, 4, [0, 1], 5, [0, 0]
    )
    case = (q, k, v, a, b, A_log, dt_bias, pool, kc, uc.half(), gc, hl, cb, idx)
    with pytest.raises(ValueError, match="bf16 k and u rings"):
        _public_call(case, backend="umma_ring64")


def test_auto_backend_never_picks_umma_ring64():
    """backend='auto' keeps the 32-slot contract: 64-slot rings fail its ring validation instead of being
    served by the ring-64 kernel behind the caller's back."""
    _skip_if_not_sm100()
    case = _make_case(2, 4, [0, 1], 5, [0, 0])
    with pytest.raises(AssertionError, match="k_cache must be"):
        _public_call(case, backend="auto")
