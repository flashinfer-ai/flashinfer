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
Correctness tests for the KDA ucache verify+flush kernel
(flashinfer/kda_kernels/kda_decode_bf16_wy_ucache_flush.py) — the KDA
(Kimi Delta Attention / Kimi K3) analogue of the GDN replayssm kernel in
tests/gdn/test_decode_ucache.py.

HOW THIS TESTS, IN PLAIN WORDS
  The kernel is a fast, fancy implementation (CuTe-DSL, tensor cores, bf16
  storage) of math that is simple to state. So we compute the same answer
  twice:

    1. with the KERNEL (fast, low precision, complicated code), and
    2. with `_ref_fp32` below - a deliberately slow, obviously-correct
       PyTorch loop that follows the KDA recurrence one token at a time,
       entirely in fp32. ~40 lines, no tricks. Read it top to bottom.

  If |kernel - reference| stays within low-precision rounding noise, the
  kernel is computing the right function.

  KDA vs GDN, the one structural difference: the decay gate is a
  PER-KEY-CHANNEL vector (g in R^128 per token per head, Kimi K3
  lower-bound gate  g_log = lb * sigmoid(exp(A_log) * (g + dt_bias)),
  lb = -5), not a per-head scalar. Consequently the cumulative-log-decay
  ring g_cache is [pool, H, RING, K] fp32 (a vector per slot) instead of
  GDN's [pool, HV, RING] scalar-per-slot, and the replay weights
  w_j = exp(G_P - G_j) are per-channel tiles applied to the cached keys.
  Everything else — the ring addressing, cursor semantics, flush
  condition, u-not-v caching (u_t = beta_t * (v_t - S_decayed k_t)) —
  matches the GDN PR #4081 contract:

    - verify from a bare checkpoint          (hist_len = 0)
    - verify with ring-history replay        (hist_len = 12)
    - the fold: ring folded into the state   (hist_len = 13 >= flush_min)
      -> here we also check the COMMITTED STATE the kernel wrote back,
         not just the output tokens.

  Geometry: Kimi K3 at serving TP — H == HV == 12 (no GQA), K == V == 128.
  Dtypes: bf16 io / bf16 state / bf16 rings (KDA is bf16-enforced across
  the stack; "16-bit state only"); g_cache is fp32 always.

Run:
  pytest tests/kda/test_decode_ucache.py -v
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.long_running

DEV = "cuda"
# Kimi K3 KDA geometry at serving TP; T=4 == MTP draft-3 verify window.
H, K, V = 12, 128, 128  # H == HV (KDA has no GQA); per-channel gate dim == K
T, W = 4, 16  # W = max history WINDOW (kernel W_RING)
RING = 32  # physical ring depth (kernel RING_SLOTS); window wraps mod RING
FLUSH_MIN = 13  # == W - T + 1
SCALE = 1.0 / math.sqrt(K)
LOWER_BOUND = -5.0  # Kimi K3 gate lower bound

_FLUSH_PATH = str(
    Path(__file__).resolve().parents[2]
    / "flashinfer/kda_kernels/kda_decode_bf16_wy_ucache_flush.py"
)
_MODULE_CACHE: dict = {}


def _skip_if_not_sm90_or_later():
    from flashinfer.utils import get_compute_capability

    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"KDA ucache requires SM90+, got SM{cc[0]}{cc[1]}")


def _load_flush():
    if "flush" in _MODULE_CACHE:
        return _MODULE_CACHE["flush"]
    spec = importlib.util.spec_from_file_location("kda_uc_flush", _FLUSH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULE_CACHE["flush"] = mod
    return mod


# ---------------------------------------------------------------------------
# The fp32 oracle. Slow and simple on purpose - this IS the spec.
# One request at a time. State S is [H, V, K]; decay scales the K axis.
# ---------------------------------------------------------------------------
def _gate_log(g_raw, A_log, dt_bias):
    """Kimi K3 lower-bound gate, fp32: lb * sigmoid(exp(A_log) * (g + bias)).
    g_raw [.., H, K]; A_log [H] fp32; dt_bias [H*K] fp32. Returns [.., H, K]."""
    x = g_raw.float() + dt_bias.float().view(H, K)
    return LOWER_BOUND * torch.sigmoid(torch.exp(A_log.float())[:, None] * x)


def _ref_fp32(q, k, v, g, b, A_log, dt_bias, S0, kc, uc, gc, P, t_len=T):
    f = torch.float32
    S = S0.to(f).clone()

    # 1) replay the P live ring entries into the state (per-channel decay):
    #    S <- exp(G_P) (.) S0 + sum_j u_j (x) (exp(G_P - G_j) (.) k_j)
    if P > 0:
        GP = gc[:, P - 1].to(f)  # [H, K]
        w = torch.exp(GP[:, None, :] - gc[:, :P].to(f))  # [H, P, K]
        S = torch.exp(GP)[:, None, :] * S + torch.einsum(
            "hpv,hpk->hvk", uc[:, :P].to(f), w * kc[:, :P].to(f)
        )
    S_after_history = S.clone()  # == the state a fold commits to the pool

    # 2) run the t_len new draft tokens through the exact KDA delta-rule
    #    recurrence: decay (per channel) -> predict -> update -> read out
    khat = F.normalize(k.to(f), dim=-1)
    qhat = F.normalize(q.to(f), dim=-1) * SCALE
    y = torch.zeros(t_len, H, V, dtype=f, device=q.device)
    for t in range(t_len):
        glog = _gate_log(g[t], A_log, dt_bias)  # [H, K]
        beta = torch.sigmoid(b[t].to(f))  # [H]
        S = S * torch.exp(glog)[:, None, :]  # per-channel decay
        pred = torch.einsum("hvk,hk->hv", S, khat[t])  # S k
        u_t = (v[t].to(f) - pred) * beta[:, None]  # delta rule (u, not v)
        S = S + u_t[:, :, None] * khat[t][:, None, :]  # + u (x) k
        y[t] = torch.einsum("hvk,hk->hv", S, qhat[t])  # out = S q
    return y, S_after_history


# ---------------------------------------------------------------------------
# Case builder: consistent inputs + rings for B requests.
# ---------------------------------------------------------------------------
def _make_case(B, hist_lens, seed, bases=None, t=T, strong=False, pool_size=None):
    """strong=True uses realistic Kimi-K3 decay magnitudes: saturating gates
    (glog near lower_bound = -5) and ring windows whose cumulative log-decay
    reaches ~-40 per channel at hist 16 — stressing the exp() paths (bdec
    down to e^{-40}, w spanning 17 decades) instead of the mild default."""
    io, st, rg = torch.bfloat16, torch.bfloat16, torch.bfloat16
    gen = torch.Generator(device=DEV).manual_seed(seed)
    pool_size = pool_size or B

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=gen, device=DEV) * sc).to(io)

    q, k = rn(B, t, H, K), rn(B, t, H, K)
    v, b = rn(B, t, H, V, sc=0.5), rn(B, t, H)
    # raw per-channel gate pre-activation (sc=3 saturates the sigmoid)
    g = rn(B, t, H, K, sc=3.0 if strong else 0.5)
    A_log = (torch.rand(H, generator=gen, device=DEV) * 0.6 - 0.3).float()
    dt_bias = (torch.randn(H * K, generator=gen, device=DEV) * 0.5).float()
    pool = (torch.randn(pool_size, H, V, K, generator=gen, device=DEV) * 0.5).to(st)
    kc = torch.zeros(pool_size, H, RING, K, dtype=rg, device=DEV)
    uc = torch.zeros(pool_size, H, RING, V, dtype=rg, device=DEV)
    gc = torch.zeros(pool_size, H, RING, K, dtype=torch.float32, device=DEV)
    hl = torch.tensor(hist_lens, dtype=torch.int32, device=DEV)
    bases = bases or [0] * B
    cb = torch.tensor(bases, dtype=torch.int32, device=DEV)
    for r in range(B):
        P = int(hl[r])
        if P == 0:
            continue
        # logical history rows j land at PHYSICAL ring rows (base + j) % RING
        rows = torch.tensor(
            [(bases[r] + j) % RING for j in range(P)], dtype=torch.long, device=DEV
        )
        kh = torch.randn(H, P, K, generator=gen, device=DEV)
        kc[r, :, rows] = F.normalize(kh, dim=-1).to(rg)
        uc[r, :, rows] = (torch.randn(H, P, V, generator=gen, device=DEV) * 0.3).to(rg)
        # per-channel cumulative log-decay: monotone non-increasing per
        # channel, each step in (LOWER_BOUND, 0) like the real gate
        if strong:
            la = -(torch.rand(H, P, K, generator=gen, device=DEV) * 2.4 + 0.1)
        else:
            la = -(torch.rand(H, P, K, generator=gen, device=DEV) * 0.3 + 0.003)
        gc[r, :, rows] = torch.cumsum(la, dim=1)
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx


def _logical_rings(kc_r, uc_r, gc_r, base, w=W):
    """Gather one request's PHYSICAL ring rows into logical window order
    (row j = physical (base+j) % RING) so the fp32 oracle stays unchanged."""
    rows = torch.tensor(
        [(base + j) % RING for j in range(w)], dtype=torch.long, device=kc_r.device
    )
    return (
        kc_r.index_select(1, rows),
        uc_r.index_select(1, rows),
        gc_r.index_select(1, rows),
    )


def _run(mod, case, flush_min=FLUSH_MIN, restart=False, w_ring=16, tma_late=None):
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    return mod.kda_delta_rule_mtp_ucache_flush(
        A_log,
        g,
        dt_bias,
        lower_bound=LOWER_BOUND,
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
        w_ring=w_ring,
        tma_late=tma_late,
    )


def _histories(t):
    """Per-T history cases: fm = W - t + 1 (13 at T=4, 9 at T=8)."""
    fm = W - t + 1
    return {
        "empty_P0": [0, 0, 0, 0],
        "replay_max": [fm - 1] * 4,
        "fold_mixed": [fm, fm - 1, fm, fm - 1],  # rows 0 and 2 fold
    }


BASES = {
    "base0": [0, 0, 0, 0],
    # wrapped windows: base+P crosses RING for some rows (the ring path)
    "wrap": [28, 5, 30, 17],
}
Y_TOL = 8e-3
STATE_TOL = 2e-2  # committed state accumulates P outer products first


def _check_case(
    mod,
    case,
    bases,
    t,
    flush_min,
    y_tol=Y_TOL,
    s_tol=STATE_TOL,
    w_ring=16,
    tma_late=None,
):
    """Run the kernel on (a clone-armored copy of) `case` and compare every
    row's output — and, for folding rows, the committed state — to the fp32
    oracle. Returns (max_y_err, max_s_err)."""
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    pool_before = pool.clone()
    run_case = (
        q,
        k,
        v,
        g,
        b,
        A_log,
        dt_bias,
        pool,
        kc.clone(),
        uc.clone(),
        gc.clone(),
        hl.clone(),
        cb.clone(),
        idx,
    )
    y = _run(mod, run_case, flush_min=flush_min, w_ring=w_ring, tma_late=tma_late)
    B = q.shape[0]
    ymax = smax = 0.0
    for r in range(B):
        slot = int(idx[r])
        kc_l, uc_l, gc_l = _logical_rings(
            kc[slot], uc[slot], gc[slot], int(cb[r]), w=w_ring
        )
        y_ref, S_ref = _ref_fp32(
            q[r],
            k[r],
            v[r],
            g[r],
            b[r],
            A_log,
            dt_bias,
            pool_before[slot],
            kc_l,
            uc_l,
            gc_l,
            int(hl[r]),
            t_len=t,
        )
        err = (y[r].float() - y_ref).abs().max().item()
        ymax = max(ymax, err)
        assert err < y_tol, f"row {r} (P={int(hl[r])}): |y - fp32 ref| = {err:.2e}"
        if int(hl[r]) >= flush_min:
            serr = (pool[slot].float() - S_ref).abs().max().item()
            smax = max(smax, serr)
            assert serr < s_tol, f"row {r}: |committed - fp32 ref| = {serr:.2e}"
    return ymax, smax


@pytest.mark.parametrize("t", [4, 8])
@pytest.mark.parametrize("history", ["empty_P0", "replay_max", "fold_mixed"])
@pytest.mark.parametrize("basecase", list(BASES))
def test_output_matches_fp32_reference(t, history, basecase):
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    bases = BASES[basecase]
    case = _make_case(4, _histories(t)[history], seed=1234, bases=bases, t=t)
    _check_case(mod, case, bases, t, flush_min=W - t + 1)


@pytest.mark.parametrize("t", [4, 8])
@pytest.mark.parametrize("basecase", list(BASES))
def test_folded_state_matches_fp32_reference(t, basecase):
    """On a fold (hist_len >= flush_min) the kernel writes the ring-folded
    checkpoint back to the pool. Compare that committed state to the oracle."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    fm = W - t + 1
    bases = BASES[basecase]
    case = _make_case(4, [fm] * 4, seed=99, bases=bases, t=t)
    _check_case(mod, case, bases, t, flush_min=fm)


@pytest.mark.parametrize("t", [4, 8])
def test_strong_decay_matches_fp32_reference(t):
    """Realistic Kimi-K3 decay magnitudes: saturating gates (glog ~ -5/token)
    and ring windows with cumulative log-decay down to ~-40 per channel at
    the max window (hist 16). Exercises bdec = e^{G_P} ~ e^{-40}, replay
    weights w spanning ~17 decades, and ktil = k*e^{-cum} up to ~e^{+40} —
    the exp()-range corners the mild default case never reaches."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    fm = W - t + 1
    hist = [16, fm, fm - 1, 0]
    case = _make_case(4, hist, seed=31337, bases=[30, 0, 27, 5], t=t, strong=True)
    _check_case(mod, case, [30, 0, 27, 5], t, flush_min=fm)


def test_flush_never_overwrites_live_window():
    """THE ring property: a flush must not modify any physical ring row
    inside the live window [base, base+P) — sibling CTAs read those rows as
    their fold source. Appends must land at (base+P+s) & RING_MASK only."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    B = 4
    bases = [28, 0, 30, 12]  # rows 0/2 wrap through the ring end
    hist = [13, 13, 14, 16]  # ALL rows flush (P >= 13)
    case = _make_case(B, hist, seed=4242, bases=bases)
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    kc_in, uc_in, gc_in = kc.clone(), uc.clone(), gc.clone()

    run_case = (
        q,
        k,
        v,
        g,
        b,
        A_log,
        dt_bias,
        pool,
        kc,
        uc,
        gc,
        hl.clone(),
        cb.clone(),
        idx,
    )
    _run(mod, run_case)

    for r in range(B):
        P = hist[r]
        window = [(bases[r] + j) % RING for j in range(P)]
        appends = [(bases[r] + P + s) % RING for s in range(T)]
        for rows, name, before, after in (
            (window, "k", kc_in, kc),
            (window, "u", uc_in, uc),
            (window, "g", gc_in, gc),
        ):
            wnd = torch.tensor(rows, dtype=torch.long, device=DEV)
            assert torch.equal(
                before[r].index_select(1, wnd), after[r].index_select(1, wnd)
            ), f"row {r}: flush modified live {name} window rows {rows}"
        # and the appends actually landed (k rows are L2-normed, non-zero)
        ap = torch.tensor(appends, dtype=torch.long, device=DEV)
        assert kc[r].index_select(1, ap).float().abs().sum() > 0, (
            f"row {r}: no k append at {appends}"
        )


def test_chained_steps_match_sequential_oracle():
    """End-to-end validation of the APPENDED ring rows (u, k, g): run the
    kernel twice back-to-back — step 1 from a bare checkpoint (P=0), commit
    all T tokens (hist_len += T), step 2 replays step 1's appends from the
    ring. Step 2's outputs must match a single 2T-token sequential fp32
    recurrence. Any error in an appended u/k/g row shows up here.

    This closes a hole in the GDN W16 suite, which never numerically checks
    appended values (only that they landed / didn't clobber the window)."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    B = 4
    bases = [0, 5, 28, 30]  # include wraps: appends of step 1 cross the ring
    gen = torch.Generator(device=DEV).manual_seed(777)
    io = torch.bfloat16

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=gen, device=DEV) * sc).to(io)

    # 2T tokens of fresh input; step 1 sees [:T], step 2 sees [T:]
    q2, k2 = rn(B, 2 * T, H, K), rn(B, 2 * T, H, K)
    v2, b2 = rn(B, 2 * T, H, V, sc=0.5), rn(B, 2 * T, H)
    g2 = rn(B, 2 * T, H, K, sc=0.5)
    A_log = (torch.rand(H, generator=gen, device=DEV) * 0.6 - 0.3).float()
    dt_bias = (torch.randn(H * K, generator=gen, device=DEV) * 0.5).float()
    pool = (torch.randn(B, H, V, K, generator=gen, device=DEV) * 0.5).to(io)
    pool_before = pool.clone()
    kc = torch.zeros(B, H, RING, K, dtype=io, device=DEV)
    uc = torch.zeros(B, H, RING, V, dtype=io, device=DEV)
    gc = torch.zeros(B, H, RING, K, dtype=torch.float32, device=DEV)
    cb = torch.tensor(bases, dtype=torch.int32, device=DEV)
    idx = torch.arange(B, dtype=torch.int32, device=DEV)

    for step in range(2):
        sl = slice(step * T, (step + 1) * T)
        hl = torch.full((B,), step * T, dtype=torch.int32, device=DEV)
        y = mod.kda_delta_rule_mtp_ucache_flush(
            A_log,
            g2[:, sl].contiguous(),
            dt_bias,
            lower_bound=LOWER_BOUND,
            q=q2[:, sl].contiguous(),
            k=k2[:, sl].contiguous(),
            v=v2[:, sl].contiguous(),
            b=b2[:, sl].contiguous(),
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb.clone(),
            scale=SCALE,
            flush_min=FLUSH_MIN,
            restart_hist_on_flush=False,  # caller-owned commit (serving mode)
        )

    # sequential fp32 truth over all 2T tokens; compare step-2 outputs
    f = torch.float32
    for r in range(B):
        S = pool_before[r].to(f).clone()
        khat = F.normalize(k2[r].to(f), dim=-1)
        qhat = F.normalize(q2[r].to(f), dim=-1) * SCALE
        for t in range(2 * T):
            glog = _gate_log(g2[r, t], A_log, dt_bias)
            beta = torch.sigmoid(b2[r, t].to(f))
            S = S * torch.exp(glog)[:, None, :]
            pred = torch.einsum("hvk,hk->hv", S, khat[t])
            u_t = (v2[r, t].to(f) - pred) * beta[:, None]
            S = S + u_t[:, :, None] * khat[t][:, None, :]
            if t >= T:
                y_ref = torch.einsum("hvk,hk->hv", S, qhat[t])
                err = (y[r, t - T].float() - y_ref).abs().max().item()
                # step 2 consumes bf16 ring rows -> a bit looser than Y_TOL
                assert err < 2e-2, (
                    f"row {r} token {t}: chained |y - fp32 ref| = {err:.2e}"
                )


@pytest.mark.parametrize("t", [4, 8])
def test_ring_append_values_match_reference(t):
    """Validate each APPENDED ring quantity in isolation (sharper than the
    chained end-to-end test): after one launch, the T new rows must hold
      k_cache: the L2-normalized key            (bf16 quantum tolerance)
      u_cache: u_s = beta_s * (v_s - S_{s-1}' k̂_s) from the recurrence
      g_cache: (G_P for verify rows | 0 for flush rows) + per-channel
               cumsum of the gate log — in fp32, so tight tolerance.
    Covers verify rows, flush rows (LOCAL-decay g restart), and wraps."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    fm = W - t + 1
    hist = [0, 7, fm, 16]  # bare, mid-window, flush-restart, max-window flush
    bases = [0, 29, 30, 28]
    case = _make_case(4, hist, seed=91, bases=bases, t=t)
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    kc_in, uc_in, gc_in = kc.clone(), uc.clone(), gc.clone()

    _run(
        mod,
        (q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl.clone(), cb.clone(), idx),
        flush_min=fm,
    )

    f = torch.float32
    for r in range(4):
        P = hist[r]
        kc_l, uc_l, gc_l = _logical_rings(kc_in[r], uc_in[r], gc_in[r], bases[r])
        # reference recurrence, capturing per-token u and the gate cumsum.
        # Flush rows overwrote pool[r], so rebuild the pre-run pool from the
        # deterministic case builder (same seed) and replay the history.
        pool_pre = _make_case(4, hist, seed=91, bases=bases, t=t)[7][r]
        _, S = _ref_fp32(
            q[r],
            k[r],
            v[r],
            g[r],
            b[r],
            A_log,
            dt_bias,
            pool_pre,
            kc_l,
            uc_l,
            gc_l,
            P,
            t_len=t,
        )
        khat = F.normalize(k[r].to(f), dim=-1)
        GP = gc_l[:, P - 1].to(f) if P > 0 else torch.zeros(H, K, device=DEV)
        g_base = torch.zeros_like(GP) if fm <= P else GP
        cum = torch.zeros(H, K, dtype=f, device=DEV)
        for s in range(t):
            row = (bases[r] + P + s) % RING
            glog = _gate_log(g[r, s], A_log, dt_bias)
            beta = torch.sigmoid(b[r, s].to(f))
            cum = cum + glog
            S = S * torch.exp(glog)[:, None, :]
            u_ref = (v[r, s].to(f) - torch.einsum("hvk,hk->hv", S, khat[s])) * beta[
                :, None
            ]
            S = S + u_ref[:, :, None] * khat[s][:, None, :]
            k_err = (kc[r, :, row].float() - khat[s]).abs().max().item()
            assert k_err < 5e-3, f"row {r} s={s}: k append |err|={k_err:.2e}"
            u_err = (uc[r, :, row].float() - u_ref).abs().max().item()
            assert u_err < 2e-2, f"row {r} s={s}: u append |err|={u_err:.2e}"
            g_err = (gc[r, :, row] - (g_base + cum)).abs().max().item()
            assert g_err < 1e-4, f"row {r} s={s}: g append |err|={g_err:.2e}"


def test_capacity_validation():
    """flush_min and hist_len beyond the W16/RING32 capacity algebra must
    raise: flush_min <= min(W - T + 1, RING - 2T + 1) and
    hist_len <= min(W, RING - T)."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    B = 4
    case = _make_case(B, [12, 12, 12, 12], seed=7, bases=[0, 5, 28, 30])
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case

    with pytest.raises(AssertionError):
        run_case = (
            q,
            k,
            v,
            g,
            b,
            A_log,
            dt_bias,
            pool,
            kc,
            uc,
            gc,
            hl.clone(),
            cb.clone(),
            idx,
        )
        _run(mod, run_case, flush_min=FLUSH_MIN + 1, restart=True)

    hl_bad = hl.clone()
    hl_bad[1] = 17  # > W - allowed by RING but not by the 16-row tile
    with pytest.raises(AssertionError, match="hist_len out of legal range"):
        run_case = (
            q,
            k,
            v,
            g,
            b,
            A_log,
            dt_bias,
            pool,
            kc,
            uc,
            gc,
            hl_bad,
            cb.clone(),
            idx,
        )
        _run(mod, run_case, restart=True)


def test_padded_rows_and_permuted_pool():
    """CUDA-graph batch padding + non-identity pool slots: rows with
    initial_state_indices < 0 must retire their CTAs before ANY work (zero
    output rows via the preallocated buffer, no ring/pool writes anywhere),
    while real rows address a LARGER pool through permuted slots and stay
    oracle-exact. This pins the slot-indexed addressing (cache_idx * pool
    strides) that the arange-idx tests never exercise."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    B, POOL = 4, 8
    hist = [13, 5, 13, 9]
    bases = [28, 0, 30, 12]
    case = _make_case(B, hist, seed=555, bases=bases, pool_size=POOL)
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    # rows 1 and 3 are padding; rows 0 and 2 use permuted slots 5 and 2.
    idx = torch.tensor([5, -1, 2, -1], dtype=torch.int32, device=DEV)
    # _make_case seeded rings at slots [0, B); move row 0/2 windows to 5/2.
    for r, slot in ((0, 5), (2, 2)):
        for ring in (kc, uc, gc):
            ring[slot] = ring[r]
            if slot != r:
                ring[r].zero_()
    pool_before = pool.clone()
    kc_in, uc_in, gc_in = kc.clone(), uc.clone(), gc.clone()
    out = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=DEV)

    mod.kda_delta_rule_mtp_ucache_flush(
        A_log,
        g,
        dt_bias,
        lower_bound=LOWER_BOUND,
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
        flush_min=FLUSH_MIN,
        restart_hist_on_flush=False,
        output=out,
    )

    # real rows match the oracle (row 0 folds into pool slot 5)
    for r, slot in ((0, 5), (2, 2)):
        kc_l, uc_l, gc_l = _logical_rings(
            kc_in[slot], uc_in[slot], gc_in[slot], bases[r]
        )
        y_ref, S_ref = _ref_fp32(
            q[r],
            k[r],
            v[r],
            g[r],
            b[r],
            A_log,
            dt_bias,
            pool_before[slot],
            kc_l,
            uc_l,
            gc_l,
            hist[r],
        )
        assert (out[r].float() - y_ref).abs().max().item() < Y_TOL
        if hist[r] >= FLUSH_MIN:
            assert (pool[slot].float() - S_ref).abs().max().item() < STATE_TOL
    # padded rows: zero output, and NOTHING else in the pool/rings moved —
    # every slot not owned by a real row is bit-identical.
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.equal(out[3], torch.zeros_like(out[3]))
    untouched = [s for s in range(POOL) if s not in (5, 2)]
    w = torch.tensor(untouched, dtype=torch.long, device=DEV)
    assert torch.equal(pool.index_select(0, w), pool_before.index_select(0, w))
    for name, before, after in (("k", kc_in, kc), ("u", uc_in, uc), ("g", gc_in, gc)):
        assert torch.equal(before.index_select(0, w), after.index_select(0, w)), (
            f"padded rows wrote the {name} ring outside their slots"
        )


def test_cuda_graph_replay_matches_eager():
    """The serving path replays this kernel from a CUDA graph. Capture with
    static tensors, restore the pre-run ring/pool bytes, replay, and demand
    BITWISE equality with an eager run on identical inputs (the kernel is
    deterministic: fixed MMA order, no atomics)."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    B = 4
    hist = [13, 0, 12, 16]
    bases = [30, 0, 5, 28]
    case = _make_case(B, hist, seed=808, bases=bases)
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    snap = [t.clone() for t in (pool, kc, uc, gc)]
    out = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=DEV)

    def call():
        mod.kda_delta_rule_mtp_ucache_flush(
            A_log,
            g,
            dt_bias,
            lower_bound=LOWER_BOUND,
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
            flush_min=FLUSH_MIN,
            restart_hist_on_flush=False,
            output=out,
        )

    def restore():
        for dst, src in zip((pool, kc, uc, gc), snap, strict=True):
            dst.copy_(src)

    # eager reference run on pristine state
    restore()
    call()
    torch.cuda.synchronize()
    eager = (out.clone(), pool.clone(), kc.clone(), uc.clone(), gc.clone())

    # warm up + capture, then replay on restored pristine state
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    restore()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()

    for name, a, bfr in zip(
        ("output", "pool", "k_cache", "u_cache", "g_cache"),
        (out, pool, kc, uc, gc),
        eager,
        strict=True,
    ):
        assert torch.equal(a, bfr), f"graph replay diverged from eager on {name}"


@pytest.mark.parametrize("t", [4, 8])
def test_large_batch_randomized_oracle_and_determinism(t):
    """B=64 with fully random hist_len [0,16], random bases [0,32), and a
    permuted pool: every row oracle-checked (output + fold), then the whole
    launch repeated on bit-identical inputs and required to reproduce output,
    rings, and pool BITWISE (a probabilistic sync/race tripwire — a missing
    barrier shows up as run-to-run byte drift)."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    B = 64
    fm = W - t + 1
    gen = torch.Generator().manual_seed(2024 + t)
    hist = torch.randint(0, 17, (B,), generator=gen).tolist()
    bases = torch.randint(0, RING, (B,), generator=gen).tolist()
    case = _make_case(B, hist, seed=6000 + t, bases=bases, t=t, pool_size=B)
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    perm = torch.randperm(B, generator=gen).to(torch.int32).to(DEV)
    # move each row's seeded ring window/pool to its permuted slot
    pool = pool[perm.long()].contiguous()
    kc = kc[perm.long()].contiguous()
    uc = uc[perm.long()].contiguous()
    gc = gc[perm.long()].contiguous()
    inv = torch.empty_like(perm)
    inv[perm.long()] = torch.arange(B, dtype=torch.int32, device=DEV)
    idx = inv  # row r's data now lives at slot inv[r]
    case = (q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx)

    pool0, kc0, uc0, gc0 = pool.clone(), kc.clone(), uc.clone(), gc.clone()
    # oracle check every row (output + committed state); mutates only the
    # real pool (_check_case runs the kernel against ring CLONES).
    _check_case(mod, case, bases, t, flush_min=fm)

    # determinism: two direct runs on the REAL tensors from bit-identical
    # restored state must reproduce output, rings, and pool bitwise.
    def direct_run():
        pool.copy_(pool0)
        kc.copy_(kc0)
        uc.copy_(uc0)
        gc.copy_(gc0)
        y = _run(
            mod,
            (
                q,
                k,
                v,
                g,
                b,
                A_log,
                dt_bias,
                pool,
                kc,
                uc,
                gc,
                hl.clone(),
                cb.clone(),
                idx,
            ),
            flush_min=fm,
        )
        torch.cuda.synchronize()
        return (y.clone(), pool.clone(), kc.clone(), uc.clone(), gc.clone())

    r1 = direct_run()
    r2 = direct_run()
    for name, a, bfr in zip(
        ("output", "pool", "k_cache", "u_cache", "g_cache"), r1, r2, strict=True
    ):
        assert torch.equal(a, bfr), f"non-deterministic {name} across identical runs"


# ---------------------------------------------------------------------------
# W32 deep-window mode (w_ring=32 over the same 32-slot ring): flush_min cap
# = RING - 2T + 1 (25 @T=4, 17 @T=8), hist cap = RING - T (28 / 24).
# ---------------------------------------------------------------------------
def _w32_fm(t):
    return min(32 - t + 1, RING - 2 * t + 1)  # 25 @T=4, 17 @T=8


@pytest.mark.parametrize("t", [4, 8])
@pytest.mark.parametrize("basecase", list(BASES))
def test_w32_output_and_fold_match_fp32_reference(t, basecase):
    """Deep windows the W16 kernel can never see: replay at P in
    (16, hist_cap] (the second j-block), folds at P >= 25/17, exact-peak
    fill, all with wrapped bases."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    fm = _w32_fm(t)
    cap = RING - t  # 28 @T=4, 24 @T=8
    bases = BASES[basecase]
    for hist in ([0, 7, 16, 20], [fm - 1, fm, fm + 1, cap]):
        case = _make_case(4, hist, seed=2468 + t, bases=bases, t=t)
        _check_case(mod, case, bases, t, flush_min=fm, w_ring=32)


@pytest.mark.parametrize("t", [4, 8])
def test_w32_tma_late_forced(t):
    """tma_late is a batch-adaptive scheduling knob (auto-selected from
    B*HV vs. SM count); w_ring=32 changes which SMEM tenant that knob
    aliases with. Every other w32 test uses B=4, which never crosses the
    auto threshold, so tma_late=True has never actually run combined with
    w_ring=32. Force it explicitly here — replay AND fold, both j-blocks —
    so that combination is oracle-checked at least once."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    fm = _w32_fm(t)
    cap = RING - t
    bases = BASES["wrap"]
    for hist in ([0, 7, 16, 20], [fm - 1, fm, fm + 1, cap]):
        case = _make_case(4, hist, seed=2468 + t, bases=bases, t=t)
        _check_case(mod, case, bases, t, flush_min=fm, w_ring=32, tma_late=True)


@pytest.mark.parametrize("t", [4, 8])
def test_w32_strong_decay_deep_window(t):
    """Realistic Kimi-K3 decay at the DEEP window: cumulative log-decay to
    ~-70 per channel at hist 28 — the widest exp() range this kernel can
    produce (w spans ~30 decades; bdec ~ e^{-70})."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    fm = _w32_fm(t)
    cap = RING - t
    hist = [cap, fm, 17, 0]
    bases = [30, 0, 27, 5]
    case = _make_case(4, hist, seed=1357 + t, bases=bases, t=t, strong=True)
    _check_case(mod, case, bases, t, flush_min=fm, w_ring=32)


def test_w32_equivalence_with_w16_at_shallow_windows():
    """For P <= 12 (both windows see identical inputs and the second j-block
    is all-zero) the W32 build must reproduce the W16 build BITWISE:
    outputs, ring appends, and (non-fold) pool."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    hist = [0, 5, 9, 12]
    bases = [28, 0, 30, 17]
    outs = {}
    for wr in (16, 32):
        case = _make_case(4, hist, seed=97, bases=bases)
        q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
        y = _run(
            mod,
            (q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx),
            flush_min=13 if wr == 16 else 25,
            w_ring=wr,
        )
        torch.cuda.synchronize()
        outs[wr] = (y.clone(), pool.clone(), kc.clone(), uc.clone(), gc.clone())
    for name, a, bfr in zip(
        ("output", "pool", "k_cache", "u_cache", "g_cache"),
        outs[16],
        outs[32],
        strict=True,
    ):
        assert torch.equal(a, bfr), f"w16/w32 diverge on {name} at shallow P"


def test_w32_capacity_validation():
    """flush_min caps at min(w-T+1, RING-2T+1) = 25 @T=4; hist_len caps at
    RING-T = 28. One past each must raise."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush()
    case = _make_case(4, [12, 12, 12, 12], seed=7, bases=[0, 5, 28, 30])
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    with pytest.raises(AssertionError):
        _run(
            mod,
            (
                q,
                k,
                v,
                g,
                b,
                A_log,
                dt_bias,
                pool,
                kc,
                uc,
                gc,
                hl.clone(),
                cb.clone(),
                idx,
            ),
            flush_min=26,
            restart=True,
            w_ring=32,
        )
    hl_bad = hl.clone()
    hl_bad[1] = 29  # appends would wrap into the live window
    with pytest.raises(AssertionError, match="hist_len out of legal range"):
        _run(
            mod,
            (q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl_bad, cb.clone(), idx),
            flush_min=25,
            restart=True,
            w_ring=32,
        )
    # shallow flush thresholds are w16 territory (round-2 zeroing is
    # skipped CTA-uniformly at P <= 16, so a fold there would read stale
    # khist rows).
    with pytest.raises(AssertionError, match="flush_min > 16"):
        _run(
            mod,
            (
                q,
                k,
                v,
                g,
                b,
                A_log,
                dt_bias,
                pool,
                kc,
                uc,
                gc,
                hl.clone(),
                cb.clone(),
                idx,
            ),
            flush_min=13,
            restart=True,
            w_ring=32,
        )
