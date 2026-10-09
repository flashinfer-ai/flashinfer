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
Correctness tests for the UMMA (tcgen05 / TMEM) MTP ReplaySSM kernel
(flashinfer/gdn_kernels/gdn_replay_mtp_umma.py), the SM100 backend of
``gated_delta_rule_mtp_ucache_flush`` for the fp16-state arm.

The oracle is FlashInfer's ``_ref_fp32`` (tests/gdn/test_decode_ucache.py), generalised to
any T: an fp32, token-at-a-time PyTorch loop over the ring replay and the delta rule.  Case
builder follows ``_make_case`` (32-slot physical ring, logical row j at (base + j) % 32) and
additionally NaN-poisons every ring slot outside the live window, so a kernel that lets a
stale slot reach an MMA (0 * NaN = NaN) fails loudly.

Covered: T in 1..8, wrapped windows, mixed verify/flush batches, padded rows, row order,
predictive flush_min, ring appends (k̂, u, g) and the committed state, a multi-step horizon
with cursor commits, and parity with ``gated_delta_rule_mtp_ucache_flush`` (fp16_state arm).

Run:  pytest tests/gdn/test_mtp_umma.py -q
"""

from __future__ import annotations

import functools
import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

DEV = "cuda"
H, HV, K, V = 16, 64, 128, 128
W, RING = 16, 32
SCALE = 1.0 / math.sqrt(K)
Y_TOL = 8e-3  # FlashInfer's tolerance for outputs
STATE_TOL = 2e-2  # FlashInfer's tolerance for the committed state
U_TOL = 1.5e-2  # appended corrections (bf16 storage of u)

_ROOT = Path(__file__).resolve().parents[2]
_CVT_RS_CC = ((10, 0), (10, 3), (10, 7))  # stochastic rounding needs cvt.rs


def _skip_if_not_sm100():
    from flashinfer.utils import get_compute_capability

    if not torch.cuda.is_available():
        pytest.skip("UMMA kernel requires a CUDA GPU")
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("UMMA kernel requires SM100 (Blackwell)")


def _skip_if_no_cvt_rs():
    from flashinfer.utils import get_compute_capability

    _skip_if_not_sm100()
    if get_compute_capability(torch.device("cuda")) not in _CVT_RS_CC:
        pytest.skip("stochastic rounding needs cvt.rs (sm_100a / sm_103a / sm_107a)")


def _umma():
    from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
        gated_delta_rule_mtp_ucache_flush_umma,
    )

    return gated_delta_rule_mtp_ucache_flush_umma


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


def _run(fn, case, flush_min, row_order=None, restart=False):
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
    return fn(A_log, a, dt_bias, **kw)


def _check_case(
    T, hist, bases, flush_min, seed, row_order=None, check_rings=True, hpc=None
):
    fn = _umma()
    if hpc is not None:
        fn = functools.partial(fn, heads_per_cta=hpc)
    B = len(hist)
    case = _make_case(B, T, hist, seed, bases)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    pool0, kc0, uc0, gc0 = pool.clone(), kc.clone(), uc.clone(), gc.clone()
    ro = None
    if row_order == "flush_first":
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma import flush_first_row_order

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
        flushed = flush_min <= P
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
    fm = W - T + 1
    hist = [0, fm - 1, fm, W - T + 1 if fm == W - T + 1 else fm, 1, fm - 1, fm, 3]
    hist = [min(x, W) for x in hist]
    b = [0] * 8 if bases == "base0" else [28, 5, 30, 17, 31, 20, 25, 9]
    _check_case(T, hist, b, fm, seed=100 + T)


@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("order", [None, "flush_first", "reverse"])
def test_row_order_and_predictive_flush(T, order):
    _skip_if_not_sm100()
    fm = 9 - (T == 8) * 4  # predictive flush_min (< lazy default)
    hist = [fm, 0, fm + 2, fm - 1, 2, W - T + 1, fm, 1]
    _check_case(T, hist, [3, 30, 27, 0, 31, 16, 29, 7], fm, seed=7 + T, row_order=order)


@pytest.mark.parametrize("T", [1, 4, 6, 8])
@pytest.mark.parametrize("hpc", [1, 2, 4])
def test_heads_per_cta(T, hpc):
    """Every heads-per-CTA split (4: one CTA per key head; 2 / 1: a key head over 2 / 4 CTAs)
    against the fp32 reference: mixed verify / flush rows, wrapped windows, rings."""
    _skip_if_not_sm100()
    fm = W - T + 1
    hist = [fm, 0, fm - 1, fm, 3, fm - 1, 1, fm]
    _check_case(T, hist, [28, 5, 30, 0, 31, 17, 9, 26], fm, seed=50 + T + hpc, hpc=hpc)


def test_full_window_flush_P16_and_T1():
    _skip_if_not_sm100()
    _check_case(1, [16, 15, 16, 0], [20, 31, 0, 5], 16, seed=3)


def test_padded_rows_untouched():
    _skip_if_not_sm100()
    fn = _umma()
    T, B = 4, 4
    case = list(_make_case(B, T, [12, 13, 13, 5], seed=11, bases=[0, 28, 3, 9]))
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
        for before, after in zip(snap, (pool, kc, uc, gc), strict=True):
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
    fm = W - T + 1
    g = torch.Generator(device=DEV).manual_seed(5)
    case = _make_case(B, T, [0] * B, seed=5, bases=[0, 9, 30, 17])
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


@pytest.mark.parametrize("T", [4, 8])
def test_parity_with_flashinfer_fp16_state(T):
    """Same inputs through the HMMA kernel behind gated_delta_rule_mtp_ucache_flush
    (fp16_state arm, backend pinned: on SM100 "auto" would run this kernel again)."""
    _skip_if_not_sm100()
    fi = functools.partial(
        _flashinfer_fp16_state().gated_delta_rule_mtp_ucache_flush, backend="hmma"
    )
    fm = W - T + 1
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


def _sr_case(T, seed):
    fm = W - T + 1
    hist = [fm, fm - 1, W, fm, 0, fm]
    bases = [28, 5, 30, 17, 9, 0]
    return _make_case(len(hist), T, hist, seed, bases), hist, fm


def _sr_call(case, fm, **kw):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() if isinstance(x, torch.Tensor) else x for x in case
    ]
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
        **kw,
    )
    torch.cuda.synchronize()
    return y, pool, (kc, uc, gc)


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("mode", ["philox10", "philox5", "lcg"])
@pytest.mark.parametrize("hpc", [1, 2, 4])
def test_stochastic_rounding_matches_reference_bits(T, mode, hpc):
    """SR only changes how the flushed fp16 state is rounded: outputs and rings are
    bit-identical to round-to-nearest, verify rows keep their state, every flushed element
    is RN or its neighbour, and each rounding direction agrees with the reference noise
    bits (rounding up where RN went down needs noise > 4096 of 8192; down where RN went up
    needs noise < 4096). Noise from a different seed must violate that, which proves the
    check can fail. hpc 1 and 4 cover both RMW group sizes (16 / 32 TMEM columns); hpc 2 at
    T = 8 the 256-thread register-split builds of Philox-5 and LCG."""
    _skip_if_no_cvt_rs()
    case, hist, fm = _sr_case(T, seed=70 + T)
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
    y_rn, s_rn, rings_rn = _sr_call(case, fm, heads_per_cta=hpc)
    y_sr, s_sr, rings_sr = _sr_call(
        case,
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
        n_bad += int(((noise <= 4096) & up).sum() + ((noise >= 4096) & down).sum())
        n_bad_ctl += int(((ctl <= 4096) & up).sum() + ((ctl >= 4096) & down).sum())
    total = sum(HV * V * K for hist_r in hist if hist_r >= fm)
    assert abs(n_diff / total - 0.25) < 0.05, n_diff / total  # E[min(f, 1 - f)] = 1/4
    assert n_bad == 0, (
        f"{n_bad} of {n_diff} rounding directions contradict the {mode} bits"
    )
    assert n_bad_ctl > 0.3 * n_diff, (n_bad_ctl, n_diff)


def test_stochastic_rounding_seed_and_graph():
    """Same seed -> bit-identical state; another seed -> different; the seed is read on
    device, so an in-place update reaches a captured CUDA graph; a seed alone (no mode)
    is RN."""
    _skip_if_no_cvt_rs()
    T = 4
    case, hist, fm = _sr_case(T, seed=91)
    seed_t = torch.tensor([7], dtype=torch.int64, device=DEV)
    ph = dict(stochastic_rounding="philox")
    _, a1, _ = _sr_call(case, fm, rand_seed=seed_t, **ph)
    _, a2, _ = _sr_call(case, fm, rand_seed=seed_t, **ph)
    _, b1, _ = _sr_call(
        case, fm, rand_seed=torch.tensor([8], dtype=torch.int64, device=DEV), **ph
    )
    _, rn, _ = _sr_call(case, fm)
    # a seed without a mode does not switch stochastic rounding on
    _, rn_seed, _ = _sr_call(case, fm, rand_seed=seed_t)
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b1) and not torch.equal(a1, rn)
    assert torch.equal(rn_seed, rn)

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
        stochastic_rounding="philox",
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


def test_lcg_unseeded_uses_clock():
    """stochastic_rounding="lcg" without a seed: %clock-seeded, so two runs differ, while each
    stays RN or its neighbour with the expected ~1/4 of elements rounded the other way;
    "none" and the default are round-to-nearest."""
    _skip_if_no_cvt_rs()
    case, hist, fm = _sr_case(4, seed=12)
    _, rn, _ = _sr_call(case, fm)
    _, rn2, _ = _sr_call(
        case,
        fm,
        stochastic_rounding="none",
        rand_seed=torch.tensor([3], dtype=torch.int64, device=DEV),
    )
    _, c1, _ = _sr_call(case, fm, stochastic_rounding="lcg")
    _, c2, _ = _sr_call(case, fm, stochastic_rounding="lcg")
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
    _skip_if_no_cvt_rs()
    case, _, fm = _sr_case(4, seed=5)
    for bad in (
        torch.tensor([1, 2], dtype=torch.int64, device=DEV),
        torch.tensor([1], dtype=torch.int32, device=DEV),
        torch.tensor([1], dtype=torch.int64),
    ):
        with pytest.raises(ValueError):
            _sr_call(case, fm, stochastic_rounding="philox", rand_seed=bad)
    with pytest.raises(ValueError):
        _sr_call(
            case,
            fm,
            stochastic_rounding="philox",
            rand_seed=torch.tensor([1], dtype=torch.int64, device=DEV),
            philox_rounds=0,
        )
    with pytest.raises(ValueError):
        _sr_call(case, fm, stochastic_rounding="philox")  # philox needs a seed
    with pytest.raises(ValueError):
        _sr_call(case, fm, stochastic_rounding="pcg")


if __name__ == "__main__":
    import sys

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    fm = W - T + 1
    print(_check_case(T, [0, fm - 1, fm, 5], [0, 28, 30, 3], fm, seed=1))
