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
Correctness tests for the GDN ucache verify+flush kernel
(flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py), plus a thin
oracle test for the companion verify-only kernel at the bottom
(gdn_decode_bf16_wy_ucache.py — legacy 16-deep flat ring, read-only state).

HOW THIS TESTS, IN PLAIN WORDS
  The kernel is a fast, fancy implementation (CuTe-DSL, tensor cores,
  bf16/fp16 storage) of math that is simple to state. So we compute the
  same answer twice:

    1. with the KERNEL (fast, low precision, complicated code), and
    2. with `_ref_fp32` below - a deliberately slow, obviously-correct
       PyTorch loop that follows the GDN recurrence one token at a time,
       entirely in fp32. ~40 lines, no tricks. Read it top to bottom.

  If |kernel - reference| stays within low-precision rounding noise, the
  kernel is computing the right function. The reference is ALWAYS fp32;
  the dtype arms below change only what the KERNEL stores:

    - "bf16"      : bf16 inputs, bf16 state pool   (default serving config)
    - "fp16_state": bf16 inputs, fp16 state pool   (GDN_UCACHE_STATE_DTYPE)
    - "fp16_io"   : fp16 inputs, fp16 state pool   (GDN_UCACHE_IO_DTYPE)

  Both arms are judged against the SAME fp32 oracle, on the SAME values
  (the oracle reads the already-rounded bf16/fp16 inputs and upcasts, so
  the only difference left is the kernel's internal arithmetic).

  Three behaviors are covered per arm:
    - verify from a bare checkpoint          (hist_len = 0)
    - verify with ring-history replay        (hist_len = 12)
    - the fold: ring folded into the state   (hist_len = 13 >= flush_min)
      -> here we also check the COMMITTED STATE the kernel wrote back,
         not just the output tokens.

  Finally, `test_fp16_state_commits_more_precisely_than_bf16` checks the
  reason the fp16-state mode exists: fp16 keeps 10 mantissa bits vs bf16's
  7, so the state written back at a fold should sit closer to the fp32
  truth. That is the mechanism behind the reduced long-context drift.

Run:
  source env.sh && pytest tests/gdn/test_decode_ucache.py -v
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.long_running

DEV = "cuda"
# Qwen3.5-122B GDN geometry at TP1; T=4 == MTP draft-3 verify window.
H, HV, K, V = 16, 64, 128, 128
T, W = 4, 16  # W = max history WINDOW (kernel W_RING)
RING = 32  # physical ring depth (kernel RING_SLOTS); window wraps mod RING
FLUSH_MIN = 13  # == W - T + 1
SCALE = 1.0 / math.sqrt(K)

_FLUSH_PATH = str(
    Path(__file__).resolve().parents[2]
    / "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py"
)
_MODULE_CACHE: dict = {}

ARMS = {
    #  name        io env  state env ring env io dtype     state dtype    ring dtype
    "bf16": (None, None, None, torch.bfloat16, torch.bfloat16, torch.bfloat16),
    "fp16_state": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "fp16_io": ("fp16", None, None, torch.float16, torch.float16, torch.float16),
    "ring_fp16": (None, None, "fp16", torch.bfloat16, torch.bfloat16, torch.float16),
    # bf16 inputs, BOTH state pool AND u/k rings fp16 (state+ring combined)
    "fp16_state_cache": (
        None,
        "fp16",
        "fp16",
        torch.bfloat16,
        torch.float16,
        torch.float16,
    ),
}


def _skip_if_not_sm90_or_later():
    from flashinfer.utils import get_compute_capability

    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN ucache requires SM90+, got SM{cc[0]}{cc[1]}")


def _load_flush(arm: str, path: str = _FLUSH_PATH):
    """Load one module copy per (dtype arm, module path); the dtype set is
    chosen at import time via the GDN_UCACHE_* env vars."""
    cache_key = (arm, path)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    io_env, state_env, ring_env, _, _, _ = ARMS[arm]
    old = {
        k: os.environ.pop(k, None)
        for k in (
            "GDN_UCACHE_IO_DTYPE",
            "GDN_UCACHE_STATE_DTYPE",
            "GDN_UCACHE_RING_DTYPE",
        )
    }
    if io_env:
        os.environ["GDN_UCACHE_IO_DTYPE"] = io_env
    if state_env:
        os.environ["GDN_UCACHE_STATE_DTYPE"] = state_env
    if ring_env:
        os.environ["GDN_UCACHE_RING_DTYPE"] = ring_env
    try:
        spec = importlib.util.spec_from_file_location(
            f"uc_{Path(path).stem}_{arm}", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _MODULE_CACHE[cache_key] = mod
    return mod


# ---------------------------------------------------------------------------
# The fp32 oracle. Slow and simple on purpose - this IS the spec.
# One request at a time. State S is [HV, V, K].
# ---------------------------------------------------------------------------
def _ref_fp32(q, k, v, a, b, A_log, dt_bias, S0, kc, uc, gc, P):
    f = torch.float32
    grp = HV // H
    S = S0.to(f).clone()

    # 1) replay the P live ring entries into the state:
    #    S <- exp(G_P) * S0 + sum_j exp(G_P - g_j) * u_j (x) k_j
    if P > 0:
        GP = gc[:, P - 1].to(f)  # [HV]
        w = torch.exp(GP[:, None] - gc[:, :P].to(f))  # [HV, P]
        kc_hv = kc[:, :P].to(f).repeat_interleave(grp, dim=0)  # [HV, P, K]
        S = torch.exp(GP)[:, None, None] * S + torch.einsum(
            "hpv,hpk->hvk", w[:, :, None] * uc[:, :P].to(f), kc_hv
        )
    S_after_history = S.clone()  # == the state a fold commits to the pool

    # 2) run the T new draft tokens through the exact delta-rule recurrence
    khat = F.normalize(k.to(f), dim=-1)
    qhat = F.normalize(q.to(f), dim=-1) * SCALE
    n_tok = q.shape[0]  # T-generic: the STP (T=1) tests below reuse this oracle
    y = torch.zeros(n_tok, HV, V, dtype=f, device=q.device)
    for t in range(n_tok):
        la = -torch.exp(A_log.to(f)) * F.softplus(a[t].to(f) + dt_bias.to(f))
        beta = torch.sigmoid(b[t].to(f))  # [HV]
        k_hv = khat[t].repeat_interleave(grp, dim=0)  # [HV, K]
        q_hv = qhat[t].repeat_interleave(grp, dim=0)
        S = S * torch.exp(la)[:, None, None]  # decay
        pred = torch.einsum("hvk,hk->hv", S, k_hv)  # S k
        u_t = (v[t].to(f) - pred) * beta[:, None]  # delta rule
        S = S + u_t[:, :, None] * k_hv[:, None, :]  # + u (x) k
        y[t] = torch.einsum("hvk,hk->hv", S, q_hv)  # out = S q
    return y, S_after_history


# ---------------------------------------------------------------------------
# Case builder: consistent inputs + rings for B requests.
# ---------------------------------------------------------------------------
def _make_case(
    B,
    hist_lens,
    io_dtype,
    state_dtype,
    seed,
    ring_dtype=None,
    bases=None,
    t_tokens=T,
    ring_slots=RING,
):
    ring_dtype = ring_dtype or io_dtype
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io_dtype)

    q, k = rn(B, t_tokens, H, K), rn(B, t_tokens, H, K)
    v, a, b = (
        rn(B, t_tokens, HV, V, sc=0.5),
        rn(B, t_tokens, HV, sc=0.5),
        rn(B, t_tokens, HV),
    )
    A_log = (
        torch.full((HV,), -3.0, device=DEV)
        + torch.rand(HV, generator=g, device=DEV) * 0.3
    ).to(io_dtype)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(state_dtype)
    kc = torch.zeros(B, H, ring_slots, K, dtype=ring_dtype, device=DEV)
    uc = torch.zeros(B, HV, ring_slots, V, dtype=ring_dtype, device=DEV)
    gc = torch.zeros(B, HV, ring_slots, dtype=torch.float32, device=DEV)
    hl = torch.tensor(hist_lens, dtype=torch.int32, device=DEV)
    bases = bases or [0] * B
    cb = torch.tensor(bases, dtype=torch.int32, device=DEV)
    for r in range(B):
        P = int(hl[r])
        if P == 0:
            continue
        # logical history rows j land at PHYSICAL ring rows (base + j) % RING
        rows = torch.tensor(
            [(bases[r] + j) % ring_slots for j in range(P)],
            dtype=torch.long,
            device=DEV,
        )
        kh = torch.randn(H, P, K, generator=g, device=DEV)
        kc[r, :, rows] = F.normalize(kh, dim=-1).to(ring_dtype)
        uc[r, :, rows] = (torch.randn(HV, P, V, generator=g, device=DEV) * 0.3).to(
            ring_dtype
        )
        la = -(torch.rand(HV, P, generator=g, device=DEV) * 0.3 + 0.003)
        gc[r, :, rows] = torch.cumsum(la, dim=-1)
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx


def _logical_rings(kc_r, uc_r, gc_r, base):
    """Gather one request's PHYSICAL ring rows into logical window order
    (row j = physical (base+j) % RING) so the fp32 oracle stays unchanged."""
    rows = torch.tensor(
        [(base + j) % RING for j in range(W)], dtype=torch.long, device=kc_r.device
    )
    return (
        kc_r.index_select(1, rows),
        uc_r.index_select(1, rows),
        gc_r.index_select(1, rows),
    )


HISTORIES = {
    "empty_P0": [0, 0, 0, 0],
    "replay_P12": [12, 12, 12, 12],
    "fold_mixed": [13, 12, 13, 12],  # rows 0 and 2 fold
}
BASES = {
    "base0": [0, 0, 0, 0],
    # wrapped windows: base+P crosses RING for some rows (the ring path)
    "wrap": [28, 5, 30, 17],
}
Y_TOL = 8e-3  # observed max ~7e-4 across arms; 10x margin
STATE_TOL = 2e-2  # committed state accumulates P outer products first


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("history", list(HISTORIES))
@pytest.mark.parametrize("basecase", list(BASES))
def test_output_matches_fp32_reference(arm, history, basecase):
    _skip_if_not_sm90_or_later()
    mod = _load_flush(arm)
    _, _, _, io_dtype, state_dtype, ring_dtype = ARMS[arm]
    B = 4
    bases = BASES[basecase]
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _make_case(
        B,
        HISTORIES[history],
        io_dtype,
        state_dtype,
        seed=1234,
        ring_dtype=ring_dtype,
        bases=bases,
    )
    pool_before = pool.clone()  # fold rows mutate the pool; ref needs the old state

    y = mod.gated_delta_rule_mtp_ucache_flush(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc.clone(),
        u_cache=uc.clone(),
        g_cache=gc.clone(),
        hist_len=hl.clone(),  # kernel mutates rings (appends); ref reads originals
        cache_base=cb.clone(),
        scale=SCALE,
        flush_min=FLUSH_MIN,
    )

    for r in range(B):
        kc_l, uc_l, gc_l = _logical_rings(kc[r], uc[r], gc[r], bases[r])
        y_ref, _ = _ref_fp32(
            q[r],
            k[r],
            v[r],
            a[r],
            b[r],
            A_log,
            dt_bias,
            pool_before[r],
            kc_l,
            uc_l,
            gc_l,
            int(hl[r]),
        )
        err = (y[r].float() - y_ref).abs().max().item()
        assert err < Y_TOL, (
            f"row {r} ({history}, {arm}, {basecase}): |y - fp32 ref| = {err:.2e}"
        )


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("basecase", list(BASES))
def test_folded_state_matches_fp32_reference(arm, basecase):
    """On a fold (hist_len >= flush_min) the kernel writes the ring-folded
    checkpoint back to the pool. Compare that committed state to the oracle."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush(arm)
    _, _, _, io_dtype, state_dtype, ring_dtype = ARMS[arm]
    B = 4
    bases = BASES[basecase]
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _make_case(
        B,
        [13, 13, 13, 13],
        io_dtype,
        state_dtype,
        seed=99,
        ring_dtype=ring_dtype,
        bases=bases,
    )
    pool_before = pool.clone()

    mod.gated_delta_rule_mtp_ucache_flush(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc.clone(),
        u_cache=uc.clone(),
        g_cache=gc.clone(),
        hist_len=hl.clone(),  # kernel mutates rings (appends); ref reads originals
        cache_base=cb.clone(),
        scale=SCALE,
        flush_min=FLUSH_MIN,
    )

    for r in range(B):
        kc_l, uc_l, gc_l = _logical_rings(kc[r], uc[r], gc[r], bases[r])
        _, S_ref = _ref_fp32(
            q[r],
            k[r],
            v[r],
            a[r],
            b[r],
            A_log,
            dt_bias,
            pool_before[r],
            kc_l,
            uc_l,
            gc_l,
            13,
        )
        err = (pool[r].float() - S_ref).abs().max().item()
        assert err < STATE_TOL, (
            f"row {r} ({arm}, {basecase}): |committed - fp32 ref| = {err:.2e}"
        )


def test_flush_never_overwrites_live_window():
    """THE ring property (the old single-buffer design's race): a flush must
    not modify any physical ring row inside the live window [base, base+P) —
    sibling CTAs read those rows as their fold source. Appends must land at
    (base+P+s) & RING_MASK only."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16")
    _, _, _, io_dtype, state_dtype, ring_dtype = ARMS["bf16"]
    B = 4
    bases = [28, 0, 30, 12]  # rows 0/2 wrap through the ring end
    hist = [13, 13, 14, 16]  # ALL rows flush (P >= 13)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _make_case(
        B, hist, io_dtype, state_dtype, seed=4242, ring_dtype=ring_dtype, bases=bases
    )
    kc_in, uc_in, gc_in = kc.clone(), uc.clone(), gc.clone()

    mod.gated_delta_rule_mtp_ucache_flush(
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
        hist_len=hl.clone(),
        cache_base=cb.clone(),
        scale=SCALE,
        flush_min=FLUSH_MIN,
    )

    for r in range(B):
        P = hist[r]
        window = [(bases[r] + j) % RING for j in range(P)]
        appends = [(bases[r] + P + s) % RING for s in range(T)]
        for rows, name, before, after in (
            (window, "k", kc_in, kc),
            (window, "u", uc_in, uc),
            (window, "g", gc_in, gc),
        ):
            w = torch.tensor(rows, dtype=torch.long, device=DEV)
            assert torch.equal(
                before[r].index_select(1, w), after[r].index_select(1, w)
            ), f"row {r}: flush modified live {name} window rows {rows}"
        # and the appends actually landed (k rows are L2-normed, non-zero)
        ap = torch.tensor(appends, dtype=torch.long, device=DEV)
        assert kc[r].index_select(1, ap).float().abs().sum() > 0, (
            f"row {r}: no k append at {appends}"
        )


def test_fp16_state_commits_more_precisely_than_bf16():
    """The point of the fp16-state mode: 10 mantissa bits vs bf16's 7 means
    the committed checkpoint sits closer to fp32 truth. Same bf16 inputs,
    only the pool dtype differs; compare mean committed-state error."""
    _skip_if_not_sm90_or_later()
    errs = {}
    for arm in ("bf16", "fp16_state"):
        mod = _load_flush(arm)
        _, _, _, io_dtype, state_dtype, ring_dtype = ARMS[arm]
        tot = 0.0
        for seed in (7, 8, 9, 10):
            B = 4
            q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _make_case(
                B, [13] * B, io_dtype, state_dtype, seed=seed, ring_dtype=ring_dtype
            )
            pool_before = pool.clone()
            mod.gated_delta_rule_mtp_ucache_flush(
                A_log,
                a,
                dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=pool,
                initial_state_indices=idx,
                k_cache=kc.clone(),
                u_cache=uc.clone(),
                g_cache=gc.clone(),
                hist_len=hl.clone(),
                cache_base=cb.clone(),
                scale=SCALE,
                flush_min=FLUSH_MIN,
            )
            for r in range(B):
                _, S_ref = _ref_fp32(
                    q[r],
                    k[r],
                    v[r],
                    a[r],
                    b[r],
                    A_log,
                    dt_bias,
                    pool_before[r],
                    kc[r],
                    uc[r],
                    gc[r],
                    13,
                )
                tot += (pool[r].float() - S_ref).abs().mean().item()
        errs[arm] = tot
    assert errs["fp16_state"] < errs["bf16"], (
        f"fp16 state should commit closer to fp32 truth: {errs}"
    )


def _load_flush_strided():
    """Load a flush-module copy with the strided-QKV path enabled
    (SGLANG_GDN_WY_STRIDED_QKV read at import). This is the path vLLM uses:
    q/k/v (and a/b) arrive as chunk views of packed tensors."""
    old = os.environ.get("SGLANG_GDN_WY_STRIDED_QKV")
    os.environ["SGLANG_GDN_WY_STRIDED_QKV"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("uc_flush_strided", _FLUSH_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("SGLANG_GDN_WY_STRIDED_QKV", None)
        else:
            os.environ["SGLANG_GDN_WY_STRIDED_QKV"] = old
    return mod


def test_strided_ab_matches_compact():
    """Regression for the a/b token-stride bug (sb_t/sb_b): when q/k/v are
    chunk views of one packed tensor (matching stride -> strided-QKV path,
    the vLLM setup) and a/b are chunk views (stride(1)=2*HV != HV), the
    output must equal a fully-compact run with the SAME values. With the old
    `sb_t = HV` the packed b was read from the wrong rows (~2e-2 error)."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush_strided()
    io = getattr(mod, "IO_TORCH", torch.bfloat16)
    ring = getattr(mod, "RING_TORCH", io)
    B = 4
    g = torch.Generator(device=DEV).manual_seed(0)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io)

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, a, b = rn(B, T, HV, V, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (torch.rand(HV, generator=g, device=DEV) * 6 - 4.5).to(io)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(torch.bfloat16)

    def run(q_, k_, v_, a_, b_):
        idx = torch.arange(B, dtype=torch.int32, device=DEV)
        return mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a_,
            dt_bias,
            q=q_,
            k=k_,
            v=v_,
            b=b_,
            initial_state_source=pool.clone(),
            initial_state_indices=idx,
            k_cache=torch.zeros(B, H, RING, K, dtype=ring, device=DEV),
            u_cache=torch.zeros(B, HV, RING, V, dtype=ring, device=DEV),
            g_cache=torch.zeros(B, HV, RING, dtype=torch.float32, device=DEV),
            hist_len=torch.zeros(B, dtype=torch.int32, device=DEV),
            cache_base=torch.zeros(B, dtype=torch.int32, device=DEV),
            scale=SCALE,
            flush_min=FLUSH_MIN,
        ).float()

    # pack q/k/v into one wide tensor -> matching token stride (strided-QKV)
    qw = H * K + H * K + HV * V
    wqkv = torch.zeros(B, T, qw, dtype=io, device=DEV)
    wqkv[:, :, : H * K] = q.reshape(B, T, H * K)
    wqkv[:, :, H * K : 2 * H * K] = k.reshape(B, T, H * K)
    wqkv[:, :, 2 * H * K :] = v.reshape(B, T, HV * V)
    q_s = wqkv[:, :, : H * K].reshape(B, T, H, K)
    k_s = wqkv[:, :, H * K : 2 * H * K].reshape(B, T, H, K)
    v_s = wqkv[:, :, 2 * H * K :].reshape(B, T, HV, V)
    wab = torch.zeros(B, T, 2 * HV, dtype=io, device=DEV)
    wab[:, :, :HV], wab[:, :, HV:] = a, b
    a_s, b_s = wab[:, :, :HV], wab[:, :, HV:]
    assert a_s.stride(1) == 2 * HV and q_s.stride(1) == qw  # confirm strided

    y_compact = run(q, k, v, a, b)
    y_strided = run(q_s, k_s, v_s, a_s, b_s)
    err = (y_compact - y_strided).abs().max().item()
    assert err < 2e-3, (
        f"strided (chunk-view) a/b must match compact; got max|d|={err:.3e} "
        "(the sb_t=HV bug reads packed b from the wrong rows)"
    )


def test_standalone_commit_guards_reject_silent_corruption_patterns():
    """The standalone-commit path (restart_hist_on_flush=True) mutates the
    cursors in place, so the wrapper must fail LOUDLY on the calling patterns
    that would otherwise corrupt silently: a throwaway default base (the
    commit lands in a temp the caller never sees), a non-contiguous cursor
    (the commit lands in a .contiguous() copy), and out-of-range cursors
    (the ring's & RING_MASK addressing keeps bad windows in-bounds, so they
    would produce well-formed wrong numbers instead of crashes)."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16")
    B = 4
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _make_case(
        B,
        [12, 13, 12, 13],
        torch.bfloat16,
        torch.bfloat16,
        seed=7,
        ring_dtype=torch.bfloat16,
        bases=[0, 5, 28, 30],
    )
    common = dict(
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        scale=SCALE,
        flush_min=FLUSH_MIN,
    )

    # commit mode without a caller-owned base: was silent corruption after
    # the first flush (base slid on an internal temp), now a ValueError.
    with pytest.raises(ValueError, match="caller-owned cache_base"):
        mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            hist_len=hl.clone(),
            cache_base=None,
            restart_hist_on_flush=True,
            **common,
        )

    # commit mode with a non-contiguous base: was a lost commit inside a
    # silent .contiguous() copy, now an assertion.
    cb_col = torch.zeros(B, 2, dtype=torch.int32, device=DEV)[:, 0]
    assert not cb_col.is_contiguous()
    with pytest.raises(AssertionError, match="contiguous"):
        mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            hist_len=hl.clone(),
            cache_base=cb_col,
            restart_hist_on_flush=True,
            **common,
        )

    # oversized window: was silent wrong output + corrupted fold (17..28)
    # or a revived append-into-window race (>= 29), now an assertion.
    hl_bad = hl.clone()
    hl_bad[1] = 17
    with pytest.raises(AssertionError, match="hist_len out of legal range"):
        mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            hist_len=hl_bad,
            cache_base=cb.clone(),
            restart_hist_on_flush=True,
            **common,
        )

    # out-of-range base: masked into range by the kernel, so it would hide a
    # caller bookkeeping bug forever; now an assertion.
    cb_bad = cb.clone()
    cb_bad[0] = 32
    with pytest.raises(AssertionError, match="cache_base out of legal range"):
        mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            hist_len=hl.clone(),
            cache_base=cb_bad,
            restart_hist_on_flush=True,
            **common,
        )

    # the caller-owned-commit path (vLLM serving) is untouched: cache_base
    # may be None (read-only base=0), no host sync, no raise.
    y = mod.gated_delta_rule_mtp_ucache_flush(
        A_log,
        a,
        dt_bias,
        hist_len=hl.clone(),
        cache_base=None,
        restart_hist_on_flush=False,
        **common,
    )
    assert y.shape[1] == T


# ---------------------------------------------------------------------------
# Verify-only kernel (gdn_decode_bf16_wy_ucache.py): thin oracle coverage.
# Same recurrence as the flush kernel's verify path, judged by the SAME
# _ref_fp32 oracle, but with this kernel's own contract: the LEGACY 16-deep
# flat ring (history at physical rows [0, P), no cache_base — so physical
# order IS logical order), a READ-ONLY checkpoint pool, bf16 only.
# ---------------------------------------------------------------------------
_VERIFY_PATH = str(
    Path(__file__).resolve().parents[2]
    / "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache.py"
)


def _load_verify():
    if "verify_only" in _MODULE_CACHE:
        return _MODULE_CACHE["verify_only"]
    spec = importlib.util.spec_from_file_location("uc_verify", _VERIFY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULE_CACHE["verify_only"] = mod
    return mod


def test_verify_only_output_matches_fp32_reference():
    """Thin oracle test for the verify-only kernel: bare checkpoint (P=0),
    mid-window replay (P=5), and the max legal window (P=12; P+T == 16) in
    one launch. Also pins the two contracts the flush kernel doesn't share:
    the checkpoint pool is never written, and the speculative appends land
    at flat-ring rows [P, P+T) without touching the live window [0, P)."""
    _skip_if_not_sm90_or_later()
    mod = _load_verify()
    B = 4
    hist = [0, 5, 12, 12]
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, _, idx = _make_case(
        B, hist, torch.bfloat16, torch.bfloat16, seed=321
    )
    # this kernel wants the legacy 16-deep flat ring; base 0 in _make_case
    # fills rows [0, P), so truncating the 32-slot rings is layout-exact
    kc = kc[:, :, :W].contiguous()
    uc = uc[:, :, :W].contiguous()
    gc = gc[:, :, :W].contiguous()
    pool_before = pool.clone()
    kc_in, uc_in, gc_in = kc.clone(), uc.clone(), gc.clone()

    y = mod.gated_delta_rule_mtp_ucache(
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
        hist_len=hl.clone(),
        scale=SCALE,
    )

    for r in range(B):
        y_ref, _ = _ref_fp32(
            q[r],
            k[r],
            v[r],
            a[r],
            b[r],
            A_log,
            dt_bias,
            pool_before[r],
            kc_in[r],
            uc_in[r],
            gc_in[r],
            hist[r],
        )
        err = (y[r].float() - y_ref).abs().max().item()
        assert err < Y_TOL, (
            f"row {r} (verify-only, P={hist[r]}): |y - fp32 ref| = {err:.2e}"
        )

    # the checkpoint is read-only here; folding belongs to the flush kernel
    assert torch.equal(pool, pool_before), (
        "verify-only kernel must never write the state pool"
    )

    for r in range(B):
        P = hist[r]
        if P > 0:
            wnd = torch.arange(P, device=DEV)
            for name, before, after in (
                ("k", kc_in, kc),
                ("u", uc_in, uc),
                ("g", gc_in, gc),
            ):
                assert torch.equal(
                    before[r].index_select(1, wnd), after[r].index_select(1, wnd)
                ), f"row {r}: verify modified live {name} window [0, {P})"
        # and the appends actually landed (k rows are L2-normed, non-zero)
        ap = torch.arange(P, P + T, device=DEV)
        assert kc[r].index_select(1, ap).float().abs().sum() > 0, (
            f"row {r}: no k append at rows [{P}, {P + T})"
        )


def test_verify_only_rejects_non_bf16_activations():
    """GDN-C1/C3 guard (#4214): the verify-only kernel body is bf16-only, so
    non-bf16 activations must fail loudly at the wrapper boundary instead of
    compiling a silently-reinterpreting cubin (or throwing on a cache hit)."""
    _skip_if_not_sm90_or_later()
    mod = _load_verify()
    B = 4
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, _, idx = _make_case(
        B, [0] * B, torch.bfloat16, torch.bfloat16, seed=5
    )
    kc = kc[:, :, :W].contiguous()
    uc = uc[:, :, :W].contiguous()
    gc = gc[:, :, :W].contiguous()
    with pytest.raises(AssertionError, match="bf16-only"):
        mod.gated_delta_rule_mtp_ucache(
            A_log,
            a,
            dt_bias,
            q=q.half(),
            k=k,
            v=v,
            b=b,
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl.clone(),
            scale=SCALE,
        )


# ===========================================================================
# STP (T=1, single-token-prediction) fold-absorb kernel
# (flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_stp.py — read its module
# docstring; it is a fork of the T=4 flush kernel above with vLLM-ReplaySSM-
# compatible flush semantics).
#
# Reuses this file's fp32 oracle / dtype arms / case builder. STP-specific
# risks pinned below:
#   - FOLD-ABSORB: a flush commits S_new = a0*S_h + u0 (x) k0^T (the
#     POST-token state; the oracle's S_after_history plus one token step) —
#     checked at fm=15 (full window) AND mid-window fm (weight-0 tail rows);
#   - SINGLE-ROW appends: exactly one (k, u, g) entry per verify step, no
#     T=4 filler entries (FLAT 16-slot buffer, slot P <= 15, base pinned 0);
#   - fused commit: flush -> hist_len = 0 (window restarts EMPTY), verify ->
#     P + 1;
#   - exhaustive P in [0, 15]; shuffled non-identity state indices; 200-step
#     drift vs a never-resynced fp32 reference; 16-token-cycle multistep;
#     prepadded [B,4] zero-copy path bitwise == the staging path.
# ===========================================================================
_STP_PATH = str(
    Path(__file__).resolve().parents[2]
    / "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_stp.py"
)
STP_RING = 16  # flat physical buffer (fork's RING_SLOTS)
STP_FM = 15  # default flush_min == vLLM's flush-at-(L-1)


def _stp_case(B, hist_lens, arm, seed):
    """T=1 case on the fork's flat 16-slot buffer (live entries at [0, P))."""
    _, _, _, io_dtype, state_dtype, ring_dtype = ARMS[arm]
    return _make_case(
        B,
        hist_lens,
        io_dtype,
        state_dtype,
        seed,
        ring_dtype=ring_dtype,
        t_tokens=1,
        ring_slots=STP_RING,
    )


def _stp_token_step(S_hist, q1, k1, v1, a1, b1, A_log, dt_bias):
    """One fp32 delta-rule step from the oracle's replayed (pre-token) state.
    Supplements _ref_fp32 for the fold-absorb check: the fork's flush commits
    the POST-token state, which _ref_fp32 does not return."""
    f = torch.float32
    grp = HV // H
    khat = F.normalize(k1.to(f), dim=-1).repeat_interleave(grp, dim=0)
    qhat = (F.normalize(q1.to(f), dim=-1) * SCALE).repeat_interleave(grp, dim=0)
    la = -torch.exp(A_log.to(f)) * F.softplus(a1.to(f) + dt_bias.to(f))
    beta = torch.sigmoid(b1.to(f))
    S = S_hist * torch.exp(la)[:, None, None]
    pred = torch.einsum("hvk,hk->hv", S, khat)
    u_t = (v1.to(f) - pred) * beta[:, None]
    S = S + u_t[:, :, None] * khat[:, None, :]
    y = torch.einsum("hvk,hk->hv", S, qhat)
    return y, S, u_t, khat


STP_HISTORIES = {
    "empty_P0": [0, 0, 0, 0],
    "replay_P14": [14, 14, 14, 14],  # deepest verify at fm=15
    "fold_mixed": [15, 14, 15, 0],  # rows 0 and 2 fold (absorb)
}


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("history", list(STP_HISTORIES))
def test_stp_output_and_absorbed_fold_match_fp32_reference(arm, history):
    _skip_if_not_sm90_or_later()
    mod = _load_flush(arm, _STP_PATH)
    assert mod.RING_SLOTS == STP_RING
    B = 4
    hist = STP_HISTORIES[history]
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, hist, arm, seed=1234
    )
    pool_before, uc0 = pool.clone(), uc.clone()

    y = mod.gated_delta_rule_stp_ucache_flush(
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
        flush_min=STP_FM,
    )

    for r in range(B):
        P = hist[r]
        y_ref, S_hist = _ref_fp32(
            q[r],
            k[r],
            v[r],
            a[r],
            b[r],
            A_log,
            dt_bias,
            pool_before[r],
            kc[r],
            uc0[r],
            gc[r],
            P,
        )
        err = (y[r].float() - y_ref).abs().max().item()
        assert err < Y_TOL, f"row {r} ({history}, {arm}): |y - ref| = {err:.2e}"
        if P >= STP_FM:
            # fold-absorb: the committed state is the POST-token state
            _, S_tok, _, _ = _stp_token_step(
                S_hist,
                q[r, 0],
                k[r, 0],
                v[r, 0],
                a[r, 0],
                b[r, 0],
                A_log,
                dt_bias,
            )
            serr = (pool[r].float() - S_tok).abs().max().item()
            assert serr < STATE_TOL, f"row {r} ({arm}): |absorbed - ref| = {serr:.2e}"
            assert int(hl[r]) == 0, "flush must restart the window EMPTY"
        else:
            assert torch.equal(pool[r], pool_before[r])
            assert int(hl[r]) == P + 1
        assert int(cb[r]) == 0  # flat: base pinned at 0


def test_stp_single_row_appends_no_filler():
    """Exactly one (k, u, g) entry written per verify call, at slot P; every
    other slot bitwise-untouched (the parent T=4 kernel would have written 3
    filler rows at P+1..P+3)."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    B = 4
    hist = [0, 5, 10, 14]
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, hist, "bf16", seed=7
    )
    kc0, uc0, gc0 = kc.clone(), uc.clone(), gc.clone()

    mod.gated_delta_rule_stp_ucache_flush(
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
        flush_min=STP_FM,
    )

    for r in range(B):
        P = hist[r]
        for s in range(STP_RING):
            same = (
                torch.equal(kc[r, :, s], kc0[r, :, s])
                and torch.equal(uc[r, :, s], uc0[r, :, s])
                and torch.equal(gc[r, :, s], gc0[r, :, s])
            )
            if s == P:
                assert not same, f"row {r}: append slot {s} not written"
            else:
                assert same, f"row {r}: slot {s} modified (filler leak)"


def test_stp_mid_window_flush_min():
    """fm=10 (predictive flushing): the fold's weight-0 tail rows (P < r <
    16) must not contaminate the absorbed state even though the token row P
    is live in the same GEMM."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    B = 4
    hist = [10, 10, 9, 0]  # rows 0,1 fold at fm=10 (self-commit cap); 2,3 verify
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, hist, "bf16", seed=99
    )
    pool_before, uc0 = pool.clone(), uc.clone()

    y = mod.gated_delta_rule_stp_ucache_flush(
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
        flush_min=10,
    )

    for r in range(B):
        P = hist[r]
        y_ref, S_hist = _ref_fp32(
            q[r],
            k[r],
            v[r],
            a[r],
            b[r],
            A_log,
            dt_bias,
            pool_before[r],
            kc[r],
            uc0[r],
            gc[r],
            P,
        )
        assert (y[r].float() - y_ref).abs().max().item() < Y_TOL
        if P >= 10:
            _, S_tok, _, _ = _stp_token_step(
                S_hist,
                q[r, 0],
                k[r, 0],
                v[r, 0],
                a[r, 0],
                b[r, 0],
                A_log,
                dt_bias,
            )
            serr = (pool[r].float() - S_tok).abs().max().item()
            assert serr < STATE_TOL, f"row {r}: |absorbed - ref| = {serr:.2e}"
            assert int(hl[r]) == 0
        else:
            assert torch.equal(pool[r], pool_before[r])
            assert int(hl[r]) == P + 1


@pytest.mark.parametrize("flush_min", [15, 8])
def test_stp_exhaustive_history_depths(flush_min):
    """One batch row per legal P in [0, flush_min]: every replay depth AND
    every legal fold depth checked against the oracle."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    hist = list(range(flush_min + 1))  # P == fm folds
    B = len(hist)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, hist, "bf16", seed=101
    )
    pool0, uc0 = pool.clone(), uc.clone()

    y = mod.gated_delta_rule_stp_ucache_flush(
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
    )

    for r in range(B):
        P = hist[r]
        y_ref, S_hist = _ref_fp32(
            q[r],
            k[r],
            v[r],
            a[r],
            b[r],
            A_log,
            dt_bias,
            pool0[r],
            kc[r],
            uc0[r],
            gc[r],
            P,
        )
        err = (y[r].float() - y_ref).abs().max().item()
        assert err < Y_TOL, f"P={P} (fm={flush_min}): |y - ref| = {err:.2e}"
        if flush_min <= P:
            _, S_tok, _, _ = _stp_token_step(
                S_hist,
                q[r, 0],
                k[r, 0],
                v[r, 0],
                a[r, 0],
                b[r, 0],
                A_log,
                dt_bias,
            )
            serr = (pool[r].float() - S_tok).abs().max().item()
            assert serr < STATE_TOL, f"P={P}: |absorbed - ref| = {serr:.2e}"
            assert int(hl[r]) == 0
        else:
            assert torch.equal(pool[r], pool0[r])
            assert int(hl[r]) == P + 1


@pytest.mark.parametrize("B", [1, 3, 16])
def test_stp_nonidentity_state_indices(B):
    """Shuffled state-slot indices into a pool of 2B+3 slots: pool reads,
    fold writes, AND all three ring streams must follow the indices (the
    other tests use arange). Non-indexed slots must be untouched."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    pool_n = 2 * B + 3
    perm = torch.randperm(pool_n, generator=torch.Generator().manual_seed(7))
    idx = perm[:B].to(torch.int32).to(DEV)
    hist_by_row = [[15, 7, 0, 12, 15, 3, 14, 15][r % 8] for r in range(B)]
    # build a pool-sized case, then place each row's history at its SLOT
    case = _stp_case(pool_n, [0] * pool_n, "bf16", seed=202 + B)
    _, _, _, _, _, A_log, dt_bias, pool, kc, uc, gc, _, _, _ = case
    gsrc = torch.Generator(device=DEV).manual_seed(400 + B)
    for r in range(B):
        s, P = int(idx[r]), hist_by_row[r]
        if P == 0:
            continue
        kh = torch.randn(H, P, K, generator=gsrc, device=DEV)
        kc[s, :, :P] = F.normalize(kh, dim=-1).to(kc.dtype)
        uc[s, :, :P] = (torch.randn(HV, P, V, generator=gsrc, device=DEV) * 0.3).to(
            uc.dtype
        )
        la = -(torch.rand(HV, P, generator=gsrc, device=DEV) * 0.3 + 0.003)
        gc[s, :, :P] = torch.cumsum(la, dim=-1)
    q = (torch.randn(B, 1, H, K, generator=gsrc, device=DEV)).to(torch.bfloat16)
    k = (torch.randn(B, 1, H, K, generator=gsrc, device=DEV)).to(torch.bfloat16)
    v = (torch.randn(B, 1, HV, V, generator=gsrc, device=DEV) * 0.5).to(torch.bfloat16)
    a = (torch.randn(B, 1, HV, generator=gsrc, device=DEV) * 0.5).to(torch.bfloat16)
    b = (torch.randn(B, 1, HV, generator=gsrc, device=DEV)).to(torch.bfloat16)
    hl = torch.tensor(hist_by_row, dtype=torch.int32, device=DEV)
    cb = torch.zeros(B, dtype=torch.int32, device=DEV)
    pool0, kc0, uc0, gc0 = pool.clone(), kc.clone(), uc.clone(), gc.clone()

    y = mod.gated_delta_rule_stp_ucache_flush(
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
        flush_min=STP_FM,
    )

    used = {int(s) for s in idx}
    for s in range(pool_n):
        if s not in used:
            assert torch.equal(pool[s], pool0[s]), f"unindexed pool slot {s} touched"
            assert torch.equal(uc[s], uc0[s]) and torch.equal(kc[s], kc0[s])
            assert torch.equal(gc[s], gc0[s]), f"unindexed ring slot {s} touched"
    for r in range(B):
        s, P = int(idx[r]), hist_by_row[r]
        y_ref, S_hist = _ref_fp32(
            q[r],
            k[r],
            v[r],
            a[r],
            b[r],
            A_log,
            dt_bias,
            pool0[s],
            kc0[s],
            uc0[s],
            gc0[s],
            P,
        )
        err = (y[r].float() - y_ref).abs().max().item()
        assert err < Y_TOL, f"row {r} slot {s} P={P}: |y - ref| = {err:.2e}"
        _, S_tok, u_ref, _ = _stp_token_step(
            S_hist, q[r, 0], k[r, 0], v[r, 0], a[r, 0], b[r, 0], A_log, dt_bias
        )
        if P >= STP_FM:
            serr = (pool[s].float() - S_tok).abs().max().item()
            assert serr < STATE_TOL, f"row {r} slot {s}: fold err {serr:.2e}"
        else:
            du = (uc[s, :, P].float() - u_ref).abs().max().item()
            dk = (
                (kc[s, :, P].float() - F.normalize(k[r, 0].float(), dim=-1))
                .abs()
                .max()
                .item()
            )
            assert du < 2e-2 and dk < 8e-3, f"row {r} slot {s}: append errs {du} {dk}"


def test_stp_multistep_decode_16_token_cycles():
    """40 real STP steps at fm=15: hist cycles 0..14 -> fold -> 0 (16-token
    cycles, exactly 2 folds per request), each step oracle-checked from the
    kernel's own ring/pool state."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    B, n_steps = 4, 40
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, [0, 0, 0, 0], "bf16", seed=55
    )

    g = torch.Generator(device=DEV).manual_seed(777)
    bf = torch.bfloat16
    n_folds = 0
    for step in range(n_steps):
        qs = (torch.randn(B, 1, H, K, generator=g, device=DEV)).to(bf)
        ks = (torch.randn(B, 1, H, K, generator=g, device=DEV)).to(bf)
        vs = (torch.randn(B, 1, HV, V, generator=g, device=DEV) * 0.5).to(bf)
        as_ = (torch.randn(B, 1, HV, generator=g, device=DEV) * 0.5).to(bf)
        bs = (torch.randn(B, 1, HV, generator=g, device=DEV)).to(bf)

        pool_before, uc_before, hl_before = pool.clone(), uc.clone(), hl.clone()

        y = mod.gated_delta_rule_stp_ucache_flush(
            A_log,
            as_,
            dt_bias,
            q=qs,
            k=ks,
            v=vs,
            b=bs,
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=STP_FM,
        )

        for r in range(B):
            P = int(hl_before[r])
            y_ref, S_hist = _ref_fp32(
                qs[r],
                ks[r],
                vs[r],
                as_[r],
                bs[r],
                A_log,
                dt_bias,
                pool_before[r],
                kc[r],
                uc_before[r],
                gc[r],
                P,
            )
            err = (y[r, 0].float() - y_ref[0]).abs().max().item()
            assert err < Y_TOL, f"step {step} row {r}: |y - ref| = {err:.2e}"
            if P >= STP_FM:
                n_folds += 1
                _, S_tok, _, _ = _stp_token_step(
                    S_hist,
                    qs[r, 0],
                    ks[r, 0],
                    vs[r, 0],
                    as_[r, 0],
                    bs[r, 0],
                    A_log,
                    dt_bias,
                )
                serr = (pool[r].float() - S_tok).abs().max().item()
                assert serr < STATE_TOL, (
                    f"step {step} row {r}: |absorbed - ref| = {serr:.2e}"
                )
                assert int(hl[r]) == 0
            else:
                assert torch.equal(pool[r], pool_before[r])
                assert int(hl[r]) == P + 1
        assert int(cb.max()) == 0
    assert n_folds == 2 * B, f"expected {2 * B} folds, saw {n_folds}"


def test_stp_long_run_drift_200_steps():
    """200 real decode steps (12 full fold cycles) against a fp32 reference
    state that is NEVER resynced from the kernel — bounds the accumulated
    rounding of the absorbed-fold chain. MEASURED: max drift 6.5e-4 and FLAT
    across all 200 steps (the delta rule's contractive decay washes out
    per-fold rounding); 5e-2 is ~77x that."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    B, n_steps = 2, 200
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, [0, 0], "bf16", seed=303
    )
    S_ref = pool.float().clone()  # continuous fp32 reference, per row

    g = torch.Generator(device=DEV).manual_seed(777)
    bf = torch.bfloat16
    max_drift = 0.0
    for _step in range(n_steps):
        qs = (torch.randn(B, 1, H, K, generator=g, device=DEV)).to(bf)
        ks = (torch.randn(B, 1, H, K, generator=g, device=DEV)).to(bf)
        vs = (torch.randn(B, 1, HV, V, generator=g, device=DEV) * 0.5).to(bf)
        as_ = (torch.randn(B, 1, HV, generator=g, device=DEV) * 0.5).to(bf)
        bs = (torch.randn(B, 1, HV, generator=g, device=DEV)).to(bf)
        y = mod.gated_delta_rule_stp_ucache_flush(
            A_log,
            as_,
            dt_bias,
            q=qs,
            k=ks,
            v=vs,
            b=bs,
            initial_state_source=pool,
            initial_state_indices=idx,
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=STP_FM,
        )
        for r in range(B):
            y_ref, S_new, _, _ = _stp_token_step(
                S_ref[r],
                qs[r, 0],
                ks[r, 0],
                vs[r, 0],
                as_[r, 0],
                bs[r, 0],
                A_log,
                dt_bias,
            )
            S_ref[r] = S_new
            max_drift = max(max_drift, (y[r, 0].float() - y_ref).abs().max().item())
    assert max_drift < 5e-2, f"max output drift over {n_steps} steps: {max_drift:.3e}"


def test_stp_prepadded_matches_staging_path():
    """The zero-copy prepadded [B,4,...] path (rows 1..3 zero) must be
    bitwise identical to the default [B,1,...] staging path."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    B = 4
    hist = [15, 14, 0, 7]
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, hist, "bf16", seed=31
    )

    def pad4(t):
        buf = torch.zeros(
            (t.shape[0], 4) + tuple(t.shape[2:]), dtype=t.dtype, device=DEV
        )
        buf[:, 0] = t[:, 0]
        return buf

    results = {}
    for mode in ("staged", "prepadded"):
        p2, kc2, uc2, gc2 = pool.clone(), kc.clone(), uc.clone(), gc.clone()
        hl2, cb2 = hl.clone(), cb.clone()
        if mode == "prepadded":
            kw = dict(q=pad4(q), k=pad4(k), v=pad4(v), prepadded=True)
            aa, bb = pad4(a), pad4(b)
        else:
            kw = dict(q=q, k=k, v=v)
            aa, bb = a, b
        y = mod.gated_delta_rule_stp_ucache_flush(
            A_log,
            aa,
            dt_bias,
            b=bb,
            **kw,
            initial_state_source=p2,
            initial_state_indices=idx,
            k_cache=kc2,
            u_cache=uc2,
            g_cache=gc2,
            hist_len=hl2,
            cache_base=cb2,
            scale=SCALE,
            flush_min=STP_FM,
        )
        results[mode] = (y.clone(), p2, kc2, uc2, gc2, hl2)

    names = ["y", "pool", "k_cache", "u_cache", "g_cache", "hist_len"]
    for name, st_, pp_ in zip(
        names, results["staged"], results["prepadded"], strict=True
    ):
        assert torch.equal(st_, pp_), f"prepadded vs staged mismatch on {name}"


def test_stp_contract_guards():
    """32-deep rings, hist_len = 16, and flush_min = 16 must be rejected
    loudly (the flat-16 buffer's bounds) rather than corrupting."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush("bf16", _STP_PATH)
    B = 2
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = _stp_case(
        B, [0, 0], "bf16", seed=9
    )

    def call(**over):
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
            flush_min=STP_FM,
        )
        kw.update(over)
        return mod.gated_delta_rule_stp_ucache_flush(A_log, a, dt_bias, **kw)

    with pytest.raises(AssertionError):  # 32-deep ring rejected
        call(k_cache=torch.zeros(B, H, 32, K, dtype=kc.dtype, device=DEV))
    with pytest.raises(AssertionError, match="flush_min"):
        call(flush_min=16)
    with pytest.raises(AssertionError, match="hist_len"):
        call(hist_len=torch.full((B,), 16, dtype=torch.int32, device=DEV))


def test_stp_lockstep_equivalence_vs_vllm_triton():
    """64 steps of the SAME token stream through this kernel and vLLM PR
    #48792's Triton ReplaySSM kernel at identical cadence (fm=15 <->
    is_flush at write_pos 15): outputs must agree every step and the
    committed pool states after every fold. Requires the reference kernel
    (benchmarks/gdn_vllm_replayssm_triton.py, vendored locally from that PR;
    not shipped) — SKIPS when absent."""
    _skip_if_not_sm90_or_later()
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    vllm_mod = pytest.importorskip(
        "gdn_vllm_replayssm_triton",
        reason="reference vLLM Triton ReplaySSM kernel not vendored locally",
    )
    vllm_fn = vllm_mod.fused_recurrent_gated_delta_rule_replayssm

    mod = _load_flush("bf16", _STP_PATH)
    B, n_steps = 4, 64
    bf = torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(404)
    A_log = (
        torch.full((HV,), -3.0, device=DEV)
        + torch.rand(HV, generator=g, device=DEV) * 0.3
    ).to(bf)
    dt_bias = (torch.randn(HV, generator=g, device=DEV) * 0.5).to(bf)
    pool_init = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(bf)

    # --- STP kernel state ---
    pool_s = pool_init.clone()
    kc_s = torch.zeros(B, H, STP_RING, K, dtype=bf, device=DEV)
    uc_s = torch.zeros(B, HV, STP_RING, V, dtype=bf, device=DEV)
    gc_s = torch.zeros(B, HV, STP_RING, dtype=torch.float32, device=DEV)
    hl = torch.zeros(B, dtype=torch.int32, device=DEV)
    cb = torch.zeros(B, dtype=torch.int32, device=DEV)
    idx_s = torch.arange(B, dtype=torch.int32, device=DEV)

    # --- vLLM kernel state (slot 0 = padding sentinel; per-step g cache) ---
    pool_v = torch.cat([torch.zeros_like(pool_init[:1]), pool_init]).contiguous()
    dc_v = torch.zeros(B + 1, HV, STP_RING, V, dtype=bf, device=DEV)
    kc_v = torch.zeros(B + 1, H, STP_RING, K, dtype=bf, device=DEV)
    gc_v = torch.zeros(B + 1, HV, STP_RING, dtype=torch.float32, device=DEV)
    idx_v = torch.arange(1, B + 1, dtype=torch.int32, device=DEV)
    out_v = torch.empty(B, HV * V, dtype=bf, device=DEV)

    for step in range(n_steps):
        qs = (torch.randn(B, 1, H, K, generator=g, device=DEV)).to(bf)
        ks = (torch.randn(B, 1, H, K, generator=g, device=DEV)).to(bf)
        vs = (torch.randn(B, 1, HV, V, generator=g, device=DEV) * 0.5).to(bf)
        as_ = (torch.randn(B, 1, HV, generator=g, device=DEV) * 0.5).to(bf)
        bs = (torch.randn(B, 1, HV, generator=g, device=DEV)).to(bf)

        wp = step % 16
        is_flush_step = wp == 15

        y_s = mod.gated_delta_rule_stp_ucache_flush(
            A_log,
            as_,
            dt_bias,
            q=qs,
            k=ks,
            v=vs,
            b=bs,
            initial_state_source=pool_s,
            initial_state_indices=idx_s,
            k_cache=kc_s,
            u_cache=uc_s,
            g_cache=gc_s,
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=STP_FM,
        )
        assert int(hl[0]) == (0 if is_flush_step else wp + 1)

        mixed = torch.cat(
            [qs.reshape(B, -1), ks.reshape(B, -1), vs.reshape(B, -1)], dim=1
        ).contiguous()
        wp_t = torch.full((B,), wp, dtype=torch.int32, device=DEV)
        fl_t = torch.full((B,), 1 if is_flush_step else 0, dtype=torch.int8, device=DEV)
        vllm_fn(
            mixed,
            as_[:, 0],
            bs[:, 0],
            A_log,
            dt_bias,
            SCALE,
            pool_v,
            dc_v,
            kc_v,
            gc_v,
            out_v,
            idx_v,
            wp_t,
            fl_t,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        d = (y_s[:, 0].float() - out_v.view(B, HV, V).float()).abs().max().item()
        assert d < 2e-2, f"step {step}: |y_stp - y_vllm| = {d:.3e}"
        if is_flush_step:
            dp = (pool_s.float() - pool_v[1:].float()).abs().max().item()
            assert dp < 4e-2, f"step {step}: |pool_stp - pool_vllm| = {dp:.3e}"
