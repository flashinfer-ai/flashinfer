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

from tests.test_helpers.test_helpers import skip_if_cute_dsl_arch_unsupported

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

    # The decode kernels are CuTe-DSL kernels; an older DSL raises a bare
    # KeyError (e.g. 'sm_107a' on CuTe DSL 4.7 / Rubin).  Environment gap, so
    # skip.  native_only=False: these kernels do not pin an arch and still run
    # against the family target.
    skip_if_cute_dsl_arch_unsupported(torch.device("cuda"), native_only=False)
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN ucache requires SM90+, got SM{cc[0]}{cc[1]}")


def _load_flush(arm: str):
    """Load one module copy per dtype arm (dtype is chosen at import time)."""
    if arm in _MODULE_CACHE:
        return _MODULE_CACHE[arm]
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
        spec = importlib.util.spec_from_file_location(f"uc_flush_{arm}", _FLUSH_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _MODULE_CACHE[arm] = mod
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
    y = torch.zeros(T, HV, V, dtype=f, device=q.device)
    for t in range(T):
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
def _make_case(B, hist_lens, io_dtype, state_dtype, seed, ring_dtype=None, bases=None):
    ring_dtype = ring_dtype or io_dtype
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io_dtype)

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, a, b = rn(B, T, HV, V, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (
        torch.full((HV,), -3.0, device=DEV)
        + torch.rand(HV, generator=g, device=DEV) * 0.3
    ).to(io_dtype)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(state_dtype)
    kc = torch.zeros(B, H, RING, K, dtype=ring_dtype, device=DEV)
    uc = torch.zeros(B, HV, RING, V, dtype=ring_dtype, device=DEV)
    gc = torch.zeros(B, HV, RING, dtype=torch.float32, device=DEV)
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
