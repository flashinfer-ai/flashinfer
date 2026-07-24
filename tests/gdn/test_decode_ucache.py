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
(flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py).

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
T, W = 4, 16
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
    "fp16_state_cache": (None, "fp16", "fp16",
                         torch.bfloat16, torch.float16, torch.float16),
}


def _skip_if_not_sm90_or_later():
    from flashinfer.utils import get_compute_capability

    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [9, 10, 11, 12]:
        pytest.skip(f"GDN ucache requires SM90+, got SM{cc[0]}{cc[1]}")


def _load_flush(arm: str):
    """Load one module copy per dtype arm (dtype is chosen at import time)."""
    if arm in _MODULE_CACHE:
        return _MODULE_CACHE[arm]
    io_env, state_env, ring_env, _, _, _ = ARMS[arm]
    old = {k: os.environ.pop(k, None)
           for k in ("GDN_UCACHE_IO_DTYPE", "GDN_UCACHE_STATE_DTYPE",
                     "GDN_UCACHE_RING_DTYPE")}
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
        GP = gc[:, P - 1].to(f)                                   # [HV]
        w = torch.exp(GP[:, None] - gc[:, :P].to(f))              # [HV, P]
        kc_hv = kc[:, :P].to(f).repeat_interleave(grp, dim=0)     # [HV, P, K]
        S = torch.exp(GP)[:, None, None] * S + torch.einsum(
            "hpv,hpk->hvk", w[:, :, None] * uc[:, :P].to(f), kc_hv)
    S_after_history = S.clone()   # == the state a fold commits to the pool

    # 2) run the T new draft tokens through the exact delta-rule recurrence
    khat = F.normalize(k.to(f), dim=-1)
    qhat = F.normalize(q.to(f), dim=-1) * SCALE
    y = torch.zeros(T, HV, V, dtype=f, device=q.device)
    for t in range(T):
        la = -torch.exp(A_log.to(f)) * F.softplus(a[t].to(f) + dt_bias.to(f))
        beta = torch.sigmoid(b[t].to(f))                          # [HV]
        k_hv = khat[t].repeat_interleave(grp, dim=0)              # [HV, K]
        q_hv = qhat[t].repeat_interleave(grp, dim=0)
        S = S * torch.exp(la)[:, None, None]                      # decay
        pred = torch.einsum("hvk,hk->hv", S, k_hv)                # S k
        u_t = (v[t].to(f) - pred) * beta[:, None]                 # delta rule
        S = S + u_t[:, :, None] * k_hv[:, None, :]                # + u (x) k
        y[t] = torch.einsum("hvk,hk->hv", S, q_hv)                # out = S q
    return y, S_after_history


# ---------------------------------------------------------------------------
# Case builder: consistent inputs + rings for B requests.
# ---------------------------------------------------------------------------
def _make_case(B, hist_lens, io_dtype, state_dtype, seed, ring_dtype=None):
    ring_dtype = ring_dtype or io_dtype
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io_dtype)

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, a, b = rn(B, T, HV, V, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (torch.full((HV,), -3.0, device=DEV)
             + torch.rand(HV, generator=g, device=DEV) * 0.3).to(io_dtype)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(state_dtype)
    kc = torch.zeros(B, H, W, K, dtype=ring_dtype, device=DEV)
    uc = torch.zeros(B, HV, W, V, dtype=ring_dtype, device=DEV)
    gc = torch.zeros(B, HV, W, dtype=torch.float32, device=DEV)
    hl = torch.tensor(hist_lens, dtype=torch.int32, device=DEV)
    for r in range(B):
        P = int(hl[r])
        if P == 0:
            continue
        kh = torch.randn(H, P, K, generator=g, device=DEV)
        kc[r, :, :P] = F.normalize(kh, dim=-1).to(ring_dtype)
        uc[r, :, :P] = (torch.randn(HV, P, V, generator=g, device=DEV) * 0.3).to(ring_dtype)
        la = -(torch.rand(HV, P, generator=g, device=DEV) * 0.3 + 0.003)
        gc[r, :, :P] = torch.cumsum(la, dim=-1)
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, idx


HISTORIES = {
    "empty_P0": [0, 0, 0, 0],
    "replay_P12": [12, 12, 12, 12],
    "fold_mixed": [13, 12, 13, 12],  # rows 0 and 2 fold
}
Y_TOL = 8e-3      # observed max ~7e-4 across arms; 10x margin
STATE_TOL = 2e-2  # committed state accumulates P outer products first


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("history", list(HISTORIES))
def test_output_matches_fp32_reference(arm, history):
    _skip_if_not_sm90_or_later()
    mod = _load_flush(arm)
    _, _, _, io_dtype, state_dtype, ring_dtype = ARMS[arm]
    B = 4
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, idx = _make_case(
        B, HISTORIES[history], io_dtype, state_dtype, seed=1234,
        ring_dtype=ring_dtype)
    pool_before = pool.clone()  # fold rows mutate the pool; ref needs the old state

    y = mod.gated_delta_rule_mtp_ucache_flush(
        A_log, a, dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=pool, initial_state_indices=idx,
        k_cache=kc.clone(), u_cache=uc.clone(), g_cache=gc.clone(),
        hist_len=hl.clone(),  # kernel mutates rings (appends/restarts); ref reads originals
        scale=SCALE, flush_min=FLUSH_MIN)

    for r in range(B):
        y_ref, _ = _ref_fp32(q[r], k[r], v[r], a[r], b[r], A_log, dt_bias,
                             pool_before[r], kc[r], uc[r], gc[r], int(hl[r]))
        err = (y[r].float() - y_ref).abs().max().item()
        assert err < Y_TOL, f"row {r} ({history}, {arm}): |y - fp32 ref| = {err:.2e}"


@pytest.mark.parametrize("arm", list(ARMS))
def test_folded_state_matches_fp32_reference(arm):
    """On a fold (hist_len >= flush_min) the kernel writes the ring-folded
    checkpoint back to the pool. Compare that committed state to the oracle."""
    _skip_if_not_sm90_or_later()
    mod = _load_flush(arm)
    _, _, _, io_dtype, state_dtype, ring_dtype = ARMS[arm]
    B = 4
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, idx = _make_case(
        B, [13, 13, 13, 13], io_dtype, state_dtype, seed=99,
        ring_dtype=ring_dtype)
    pool_before = pool.clone()

    mod.gated_delta_rule_mtp_ucache_flush(
        A_log, a, dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=pool, initial_state_indices=idx,
        k_cache=kc.clone(), u_cache=uc.clone(), g_cache=gc.clone(),
        hist_len=hl.clone(),  # kernel mutates rings (appends/restarts); ref reads originals
        scale=SCALE, flush_min=FLUSH_MIN)

    for r in range(B):
        _, S_ref = _ref_fp32(q[r], k[r], v[r], a[r], b[r], A_log, dt_bias,
                             pool_before[r], kc[r], uc[r], gc[r], 13)
        err = (pool[r].float() - S_ref).abs().max().item()
        assert err < STATE_TOL, f"row {r} ({arm}): |committed - fp32 ref| = {err:.2e}"


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
            q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, idx = _make_case(
                B, [13] * B, io_dtype, state_dtype, seed=seed,
                ring_dtype=ring_dtype)
            pool_before = pool.clone()
            mod.gated_delta_rule_mtp_ucache_flush(
                A_log, a, dt_bias, q=q, k=k, v=v, b=b,
                initial_state_source=pool, initial_state_indices=idx,
                k_cache=kc.clone(), u_cache=uc.clone(), g_cache=gc.clone(),
        hist_len=hl.clone(),  # kernel mutates rings (appends/restarts); ref reads originals
                scale=SCALE, flush_min=FLUSH_MIN)
            for r in range(B):
                _, S_ref = _ref_fp32(q[r], k[r], v[r], a[r], b[r], A_log,
                                     dt_bias, pool_before[r], kc[r], uc[r],
                                     gc[r], 13)
                tot += (pool[r].float() - S_ref).abs().mean().item()
        errs[arm] = tot
    assert errs["fp16_state"] < errs["bf16"], (
        f"fp16 state should commit closer to fp32 truth: {errs}")


def _load_flush_strided():
    """Load a flush-module copy with the strided-QKV path enabled
    (SGLANG_GDN_WY_STRIDED_QKV read at import). This is the path vLLM uses:
    q/k/v (and a/b) arrive as chunk views of packed tensors."""
    old = os.environ.get("SGLANG_GDN_WY_STRIDED_QKV")
    os.environ["SGLANG_GDN_WY_STRIDED_QKV"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("uc_flush_strided",
                                                      _FLUSH_PATH)
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
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(
        torch.bfloat16)

    def run(q_, k_, v_, a_, b_):
        idx = torch.arange(B, dtype=torch.int32, device=DEV)
        return mod.gated_delta_rule_mtp_ucache_flush(
            A_log, a_, dt_bias, q=q_, k=k_, v=v_, b=b_,
            initial_state_source=pool.clone(), initial_state_indices=idx,
            k_cache=torch.zeros(B, H, W, K, dtype=ring, device=DEV),
            u_cache=torch.zeros(B, HV, W, V, dtype=ring, device=DEV),
            g_cache=torch.zeros(B, HV, W, dtype=torch.float32, device=DEV),
            hist_len=torch.zeros(B, dtype=torch.int32, device=DEV),
            scale=SCALE, flush_min=FLUSH_MIN).float()

    # pack q/k/v into one wide tensor -> matching token stride (strided-QKV)
    qw = H * K + H * K + HV * V
    wqkv = torch.zeros(B, T, qw, dtype=io, device=DEV)
    wqkv[:, :, :H * K] = q.reshape(B, T, H * K)
    wqkv[:, :, H * K:2 * H * K] = k.reshape(B, T, H * K)
    wqkv[:, :, 2 * H * K:] = v.reshape(B, T, HV * V)
    q_s = wqkv[:, :, :H * K].reshape(B, T, H, K)
    k_s = wqkv[:, :, H * K:2 * H * K].reshape(B, T, H, K)
    v_s = wqkv[:, :, 2 * H * K:].reshape(B, T, HV, V)
    wab = torch.zeros(B, T, 2 * HV, dtype=io, device=DEV)
    wab[:, :, :HV], wab[:, :, HV:] = a, b
    a_s, b_s = wab[:, :, :HV], wab[:, :, HV:]
    assert a_s.stride(1) == 2 * HV and q_s.stride(1) == qw  # confirm strided

    y_compact = run(q, k, v, a, b)
    y_strided = run(q_s, k_s, v_s, a_s, b_s)
    err = (y_compact - y_strided).abs().max().item()
    assert err < 2e-3, (
        f"strided (chunk-view) a/b must match compact; got max|d|={err:.3e} "
        "(the sb_t=HV bug reads packed b from the wrong rows)")
