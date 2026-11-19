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
The KDA ucache verify+flush tests with an fp16 state pool on ``backend="umma"``,
plus stochastic rounding (``stochastic_rounding="philox" / "lcg"``) of the flushed
fp16 state.

Reuse: the star import re-collects every test of test_kda_ucache_flush_umma.py (which
itself re-collects test_kda_ucache_flush.py). The autouse fixture pins
``_base._BACKEND_KW`` to UMMA, as that module's fixture does, and patches
``_base._STATE_DTYPE`` to fp16, so every case ``_make_case`` (and the two tests that
build their own pools) creates carries an fp16 pool. The oracle reads the pool as fp32,
so the same fp32 reference and the same tolerances apply: fp16 holds the pool values
(|S| < 2^15, here O(1)) with 3 more mantissa bits than bf16, and the readout operand
X_r = [khat; qhat] . e^{G_P} (|x| <= 1) goes subnormal (and below 2^-24 to
zero) only when every channel has decayed; with O(1) states that is far below Y_TOL, and
``test_fp16_large_state_strong_decay`` bounds the large-|S| case (|S| ~ 3000, G_P ~ -11).

Tests that need the HMMA backend on the same case (it keeps a bf16 pool) are skipped
here; ``test_fp16_hmma_rejects`` checks that HMMA refuses an fp16 pool instead.

Run:
  pytest tests/experimental/test_kda_ucache_flush_umma_fp16.py -v
"""

from __future__ import annotations

import pytest
import torch

from . import test_kda_ucache_flush as _base
from .test_kda_ucache_flush_umma import *  # noqa: F401,F403  re-collects all UMMA cases
from .test_kda_ucache_flush import DEV, H, K, LOWER_BOUND, SCALE, V, _make_case
from .test_kda_ucache_flush_umma import _random_case

# Re-collected tests that compare against (or call) the HMMA backend on the same case:
# HMMA has no fp16 pool, so these cannot run with one (see test_fp16_hmma_rejects).
_HMMA_ON_CASE = {
    "test_umma_parity_with_hmma",
    "test_umma_backend_dispatch",
    "test_umma_only_t_dispatch",
    "test_restart_commit_skips_padded_rows",
}


@pytest.fixture(autouse=True)
def _umma_fp16(request, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("KDA UMMA backend needs SM10x (tcgen05)")
    if request.node.originalname in _HMMA_ON_CASE:
        pytest.skip("compares with the HMMA backend, which has no fp16 state pool")
    monkeypatch.setattr(_base, "_BACKEND_KW", {"backend": "umma"})
    monkeypatch.setattr(_base, "_STATE_DTYPE", torch.float16)


def _skip_if_no_cvt_rs():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("stochastic rounding needs cvt.rs (sm_100a / sm_103a / sm_107a)")


# --------------------------------------------------------------------------- helpers
def _fm(t, w_ring=16):
    fm = min(w_ring - t + 1, 33 - 2 * t)
    return max(fm, 17) if w_ring == 32 else fm


def _sr_case(t, seed, w_ring=16):
    """6 rows: 4 fold (one at the window cap, wrapped bases), one verify, one empty."""
    fm = _fm(t, w_ring)
    cap = min(w_ring, 32 - t)
    hist = [fm, fm - 1, cap, fm, 0, fm]
    bases = [28, 5, 30, 17, 9, 0]
    return _make_case(len(hist), hist, seed, bases=bases, t=t), hist, fm


def _sr_call(case, fm, t, w_ring=16, **kw):
    """One UMMA call on clones of ``case``: (y, pool, (k, u, g rings))."""
    from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush as f

    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    y = f(
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
        flush_min=fm,
        restart_hist_on_flush=False,
        w_ring=w_ring,
        backend=kw.pop("backend", "umma"),
        **kw,
    )
    torch.cuda.synchronize()
    return y, pool, (kc, uc, gc)


_PHILOX = (0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85)  # round A, B, key A, B


def _philox_noise13(seed, sidx, rounds):
    """13-bit cvt.rs noise of every element of pool slot ``sidx`` ([H, V, K]) under the
    kernel's "philox" scheme (FlashInfer Mamba's): Philox4x32, key = seed, counter = flat
    pool offset of the element's 8-element chunk; word i -> pair i, the even element
    rbits[12:0], the odd one rbits[28:16]. numpy reference."""
    import numpy as np

    u64, m32 = np.uint64, np.uint64(0xFFFFFFFF)
    ra, rb, ka, kb = (u64(x) for x in _PHILOX)
    chunk = np.arange(H * V * (K // 8), dtype=np.uint64) * u64(8) + u64(
        sidx * H * V * K
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
    return torch.from_numpy(pairs.reshape(H, V, K).astype(np.int32))


def _lcg_noise13(seed, sidx):
    """13-bit cvt.rs noise of pool slot ``sidx`` ([H, V, K]) under the kernel's "lcg" mode:
    the (state row, K half) start = murmur3(id_lo * golden + id_hi * c + key), key =
    murmur3 of the seed; pair n of the half (n = 4 chunk + i) takes LCG state n folded as
    s ^ (s >> 16). numpy reference of _rmw (fmix32 / LCG_* of the GDN UMMA helpers)."""
    import numpy as np

    u64, m = np.uint64, np.uint64(0xFFFFFFFF)

    def fmix(x):
        x = x ^ (x >> u64(16))
        x = (x * u64(0x85EBCA6B)) & m
        x = x ^ (x >> u64(13))
        x = (x * u64(0xC2B2AE35)) & m
        return x ^ (x >> u64(16))

    key = fmix(
        u64(seed & 0xFFFFFFFF) ^ fmix(u64((seed >> 32) & 0xFFFFFFFF) ^ u64(0x9E3779B9))
    )
    hh, vv, gg = np.meshgrid(
        np.arange(H, dtype=np.uint64),
        np.arange(V, dtype=np.uint64),
        np.arange(2, dtype=np.uint64),
        indexing="ij",
    )
    cid = ((u64(sidx * H) + hh) * u64(V) + vv) * u64(2) + gg
    x = fmix(
        ((cid & m) * u64(0x9E3779B9) + (cid >> u64(32)) * u64(0x85EBCA6B) + key) & m
    )
    words = []
    for _ in range(32):  # pair n = 4 chunk + i of the K half
        words.append(x ^ (x >> u64(16)))
        x = (x * u64(1664525) + u64(1013904223)) & m
    w = np.stack(words, axis=-1)  # [H, V, 2, 32]
    pairs = np.stack([w & u64(0x1FFF), (w >> u64(16)) & u64(0x1FFF)], axis=-1)
    return torch.from_numpy(pairs.reshape(H, V, K).astype(np.int32))


def _seed(x):
    return torch.tensor([x], dtype=torch.int64, device=DEV)


# --------------------------------------------------------------------------- fp16 pool
def test_fp16_hmma_rejects():
    """An fp16 pool and stochastic rounding are UMMA-only: "hmma" raises ValueError,
    "auto" routes both to UMMA (bitwise equal to "umma"), SR on a bf16 pool raises."""
    case, hist, fm = _sr_case(4, seed=3)
    with pytest.raises(ValueError, match="UMMA backend only"):
        _sr_call(case, fm, 4, backend="hmma")
    with pytest.raises(ValueError, match="UMMA backend only"):
        _sr_call(
            case,
            fm,
            4,
            backend="hmma",
            stochastic_rounding="philox",
            rand_seed=_seed(1),
        )
    a = _sr_call(case, fm, 4, backend="auto")
    b = _sr_call(case, fm, 4, backend="umma")
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    bf = list(case)
    bf[7] = bf[7].to(torch.bfloat16)
    with pytest.raises(ValueError, match="fp16 state pool"):
        _sr_call(bf, fm, 4, stochastic_rounding="lcg", rand_seed=_seed(1))
    f32 = list(case)
    f32[7] = f32[7].float()
    with pytest.raises(ValueError, match="initial_state_source"):
        _sr_call(f32, fm, 4)


@pytest.mark.parametrize("t", [1, 4, 8])
@pytest.mark.parametrize("w_ring", [16, 32])
def test_fp16_pool_no_worse_than_bf16(t, w_ring):
    """Same case with a bf16 and with an fp16 pool (the fp16 pool holds the bf16 values
    exactly): outputs agree, and the fp16 fold is at least as close to the fp32 oracle
    as the bf16 one (3 more mantissa bits in the stored state)."""
    fm = _fm(t, w_ring)
    hist = [fm, fm - 1, min(w_ring, 32 - t), 0, fm, fm]
    bases = [28, 5, 30, 9, 0, 17]
    _base._STATE_DTYPE = torch.bfloat16
    c_bf = _make_case(len(hist), hist, seed=71 + t, bases=bases, t=t)
    _base._STATE_DTYPE = torch.float16
    c_h = list(c_bf)
    c_h[7] = c_bf[7].to(torch.float16)
    y_bf, s_bf, _ = _sr_call(c_bf, fm, t, w_ring)
    y_h, s_h, _ = _sr_call(c_h, fm, t, w_ring)
    assert (y_bf.float() - y_h.float()).abs().max().item() < 2e-3
    err_bf = err_h = 0.0
    for r, p in enumerate(hist):
        if p < fm:
            assert torch.equal(s_h[r], c_h[7][r])
            continue
        kc_l, uc_l, gc_l = _base._logical_rings(
            c_bf[8][r], c_bf[9][r], c_bf[10][r], bases[r], w=w_ring
        )
        _, S_ref = _base._ref_fp32(
            *(c_bf[i][r] for i in range(5)),
            c_bf[5],
            c_bf[6],
            c_bf[7][r],
            kc_l,
            uc_l,
            gc_l,
            p,
            t_len=t,
        )
        err_bf = max(err_bf, (s_bf[r].float() - S_ref).abs().max().item())
        err_h = max(err_h, (s_h[r].float() - S_ref).abs().max().item())
    assert err_h <= err_bf, (err_h, err_bf)


# --------------------------------------------------------------------------- SR
@pytest.mark.parametrize("t", [1, 4, 8])
@pytest.mark.parametrize("mode", ["philox10", "philox5", "lcg"])
@pytest.mark.parametrize("path", ["one_item", "persistent"])
def test_sr_matches_reference_bits(t, mode, path, monkeypatch):
    """SR only changes how the flushed fp16 state is rounded: outputs and rings are
    bit-identical to round-to-nearest, verify rows keep their state, and every flushed
    (normal fp16) element is the RN result or its neighbour, with the rounding direction
    the reference noise bits allow. cvt.rs adds the 13 noise bits r to the 13 dropped
    mantissa bits d and truncates: up iff d + r >= 8192. RN went down (d < 4096, or the
    tie d = 4096 broken to even) -> SR up needs r >= 8192 - d, i.e. r >= 4096 (= 4096
    only at the tie); RN went up (d > 4096, or the tie) -> SR down needs r < 8192 - d,
    i.e. r < 4096. Noise from another seed must violate these (the check can fail)."""
    _skip_if_no_cvt_rs()
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    monkeypatch.setattr(um, "_FORCE_PERSISTENT", path == "persistent")
    case, hist, fm = _sr_case(t, seed=70 + t)
    seed = 0x1234_5678_9ABC_DEF1 + t
    if mode == "lcg":
        kw = dict(stochastic_rounding="lcg")
        noise_of = _lcg_noise13
    else:
        rounds = int(mode[len("philox") :])
        kw = dict(stochastic_rounding="philox", philox_rounds=rounds)

        def noise_of(sd, sidx):
            return _philox_noise13(sd, sidx, rounds)

    pool0 = case[7].clone()
    y_rn, s_rn, rings_rn = _sr_call(case, fm, t)
    y_sr, s_sr, rings_sr = _sr_call(case, fm, t, rand_seed=_seed(seed), **kw)
    assert torch.equal(y_rn, y_sr)
    for x, z in zip(rings_rn, rings_sr, strict=True):
        assert torch.equal(x, z)
    n_diff = n_bad = n_bad_ctl = total = 0
    for r, p in enumerate(hist):
        if p < fm:
            assert torch.equal(s_sr[r], pool0[r]) and torch.equal(s_rn[r], pool0[r])
            continue
        rn = s_rn[r].view(torch.int16).int().cpu()
        sr = s_sr[r].view(torch.int16).int().cpu()
        mag_rn, mag_sr = rn & 0x7FFF, sr & 0x7FFF
        normal = (mag_rn >= 0x0400) & (mag_rn < 0x7C00)  # 13-bit noise model
        assert torch.equal((rn >> 15)[normal], (sr >> 15)[normal]), "sign flipped"
        step = mag_sr - mag_rn
        assert step.abs().max().item() <= 1, "SR result is not RN or its neighbour"
        noise = noise_of(seed, r)
        ctl = noise_of(seed + 1, r)
        up, down = normal & (step == 1), normal & (step == -1)
        n_diff += int(up.sum() + down.sum())
        n_bad += int(((noise < 4096) & up).sum() + ((noise >= 4096) & down).sum())
        n_bad_ctl += int(((ctl < 4096) & up).sum() + ((ctl >= 4096) & down).sum())
        total += int(normal.sum())
    # E[min(f, 1 - f)] = 1/4 of the elements round the other way
    assert abs(n_diff / total - 0.25) < 0.05, n_diff / total
    assert n_bad == 0, f"{n_bad} of {n_diff} directions contradict the {mode} bits"
    assert n_bad_ctl > 0.3 * n_diff, (n_bad_ctl, n_diff)


@pytest.mark.parametrize("mode", ["philox", "lcg"])
def test_sr_unbiased(mode):
    """The mean over 64 seeds of the stochastically rounded fold converges to the exact
    value: its mean signed error is ~0 and it is much closer than round-to-nearest. The
    u ring is zeroed, so the fold is S <- e^{G_P} . S0 with F = 0 and the kernel's
    pre-rounding fp32 value equals the fp32 oracle to ~1e-7 relative (ex2.approx vs
    torch.exp): what remains is the fp16 rounding under test (the full fold also carries
    the bf16 rounding of K~ = w . k̂, as large as an fp16 ulp)."""
    _skip_if_no_cvt_rs()
    t = 4
    case, hist, fm = _sr_case(t, seed=901)
    case = list(case)
    case[9] = torch.zeros_like(case[9])
    _, s_rn, _ = _sr_call(case, fm, t)
    acc = torch.zeros_like(s_rn, dtype=torch.float32)
    n = 64
    for i in range(n):
        _, s, _ = _sr_call(
            case, fm, t, stochastic_rounding=mode, rand_seed=_seed(1000 + 7919 * i)
        )
        acc += s.float()
    mean = acc / n
    e_rn = e_sr = bias = 0.0
    cnt = 0
    for r, p in enumerate(hist):
        if p < fm:
            continue
        kc_l, uc_l, gc_l = _base._logical_rings(
            case[8][r], case[9][r], case[10][r], [28, 5, 30, 17, 9, 0][r]
        )
        _, S_ref = _base._ref_fp32(
            *(case[i][r] for i in range(5)),
            case[5],
            case[6],
            case[7][r],
            kc_l,
            uc_l,
            gc_l,
            p,
            t_len=t,
        )
        e_rn += (s_rn[r].float() - S_ref).abs().sum().item()
        e_sr += (mean[r] - S_ref).abs().sum().item()
        bias += (mean[r] - S_ref).sum().item()
        cnt += S_ref.numel()
    # RN: E|err| = ulp / 4; the 64-sample mean: ~ ulp / (8 sqrt(2 pi)) -> about 0.2 x RN
    assert e_sr < 0.35 * e_rn, (e_sr / cnt, e_rn / cnt)
    assert abs(bias) < 0.02 * e_rn, (bias / cnt, e_rn / cnt)


def test_sr_seed_and_graph():
    """Same seed -> bit-identical state; another seed -> different; the seed is read on
    device, so an in-place update reaches a captured CUDA graph; a seed alone (no mode)
    is RN."""
    _skip_if_no_cvt_rs()
    from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush as f

    t = 4
    case, hist, fm = _sr_case(t, seed=91)
    seed_t = _seed(7)
    ph = dict(stochastic_rounding="philox")
    _, a1, _ = _sr_call(case, fm, t, rand_seed=seed_t, **ph)
    _, a2, _ = _sr_call(case, fm, t, rand_seed=seed_t, **ph)
    _, b1, _ = _sr_call(case, fm, t, rand_seed=_seed(8), **ph)
    _, rn, _ = _sr_call(case, fm, t)
    _, rn_seed, _ = _sr_call(case, fm, t, rand_seed=seed_t)
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b1) and not torch.equal(a1, rn)
    assert torch.equal(rn_seed, rn)

    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    pool_init = pool.clone()
    kw = dict(
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
        flush_min=fm,
        restart_hist_on_flush=False,
        backend="umma",
        stochastic_rounding="philox",
        rand_seed=seed_t,
    )
    f(A_log, g, dt_bias, **kw)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        f(A_log, g, dt_bias, **kw)
    for want_seed, want in ((7, a1), (8, b1)):
        pool.copy_(pool_init)
        seed_t.fill_(want_seed)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool, want), f"graph replay with seed {want_seed}"


def test_lcg_unseeded_uses_clock():
    """ "lcg" without a seed is %clock-seeded: two runs differ, each RN or its neighbour
    with ~1/4 of the elements rounded the other way; "none" (even with a seed) is RN."""
    _skip_if_no_cvt_rs()
    t = 4
    case, hist, fm = _sr_case(t, seed=12)
    _, rn, _ = _sr_call(case, fm, t)
    _, rn2, _ = _sr_call(case, fm, t, stochastic_rounding="none", rand_seed=_seed(3))
    _, c1, _ = _sr_call(case, fm, t, stochastic_rounding="lcg")
    _, c2, _ = _sr_call(case, fm, t, stochastic_rounding="lcg")
    assert torch.equal(rn, rn2)
    assert not torch.equal(c1, c2)
    rows = [r for r, p in enumerate(hist) if p >= fm]
    for st in (c1, c2):
        step = (st[rows].view(torch.int16).int() & 0x7FFF) - (
            rn[rows].view(torch.int16).int() & 0x7FFF
        )
        assert step.abs().max().item() <= 1
        assert abs((step != 0).float().mean().item() - 0.25) < 0.05


@pytest.mark.parametrize("t", [1, 4, 8])
@pytest.mark.parametrize("mode", ["none", "philox", "lcg"])
@pytest.mark.parametrize("w_ring", [16, 32])
def test_sr_persistent_matches_single_item_bitwise(t, mode, w_ring, monkeypatch):
    """fp16 pool, RN and seeded SR: the persistent kernel is bitwise equal to one item per
    CTA (same counters / LCG ids per (slot, head, row, half)), with padded rows and more
    items than CTAs."""
    if mode != "none":
        _skip_if_no_cvt_rs()
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    res = {}
    for pz in (False, True):
        monkeypatch.setattr(um, "_FORCE_PERSISTENT", pz)
        case, _, _ = _random_case(700, t, w_ring, "fold", seed=5151 + 10 * t + w_ring)
        case = list(case)
        idx = case[13].clone()
        idx[::37] = -1
        case[13] = idx
        kw = {} if mode == "none" else dict(stochastic_rounding=mode)
        y, pool, rings = _sr_call(
            case, _fm(t, w_ring), t, w_ring, rand_seed=_seed(99 + t), **kw
        )
        live = idx >= 0
        res[pz] = [y[live], pool, *rings]
    for name, a, b in zip(
        ("output", "pool", "k_cache", "u_cache", "g_cache"),
        res[False],
        res[True],
        strict=True,
    ):
        assert torch.equal(a, b), f"persistent vs single-item differ on {name}"


def test_sr_rejects_bad_arguments():
    _skip_if_no_cvt_rs()
    case, _, fm = _sr_case(4, seed=5)
    for bad in (
        torch.tensor([1, 2], dtype=torch.int64, device=DEV),
        torch.tensor([1], dtype=torch.int32, device=DEV),
        torch.tensor([1], dtype=torch.int64),
    ):
        with pytest.raises(ValueError, match="rand_seed"):
            _sr_call(case, fm, 4, stochastic_rounding="philox", rand_seed=bad)
    with pytest.raises(ValueError, match="philox_rounds"):
        _sr_call(
            case,
            fm,
            4,
            stochastic_rounding="philox",
            rand_seed=_seed(1),
            philox_rounds=0,
        )
    with pytest.raises(ValueError, match="needs rand_seed"):
        _sr_call(case, fm, 4, stochastic_rounding="philox")
    with pytest.raises(ValueError, match="stochastic_rounding"):
        _sr_call(case, fm, 4, stochastic_rounding="pcg")


def test_fp16_pool_through_auto_dispatch_at_hmma_t():
    """T in {4, 8} with an fp16 pool on backend="auto": served by UMMA (the launch counter
    moves), never by HMMA."""
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    case, _, fm = _sr_case(8, seed=17)
    n0 = um._LAUNCHES
    _sr_call(case, fm, 8, backend="auto")
    assert n0 + 1 == um._LAUNCHES


@pytest.mark.parametrize("t", [1, 4, 8])
@pytest.mark.parametrize("w_ring", [16, 32])
def test_fp16_large_state_strong_decay(t, w_ring):
    """Large state (|S| ~ 3000) under strong decay (every channel's G_P ~ -11): the
    readout's S0 term rides on X_r = [khat; qhat] e^{G_P} ~ 2^-16, which fp16 stores as
    a subnormal (the documented limit of the fp16 pool: the operand is not rescaled).
    Bound: the fp16 pool's output error exceeds the bf16 pool's (same values) by at most
    2^-25 sqrt(K) max|S0| (half an ulp of the smallest fp16 subnormal per X_r entry,
    summed over K channels as a random walk). Measured: 6e-4 vs 1e-4, bound 2.5e-3."""
    fm = _fm(t, w_ring)
    hist = [fm - 1, fm - 2, 3, 1, fm - 1, 2]  # verify rows (the readout path)
    bases = [28, 5, 30, 9, 0, 17]
    _base._STATE_DTYPE = torch.bfloat16
    c_bf = list(_make_case(len(hist), hist, seed=91 + t, bases=bases, t=t))
    _base._STATE_DTYPE = torch.float16
    c_bf[7] = (c_bf[7].float() * 3000.0).to(torch.bfloat16)  # exact in fp16 too
    c_bf[10] = c_bf[10] - 11.0  # shift every G: same w_j, e^{G_P} ~ 1.7e-5
    c_h = list(c_bf)
    c_h[7] = c_bf[7].to(torch.float16)
    assert torch.equal(c_h[7].float(), c_bf[7].float())
    y_bf, _, _ = _sr_call(c_bf, fm, t, w_ring)
    y_h, _, _ = _sr_call(c_h, fm, t, w_ring)
    err_bf = err_h = 0.0
    for r, p in enumerate(hist):
        kc_l, uc_l, gc_l = _base._logical_rings(
            c_bf[8][r], c_bf[9][r], c_bf[10][r], bases[r], w=w_ring
        )
        y_ref, _ = _base._ref_fp32(
            *(c_bf[i][r] for i in range(5)),
            c_bf[5],
            c_bf[6],
            c_bf[7][r],
            kc_l,
            uc_l,
            gc_l,
            p,
            t_len=t,
        )
        err_bf = max(err_bf, (y_bf[r].float() - y_ref).abs().max().item())
        err_h = max(err_h, (y_h[r].float() - y_ref).abs().max().item())
    bound = 2.0**-25 * K**0.5 * c_bf[7].float().abs().max().item()
    assert err_h <= err_bf + bound, (err_h, err_bf, bound)


def test_philox_rounds_below_two_rejected():
    """The shared Philox helper always runs its first two rounds, so philox_rounds = 1
    would silently run 2: the dispatcher rejects it."""
    _skip_if_no_cvt_rs()
    case, _, fm = _sr_case(4, seed=5)
    for bad in (1, 0, -3):
        with pytest.raises(ValueError, match="philox_rounds must be >= 2"):
            _sr_call(
                case,
                fm,
                4,
                stochastic_rounding="philox",
                rand_seed=_seed(1),
                philox_rounds=bad,
            )
