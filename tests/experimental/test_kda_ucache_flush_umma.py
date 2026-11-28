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
The KDA ucache verify+flush tests, re-run on ``backend="umma"`` (the tcgen05 /
TMEM kernel in flashinfer/experimental/kda_ucache_flush/kda_replay_mtp_umma.py),
plus UMMA-only tests.

How the reuse works: the star import below re-collects every test of
test_kda_ucache_flush.py in this module. Those functions keep the base
module's globals, and every one of them reaches the kernel through
``_base._load_flush()``, which binds ``**_base._BACKEND_KW``. The autouse
fixture patches that dict to ``{"backend": "umma"}`` for every test collected
here, so the base file's oracle, cases and tolerances apply unchanged.

The base tests cover T in {4, 8} (all the HMMA kernel builds). The UMMA kernel
also runs T = 1 (STP) and T = 2, 3, 5, 6, 7: the ``test_umma_only_t_*`` tests
below re-run the base test bodies at those T (same oracle, cases, tolerances).

Run:
  pytest tests/experimental/test_kda_ucache_flush_umma.py -v
"""

from __future__ import annotations

import pytest
import torch

from . import test_kda_ucache_flush as _base
from .test_kda_ucache_flush import *  # noqa: F401,F403  re-collects all base cases
from .test_kda_ucache_flush import BASES, _load_flush, _make_case

# Draft lengths only the UMMA backend builds (the HMMA kernel: T in {4, 8}).
_UMMA_ONLY_T = [1, 2, 3, 5, 6, 7]
_ALL_T = list(range(1, 9))

# Base tests the UMMA backend cannot run yet at the current stage. Empty since
# stage S3c: the kernel covers T in {4, 8} x w_ring in {16, 32}.
_PENDING_NAMES: set = set()


def _PENDING_PARAMS(params: dict) -> bool:
    """Per-parameter pending filter (nothing pending since S3c)."""
    return False


@pytest.fixture(autouse=True)
def _umma_backend(request, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("KDA UMMA backend needs SM10x (tcgen05)")
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    if request.node.originalname in _PENDING_NAMES or _PENDING_PARAMS(params):
        pytest.skip("not yet supported by the UMMA backend at this stage")
    monkeypatch.setattr(_base, "_BACKEND_KW", {"backend": "umma"})


# ---------------------------------------------------------------------------
# UMMA-only tests
# ---------------------------------------------------------------------------
def _call(case, backend, t, w_ring=16, pdl_trigger=False):
    from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush as f

    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    flush_min = min(w_ring - t + 1, 33 - 2 * t)
    return f(
        A_log,
        g,
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
        flush_min=flush_min,
        restart_hist_on_flush=False,
        w_ring=w_ring,
        pdl_trigger=pdl_trigger,
        backend=backend,
    )


def test_umma_backend_dispatch():
    """backend validation, 'umma' availability errors, and which kernel
    'auto' / 'hmma' / 'umma' actually launch (via the UMMA launch counter)."""
    from flashinfer.experimental.kda_ucache_flush import (
        kda_decode_bf16_wy_ucache_flush as disp,
    )
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    def launches() -> int:
        return um._LAUNCHES

    dev = torch.device("cuda")
    mod = _load_flush()  # pinned to backend="umma" by the autouse fixture
    assert mod.kda_delta_rule_mtp_ucache_flush.keywords == {"backend": "umma"}

    # Unknown backend: ValueError naming the accepted values.
    case = _make_case(4, [0, 3, 12, 13], seed=7)
    with pytest.raises(ValueError, match="backend must be 'auto', 'hmma' or 'umma'"):
        _call(case, "bogus", t=4)

    # pdl_trigger is not supported by the UMMA kernel yet (until stage S4).
    assert disp.kda_umma_unavailable_reason(dev, 4, 16, True) is not None
    case = _make_case(4, [0, 3, 12, 13], seed=7)
    with pytest.raises(ValueError, match="backend='umma' unavailable"):
        _call(case, "umma", t=4, pdl_trigger=True)

    for t in (4, 8):
        for w_ring in (16, 32):
            fm = min(w_ring - t + 1, 33 - 2 * t)
            hist = [0, fm - 1, fm, 1]
            why = disp.kda_umma_unavailable_reason(dev, t, w_ring, False)

            # Explicit 'umma': raises with the reason, or launches UMMA.
            n0 = launches()
            case = _make_case(4, hist, seed=11, t=t)
            if why is not None:
                with pytest.raises(ValueError, match="backend='umma' unavailable"):
                    _call(case, "umma", t=t, w_ring=w_ring)
                assert launches() == n0
            else:
                _call(case, "umma", t=t, w_ring=w_ring)
                assert launches() == n0 + 1

            # 'hmma' never launches UMMA.
            n0 = launches()
            _call(_make_case(4, hist, seed=11, t=t), "hmma", t=t, w_ring=w_ring)
            assert launches() == n0

            # 'auto' launches UMMA iff available and the auto policy is on.
            n0 = launches()
            _call(_make_case(4, hist, seed=11, t=t), "auto", t=t, w_ring=w_ring)
            expect = int(why is None and disp._auto_umma(torch.device("cuda")))
            assert launches() == n0 + expect


def test_umma_backend_validation_precedes_dispatch():
    """Contract violations raise the shared validation error on every
    backend value, before the backend split: 'umma' rejects bad inputs the
    same way 'hmma' does, whatever its availability."""
    for backend in ("hmma", "umma", "auto", "bogus"):
        case = list(_make_case(4, [0, 3, 12, 13], seed=3))
        case[11] = torch.tensor([0, 3, 12, 17], dtype=torch.int32, device="cuda")
        with pytest.raises(ValueError, match="hist_len out of legal range"):
            _call(tuple(case), backend, t=4)


def _fm(t: int, w_ring: int) -> int:
    return min(w_ring - t + 1, 33 - 2 * t)


def _random_case(B, t, w_ring, history, seed):
    """B rows with random hist_len / bases and a permuted pool. history
    "verify": hist in [0, fm); "fold": about 25 % of the rows in [fm, cap]."""
    fm, cap = _fm(t, w_ring), min(w_ring, 32 - t)
    gen = torch.Generator().manual_seed(seed)
    hist = torch.randint(0, fm, (B,), generator=gen)
    if history == "fold":
        deep = torch.randint(fm, cap + 1, (B,), generator=gen)
        hist = torch.where(torch.rand(B, generator=gen) < 0.25, deep, hist)
    hist = hist.tolist()
    bases = torch.randint(0, 32, (B,), generator=gen).tolist()
    perm = torch.randperm(B, generator=gen).cuda()
    case = list(_make_case(B, hist, seed=seed, bases=bases, t=t, pool_size=B))
    for i in (7, 8, 9, 10):  # pool, k / u / g rings: row r's data at slot inv[r]
        case[i] = case[i][perm].contiguous()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(B, device=perm.device)
    case[13] = inv.to(torch.int32)
    return tuple(case), hist, bases


def _append_slots(hist, bases, t):
    return [[(bases[r] + hist[r] + s) % 32 for s in range(t)] for r in range(len(hist))]


@pytest.mark.parametrize("t", [4, 8])
@pytest.mark.parametrize("w_ring", [16, 32])
@pytest.mark.parametrize("history", ["verify", "fold"])
def test_umma_parity_with_hmma(t, w_ring, history):
    """UMMA vs HMMA on identical inputs (tolerance-equal, not bitwise): outputs,
    folded pool rows and the k / u / g ring appends."""
    B = 64
    fm = _fm(t, w_ring)
    res = {}
    for backend in ("hmma", "umma"):
        case, hist, bases = _random_case(B, t, w_ring, history, seed=4100 + t + w_ring)
        y = _call(case, backend, t=t, w_ring=w_ring)
        torch.cuda.synchronize()
        res[backend] = (y.clone(), *[case[i].clone() for i in (7, 8, 9, 10)])
    idx = case[13].long()
    (yh, ph, kh, uh, gh), (yu, pu, ku, uu, gu) = res["hmma"], res["umma"]
    dy = (yh.float() - yu.float()).abs().max().item()
    assert dy < _base.Y_TOL, f"|y_umma - y_hmma| = {dy:.2e}"
    fold = [r for r in range(B) if hist[r] >= fm]
    if fold:
        sl = idx[fold]
        ds = (ph[sl].float() - pu[sl].float()).abs().max().item()
        assert ds < _base.STATE_TOL, f"folded pool |umma - hmma| = {ds:.2e}"
    else:
        ds = 0.0
    slots = _append_slots(hist, bases, t)
    dk = du = dg = 0.0
    for r in range(B):
        s_ = idx[r].item()
        rows = torch.tensor(slots[r], device=yh.device)
        dk = max(
            dk, (kh[s_][:, rows].float() - ku[s_][:, rows].float()).abs().max().item()
        )
        du = max(
            du, (uh[s_][:, rows].float() - uu[s_][:, rows].float()).abs().max().item()
        )
        dg = max(dg, (gh[s_][:, rows] - gu[s_][:, rows]).abs().max().item())
    assert dk < 5e-3, f"k append |umma - hmma| = {dk:.2e}"
    assert du < 2e-2, f"u append |umma - hmma| = {du:.2e}"
    assert dg < 1e-4, f"g append |umma - hmma| = {dg:.2e}"
    print(
        f"\n[parity t={t} w={w_ring} {history}] dy={dy:.2e} dS={ds:.2e} dk={dk:.2e} "
        f"du={du:.2e} dg={dg:.2e}; bitwise y {(yh == yu).float().mean().item():.3f} "
        f"k {(kh == ku).float().mean().item():.3f} g {(gh == gu).float().mean().item():.3f}"
    )


def _poison_outside_window(case, hist, bases):
    """NaN every k / u / g ring slot outside [base, base + P) mod 32 of each row."""
    kc, uc, gc, idx = case[8], case[9], case[10], case[13]
    for r in range(len(hist)):
        live = {(bases[r] + j) % 32 for j in range(hist[r])}
        dead = torch.tensor(
            [s for s in range(32) if s not in live], dtype=torch.long, device=kc.device
        )
        s_ = int(idx[r])
        for ring in (kc, uc, gc):
            ring[s_, :, dead] = float("nan")


@pytest.mark.parametrize("t", _ALL_T)
@pytest.mark.parametrize("w_ring", [16, 32])
@pytest.mark.parametrize("history", ["verify", "fold"])
@pytest.mark.parametrize("strong", [False, True], ids=["mild", "strong"])
def test_umma_nan_poisoned_rings(t, w_ring, history, strong):
    """Stale ring slots are NaN: any read past the live window (or a 0 * NaN
    mask) shows up as a non-finite output / pool or an oracle mismatch."""
    fm, cap = _fm(t, w_ring), min(w_ring, 32 - t)
    if history == "verify":
        hist = [0, 5, fm - 1, 1]
    else:
        hist = [fm, cap, fm - 1, 0]
    bases = [30, 0, 27, 5]
    case = _make_case(4, hist, seed=606 + t, bases=bases, t=t, strong=strong)
    _poison_outside_window(case, hist, bases)
    pool = case[7]
    _base._check_case(_load_flush(), case, bases, t, flush_min=fm, w_ring=w_ring)
    assert torch.isfinite(pool.float()).all()


@pytest.mark.parametrize("t", _ALL_T)
def test_umma_repeat_determinism_w16(t):
    """B=1024 random rows, 50 launches on restored rings / pool: every launch
    bitwise equal to the first (TMEM / mbarrier race tripwire)."""
    B, REPEATS = 1024, 50
    gen = torch.Generator().manual_seed(77 + t)
    hist = torch.randint(0, 17, (B,), generator=gen).tolist()
    bases = torch.randint(0, 32, (B,), generator=gen).tolist()
    case = _make_case(B, hist, seed=78 + t, bases=bases, t=t)
    snap = [x.clone() for x in case[7:11]]

    def once():
        for dst, src in zip(case[7:11], snap, strict=True):
            dst.copy_(src)
        y = _call(case, "umma", t=t, w_ring=16)
        torch.cuda.synchronize()
        return [y.clone()] + [x.clone() for x in case[7:11]]

    ref = once()
    assert torch.isfinite(ref[0].float()).all()
    bad = 0
    for _ in range(REPEATS):
        got = once()
        bad += int(not all(torch.equal(a, b) for a, b in zip(ref, got, strict=True)))
    assert bad == 0, f"{bad}/{REPEATS} identical UMMA launches differ"


@pytest.mark.parametrize("t", _ALL_T)
def test_umma_u_src_smem_vs_ldg_bitwise(t, monkeypatch):
    """The u window from the SMEM bulk copy and from direct loads gives the same
    D tile: outputs, rings and pool bitwise equal."""
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    res = {}
    for src in ("smem", "ldg"):
        monkeypatch.setattr(um, "_FORCE_U_SRC", src)
        case, _, _ = _random_case(64, t, 16, "fold", seed=909 + t)
        y = _call(case, "umma", t=t, w_ring=16)
        torch.cuda.synchronize()
        res[src] = [y.clone()] + [case[i].clone() for i in (7, 8, 9, 10)]
    for name, a, b in zip(
        ("output", "pool", "k_cache", "u_cache", "g_cache"),
        res["smem"],
        res["ldg"],
        strict=True,
    ):
        assert torch.equal(a, b), f"u_src smem vs ldg differ on {name}"


@pytest.mark.parametrize("t", _ALL_T)
@pytest.mark.parametrize("w_ring", [16, 32])
def test_umma_history_dominated_matches_fp32_reference(t, w_ring):
    """History-dominated case: every live window key is (a normalized copy
    of) the step's own first key and q = k, so k̂_j . k̂_t ~ 1 and every window
    slot's u_j reaches y and the u update at full weight. With the random
    keys of the base cases a dropped window slot moves y by about
    |u| / sqrt(K) * SCALE, below Y_TOL; here it moves y by O(|u| * SCALE),
    so a missing or misplaced j-block (w_ring=32: slots 16..31) fails the
    fp32 oracle check. Shallow / deep verify rows and fold rows."""
    fm, cap = _fm(t, w_ring), min(w_ring, 32 - t)
    hist = [0, fm - 1, cap, fm, 3, max(fm - 5, 1)]
    bases = [30, 0, 27, 5, 31, 12]
    case = list(_make_case(len(hist), hist, seed=1717 + t + w_ring, bases=bases, t=t))
    q, k, kc = case[0], case[1], case[8]
    q.copy_(k)
    for r, P in enumerate(hist):
        if P == 0:
            continue
        rows = torch.tensor([(bases[r] + j) % 32 for j in range(P)], device=kc.device)
        kr = torch.nn.functional.normalize(k[r, 0].float(), dim=-1)  # [H, K]
        kc[r, :, rows] = kr[:, None, :].expand(-1, P, -1).to(kc.dtype)
    _base._check_case(_load_flush(), tuple(case), bases, t, flush_min=fm, w_ring=w_ring)


# ---------------------------------------------------------------------------
# T = 1 (STP) and T = 2, 3, 5, 6, 7: UMMA-only draft lengths
# ---------------------------------------------------------------------------
def test_umma_only_t_dispatch(monkeypatch):
    """T outside {4, 8}: 'hmma' keeps its NotImplementedError, 'umma' and
    'auto' launch the UMMA kernel (bitwise the same result), T = 9 is rejected
    by every backend, and 'auto' raises clearly when UMMA is unavailable
    (T in {4, 8} then stays on HMMA)."""
    from flashinfer.experimental.kda_ucache_flush import (
        kda_decode_bf16_wy_ucache_flush as disp,
    )
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    def launches() -> int:
        return um._LAUNCHES

    dev = torch.device("cuda")
    for t in _UMMA_ONLY_T:
        fm = _fm(t, 16)
        hist = [0, fm - 1, fm, 1]
        assert disp.kda_umma_unavailable_reason(dev, t, 16, False) is None
        with pytest.raises(NotImplementedError, match=r"native T in \{4, 8\} only"):
            _call(_make_case(4, hist, seed=13, t=t), "hmma", t=t)
        outs = {}
        for backend in ("umma", "auto"):
            n0 = launches()
            case = _make_case(4, hist, seed=13, t=t)
            y = _call(case, backend, t=t)
            assert launches() == n0 + 1, backend
            outs[backend] = [y.clone()] + [case[i].clone() for i in (7, 8, 9, 10)]
        for a, b in zip(outs["umma"], outs["auto"], strict=True):
            assert torch.equal(a, b)

    for backend in ("hmma", "umma", "auto"):
        with pytest.raises(NotImplementedError, match="T=9 unsupported"):
            _call(_make_case(4, [0, 0, 0, 0], seed=13, t=9), backend, t=8)

    monkeypatch.setattr(
        disp, "kda_umma_unavailable_reason", lambda *a, **kw: "forced off"
    )
    with pytest.raises(NotImplementedError, match="needs the UMMA backend.*forced off"):
        _call(_make_case(4, [0, 15, 16, 1], seed=13, t=1), "auto", t=1)
    with pytest.raises(ValueError, match="backend='umma' unavailable: forced off"):
        _call(_make_case(4, [0, 15, 16, 1], seed=13, t=1), "umma", t=1)
    n0 = launches()
    _call(_make_case(4, [0, 12, 13, 1], seed=13, t=4), "auto", t=4)
    assert launches() == n0


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
@pytest.mark.parametrize("history", ["empty_P0", "replay_max", "fold_mixed"])
@pytest.mark.parametrize("basecase", list(BASES))
def test_umma_only_t_output_matches_fp32_reference(t, history, basecase):
    _base.test_output_matches_fp32_reference(t, history, basecase)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
@pytest.mark.parametrize("basecase", list(BASES))
def test_umma_only_t_folded_state_matches_fp32_reference(t, basecase):
    _base.test_folded_state_matches_fp32_reference(t, basecase)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_strong_decay_matches_fp32_reference(t):
    _base.test_strong_decay_matches_fp32_reference(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_ring_append_values_match_reference(t):
    _base.test_ring_append_values_match_reference(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_flush_never_overwrites_live_window(t):
    _base._flush_never_overwrites_live_window(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_chained_steps_match_sequential_oracle(t):
    _base._chained_steps_match_sequential_oracle(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_padded_rows_and_permuted_pool(t):
    _base._padded_rows_and_permuted_pool(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_cuda_graph_replay_matches_eager(t):
    _base._cuda_graph_replay_matches_eager(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_large_batch_randomized_oracle_and_determinism(t):
    _base.test_large_batch_randomized_oracle_and_determinism(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_lower_bound_range(t):
    """lower_bound in [-80 / T, 0): at T = 1 the boundary is -80 (one
    saturated token spans the whole e^{80} exp(-cum) budget)."""
    _base.test_lower_bound_range(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
@pytest.mark.parametrize("basecase", list(BASES))
def test_umma_only_t_w32_output_and_fold_match_fp32_reference(t, basecase):
    _base.test_w32_output_and_fold_match_fp32_reference(t, basecase)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_w32_strong_decay_deep_window(t):
    _base.test_w32_strong_decay_deep_window(t)


@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_w32_deep_window_repeat_determinism(t):
    # tma_late only selects the HMMA load schedule; UMMA ignores it
    _base.test_w32_deep_window_repeat_determinism(t, False)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "cuda_graph"])
@pytest.mark.parametrize("w_ring", [16, 32])
@pytest.mark.parametrize("t", _UMMA_ONLY_T)
def test_umma_only_t_restart_hist_on_flush_across_steps(t, w_ring, graph):
    _base._restart_hist_on_flush_across_steps(graph, w_ring, t)


@pytest.mark.parametrize("t", _ALL_T)
@pytest.mark.parametrize("w_ring", [16, 32])
@pytest.mark.parametrize("history", ["verify", "fold"])
def test_umma_persistent_matches_single_item_bitwise(t, w_ring, history, monkeypatch):
    """The persistent kernel (kda_replay_mtp_umma_persistent: 4 CTAs / SM walking the
    (request, head) items) only reschedules work: outputs, rings and pool are bitwise
    equal to one item per CTA, with padded rows (every 37th) and more items than CTAs."""
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    B = 700  # 8400 (request, head) items: about 14 per CTA on 148 SMs
    res = {}
    for pz in (False, True):
        monkeypatch.setattr(um, "_FORCE_PERSISTENT", pz)
        case, _, _ = _random_case(B, t, w_ring, history, seed=4242 + 10 * t + w_ring)
        case = list(case)
        idx = case[13].clone()
        idx[::37] = -1
        case[13] = idx
        y = _call(tuple(case), "umma", t=t, w_ring=w_ring)
        torch.cuda.synchronize()
        live = idx >= 0
        res[pz] = [y[live].clone()] + [case[i].clone() for i in (7, 8, 9, 10)]
    for name, a, b in zip(
        ("output", "pool", "k_cache", "u_cache", "g_cache"),
        res[False],
        res[True],
        strict=True,
    ):
        assert torch.equal(a, b), f"persistent vs single-item differ on {name}"


def _pz_cases(n, t, seed):
    """n independent persistent-sized cases (B=600: about 12 items per CTA slot)."""
    return [list(_random_case(600, t, 16, "fold", seed=seed + i)[0]) for i in range(n)]


def _pz_snapshot(case, y):
    return [y.clone()] + [case[i].clone() for i in (7, 8, 9, 10)]


@pytest.mark.parametrize("mode", ["streams", "graphs"])
def test_umma_persistent_concurrent_launches(mode, monkeypatch):
    """Persistent launches on two streams claim from separate work counters. "graphs": two
    CUDA graphs captured with torch.cuda.graph's defaults (one capture stream) and replayed
    concurrently on two streams -- with one shared counter this corrupts the results. "streams":
    eager calls on two streams (the eager range check syncs, so these only exercise the
    per-stream counters and their reset). Results are bitwise equal to running the cases one
    after another, and every counter ends at zero."""
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um
    from flashinfer.experimental.kda_ucache_flush import (
        kda_replay_mtp_umma_persistent as pzm,
    )

    monkeypatch.setattr(um, "_FORCE_PERSISTENT", True)
    t = 4
    # serial reference (fresh copies of the same cases)
    ref = []
    for c in _pz_cases(2, t, 9100):
        y = _call(tuple(c), "umma", t=t)
        torch.cuda.synchronize()
        ref.append(_pz_snapshot(c, y))
    cases = _pz_cases(2, t, 9100)
    s0, s1 = torch.cuda.Stream(), torch.cuda.Stream()
    outs = [None, None]
    if mode == "streams":
        torch.cuda.synchronize()
        for _ in range(4):  # several rounds: the counters must reset between launches
            cases = _pz_cases(2, t, 9100)
            torch.cuda.synchronize()
            for i, s in enumerate((s0, s1)):
                with torch.cuda.stream(s):
                    torch.cuda._sleep(200_000)
                    outs[i] = _call(tuple(cases[i]), "umma", t=t)
            torch.cuda.synchronize()
            for i in range(2):
                for a, b in zip(ref[i], _pz_snapshot(cases[i], outs[i]), strict=True):
                    assert torch.equal(a, b)
    else:
        _call(tuple(_pz_cases(1, t, 1)[0]), "umma", t=t)  # eager call: arena topped up
        torch.cuda.synchronize()
        graphs = []
        for i in range(2):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                outs[i] = _call(tuple(cases[i]), "umma", t=t)
            graphs.append(gr)
        snap = [[cases[i][j].clone() for j in (7, 8, 9, 10)] for i in range(2)]
        for _ in range(3):
            for i in range(2):  # restore the inputs the kernel updates in place
                for j, v in zip((7, 8, 9, 10), snap[i], strict=True):
                    cases[i][j].copy_(v)
            torch.cuda.synchronize()
            for i, s in enumerate((s0, s1)):
                with torch.cuda.stream(s):
                    torch.cuda._sleep(200_000)
                    graphs[i].replay()
            torch.cuda.synchronize()
            for i in range(2):
                for a, b in zip(ref[i], _pz_snapshot(cases[i], outs[i]), strict=True):
                    assert torch.equal(a, b)

    for c in pzm._COUNTERS.values():
        assert c.tolist() == [0, 0]
    for a in pzm._ARENA.values():
        for c in a["chunks"]:
            assert int(c.abs().sum()) == 0


def test_umma_persistent_dirty_counter_recovery(monkeypatch):
    """A completed persistent launch leaves its claim counter at zero; a counter left
    non-zero (a launch that did not complete, or any corruption) is recovered by
    persistent_counter_reset(): every launch after it is bitwise equal to the one-item
    kernel and leaves the counter at zero again."""
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um
    from flashinfer.experimental.kda_ucache_flush import (
        kda_replay_mtp_umma_persistent as pzm,
    )

    t = 4
    monkeypatch.setattr(um, "_FORCE_PERSISTENT", False)
    c = _pz_cases(1, t, 9300)[0]
    ref = _pz_snapshot(c, _call(tuple(c), "umma", t=t))
    monkeypatch.setattr(um, "_FORCE_PERSISTENT", True)
    _call(tuple(_pz_cases(1, t, 9300)[0]), "umma", t=t)  # creates this stream's counter
    torch.cuda.synchronize()
    key = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
    word = pzm._COUNTERS[key]
    assert word.tolist() == [0, 0]
    for gv in ([5, 0], [0, 3], [7, 2], [100000, 0], [-1, -1]):
        word.copy_(torch.tensor(gv, dtype=torch.int32))
        pzm.persistent_counter_reset()
        for _ in range(2):
            c = _pz_cases(1, t, 9300)[0]
            got = _pz_snapshot(c, _call(tuple(c), "umma", t=t))
            torch.cuda.synchronize()
            for a, b in zip(ref, got, strict=True):
                assert torch.equal(a, b), f"wrong results after counter {gv} + reset"
            assert word.tolist() == [0, 0]


@pytest.mark.parametrize("nh", [129, 160, 256])
def test_umma_persistent_large_h_bitwise(nh, monkeypatch):
    """H > 128: the persistent kernel's SMEM A_log table holds every head (it held 128 and
    read past the end for heads >= 128). Persistent == one-item kernel, bitwise, for
    verify and fold rows; H > 256 never takes the persistent kernel."""
    from flashinfer.experimental.kda_ucache_flush import kda_replay_mtp_umma as um

    monkeypatch.setattr(_base, "H", nh)
    t, B = 4, 24
    res = {}
    for pz in (False, True):
        monkeypatch.setattr(um, "_FORCE_PERSISTENT", pz)
        assert um._use_persistent(B, t, nh, 16, torch.device("cuda")) is pz
        case, _, _ = _random_case(B, t, 16, "fold", seed=777 + nh)
        y = _call(case, "umma", t=t)
        torch.cuda.synchronize()
        res[pz] = [y.clone()] + [case[i].clone() for i in (7, 8, 9, 10)]
    for a, b in zip(res[False], res[True], strict=True):
        assert torch.equal(a, b)
    monkeypatch.setattr(um, "_FORCE_PERSISTENT", True)
    assert not um._use_persistent(B, t, 257, 16, torch.device("cuda"))


def test_umma_one_item_grid_limit_raises(monkeypatch):
    """The one-item kernel puts the request on grid.y (<= 65535): larger batches on that
    path raise a clear ValueError instead of a launch failure."""
    from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush as f

    t, B, pool = 1, 65536, 4
    case = list(_make_case(pool, [0] * pool, seed=5, t=t, pool_size=pool))
    for i in (0, 1, 2, 3, 4):  # q, k, v, g, b: B rows (inputs only read for live rows)
        case[i] = case[i][:1].expand(B, *case[i].shape[1:]).contiguous()
    idx = torch.full((B,), -1, dtype=torch.int32, device="cuda")
    idx[:pool] = torch.arange(pool, dtype=torch.int32, device="cuda")
    hl = torch.zeros(B, dtype=torch.int32, device="cuda")
    cb = torch.zeros(B, dtype=torch.int32, device="cuda")
    q, k, v, g, b, A_log, dt_bias, st, kc, uc, gc = case[:11]
    with pytest.raises(ValueError, match="65535"):
        f(
            A_log, g, dt_bias, q=q, k=k, v=v, b=b, initial_state_source=st,
            initial_state_indices=idx, k_cache=kc, u_cache=uc, g_cache=gc,
            hist_len=hl, cache_base=cb, w_ring=32, flush_min=17,
            restart_hist_on_flush=False, backend="umma",
        )  # fmt: skip
