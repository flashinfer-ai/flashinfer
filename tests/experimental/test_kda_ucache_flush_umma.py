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

Run:
  pytest tests/experimental/test_kda_ucache_flush_umma.py -v
"""

from __future__ import annotations

import pytest
import torch

from . import test_kda_ucache_flush as _base
from .test_kda_ucache_flush import *  # noqa: F401,F403  re-collects all base cases
from .test_kda_ucache_flush import _load_flush, _make_case

# Base tests the UMMA backend cannot run yet at the current stage. MUST be
# empty once the kernel covers T in {4, 8} x w_ring in {16, 32} (stage S3).
# S2: verify rows at T=4 / w_ring=16 only. Rows with hist_len >= flush_min run
# (outputs and ring appends are exact) but do not fold their ring into the
# pool yet, so every test that checks a folded pool row is pending, and so is
# every w_ring=32 test. Base tests that run flush rows but never read the
# pool back (ring appends, live-window protection, graph-vs-eager bitwise,
# lower_bound outputs) run.
_PENDING_NAMES: set = {
    "test_folded_state_matches_fp32_reference",
    "test_strong_decay_matches_fp32_reference",
    "test_padded_rows_and_permuted_pool",
    "test_large_batch_randomized_oracle_and_determinism",
    "test_restart_hist_on_flush_across_steps",
    "test_w32_output_and_fold_match_fp32_reference",
    "test_w32_tma_late_forced",
    "test_w32_strong_decay_deep_window",
    "test_w32_equivalence_with_w16_at_shallow_windows",
    "test_w32_deep_window_repeat_determinism",
}


def _PENDING_PARAMS(params: dict) -> bool:
    """Per-parameter pending filter: T=8, w_ring=32 and fold cases (S2)."""
    return (
        params.get("t") == 8
        or params.get("w_ring") == 32
        or params.get("history") in ("fold_mixed", "fold")
    )


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
            expect = int(why is None and disp._AUTO_UMMA)
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


@pytest.mark.parametrize("t", [4, 8])
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


@pytest.mark.parametrize("t", [4, 8])
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


@pytest.mark.parametrize("t", [4, 8])
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
