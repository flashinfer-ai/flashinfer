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
Correctness tests for the Rubin (sm_107a) build of the UMMA MTP ReplaySSM kernel
(flashinfer/gdn_kernels/gdn_replay_mtp_umma_rubin.py).

1. Every test of tests/gdn/test_mtp_umma.py, re-run against
   ``gated_delta_rule_mtp_ucache_flush_umma_rubin`` (same fp32 oracle, cases and
   tolerances; the base file's kernel entry is swapped by an autouse fixture).
2. What only the Rubin build has: bit-identity with the base kernel, the fp32 gate-parameter
   cache across streams and in-place updates, and the persistent kernel (sm_107a):
   bit-identity with the non-persistent launch at T = 1-8 on any row mix and grid, the automatic
   choice, the dispatch rule and ``flush_hint``, a caller-owned claim counter, and concurrent
   replays of separately captured CUDA graphs; every heads-per-CTA split bit-identical at
   T = 1-8 in every rounding mode; uGPU buffers (``ugpu_ucache_buffers``: one launch per die)
   bit-identical to the 4-D buffers, eager and captured; the routing of
   ``gated_delta_rule_mtp_ucache_flush`` to this build.

Run:  pytest tests/gdn/test_mtp_umma_rubin.py -q
"""

from __future__ import annotations

import importlib

import pytest
import torch

from . import test_mtp_umma as _base
from .test_mtp_umma import *  # noqa: F401,F403  (re-collects every base test here)
from .test_mtp_umma import DEV, RING, SCALE, W, _make_case

_RUBIN = (
    "flashinfer.gdn_kernels.gdn_replay_mtp_umma_rubin",
    "gated_delta_rule_mtp_ucache_flush_umma_rubin",
)
_RUBIN_CC = ((10, 0), (10, 7))  # archs the Rubin build runs on


def _rubin_mod():
    return importlib.import_module(_RUBIN[0])


def _umma():
    return getattr(_rubin_mod(), _RUBIN[1])


def _skip_if_no_rubin_build():
    from flashinfer.utils import get_compute_capability

    _base._skip_if_not_sm100()
    if get_compute_capability(torch.device("cuda")) not in _RUBIN_CC:
        pytest.skip("the Rubin build runs on sm_107a and sm_100a only")


@pytest.fixture(autouse=True)
def _rubin_entry(monkeypatch):
    """Point the base file's tests (collected here by the star import) at the Rubin build."""
    if torch.cuda.is_available():
        from flashinfer.utils import get_compute_capability

        cc = get_compute_capability(torch.device("cuda"))
        if cc[0] == 10 and cc not in _RUBIN_CC:
            pytest.skip("the Rubin build runs on sm_107a and sm_100a only")
    monkeypatch.setattr(_base, "_ENTRY", _RUBIN)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("hpc", [1, 2, 4])
def test_bit_identical_to_base_kernel(T, hpc):
    """Round to nearest: outputs, state pool and rings bit-identical to the base kernel on a mixed
    verify / flush / padded batch (wrapped windows)."""
    _skip_if_no_rubin_build()
    from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
        gated_delta_rule_mtp_ucache_flush_umma as base_fn,
    )

    case, fm = _pz_case(T, 500 + 11 * T + hpc, "FVFFPVVFPFFFVVPF")
    ref = _pz_call(case, fm, fn=base_fn, heads_per_cta=hpc)
    got = _pz_call(case, fm, heads_per_cta=hpc)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), name


def test_auto_heads_per_cta_rubin_caps():
    """auto_heads_per_cta with T (the sm_107a caps): hpc = 1 while the hpc = 1 grid fits one wave
    of CTA slots, hpc = 2 while the hpc = 2 grid fits 1.5 waves at T >= 2 (1.75 with LCG or
    Philox of at most 6 rounds), 2.4 (flush rows first, partial-device launch) / 2.7 waves at
    T = 1; Philox with more than 6 rounds: 2.75 / 2.5 waves (T = 1 / T = 2-4) with flush rows
    first on a partial-device launch, 8 / 2.1 (T <= 4 / T >= 5) on a
    partial-device launch without them, 3.6 on a whole-device launch at T <= 4, else the 1.75
    of stochastic rounding; no flush rows: 0.65 / 0.8 waves, cap2 one wave at T = 1 and on
    whole-device launches at T = 2-4 (odd heads per key: one hpc = 1 wave at T >= 5). Without
    T: the 8-wave cap whatever the hint and rounding (also the base file's policy test, run here
    against this build)."""
    f = _rubin_mod().auto_heads_per_cta
    die = (8, 106)  # one die of a Rubin uGPU launch: key heads and SMs (424 CTA slots)
    assert [f(b, *die) for b in (13, 14, 212, 213)] == [1, 2, 2, 4]
    got = [f(b, *die, 4, None, "none", "philox", 10, True) for b in (13, 14, 212, 213)]
    assert got == [1, 2, 2, 4]
    # caps that land on a whole grid: hpc = 1 / 2 while the grid fits, not only below (one key
    # head, 106 SMs: 4 x 106 hpc = 1 CTAs is one wave, 2 x 318 hpc = 2 CTAs 1.5 waves)
    for T in (None, 2):
        assert [f(b, 1, 106, 4, T) for b in (106, 107)] == [1, 2], T
    assert [f(b, 1, 106, 4, 2) for b in (318, 319)] == [2, 4]
    for T in (2, 4, 8):
        for hint in (None, "some", "flush_first"):
            got = [f(b, *die, 4, T, hint) for b in (13, 14, 39, 40, 212)]
            assert got == [1, 2, 2, 4, 4], (T, hint)
            for sr, rounds in (
                ("philox", 5),
                ("philox", 6),
                ("lcg", 0),
                ("lcg_clock", 0),
            ):
                got = [f(b, *die, 4, T, hint, sr, rounds) for b in (13, 14, 46, 47)]
                assert got == [1, 2, 2, 4], (T, hint, sr, rounds)
    for sr in ("", "philox"):  # T = 1: the hint moves the cap, Philox-5 / 6 do not
        for rounds in (5, 6):
            got = [
                f(b, *die, 4, 1, "flush_first", sr, rounds) for b in (13, 14, 63, 64)
            ]
            assert got == [1, 2, 2, 4], (sr, rounds)
            for hint in (None, "some"):
                got = [f(b, *die, 4, 1, hint, sr, rounds) for b in (13, 14, 71, 72)]
                assert got == [1, 2, 2, 4], (hint, sr, rounds)
    # Philox with 10 rounds on a die: with flush rows first hpc = 2 to 72 rows (T = 1) / 66
    # (T = 2-4) / 46 (T >= 5); without them to 212 (T <= 4) / 55 (T >= 5)
    assert [f(b, *die, 4, 1, "flush_first", "philox", 10) for b in (72, 73)] == [2, 4]
    for T in (2, 3, 4):
        got = [f(b, *die, 4, T, "flush_first", "philox", 10) for b in (13, 14, 66, 67)]
        assert got == [1, 2, 2, 4], T
    for hint in (None, "some"):
        got = [f(b, *die, 4, 1, hint, "philox", 10) for b in (13, 14, 128, 212, 213)]
        assert got == [1, 2, 2, 2, 4], hint
        for T in (2, 3, 4):
            got = [
                f(b, *die, 4, T, hint, "philox", 10) for b in (13, 14, 128, 212, 213)
            ]
            assert got == [1, 2, 2, 2, 4], (T, hint)
    for T in (5, 6, 7, 8):
        got = [f(b, *die, 4, T, "flush_first", "philox", 10) for b in (46, 47)]
        assert got == [2, 4], T
        for hint in (None, "some"):
            got = [f(b, *die, 4, T, hint, "philox", 10) for b in (47, 55, 56)]
            assert got == [2, 2, 4], (T, hint)
    # the Philox rounds threshold: up to 6 rounds the stochastic-rounding cap, from 7 the
    # Philox caps (rounds 7 and 9 untimed)
    for rounds in (1, 5, 6):
        for T in (2, 4, 5):
            got = [f(b, *die, 4, T, None, "philox", rounds) for b in (46, 47)]
            assert got == [2, 4], (T, rounds)
        got = [f(b, *die, 4, 1, "flush_first", "philox", rounds) for b in (63, 64)]
        assert got == [2, 4], rounds
        assert [f(b, *die, 4, 1, None, "philox", rounds) for b in (71, 72)] == [2, 4]
    for rounds in (7, 8, 9):
        for T in (2, 4):
            got = [f(b, *die, 4, T, None, "philox", rounds) for b in (47, 212, 213)]
            assert got == [2, 2, 4], (T, rounds)
            got = [f(b, *die, 4, T, "flush_first", "philox", rounds) for b in (66, 67)]
            assert got == [2, 4], (T, rounds)
        got = [f(b, *die, 4, 1, "flush_first", "philox", rounds) for b in (72, 73)]
        assert got == [2, 4], rounds
        assert [f(b, *die, 4, 1, None, "philox", rounds) for b in (212, 213)] == [2, 4]
        assert [f(b, *die, 4, 5, None, "philox", rounds) for b in (55, 56)] == [2, 4]
        assert [
            f(b, *die, 4, 5, "flush_first", "philox", rounds) for b in (46, 47)
        ] == [
            2,
            4,
        ]
    # no flush rows: hpc = 2 while its grid fits one wave at T = 1, 0.8 waves at T >= 2
    for sr, rounds in (("", 0), ("lcg", 0), ("philox", 10)):
        got = [f(b, *die, 4, 1, "none", sr, rounds) for b in (8, 9, 26, 27)]
        assert got == [1, 2, 2, 4], sr
        for T in (2, 4, 5, 8):
            got = [f(b, *die, 4, T, "none", sr, rounds) for b in (8, 9, 21, 22)]
            assert got == [1, 2, 2, 4], (T, sr)
    # one whole-device launch (16 key heads, 212 SMs): the same waves; flush rows first counts
    # as rows in any order, Philox above 6 rounds keeps 3.6 waves at T <= 4 and 1.75 at T >= 5,
    # and without flush rows hpc = 2 holds to one wave at T = 1-4
    whole = (16, 212, 4)
    got = [f(b, *whole, 4, None, "", 0, True) for b in (13, 14, 39, 40)]
    assert got == [1, 2, 2, 4]
    assert [f(b, *whole, 1, "flush_first", "", 0, True) for b in (71, 72)] == [2, 4]
    assert [f(b, 16, 212, 4, 1, "flush_first") for b in (63, 64)] == [2, 4]
    for T in (1, 2, 4):
        for hint in (None, "flush_first"):
            got = [f(b, *whole, T, hint, "philox", 10, True) for b in (95, 96)]
            assert got == [2, 4], (T, hint)
    for hint in (None, "flush_first"):
        assert [f(b, *whole, 6, hint, "philox", 10, True) for b in (46, 47)] == [2, 4]
    for T in (1, 2, 3, 4):
        got = [f(b, *whole, T, "none", "", 0, True) for b in (8, 9, 26, 27)]
        assert got == [1, 2, 2, 4], T
    for T in (5, 8):
        got = [f(b, *whole, T, "none", "", 0, True) for b in (21, 22)]
        assert got == [2, 4], T
    # 2 value heads per key head: both past one hpc = 1 wave; 3: no hpc = 2 (and without flush
    # rows hpc = 1 to one wave at T >= 5, 0.65 waves at T <= 4)
    assert [f(b, *die, 2, 4) for b in (26, 27, 500)] == [1, 2, 2]
    assert [f(b, *die, 3, 4) for b in (17, 18)] == [1, 3]
    # odd heads per key without flush rows: 0.65 hpc = 1 waves to T = 4, one wave from T = 5
    for T in (1, 2, 3, 4):
        got = [f(b, *die, 3, T, "none") for b in (11, 12, 17, 18)]
        assert got == [1, 3, 3, 3], T
    for T in (5, 6, 7, 8):
        got = [f(b, *die, 3, T, "none") for b in (11, 12, 17, 18)]
        assert got == [1, 1, 1, 3], T
    # even heads per key keep the 0.65 waves of hpc = 1 at every T
    for T in (1, 4, 5):
        assert [f(b, *die, 4, T, "none") for b in (8, 9)] == [1, 2], T


def test_persistent_auto_items():
    """_pz_auto's minimum (key head, row) items per CTA slot (CPU only): with flush_hint "none"
    _PZ_AUTO_MIN_ITEMS["none"][T] (H = 8 on a 106-SM die: B >= 53 x items, e.g. T = 2 186,
    T = 4 128, T = 8 292), 8 otherwise (B >= 424); never on a whole-device launch, at T = 1, with
    fewer heads per CTA or with grid_override."""
    from types import SimpleNamespace

    M = _rubin_mod()
    rubin = SimpleNamespace(major=10, minor=7, num_sms=212)

    def auto(T, B, hint, nsm=106, hpc=None, grid=None, target=rubin):
        return M._pz_auto(T, 4, hpc, 8 * B, nsm, grid, target, hint)

    tab = M._PZ_AUTO_MIN_ITEMS["none"]
    assert tab == {2: 3.5, 3: 2.4, 4: 2.4, 5: 2.4, 6: 5.0, 7: 5.5, 8: 5.5}
    # the first B per die with 8 B >= items x 424 CTA slots
    first_b = {2: 186, 3: 128, 4: 128, 5: 128, 6: 265, 7: 292, 8: 292}
    for T, first in first_b.items():
        assert 8 * first >= tab[T] * 424 > 8 * (first - 1), T
        got = [auto(T, b, "none") for b in (53, first - 1, first)]
        assert got == [False, False, True], T
    for T in (2, 4, 8):
        for hint in (None, "some", "flush_first"):
            assert [auto(T, b, hint) for b in (423, 424)] == [False, True], (T, hint)
    assert not auto(1, 4096, "none")
    assert not auto(4, 4096, "none", nsm=212)
    assert not auto(4, 4096, "none", hpc=2)
    assert not auto(4, 4096, "none", grid=3)
    assert not auto(
        4, 4096, "none", target=SimpleNamespace(major=10, minor=0, num_sms=148)
    )


# ---------------------------------------------------------------------------
# Host-side launch choices without a GPU: _launch on meta tensors with _get_kernel replaced by a
# recorder (CPU-only; the bitwise checks of the same choices are the GPU tests further down).
def _targets():
    from flashinfer.gdn_kernels.device_target import GdnDeviceTarget

    return (
        GdnDeviceTarget(0, "sm_107a", 10, 7, 212, True),
        GdnDeviceTarget(0, "sm_100a", 10, 0, 148, True),
    )


def _cpu_launch(
    monkeypatch,
    B,
    T,
    target,
    *,
    H=8,
    HV=32,
    num_sms=106,
    flush_hint="flush_first",
    sr="",
    persistent=None,
    heads_per_cta=None,
    use_ro=True,
    **extra,
):
    """One _launch over meta tensors (a die's 8 / 32 heads on 106 SMs by default); returns the
    recorded _get_kernel call: dict(ro, hpc, pz, sr, kw)."""
    M = _rubin_mod()
    seen = []

    def spy(device, T_, h, hv, u_bf16, ro, hpc, sr_, sr_rounds, pz=False, *rest, **kw):
        seen.append(dict(ro=ro, hpc=hpc, pz=pz, sr=sr_, kw=kw))
        return lambda *a: None

    monkeypatch.setattr(M, "_get_kernel", spy)
    monkeypatch.setattr(M, "_pz_counter", lambda dev: None)
    meta = torch.device("meta")

    def e(*shape, dt=torch.bfloat16):
        return torch.empty(shape, dtype=dt, device=meta)

    q = e(B, T, H, 128)
    v = e(B, T, HV, 128)
    ab = e(B, T, HV)
    i32 = e(B, dt=torch.int32)
    M._launch(
        q,
        q,
        v,
        ab,
        ab,
        e(HV, dt=torch.float32),
        e(HV, dt=torch.float32),
        e(4, HV, 128, 128, dt=torch.float16),
        e(4, H, 32, 128),
        e(4, HV, 32, 128),
        e(4, HV, 32, dt=torch.float32),
        v,
        0,
        num_sms,
        None,
        state_idx=i32,
        hist_len=i32,
        cache_base=i32,
        row_order=i32,
        rand_seed=e(1, dt=torch.int64),
        T=T,
        u_bf16=True,
        use_ro=use_ro,
        heads_per_cta=heads_per_cta,
        sr=sr,
        sr_rounds=5 if sr == "philox" else 0,
        persistent=persistent,
        grid_override=None,
        flush_hint=flush_hint,
        target=target,
        scale=1.0,
        flush_min=17 - T,
        h_io=H,
        hv_io=HV,
        **extra,
    )
    assert len(seen) == 1, seen
    return seen[0]


def test_small_b_row_order_rule(monkeypatch):
    """_SMALL_B_DISPATCH_CC (sm_107a): a non-persistent launch of at most 2 CTAs per SM (8 key heads
    x 4 / hpc CTAs per row on 106 SMs: B <= 6 / 13 / 26 at hpc = 1 / 2 / 4), or without flush rows
    (flush_hint "none"), ignores row_order; larger grids, the persistent kernel and sm_100a keep
    it; a call without row_order stays without. CPU-only."""
    rubin, b200 = _targets()

    def ro(B, T, tgt=rubin, **kw):
        return _cpu_launch(monkeypatch, B, T, tgt, **kw)["ro"]

    for hpc, edge in ((1, 6), (2, 13), (4, 26)):
        got = [ro(B, 4, heads_per_cta=hpc, persistent=False) for B in (edge, edge + 1)]
        assert got == [False, True], (hpc, got)
    # automatic heads per CTA: hpc = 1 up to B = 13 (grid 4 CTAs per SM), 2 from 14 (B > 13)
    assert [ro(B, 1) for B in (1, 6, 7, 13, 14, 40)] == [False] * 2 + [True] * 4
    # no flush rows: any B on the non-persistent kernel; the persistent one keeps its order
    assert [ro(B, 4, flush_hint="none", persistent=False) for B in (1, 64, 512)] == [
        False
    ] * 3
    assert ro(512, 4, flush_hint="none", persistent=True) is True
    assert ro(4, 4, use_ro=False) is False
    monkeypatch.setattr(_rubin_mod(), "_SMALL_B_DISPATCH_CC", ())
    assert ro(4, 4) is True and ro(64, 4, flush_hint="none", persistent=False) is True
    monkeypatch.undo()
    assert ro(4, 4, b200, num_sms=74) is True


def test_small_b_launch_bound(monkeypatch):
    """_SMALL_GRID_LB_CC (sm_107a): a small non-persistent grid (at most 2 CTAs per SM) with 4
    value heads per key head compiles with a 2-CTA/SM launch bound, except the register-split
    builds (setmaxnreg assumes 64 registers per thread) and Philox (no measured gain); larger
    grids, other heads per key head, the persistent kernel and sm_100a keep 4. CPU-only."""
    M = _rubin_mod()
    rubin, b200 = _targets()

    def mb(B, T, tgt=rubin, **kw):
        return _cpu_launch(monkeypatch, B, T, tgt, **kw)["kw"]["min_blocks"]

    for hpc, edge in ((1, 6), (2, 13), (4, 26)):
        got = [mb(B, 4, heads_per_cta=hpc, persistent=False) for B in (edge, edge + 1)]
        assert got == [2, 4], (hpc, got)
    # register split: RN 4 heads per CTA at T >= 5, Philox 2 heads at T = 8
    assert M._use_regsplit(8, 4, "") and mb(4, 8, heads_per_cta=4) == 4
    assert mb(4, 8, heads_per_cta=2) == 2
    assert mb(4, 8, heads_per_cta=2, sr="philox") == 4
    assert mb(4, 8, heads_per_cta=2, sr="lcg") == 2
    assert mb(4, 4, heads_per_cta=1, sr="philox") == 4
    assert mb(4, 4, persistent=True) == 4
    assert mb(4, 4, b200, num_sms=74) == 4
    # 4 value heads per key head only (the measured shape): HV / H = 2 or 3 keep 4
    assert (
        mb(4, 4, HV=16, heads_per_cta=1) == 4 and mb(4, 4, HV=24, heads_per_cta=1) == 4
    )
    monkeypatch.setattr(M, "_SMALL_GRID_LB_CC", ())
    assert mb(4, 4) == 4
    # the build: the launch bound only, nothing else
    o2 = M.GdnReplayMtpUmmaRubin(4, 8, 32, True, False, 2, min_blocks=2)
    o4 = M.GdnReplayMtpUmmaRubin(4, 8, 32, True, False, 2)
    assert o2.min_blocks == 2 and o4.min_blocks == 4
    assert {k: v for k, v in vars(o2).items() if k != "min_blocks"} == {
        k: v for k, v in vars(o4).items() if k != "min_blocks"
    }
    with pytest.raises(AssertionError):
        M.GdnReplayMtpUmmaRubin(
            8, 8, 32, True, False, 4, min_blocks=2
        )  # register split


def test_small_b_build_options():
    """GdnReplayMtpUmmaRubin(small_b=True) (compile passes it on _SMALL_B_BUILD_CC, sm_107a): the
    RMW loop in the round-to-nearest builds with 1-2 heads per CTA, the prologue reorder (no
    vsmem / D-build masks) at T >= 5 with 1 head per CTA, no rolled head loop in the
    round-to-nearest 2-head builds; 4 heads per CTA, stochastic rounding's roll and the
    persistent build unchanged. CPU-only."""
    M = _rubin_mod()
    keys = ("rmw_loop", "pro_reorder", "vsmem", "d_mask", "roll", "roll_f", "mixed_fma")

    def flags(T, hpc, sr="", small_b=False, persistent=False):
        o = M.GdnReplayMtpUmmaRubin(
            T, 8, 32, True, False, hpc, sr, 5 if sr == "philox" else 0, True,
            persistent, small_b=small_b,
        )  # fmt: skip
        return {k: getattr(o, k) for k in keys}

    for T in range(1, 9):
        for sr in ("", "philox", "lcg"):
            for hpc in (1, 2, 4):
                old, new = flags(T, hpc, sr), flags(T, hpc, sr, True)
                want = dict(old)
                if sr == "" and hpc < 4:
                    want.update(rmw_loop=True, mixed_fma=True)
                if T >= 5 and hpc == 1:
                    want.update(pro_reorder=True, vsmem=False, d_mask=False)
                if sr == "" and hpc == 2:
                    want.update(roll=False)
                want["roll_f"] = want["roll"] or (
                    want["rmw_loop"]
                    and _rubin_mod()._use_regsplit(T, hpc, sr)
                    and hpc == 4
                )
                assert new == want, (T, sr, hpc, old, new)
        assert flags(T, 4, "", True, True) == flags(T, 4, "", False, True)
    # compile passes small_b on sm_107a with 4 value heads per key head only: with 2 or 3 the
    # hpc = 2 / 3 builds serve every batch size
    rubin, b200 = _targets()
    sb = M._small_b_build
    assert [
        sb(rubin, 8, 32),
        sb(rubin, 16, 64),
        sb(rubin, 8, 16),
        sb(rubin, 8, 24),
    ] == [
        True,
        True,
        False,
        False,
    ]
    assert not sb(b200, 8, 32)


# the sm_107a small-batch switches: each () gives the code without it (same bits)
_SMALL_B_SWITCHES = ("_SMALL_B_DISPATCH_CC", "_SMALL_GRID_LB_CC", "_SMALL_B_BUILD_CC")


@pytest.mark.parametrize("mode", ["none", "philox5", "lcg"])
def test_small_b_bit_identical(mode, monkeypatch):
    """The small-batch choices (_SMALL_B_SWITCHES) change no bit: outputs, state pool and rings
    equal with every switch off, on small and larger grids, with and without flush rows, row
    order flush-first, automatic heads per CTA and persistent choice. One whole-device launch."""
    _skip_if_no_rubin_build()
    M = _rubin_mod()
    sr = _sr_kw(mode)
    for T, pattern, hint in (
        (1, "FV", "flush_first"),
        (2, "VFPVFFVV" * 3, "flush_first"),
        (5, "FVVF", None),
        (6, "FVVFVFVVPF" * 2, "flush_first"),
        (8, "VFP", None),
        (4, "V" * 36, "none"),
        (4, "FV" * 20, "flush_first"),
    ):
        case, fm = _pz_case(T, 880 + len(pattern), pattern)
        ro = M.flush_first_row_order(case[11], fm)
        for hpc in (None, 1, 2):
            kw = dict(flush_hint=hint, row_order=ro, heads_per_cta=hpc, **sr)
            with monkeypatch.context() as m:
                for name in _SMALL_B_SWITCHES:
                    m.setattr(M, name, ())
                ref = _pz_call(case, fm, **kw)
            got = _pz_call(case, fm, **kw)
            for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
                assert torch.equal(x, y), (T, pattern, hint, hpc, mode, name)


def test_auto_heads_per_cta_dispatch(monkeypatch):
    """_launch passes T, flush_hint, the rounding mode with its Philox rounds and whether the
    launch covers the whole device to auto_heads_per_cta on sm_107a only (sm_100a keeps the
    8-wave cap: 2 heads per CTA in every case below), and the automatic choice gives the same
    bits as 4 heads per CTA. One whole-device launch over 16 key heads."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    M = _rubin_mod()
    on_rubin = get_compute_capability(torch.device("cuda")) == (10, 7)
    nsm = torch.cuda.get_device_properties(DEV).multi_processor_count
    seen = []
    real = M._get_kernel

    def spy(*args, **kw):
        seen.append(args[6])  # heads per CTA
        return real(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy)
    # T, rows, flush_hint, rounding, heads per CTA on Rubin (212 SMs)
    for T, B, hint, mode, rubin_hpc in (
        (4, 48, None, "none", 4),
        (4, 44, None, "lcg", 2),  # stochastic rounding: hpc = 2 up to 46 rows
        (4, 48, None, "philox5", 4),
        (4, 60, None, "philox10", 2),  # Philox-10 at T <= 4: hpc = 2 up to 95 rows
        (4, 60, "flush_first", "philox10", 2),  # whole device: flush-first as any order
        (8, 48, None, "philox10", 4),  # T >= 5: the stochastic-rounding cap (46 rows)
        (1, 66, "flush_first", "none", 2),  # whole device: flush-first as any order
        (1, 72, None, "none", 4),  # T = 1, flush rows anywhere: hpc = 2 up to 71 rows
        (4, 24, "none", "none", 2),  # no flush rows, T <= 4: hpc = 2 up to 26 rows
        (4, 28, "none", "none", 4),
        (8, 24, "none", "none", 4),  # no flush rows, T >= 5: hpc = 2 up to 21 rows
    ):
        case, fm = _pz_case(T, 770 + B, _ugpu_pattern(B, 770 + B))
        H = case[0].shape[2]
        sr_ = {"none": "", "lcg": "lcg"}.get(mode, "philox")
        rounds = int(mode[len("philox") :]) if sr_ == "philox" else 0
        if not on_rubin:
            want = M.auto_heads_per_cta(B, H, nsm, 4)
        elif nsm == 212:
            want = rubin_hpc
        else:
            want = M.auto_heads_per_cta(B, H, nsm, 4, T, hint, sr_, rounds, True)
        sr = _sr_kw(mode)
        ref = _pz_call(case, fm, heads_per_cta=4, **sr)
        seen.clear()
        got = _pz_call(
            case, fm, heads_per_cta=None, persistent=False, flush_hint=hint, **sr
        )
        assert seen == [want], (T, B, hint, mode, seen)
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), (T, B, hint, mode, name)


def test_auto_heads_per_cta_ugpu_dies(monkeypatch):
    """uGPU buffers on Rubin: _launch_ugpu's per-die launches (8 of the 16 key heads on one
    die's 106 SMs, a partial-device launch, so "flush_first" moves the caps) pick hpc = 4 from
    B = 64 at T = 1 with "flush_first" (72 without), from 73 (T = 1) / 67 (T = 2-4) / 47
    (T >= 5) with Philox-10 and "flush_first" (213 / 213 / 56 without), from 40 at T >= 2 with round
    to nearest, from 47 with Philox-6 (Philox-7: the Philox-10 caps), and without flush rows
    from 27 at T = 1 and 22 at T = 2; the same bits as the 4-D buffers with 4 heads per CTA.
    Eager calls this small normally run the per-die launches back to back on the whole device;
    the test lowers that threshold so they take the green-context path of captured serving."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    M = _rubin_mod()
    if get_compute_capability(torch.device("cuda")) != (10, 7):
        pytest.skip("the per-die caps are sm_107a only")
    dev = torch.cuda.current_device()
    if M._locality_domains(dev) != 2:
        pytest.skip("needs a device with two locality domains")
    if M._ugpu_streams(dev)[1] != [106, 106]:
        pytest.skip("the boundaries below are for 106-SM dies")
    monkeypatch.setattr(M, "_UGPU_EAGER_MIN_STATES", 0)
    # the per-die launches at every B (few rows otherwise take the fused launch: test_ugpu_path_rule)
    monkeypatch.setattr(M, "_UGPU_FUSED_CC", ())
    seen, rule_args = [], []
    real_kernel, real_rule = M._get_kernel, M.auto_heads_per_cta

    def spy_kernel(*args, **kw):
        seen.append(args[6])  # heads per CTA
        return real_kernel(*args, **kw)

    def spy_rule(*args, **kw):
        rule_args.append((args[1], args[2], kw.get("whole_device")))
        return real_rule(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy_kernel)
    monkeypatch.setattr(M, "auto_heads_per_cta", spy_rule)
    # T, rows, flush_hint, rounding, heads per CTA on each die
    for T, B, hint, mode, want in (
        (1, 63, "flush_first", "none", 2),
        (1, 64, "flush_first", "none", 4),
        (1, 71, None, "none", 2),
        (1, 72, None, "none", 4),
        (1, 72, "flush_first", "philox10", 2),
        (1, 73, "flush_first", "philox10", 4),
        (2, 66, "flush_first", "philox10", 2),
        (2, 67, "flush_first", "philox10", 4),
        (4, 67, None, "philox10", 2),  # no hint, T <= 4: the 8-wave cap
        (4, 128, None, "philox10", 2),
        (1, 128, None, "philox10", 2),  # T = 1, no hint: the 8-wave cap
        (3, 213, None, "philox10", 4),
        (8, 47, "flush_first", "philox10", 4),
        (8, 55, None, "philox10", 2),  # T >= 5, no hint: hpc = 2 up to 55 rows
        (8, 56, None, "philox10", 4),
        (4, 66, "flush_first", "philox5", 4),  # 5 rounds: the stochastic-rounding cap
        (2, 47, None, "philox6", 4),  # 6 rounds too
        (2, 47, None, "philox7", 2),  # 7: the Philox caps
        (1, 26, "none", "none", 2),  # no flush rows, T = 1: hpc = 2 up to 26 rows
        (1, 27, "none", "none", 4),
        (2, 21, "none", "none", 2),  # T >= 2 on a die: up to 21
        (2, 22, "none", "none", 4),
        (2, 39, None, "none", 2),
        (2, 40, None, "none", 4),
        (8, 39, "flush_first", "none", 2),
        (8, 40, "flush_first", "none", 4),
    ):
        case, fm = _pz_case(T, 880 + B, _ugpu_pattern(B, 880 + B))
        sr = _sr_kw(mode)
        ref = _pz_call(case, fm, heads_per_cta=4, **sr)
        seen.clear()
        rule_args.clear()
        got = _ugpu_call(
            case, fm, heads_per_cta=None, persistent=False, flush_hint=hint, **sr
        )
        assert seen == [want, want], (T, B, hint, mode, seen)
        assert rule_args == [(8, 106, False)] * 2, (T, B, hint, mode, rule_args)
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), (T, B, hint, mode, name)


def test_persistent_rejected_off_sm107():
    """persistent=True is sm_107a only: elsewhere a clear ValueError, not a silent fallback."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        pytest.skip("sm_107a runs the persistent kernel")
    case, fm = _pz_case(8, 71, "FVVPF")
    with pytest.raises(ValueError, match="sm_107a"):
        _pz_call(case, fm, persistent=True)


# ---------------------------------------------------------------------------
# persistent kernel (opt-in; sm_107a): CTAs loop over (key head, row) items claimed from a
# counter. It must be bit-identical to the non-persistent launch on any mix and order of
# verify (V), flush (F) and padded (P) rows, at any grid size.
_PZ_PATTERNS = (
    "FVFFPVVFPFFFVVPFVF",  # every transition: V->F, F->V, F->F, V->V, a padded row between
    "FFFFFFFFFFFF",  # every row flushes
    "VVVVVVVVVVVF",  # one flush row, the last item
    "FPVVVVPVVVVV",  # one flush row, the first item
)


def _skip_if_no_persistent():
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    if get_compute_capability(torch.device("cuda")) != (10, 7):
        pytest.skip("the persistent kernel is enabled on sm_107a only")


def _pz_case(T, seed, pattern):
    import random

    fm = W - T + 1
    rng = random.Random(seed)
    hist = [rng.randint(fm, W) if c == "F" else rng.randint(0, fm - 1) for c in pattern]
    bases = [rng.randrange(RING) for _ in pattern]
    case = list(_make_case(len(pattern), T, hist, seed, bases))
    case[13] = torch.tensor(
        [-1 if c == "P" else r for r, c in enumerate(pattern)],
        dtype=torch.int32,
        device=DEV,
    )
    return case, fm


def _pz_bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _pz_call(case, fm, fn=None, **kw):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)
    kw.setdefault("heads_per_cta", 4)
    (fn or _umma())(
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
        output=out,
        **kw,
    )
    torch.cuda.synchronize()
    return [_pz_bits(x) for x in (out, pool, kc, uc, gc)]


def test_bf16_gate_params_two_streams():
    """bf16 A_log / dt_bias are converted to fp32 once and cached: a launch on another stream (e.g. the
    other die's stream of a die-localized split) must not read the copy before the converting stream
    has written it. Stream 0 is held back, so an unsynchronized read on stream 1 sees no copy yet."""
    _skip_if_no_rubin_build()
    T, B = 4, 8
    fm = W - T + 1
    hist = [fm, 3, 0, fm, 7, fm, 1, 9]
    case = _make_case(B, T, hist, seed=55, bases=[3, 30, 0, 17, 9, 28, 5, 12])
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    fn = _umma()
    ss = [torch.cuda.Stream(), torch.cuda.Stream()]

    def run(A_, dt_, delay):
        bufs = [x.clone() for x in (pool, kc, uc, gc)]
        ys = []
        torch.cuda.synchronize()
        for d, s in enumerate(ss):
            r = slice(4 * d, 4 * d + 4)
            with torch.cuda.stream(s):
                if d == 0 and delay:
                    torch.cuda._sleep(20_000_000)
                y = fn(
                    A_,
                    a[r],
                    dt_,
                    q=q[r],
                    k=k[r],
                    v=v[r],
                    b=b[r],
                    initial_state_source=bufs[0],
                    initial_state_indices=idx[r],
                    k_cache=bufs[1],
                    u_cache=bufs[2],
                    g_cache=bufs[3],
                    hist_len=hl[r],
                    cache_base=cb[r],
                    scale=SCALE,
                    flush_min=fm,
                    restart_hist_on_flush=False,
                )
                ys.append(y)
        torch.cuda.synchronize()
        return [_pz_bits(x) for x in [torch.cat(ys)] + bufs]

    ref = run(A_log.float(), dt_bias.float(), False)
    # fresh bf16 objects: their first conversion is enqueued on the held-back stream 0
    A_, dt_ = A_log.clone(), dt_bias.clone()
    got = run(A_, dt_, True)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), name
    # warm cache, both streams already synchronized with the conversion
    got = run(A_, dt_, True)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), ("warm", name)


def test_bf16_gate_params_inplace_update():
    """The fp32 copy of a bf16 A_log / dt_bias is re-made after an in-place update of the same
    object (e.g. a weight reload), not served stale."""
    _skip_if_no_rubin_build()
    T, B = 4, 8
    fm = W - T + 1
    hist = [fm, 3, 0, fm, 7, fm, 1, 9]
    case = _make_case(B, T, hist, seed=56, bases=[3, 30, 0, 17, 9, 28, 5, 12])
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    fn = _umma()

    def run(A_, dt_):
        bufs = [x.clone() for x in (pool, kc, uc, gc)]
        y = fn(
            A_,
            a,
            dt_,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
        )
        torch.cuda.synchronize()
        return [_pz_bits(x) for x in [y] + bufs]

    A_, dt_ = A_log.clone(), dt_bias.clone()
    run(A_, dt_)  # caches A_ / dt_
    A_new = (A_log.float() + 0.25).to(A_log.dtype)
    dt_new = (dt_bias.float() - 0.25).to(dt_bias.dtype)
    ref = run(A_new.float(), dt_new.float())
    A_.copy_(A_new)
    dt_.copy_(dt_new)
    got = run(A_, dt_)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), name
    # inference-mode tensors (no version counter) are cached by identity
    with torch.inference_mode():
        A_i, dt_i = A_new.clone(), dt_new.clone()
    got = run(A_i, dt_i)
    got2 = run(A_i, dt_i)
    for name, x, y, z in zip(("y", "pool", "k", "u", "g"), ref, got, got2, strict=True):
        assert torch.equal(x, y) and torch.equal(x, z), ("inference", name)


@pytest.mark.parametrize("arena", [True, False])
def test_persistent_graphs_concurrent(arena, monkeypatch):
    """Two persistent launches captured in two separate CUDA graphs (torch.cuda.graph defaults: one
    capture stream) and replayed concurrently on two streams must not share a claim counter (the
    captured counters come from the eagerly zeroed arena, or without it from a memset node)."""
    _skip_if_no_persistent()
    M = _rubin_mod()

    if not arena:
        monkeypatch.setattr(M, "_PZ_ARENA_SLOTS", 0)
        monkeypatch.setattr(M, "_PZ_ARENA", {})

    case, fm = _pz_case(8, 91, "FVVVPVFVVFVVVVVVFVPVVFVV")
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    n = q.shape[0] // 2
    parts = [slice(0, n), slice(n, 2 * n)]
    bufs = [x.clone() for x in (pool, kc, uc, gc)]
    outs = [
        torch.full(v[r].shape, -7.0, dtype=torch.bfloat16, device=DEV) for r in parts
    ]
    fn = _umma()

    def call(d):
        r = parts[d]
        fn(
            A_log,
            a[r],
            dt_bias,
            q=q[r],
            k=k[r],
            v=v[r],
            b=b[r],
            initial_state_source=bufs[0],
            initial_state_indices=idx[r],
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl[r],
            cache_base=cb[r],
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            heads_per_cta=4,
            output=outs[d],
            persistent=True,
            num_sms=8,
        )

    ref = _pz_call(case, fm)
    call(0)  # eager warm-up (compile, fp32 parameter cache)
    call(1)
    torch.cuda.synchronize()
    graphs = []
    for d in range(2):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call(d)
        graphs.append(g)
    ss = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(6):
        for x, x0 in zip(bufs, (pool, kc, uc, gc), strict=True):
            x.copy_(x0)
        torch.cuda.synchronize()
        for g, s in zip(graphs, ss, strict=True):
            with torch.cuda.stream(s):
                g.replay()
        torch.cuda.synchronize()
        got = [_pz_bits(x) for x in [torch.cat(outs)] + bufs]
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), name
    assert all(c.tolist() == [0, 0] for c in M._PZ_CTR.values())
    assert all(
        int(c.abs().sum()) == 0 for a in M._PZ_ARENA.values() for c in a["chunks"]
    )
    assert bool(M._PZ_ARENA) == arena


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("mode", ["none", "philox5", "lcg"])
def test_persistent_bit_identical(T, mode):
    _skip_if_no_persistent()
    sr = {}
    if mode != "none":
        sr = dict(
            stochastic_rounding=mode.rstrip("5"),
            rand_seed=torch.tensor([1234], dtype=torch.int64, device=DEV),
        )
        if mode == "philox5":
            sr["philox_rounds"] = 5
    for i, pattern in enumerate(_PZ_PATTERNS):
        case, fm = _pz_case(T, 300 + 7 * T + i, pattern)
        ref = _pz_call(case, fm, **sr)
        for grid in (1, 3, None):
            got = _pz_call(case, fm, persistent=True, grid_override=grid, **sr)
            for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
                assert torch.equal(x, y), (pattern, grid, name)


def test_persistent_dispatch_rule(monkeypatch):
    """persistent=True runs the non-persistent kernel where the measured rule says so (Philox,
    T = 8, few items per CTA slot), the persistent one otherwise or with grid_override; same bits."""
    _skip_if_no_persistent()
    M = _rubin_mod()

    seen = []
    real = M._get_kernel

    def spy(*args, **kw):
        seen.append(args[9] if len(args) > 9 else kw.get("persistent", False))
        return real(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy)
    case, fm = _pz_case(8, 79, "FVVPFVVFVVVF")
    nkh = case[0].shape[2]
    sr = dict(
        stochastic_rounding="philox",
        rand_seed=torch.tensor([99], dtype=torch.int64, device=DEV),
        philox_rounds=5,
    )
    ref = _pz_call(case, fm, **sr)
    B = case[0].shape[0]
    lim = M._PZ_PLAIN_MAX_ITEMS["some"][("philox", 8)] * M._CTA_SLOTS_PER_SM
    few = -(-nkh * B // lim)  # the fewest SMs at which the rule still applies
    for kw, want in (
        (dict(num_sms=few), False),
        (dict(num_sms=few - 1), True)
        if few > 1
        else (dict(num_sms=few, grid_override=3), True),
        (dict(num_sms=few, grid_override=3), True),
    ):
        seen.clear()
        got = _pz_call(case, fm, persistent=True, **kw, **sr)
        assert seen == [want], (kw, seen)
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True)), kw
    seen.clear()
    _pz_call(case, fm, persistent=True, num_sms=few)  # round to nearest: no rule
    assert seen == [True]


def test_persistent_flush_hint(monkeypatch):
    """flush_hint: None / "some" use the dispatch rule's flush threshold (non-persistent kernel at
    <= 10 items per CTA slot), "none" / "flush_first" the lower one (<= 5): between the two only
    they run the persistent kernel; same bits either way, also from a captured CUDA graph (the
    hint is read at capture); anything else is a ValueError."""
    _skip_if_no_persistent()
    M = _rubin_mod()

    seen = []
    real = M._get_kernel

    def spy(*args, **kw):
        seen.append(args[9] if len(args) > 9 else kw.get("persistent", False))
        return real(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy)
    case, fm = _pz_case(8, 83, "FFFVVPVVVFVV")
    sr = dict(
        stochastic_rounding="philox",
        rand_seed=torch.tensor([7], dtype=torch.int64, device=DEV),
        philox_rounds=5,
    )
    ref = _pz_call(case, fm, **sr)
    n_items = case[0].shape[2] * case[0].shape[0]
    slots = M._CTA_SLOTS_PER_SM
    hi, lo = (M._PZ_PLAIN_MAX_ITEMS[k][("philox", 8)] for k in ("some", "few"))
    few = -(-n_items // (hi * slots))  # the fewest SMs at which the "some" rule applies
    more = -(-n_items // (lo * slots))  # the fewest at which the "few" rule applies too
    assert few < more
    assert M._pz_runs_plain(8, "philox", n_items, few)
    assert not M._pz_runs_plain(8, "philox", n_items, few, "none")
    assert not M._pz_runs_plain(8, "philox", n_items, more - 1, "flush_first")
    assert M._pz_runs_plain(8, "philox", n_items, more, "flush_first")
    for nsm, hint, want in (
        (few, None, False),
        (few, "some", False),
        (few, "none", True),
        (few, "flush_first", True),
        (more, "none", False),
        (more, "flush_first", False),
    ):
        seen.clear()
        got = _pz_call(case, fm, persistent=True, num_sms=nsm, flush_hint=hint, **sr)
        assert seen == [want], (nsm, hint, seen)
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True)), hint
    # outside the rule (round to nearest) every hint runs the persistent kernel
    for hint in (None, "some", "none", "flush_first"):
        seen.clear()
        _pz_call(case, fm, persistent=True, num_sms=few, flush_hint=hint)
        assert seen == [True], hint
    # under graph capture: the kernel chosen at capture is replayed
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    bufs = [x.clone() for x in (pool, kc, uc, gc)]
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)

    def call():
        _umma()(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            heads_per_cta=4,
            output=out,
            persistent=True,
            num_sms=few,
            flush_hint="none",
            **sr,
        )

    call()
    torch.cuda.synchronize()
    seen.clear()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    assert seen == [True]
    for _ in range(3):
        for x, x0 in zip(bufs, (pool, kc, uc, gc), strict=True):
            x.copy_(x0)
        g.replay()
        torch.cuda.synchronize()
        got = [_pz_bits(x) for x in [out] + bufs]
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))
    for bad in ("all", "None", 0):
        with pytest.raises(ValueError):
            _pz_call(case, fm, persistent=True, num_sms=few, flush_hint=bad)
    with pytest.raises(ValueError):
        _pz_call(case, fm, flush_hint="mixed")  # validated without persistent too


def test_persistent_auto(monkeypatch):
    """persistent=None (the default): the persistent kernel at T in _PZ_AUTO_T (2-8) with all value
    heads per CTA when num_sms names fewer SMs than the device (a die-localized launch) and there are
    enough items per CTA slot; the non-persistent kernel otherwise. Same bits either way."""
    _skip_if_no_persistent()
    M = _rubin_mod()
    seen = []
    real = M._get_kernel

    def spy(*args, **kw):
        seen.append(args[9] if len(args) > 9 else kw.get("persistent", False))
        return real(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy)
    monkeypatch.setattr(M, "_PZ_AUTO_MIN_ITEMS", {"none": 0, "some": 0})
    nsm = torch.cuda.get_device_properties(DEV).multi_processor_count
    for T, pat in ((1, "FVVPF"), (2, "VFVVP"), (4, "FVVPFV"), (8, "VFVVF")):
        case, fm = _pz_case(T, 400 + T, pat)
        ref = _pz_call(case, fm, persistent=False)
        auto = T in M._PZ_AUTO_T
        for kw, want in (
            (dict(), False),  # no num_sms: a whole-device launch
            (dict(num_sms=nsm), False),
            (dict(num_sms=nsm // 2), auto),
            (dict(num_sms=nsm // 2, heads_per_cta=2), False),
            (dict(num_sms=nsm // 2, grid_override=3), False),
        ):
            seen.clear()
            got = _pz_call(case, fm, **kw)
            assert seen == [want], (T, kw, seen)
            assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True)), (
                T,
                kw,
            )
    # too few items per CTA slot: non-persistent, unless the launch has no flush rows
    monkeypatch.setattr(M, "_PZ_AUTO_MIN_ITEMS", {"none": 0, "some": 10**6})
    case, fm = _pz_case(4, 404, "VVVPVV")
    for hint, want in (
        (None, False),
        ("some", False),
        ("flush_first", False),
        ("none", True),
    ):
        seen.clear()
        _pz_call(case, fm, num_sms=nsm // 2, flush_hint=hint)
        assert seen == [want], (hint, seen)


def test_persistent_api():
    """Scope checks, and a caller-owned claim counter (zeroed again by the launch's last CTA)."""
    _skip_if_no_persistent()
    case, fm = _pz_case(8, 77, "FVVPF")
    ref = _pz_call(case, fm)
    ctr = torch.zeros(2, dtype=torch.int32, device=DEV)
    got = _pz_call(case, fm, persistent=True, grid_override=2, persistent_counter=ctr)
    assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))
    assert ctr.tolist() == [0, 0]
    with pytest.raises(ValueError):
        _pz_call(case, fm, persistent=True, persistent_counter=ctr.to(torch.int64))
    with pytest.raises(
        ValueError
    ):  # the persistent kernel runs all 4 value heads per CTA
        _pz_call(case, fm, persistent=True, heads_per_cta=2)


# ---------------------------------------------------------------------------
# uGPU buffers ([pool, dies, heads / dies, ...], see ugpu_ucache_buffers): one launch per die's
# key heads, bitwise the same as the 4-D buffers in every rounding mode. Without two locality
# domains (older drivers), and for small eager calls, the per-die launches run back to back on
# the current stream; otherwise (Rubin, B200 on recent drivers) on the dies' green contexts.
def _sr_kw(mode):
    """Keyword arguments of a rounding mode of _SR_MODES (seeded, so the bits are reproducible)."""
    if mode == "none":
        return {}
    kw = dict(
        stochastic_rounding="lcg" if mode == "lcg" else "philox",
        rand_seed=torch.tensor([1234], dtype=torch.int64, device=DEV),
    )
    if mode != "lcg":
        kw["philox_rounds"] = int(mode[len("philox") :])
    return kw


# rounding modes of the bit-identity tests: round to nearest, seeded LCG, Philox 5 / 10 rounds
_SR_MODES = ("none", "lcg", "philox5", "philox10")


def _ugpu_split(x, n, layout):
    """[pool, heads, ...] -> [pool, n, heads / n, ...]: a view of x ("view"), or a copy whose
    parts are separate dense blocks, as ugpu_ucache_buffers allocates them ("parts")."""
    v5 = x.view(x.shape[0], n, x.shape[1] // n, *x.shape[2:])
    if layout == "view":
        return v5
    parts = torch.empty(
        (n, x.shape[0], x.shape[1] // n, *x.shape[2:]), dtype=x.dtype, device=x.device
    ).transpose(0, 1)
    parts.copy_(v5)
    return parts


def _ugpu_logical(x):
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])


def _ugpu_call(case, fm, bufs=None, layout="parts", **kw):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    if bufs is None:
        bufs = [_ugpu_split(x, 2, layout) for x in (pool, kc, uc, gc)]
    else:
        for dst, src in zip(bufs, (pool, kc, uc, gc), strict=True):
            dst.copy_(_ugpu_split(src, dst.shape[1], "view"))
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)
    kw.setdefault("heads_per_cta", 4)
    _umma()(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=bufs[0],
        initial_state_indices=idx,
        k_cache=bufs[1],
        u_cache=bufs[2],
        g_cache=bufs[3],
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=fm,
        restart_hist_on_flush=False,
        output=out,
        **kw,
    )
    torch.cuda.synchronize()
    return [_pz_bits(x) for x in [out] + [_ugpu_logical(y) for y in bufs]]


def _ugpu_pattern(B, seed):
    import random

    rng = random.Random(seed)
    return "".join(rng.choice("VVVVFFP") for _ in range(B))


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mode", _SR_MODES)
@pytest.mark.parametrize("layout", ["view", "parts"])
def test_ugpu_buffers_bit_identical(T, mode, layout):
    """5-D uGPU buffers: same bits as the 4-D buffers with 4 heads per CTA (stochastic rounding
    included: its ids use the global value head), for both kernels and every heads-per-CTA, with
    and without a row order (flush rows first), at every T and rounding mode (the heads per CTA
    pick different builds, e.g. the register split with 4 heads at T >= 5 and with Philox's 2
    heads at T = 8, and auto_heads_per_cta changes them by batch size, T, flush hint and rounding
    mode)."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    sr = _sr_kw(mode)
    case, fm = _pz_case(T, 700 + 7 * T, _PZ_PATTERNS[0])
    ref = _pz_call(case, fm, **sr)
    ff = _rubin_mod().flush_first_row_order(case[11], fm)
    runs = [dict(heads_per_cta=h) for h in (1, 2, 4)]
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        runs += [dict(persistent=True), dict(persistent=True, grid_override=3)]
    for kw in runs:
        for ro in (None, ff):
            got = _ugpu_call(case, fm, layout=layout, row_order=ro, **kw, **sr)
            for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
                assert torch.equal(x, y), (kw, ro is not None, name)


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mode", _SR_MODES)
def test_heads_per_cta_bit_identical(T, mode):
    """4-D buffers: 1, 2 and 4 heads per CTA (and on sm_107a the persistent kernel), with and
    without a row order (flush rows first), give the same bits as 4 heads per CTA without one on a
    mixed verify / flush / padded batch, at every T and rounding mode (the heads per CTA are a
    performance choice only, see auto_heads_per_cta)."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    sr = _sr_kw(mode)
    case, fm = _pz_case(T, 760 + 7 * T, _PZ_PATTERNS[0])
    ref = _pz_call(case, fm, **sr)  # 4 heads per CTA, no row order
    ff = _rubin_mod().flush_first_row_order(case[11], fm)
    runs = [dict(heads_per_cta=h) for h in (1, 2, 4)]
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        runs += [dict(persistent=True), dict(persistent=True, grid_override=3)]
    for kw in runs:
        for ro in (None, ff):
            got = _pz_call(case, fm, row_order=ro, **kw, **sr)
            for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
                assert torch.equal(x, y), (kw, ro is not None, name)


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("mode", ["none", "philox5", "lcg"])
@pytest.mark.parametrize("layout", ["view", "parts"])
def test_ugpu_paths_bit_identical(T, mode, layout):
    """Every uGPU launch path (one launch over both parts' heads, per-part launches back to back,
    on two ordinary streams, on the dies' green contexts): same bits as the 4-D buffers, eager and
    replayed from a CUDA graph."""
    _skip_if_no_rubin_build()
    M = _rubin_mod()
    sr = _sr_kw(mode)
    B = 23
    case, fm = _pz_case(T, 990 + 7 * T, _ugpu_pattern(B, 990 + T))
    ref = _pz_call(case, fm, **sr)
    paths = ["fused", "seq", "streams"]
    if M._locality_domains(torch.cuda.current_device()) == 2:
        paths.append("dies")
    try:
        for path in paths:
            M._UGPU_PATH_OVERRIDE = path
            for kw in (
                dict(heads_per_cta=1),
                dict(heads_per_cta=2),
                dict(heads_per_cta=4),
                dict(heads_per_cta=None),
            ):
                got = _ugpu_call(case, fm, layout=layout, **kw, **sr)
                for name, x, y in zip(
                    ("y", "pool", "k", "u", "g"), ref, got, strict=True
                ):
                    assert torch.equal(x, y), (path, kw, name)
            # captured
            q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
            bufs = [_ugpu_split(x.clone(), 2, layout) for x in (pool, kc, uc, gc)]
            out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)

            def call():
                _umma()(
                    A_log,
                    a,
                    dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    b=b,
                    initial_state_source=bufs[0],
                    initial_state_indices=idx,
                    k_cache=bufs[1],
                    u_cache=bufs[2],
                    g_cache=bufs[3],
                    hist_len=hl,
                    cache_base=cb,
                    scale=SCALE,
                    flush_min=fm,
                    restart_hist_on_flush=False,
                    output=out,
                    **sr,
                )

            call()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            for _ in range(2):
                for x, x0 in zip(bufs, (pool, kc, uc, gc), strict=True):
                    x.copy_(_ugpu_split(x0, 2, "view"))
                out.fill_(-7.0)
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                got = [_pz_bits(x) for x in [out] + [_ugpu_logical(y) for y in bufs]]
                for name, x, y in zip(
                    ("y", "pool", "k", "u", "g"), ref, got, strict=True
                ):
                    assert torch.equal(x, y), (path, "graph", name)
    finally:
        M._UGPU_PATH_OVERRIDE = None


def test_ugpu_path_rule():
    """_ugpu_path: one fused launch for few rows on sm_107a (H x B <= _UGPU_FUSED_MAX_ITEMS[T] x
    SMs: B <= 12 / 8 / 4 at T = 1-2 / 3-5 / 6-8 on 212 SMs; every small eager call), the per-die
    green-context launches
    above it under capture; sm_100a, 2 or 3 value heads per key head and persistent /
    grid_override calls keep the old paths."""
    M = _rubin_mod()
    from flashinfer.gdn_kernels.device_target import GdnDeviceTarget

    rubin = GdnDeviceTarget(0, "sm_107a", 10, 7, 212, True)
    b200 = GdnDeviceTarget(0, "sm_100a", 10, 0, 148, True)

    def p(B, T, tgt, capturing, persistent=None, grid_override=None, HV=64):
        return M._ugpu_path(
            B, 16, HV, T, None, tgt, capturing, persistent, grid_override
        )

    assert M._UGPU_PATH_OVERRIDE is None
    assert [p(B, 4, rubin, True) for B in (1, 8, 9, 64)] == [
        "fused",
        "fused",
        "dies",
        "dies",
    ]
    for T, last in ((1, 12), (2, 12), (3, 8), (5, 8), (6, 4), (7, 4), (8, 4)):
        assert [p(B, T, rubin, True) for B in (last, last + 1)] == ["fused", "dies"], T
    assert [p(B, 4, rubin, False) for B in (64, 127, 128)] == ["fused", "fused", "dies"]
    assert p(8, 4, rubin, True, persistent=True) == "dies"
    assert p(8, 4, rubin, False, persistent=True) == "seq"
    assert p(8, 4, rubin, True, grid_override=3) == "dies"
    assert [p(8, 4, b200, c) for c in (True, False)] == ["dies", "seq"]
    # 4 value heads per key head only (the measured shape)
    assert [p(8, 4, rubin, c, HV=hv) for c in (True, False) for hv in (32, 48)] == [
        "dies",
        "dies",
        "seq",
        "seq",
    ]
    try:
        M._UGPU_FUSED_CC = ()
        assert [p(8, 4, rubin, c) for c in (True, False)] == ["dies", "seq"]
    finally:
        M._UGPU_FUSED_CC = ((10, 7),)


def test_ugpu_buffers_graph():
    """A captured uGPU call (on Rubin: the per-die green-context launches) replays to the same bits
    as the 4-D buffers, with a per-die caller-owned claim counter where persistent."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    T, B = 4, 48
    case, fm = _pz_case(T, 811, _ugpu_pattern(B, 811))
    ref = _pz_call(case, fm)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    bufs = [_ugpu_split(x.clone(), 2, "parts") for x in (pool, kc, uc, gc)]
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)
    kw = {}
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        kw = dict(
            persistent=True,
            persistent_counter=torch.zeros(2, 2, dtype=torch.int32, device=DEV),
        )

    def call():
        _umma()(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            output=out,
            **kw,
        )

    call()  # eager warm-up: compile, green-context streams, fp32 parameter cache
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(3):
        for x, x0 in zip(bufs, (pool, kc, uc, gc), strict=True):
            x.copy_(_ugpu_split(x0, 2, "view"))
        out.fill_(-7.0)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        got = [_pz_bits(x) for x in [out] + [_ugpu_logical(y) for y in bufs]]
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), name
    if kw:
        assert kw["persistent_counter"].tolist() == [[0, 0], [0, 0]]


def test_ugpu_ucache_buffers():
    """Buffers from ugpu_ucache_buffers (each die's part in its own memory) and an eager call large
    enough for the per-die green-context launches: same bits as the 4-D buffers. Needs a device
    with two locality domains (Rubin; B200 with a driver that exposes them)."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    M = _rubin_mod()
    if M._locality_domains(torch.cuda.current_device()) != 2:
        pytest.skip("needs a device with two locality domains")
    T, B = 8, 192  # B x HV = 12288: above the eager fork / join threshold
    case, fm = _pz_case(T, 905, _ugpu_pattern(B, 905))
    ref = _pz_call(case, fm)
    runs = [{}, dict(persistent=False)]
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        runs.append(dict(persistent=True))
    for kw in runs:
        bufs = list(M.ugpu_ucache_buffers(B, 16, 64, device=DEV))
        assert [tuple(x.shape) for x in bufs] == [
            (B, 2, 32, 128, 128),
            (B, 2, 8, 32, 128),
            (B, 2, 32, 32, 128),
            (B, 2, 32, 32),
        ]
        got = _ugpu_call(case, fm, bufs=bufs, **kw)
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), (kw, name)


def test_ugpu_buffers_rejected():
    """Mismatched 5-D buffers: a ValueError naming the expected layout."""
    _skip_if_no_rubin_build()
    case, fm = _pz_case(4, 61, "FVVPF")
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    bufs = [_ugpu_split(x, 2, "view") for x in (pool, kc, uc, gc)]
    bad = [bufs[0], kc, bufs[2], bufs[3]]  # a 4-D k ring next to a 5-D pool
    with pytest.raises(ValueError, match="uGPU buffers"):
        _umma()(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bad[0],
            initial_state_indices=idx,
            k_cache=bad[1],
            u_cache=bad[2],
            g_cache=bad[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
        )


def test_dispatcher_routes_rubin_build(monkeypatch):
    """gated_delta_rule_mtp_ucache_flush (fp16-state arm, backend "auto"): 4-D buffers run the Rubin
    build on sm_107a and the base UMMA kernel elsewhere; uGPU buffers always run the Rubin build.
    Same bits as the Rubin build called directly."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    mod = _base._flashinfer_fp16_state()
    calls = []

    def spy(name, real):
        def wrapped(*args, **kw):
            calls.append(name)
            return real(*args, **kw)

        return wrapped

    for name in ("_UMMA", "_UMMA_RUBIN"):
        monkeypatch.setattr(mod, name, spy(name, getattr(mod, name)))

    def run(case, fm, ugpu):
        q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
            x.clone() for x in case
        ]
        bufs = [pool, kc, uc, gc]
        if ugpu:
            bufs = [_ugpu_split(x, 2, "parts") for x in bufs]
        out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)  # padded rows
        y = mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            output=out,
        )
        torch.cuda.synchronize()
        assert y is out
        if ugpu:
            bufs = [_ugpu_logical(x) for x in bufs]
        return [_pz_bits(x) for x in [y] + bufs]

    case, fm = _pz_case(4, 123, _PZ_PATTERNS[0])
    ref = _pz_call(case, fm, heads_per_cta=None)
    on_rubin = get_compute_capability(torch.device("cuda")) == (10, 7)
    for ugpu, want in (
        (False, "_UMMA_RUBIN" if on_rubin else "_UMMA"),
        (True, "_UMMA_RUBIN"),
    ):
        calls.clear()
        got = run(case, fm, ugpu)
        assert calls == [want], (ugpu, calls)
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), (ugpu, name)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    with pytest.raises(ValueError, match="uGPU buffers"):
        mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=_ugpu_split(pool.clone(), 2, "view"),
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            restart_hist_on_flush=False,
            backend="hmma",
        )


def test_dispatcher_ugpu_cursor_checks():
    """gated_delta_rule_mtp_ucache_flush checks the ring cursors of a standalone call
    (restart_hist_on_flush=True) before handing uGPU buffers to the Rubin build, as for 4-D buffers:
    an out-of-range hist_len raises instead of reaching the kernel."""
    _skip_if_no_rubin_build()
    mod = _base._flashinfer_fp16_state()
    case, fm = _pz_case(4, 131, "FVVPF")
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    bufs = [_ugpu_split(x, 2, "parts") for x in (pool, kc, uc, gc)]
    hl[1] = W + 4  # beyond the 16-slot window
    for ugpu in (False, True):
        b4 = bufs if ugpu else [pool, kc, uc, gc]
        with pytest.raises(AssertionError, match="hist_len out of legal range"):
            mod.gated_delta_rule_mtp_ucache_flush(
                A_log,
                a,
                dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=b4[0],
                initial_state_indices=idx,
                k_cache=b4[1],
                u_cache=b4[2],
                g_cache=b4[3],
                hist_len=hl,
                cache_base=cb,
                scale=SCALE,
                flush_min=fm,
                restart_hist_on_flush=True,
            )
