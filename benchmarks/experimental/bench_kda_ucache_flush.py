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
Flush-rate sweep for the KDA ucache verify+flush kernel
(the experimental ``flashinfer.kda_decode.kda_delta_rule_mtp_ucache_flush``).

Methodology matches bench_gdn_ucache_flush_w32.py: CUDA-graph replay,
pure-kernel timing (restart_hist_on_flush=False), CUPTI cold-L2 median (and
min) of ``--iters`` replays + a steady-state warm-L2 CUDA-event mean over
max(iters, 200) back-to-back replays. Unlocked clocks inflate cold-L2
absolutes on some hosts, so read the trend, not the absolutes.

Steady state across repeated calls: with restart_hist_on_flush=False the
kernel never writes hist_len / cache_base, so every replay sees the same
cursors: the T ring appends land on the same slots, folding rows fold again
(the pool is rewritten in place, the work is identical) and verify rows stay
verify rows. The fold set is therefore fixed for the whole measurement.

Fold rows (``--folds``, percent, fractional allowed, e.g. 6.25 = 1/16):
  n_flush = 0                                       if f == 0
          = clamp(floor(B * f / 100 + 0.5), 1, B)   otherwise
i.e. round-half-up of B*f/100, but at least one row for any f > 0. When
B*f/100 < 0.5 (e.g. B < 8 at 6.25 %) the one forced row makes the effective
rate 1/B > f; such points are tagged ``min1`` in the table and carry
``n_flush_forced_min: true`` in the JSON. Every point prints n_flush and the
effective rate n_flush/B. Fold rows get hist_len = flush_min; the others
hist_len ~ U[0, flush_min - 1] (verify path).

Row order (``--order``, applied identically to every backend by a host-side
batch permutation; the requests are identical across orders):
  random       the fold rows sit at random batch positions (seeded randperm).
  flush_first  the same requests (same hist_len, pool/ring slot, q/k/v/g/b)
               stably permuted so the fold rows occupy batch rows 0..n_flush-1
               and their CTAs launch first (cf. GDN flush_first_row_order).
               Only the per-row inputs (q, k, v, g, b, hist_len, cache_base)
               and initial_state_indices are permuted; each request keeps its
               pool/ring slot, so state memory is identical across orders and
               only the request -> CTA mapping changes.

Geometry: Kimi K3 at serving TP -- H == HV == 12, K == V == 128,
lower_bound = -5. ``--t`` in 1..8; flush_min = min(w - T + 1, 33 - 2T).

Backends: ``--backend hmma umma`` (default). Each arm is probed once per T;
an arm that rejects the configuration (NotImplementedError / ValueError, e.g.
HMMA outside T in {4, 8}) is reported as N/A instead of timed. Unless
``--no-check``, every supported arm first runs once eagerly on identical
inputs per (B, fold, order) and the run aborts if one drifts from the first
supported arm (y, folded pool rows, k/u/g ring appends); the flush_first
output is also checked against the random-order output un-permuted.
``spd`` = first supported arm's cold median / this arm's cold median.

``--json PATH`` writes every measured point (and N/A points with
cold_us = null) plus run metadata (GPU, timing method, iters, git sha).

Run:
  python benchmarks/experimental/bench_kda_ucache_flush.py [--iters 500] [--t 8] \
      [--backend hmma umma] [--w-ring 16] [--batches 8 32 64 128 256] \
      [--folds 0 6.25 13 25 50] [--order random flush_first] [--json out.json]
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import torch

from flashinfer.testing import bench_gpu_time

DEV = "cuda"
H, K, V = 12, 128, 128  # Kimi K3 KDA at serving TP (H == HV)
RING = 32
SCALE = 1.0 / math.sqrt(K)
LOWER_BOUND = -5.0


def load(backend="hmma"):
    from flashinfer.api_logging import ExperimentalWarning
    from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush

    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    return functools.partial(kda_delta_rule_mtp_ucache_flush, backend=backend)


torch.manual_seed(0)


def graphed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


def make_case(B, T, seed, hist):
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=gen, device=DEV) * sc).bfloat16()

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, b = rn(B, T, H, V, sc=0.5), rn(B, T, H)
    g = rn(B, T, H, K, sc=0.5)  # raw per-channel gate pre-activation
    A_log = (torch.rand(H, generator=gen, device=DEV) * 0.6 - 0.3).float()
    dt_bias = (torch.randn(H * K, generator=gen, device=DEV) * 0.5).float()
    pool = (torch.randn(B, H, V, K, generator=gen, device=DEV) * 0.5).bfloat16()
    kh = torch.randn(B, H, RING, K, generator=gen, device=DEV)
    kc = (kh / kh.norm(dim=-1, keepdim=True).clamp_min(1e-6)).bfloat16()
    uc = (torch.randn(B, H, RING, V, generator=gen, device=DEV) * 0.3).bfloat16()
    la = -(torch.rand(B, H, RING, K, generator=gen, device=DEV) * 0.3 + 0.003)
    gc = torch.cumsum(la, dim=2).float().contiguous()
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    hl = torch.tensor(hist, dtype=torch.int32, device=DEV)
    cb = torch.zeros(B, dtype=torch.int32, device=DEV)
    return q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, idx, hl, cb


def n_flush_rows(B, fold_pct):
    """Number of fold rows for a fold percentage (see the module docstring).
    Returns (n_flush, forced_min) where forced_min marks the at-least-one
    clamp having raised the count above round-half-up(B * f / 100)."""
    if fold_pct == 0:
        return 0, False
    nominal = math.floor(B * fold_pct / 100.0 + 0.5)
    n = min(max(1, nominal), B)
    return n, n > nominal


def hist_dist(B, seed, hist_max, n_flush, fold_hist):
    """n_flush rows at fold_hist (>= flush_min -> they fold), at random batch
    positions; the rest ~ U[0, hist_max] (verify path)."""
    g = torch.Generator().manual_seed(seed)
    hist = torch.randint(0, hist_max + 1, (B,), generator=g).tolist()
    for i in torch.randperm(B, generator=g)[:n_flush].tolist():
        hist[i] = fold_hist
    return hist


def order_perm(hist, flush_min, order):
    """Batch permutation (new row i <- request perm[i]) for a row order."""
    hl = torch.tensor(hist, dtype=torch.int32)
    if order == "random":
        return torch.arange(len(hist), dtype=torch.int64)
    assert order == "flush_first", order
    # stable: fold rows first, each group keeps its relative order
    return torch.argsort((hl < flush_min).to(torch.int32), stable=True)


def make_ordered_case(B, T, seed, hist, perm):
    """make_case() in the random layout, then permute the per-row inputs and
    point each row at its request's original pool/ring slot."""
    case = list(make_case(B, T, seed, hist))
    if torch.equal(perm, torch.arange(B)):
        return tuple(case)
    pd = perm.to(DEV)
    for j in (0, 1, 2, 3, 4, 12, 13):  # q, k, v, g, b, hist_len, cache_base
        case[j] = case[j][pd].contiguous()
    case[11] = case[11][pd].contiguous()  # initial_state_indices
    return tuple(case)


def _call(fn, case, flush_min, w_ring):
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, idx, hl, cb = case
    return fn(
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
        restart_hist_on_flush=False,
        w_ring=w_ring,
    )


# Cross-arm agreement bounds (the same bounds the tests use vs the oracle).
_CHECK_TOL = {"y": 8e-3, "pool": 2e-2, "k": 5e-3, "u": 2e-2, "g": 1e-4}


def run_eager(fn, flush_min, B, T, hist, seed, w_ring, perm):
    """One eager call; returns y (in the random/request order) and the
    request-indexed folded pool slots and ring append slots."""
    case = make_ordered_case(B, T, seed, hist, perm)
    y = _call(fn, case, flush_min, w_ring)
    torch.cuda.synchronize()
    y_req = torch.empty_like(y)
    y_req[perm.to(DEV)] = y  # back to request order
    pool, kc, uc, gc = case[7], case[8], case[9], case[10]
    # request r lives in slot r (make_case idx = arange) in every order
    hl = torch.tensor(hist, dtype=torch.int32, device=DEV)
    fold = hl >= flush_min
    # append slots (base 0): (P + s) & 31 for s < T
    app = (hl[:, None] + torch.arange(T, device=DEV)[None, :]) & (RING - 1)
    rows = torch.arange(B, device=DEV)[:, None]
    return {
        "y": y_req.float(),
        "pool": pool[fold].float(),
        "k": kc[rows, :, app].float(),
        "u": uc[rows, :, app].float(),
        "g": gc[rows, :, app].float(),
    }


def compare(ref, o, what):
    for key, tol in _CHECK_TOL.items():
        if ref[key].numel() == 0:
            continue
        err = (o[key] - ref[key]).abs().max().item()
        if not err < tol:  # also catches NaN
            raise SystemExit(
                f"refcheck FAILED: {what}: max|d{key}| = {err:.3e} >= {tol:g}"
            )


def check_arms(fns, names, flush_min, B, T, hist, seed, w_ring, perm, order, base):
    """Every arm on identical fresh inputs must match the first arm: output,
    folded pool rows, and the k/u/g ring append slots. ``base`` (the first
    arm's random-order result, or None) also pins flush_first to random."""
    outs = [run_eager(fn, flush_min, B, T, hist, seed, w_ring, perm) for fn in fns]
    tag = f"B={B} T={T} w_ring={w_ring} order={order}"
    for name, o in zip(names[1:], outs[1:], strict=True):
        compare(outs[0], o, f"backend {name} vs {names[0]} at {tag}")
    if base is not None:
        compare(base, outs[0], f"{names[0]} {order} vs random at {tag}")
    return outs[0]


def bench_point(fn, flush_min, B, T, case, iters, w_ring=16):
    def call():
        _call(fn, case, flush_min, w_ring)

    gr = graphed(call)
    times = bench_gpu_time(
        gr,
        enable_cupti=True,
        cold_l2_cache=True,
        dry_run_iters=10,
        repeat_iters=iters,
    )
    cold_med = float(np.median(times)) * 1000.0
    cold_min = float(np.min(times)) * 1000.0
    # steady-state warm-L2 CUDA events (clock-sag-proof cross-check)
    for _ in range(200):
        gr()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    n = max(iters, 200)
    s.record()
    for _ in range(n):
        gr()
    e.record()
    torch.cuda.synchronize()
    warm = s.elapsed_time(e) / n * 1000.0
    return {"cold_us": cold_med, "cold_min_us": cold_min, "warm_us": warm}


def probe(fn, T, flush_min, w_ring):
    """None if the arm runs this (T, w_ring) config, else the reason."""
    B = 2
    hist = [flush_min, 0]
    try:
        _call(fn, make_case(B, T, 7, hist), flush_min, w_ring)
        torch.cuda.synchronize()
    except (NotImplementedError, ValueError) as ex:
        return f"{type(ex).__name__}: {ex}"
    return None


def git_info():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument(
        "--t",
        type=int,
        default=8,
        choices=range(1, 9),
        help="draft length; arms that reject it are reported N/A",
    )
    ap.add_argument(
        "--folds",
        type=float,
        nargs="+",
        default=[0, 13, 25, 50],
        help="percent of rows that fold per step, each in [0, 100]; fractional "
        "allowed (6.25 = 1/16). n_flush = round-half-up(B*f/100), at least 1 if f > 0",
    )
    ap.add_argument(
        "--order",
        nargs="+",
        default=["random", "flush_first"],
        choices=["random", "flush_first"],
        help="batch placement of the fold rows (host-side permutation, all arms)",
    )
    ap.add_argument(
        "--backend",
        nargs="+",
        default=["hmma", "umma"],
        choices=["hmma", "umma", "auto"],
        help="kernel arms to time; the first supported one is the speed-up reference",
    )
    ap.add_argument("--w-ring", type=int, default=16, choices=[16, 32])
    ap.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cross-check every arm (and flush_first vs random) before timing",
    )
    ap.add_argument("--json", default=None, help="write every point to this path")
    args = ap.parse_args()
    bad = [p for p in args.folds if not 0 <= p <= 100]
    if bad:
        ap.error(f"--folds values must be in [0, 100]; got {bad}")
    if any(b < 1 for b in args.batches):
        ap.error("--batches values must be >= 1")
    orders = list(dict.fromkeys(args.order))

    names = list(args.backend)
    fns = [load(n) for n in names]
    T = args.t
    w_ring = args.w_ring
    # lazy flush: min(W - T + 1, RING - 2T + 1); 13/9 at w16, 25/17 at w32
    flush_min = min(w_ring - T + 1, RING - 2 * T + 1)
    hist_max = flush_min - 1
    gpu = torch.cuda.get_device_name(0)
    timing = (
        f"CUDA-graph replay, pure kernel (restart_hist_on_flush=False); cold: "
        f"bench_gpu_time CUPTI cold-L2 median/min of {args.iters} replays "
        f"(10 dry-run); warm: CUDA-event mean of {max(args.iters, 200)} "
        f"back-to-back replays after 200 warm-up replays"
    )
    print(
        f"GPU: {gpu} | KDA ucache flush | T={T} w_ring={w_ring} H=HV={H} "
        f"K=V={K} lb={LOWER_BOUND} flush_min={flush_min} | arms {names} | "
        f"orders {orders} | {timing}",
        flush=True,
    )

    unsupported = {}
    if w_ring == 32 and flush_min <= 16:
        unsupported = {n: f"w_ring=32 needs flush_min > 16 (T={T})" for n in names}
    else:
        for n, fn in zip(names, fns, strict=True):
            why = probe(fn, T, flush_min, w_ring)
            if why is not None:
                unsupported[n] = why
    for n, why in unsupported.items():
        print(f"  [N/A] backend {n} at T={T} w_ring={w_ring}: {why}", flush=True)
    live = [i for i, n in enumerate(names) if n not in unsupported]
    live_names = [names[i] for i in live]
    live_fns = [fns[i] for i in live]
    multi = len(names) > 1

    def fold_str(p):
        return f"{p:g}"

    hdr = f"{'B':>5} | {'fold%':>6} | {'n_fl':>4} {'eff%':>6} | {'order':>11}"
    for n in names:
        hdr += f" | {n + ' cold':>11} {'min':>8} {'warm':>8}"
        if multi and live_names and n != live_names[0]:
            hdr += f" {'spd':>6}"
    print(hdr)
    print("-" * len(hdr))

    points = []
    for B in args.batches:
        for p in args.folds:
            nf, forced = n_flush_rows(B, p)
            hist = hist_dist(B, 1000 + B, hist_max, nf, flush_min)
            assert sum(h >= flush_min for h in hist) == nf
            base = None
            for order in orders:
                perm = order_perm(hist, flush_min, order)
                if order == "flush_first" and nf:
                    hl_new = [hist[i] for i in perm.tolist()]
                    assert all(h >= flush_min for h in hl_new[:nf])
                    assert all(h < flush_min for h in hl_new[nf:])
                if args.check and live_fns:
                    r = check_arms(
                        live_fns,
                        live_names,
                        flush_min,
                        B,
                        T,
                        hist,
                        1000 + B,
                        w_ring,
                        perm,
                        order,
                        base if order != "random" else None,
                    )
                    if order == "random":
                        base = r
                eff = 100.0 * nf / B
                line = f"{B:5d} | {fold_str(p):>6} | {nf:4d} {eff:6.2f} | {order:>11}"
                cold0 = None
                for n, fn in zip(names, fns, strict=True):
                    rec = {
                        "T": T,
                        "w_ring": w_ring,
                        "B": B,
                        "H": H,
                        "fold_pct": p,
                        "n_flush_rows": nf,
                        "eff_fold_pct": eff,
                        "n_flush_forced_min": forced,
                        "flush_min": flush_min,
                        "order": order,
                        "backend": n,
                    }
                    if n in unsupported:
                        rec.update(
                            supported=False,
                            reason=unsupported[n],
                            cold_us=None,
                            cold_min_us=None,
                            warm_us=None,
                        )
                        line += f" | {'N/A':>11} {'':>8} {'':>8}"
                        if multi and live_names and n != live_names[0]:
                            line += f" {'':>6}"
                        points.append(rec)
                        continue
                    case = make_ordered_case(B, T, 1000 + B, hist, perm)
                    t = bench_point(fn, flush_min, B, T, case, args.iters, w_ring)
                    del case
                    rec.update(supported=True, **t)
                    line += (
                        f" | {t['cold_us']:11.2f} {t['cold_min_us']:8.2f} "
                        f"{t['warm_us']:8.2f}"
                    )
                    if cold0 is None:
                        cold0 = t["cold_us"]
                    elif multi:
                        rec["spd_vs_" + live_names[0]] = cold0 / t["cold_us"]
                        line += f" {cold0 / t['cold_us']:6.3f}"
                    points.append(rec)
                if forced:
                    line += "  min1"
                print(line, flush=True)

    if args.json:
        sha, dirty = git_info()
        meta = {
            "gpu": gpu,
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "timing_method": timing,
            "cold_us": "CUPTI cold-L2 median (us)",
            "cold_min_us": "CUPTI cold-L2 min (us)",
            "warm_us": "warm-L2 CUDA-event mean (us)",
            "iters": args.iters,
            "warm_iters": max(args.iters, 200),
            "git_sha": sha,
            "git_dirty": dirty,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "T": T,
            "w_ring": w_ring,
            "H": H,
            "K": K,
            "V": V,
            "lower_bound": LOWER_BOUND,
            "flush_min": flush_min,
            "backends": names,
            "unsupported": unsupported,
            "orders": orders,
            "check": args.check,
            "n_flush_rule": "0 if f == 0 else clamp(floor(B*f/100 + 0.5), 1, B)",
        }
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "points": points}, f, indent=1)
        print(f"wrote {len(points)} points to {args.json}", flush=True)


if __name__ == "__main__":
    main()
