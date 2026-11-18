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

One table per fold rate (methodology matches bench_gdn_ucache_flush_w32.py:
CUDA-graph replay, pure-kernel timing i.e. restart_hist_on_flush=False,
CUPTI cold-L2 median + a steady-state warm-L2 CUDA-event column — unlocked
clocks inflate cold-L2 absolutes on some hosts, so read the trend, not the
absolutes):

  fold 0%   — hist ~ U[0,12], nothing folds: the every-step verify cost.
  fold 13%  — the deep-window steady state (accept ~3.5 tok/step crossing a
              25-threshold; here it maps to how often a 13-threshold ring
              would fold with a deeper commit policy).
  fold 25%  — the W16 steady state at accept ~3.5 tok/step (a request
              crosses flush_min=13 every ~3.7 steps).
  fold 50%  — stress: half the batch folds every step.

Geometry: Kimi K3 at serving TP — H == HV == 12, K == V == 128,
lower_bound = -5. T in {4, 8} via --t (default 8 — Kimi K3's draft length).

Backends: ``--backend hmma umma`` (default) times each arm per point and,
unless ``--no-check``, first runs every arm once on identical inputs and
aborts if an arm drifts from the first one (y, folded pool rows, k/u/g ring
appends). ``spd`` = first arm's cold time / this arm's cold time.
``--w-ring 32`` benches the deep window (flush_min = min(w - T + 1, 33 - 2T)).

Run:
  python benchmarks/experimental/bench_kda_ucache_flush.py [--iters 500] [--t 8] \
      [--backend hmma umma] [--w-ring 16]
"""

from __future__ import annotations

import argparse
import functools
import math
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


def hist_dist(B, seed, hist_max, fold_pct, fold_hist):
    """fold_pct% of rows at fold_hist (>= flush_min -> they fold), the rest
    ~ U[0, hist_max] (verify path). Exact per-row fold count."""
    g = torch.Generator().manual_seed(seed)
    hist = torch.randint(0, hist_max + 1, (B,), generator=g).tolist()
    nf = 0 if fold_pct == 0 else max(1, round(B * fold_pct / 100))
    for i in torch.randperm(B, generator=g)[:nf].tolist():
        hist[i] = fold_hist
    return hist


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


def check_arms(fns, names, flush_min, B, T, hist, seed, w_ring):
    """One eager call per arm on identical fresh inputs; every arm must match
    the first: output, folded pool rows, and the k/u/g ring append slots."""
    hl = torch.tensor(hist, dtype=torch.int32, device=DEV)
    fold = hl >= flush_min
    # append slots (base 0): (P + s) & 31 for s < T
    app = (hl[:, None] + torch.arange(T, device=DEV)[None, :]) & (RING - 1)
    rows = torch.arange(B, device=DEV)[:, None]
    outs = []
    for fn in fns:
        case = make_case(B, T, seed, hist)
        y = _call(fn, case, flush_min, w_ring)
        torch.cuda.synchronize()
        pool, kc, uc, gc = case[7], case[8], case[9], case[10]
        outs.append(
            {
                "y": y.float(),
                "pool": pool[fold].float(),
                "k": kc[rows, :, app].float(),
                "u": uc[rows, :, app].float(),
                "g": gc[rows, :, app].float(),
            }
        )
    ref = outs[0]
    for name, o in zip(names[1:], outs[1:], strict=True):
        for key, tol in _CHECK_TOL.items():
            if ref[key].numel() == 0:
                continue
            err = (o[key] - ref[key]).abs().max().item()
            if not err < tol:  # also catches NaN
                raise SystemExit(
                    f"refcheck FAILED: backend {name} vs {names[0]} at B={B} T={T} "
                    f"w_ring={w_ring}: max|d{key}| = {err:.3e} >= {tol:g}"
                )


def bench_point(fn, flush_min, B, T, hist, iters, seed, w_ring=16):
    case = make_case(B, T, seed, hist)

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
    cold = float(np.median(times)) * 1000.0
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
    return cold, warm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--t", type=int, default=8, choices=[4, 8])
    ap.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 13, 25, 50],
        help="percent of rows that fold per step, each in [0, 100]",
    )
    ap.add_argument(
        "--backend",
        nargs="+",
        default=["hmma", "umma"],
        choices=["hmma", "umma", "auto"],
        help="kernel arms to time; the first one is the speed-up reference",
    )
    ap.add_argument("--w-ring", type=int, default=16, choices=[16, 32])
    ap.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cross-check every arm against the first before timing",
    )
    args = ap.parse_args()
    bad = [p for p in args.folds if not 0 <= p <= 100]
    if bad:
        ap.error(f"--folds values must be in [0, 100]; got {bad}")

    names = list(args.backend)
    fns = [load(n) for n in names]
    T = args.t
    w_ring = args.w_ring
    # lazy flush: min(W - T + 1, RING - 2T + 1); 13/9 at w16, 25/17 at w32
    flush_min = min(w_ring - T + 1, RING - 2 * T + 1)
    hist_max = flush_min - 1
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | KDA ucache flush | T={T} "
        f"w_ring={w_ring} H=HV={H} K=V={K} lb={LOWER_BOUND} "
        f"flush_min={flush_min} | arms {names} | CUDA-graph replay, pure "
        f"kernel (no commit), CUPTI cold-L2 median of {args.iters} + "
        f"steady-state warm events",
        flush=True,
    )

    multi = len(names) > 1

    def cell_hdr(p, n):
        tag = f"f{p:02d}% {n}" if multi else f"f{p:02d}%"
        h = f"{tag} cold(us) | {tag} warm(us)"
        return h + (f" | {tag} spd" if multi and n != names[0] else "")

    hdr = "   B | " + " | ".join(cell_hdr(p, n) for p in args.folds for n in names)
    print(hdr)
    print("-" * len(hdr))
    for B in args.batches:
        cells = []
        for p in args.folds:
            hist = hist_dist(B, 1000 + B, hist_max, p, flush_min)
            if args.check and multi:
                check_arms(fns, names, flush_min, B, T, hist, 1000 + B, w_ring)
            cold0 = None
            for fn in fns:
                c, w = bench_point(
                    fn, flush_min, B, T, hist, args.iters, 1000 + B, w_ring
                )
                cell = f"{c:13.2f} | {w:13.2f}"
                if cold0 is None:
                    cold0 = c
                elif multi:
                    cell += f" | {cold0 / c:6.3f}"
                cells.append(cell)
        print(f"{B:4d} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
