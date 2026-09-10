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
(flashinfer/kda_kernels/kda_decode_bf16_wy_ucache_flush.py).

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

Run:
  python benchmarks/bench_kda_ucache_flush.py [--iters 500] [--t 8]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from flashinfer.testing import bench_gpu_time

DEV = "cuda"
H, K, V = 12, 128, 128  # Kimi K3 KDA at serving TP (H == HV)
RING = 32
SCALE = 1.0 / math.sqrt(K)
LOWER_BOUND = -5.0

_KPATH = (
    Path(__file__).resolve().parents[1]
    / "flashinfer/kda_kernels/kda_decode_bf16_wy_ucache_flush.py"
)


def load():
    spec = importlib.util.spec_from_file_location("kda_uc_bench", str(_KPATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kda_delta_rule_mtp_ucache_flush


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


def bench_point(fn, flush_min, B, T, hist, iters, seed):
    q, k, v, g, b, A_log, dt_bias, pool, kc, uc, gc, idx, hl, cb = make_case(
        B, T, seed, hist
    )

    def call():
        fn(
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
            restart_hist_on_flush=False,  # pure kernel: no cursor-commit ops
        )

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
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 13, 25, 50])
    args = ap.parse_args()

    fn = load()
    T = args.t
    flush_min = 16 - T + 1  # W_RING - T + 1 (lazy flush)
    hist_max = flush_min - 1
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | KDA ucache flush | T={T} "
        f"H=HV={H} K=V={K} lb={LOWER_BOUND} flush_min={flush_min} | CUDA-graph "
        f"replay, pure kernel (no commit), CUPTI cold-L2 median of "
        f"{args.iters} + steady-state warm events",
        flush=True,
    )

    hdr = "   B | " + " | ".join(
        f"f{p:02d}% cold(us) | f{p:02d}% warm(us)" for p in args.folds
    )
    print(hdr)
    print("-" * len(hdr))
    for B in args.batches:
        cells = []
        for p in args.folds:
            hist = hist_dist(B, 1000 + B, hist_max, p, flush_min)
            c, w = bench_point(fn, flush_min, B, T, hist, args.iters, 1000 + B)
            cells.append(f"{c:13.2f} | {w:13.2f}")
        print(f"{B:4d} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
