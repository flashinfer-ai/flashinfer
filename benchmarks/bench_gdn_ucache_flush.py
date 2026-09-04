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
Simple perf bench for the GDN ucache verify+flush kernel (fused scheme).

Prints one row per batch size, one column per flush rate. Methodology
matches BENCHMARK.md's scheme sweep: T=4, flush_min=13, verify rows at
P=12, flush rows at P=13 scattered at exact counts, the closure captured
as a CUDA graph and benched on the replay (CUPTI, cold L2). Timing under
graph replay matters: eager calls carry ~25 us of host launch overhead
that serving (always graph-captured) never pays.

Anchors (B200, median of 1000, 2026-07-19): B=32/20% ~= 32 us,
B=256/20% ~= 163 us, B=256/0% ~= 134 us. Regressions >5% are real.

Run:
  source env.sh && python benchmarks/bench_gdn_ucache_flush.py [--iters 200]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch

from flashinfer.testing import bench_gpu_time

DEV = "cuda"
H, HV, K, V = 16, 64, 128, 128  # Qwen3.5-122B GDN @ TP1
T, W = 4, 16  # W = max history window (kernel W_RING)
RING = 32  # physical ring depth (kernel RING_SLOTS)
FLUSH_MIN = 13
SCALE = 1.0 / math.sqrt(K)

# --arm choices: dtype is fixed at module import (env-gated), so each arm
# loads its own copy of the flush module.
#   bf16       : bf16 inputs + bf16 state pool (default serving config)
#   fp16_state : bf16 inputs + fp16 state pool (GDN_UCACHE_STATE_DTYPE=fp16)
#   fp16_io    : fp16 inputs + fp16 state pool (GDN_UCACHE_IO_DTYPE=fp16)
#   ring_fp16  : bf16 inputs + bf16 state + fp16 u/k rings
#                (GDN_UCACHE_RING_DTYPE=fp16 — the vLLM/Triton ring rule)
# tuple: (io_env, state_env, ring_env, io_dtype, state_dtype, ring_dtype)
ARMS = {
    "bf16": (None, None, None, torch.bfloat16, torch.bfloat16, torch.bfloat16),
    "fp16_state": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "fp16_io": ("fp16", None, None, torch.float16, torch.float16, torch.float16),
    "ring_fp16": (None, None, "fp16", torch.bfloat16, torch.bfloat16, torch.float16),
    "fp16_state_cache": (
        None,
        "fp16",
        "fp16",
        torch.bfloat16,
        torch.float16,
        torch.float16,
    ),
}
_KDIR = Path(__file__).resolve().parents[1] / "flashinfer/gdn_kernels"
_FLUSH_PATH = str(_KDIR / "gdn_decode_bf16_wy_ucache_flush.py")
# STP (T=1) fold-absorb fork: flat 16-slot buffer, fm=15, single-row appends
_STP_PATH = str(_KDIR / "gdn_decode_bf16_wy_ucache_stp.py")
STP_RING = 16
STP_FM = 15


def load_flush(arm, path=_FLUSH_PATH):
    io_env, state_env, ring_env, io_dtype, state_dtype, ring_dtype = ARMS[arm]
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
    fn = (
        mod.gated_delta_rule_stp_ucache_flush
        if path == _STP_PATH
        else mod.gated_delta_rule_mtp_ucache_flush
    )
    return fn, io_dtype, state_dtype, ring_dtype


torch.manual_seed(0)


def graphed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


def make_case(
    B,
    seed,
    io_dtype=torch.bfloat16,
    state_dtype=torch.bfloat16,
    ring_dtype=torch.bfloat16,
    t_tokens=T,
    ring_slots=RING,
):
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
    # 32-deep physical rings, fully populated (rows outside the live window
    # are masked by the kernel; values just need to be finite).
    kh = torch.randn(B, H, ring_slots, K, generator=g, device=DEV)
    kc = (kh / kh.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(ring_dtype)
    uc = (torch.randn(B, HV, ring_slots, V, generator=g, device=DEV) * 0.3).to(
        ring_dtype
    )
    la = -(torch.rand(B, HV, ring_slots, generator=g, device=DEV) * 0.3 + 0.003)
    gc = torch.cumsum(la, dim=-1).float().contiguous()
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, idx


def bench_point(
    uc_flush,
    B,
    rate_pct,
    iters,
    seed,
    io_dtype,
    state_dtype,
    ring_dtype=torch.bfloat16,
    base=0,
    no_commit=False,
):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, idx = make_case(
        B, seed, io_dtype, state_dtype, ring_dtype
    )
    nf = 0 if rate_pct == 0 else max(1, round(B * rate_pct / 100))
    mask = torch.zeros(B, dtype=torch.bool, device=DEV)
    if nf:
        g_cpu = torch.Generator().manual_seed(seed + 3)
        mask[torch.randperm(B, generator=g_cpu)[:nf].to(DEV)] = True
    hl_src = torch.where(
        mask,
        torch.tensor(13, dtype=torch.int32, device=DEV),
        torch.tensor(12, dtype=torch.int32, device=DEV),
    )
    hl = hl_src.clone()
    cb_src = torch.full((B,), base, dtype=torch.int32, device=DEV)
    cb = cb_src.clone()

    def fn():
        uc_flush(
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
            flush_min=FLUSH_MIN,
            restart_hist_on_flush=not no_commit,
        )
        if not no_commit:
            # wrapper committed cursors for flushed rows; restore them
            hl.copy_(hl_src)
            cb.copy_(cb_src)

    times = bench_gpu_time(
        graphed(fn),
        enable_cupti=True,
        cold_l2_cache=True,
        dry_run_iters=10,
        repeat_iters=iters,
    )
    return float(np.median(times)) * 1000.0  # us


def bench_point_stp(
    uc_stp,
    B,
    rate_pct,
    iters,
    seed,
    io_dtype,
    state_dtype,
    ring_dtype=torch.bfloat16,
):
    """One (batch, flush-rate) point for the STP (T=1) fold-absorb kernel.

    Steady-state operating points: verify rows at hist_len = STP_FM - 1 (the
    deepest replay), flush rows at STP_FM (fold absorbs the current token).
    Inputs use the wrapper's zero-copy ``prepadded`` [B, 4, ...] contract
    (row 0 real, rows 1..3 zero — the producer-writes-row-0 serving
    pattern), so the graph carries no staging nodes. The wrapper always
    self-commits cursors; the per-iter restore keeps every replay at the
    same pre-call state, like bench_point's commit/restore.
    """
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, idx = make_case(
        B,
        seed,
        io_dtype,
        state_dtype,
        ring_dtype,
        t_tokens=1,
        ring_slots=STP_RING,
    )

    def pad4(t):
        buf = torch.zeros(
            (t.shape[0], 4) + tuple(t.shape[2:]), dtype=t.dtype, device=DEV
        )
        buf[:, 0] = t[:, 0]
        return buf

    q, k, v, a, b = pad4(q), pad4(k), pad4(v), pad4(a), pad4(b)
    nf = 0 if rate_pct == 0 else max(1, round(B * rate_pct / 100))
    mask = torch.zeros(B, dtype=torch.bool, device=DEV)
    if nf:
        g_cpu = torch.Generator().manual_seed(seed + 3)
        mask[torch.randperm(B, generator=g_cpu)[:nf].to(DEV)] = True
    hl_src = torch.where(
        mask,
        torch.tensor(STP_FM, dtype=torch.int32, device=DEV),
        torch.tensor(STP_FM - 1, dtype=torch.int32, device=DEV),
    )
    hl = hl_src.clone()
    cb = torch.zeros(B, dtype=torch.int32, device=DEV)

    def fn():
        uc_stp(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            prepadded=True,
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
        hl.copy_(hl_src)

    times = bench_gpu_time(
        graphed(fn),
        enable_cupti=True,
        cold_l2_cache=True,
        dry_run_iters=10,
        repeat_iters=iters,
    )
    return float(np.median(times)) * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--rates", type=int, nargs="+", default=[0, 20, 40, 80])
    ap.add_argument(
        "--arm",
        choices=list(ARMS),
        default="bf16",
        help="dtype config: bf16 | fp16_state | fp16_io",
    )
    ap.add_argument(
        "--kernel",
        choices=["mtp", "stp"],
        default="mtp",
        help="mtp = the T=4/8 verify+flush kernel (default, unchanged); "
        "stp = the T=1 fold-absorb kernel "
        "(gdn_decode_bf16_wy_ucache_stp.py; flat 16-slot buffer, fm=15)",
    )
    ap.add_argument(
        "--no-commit",
        action="store_true",
        help="pure-kernel timing: disable the wrapper's standalone "
        "cursor commit AND the per-iter cursor restores (the "
        "kernel never mutates cursors, so iterations are "
        "identical). Without this flag the timed graph also "
        "contains ~4-6us of commit/restore elementwise ops — "
        "fine for wrapper-level A/Bs, misleading for "
        "kernel-level ones.",
    )
    ap.add_argument(
        "--base",
        type=int,
        default=0,
        help="ring window origin for all rows (28 exercises the "
        "wrapped-window path: base+P crosses RING_SLOTS)",
    )
    args = ap.parse_args()

    stp = args.kernel == "stp"
    uc_flush, io_dtype, state_dtype, ring_dtype = load_flush(
        args.arm, _STP_PATH if stp else _FLUSH_PATH
    )
    geom = (
        f"T=1 ring={STP_RING} (flat) fm={STP_FM}"
        if stp
        else f"T={T} W={W} ring={RING} base={args.base} fm={FLUSH_MIN}"
    )
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | "
        f"{'STP fold-absorb' if stp else 'fused verify+flush'}, "
        f"arm={args.arm} (io={io_dtype}, state={state_dtype}, "
        f"ring={ring_dtype}), "
        f"{geom} "
        f"H={H} HV={HV} K=V={K} | "
        f"CUDA-graph replay, CUPTI cold-L2, median of {args.iters}",
        flush=True,
    )
    hdr = "   B | " + " | ".join(f"{r:3d}% (us)" for r in args.rates)
    print(hdr)
    print("-" * len(hdr))
    for B in args.batches:
        if stp:
            row = [
                bench_point_stp(
                    uc_flush,
                    B,
                    r,
                    args.iters,
                    1000 + B + r,
                    io_dtype,
                    state_dtype,
                    ring_dtype,
                )
                for r in args.rates
            ]
        else:
            row = [
                bench_point(
                    uc_flush,
                    B,
                    r,
                    args.iters,
                    1000 + B + r,
                    io_dtype,
                    state_dtype,
                    ring_dtype,
                    base=args.base,
                    no_commit=args.no_commit,
                )
                for r in args.rates
            ]
        print(f"{B:4d} | " + " | ".join(f"{t:9.2f}" for t in row), flush=True)


if __name__ == "__main__":
    main()
