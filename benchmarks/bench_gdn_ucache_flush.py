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
T, W = 4, 16
FLUSH_MIN = 13
SCALE = 1.0 / math.sqrt(K)

# --arm choices: dtype is fixed at module import (env-gated), so each arm
# loads its own copy of the flush module.
#   bf16       : bf16 inputs + bf16 state pool (default serving config)
#   fp16_state : bf16 inputs + fp16 state pool (GDN_UCACHE_STATE_DTYPE=fp16)
#   fp16_io    : fp16 inputs + fp16 state pool (GDN_UCACHE_IO_DTYPE=fp16)
ARMS = {
    "bf16": (None, None, torch.bfloat16, torch.bfloat16),
    "fp16_state": (None, "fp16", torch.bfloat16, torch.float16),
    "fp16_io": ("fp16", None, torch.float16, torch.float16),
}
_FLUSH_PATH = str(Path(__file__).resolve().parents[1]
                  / "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py")


def load_flush(arm):
    io_env, state_env, io_dtype, state_dtype = ARMS[arm]
    old = {k: os.environ.pop(k, None)
           for k in ("GDN_UCACHE_IO_DTYPE", "GDN_UCACHE_STATE_DTYPE")}
    if io_env:
        os.environ["GDN_UCACHE_IO_DTYPE"] = io_env
    if state_env:
        os.environ["GDN_UCACHE_STATE_DTYPE"] = state_env
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
    return mod.gated_delta_rule_mtp_ucache_flush, io_dtype, state_dtype


torch.manual_seed(0)


def graphed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


def make_case(B, seed, io_dtype=torch.bfloat16, state_dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io_dtype)

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, a, b = rn(B, T, HV, V, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (torch.full((HV,), -3.0, device=DEV)
             + torch.rand(HV, generator=g, device=DEV) * 0.3).to(io_dtype)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(state_dtype)
    kh = torch.randn(B, H, W, K, generator=g, device=DEV)
    kc = (kh / kh.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(io_dtype)
    uc = (torch.randn(B, HV, W, V, generator=g, device=DEV) * 0.3).to(io_dtype)
    la = -(torch.rand(B, HV, W, generator=g, device=DEV) * 0.3 + 0.003)
    gc = torch.cumsum(la, dim=-1).float().contiguous()
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    return q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, idx


def bench_point(uc_flush, B, rate_pct, iters, seed, io_dtype, state_dtype):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, idx = make_case(
        B, seed, io_dtype, state_dtype)
    nf = 0 if rate_pct == 0 else max(1, round(B * rate_pct / 100))
    mask = torch.zeros(B, dtype=torch.bool, device=DEV)
    if nf:
        g_cpu = torch.Generator().manual_seed(seed + 3)
        mask[torch.randperm(B, generator=g_cpu)[:nf].to(DEV)] = True
    hl_src = torch.where(mask,
                         torch.tensor(13, dtype=torch.int32, device=DEV),
                         torch.tensor(12, dtype=torch.int32, device=DEV))
    hl = hl_src.clone()

    def fn():
        uc_flush(A_log, a, dt_bias, q=q, k=k, v=v, b=b,
                 initial_state_source=pool, initial_state_indices=idx,
                 k_cache=kc, u_cache=uc, g_cache=gc, hist_len=hl,
                 scale=SCALE, flush_min=FLUSH_MIN)
        hl.copy_(hl_src)  # kernel zeroes flushed rows; restore for next iter

    times = bench_gpu_time(graphed(fn), enable_cupti=True, cold_l2_cache=True,
                           dry_run_iters=10, repeat_iters=iters)
    return float(np.median(times)) * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[8, 32, 64, 128, 256])
    ap.add_argument("--rates", type=int, nargs="+",
                    default=[0, 20, 40, 80])
    ap.add_argument("--arm", choices=list(ARMS), default="bf16",
                    help="dtype config: bf16 | fp16_state | fp16_io")
    args = ap.parse_args()

    uc_flush, io_dtype, state_dtype = load_flush(args.arm)
    print(f"GPU: {torch.cuda.get_device_name(0)} | fused verify+flush, "
          f"arm={args.arm} (io={io_dtype}, state={state_dtype}), "
          f"T={T} W={W} fm={FLUSH_MIN} H={H} HV={HV} K=V={K} | "
          f"CUDA-graph replay, CUPTI cold-L2, median of {args.iters}",
          flush=True)
    hdr = "   B | " + " | ".join(f"{r:3d}% (us)" for r in args.rates)
    print(hdr)
    print("-" * len(hdr))
    for B in args.batches:
        row = [bench_point(uc_flush, B, r, args.iters, 1000 + B + r,
                           io_dtype, state_dtype) for r in args.rates]
        print(f"{B:4d} | " + " | ".join(f"{t:9.2f}" for t in row), flush=True)


if __name__ == "__main__":
    main()
