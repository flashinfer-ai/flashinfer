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
Perf bench for the GDN ucache verify+flush kernels: the HMMA kernel's dtype
arms (``--arm bf16 | fp16_state | fp16_io | ring_fp16 | fp16_state_cache``,
backend pinned to HMMA) and the UMMA (tcgen05 / TMEM) backend of the
fp16_state arm (``--arm umma``, plus ``umma_ph10`` / ``umma_ph5`` / ``umma_lcg``
with stochastic rounding of the flushed state).

Prints one row per batch size, one column per flush rate. Steady-state
operating points: at ``--T 4`` flush_min = 13, verify rows at P = 12, flush
rows at P = 13 scattered at exact counts; ``--T 8`` uses the lazy
flush_min = 17 - T = 9 (verify rows at P = 8, flush rows at P = 9). The
closure is captured as a CUDA graph and benched on the replay (CUPTI, cold
L2): eager calls carry ~25 us of host launch overhead that serving (always
graph-captured) never pays.

Cold-L2 method (``--l2``):
  zero (default): ``bench_gpu_time(cold_l2_cache=True)``, which flushes with
      ``buffer.zero_()``. That leaves L2 full of DIRTY lines, and the timed
      kernel pays their write-back to DRAM -- several us on short launches.
  read: L2 is flushed by a read-only pass over a 2x-L2 buffer (clean lines),
      then the kernel's own CUPTI record is taken (exactly one kernel per
      call is asserted). Cold L2 without the flush's write-back in the
      measurement.

Anchors (B200, median of 1000, 2026-07-19, ``--l2 zero``): B=32/20% ~= 32 us,
B=256/20% ~= 163 us, B=256/0% ~= 134 us. Regressions >5% are real.

Run:
  python benchmarks/bench_gdn_ucache_flush.py --arm umma --T 4 --l2 read \
      --no-commit --batches 512 --rates 0 30 --iters 200
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch

from flashinfer.testing import bench_gpu_time

DEV = "cuda"
H, HV, K, V = 16, 64, 128, 128  # Qwen3.5-122B GDN @ TP1
T, W = 4, 16  # W = max history window (kernel W_RING); T is set by --T
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
    # UMMA (tcgen05 / TMEM) drop-in for the fp16_state arm: same tensors and contract
    # (flashinfer/gdn_kernels/gdn_replay_mtp_umma.py)
    "umma": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    # ... with stochastic rounding of the flushed state (fixed seed; the bits do not
    # change the work): Philox 5 / 10 rounds, LCG
    "umma_ph5": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "umma_ph10": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
    "umma_lcg": (None, "fp16", None, torch.bfloat16, torch.float16, torch.bfloat16),
}
_UMMA_SR = {
    "umma": {},
    "umma_ph5": dict(stochastic_rounding="philox", philox_rounds=5),
    "umma_ph10": dict(stochastic_rounding="philox", philox_rounds=10),
    "umma_lcg": dict(stochastic_rounding="lcg"),
}
# history length of the non-flushing (verify) rows: None = flush_min - 1 (--hist full)
P_VERIFY = None
# --row-order: which request each CTA runs (umma arms only; the kernel reads the permutation)
ROW_ORDER = "none"


def make_row_order(hl_src, flush_min, mode):
    """Row permutation for ``mode``: identity (same work, exercises the kernel's row_order
    load), flush_first (longest CTAs first), verify_tail (the last two waves of CTAs hold no
    flushing row: their flush rows are swapped with verify rows from the front)."""
    B = hl_src.numel()
    order = torch.arange(B, dtype=torch.int64, device=hl_src.device)
    if mode == "identity":
        return order.to(torch.int32)
    is_flush = hl_src >= flush_min
    if mode == "flush_first":
        return torch.argsort((~is_flush).to(torch.int32), stable=True).to(torch.int32)
    assert mode == "verify_tail", mode
    slots = torch.cuda.get_device_properties(hl_src.device).multi_processor_count * 4
    tail = min(B, 2 * -(-slots // H))  # two waves of CTAs, in rows (H CTAs per row)
    tail_pos = order[B - tail :]
    tail_flush = tail_pos[is_flush[tail_pos]]
    head_pos = order[: B - tail]
    head_verify = head_pos[~is_flush[head_pos]][: tail_flush.numel()]
    n = head_verify.numel()
    if n:
        a, b = tail_flush[:n].clone(), head_verify.clone()
        order[a], order[b] = b, a
    return order.to(torch.int32)


_FLUSH_PATH = str(
    Path(__file__).resolve().parents[1]
    / "flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py"
)


def load_flush(arm):
    io_env, state_env, ring_env, io_dtype, state_dtype, ring_dtype = ARMS[arm]
    if arm in _UMMA_SR:
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
            gated_delta_rule_mtp_ucache_flush_umma,
        )

        fn = gated_delta_rule_mtp_ucache_flush_umma
        if _UMMA_SR[arm]:
            seed = torch.tensor([0x5EED], dtype=torch.int64, device=DEV)
            fn = functools.partial(fn, rand_seed=seed, **_UMMA_SR[arm])
        return fn, io_dtype, state_dtype, ring_dtype
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
        spec = importlib.util.spec_from_file_location(f"uc_flush_{arm}", _FLUSH_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # the dtype arms time the HMMA kernel (on SM100 the fp16_state arm would
    # otherwise auto-route to the UMMA backend, which has its own arms above)
    fn = functools.partial(mod.gated_delta_rule_mtp_ucache_flush, backend="hmma")
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
):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return (torch.randn(*s, generator=g, device=DEV) * sc).to(io_dtype)

    q, k = rn(B, T, H, K), rn(B, T, H, K)
    v, a, b = rn(B, T, HV, V, sc=0.5), rn(B, T, HV, sc=0.5), rn(B, T, HV)
    A_log = (
        torch.full((HV,), -3.0, device=DEV)
        + torch.rand(HV, generator=g, device=DEV) * 0.3
    ).to(io_dtype)
    dt_bias = rn(HV, sc=0.5)
    pool = (torch.randn(B, HV, V, K, generator=g, device=DEV) * 0.5).to(state_dtype)
    # 32-deep physical rings, fully populated (rows outside the live window
    # are masked by the kernel; values just need to be finite).
    kh = torch.randn(B, H, RING, K, generator=g, device=DEV)
    kc = (kh / kh.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(ring_dtype)
    uc = (torch.randn(B, HV, RING, V, generator=g, device=DEV) * 0.3).to(ring_dtype)
    la = -(torch.rand(B, HV, RING, generator=g, device=DEV) * 0.3 + 0.003)
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
    l2="zero",
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
        torch.tensor(FLUSH_MIN, dtype=torch.int32, device=DEV),
        torch.tensor(
            FLUSH_MIN - 1 if P_VERIFY is None else P_VERIFY,
            dtype=torch.int32,
            device=DEV,
        ),
    )
    hl = hl_src.clone()
    cb_src = torch.full((B,), base, dtype=torch.int32, device=DEV)
    cb = cb_src.clone()
    extra = {}
    if ROW_ORDER != "none":
        extra["row_order"] = make_row_order(hl_src, FLUSH_MIN, ROW_ORDER)

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
            **extra,
        )
        if not no_commit:
            # wrapper committed cursors for flushed rows; restore them
            hl.copy_(hl_src)
            cb.copy_(cb_src)

    if l2 == "read":
        return kernel_us_read_flush(graphed(fn), iters)
    times = bench_gpu_time(
        graphed(fn),
        enable_cupti=True,
        cold_l2_cache=True,
        dry_run_iters=10,
        repeat_iters=iters,
    )
    return float(np.median(times)) * 1000.0  # us


_READ_FLUSH = {}


def kernel_us_read_flush(runner, iters):
    """Median kernel time (us) with a cold, CLEAN L2: before each call a read-only pass
    over a 2x-L2 buffer evicts everything (no dirty lines left behind), then the
    graph replay runs alone. The kernel's own CUPTI activity record is the time
    (torch.profiler), and exactly one kernel per call is asserted."""
    from torch.profiler import ProfilerActivity, profile

    from flashinfer.testing.utils import get_l2_cache_size

    if not _READ_FLUSH:
        _READ_FLUSH["buf"] = torch.ones(
            2 * get_l2_cache_size() // 4, dtype=torch.int32, device=DEV
        )
        _READ_FLUSH["out"] = torch.empty((), dtype=torch.int64, device=DEV)
    buf, out = _READ_FLUSH["buf"], _READ_FLUSH["out"]

    def gpu_events(prof):
        return [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]

    for _ in range(10):
        runner()
    torch.cuda.synchronize()
    # The call alone must be exactly one GPU kernel (no memsets / copies / extra launches).
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            runner()
        torch.cuda.synchronize()
    names = {e.name for e in gpu_events(prof)}
    assert len(gpu_events(prof)) == 5 and len(names) == 1, (
        f"expected one kernel per call, got {len(gpu_events(prof))} for 5 calls: {names}"
    )
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            torch.sum(
                buf, dim=0, out=out
            )  # read-only L2 flush (its kernels are dropped below)
            torch.cuda.synchronize()
            runner()
            torch.cuda.synchronize()
    kern = [e for e in gpu_events(prof) if e.name in names]
    assert len(kern) == iters, (len(kern), iters)
    return float(np.median([e.device_time for e in kern]))


def main():
    global T, FLUSH_MIN, P_VERIFY
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--T", type=int, default=4, help="draft tokens (flush_min = 17 - T)"
    )
    ap.add_argument(
        "--l2",
        choices=["zero", "read"],
        default="zero",
        help="cold-L2 method: zero = bench_gpu_time's zero_() flush; "
        "read = read-only flush + kernel-only CUPTI record (see module docstring)",
    )
    ap.add_argument(
        "--json", type=str, default=None, help="append rows to this JSON file"
    )
    ap.add_argument(
        "--hist",
        choices=["full", "mid", "zero"],
        default="full",
        help="history length of the non-flushing rows: full = flush_min - 1 (default), "
        "mid = (flush_min - 1) // 2, zero = 0",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--rates", type=int, nargs="+", default=[0, 20, 40, 80])
    ap.add_argument(
        "--arm",
        choices=list(ARMS),
        default="bf16",
        help="dtype config: bf16 | fp16_state | fp16_io | ring_fp16 | fp16_state_cache | umma (UMMA kernel, "
        "fp16 state) | umma_ph5 / umma_ph10 / umma_lcg (UMMA with stochastic rounding)",
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
    ap.add_argument(
        "--row-order",
        choices=["none", "identity", "flush_first", "verify_tail"],
        default="none",
        help="umma arms: which request each CTA runs (see make_row_order); identity "
        "isolates the cost of the kernel's row_order load from the ordering itself",
    )
    args = ap.parse_args()
    global ROW_ORDER
    ROW_ORDER = args.row_order
    T = args.T
    FLUSH_MIN = W - T + 1
    P_VERIFY = {"full": FLUSH_MIN - 1, "mid": (FLUSH_MIN - 1) // 2, "zero": 0}[
        args.hist
    ]

    uc_flush, io_dtype, state_dtype, ring_dtype = load_flush(args.arm)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | fused verify+flush, "
        f"arm={args.arm} (io={io_dtype}, state={state_dtype}, "
        f"ring={ring_dtype}), "
        f"T={T} W={W} ring={RING} base={args.base} fm={FLUSH_MIN} P_verify={P_VERIFY} "
        f"H={H} HV={HV} K=V={K} | "
        f"CUDA-graph replay, CUPTI cold-L2 ({args.l2} flush), median of {args.iters}",
        flush=True,
    )
    recs = []
    hdr = "   B | " + " | ".join(f"{r:3d}% (us)" for r in args.rates)
    print(hdr)
    print("-" * len(hdr))
    for B in args.batches:
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
                l2=args.l2,
            )
            for r in args.rates
        ]
        print(f"{B:4d} | " + " | ".join(f"{t:9.2f}" for t in row), flush=True)
        recs += [
            dict(
                arm=args.arm,
                T=T,
                B=B,
                rate=r,
                l2=args.l2,
                no_commit=args.no_commit,
                folded_rows=0 if r == 0 else max(1, round(B * r / 100)),
                hist=args.hist,
                p_verify=P_VERIFY,
                row_order=args.row_order,
                us=t,
            )
            for r, t in zip(args.rates, row, strict=True)
        ]
    if args.json:
        import json

        path = Path(args.json)
        prev = json.loads(path.read_text()) if path.exists() else []
        key = lambda d: (  # noqa: E731
            d["arm"],
            d["T"],
            d["B"],
            d["rate"],
            d["l2"],
            d["no_commit"],
            d.get("hist", "full"),
        )
        new = {key(d) for d in recs}
        path.write_text(
            json.dumps([d for d in prev if key(d) not in new] + recs, indent=1)
        )


if __name__ == "__main__":
    main()
