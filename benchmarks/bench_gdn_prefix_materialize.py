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
Benchmark the GDN prefix-cache materialization kernel.

The headline number is NOT peak throughput. Materialization runs roughly once
per prefix-cache block per request (say once per 128 tokens), so what matters
in serving is the cost of a launch where the batch is padded to a fixed size
but only a handful of rows actually need work.

The ``active=0`` row of the active-request sweep measures that directly: the
kernel runs with nobody needing work, so every microsecond it takes is pure
scheduling overhead. The kernel's persistent grid + device-side ``num_active``
trip count make this ~2.5 us at B=256/HV=64 on B200; the same launch cost 32 us
under the naive one-CTA-per-item grid it replaced, so treat regressions of this
row as scheduling breakage, not noise.

Methodology follows PR #4815's mamba materialize benchmark: active requests are
SPREAD through the batch rather than front-loaded (front-loading flatters a
schedule that finds its work early), the sweep includes the zero-active case,
and the sweep's replay length defaults to 1 so launch cost is not hidden behind
arithmetic.

Examples:
    # Default: Qwen3.5-122B GDN at TP1, CUPTI timing.
    python benchmarks/bench_gdn_prefix_materialize.py

    # Tensor-parallel shard shapes.
    python benchmarks/bench_gdn_prefix_materialize.py --tp 4

    # CUDA-graph replay timing -- how serving actually runs.
    python benchmarks/bench_gdn_prefix_materialize.py --timing graph
"""

import argparse
import statistics

import cuda.bindings.driver as cuda
import torch
import torch.nn.functional as F
from cutlass.cute.runtime import from_dlpack

import flashinfer.gdn_kernels.gdn_prefix_materialize as _mat
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush import W_RING
from flashinfer.gdn_kernels.gdn_prefix_materialize import (
    gdn_prefix_materialize,
    state_ty_torch,
)
from flashinfer.testing import bench_gpu_time

DEV = "cuda"
K = V = 128

# (H, HV) per tensor-parallel shard, from bench_gdn_prefill.py's HEAD_CONFIGS.
# Qwen3.5-397B and 122B share this family (h_k=16, h_v=64 at TP1); K == V == 128
# throughout, which is what this kernel is fixed at.
TP_HEADS = {1: (16, 64), 2: (8, 32), 4: (4, 16), 8: (2, 8)}

# Set from --tp in main(); build() and bytes_moved() read them.
H, HV = TP_HEADS[1]
# 32 = MTP/spec-decode ring, 16 = STP ring; set from --ring-slots.
RING_SLOTS = 32


def build(batch, seed=0):
    """Pool with `batch` sources in [0, B) and `batch` destinations in [B, 2B)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    dt = state_ty_torch()
    n = 2 * batch
    state = (torch.randn(n, HV, V, K, generator=g, device=DEV) * 0.5).to(dt)
    kc = torch.zeros(n, H, RING_SLOTS, K, dtype=dt, device=DEV)
    uc = torch.zeros(n, HV, RING_SLOTS, V, dtype=dt, device=DEV)
    gc = torch.zeros(n, HV, RING_SLOTS, dtype=torch.float32, device=DEV)
    P = W_RING - 1
    kh = torch.randn(batch, H, P, K, generator=g, device=DEV)
    kc[:batch, :, :P] = F.normalize(kh, dim=-1).to(dt)
    uc[:batch, :, :P] = (
        torch.randn(batch, HV, P, V, generator=g, device=DEV) * 0.3
    ).to(dt)
    la = -(torch.rand(batch, HV, P, generator=g, device=DEV) * 0.3 + 0.003)
    gc[:batch, :, :P] = torch.cumsum(la, dim=-1)
    src = torch.arange(batch, dtype=torch.int32, device=DEV)
    dst = src + batch
    base = torch.zeros(batch, dtype=torch.int32, device=DEV)
    return state, kc, uc, gc, src, dst, base


def bytes_moved(n_active, count):
    """Minimum DRAM traffic for `n_active` requests replaying `count` entries.

    Per (request, value head): read the source state and write the destination
    state (V*K elements each), plus `count` ring rows of u (V) and g (1 f32),
    plus `count` rows of k (K) shared across the HV/H heads of a k-group.
    """
    esz = torch.finfo(state_ty_torch()).bits // 8
    per_hv = 2 * V * K * esz + count * V * esz + count * 4
    per_h = count * K * esz
    return n_active * (HV * per_hv + H * per_h)


def compiled_call(state, src, dst, kc, uc, gc, base, cnt, min_blocks_per_mp=8):
    """Bind the compiled kernel to pre-built descriptors.

    The public wrapper rebuilds a DLPack descriptor per tensor on every call.
    That host work is ~40 us and shows up as GPU idle inside the timed region,
    which would swamp a kernel this small. Hoisting it out is what the decode
    benchmarks do via CUDA-graph capture; doing it directly here keeps the
    measurement about the kernel.
    """
    active = torch.full_like(cnt, -1)
    live = torch.nonzero(cnt >= 0, as_tuple=False).flatten().to(torch.int32)
    active[: live.numel()] = live
    n_act = torch.tensor([live.numel()], dtype=torch.int32, device=cnt.device)
    gdn_prefix_materialize(
        state,
        src,
        dst,
        kc,
        uc,
        gc,
        base,
        cnt,
        active,
        n_act,
        min_blocks_per_mp=min_blocks_per_mp,
    )
    torch.cuda.synchronize()

    def mk_dyn(t):
        return from_dlpack(t, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=tuple(range(t.dim())), divisibility=1
        )

    args = [
        mk_dyn(state),
        mk_dyn(src),
        mk_dyn(dst),
        mk_dyn(kc),
        mk_dyn(uc),
        mk_dyn(gc),
        mk_dyn(base),
        mk_dyn(cnt),
        mk_dyn(active),
        mk_dyn(n_act),
        HV,
        V,
        H,
        int(kc.shape[2]),
        min(
            cnt.shape[0] * HV,
            torch.cuda.get_device_properties(state.device).multi_processor_count * 8,
        ),
    ]
    # Reconstruct the module's cache key rather than taking the most recently
    # inserted entry: sweeping several --tp shapes in one process leaves
    # multiple entries, and insertion order does not track the last CALL.
    key = (
        str(state.device),
        int(HV),
        int(H),
        int(V),
        int(kc.shape[2]),
        int(min_blocks_per_mp),
        str(state.dtype),
        str(kc.dtype),
    )
    fn = _mat._CACHE[key]

    # Read the CURRENT stream at call time, not at build time. Under
    # --timing graph the harness captures on a side stream; a stream bound
    # here would launch outside the capture, the graph would record nothing,
    # and replaying that empty graph reports a fictitious ~0.7 us for every
    # configuration (caught by the 1745 TB/s it implied).
    dev = state.device

    def call():
        fn(*args, cuda.CUstream(torch.cuda.current_stream(device=dev).cuda_stream))

    return call


def active_positions(batch, n_active):
    """Spread the active requests evenly through the batch.

    Deliberately NOT the first n_active slots. Front-loading rewards an
    implementation that happens to find its work early and bail, which is
    exactly the property the persistent rework will add -- measuring it against
    a front-loaded batch would flatter it. Same placement rule as PR #4815's
    benchmark.
    """
    if n_active <= 0:
        return []
    return [(2 * i + 1) * batch // (2 * n_active) for i in range(n_active)]


def counts_for(batch, arm, n_active=4):
    P = W_RING - 1
    if arm == "full":
        return torch.full((batch,), P, dtype=torch.int32, device=DEV)
    if arm == "copy":
        return torch.zeros(batch, dtype=torch.int32, device=DEV)
    if arm == "sparse":
        # A realistic serving moment: the batch is padded to B, but only
        # n_active requests just crossed a prefix-cache block boundary.
        c = torch.full((batch,), -1, dtype=torch.int32, device=DEV)
        for pos in active_positions(batch, min(n_active, batch)):
            c[pos] = P
        return c
    raise ValueError(arm)


def timing_kwargs(args):
    """Map --timing onto bench_gpu_time's backend selection.

    cupti  hardware profiler: kernel execution time, excluding launch bookkeeping
    event  CUDA events on the stream: wall time, so launch cost is included
    graph  capture and replay: how serving actually runs, where launches are cheap

    The three disagree for a kernel this small, which is the point of offering
    all of them -- a launch-bound kernel looks free under cupti and expensive
    under event, and neither is what serving sees.
    """
    kw = {
        "dry_run_iters": args.warmup,
        "repeat_iters": args.iters,
        "cold_l2_cache": args.cold_l2,
    }
    if args.timing == "cupti":
        kw["enable_cupti"] = True
    elif args.timing == "graph":
        kw["use_cuda_graph"] = True
        kw["num_iters_within_graph"] = args.graph_iters
    return kw


def bench(call, args):
    return statistics.median(bench_gpu_time(call, **timing_kwargs(args))) * 1e3


def sweep_active(batch, args, replay=1):
    """Cost vs how many requests are active, at a fixed batch size.

    active=0 is the pure no-op launch: every CTA starts, reads a sentinel and
    retires. That is the overhead measured directly, with nothing subtracted.
    """
    print(f"\nactive-request sweep at B={batch}, replay={replay}")
    print(f"{'active':>7} | {'% of B':>7} | {'us':>8}")
    print("-" * 30)
    state, kc, uc, gc, src, dst, base = build(batch)
    for n in (0, 1, 2, 4, 8, 32, batch):
        cnt = torch.full((batch,), -1, dtype=torch.int32, device=DEV)
        for pos in active_positions(batch, n):
            cnt[pos] = replay
        call = compiled_call(state, src, dst, kc, uc, gc, base, cnt)
        print(f"{n:>7} | {100.0 * n / batch:>6.1f}% | {bench(call, args):>8.2f}")


def sweep_counts(batch, args):
    """Cost vs replay depth, with every request active."""
    print(f"\nreplay-depth sweep at B={batch} (all requests active)")
    print(f"{'count':>6} | {'us':>8} | {'GB':>7} | {'TB/s':>7}")
    print("-" * 36)
    state, kc, uc, gc, src, dst, base = build(batch)
    for c in (0, 1, 2, 4, 8, 12, 15):
        cnt = torch.full((batch,), c, dtype=torch.int32, device=DEV)
        call = compiled_call(state, src, dst, kc, uc, gc, base, cnt)
        us = bench(call, args)
        gb = bytes_moved(batch, c) / 1e9
        print(f"{c:>6} | {us:>8.2f} | {gb:>7.3f} | {gb / (us * 1e-6) / 1e3:>7.2f}")


def main():
    global H, HV, RING_SLOTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default="8,32,64,128,256")
    ap.add_argument(
        "--tp",
        type=int,
        default=1,
        choices=sorted(TP_HEADS),
        help="tensor-parallel shard: selects (H, HV) for the Qwen3.5-397B/122B family",
    )
    ap.add_argument("--n-active", type=int, default=4)
    ap.add_argument(
        "--ring-slots",
        type=int,
        default=32,
        choices=(16, 32),
        help="ring depth: 32 = MTP/spec-decode ring, 16 = STP ring",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--graph-iters", type=int, default=10)
    ap.add_argument(
        "--timing",
        choices=("cupti", "event", "graph"),
        default="cupti",
        help="CUPTI kernel timing, CUDA events (includes launch), or graph replay",
    )
    ap.add_argument(
        "--cold-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flush L2 between iterations (measured to be a no-op for this kernel)",
    )
    args = ap.parse_args()
    H, HV = TP_HEADS[args.tp]
    RING_SLOTS = args.ring_slots
    batches = [int(x) for x in args.batches.split(",")]

    print(f"GDN prefix materialize (persistent grid) — {torch.cuda.get_device_name()}")
    print(
        f"Qwen3.5-397B/122B GDN @ TP{args.tp}: H={H} HV={HV} K=V={K} "
        f"ring={RING_SLOTS} replay={W_RING - 1}"
    )
    print(f"timing={args.timing}  cold_l2={args.cold_l2}")
    print(f"sparse arm: {args.n_active} active requests, spread through B\n")
    print("legend:")
    print("  B        batch SLOTS (fixed by the CUDA-graph shape, not the live count)")
    print("  full     all B requests replay the full window -- saturated, no idle rows")
    print("  copy     all B requests have count=0 (checkpoint copy, no ring loads).")
    print("           Same code path as full, just without the u/k ring reads.")
    print(f"  sparse   B slots, but only {args.n_active} requests crossed a block")
    print("           boundary; the rest are skip sentinels. The realistic case.")
    print(f"  ideal    the same {args.n_active} requests with the batch sized to")
    print("           them instead of padded to B -- the floor sparse could reach")
    print("  pad tax  sparse - ideal. Near zero by construction with the")
    print("           persistent schedule; see the active=0 row below for the same")
    print("           overhead measured directly, with nothing subtracted.\n")
    print(
        f"{'B':>5} | {'full us':>9} | {'copy us':>9} | {'sparse us':>10} | "
        f"{'ideal us':>10} | {'pad tax':>8} | {'full TB/s':>9}"
    )
    print("-" * 82)

    for batch in batches:
        row = {}
        for arm in ("full", "copy", "sparse"):
            state, kc, uc, gc, src, dst, base = build(batch)
            cnt = counts_for(batch, arm, args.n_active)
            row[arm] = bench(
                compiled_call(state, src, dst, kc, uc, gc, base, cnt), args
            )

        nb = min(args.n_active, batch)
        state, kc, uc, gc, src, dst, base = build(nb)
        cnt = counts_for(nb, "full")
        row["ideal"] = bench(
            compiled_call(state, src, dst, kc, uc, gc, base, cnt), args
        )

        gb = bytes_moved(batch, W_RING - 1) / 1e9
        bw = gb / (row["full"] * 1e-6) / 1e3
        tax = row["sparse"] - row["ideal"]
        print(
            f"{batch:>5} | {row['full']:>9.2f} | {row['copy']:>9.2f} | "
            f"{row['sparse']:>10.2f} | {row['ideal']:>10.2f} | {tax:>7.2f} | {bw:>9.2f}"
        )

    sweep_active(max(batches), args)
    sweep_counts(max(batches), args)


if __name__ == "__main__":
    main()
