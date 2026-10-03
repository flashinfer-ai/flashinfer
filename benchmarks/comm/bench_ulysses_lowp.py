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

Rank-local general Ulysses timing on SM90 or SM89/SM120.
Collectives are excluded: peer descriptors are prepared before timing.
All lengths use BOUNDARY_MERGE; --hot-l2 controls cache flushing.
Use --zero-v to measure zero-amax protection with identical shapes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import flashinfer.comm._ulysses_lowp as lowp
from flashinfer.testing.utils import bench_gpu_time

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _timing_backend(force_cuda_events: bool) -> str:
    if force_cuda_events:
        return "cuda_event"
    try:
        import cupti  # noqa: F401

        return "cupti"
    except ImportError:
        return "cuda_event"


def _median_us(fn, *, args: argparse.Namespace, backend: str) -> float:
    times_ms = bench_gpu_time(
        fn,
        dry_run_iters=args.dry_run_iters,
        repeat_iters=args.repeat_iters,
        enable_cupti=(backend == "cupti"),
        cold_l2_cache=not args.hot_l2,
    )
    return float(np.median(times_ms)) * 1e3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    p.add_argument("--local-sequence", type=int, default=4736)
    p.add_argument("--num-heads", type=int, default=56)
    p.add_argument("--world-size", type=int, default=8, choices=(2, 4, 8))
    p.add_argument(
        "--rank",
        type=int,
        default=None,
        help="rank whose shard is packed (default: world_size // 2)",
    )
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    p.add_argument("--dry-run-iters", type=int, default=10)
    p.add_argument("--repeat-iters", type=int, default=100)
    p.add_argument(
        "--hot-l2",
        action="store_true",
        help="do not flush L2 between iterations (inputs stay cache-resident)",
    )
    p.add_argument(
        "--cuda-events",
        action="store_true",
        help="time with CUDA events even if CUPTI is installed",
    )
    p.add_argument(
        "--json", type=str, default=None, help="write rows to this JSON file"
    )
    p.add_argument("--zero-v", action="store_true")
    args = p.parse_args()
    if args.rank is None:
        args.rank = args.world_size // 2
    if not 0 <= args.rank < args.world_size:
        p.error(f"--rank must lie in [0, {args.world_size})")
    if args.num_heads % args.world_size:
        p.error("--num-heads must be divisible by --world-size")
    return args


def main() -> None:
    args = _parse_args()
    cap = lowp.capability("cuda")
    if not cap["supported"] or cap["layout_class"] is None:
        raise SystemExit(f"unsupported layout: {cap}")
    layout = getattr(lowp, cap["layout_class"])(head_dim=args.head_dim)
    L, H, P, r = args.local_sequence, args.num_heads, args.world_size, args.rank
    dtype = _DTYPES[args.dtype]
    torch.manual_seed(0)
    q = torch.randn((1, L, H, args.head_dim), device="cuda", dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    if args.zero_v:
        v.zero_()
    # Equal peer values still need rank-specific boundary descriptors.
    prepared = [layout.local_stats(q, k, v, rank=i, world_size=P) for i in range(P)]
    gathered = torch.stack([x[0] for x in prepared])
    ctx = prepared[r][1]
    stats = layout.finalize_stats(gathered, ctx, k)
    spec = layout.payload_spec(
        batch_size=1, local_sequence=L, num_heads=H, world_size=P
    )
    send = layout.quant_and_pack(q, k, v, stats)
    recv = torch.stack(
        [
            layout.quant_and_pack(q, k, v, layout.finalize_stats(gathered, c, k))[r]
            for _, c in prepared
        ]
    )
    kwargs = dict(batch_size=1, local_sequence=L, local_heads=H // P, world_size=P)
    unpacked = layout.unpack_for_sage(recv, **kwargs)

    def chain():
        local_send, local_ctx = layout.local_stats(q, k, v, rank=r, world_size=P)
        gathered[r].copy_(local_send)
        final = layout.finalize_stats(gathered, local_ctx, k)
        return layout.quant_and_pack(q, k, v, final, out=send)

    cases = [
        ("local_stats", lambda: layout.local_stats(q, k, v, rank=r, world_size=P)),
        ("finalize_stats", lambda: layout.finalize_stats(gathered, ctx, k)),
        ("quant_and_pack", lambda: layout.quant_and_pack(q, k, v, stats, out=send)),
        ("general_chain_without_collectives", chain),
        (
            "general_unpack",
            lambda: layout.unpack_for_sage(recv, out=unpacked, **kwargs),
        ),
    ]
    backend = _timing_backend(args.cuda_events)
    rows = []
    for name, fn in cases:
        us = _median_us(fn, args=args, backend=backend)
        rows.append(dict(case=name, median_us=us))
        print(f"{name:36s} {us:10.2f} us")
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        layout=cap["layout_class"],
        stats_protocol=layout.stats_protocol_for(L, P),
        scale_max=2.25,
        local_sequence=L,
        num_heads=H,
        head_dim=args.head_dim,
        world_size=P,
        dtype=args.dtype,
        zero_v=args.zero_v,
        timing_backend=backend,
        cold_l2_cache=not args.hot_l2,
        payload_bytes=spec["payload_bytes"],
        stats_bytes=prepared[r][0].numel() * 4,
        rows=rows,
    )
    if args.json:
        with open(args.json, "x") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
