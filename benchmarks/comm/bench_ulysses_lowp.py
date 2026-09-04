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

Kernel-level latency of the low-precision Ulysses A2A primitives
(``flashinfer.comm.ulysses_lowp``) on one rank's shard.

Everything here is rank-local -- the collectives (stats AllGather, uint8
all-to-all) belong to the caller -- so a single GPU measures the quantize /
pack / unpack work a rank does per attention layer, and the payload bytes it
would put on the wire versus the BF16 exchange it replaces.  The "routed
chain" row runs the protocol-routed API (local_stats -> finalize_stats ->
quant_and_pack) with the AllGather replaced by a local stack, so both
protocols report their true rank-local cost (protocol 2 includes the boundary
machinery).

Defaults reproduce the MiniMax-H3 deployment: 56 heads, D=128, Ulysses P=8,
shard L=4736 (37 x 128 -> stats protocol 3, fused fast path).
``--local-sequence 4720`` exercises the protocol-2 shapes (split kernels,
boundary machinery, per-token unpack).

Timing flushes L2 before every iteration (cold_l2_cache) because the kernels
are memory-bound and one Q/K/V shard (~65 MiB) fits in a 96 MiB L2: hot-L2
numbers overstate bandwidth ~2.5x for the single-tensor kernels.  Pass
``--hot-l2`` to measure the L2-resident case.  CUPTI is used when installed,
else CUDA events (``--cuda-events`` forces events).

    python benchmarks/comm/bench_ulysses_lowp.py
    python benchmarks/comm/bench_ulysses_lowp.py --local-sequence 4720 --dtype float16
    python benchmarks/comm/bench_ulysses_lowp.py --world-size 4 --json lowp_p4.json

Requires an SM120 GPU (the kernels are byte-anchored to sm_120a).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import flashinfer.comm.ulysses_lowp as lowp
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
    cap = lowp.capability("cuda") if torch.cuda.is_available() else {"supported": False}
    if not cap.get("supported"):
        raise SystemExit(
            f"ulysses_lowp needs an SM120 device with payload ABI {lowp.ABI_VERSION}; "
            f"capability = {cap}"
        )
    backend = _timing_backend(args.cuda_events)
    dtype = _DTYPES[args.dtype]
    L, H, D, P, r = (
        args.local_sequence,
        args.num_heads,
        lowp.HEAD_DIM,
        args.world_size,
        args.rank,
    )
    protocol = lowp.stats_protocol_for(L, P)
    device = torch.device("cuda")

    torch.manual_seed(0)
    q = torch.randn((1, L, H, D), device=device, dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    # Global statistics a rank would hold after the AllGather (single-rank stand-in).
    k_mean = k.float().mean(dim=1).to(dtype).contiguous()
    v_scale = (v.float().abs().amax(dim=1) / lowp.V_SCALE_MAX).contiguous()
    spec = lowp.payload_spec(
        batch_size=1, local_sequence=L, num_heads=H, head_dim=D, world_size=P
    )
    chunk_bytes = int(spec["chunk_bytes"])
    main_bytes = int(spec["main_bytes"])
    send = torch.empty((P, chunk_bytes), dtype=torch.uint8, device=device)
    q_amax = lowp.q_grouped_amax(q, rank=r, world_size=P)
    k_amax = lowp.k_grouped_amax(k, k_mean, rank=r, world_size=P)
    split_payload = lowp.quant_qkv_pack(
        q, k, v, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=P
    )
    if protocol == 3:
        fused_payload = lowp.quant_qkv_pack_fused(
            q, k, v, k_mean, v_scale, rank=r, world_size=P
        )
        if not torch.equal(fused_payload, split_payload):
            raise SystemExit(
                "fused and split packs disagree; refusing to time a broken kernel"
            )

    # Routed chain with the AllGather replaced by a local stack of this rank's
    # statistics (every peer "sent" the same shard).
    def routed_chain() -> torch.Tensor:
        stats_send, ctx = lowp.local_stats(q, k, v, rank=r, world_size=P)
        gathered = stats_send.unsqueeze(0).expand(P, -1).contiguous()
        stats = lowp.finalize_stats(gathered, ctx, k)
        return lowp.quant_and_pack(q, k, v, stats, out=send)

    routed_chain()

    # A receive buffer with a valid layout: every source sent this rank's chunk.
    recv = split_payload[r : r + 1].expand(P, -1).contiguous()
    local_heads = H // P
    unpack_kwargs = dict(
        batch_size=1,
        local_sequence=L,
        local_heads=local_heads,
        head_dim=D,
        world_size=P,
        aligned=None,
    )
    unpack_out = lowp.unpack_for_sage(recv, **unpack_kwargs)
    unpack_out_bytes = sum(t.numel() * t.element_size() for t in unpack_out)

    shard_bytes = q.numel() * q.element_size()  # one of Q/K/V
    payload_bytes = P * chunk_bytes
    stats_bytes = 2 * H * D * 4  # k_sum + v_amax fp32
    # (name, fn, bytes read + written)
    cases = [
        (
            "k_sum_v_amax",
            lambda: lowp.k_sum_v_amax(k, v),
            2 * shard_bytes + stats_bytes,
        ),
        (
            "q_grouped_amax",
            lambda: lowp.q_grouped_amax(q, rank=r, world_size=P),
            shard_bytes,
        ),
        (
            "k_grouped_amax",
            lambda: lowp.k_grouped_amax(k, k_mean, rank=r, world_size=P),
            shard_bytes,
        ),
        (
            "quant_qkv_pack (split, final amax given)",
            lambda: lowp.quant_qkv_pack(
                q, k, v, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=P, out=send
            ),
            3 * shard_bytes + payload_bytes,
        ),
        (
            "split path total (Q amax + K amax + pack)",
            lambda: lowp.quant_qkv_pack(
                q,
                k,
                v,
                k_mean,
                lowp.q_grouped_amax(q, rank=r, world_size=P),
                lowp.k_grouped_amax(k, k_mean, rank=r, world_size=P),
                v_scale,
                rank=r,
                world_size=P,
                out=send,
            ),
            5 * shard_bytes + payload_bytes,
        ),
    ]
    if protocol == 3:
        cases += [
            (
                "quant_qkv_pack_fused (amax+quant fused)",
                lambda: lowp.quant_qkv_pack_fused(
                    q, k, v, k_mean, v_scale, rank=r, world_size=P, out=send
                ),
                3 * shard_bytes + payload_bytes,
            ),
            (
                "  Q half (quant_q_into_payload_fused)",
                lambda: lowp.quant_q_into_payload_fused(q, send, rank=r, world_size=P),
                shard_bytes + P * main_bytes,
            ),
            (
                "  K/V half (quant_kv_into_payload_fused)",
                lambda: lowp.quant_kv_into_payload_fused(
                    k, v, k_mean, v_scale, send, rank=r, world_size=P
                ),
                2 * shard_bytes + 2 * P * main_bytes,
            ),
        ]
    cases += [
        (
            f"routed chain (local_stats+finalize+quant_and_pack, protocol {protocol})",
            routed_chain,
            (5 if protocol == 3 else 6) * shard_bytes + payload_bytes,
        ),
        (
            f"unpack_for_sage ({'aligned' if protocol == 3 else 'unaligned'} kernel)",
            lambda: lowp.unpack_for_sage(recv, out=unpack_out, **unpack_kwargs),
            payload_bytes + unpack_out_bytes,
        ),
    ]

    bf16_a2a_bytes = 3 * shard_bytes
    gpu_name = torch.cuda.get_device_name(device)
    print(
        f"ulysses_lowp  L={L} H={H} D={D} P={P} rank={r} dtype={args.dtype}  "
        f"-> stats protocol {protocol} "
        f"({'fused fast path' if protocol == 3 else 'boundary machinery'})"
    )
    print(
        f"gpu={gpu_name}  timing={backend}  L2={'hot' if args.hot_l2 else 'cold (flushed per iteration)'}  "
        f"iters={args.dry_run_iters}+{args.repeat_iters}"
    )
    print(
        f"payload per rank: {payload_bytes / 2**20:.1f} MiB vs "
        f"{bf16_a2a_bytes / 2**20:.1f} MiB {args.dtype} Q/K/V "
        f"(-{100 * (1 - payload_bytes / bf16_a2a_bytes):.1f}% on the wire)"
    )
    if protocol != 3:
        print(
            "fused rows skipped: quant_qkv_pack_fused needs local_sequence % 128 == 0"
        )
    print(f"{'kernel / composite':66s} {'median us':>10s} {'GB/s moved':>11s}")
    rows = []
    for name, fn, moved in cases:
        us = _median_us(fn, args=args, backend=backend)
        gbps = moved / (us * 1e-6) / 1e9
        print(f"{name:66s} {us:10.1f} {gbps:11.0f}")
        rows.append(
            {"case": name.strip(), "median_us": us, "bytes_moved": moved, "gbps": gbps}
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "gpu": gpu_name,
                    "torch": torch.__version__,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "timing_backend": backend,
                    "cold_l2_cache": not args.hot_l2,
                    "dry_run_iters": args.dry_run_iters,
                    "repeat_iters": args.repeat_iters,
                    "local_sequence": L,
                    "num_heads": H,
                    "head_dim": D,
                    "world_size": P,
                    "rank": r,
                    "dtype": args.dtype,
                    "stats_protocol": protocol,
                    "chunk_bytes": chunk_bytes,
                    "payload_bytes": payload_bytes,
                    "bf16_a2a_bytes": bf16_a2a_bytes,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
