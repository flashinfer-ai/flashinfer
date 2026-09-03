"""Kernel-level latency of the low-precision Ulysses A2A primitives
(``flashinfer.comm.ulysses_lowp``) on one rank's shard.

Everything here is rank-local -- the collectives (stats AllGather, uint8
all-to-all) belong to the caller -- so a single GPU measures the full
quantize / pack / unpack cost a rank pays per attention layer, and the payload
bytes it would put on the wire versus the BF16 exchange it replaces.

Defaults reproduce the MiniMax-H3 deployment: 56 heads, D=128, Ulysses P=8,
shard L=4736 (37 x 128 -> stats protocol 3 fast path).  ``--local-sequence
4720`` exercises the protocol-2 shapes (split kernels, per-token unpack).

    python benchmarks/comm/bench_ulysses_lowp.py
    python benchmarks/comm/bench_ulysses_lowp.py --local-sequence 4720 --dtype float16
    python benchmarks/comm/bench_ulysses_lowp.py --json lowp_p8.json

Requires an SM120 GPU (the kernels are byte-anchored to sm_120a).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import flashinfer.comm.ulysses_lowp as lowp
from flashinfer.testing.utils import bench_gpu_time

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _median_us(fn, *, dry_run_iters: int, repeat_iters: int, cupti: bool) -> float:
    times_ms = bench_gpu_time(
        fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        enable_cupti=cupti,
        cold_l2_cache=False,
    )
    return float(np.median(times_ms)) * 1e3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-sequence", type=int, default=4736)
    p.add_argument("--num-heads", type=int, default=56)
    p.add_argument("--world-size", type=int, default=8, choices=(2, 4, 8))
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    p.add_argument("--dry-run-iters", type=int, default=10)
    p.add_argument("--repeat-iters", type=int, default=100)
    p.add_argument(
        "--cupti", action="store_true", help="time with CUPTI instead of CUDA events"
    )
    p.add_argument(
        "--json", type=str, default=None, help="write rows to this JSON file"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise SystemExit("ulysses_lowp requires an SM120 device")
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
    send = torch.empty((P, int(spec["chunk_bytes"])), dtype=torch.uint8, device=device)
    q_amax = lowp.q_grouped_amax(q, rank=r, world_size=P)
    k_amax = lowp.k_grouped_amax(k, k_mean, rank=r, world_size=P)
    lowp.quant_qkv_pack(
        q, k, v, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=P, out=send
    )
    # A receive buffer with a valid layout: every source sent this rank's chunk.
    recv = send[r : r + 1].expand(P, -1).contiguous()
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

    shard_bytes = q.numel() * q.element_size()  # one of Q/K/V
    cases = [
        ("k_sum_v_amax", lambda: lowp.k_sum_v_amax(k, v), 2 * shard_bytes),
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
            3 * shard_bytes + int(spec["chunk_bytes"]) * P,
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
            5 * shard_bytes + int(spec["chunk_bytes"]) * P,
        ),
        (
            "unpack_for_sage",
            lambda: lowp.unpack_for_sage(recv, out=unpack_out, **unpack_kwargs),
            int(spec["chunk_bytes"]) * P
            + sum(t.numel() * t.element_size() for t in unpack_out),
        ),
    ]
    if protocol == 3:
        cases[4:4] = [
            (
                "quant_qkv_pack_fused (amax+quant fused)",
                lambda: lowp.quant_qkv_pack_fused(
                    q, k, v, k_mean, v_scale, rank=r, world_size=P, out=send
                ),
                3 * shard_bytes + int(spec["chunk_bytes"]) * P,
            ),
            (
                "  Q half (quant_q_into_payload_fused)",
                lambda: lowp.quant_q_into_payload_fused(q, send, rank=r, world_size=P),
                shard_bytes,
            ),
            (
                "  K/V half (quant_kv_into_payload_fused)",
                lambda: lowp.quant_kv_into_payload_fused(
                    k, v, k_mean, v_scale, send, rank=r, world_size=P
                ),
                2 * shard_bytes,
            ),
        ]

    bf16_a2a_bytes = 3 * shard_bytes
    print(
        f"ulysses_lowp  L={L} H={H} D={D} P={P} rank={r} dtype={args.dtype}  "
        f"-> stats protocol {protocol} ({'fused fast path' if protocol == 3 else 'boundary machinery'})"
    )
    print(
        f"payload per rank: {P * int(spec['chunk_bytes']) / 2**20:.1f} MiB vs "
        f"{bf16_a2a_bytes / 2**20:.1f} MiB BF16 Q/K/V "
        f"(-{100 * (1 - P * int(spec['chunk_bytes']) / bf16_a2a_bytes):.1f}% on the wire)"
    )
    print(f"{'kernel / composite':46s} {'median us':>10s} {'GB/s moved':>11s}")
    rows = []
    for name, fn, moved in cases:
        us = _median_us(
            fn,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.repeat_iters,
            cupti=args.cupti,
        )
        gbps = moved / (us * 1e-6) / 1e9
        print(f"{name:46s} {us:10.1f} {gbps:11.0f}")
        rows.append(
            {"case": name.strip(), "median_us": us, "bytes_moved": moved, "gbps": gbps}
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "local_sequence": L,
                    "num_heads": H,
                    "head_dim": D,
                    "world_size": P,
                    "dtype": args.dtype,
                    "stats_protocol": protocol,
                    "chunk_bytes": int(spec["chunk_bytes"]),
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
