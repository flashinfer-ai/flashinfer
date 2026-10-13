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

Benchmark of the fused QKV projection + norm + RoPE + head exchange against the
per-operator paths, with cold-L2 CUPTI timing.

Run on a single NVLink node (two, four or eight GPUs of one architecture):

    torchrun --standalone --nproc-per-node=8 \
        benchmarks/comm/bench_qkv_norm_rope_exchange_overlap.py --rows representative --json out.json

Arms (every arm starts from the same ``x``, ``w_qkv``, norm weights and cos/sin
table and ends with this rank's head band of the whole sequence):

- ``fused``: :class:`flashinfer.comm.QkvNormRopeExchangeOverlap` (Cake backend);
- ``b1_ordinary``: cuBLAS projection, ``rmsnorm`` per head, RoPE, then three
  ``UlyssesCommunicator.scatter_heads`` collectives (Q, K, V);
- ``b2_whole_qkv``: the same operators, then one
  ``scatter_qkv_head_chunk`` collective over the whole head band;
- ``b3_head_chunk``: the same operators, then one ``scatter_qkv_head_chunk``
  per head band of the default head-chunk schedule;
- ``b4_nccl``: the same operators, then
  ``parallel_wrapper.ulysses_a2a_in(fuse_qkv=True)`` (NCCL all-to-all).

The Ulysses communicators of B1-B3 use ``backend="auto"`` (the NVLink kernels on
SM100 / SM103, NCCL elsewhere); the JSON records the backend they resolved.
Timing uses ``flashinfer.testing.bench_gpu_time_with_cupti`` with a cold L2
cache and the per-rank maximum of every sample; CUPTI unavailability is an
error.  Every arm is checked once against the fused arm before it is timed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flashinfer.comm import QkvNormRopeExchangeOverlap, UlyssesCommunicator
from flashinfer.norm import rmsnorm
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from flashinfer.testing import bench_gpu_time_with_cupti

HEAD_DIM = 128
EPS = 1e-6
HIDDENS = (3072, 4096, 6144, 8192)
S_GLOBALS = (4096, 8192, 16384, 32768, 65536, 131072)
REPRESENTATIVE = ((4096, 16384), (8192, 16384), (4096, 65536), (8192, 65536))


def default_schedule(local_heads: int) -> tuple[int, ...]:
    """Default head-chunk schedule of ``benchmarks/comm/bench_ulysses_head_chunk.py``."""

    if local_heads == 14:
        return (2, 10, 2)
    if local_heads == 28:
        return (7, 14, 7)
    left = max(1, local_heads // 4)
    middle = local_heads - 2 * left
    return (left, middle, left) if middle > 0 else (local_heads,)


def operands(device, *, rows, hidden, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (1, rows, hidden), dtype=torch.float32, device=device, generator=generator
    ).to(torch.bfloat16)
    w_qkv = (
        torch.randn(
            (3 * hidden, hidden),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * hidden**-0.5
    ).to(torch.bfloat16)
    q_scale = (
        1.0
        + 0.1
        * torch.randn(
            (HEAD_DIM,), dtype=torch.float32, device=device, generator=generator
        )
    ).to(torch.bfloat16)
    k_scale = (
        1.0
        + 0.1
        * torch.randn(
            (HEAD_DIM,), dtype=torch.float32, device=device, generator=generator
        )
    ).to(torch.bfloat16)
    positions = torch.arange(rows, device=device, dtype=torch.float64)
    omega = 1.0 / (
        10000.0
        ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float64) / HEAD_DIM)
    )
    angle = positions[:, None] * omega[None, :]
    cos_sin = (
        torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1).float().contiguous()
    )
    return x, w_qkv, q_scale, k_scale, cos_sin


class OperatorProducer:
    """The per-operator producer shared by B1-B4: cuBLAS linear, per-head ``rmsnorm``, in-place RoPE."""

    def __init__(self, x, w_qkv, q_scale, k_scale, cos_sin):
        self.x, self.w_qkv, self.q_scale, self.k_scale, self.cos_sin = (
            x,
            w_qkv,
            q_scale,
            k_scale,
            cos_sin,
        )
        self.rows, self.hidden = x.shape[0] * x.shape[1], x.shape[2]
        self.heads = self.hidden // HEAD_DIM
        shape = (self.rows, self.heads, HEAD_DIM)
        self.q_norm = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
        self.k_norm = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
        self.positions = torch.arange(self.rows, device=x.device, dtype=torch.int32)

    def __call__(self):
        rows, hidden, heads = self.rows, self.hidden, self.heads
        qkv = functional.linear(self.x.reshape(rows, hidden), self.w_qkv)
        q = rmsnorm(
            qkv[:, :hidden].view(rows, heads, HEAD_DIM),
            self.q_scale,
            eps=EPS,
            out=self.q_norm,
        )
        k = rmsnorm(
            qkv[:, hidden : 2 * hidden].view(rows, heads, HEAD_DIM),
            self.k_scale,
            eps=EPS,
            out=self.k_norm,
        )
        apply_rope_with_cos_sin_cache_inplace(
            self.positions,
            q.view(rows, heads * HEAD_DIM),
            k.view(rows, heads * HEAD_DIM),
            HEAD_DIM,
            self.cos_sin,
            is_neox=False,
        )
        local = (1, rows, heads, HEAD_DIM)
        return q.view(local), k.view(local), qkv[:, 2 * hidden :].view(local)


def build_arms(group, device, rank, world_size, hidden, rows, args):
    x, w_qkv, q_scale, k_scale, cos_sin = operands(
        device, rows=rows, hidden=hidden, seed=1190 + rank
    )
    heads = hidden // HEAD_DIM
    h_local = heads // world_size
    s_global = rows * world_size
    producer = OperatorProducer(x, w_qkv, q_scale, k_scale, cos_sin)
    session = QkvNormRopeExchangeOverlap(
        group,
        hidden=hidden,
        num_heads=heads,
        layout="bshd",
        max_tokens=rows,
        backend=args.backend,
        device=device,
    )
    comm_in = UlyssesCommunicator(
        group,
        max_elems=rows * 3 * hidden,
        dtype=torch.bfloat16,
        backend=args.ulysses_backend,
        device=device,
    )
    comm_ops = UlyssesCommunicator(
        group,
        max_elems=rows * hidden,
        dtype=torch.bfloat16,
        backend=args.ulysses_backend,
        device=device,
    )
    band = (1, s_global, h_local, HEAD_DIM)
    b1_out = tuple(
        torch.empty(band, dtype=torch.bfloat16, device=device) for _ in range(3)
    )
    payload = torch.empty(
        (1, s_global, h_local, 3 * HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    schedule = default_schedule(h_local)
    chunks, offset = [], 0
    for count in schedule:
        chunks.append(
            torch.empty(
                (1, s_global, count, 3 * HEAD_DIM), dtype=torch.bfloat16, device=device
            )
        )
        offset += count

    def fused():
        return session(x, w_qkv, q_scale, k_scale, cos_sin)

    def b1_ordinary():
        q, k, v = producer()
        comm_ops.scatter_heads(q, out=b1_out[0])
        comm_ops.scatter_heads(k, out=b1_out[1])
        comm_ops.scatter_heads(v.contiguous(), out=b1_out[2])
        return b1_out

    def b2_whole_qkv():
        q, k, v = producer()
        comm_in.scatter_qkv_head_chunk(
            q, k, v, head_offset=0, head_count=h_local, out=payload
        )
        return tuple(payload.split(HEAD_DIM, dim=-1))

    def b3_head_chunk():
        q, k, v = producer()
        start = 0
        for index, count in enumerate(schedule):
            comm_in.scatter_qkv_head_chunk(
                q, k, v, head_offset=start, head_count=count, out=chunks[index]
            )
            start += count
        merged = torch.cat(chunks, dim=2)
        return tuple(merged.split(HEAD_DIM, dim=-1))

    def b4_nccl():
        from flashinfer.parallel_attention.parallel_wrapper import ulysses_a2a_in

        q, k, v = producer()
        q_out, k_out, v_out, _mask = ulysses_a2a_in(
            q[0],
            k[0],
            v[0].contiguous(),
            None,
            "NHD",
            ulysses_size=world_size,
            ulysses_rank=rank,
            ulysses_group=group,
            fuse_qkv=True,
        )
        return (q_out.unsqueeze(0), k_out.unsqueeze(0), v_out.unsqueeze(0))

    arms = {
        "fused": fused,
        "b1_ordinary": b1_ordinary,
        "b2_whole_qkv": b2_whole_qkv,
        "b3_head_chunk": b3_head_chunk,
    }
    if world_size in (2, 4, 8):
        arms["b4_nccl"] = b4_nccl
    closers = (session.close, comm_in.close, comm_ops.close)
    info = {
        "fused_backend": session.backend,
        "fused_backend_reason": session.backend_reason,
        "ulysses_backend": comm_in.backend,
        "b3_schedule": list(schedule),
    }
    return arms, closers, info


def check_against_fused(arms, device):
    reference = tuple(t.clone() for t in arms["fused"]())
    torch.cuda.synchronize(device)
    verdicts = {}
    for name, arm in arms.items():
        if name == "fused":
            continue
        outputs = arm()
        torch.cuda.synchronize(device)
        max_abs = max(
            float((actual.float() - want.float()).abs().max())
            for actual, want in zip(outputs, reference, strict=True)
        )
        close = all(
            torch.allclose(actual.float(), want.float(), atol=1e-2, rtol=1e-2)
            for actual, want in zip(outputs, reference, strict=True)
        )
        verdicts[name] = {"max_abs_vs_fused": max_abs, "allclose_1e-2": close}
        # Agree on the verdict across ranks before raising: a rank that raised alone
        # would leave its peers blocked in the next collective instead of reporting.
        disagree = torch.tensor([0 if close else 1], device=device, dtype=torch.int32)
        dist.all_reduce(disagree, op=dist.ReduceOp.MAX)
        if int(disagree.item()):
            raise RuntimeError(
                f"arm {name} disagrees with the fused arm on at least one rank "
                f"(this rank: max |diff| = {max_abs:.4f}, allclose={close})"
            )
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", choices=("representative", "all"), default="representative"
    )
    parser.add_argument(
        "--hidden", type=int, action="append", help="restrict to these hidden sizes"
    )
    parser.add_argument(
        "--s-global",
        type=int,
        action="append",
        help="restrict to these global sequence lengths",
    )
    parser.add_argument(
        "--backend",
        default="nvlink",
        choices=("auto", "nvlink"),
        help="fused arm backend",
    )
    parser.add_argument(
        "--ulysses-backend",
        default="auto",
        choices=("auto", "nvlink", "nccl"),
        help="B1-B3 communicators",
    )
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeat_iters < 1 or args.warmup_iters < 0:
        parser.error("repeat-iters must be positive and warmup-iters nonnegative")

    # The same check ``bench_gpu_time_with_cupti`` performs before it would fall back to CUDA events; here the
    # fallback is an error so a published number is always a CUPTI number.
    from importlib.metadata import version as package_version

    from cupti import cupti  # noqa: F401

    if int(package_version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError(
            "cupti-python >= 13 is required (pip install -U cupti-python)"
        )
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")

    started = time.monotonic()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD
    rows_out = []
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if args.rows == "all":
            grid = [(hidden, s_global) for hidden in HIDDENS for s_global in S_GLOBALS]
        else:
            grid = list(REPRESENTATIVE)
        grid = [
            (hidden, s_global)
            for hidden, s_global in grid
            if (not args.hidden or hidden in args.hidden)
            and (not args.s_global or s_global in args.s_global)
        ]
        for hidden, s_global in grid:
            if (hidden // HEAD_DIM) % world_size or s_global % world_size:
                continue
            rows = s_global // world_size
            arms, closers, info = build_arms(
                group, device, rank, world_size, hidden, rows, args
            )
            try:
                verdicts = check_against_fused(arms, device)
                results = {}
                for name, arm in arms.items():
                    samples = bench_gpu_time_with_cupti(
                        arm,
                        dry_run_iters=args.warmup_iters,
                        repeat_iters=args.repeat_iters,
                        cold_l2_cache=True,
                        use_cuda_graph=False,
                        aggregate_op=max,
                    )
                    results[name] = {
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                    }
                baseline_ms = min(
                    results[name]["median_ms"] for name in results if name != "fused"
                )
                rows_out.append(
                    {
                        "world_size": world_size,
                        "hidden": hidden,
                        "s_global": s_global,
                        "rows_per_rank": rows,
                        **info,
                        "timing_backend": "cupti",
                        "cold_l2_cache": True,
                        "rank_aggregation": "max_then_median",
                        "correctness_vs_fused": verdicts,
                        "results": results,
                        "fastest_baseline_ms": baseline_ms,
                        "fastest_baseline": min(
                            (name for name in results if name != "fused"),
                            key=lambda n: results[n]["median_ms"],
                        ),
                        "speedup_over_fastest_baseline": baseline_ms
                        / results["fused"]["median_ms"],
                    }
                )
            finally:
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                for close in closers:
                    close()
        if rank == 0:
            report = {
                "world_size": world_size,
                "gpu": torch.cuda.get_device_name(),
                "compute_capability": torch.cuda.get_device_capability(),
                "runtime_seconds": time.monotonic() - started,
                "rows": rows_out,
            }
            encoded = json.dumps(report, indent=2)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(encoded + "\n")
            print(
                "| W | hidden | S_global | fused (us) | B1 (us) | B2 (us) | B3 (us) | B4 (us) | speedup vs fastest |"
            )
            print("|---|---|---|---|---|---|---|---|---|")
            for row in rows_out:
                cells = [
                    f"{row['results'][name]['median_ms'] * 1000:.1f}"
                    if name in row["results"]
                    else "-"
                    for name in (
                        "fused",
                        "b1_ordinary",
                        "b2_whole_qkv",
                        "b3_head_chunk",
                        "b4_nccl",
                    )
                ]
                print(
                    f"| {row['world_size']} | {row['hidden']} | {row['s_global']} | "
                    + " | ".join(cells)
                    + f" | {row['speedup_over_fastest_baseline']:.2f}x ({row['fastest_baseline']}) |"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
