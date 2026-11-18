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

The Ulysses communicators of B1-B3 default to ``ULYSSES_BACKEND_DEFAULT_BY_ARCH``
(``"auto"`` on SM90 / SM100 / SM103: the NVLink kernels on SM100 / SM103, NCCL on
SM90; ``"nccl"`` on SM107, where the NVLink Ulysses kernels are unverified and
the export protocol measures B1-B3 on NCCL, so both measurements of an arm use
the same exchange backend); ``--ulysses-backend`` overrides it and the JSON
records both the requested and the resolved backend.
``--fold-barrier auto|0|1`` selects the fused arm's launch path (the
architecture default, the separate barrier kernel, or the producer with the
folded rendezvous); the JSON records the resolved path and its launch count.
Timing uses ``flashinfer.testing.bench_gpu_time_with_cupti`` with the per-rank
maximum of every sample; CUPTI unavailability is an error.  Every arm is called
twice per timed iteration (``two_calls``; ``calls_per_iteration: 2`` in the
JSON) and its samples are divided by two: the fused session alternates its
phase program per call and the CUPTI timer requires every iteration to launch
the same kernel set; the per-operator arms are paired the same way.  The timer
flushes L2 once per iteration, before the first call of the pair (``l2_flush``
in the JSON); the second call runs with the first call's footprint in L2, so
every per-call sample is the mean of a cold and a warm call.  The same factor
applies to every arm; the arm ratios against a cold-only protocol are not
measured.
Every per-operator arm is checked once against the fused arm before it is timed
(``check_against_fused``): ``torch.allclose(atol=0.1, rtol=1e-2)`` per tensor,
both sides finite.  The fused kernel and the per-operator operators round to
BF16 at different points (fp32 projection kept against a BF16 projection store;
the normed value rounded before the BF16 scale against the scaled value rounded
once), so their outputs differ by BF16 rounding flips of O(1) normalised inputs;
at a near-zero RoPE output such a flip is an absolute difference of a few
spacings of the pair norm, up to 0.0625 on the recorded multi-GPU runs -- hence
0.1 here, where the multi-GPU test uses 0.05 against a host reference (there is
no host reference at these shapes).  The JSON records ``passed``, ``finite`` and
``max_abs_diff`` per arm and tensor, and ``bitwise_equal_to_b1`` (informational;
B1-B4 share one producer and bit-preserving transports).
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
from flashinfer.jit import cake_qkv_norm_rope_exchange_overlap as loader
from flashinfer.norm import rmsnorm
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from flashinfer.testing import bench_gpu_time_with_cupti

HEAD_DIM = 128
# B1-B3 communicator backend when ``--ulysses-backend`` is not given: FlashInfer's
# own resolution where it agrees with the export protocol's measurement, the
# protocol's pinned backend elsewhere (SM107: NCCL until the NVLink Ulysses path
# is verified there).
ULYSSES_BACKEND_DEFAULT_BY_ARCH: dict[str, str] = {
    "sm_90a": "auto",
    "sm_100a": "auto",
    "sm_103a": "auto",
    "sm_107a": "nccl",
}


def default_ulysses_backend(arch: str | None) -> str:
    """Default B1-B3 backend of the device architecture (see the module docstring)."""

    if arch not in ULYSSES_BACKEND_DEFAULT_BY_ARCH:
        raise RuntimeError(
            f"no delivered fused QKV exchange programs for architecture {arch!r}; "
            "pass --ulysses-backend explicitly to benchmark the per-operator arms"
        )
    return ULYSSES_BACKEND_DEFAULT_BY_ARCH[arch]


EPS = 1e-6
# Calls of an arm per timed iteration (``two_calls``): the fused session launches ``producer_fold0`` / ``producer_fold1``
# on alternating calls (``barrier_p0`` / ``barrier_p1`` with ``--fold-barrier 0``), and ``bench_gpu_time_with_cupti``
# raises ``Inconsistent kernel names`` when the kernel set differs between iterations; a pair launches both phases.
CALLS_PER_ITERATION = 2
HIDDENS = (3072, 4096, 6144, 8192)
S_GLOBALS = (4096, 8192, 16384, 32768, 65536, 131072)
REPRESENTATIVE = ((4096, 16384), (8192, 16384), (4096, 65536), (8192, 65536))
FOLD_BARRIER_CHOICES = {"auto": None, "0": False, "1": True}


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
        fold_barrier=FOLD_BARRIER_CHOICES[args.fold_barrier],
    )
    comm_in = UlyssesCommunicator(
        group,
        max_bytes=rows * 3 * hidden * torch.bfloat16.itemsize,
        dtype=torch.bfloat16,
        backend=args.ulysses_backend,
        device=device,
    )
    comm_ops = UlyssesCommunicator(
        group,
        max_bytes=rows * hidden * torch.bfloat16.itemsize,
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
    fused_info = session.describe()
    info = {
        "fused_backend": session.backend,
        "fused_backend_reason": session.backend_reason,
        "ulysses_backend_requested": args.ulysses_backend,
        "fused_fold_barrier": fused_info.get("fold_barrier"),
        "fused_fold_barrier_requested": fused_info.get("fold_barrier_requested"),
        "fused_launches_per_call": fused_info.get("launches_per_call"),
        "fused_shape_gates": fused_info.get("shape_gates"),
        "fused_producer_variant": fused_info.get("producer_variant"),
        "fused_group_m_scale": fused_info.get("group_m_scale"),
        "fused_prepared_row_plans": fused_info.get("prepared_row_plans"),
        "ulysses_backend": comm_in.backend,
        "b3_schedule": list(schedule),
    }
    return arms, closers, info


def two_calls(arm):
    """``arm`` called ``CALLS_PER_ITERATION`` (two) times per timed iteration, so that every iteration launches both
    phase programs of the fused session; the samples are divided by the same factor for every arm, fused and
    per-operator alike, and a per-call time is reported.  The timer's L2 flush precedes the first call only: a
    per-call sample is the mean of a cold and a warm call, and the arm ratios against a cold-only protocol are not
    measured."""

    assert CALLS_PER_ITERATION == 2, CALLS_PER_ITERATION

    def call_twice():
        arm()
        return arm()

    return call_twice


def check_against_fused(arms, device, rank):
    """Every per-operator arm against the fused arm, per rank and per tensor: both sides finite and
    ``torch.allclose(atol=0.1, rtol=1e-2)``.

    The fused kernel keeps the projection in fp32 and rounds the normed value before the BF16 scale; the
    per-operator arms store the projection in BF16 and round the scaled normed value once.  So a large fraction of
    the stored normed values differ by one spacing (a few by two), and where a RoPE output is near zero -- two
    cancelling pair terms of magnitude up to ~4 -- that input shift is an absolute difference of a few BF16
    spacings of the pair norm: up to 0.0625 on the recorded two-, four- and eight-GPU runs of three architectures.
    The absolute tolerance is 0.1 for that reason -- set above that recorded maximum, not derived as a bound --
    twice the 0.05 the multi-GPU test uses against a host reference (there is no host reference at these shapes).
    What the guard proves: every per-operator arm delivers its producer's values to the right place within the
    rounding band of the fused arm, and the fused arm within the band of the per-operator arms; it is not the fused
    kernel's correctness evidence at these shapes (the multi-GPU test is).  ``bitwise_equal_to_b1`` records whether
    an arm's outputs equal ``b1_ordinary``'s bit for bit (B1-B4 share one producer and bit-preserving transports;
    informational).  The ranks agree on the tolerance verdict before raising, so a failing rank does not leave its
    peers blocked in the next collective (the shape check is rank-uniform and raises locally)."""

    fused = tuple(t.clone() for t in arms["fused"]())
    torch.cuda.synchronize(device)
    b1 = None
    verdicts = {}
    for name, arm in arms.items():
        if name == "fused":
            continue
        outputs = arm()
        torch.cuda.synchronize(device)
        if name == "b1_ordinary":
            b1 = outputs = tuple(t.clone() for t in outputs)
        tensors = {}
        for tensor, actual, want in zip(("q", "k", "v"), outputs, fused, strict=True):
            got, ref = actual.float(), want.float()
            assert got.shape == ref.shape, (name, tensor, got.shape, ref.shape)
            finite = bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
            tensors[tensor] = {
                "passed": finite and torch.allclose(got, ref, atol=0.1, rtol=1e-2),
                "finite": finite,
                "max_abs_diff": float((got - ref).abs().max()) if finite else None,
            }
        passed = all(v["passed"] for v in tensors.values())
        verdicts[name] = {
            "passed": passed,
            "finite": all(v["finite"] for v in tensors.values()),
            "bitwise_equal_to_b1": None
            if name == "b1_ordinary" or b1 is None
            else all(torch.equal(a, b) for a, b in zip(outputs, b1, strict=True)),
            "tensors": tensors,
        }
        disagree = torch.tensor([0 if passed else 1], device=device, dtype=torch.int32)
        dist.all_reduce(disagree, op=dist.ReduceOp.MAX)
        if int(disagree.item()):
            failing = {t: v for t, v in tensors.items() if not v["passed"]}
            raise RuntimeError(
                f"arm {name} disagrees with the fused arm on at least one rank; rank "
                f"{rank}: {failing or 'every tensor passed on this rank'}"
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
        default=None,
        choices=("auto", "nvlink", "nccl"),
        help="B1-B3 communicators (default: ULYSSES_BACKEND_DEFAULT_BY_ARCH of the device)",
    )
    parser.add_argument(
        "--fold-barrier",
        default="auto",
        choices=tuple(FOLD_BARRIER_CHOICES),
        help="fused arm launch path: architecture default, barrier kernel (0) or folded rendezvous (1)",
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
    if args.ulysses_backend is None:
        args.ulysses_backend = default_ulysses_backend(
            loader.device_facts(local_rank).arch
        )
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
                verdicts = check_against_fused(arms, device, rank)
                results = {}
                for name, arm in arms.items():
                    samples = [
                        sample / CALLS_PER_ITERATION
                        for sample in bench_gpu_time_with_cupti(
                            two_calls(arm),
                            dry_run_iters=args.warmup_iters,
                            repeat_iters=args.repeat_iters,
                            cold_l2_cache=True,
                            use_cuda_graph=False,
                            aggregate_op=max,
                        )
                    ]
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
                        "rank_aggregation": "max_then_median",
                        "calls_per_iteration": CALLS_PER_ITERATION,
                        "l2_flush": "before the first of the two calls per iteration",
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
