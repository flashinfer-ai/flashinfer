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

Benchmark of the context-parallel (KCP) Cake packed KDA training step: one
process per rank (``torchrun --nproc_per_node N``), every rank timing the
module boundary of its fragments -- the KCP causal convolution
(``flashinfer.cake_causal_conv1d_train.cake_causal_conv1d_kcp``) followed by
the KCP KDA step (``flashinfer.kda_train.chunk_kda_kcp``), forward and
backward -- with CUDA events, and the slowest rank's time counting for the
iteration.

Arms, every one the same packed batch split by the same partition:

* ``cake``: the KCP path;
* ``control``: the KCP path again (the same programs; its ratio to ``cake``
  measures the protocol's own noise and must stay within the control band);
* ``local``: the single-rank Cake kernels on the rank's shard as if it were a
  batch of its own (no exchange; the per-rank work without the boundaries);
* ``fla``: flash-linear-attention's ``chunk_kda`` on the rank's shard, when
  the package is installed.

Protocol: ``--groups`` groups (at least 3); in every group the arms run in the
order cake, control, control, cake (the ABBA pattern), then the optional arms;
every visit takes ``--warmup`` untimed and ``--samples`` timed iterations (at
least 8), each after a barrier and an L2 flush; the figure of an arm in a group
is the median of its timed iterations over both visits, the figure of an arm
is the median over the groups.  The report gives, per arm, the group medians,
the overall median, tokens per second, the control ratio against its band
(0.98 .. 1.02) and the drift of the ``cake`` arm between its first and last
group.  Rank 0 prints the table and writes ``--json``.

Example::

    torchrun --nproc_per_node 4 benchmarks/bench_cake_kda_train_kcp.py \\
        --seqlen 65536 --docs 3 --heads 16 --channels 4096 --groups 3 --samples 8 --json kcp.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.cake_causal_conv1d_train import (  # noqa: E402
    cake_causal_conv1d,
    cake_causal_conv1d_kcp,
)
from flashinfer.kda_train import (  # noqa: E402
    build_kcp_context,
    chunk_kda,
    chunk_kda_kcp,
    kcp_comm_bytes,
    kcp_partition,
    kcp_saved_bytes,
    kcp_workspace_bytes,
)
from tests.test_helpers.cake_kda_train_reference import make_inputs  # noqa: E402

ARMS = ("cake", "control", "local", "fla")
CONTROL_BAND = (0.98, 1.02)
L2_FLUSH_BYTES = 256 << 20
WIDTH = 4


def split_docs(seqlen: int, docs: int) -> tuple[int, ...]:
    """``docs`` unequal document lengths summing to ``seqlen`` whose tails are not chunk-aligned."""
    if docs < 1 or seqlen < docs:
        raise ValueError(f"cannot split {seqlen} tokens into {docs} documents")
    weights = [1.0 + 0.37 * i for i in range(docs)]
    total = sum(weights)
    lengths = [max(1, int(seqlen * w / total)) for w in weights]
    lengths[-1] += seqlen - sum(lengths)
    for i in range(docs - 1):
        if lengths[i] % 64 == 0 and lengths[i + 1] > 1:
            lengths[i] += 1
            lengths[i + 1] -= 1
    return tuple(lengths)


def _init() -> tuple[int, int, torch.device]:
    """The process group of ``torchrun``: rank, world size and this rank's device (``LOCAL_RANK``)."""
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def _shard(x: torch.Tensor, cu: list[int], fragments) -> torch.Tensor:
    """The rows of ``x`` of this rank's fragments, in the partition's slot order."""
    pieces = [
        x[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
        for f in fragments
        if f.length > 0
    ]
    return torch.cat(pieces, 0) if pieces else x[:0]


def make_problem(args, rank: int, world: int, device: torch.device) -> dict[str, Any]:
    """The whole batch on the CPU (every rank draws the same tensors), this rank's shard on the device."""
    lengths = split_docs(args.seqlen, args.docs)
    inputs = make_inputs(
        lengths, args.heads, device="cpu", seed=args.seed, with_dout=True
    )
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    cu_cpu = torch.tensor(cu, dtype=torch.int32)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    x = (torch.randn(args.seqlen, args.channels, generator=gen) * 0.5).to(
        torch.bfloat16
    )
    dy = (torch.randn(args.seqlen, args.channels, generator=gen) * 0.1).to(
        torch.bfloat16
    )
    weight = (torch.randn(args.channels, WIDTH, generator=gen) * 0.3).to(device)
    part = kcp_partition(cu_cpu, world, args.layout)
    group = dist.group.WORLD
    ctx = build_kcp_context(part, group, rank, device)
    fragments = part.fragments[rank]
    local = {
        name: _shard(inputs[name], cu, fragments).to(device)
        for name in ("q", "k", "v", "g", "beta", "dout")
    }
    local["x"] = _shard(x, cu, fragments).to(device)
    local["dy"] = _shard(dy, cu, fragments).to(device)
    local_cu = [0]
    for f in fragments:
        local_cu.append(local_cu[-1] + f.length)
    local["cu_seqlens"] = torch.tensor(local_cu, dtype=torch.int32, device=device)
    return {
        "lengths": lengths,
        "cu": cu,
        "ctx": ctx,
        "part": part,
        "weight": weight,
        "local": local,
        "tokens_local": int(local["q"].shape[0]),
        "tokens_total": args.seqlen,
    }


def _step_kcp(problem: dict[str, Any]) -> None:
    local, ctx, weight = problem["local"], problem["ctx"], problem["weight"]
    x = local["x"].detach().requires_grad_(True)
    w = weight.detach().requires_grad_(True)
    q, k, v, g, beta = (
        local[n].detach().requires_grad_(True) for n in ("q", "k", "v", "g", "beta")
    )
    y = cake_causal_conv1d_kcp(x, w, ctx)
    out = chunk_kda_kcp(q, k, v, g, beta, ctx)
    torch.autograd.backward([y, out], [local["dy"], local["dout"]])


def _step_local(problem: dict[str, Any]) -> None:
    local, weight = problem["local"], problem["weight"]
    x = local["x"].detach().requires_grad_(True)
    w = weight.detach().requires_grad_(True)
    q, k, v, g, beta = (
        local[n].detach().requires_grad_(True) for n in ("q", "k", "v", "g", "beta")
    )
    y = cake_causal_conv1d(x, w, local["cu_seqlens"])
    out = chunk_kda(q, k, v, g, beta, local["cu_seqlens"])
    torch.autograd.backward([y, out], [local["dy"], local["dout"]])


def _fla_step(problem: dict[str, Any]):
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda  # type: ignore
    except Exception:  # noqa: BLE001 -- any import failure means the optional arm is absent
        return None
    local = problem["local"]
    cu = local["cu_seqlens"]

    def step() -> None:
        q, k, v, g, beta = (
            local[n].detach().unsqueeze(0).requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        )
        out, _ = fla_chunk_kda(q, k, v, g, beta, scale=None, cu_seqlens=cu)
        out.backward(local["dout"].unsqueeze(0))

    return step


def _time_visit(
    step, samples: int, warmup: int, flush: torch.Tensor, device: torch.device
) -> list[float]:
    """The slowest rank's milliseconds of ``samples`` timed iterations (after ``warmup`` untimed ones)."""
    times: list[float] = []
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for i in range(warmup + samples):
        flush.fill_(i & 0xFF)
        torch.cuda.synchronize(device)
        dist.barrier()
        start.record()
        step()
        end.record()
        torch.cuda.synchronize(device)
        t = torch.tensor([start.elapsed_time(end)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if i >= warmup:
            times.append(float(t.item()))
    return times


def run(args, rank: int, world: int, device: torch.device) -> dict[str, Any]:
    problem = make_problem(args, rank, world, device)
    steps = {
        "cake": lambda: _step_kcp(problem),
        "control": lambda: _step_kcp(problem),
        "local": lambda: _step_local(problem),
    }
    fla = _fla_step(problem)
    if fla is not None:
        steps["fla"] = fla
    arms = [arm for arm in args.arms if arm in steps]
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    for arm in (
        arms
    ):  # one untimed call per arm: the JIT builds and the first exchanges happen here
        steps[arm]()
    torch.cuda.synchronize(device)
    dist.barrier()
    order = ["cake", "control", "control", "cake"] + [
        arm for arm in arms if arm not in ("cake", "control")
    ]
    per_group: dict[str, list[float]] = {arm: [] for arm in arms}
    raw: dict[str, list[list[float]]] = {arm: [] for arm in arms}
    for _ in range(args.groups):
        visits: dict[str, list[float]] = {arm: [] for arm in arms}
        for arm in order:
            if arm in steps:
                visits[arm].extend(
                    _time_visit(steps[arm], args.samples, args.warmup, flush, device)
                )
        for arm in arms:
            per_group[arm].append(statistics.median(visits[arm]))
            raw[arm].append(visits[arm])
    medians = {arm: statistics.median(per_group[arm]) for arm in arms}
    tokens = problem["tokens_total"]
    rows = {
        arm: {
            "group_medians_ms": per_group[arm],
            "median_ms": medians[arm],
            "tokens_per_s": tokens / (medians[arm] * 1e-3),
            "samples": raw[arm],
        }
        for arm in arms
    }
    control = medians["control"] / medians["cake"] if "control" in medians else None
    drift = abs(per_group["cake"][-1] - per_group["cake"][0]) / medians["cake"]
    ctx = problem["ctx"]
    return {
        "config": {
            "seqlen": args.seqlen,
            "docs": args.docs,
            "lengths": list(problem["lengths"]),
            "heads": args.heads,
            "channels": args.channels,
            "layout": args.layout,
            "world_size": world,
            "groups": args.groups,
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "bytes": {
            "workspace": kcp_workspace_bytes(ctx, H=args.heads, C=args.channels),
            "saved": kcp_saved_bytes(ctx, H=args.heads, C=args.channels),
            "comm": kcp_comm_bytes(ctx, H=args.heads, C=args.channels),
        },
        "arms": rows,
        "control_ratio": control,
        "control_band": list(CONTROL_BAND),
        "control_ok": control is not None
        and CONTROL_BAND[0] <= control <= CONTROL_BAND[1],
        "drift": drift,
        "drift_ok": drift <= 0.02,
        "env": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "ranks": world,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def _print(report: dict[str, Any]) -> None:
    cfg = report["config"]
    print(
        f"KCP x{cfg['world_size']}: S={cfg['seqlen']} docs={cfg['docs']} H={cfg['heads']} C={cfg['channels']} "
        f"layout={cfg['layout']} groups={cfg['groups']} samples={cfg['samples']}"
    )
    for arm, row in report["arms"].items():
        groups = " ".join(f"{t:.3f}" for t in row["group_medians_ms"])
        print(
            f"  {arm:8s} {row['median_ms']:9.3f} ms  {row['tokens_per_s'] / 1e6:8.3f} Mtok/s  groups [{groups}]"
        )
    if report["control_ratio"] is not None:
        print(
            f"  control/cake = {report['control_ratio']:.4f} (band {report['control_band']}) -> {'ok' if report['control_ok'] else 'OUT OF BAND'}"
        )
    print(
        f"  drift(cake) = {report['drift']:.4f} -> {'ok' if report['drift_ok'] else 'ABOVE 2%'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seqlen", type=int, default=65536, help="tokens of the whole packed batch"
    )
    parser.add_argument("--docs", type=int, default=3)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument(
        "--channels", type=int, default=4096, help="channels of the causal convolution"
    )
    parser.add_argument(
        "--layout",
        default="contiguous",
        choices=("contiguous", "zigzag", "zigzag_per_document"),
    )
    parser.add_argument(
        "--groups", type=int, default=3, help="measurement groups (at least 3)"
    )
    parser.add_argument(
        "--samples", type=int, default=8, help="timed iterations per visit (at least 8)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="untimed iterations per visit"
    )
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--json", type=str, default=None, help="rank 0 writes the report here"
    )
    args = parser.parse_args()
    if args.groups < 3 or args.samples < 8:
        parser.error("the protocol takes at least 3 groups and 8 samples per visit")
    if "cake" not in args.arms or "control" not in args.arms:
        parser.error("the cake and control arms are required")
    rank, world, device = _init()
    report = run(args, rank, world, device)
    if rank == 0:
        _print(report)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if report["control_ok"] and report["drift_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
