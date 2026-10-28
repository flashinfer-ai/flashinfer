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

Benchmark of the context-parallel (KCP) Cake packed GDN training step: one
process per rank (``torchrun --nproc_per_node N``), every rank timing the
module boundary of its fragments -- the KCP causal convolution
(``flashinfer.cake_causal_conv1d_train.cake_causal_conv1d_kcp``, 10,240
channels) followed by the KCP GDN step (``flashinfer.gdn_train.chunk_gdn_kcp``)
-- with CUDA events, the slowest rank's time counting for the iteration.

Boundaries (``--boundary``): ``B1`` the module forward with autograd recording
(summaries, exchange and saved set written); ``B2`` the backward from the saved
set (report only); ``B3`` ``torch.utils.checkpoint`` forward + replay +
backward (``2 x forward + backward``, the training boundary).

Arms, every one on the same packed batch, the same partition and the same
input storage:

* ``cake``: the KCP path;
* ``fla``: flash-linear-attention's ``chunk_gated_delta_rule`` with its
  ``cp_context`` (and its causal convolution), when the package is installed;
  the pack is padded to a multiple of ``2 * world_size`` tokens for it and the
  padding cost is part of its time;
* ``local``: the single-rank Cake kernels on the rank's shard as a batch of
  its own (no exchange; the per-rank work without the boundaries);
* ``control``: the KCP path again -- the identical-program control.

Protocol: ``--groups`` groups (at least 3); in every group the arms run in the
pattern ``[cake, base, base, cake] x k`` (``--base`` names the baseline arm,
``control`` for the identical-program run); every call takes ``--samples``
timed iterations (at least 8) after ``--warmup`` untimed ones, each after an L2
flush and a barrier; a sample is *steady* when the previous call dispatched the
same arm, and the statistics are medians of the steady samples: the figure of
an arm in a group is the median of its steady samples, the figure of an arm is
the median over the groups.  The report gives, per arm, the group medians, the
overall median, tokens per second, the ratio ``base / cake`` and, for a
control run, the control ratio against its band (0.98 .. 1.02) and the drift
of the ``cake`` arm between its first and last group.  Rank 0 prints the table
and writes ``--json``.

Example::

    torchrun --nproc_per_node 4 benchmarks/bench_cake_gdn_train_kcp.py \\
        --seqlen 261987 --docs 3 --qk-heads 16 --v-heads 48 --layout zigzag_per_document \\
        --boundary B3 --base fla --groups 3 --samples 8 --json kcp_b3.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.cake_causal_conv1d_train import (  # noqa: E402
    cake_causal_conv1d,
    cake_causal_conv1d_kcp,
)
from flashinfer.gdn_train import (  # noqa: E402
    build_kcp_context,
    chunk_gdn_kcp,
    gdn_kcp_comm_bytes,
    gdn_kcp_saved_bytes,
    gdn_kcp_workspace_bytes,
    kcp_partition,
)

ARMS = ("cake", "fla", "local", "control")
BOUNDARIES = ("B1", "B2", "B3")
CONTROL_BAND = (0.98, 1.02)
L2_FLUSH_BYTES = 256 << 20
WIDTH = 4
HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
FLA_LAYOUT = {"contiguous": "contiguous", "zigzag": "zigzag"}


# ----------------------------------------------------------------------------------------------------------------
# Inputs (CPU, seeded: every rank draws the same tensors)


def split_docs(seqlen: int, docs: int) -> tuple[int, ...]:
    """``docs`` unequal document lengths summing to ``seqlen``; no length is a multiple of 64 and at least one has
    ``length mod 64 < 32`` (the shapes the partition must handle)."""
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


def make_inputs(
    seqlen: int, qk_heads: int, v_heads: int, channels: int, seed: int
) -> dict[str, torch.Tensor]:
    """The whole packed batch on the CPU: ``q, k`` BF16 ``[T, H_qk, 128]`` (rows L2-normalised in FP32), ``v, dout``
    BF16 ``[T, HV, 128]``, ``g`` FP32 ``[T, HV]`` (natural-log decay, ``<= 0``), ``beta`` FP32 ``[T, HV]`` in (0, 1),
    the convolution input ``x`` and its upstream gradient ``dy`` BF16 ``[T, C]`` and the FP32 weight ``[C, 4]``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def l2(x: torch.Tensor) -> torch.Tensor:
        return (x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(torch.bfloat16)

    a_log = torch.rand(v_heads, generator=gen, dtype=torch.float32) * math.log(16.0)
    g = -torch.exp(a_log) * torch.nn.functional.softplus(randn(seqlen, v_heads))
    return {
        "q": l2(randn(seqlen, qk_heads, HEAD_DIM)),
        "k": l2(randn(seqlen, qk_heads, HEAD_DIM)),
        "v": randn(seqlen, v_heads, HEAD_DIM).to(torch.bfloat16),
        "g": g,
        "beta": torch.sigmoid(randn(seqlen, v_heads)),
        "dout": randn(seqlen, v_heads, HEAD_DIM).to(torch.bfloat16),
        "x": (randn(seqlen, channels) * 0.5).to(torch.bfloat16),
        "dy": (randn(seqlen, channels) * 0.1).to(torch.bfloat16),
        "weight": randn(channels, WIDTH) * 0.3,
    }


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
    lengths = split_docs(args.seqlen, args.docs)
    inputs = make_inputs(
        args.seqlen, args.qk_heads, args.v_heads, args.channels, args.seed
    )
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    cu_cpu = torch.tensor(cu, dtype=torch.int32)
    part = kcp_partition(cu_cpu, world, args.layout)
    ctx = build_kcp_context(
        part, dist.group.WORLD, rank, device, conv_kernel_size=WIDTH
    )
    fragments = part.fragments[rank]
    local = {
        name: _shard(inputs[name], cu, fragments).to(device)
        for name in ("q", "k", "v", "g", "beta", "dout", "x", "dy")
    }
    local_cu = [0]
    for f in fragments:
        local_cu.append(local_cu[-1] + f.length)
    local["cu_seqlens"] = torch.tensor(local_cu, dtype=torch.int32, device=device)
    return {
        "lengths": lengths,
        "cu": cu,
        "inputs": inputs,
        "ctx": ctx,
        "part": part,
        "weight": inputs["weight"].to(device),
        "local": local,
        "tokens_local": int(local["q"].shape[0]),
        "tokens_total": args.seqlen,
    }


# ----------------------------------------------------------------------------------------------------------------
# Arms: one callable per (arm, boundary)


def _leaves(local: dict[str, torch.Tensor], weight: torch.Tensor) -> list[torch.Tensor]:
    return [
        local[n].detach().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta", "x")
    ] + [weight.detach().requires_grad_(True)]


def _bounded(
    forward: Callable[..., tuple[torch.Tensor, ...]],
    leaves_of: Callable[[], list[torch.Tensor]],
    grads_out: list[torch.Tensor],
    boundary: str,
) -> Callable[[], None]:
    """The callable of one boundary over ``forward(*leaves) -> outputs``."""
    if boundary == "B1":

        def b1() -> None:
            with torch.enable_grad():
                forward(*leaves_of())

        return b1
    if boundary == "B2":

        def b2() -> None:
            with torch.enable_grad():
                outs = forward(*leaves_of())
                torch.autograd.backward(list(outs), grads_out)

        return b2

    def b3() -> None:
        with torch.enable_grad():
            outs = checkpoint(forward, *leaves_of(), use_reentrant=False)
            torch.autograd.backward(list(outs), grads_out)

    return b3


def _cake_arm(problem: dict[str, Any], boundary: str) -> Callable[[], None]:
    local, ctx, weight = problem["local"], problem["ctx"], problem["weight"]

    def forward(q, k, v, g, beta, x, w):
        y = cake_causal_conv1d_kcp(x, w, ctx)
        out = chunk_gdn_kcp(q, k, v, g, beta, ctx)
        return y, out

    return _bounded(
        forward, lambda: _leaves(local, weight), [local["dy"], local["dout"]], boundary
    )


def _local_arm(problem: dict[str, Any], boundary: str) -> Callable[[], None] | None:
    """The single-rank Cake path on the rank's shard; absent until the CP = 1 entry point ships."""
    try:
        from flashinfer.gdn_train import chunk_gdn  # type: ignore
    except ImportError:
        return None
    local, weight = problem["local"], problem["weight"]
    cu = local["cu_seqlens"]

    def forward(q, k, v, g, beta, x, w):
        y = cake_causal_conv1d(x, w, cu)
        out = chunk_gdn(q, k, v, g, beta, cu)
        return y, out

    return _bounded(
        forward, lambda: _leaves(local, weight), [local["dy"], local["dout"]], boundary
    )


def _fla_arm(
    problem: dict[str, Any],
    boundary: str,
    layout: str,
    device: torch.device,
    world: int,
    rank: int,
) -> Callable[[], None] | None:
    """flash-linear-attention's KCP module on the same tokens (``use_tf32x3_affine_chain=True``).  FLA splits the
    pack itself from the global ``cu_seqlens`` padded to a multiple of ``2 * world_size`` tokens; the padding document
    and the FLA shard of this rank are built here.  Only the ``contiguous`` and ``zigzag`` layouts have an FLA form."""
    if layout not in FLA_LAYOUT:
        return None
    try:
        from fla.modules.causal_conv1d import causal_conv1d  # type: ignore
        from fla.ops.cp.context import build_cp_context  # type: ignore
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule  # type: ignore
    except Exception:  # noqa: BLE001 -- any import failure means the optional arm is absent
        return None
    inputs, cu = problem["inputs"], list(problem["cu"])
    total = cu[-1]
    pad = (-total) % (2 * world)
    if pad:
        cu.append(total + pad)
    cu_cpu = torch.tensor(cu, dtype=torch.long)
    fla_ctx = build_cp_context(
        cu_cpu.to(device),
        dist.group.WORLD,
        conv1d_kernel_size=WIDTH,
        cu_seqlens_cpu=cu_cpu,
        use_tf32x3_affine_chain=True,
        layout=FLA_LAYOUT[layout],
    )
    # This rank's FLA shard: the token ranges FLA assigns to it, read from its local cu_seqlens of the padded pack.
    bounds = [int(c) for c in fla_ctx.cu_seqlens_cpu.tolist()]
    t_r = bounds[-1]
    padded = {
        name: torch.cat(
            [
                inputs[name],
                torch.zeros((pad, *inputs[name].shape[1:]), dtype=inputs[name].dtype),
            ],
            0,
        )
        for name in ("q", "k", "v", "g", "beta", "dout", "x", "dy")
    }
    if layout == "contiguous":
        lo = rank * (cu[-1] // world)
        rows = torch.arange(lo, lo + t_r)
    else:
        piece = cu[-1] // (2 * world)
        rows = torch.cat(
            [
                torch.arange(rank * piece, (rank + 1) * piece),
                torch.arange(
                    (2 * world - 1 - rank) * piece, (2 * world - rank) * piece
                ),
            ]
        )
    local = {
        name: padded[name][rows].unsqueeze(0).contiguous().to(device) for name in padded
    }
    weight = inputs["weight"].to(device)

    def forward(q, k, v, g, beta, x, w):
        y, _ = causal_conv1d(x, w, activation="silu", cp_context=fla_ctx)
        out, _ = chunk_gated_delta_rule(
            q, k, v, g, beta, scale=SCALE, cp_context=fla_ctx
        )
        return y, out

    def leaves() -> list[torch.Tensor]:
        return [
            local[n].detach().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta", "x")
        ] + [weight.detach().requires_grad_(True)]

    return _bounded(forward, leaves, [local["dy"], local["dout"]], boundary)


# ----------------------------------------------------------------------------------------------------------------
# Timing


def _time_call(
    step: Callable[[], None],
    samples: int,
    warmup: int,
    flush: torch.Tensor,
    device: torch.device,
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
    steps: dict[str, Callable[[], None]] = {
        "cake": _cake_arm(problem, args.boundary),
        "control": _cake_arm(problem, args.boundary),
    }
    local = _local_arm(problem, args.boundary)
    if local is not None:
        steps["local"] = local
    fla = _fla_arm(problem, args.boundary, args.layout, device, world, rank)
    if fla is not None:
        steps["fla"] = fla
    if args.base not in steps:
        raise SystemExit(
            f"baseline arm {args.base!r} is not available on this rank set (arms: {sorted(steps)})"
        )
    arms = ["cake", args.base]
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    for arm in (
        arms
    ):  # one untimed call per arm: the JIT builds and the first exchanges happen here
        steps[arm]()
    torch.cuda.synchronize(device)
    dist.barrier()
    pattern = ["cake", args.base, args.base, "cake"] * args.repeat
    per_group: dict[str, list[float]] = {arm: [] for arm in arms}
    raw: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    for group in range(args.groups):
        steady: dict[str, list[float]] = {arm: [] for arm in arms}
        previous: str | None = None
        for arm in pattern:
            times = _time_call(steps[arm], args.samples, args.warmup, flush, device)
            # the first timed sample follows the warmup calls of the same arm: every sample of a call is steady when
            # the arm equals the previous call's; the first call of a group follows the other arm (not steady)
            if previous == arm or previous is None and group == 0 and arm == pattern[0]:
                steady[arm].extend(times)
            else:
                steady[arm].extend(times[1:] if args.warmup == 0 else times)
            raw[arm].append({"group": group, "previous": previous, "samples_ms": times})
            previous = arm
        for arm in arms:
            per_group[arm].append(statistics.median(steady[arm]))
    medians = {arm: statistics.median(per_group[arm]) for arm in arms}
    tokens = problem["tokens_total"]
    rows = {
        arm: {
            "group_medians_ms": per_group[arm],
            "median_ms": medians[arm],
            "tokens_per_s": tokens / (medians[arm] * 1e-3),
            "calls": raw[arm],
        }
        for arm in arms
    }
    ratio = medians[args.base] / medians["cake"]
    drift = abs(per_group["cake"][-1] - per_group["cake"][0]) / medians["cake"]
    ctx = problem["ctx"]
    heads = {"H_qk": args.qk_heads, "HV": args.v_heads}
    return {
        "config": {
            "seqlen": args.seqlen,
            "docs": args.docs,
            "lengths": list(problem["lengths"]),
            **heads,
            "channels": args.channels,
            "layout": args.layout,
            "boundary": args.boundary,
            "base": args.base,
            "world_size": world,
            "groups": args.groups,
            "repeat": args.repeat,
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "bytes": {
            "workspace": gdn_kcp_workspace_bytes(ctx, **heads),
            "saved": gdn_kcp_saved_bytes(ctx, **heads, save_mode=args.save_mode),
            "comm": gdn_kcp_comm_bytes(ctx, **heads, C=args.channels),
        },
        "arms": rows,
        "ratio_base_over_cake": ratio,
        "control_ratio": ratio if args.base == "control" else None,
        "control_band": list(CONTROL_BAND),
        "control_ok": (CONTROL_BAND[0] <= ratio <= CONTROL_BAND[1])
        if args.base == "control"
        else None,
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
        f"KCP x{cfg['world_size']} {cfg['boundary']}: S={cfg['seqlen']} docs={cfg['docs']} H_qk={cfg['H_qk']} "
        f"HV={cfg['HV']} C={cfg['channels']} layout={cfg['layout']} groups={cfg['groups']} samples={cfg['samples']}"
    )
    for arm, row in report["arms"].items():
        groups = " ".join(f"{t:.3f}" for t in row["group_medians_ms"])
        print(
            f"  {arm:8s} {row['median_ms']:9.3f} ms  {row['tokens_per_s'] / 1e6:8.3f} Mtok/s  groups [{groups}]"
        )
    print(f"  {cfg['base']}/cake = {report['ratio_base_over_cake']:.4f}")
    if report["control_ratio"] is not None:
        print(
            f"  control band {report['control_band']} -> {'ok' if report['control_ok'] else 'OUT OF BAND'}"
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
    parser.add_argument("--qk-heads", type=int, default=16, help="Q/K heads (H_qk)")
    parser.add_argument(
        "--v-heads", type=int, default=48, help="value heads (HV, a multiple of H_qk)"
    )
    parser.add_argument(
        "--channels", type=int, default=10240, help="channels of the causal convolution"
    )
    parser.add_argument(
        "--layout",
        default="contiguous",
        choices=("contiguous", "zigzag", "zigzag_per_document"),
    )
    parser.add_argument("--boundary", default="B3", choices=BOUNDARIES)
    parser.add_argument(
        "--base",
        default="fla",
        choices=("fla", "local", "control"),
        help="the baseline arm of the ABBA pattern",
    )
    parser.add_argument("--save-mode", default="light", choices=("light", "full"))
    parser.add_argument(
        "--groups", type=int, default=3, help="measurement groups (at least 3)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repetitions k of the [cake, base, base, cake] pattern per group",
    )
    parser.add_argument(
        "--samples", type=int, default=8, help="timed iterations per call (at least 8)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="untimed iterations per call"
    )
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--json", type=str, default=None, help="rank 0 writes the report here"
    )
    args = parser.parse_args()
    if args.groups < 3 or args.samples < 8:
        parser.error("the protocol takes at least 3 groups and 8 samples per call")
    if args.v_heads % args.qk_heads:
        parser.error("--v-heads must be a multiple of --qk-heads")
    rank, world, device = _init()
    report = run(args, rank, world, device)
    if rank == 0:
        _print(report)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    ok = report["drift_ok"] and (report["control_ok"] is not False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
