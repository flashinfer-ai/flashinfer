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

Benchmark of the context-parallel (KCP) Cake packed KDA training step on both
numerical modes: one process per rank (``torchrun --nproc_per_node N``), every
rank timing the module boundary of its fragments -- the KCP causal convolution
(``flashinfer.cake_causal_conv1d_train.cake_causal_conv1d_kcp``, the same
program in every arm; ``--no-conv`` leaves it out) followed by the KCP KDA
step -- with CUDA events, and the slowest rank's time counting for the
iteration (the boundary of ``benchmarks/bench_cake_kda_det_train_kcp.py``).

Arms (``--arms``), every one the same packed batch split by the same partition
(``fla_cp``: by FLA's own, which gives every rank the tokens of its shard):

* ``det``: the deterministic package, the objects of
  ``flashinfer.cake_kda_det_train_kcp`` (what ``deterministic=True`` of
  ``flashinfer.cake_kda_train_kcp`` runs), with its own partition and context;
* ``nd``: ``flashinfer.cake_kda_train_kcp`` with ``deterministic=False`` (the
  package ``flashinfer.experimental.cake_kda_train_kcp``), with its own
  partition and context;
* ``control``: the ``nd`` path again (the same programs; its ratio to ``nd``
  measures the protocol's own noise and must stay within the control band);
* ``local``: the single-rank non-deterministic Cake kernels
  (``flashinfer.cake_kda_train.chunk_kda`` with ``deterministic=False``, after
  the single-rank convolution) on the rank's shard as if it were a batch of its
  own (no exchange; the per-rank work without the boundaries);
* ``fla``: flash-linear-attention's ``chunk_kda`` on the rank's shard (after the
  single-rank Cake convolution), when the package is installed: the non-Cake
  figure of the per-rank work, comparable with ``local``.  The arm fails soft:
  one untimed ``step`` runs on every rank first, and when it raises on any rank
  every rank drops the arm, the report records ``fla_error`` (the first failing
  rank, the exception type and message) and the other arms go on.  FLA takes
  the operands as given (``scale=None``, no in-kernel L2 norm, gate or beta
  sigmoid) with ``safe_gate=True``, its path for gates bounded below (they lie
  in ``(-5, 0)``; ``--no-fla-safe-gate`` runs its general path instead).
* ``fla_cp`` (run only when ``--arms`` names it): flash-linear-attention's
  context-parallel KDA (its ``chunk_kda`` with the ``cp_context`` of
  ``fla.ops.cp.build_cp_context``, built once) on FLA's own equal-token
  partition of the same batch, after the KCP convolution of the main arms
  (the shard's ``x`` on the ``nd`` context): FLA's context-parallel figure of
  the KCP step (``fla`` stays the per-rank work without the exchange).
  ``--layout contiguous`` maps to FLA's ``contiguous`` (rank ``r`` holds part
  ``r`` of ``W``), ``zigzag`` to FLA's ``zigzag`` (parts ``r`` and ``2W-1-r``
  of ``2W`` as ``[front; back]``, the assignment of the bench's zigzag); FLA's
  equal parts are then the partition's near-equal ones, so every rank holds
  the tokens of its Cake shard, in the same order.  The operands are those of
  ``fla``, sliced from the whole batch.  It is dropped on every rank, with the
  reason in ``fla_cp_error``, with ``zigzag_per_document``, when ``seqlen`` is
  not a multiple of ``W`` (contiguous) / ``2W`` (zigzag), when ``fla.ops.cp``
  is missing, and like ``fla`` when an untimed ``step`` raises on a rank: first
  one on a one-rank group per rank (every kernel of FLA's step compiles there,
  those FLA runs on some ranks only included), then one with the exchange.  A
  failure inside the exchange itself can differ by rank: it leaves the ranks'
  collectives unpaired and aborts the run.

Boundaries (``--boundaries``): ``fwd`` the forward with autograd recording;
``bwd`` the backward of a forward run untimed before it; ``step`` forward +
backward (the deterministic benchmark's boundary); ``ckpt``
``torch.utils.checkpoint(use_reentrant=False)`` forward + recompute + backward
(two forwards and the backward).

Protocol, per boundary: one untimed call per arm (the JIT builds and the first
exchanges happen here) and one probe call per arm (peak memory), then
``--groups`` groups (at least 3); in every group the arms run in the order
det, nd, control, fla_cp, fla_cp, control, nd, det (palindromic; ``fla_cp``
when it runs), then the other optional arms; every visit takes ``--warmup``
untimed and ``--samples`` timed iterations (at least 8), each after an L2
flush, a device synchronize and a barrier; the figure of an arm in a group is
the median of its timed iterations over its visits, the figure of an arm is
the median over the groups.  The report gives, per
boundary and arm, the group medians, the overall median, p10 / p90 of the
samples, tokens per second and the probe's peak memory (the largest over the
ranks); the ratios ``det / nd`` (above 1: the non-deterministic arm is faster)
and ``<arm> / nd`` of the optional arms; the control ratio against its band
(0.98 .. 1.02) and the drift of the ``nd`` arm between its first and last group.

Before the timing, one ``step`` of ``det`` and one of ``nd`` on the same shard
are compared: ``out``, ``dq``, ``dk``, ``dv``, ``dbeta`` (and ``y``, ``dx``,
``dweight`` with the convolution) must be bitwise equal; ``dg`` differs by FP32
rounding on the ranks whose backward runs the ``S1`` pair (its rel-RMS and
max-abs difference are reported).  With ``fla_cp``, one untimed forward of it
and of ``det`` compares ``out`` in global token order (``fla_cp_out``: rel-RMS
and max-abs difference, recorded only).  The report also lists, per rank, the
local backward records (``dqkg`` and ``glm_intra`` on the replay path) the plan
of each package selects.  Rank 0 prints the table and writes ``--json``.  The
exit status is 0 when every boundary's control ratio is in band and its drift
is at most 2 %, and the bitwise comparison holds.

Example::

    torchrun --nproc_per_node 4 benchmarks/bench_cake_kda_train_kcp.py \\
        --seqlen 65536 --docs 3 --heads 16 --channels 4096 --groups 3 --samples 8 \\
        --boundaries fwd bwd step ckpt --json kcp_nd.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.cake_causal_conv1d_train import (  # noqa: E402
    cake_causal_conv1d,
    cake_causal_conv1d_kcp,
)
from flashinfer.cake_kda_det_train_kcp import (  # noqa: E402
    build_context_det_kcp,
    chunk_kda_det_kcp,
    kda_det_comm_bytes_kcp,
    kda_det_saved_bytes_kcp,
    kda_det_workspace_bytes_kcp,
    partition_det_kcp,
)
from flashinfer.cake_kda_train import chunk_kda  # noqa: E402
from flashinfer.cake_kda_train_kcp import (  # noqa: E402
    build_context_kcp,
    chunk_kda_kcp,
    kda_comm_bytes_kcp,
    kda_saved_bytes_kcp,
    kda_workspace_bytes_kcp,
    partition_kcp,
)
from flashinfer.experimental.cake_kda_det_train_kcp import (  # noqa: E402
    cake_kcp as det_kcp,
)
from flashinfer.experimental.cake_kda_train_kcp import (  # noqa: E402
    cake_kcp as nd_kcp,
)
from tests.test_helpers.cake_kda_det_train_reference import make_inputs  # noqa: E402

ARMS = ("det", "nd", "control", "local", "fla", "fla_cp")
MAIN_ARMS = ("det", "nd", "control")
BOUNDARIES = ("fwd", "bwd", "step", "ckpt")
CONTROL_BAND = (0.98, 1.02)
DRIFT_LIMIT = 0.02
L2_FLUSH_BYTES = 256 << 20
WIDTH = 4
INPUTS = ("q", "k", "v", "g", "beta")
BITWISE = ("out", "dq", "dk", "dv", "dbeta", "y", "dx", "dweight")
RECORD_STAGES = ("dqkg", "glm_intra")

Step = tuple[Callable[[], Any], Callable[[Any], Any]]


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


def _fragment_key(part) -> list[list[tuple[int, int, int]]]:
    return [[(f.doc, f.off, f.length) for f in frags] for frags in part.fragments]


def make_problem(args, rank: int, world: int, device: torch.device) -> dict[str, Any]:
    """The whole batch on the CPU (every rank draws the same tensors; with the ``fla_cp`` arm its KDA operands stay
    as ``global`` for FLA's partition), this rank's shard on the device and its fragments, and one partition and
    context per package (the two partitions must give every rank the same fragments)."""
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
    group = dist.group.WORLD
    part_det = partition_det_kcp(cu_cpu, world, args.layout)
    part_nd = partition_kcp(cu_cpu, world, args.layout, deterministic=False)
    if _fragment_key(part_det) != _fragment_key(part_nd):
        raise RuntimeError(
            "the two packages' partitions of the batch differ; the arms would not time the same shards"
        )
    ctx = {
        "det": build_context_det_kcp(part_det, group, rank, device),
        "nd": build_context_kcp(part_nd, group, rank, device, deterministic=False),
    }
    fragments = part_nd.fragments[rank]
    local = {
        name: _shard(inputs[name], cu, fragments).to(device)
        for name in (*INPUTS, "dout")
    }
    local["x"] = _shard(x, cu, fragments).to(device)
    local["dy"] = _shard(dy, cu, fragments).to(device)
    local_cu = [0]
    for f in fragments:
        local_cu.append(local_cu[-1] + f.length)
    local["cu_seqlens"] = torch.tensor(local_cu, dtype=torch.int32, device=device)
    problem: dict[str, Any] = {
        "lengths": lengths,
        "cu": cu,
        "ctx": ctx,
        "weight": weight,
        "local": local,
        "fragments": fragments,
        "tokens_local": int(local["q"].shape[0]),
        "tokens_total": args.seqlen,
        "fla_safe_gate": not args.no_fla_safe_gate,
    }
    if "fla_cp" in args.arms:
        problem["global"] = {name: inputs[name] for name in (*INPUTS, "dout")}
    return problem


def _leaves(problem: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    """Fresh leaves of the rank's shard: ``x, weight, q, k, v, g, beta``."""
    local = problem["local"]
    x = local["x"].detach().requires_grad_(True)
    w = problem["weight"].detach().requires_grad_(True)
    qkvgb = tuple(local[n].detach().requires_grad_(True) for n in INPUTS)
    return (x, w, *qkvgb)


def _fla_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda  # type: ignore
    except Exception:  # noqa: BLE001 -- any import failure means the optional arm is absent
        return None
    return fla_chunk_kda


def _fla_arm(problem: dict[str, Any], conv: bool, device: torch.device):
    """``(chunk_kda, fla_error)`` of the ``fla`` arm after one untimed ``step`` on every rank.  When the package is
    absent on a rank or the step raises on one, every rank drops the arm (``chunk_kda`` is ``None``; the timing's
    collectives stay matched), and an exception is reported as ``fla_error`` (the first failing rank, the exception
    type and message)."""
    fla_chunk_kda = _fla_chunk_kda()
    error = None
    if fla_chunk_kda is not None:
        try:
            setup, timed = _step(
                _forward("fla", problem, conv, fla_chunk_kda), problem, "step"
            )
            timed(setup())
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 -- any failure of the optional arm drops it
            error = f"{type(exc).__name__}: {exc}"
    states: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(states, (fla_chunk_kda is not None, error))
    errors = [f"rank {r}: {e}" for r, (_, e) in enumerate(states) if e is not None]
    if errors:
        return None, errors[0]
    if not all(present for present, _ in states):
        return None, None
    return fla_chunk_kda, None


def _first_error(error: str | None) -> str | None:
    """The first failing rank's ``error`` (``"rank r: ..."``) of an all-gather over the ranks, ``None`` if none."""
    errors: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    return next((f"rank {r}: {e}" for r, e in enumerate(errors) if e is not None), None)


def _fla_cp_arm(args, problem: dict[str, Any], conv: bool, device: torch.device):
    """``(chunk_kda, view, fla_cp_error)`` of the ``fla_cp`` arm.  ``view`` is the arm's problem: this rank's buffer
    of FLA's equal-token partition (rank ``r`` holds part ``r`` of ``W``, or parts ``r`` and ``2W-1-r`` of ``2W`` as
    ``[front; back]`` with ``zigzag``) sliced from the whole batch (``problem["global"]``, released here), ``dout``
    alike, the context of ``fla.ops.cp.build_cp_context`` built once, its local ``cu_seqlens`` and the global row of
    every local row (the convolution's ``x``, ``dy``, weight and context stay the shard's).  The arm is dropped
    (``chunk_kda`` and ``view`` are ``None``) with the reason as ``fla_cp_error``: with a layout without an FLA
    counterpart or a ``seqlen`` FLA's parts do not divide (read from the arguments, the same on every rank); when the
    import or the context fails on a rank, then when an untimed ``step`` raises on one (each outcome all-gathered, as
    for ``fla``): first on a context of a one-rank group per rank that makes the rank neither first nor last with no
    rank to merge, so that every kernel of FLA's step compiles on every rank before an exchange (FLA runs some on some
    ranks only: the first rank merges no carried state, the last exports none); then with the exchange, which runs
    only when every rank holds the arm and whose own failures can differ by rank (the collectives are then unpaired
    and the run aborts)."""
    batch = problem.pop("global")
    world, rank = dist.get_world_size(), dist.get_rank()
    parts = {"contiguous": world, "zigzag": 2 * world}.get(args.layout)
    if parts is None:
        return (
            None,
            None,
            f"layout {args.layout!r} has no FLA counterpart (FLA cuts 'contiguous' or 'zigzag')",
        )
    if args.seqlen % parts:
        return (
            None,
            None,
            f"seqlen {args.seqlen} is not a multiple of {parts}, the equal parts of FLA's {args.layout!r} layout",
        )
    error = None
    try:
        from fla.ops.cp import build_cp_context  # type: ignore
    except Exception as exc:  # noqa: BLE001 -- any import failure means FLA's context parallelism is absent
        error = f"fla.ops.cp unavailable: {type(exc).__name__}: {exc}"
    fla_chunk_kda = _fla_chunk_kda()
    if error is None and fla_chunk_kda is None:
        error = "fla.ops.kda.chunk_kda unavailable"
    view = None
    if error is None:
        try:
            length = args.seqlen // parts
            pieces = (rank,) if parts == world else (rank, parts - 1 - rank)
            spans = [(i * length, (i + 1) * length) for i in pieces]
            cu_cpu = torch.tensor(problem["cu"], dtype=torch.int32)
            cp_ctx = build_cp_context(
                cu_cpu.to(device),
                dist.group.WORLD,
                cu_seqlens_cpu=cu_cpu,
                layout=args.layout,
            )
            tokens = length * len(pieces)
            if int(cp_ctx.cu_seqlens_cpu[-1]) != tokens:
                raise RuntimeError(
                    f"FLA's context holds {int(cp_ctx.cu_seqlens_cpu[-1])} local tokens, its parts {tokens}"
                )
            # x, dy, the weight and the nd context stay the shard's: the convolution
            # is the main arms' program
            local = dict(problem["local"])
            for name in (*INPUTS, "dout"):
                local[name] = torch.cat([batch[name][s:e] for s, e in spans]).to(device)
            local["cu_seqlens"] = cp_ctx.cu_seqlens
            view = {
                **problem,
                "ctx": {**problem["ctx"], "fla_cp": cp_ctx},
                "local": local,
                "rows": torch.cat([torch.arange(s, e) for s, e in spans]).to(device),
                "tokens_local": tokens,
            }
        except Exception as exc:  # noqa: BLE001 -- any failure of the optional arm drops it
            error = f"{type(exc).__name__}: {exc}"
    error = _first_error(error)
    if error is None:
        # FLA runs some kernels of its step on some ranks only, so a failure to compile
        # one would come at different points of the exchange and unpair the ranks'
        # collectives: one untimed step without the convolution on a one-rank group,
        # every rank neither first nor last with no rank to merge, compiles and tunes
        # all of them on every rank first
        try:
            solo, _ = dist.new_subgroups(group_size=1)
            alone = replace(
                view["ctx"]["fla_cp"],
                group=solo,
                is_first_rank=False,
                is_last_rank=False,
                pre_num_ranks=0,
                post_num_ranks=0,
                is_first_by_part=(False, False),
                is_last_by_part=(False, False),
                pre_num_ranks_by_part=(0, 0),
                post_num_ranks_by_part=(0, 0),
            )
            warm = {**view, "ctx": {**view["ctx"], "fla_cp": alone}}
            setup, timed = _step(
                _forward("fla_cp", warm, False, fla_chunk_kda), warm, "step"
            )
            timed(setup())
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 -- any failure of the optional arm drops it
            error = f"{type(exc).__name__}: {exc}"
        error = _first_error(error)
    if error is None:
        try:
            setup, timed = _step(
                _forward("fla_cp", view, conv, fla_chunk_kda), view, "step"
            )
            timed(setup())
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 -- any failure of the optional arm drops it
            error = f"{type(exc).__name__}: {exc}"
        error = _first_error(error)
    if error is not None:
        return None, None, error
    return fla_chunk_kda, view, None


def _forward(arm: str, problem: dict[str, Any], conv: bool, fla_chunk_kda=None):
    """The module forward of ``arm``: ``(x, weight, q, k, v, g, beta) -> (y or None, out)``."""
    cu = problem["local"]["cu_seqlens"]
    if arm == "det":
        ctx = problem["ctx"]["det"]

        def fwd(x, w, q, k, v, g, beta):
            y = cake_causal_conv1d_kcp(x, w, ctx) if conv else None
            return y, chunk_kda_det_kcp(q, k, v, g, beta, ctx)

    elif arm in ("nd", "control"):
        ctx = problem["ctx"]["nd"]

        def fwd(x, w, q, k, v, g, beta):
            y = cake_causal_conv1d_kcp(x, w, ctx) if conv else None
            return y, chunk_kda_kcp(q, k, v, g, beta, ctx, deterministic=False)

    elif arm == "local":

        def fwd(x, w, q, k, v, g, beta):
            y = cake_causal_conv1d(x, w, cu) if conv else None
            return y, chunk_kda(q, k, v, g, beta, cu, deterministic=False)

    elif arm == "fla":
        safe_gate = problem["fla_safe_gate"]

        def fwd(x, w, q, k, v, g, beta):
            y = cake_causal_conv1d(x, w, cu) if conv else None
            out, _ = fla_chunk_kda(
                *(t.unsqueeze(0) for t in (q, k, v, g, beta)),
                scale=None,
                safe_gate=safe_gate,
                cu_seqlens=cu,
            )
            return y, out.squeeze(0)

    elif arm == "fla_cp":
        ctx, cp_ctx = problem["ctx"]["nd"], problem["ctx"]["fla_cp"]
        safe_gate = problem["fla_safe_gate"]

        def fwd(x, w, q, k, v, g, beta):
            # the convolution of the main arms on the shard, FLA's KDA on its own
            # partition (its cu_seqlens are the context's)
            y = cake_causal_conv1d_kcp(x, w, ctx) if conv else None
            out, _ = fla_chunk_kda(
                *(t.unsqueeze(0) for t in (q, k, v, g, beta)),
                scale=None,
                safe_gate=safe_gate,
                cp_context=cp_ctx,
            )
            return y, out.squeeze(0)

    else:
        raise ValueError(f"unknown arm {arm!r}")
    return fwd


def _step(fwd, problem: dict[str, Any], boundary: str) -> Step:
    """``(setup, timed)`` of ``boundary``: ``setup`` runs untimed before every iteration, ``timed`` is the span."""
    local = problem["local"]

    def backward(y, out) -> None:
        outs, grads = [out], [local["dout"]]
        if y is not None:
            outs.append(y)
            grads.append(local["dy"])
        torch.autograd.backward(outs, grads)

    def leaves():
        return _leaves(problem)

    if boundary == "fwd":
        return leaves, lambda xs: fwd(*xs)
    if boundary == "bwd":
        return (lambda: fwd(*_leaves(problem))), lambda yo: backward(*yo)
    if boundary == "step":
        return leaves, lambda xs: backward(*fwd(*xs))
    if boundary == "ckpt":
        return leaves, lambda xs: backward(*checkpoint(fwd, *xs, use_reentrant=False))
    raise ValueError(f"unknown boundary {boundary!r}")


def _time_visit(
    step: Step, samples: int, warmup: int, flush: torch.Tensor, device: torch.device
) -> list[float]:
    """The slowest rank's milliseconds of ``samples`` timed iterations (after ``warmup`` untimed ones)."""
    setup, timed = step
    times: list[float] = []
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for i in range(warmup + samples):
        state = setup()
        flush.fill_(i & 0xFF)
        torch.cuda.synchronize(device)
        dist.barrier()
        start.record()
        result = timed(state)
        end.record()
        torch.cuda.synchronize(device)
        del state, result
        t = torch.tensor([start.elapsed_time(end)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if i >= warmup:
            times.append(float(t.item()))
    return times


def _probe_memory(step: Step, device: torch.device) -> dict[str, int]:
    """Peak bytes of one call above the allocation before it, and the bytes still held at its end while its inputs
    and results are alive (for ``fwd``: the outputs and the saved set); the largest over the ranks."""
    setup, timed = step
    torch.cuda.synchronize(device)
    dist.barrier()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    state = setup()
    result = timed(state)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    held = torch.cuda.memory_allocated(device) - base
    del state, result
    t = torch.tensor([peak, held], dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return {"peak_bytes": int(t[0].item()), "held_bytes": int(t[1].item())}


def _agreement(
    problem: dict[str, Any], conv: bool, device: torch.device
) -> dict[str, Any]:
    """One ``step`` of ``det`` and of ``nd`` on the same shard: the bitwise tensors and the ``dg`` difference."""
    results: dict[str, dict[str, torch.Tensor]] = {}
    local = problem["local"]
    for arm in ("det", "nd"):
        x, w, q, k, v, g, beta = xs = _leaves(problem)
        y, out = _forward(arm, problem, conv)(*xs)
        outs, grads = [out], [local["dout"]]
        if y is not None:
            outs.append(y)
            grads.append(local["dy"])
        torch.autograd.backward(outs, grads)
        res = {
            "out": out.detach(),
            "dq": q.grad,
            "dk": k.grad,
            "dv": v.grad,
            "dg": g.grad,
            "dbeta": beta.grad,
        }
        if conv:
            res.update(y=y.detach(), dx=x.grad, dweight=w.grad)
        results[arm] = res
    torch.cuda.synchronize(device)
    det, nd = results["det"], results["nd"]
    names = [n for n in BITWISE if n in det]
    equal = torch.tensor(
        [
            int(det[n] is not None and nd[n] is not None and torch.equal(det[n], nd[n]))
            for n in names
        ],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(equal, op=dist.ReduceOp.MIN)
    a, b = det["dg"].double(), nd["dg"].double()
    diff = b - a
    sums = torch.stack([(diff * diff).sum(), (a * a).sum()])
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    max_abs = (
        diff.abs().max()
        if diff.numel()
        else torch.zeros((), dtype=torch.float64, device=device)
    )
    max_abs = max_abs.reshape(1).clone()
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    bitwise = {n: bool(e) for n, e in zip(names, equal.tolist(), strict=True)}
    return {
        "bitwise": bitwise,
        "bitwise_ok": all(bitwise.values()),
        "dg_rel_rms": float((sums[0] / sums[1].clamp_min(1e-300)).sqrt().item()),
        "dg_max_abs": float(max_abs.item()),
        "dg_bitwise": bool(sums[0].item() == 0.0),
    }


def _fla_cp_agreement(
    problem: dict[str, Any], view: dict[str, Any], fwd, device: torch.device
) -> dict[str, float]:
    """One untimed forward of ``fla_cp`` and of ``det``: ``fla_cp``'s ``out`` gathered in global token order (every
    rank writes its rows at their global positions in a zero buffer, summed over the ranks: exact, one rank writes
    each row) against ``det``'s at the global positions of every rank's fragments; the rel-RMS and max-abs difference
    over the batch."""
    with torch.no_grad():
        _, out = fwd(*_leaves(view))
        _, ref = _forward("det", problem, False)(*_leaves(problem))
    full = out.new_zeros((problem["tokens_total"], *out.shape[1:]))
    full[view["rows"]] = out
    dist.all_reduce(full, op=dist.ReduceOp.SUM)
    rows = _shard(
        torch.arange(problem["tokens_total"]), problem["cu"], problem["fragments"]
    )
    a = ref.double()
    diff = full[rows.to(device)].double() - a
    sums = torch.stack([(diff * diff).sum(), (a * a).sum()])
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    max_abs = (
        diff.abs().max()
        if diff.numel()
        else torch.zeros((), dtype=torch.float64, device=device)
    )
    max_abs = max_abs.reshape(1).clone()
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    return {
        "rel_rms": float((sums[0] / sums[1].clamp_min(1e-300)).sqrt().item()),
        "max_abs": float(max_abs.item()),
    }


def _local_records(
    problem: dict[str, Any], heads: int, device: torch.device
) -> list[dict[str, Any]]:
    """Per rank, the local backward records (``dqkg``, ``glm_intra`` on the replay path) the plan of each package
    selects for the rank's local problem (``None`` on a rank without tokens)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    mine: dict[str, Any] = {}
    for arm, mod in (("det", det_kcp), ("nd", nd_kcp)):
        ctx = problem["ctx"][arm]
        if int(ctx.T_r) == 0:
            mine[arm] = None
            continue
        p = mod.cake_backend.plan(
            int(ctx.T_r),
            int(ctx.n_frag),
            heads,
            mod.cake_jit.sm_count(index),
            mod.cake_jit.device_arch(index),
        )
        mine[arm] = {
            stage: p.record(stage, mod.LOCAL_PATH_BWD) for stage in RECORD_STAGES
        }
    mine["arch"] = nd_kcp.cake_jit.device_arch(index)
    mine["host"] = socket.gethostname()
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    return gathered


def _percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def run_boundary(
    args,
    problem: dict[str, Any],
    boundary: str,
    steps: dict[str, Step],
    device: torch.device,
) -> dict[str, Any]:
    arms = list(steps)
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    for arm in (
        arms
    ):  # one untimed call per arm: the JIT builds and the first exchanges happen here
        setup, timed = steps[arm]
        timed(setup())
    torch.cuda.synchronize(device)
    dist.barrier()
    memory = {arm: _probe_memory(steps[arm], device) for arm in arms}
    # fla_cp, nd's context-parallel baseline, joins the palindrome of the main arms
    main = [arm for arm in (*MAIN_ARMS, "fla_cp") if arm in steps]
    order = main + main[::-1] + [arm for arm in arms if arm not in main]
    per_group: dict[str, list[float]] = {arm: [] for arm in arms}
    raw: dict[str, list[list[float]]] = {arm: [] for arm in arms}
    for _ in range(args.groups):
        visits: dict[str, list[float]] = {arm: [] for arm in arms}
        for arm in order:
            visits[arm].extend(
                _time_visit(steps[arm], args.samples, args.warmup, flush, device)
            )
        for arm in arms:
            per_group[arm].append(statistics.median(visits[arm]))
            raw[arm].append(visits[arm])
    medians = {arm: statistics.median(per_group[arm]) for arm in arms}
    tokens = problem["tokens_total"]
    rows = {}
    for arm in arms:
        flat = [t for visit in raw[arm] for t in visit]
        rows[arm] = {
            "group_medians_ms": per_group[arm],
            "median_ms": medians[arm],
            "p10_ms": _percentile(flat, 0.1),
            "p90_ms": _percentile(flat, 0.9),
            "tokens_per_s": tokens / (medians[arm] * 1e-3),
            "memory": memory[arm],
            "samples": raw[arm],
        }
    ratios = {"det/nd": medians["det"] / medians["nd"]}
    for arm in arms:
        if arm not in MAIN_ARMS:
            ratios[f"{arm}/nd"] = medians[arm] / medians["nd"]
    control = medians["control"] / medians["nd"]
    drift = abs(per_group["nd"][-1] - per_group["nd"][0]) / medians["nd"]
    return {
        "arms": rows,
        "ratios": ratios,
        "control_ratio": control,
        "control_ok": CONTROL_BAND[0] <= control <= CONTROL_BAND[1],
        "drift": drift,
        "drift_ok": drift <= DRIFT_LIMIT,
    }


def run(args, rank: int, world: int, device: torch.device) -> dict[str, Any]:
    problem = make_problem(args, rank, world, device)
    conv = not args.no_conv
    fla_chunk_kda, fla_error = (
        _fla_arm(problem, conv, device) if "fla" in args.arms else (None, None)
    )
    fla_cp_chunk_kda, fla_cp_problem, fla_cp_error = (
        _fla_cp_arm(args, problem, conv, device)
        if "fla_cp" in args.arms
        else (None, None, None)
    )
    kda = {"fla": fla_chunk_kda, "fla_cp": fla_cp_chunk_kda}
    arms = [
        arm
        for arm in ARMS
        if arm in args.arms and (arm not in kda or kda[arm] is not None)
    ]
    # the fla_cp arm runs on its own view of the batch (FLA's partition and context)
    problems = {arm: fla_cp_problem if arm == "fla_cp" else problem for arm in arms}
    forwards = {arm: _forward(arm, problems[arm], conv, kda.get(arm)) for arm in arms}
    agreement = _agreement(problem, conv, device)
    agreement["fla_cp_out"] = (
        _fla_cp_agreement(problem, fla_cp_problem, forwards["fla_cp"], device)
        if "fla_cp" in forwards
        else None
    )
    records = _local_records(problem, args.heads, device)
    boundaries = {
        boundary: run_boundary(
            args,
            problem,
            boundary,
            {arm: _step(forwards[arm], problems[arm], boundary) for arm in arms},
            device,
        )
        for boundary in args.boundaries
    }
    hosts: list[Any] = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    H, C = args.heads, args.channels
    ctx_det, ctx_nd = problem["ctx"]["det"], problem["ctx"]["nd"]
    nbytes = {
        "det": {
            "workspace": kda_det_workspace_bytes_kcp(ctx_det, H=H, C=C),
            "saved": kda_det_saved_bytes_kcp(ctx_det, H=H, C=C),
            "comm": kda_det_comm_bytes_kcp(ctx_det, H=H, C=C),
        },
        "nd": {
            "workspace": kda_workspace_bytes_kcp(ctx_nd, H=H, C=C, deterministic=False),
            "saved": kda_saved_bytes_kcp(ctx_nd, H=H, C=C, deterministic=False),
            "comm": kda_comm_bytes_kcp(ctx_nd, H=H, C=C, deterministic=False),
        },
    }
    try:
        import fla  # type: ignore

        fla_version = getattr(fla, "__version__", "unknown")
    except Exception:  # noqa: BLE001 -- the optional package is absent
        fla_version = None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "config": {
            "seqlen": args.seqlen,
            "docs": args.docs,
            "lengths": list(problem["lengths"]),
            "heads": args.heads,
            "channels": args.channels,
            "conv": conv,
            "fla_safe_gate": problem["fla_safe_gate"],
            "layout": args.layout,
            "world_size": world,
            "arms": arms,
            "boundaries": list(args.boundaries),
            "groups": args.groups,
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "fla_error": fla_error,
        "fla_cp_error": fla_cp_error,
        "bytes": nbytes,
        "agreement": agreement,
        "local_records": records,
        "boundaries": boundaries,
        "control_band": list(CONTROL_BAND),
        "drift_limit": DRIFT_LIMIT,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fla": fla_version,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "arch": nd_kcp.cake_jit.device_arch(index),
            "sm_count": nd_kcp.cake_jit.sm_count(index),
            "ranks": world,
            "hosts": hosts,
            "argv": list(sys.argv),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def _print(report: dict[str, Any]) -> None:
    cfg = report["config"]
    print(
        f"KCP x{cfg['world_size']}: S={cfg['seqlen']} docs={cfg['docs']} H={cfg['heads']} C={cfg['channels']} "
        f"conv={cfg['conv']} layout={cfg['layout']} groups={cfg['groups']} samples={cfg['samples']} "
        f"({report['env']['device']}, {report['env']['arch']})"
    )
    agree = report["agreement"]
    bad = [n for n, ok in agree["bitwise"].items() if not ok]
    print(
        f"  nd vs det: bitwise {'ok' if agree['bitwise_ok'] else 'MISMATCH ' + ','.join(bad)}; "
        f"dg rel_rms={agree['dg_rel_rms']:.3e} max_abs={agree['dg_max_abs']:.3e}"
    )
    for rank, rec in enumerate(report["local_records"]):
        print(f"  rank {rank} ({rec['arch']}): det {rec['det']}  nd {rec['nd']}")
    if report["fla_error"]:
        print(f"  fla arm dropped: {report['fla_error'][:200]}")
    if report["fla_cp_error"]:
        print(f"  fla_cp arm dropped: {report['fla_cp_error'][:200]}")
    if agree["fla_cp_out"] is not None:
        print(
            f"  fla_cp vs det: out rel_rms={agree['fla_cp_out']['rel_rms']:.3e} "
            f"max_abs={agree['fla_cp_out']['max_abs']:.3e}"
        )
    for boundary, res in report["boundaries"].items():
        print(f"  [{boundary}]")
        for arm, row in res["arms"].items():
            groups = " ".join(f"{t:.3f}" for t in row["group_medians_ms"])
            print(
                f"    {arm:8s} {row['median_ms']:9.3f} ms  p10 {row['p10_ms']:9.3f}  p90 {row['p90_ms']:9.3f}  "
                f"{row['tokens_per_s'] / 1e6:8.3f} Mtok/s  peak {row['memory']['peak_bytes'] / 2**20:9.1f} MiB  "
                f"groups [{groups}]"
            )
        ratios = "  ".join(f"{k} = {v:.4f}" for k, v in res["ratios"].items())
        print(f"    {ratios}")
        print(
            f"    control/nd = {res['control_ratio']:.4f} (band {report['control_band']}) -> "
            f"{'ok' if res['control_ok'] else 'OUT OF BAND'}; drift(nd) = {res['drift']:.4f} -> "
            f"{'ok' if res['drift_ok'] else 'ABOVE 2%'}"
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
        "--no-conv",
        action="store_true",
        help="time the KDA step alone (no causal convolution before it)",
    )
    parser.add_argument(
        "--no-fla-safe-gate",
        action="store_true",
        help="run FLA's chunk_kda (fla, fla_cp) without safe_gate, on its general path",
    )
    parser.add_argument(
        "--layout",
        default="contiguous",
        choices=("contiguous", "zigzag", "zigzag_per_document"),
    )
    parser.add_argument(
        "--boundaries", nargs="+", default=list(BOUNDARIES), choices=BOUNDARIES
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
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[arm for arm in ARMS if arm != "fla_cp"],
        choices=ARMS,
        help="arms to time (default: all but fla_cp, which runs only when named)",
    )
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument(
        "--json", type=str, default=None, help="rank 0 writes the report here"
    )
    args = parser.parse_args()
    if args.groups < 3 or args.samples < 8:
        parser.error("the protocol takes at least 3 groups and 8 samples per visit")
    if not set(MAIN_ARMS) <= set(args.arms):
        parser.error("the det, nd and control arms are required")
    if torch.are_deterministic_algorithms_enabled():
        parser.error(
            "torch's deterministic algorithms are enabled: deterministic=False would run the deterministic package"
        )
    rank, world, device = _init()
    report = run(args, rank, world, device)
    if rank == 0:
        _print(report)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    ok = report["agreement"]["bitwise_ok"] and all(
        res["control_ok"] and res["drift_ok"] for res in report["boundaries"].values()
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
