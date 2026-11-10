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
(``flashinfer.cake_causal_conv1d_det_train_kcp.cake_causal_conv1d_det_kcp``) followed by
the KCP KDA step (``flashinfer.cake_kda_det_train_kcp.cake_kda_det_kcp``), forward and
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

    torchrun --nproc_per_node 4 benchmarks/bench_cake_kda_det_train_kcp.py \\
        --seqlen 65536 --docs 3 --heads 16 --channels 4096 --groups 3 --samples 8 --json kcp.json

The gate arms (``--gate-arms``) time instead the KCP KDA step alone (no convolution) of the forget gate's two
forms on the same raw operands (BF16 raw gate, FP32 ``A_log`` / ``dt_bias``, lower bound -5), every rank its
shard of the same packed batch, with the same groups and visits (in every group the order ``R, control, E_T,
E_F``, then the reverse):

* ``R``: the raw gate mode, ``cake_kda_det_kcp(raw, ..., use_gate_in_kernel=True, A_log=, dt_bias=, lower_bound=)``;
* ``control``: the ``R`` step again (its ratio to ``R`` must stay within the control band);
* ``E_T``: a torch FP32 gate with autograd on the rank's raw shard, then ``cake_kda_det_kcp(g)``;
* ``E_F``: flash-linear-attention's gate with autograd, then ``cake_kda_det_kcp(g)`` (where it imports and gives
  the torch gate's values on every rank).

A step is the forward and the backward from fresh gradients, then the cross-rank sum of the parameter gradients'
partial sums (``cake_kda_det_train_kcp_reduce_gate_param_grads``: the raw mode's partials in ``R``, the gate's autograd partials
in ``E_T`` / ``E_F``); every arm runs on a persistent ``CakeKDADetKcpWorkspace`` of its gate mode.  ``--ckpt`` times it
plain (``off``), under ``torch.utils.checkpoint(use_reentrant=False)`` (``on``: forward, recompute with the state
exchange, backward) and with a ``CakeKDADetKcpRecomputeHandle`` per step (``handle``: the recompute exchanges nothing).
Per row and arm: the median of the slowest rank's step time, its interquartile range and group medians, the ratio
of each explicit arm to ``R`` (above 1: ``R`` is faster), the control ratio and the drift of ``R``; per rank the
peak memory of one step above the inputs (``max_memory_allocated`` after ``reset_peak_memory_stats``) and the
memory the forward keeps for the backward; the workspace bytes of each gate mode.  ``--kernel-times STEPS``: rank
0's per-kernel GPU time per step from the torch profiler (``--trace-dir``: its chrome trace per row and arm);
``--alloc-trace``: rank 0's allocations of one step of at least a BF16 ``[T_r, H, 128]`` tensor (size and Python
frames, from the caching allocator's memory history)::

    torchrun --nproc_per_node 4 benchmarks/bench_cake_kda_det_train_kcp.py --seqlen 65536 --heads 64 \\
        --gate-arms R E_T E_F --ckpt off on handle --groups 5 --samples 16 --json gate.json

The Q/K L2 norm arms (``--l2n-arms``) time the KCP KDA step alone the same way (FP32 gate ``g``), on the same raw
BF16 ``q`` / ``k`` rows (in every group the order ``N, control, S_T, S_F``, then the reverse):

* ``N``: the in-kernel norm, ``cake_kda_det_kcp(q_raw, k_raw, ..., use_qk_l2norm_in_kernel=True)``;
* ``control``: the ``N`` step again;
* ``S_T``: a torch FP32 norm with autograd (``x / sqrt(sum x^2 + 1e-6)``, stored in BF16), then
  ``cake_kda_det_kcp`` with the flag off;
* ``S_F``: flash-linear-attention's norm with autograd (``fla.modules.l2norm.l2norm``, where it imports and gives
  the torch norm's values on every rank), then the flag-off step.

``--l2n-packed`` takes ``q`` / ``k`` as the column halves of one ``[T_r, 2 * H * 128]`` tensor (``N`` reads the
strided views, ``S_T`` normalizes them, ``S_F`` copies them to contiguous rows first).  The report is the gate
arms' with ``N`` in place of ``R``; ``--kernel-times`` shows the separate norms' kernels as ``external``::

    torchrun --nproc_per_node 4 benchmarks/bench_cake_kda_det_train_kcp.py --seqlen 131072 --heads 64 \\
        --layout zigzag_per_document --l2n-arms N S_T S_F --ckpt off on handle --groups 5 --samples 16 --json l2n.json
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
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.cake_causal_conv1d_det_train import (  # noqa: E402
    cake_causal_conv1d_det,
)
from flashinfer.cake_causal_conv1d_det_train_kcp import (  # noqa: E402
    cake_causal_conv1d_det_kcp,
)
from flashinfer.cake_kda_det_train_kcp import (  # noqa: E402
    CakeKDADetKcpRecomputeHandle,
    CakeKDADetKcpWorkspace,
    cake_kda_det_train_kcp_build_context,
    cake_kda_det_kcp,
    cake_kda_det_train_kcp_comm_bytes,
    cake_kda_det_train_kcp_saved_bytes,
    cake_kda_det_train_kcp_workspace_bytes,
    cake_kda_det_train_kcp_partition,
    cake_kda_det_train_kcp_reduce_gate_param_grads,
)
from flashinfer.cake_kda_det_train import cake_kda_det  # noqa: E402
from tests.experimental.test_helpers.cake_kda_det_train_reference import (  # noqa: E402
    LOWER_BOUND,
    check_fla_gate,
    fla_kda_gate,
    make_raw_inputs,
    raw_gate,
)
from tests.experimental.test_helpers.cake_kda_det_train_reference import make_inputs  # noqa: E402

ARMS = ("cake", "control", "local", "fla")
GATE_ARMS = ("R", "E_T", "E_F")
L2N_ARMS = ("N", "S_T", "S_F")
L2N_EPS = 1e-6  # under the square root, as in the in-kernel norm
L2N_ATOL = 2.0**-7  # BF16 rows of norm at most 1: two units in the last place
CKPT_MODES = ("off", "on", "handle")
D = 128  # head dimension
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
    part = cake_kda_det_train_kcp_partition(cu_cpu, world, args.layout)
    group = dist.group.WORLD
    ctx = cake_kda_det_train_kcp_build_context(part, group, rank, device)
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
    y = cake_causal_conv1d_det_kcp(x, w, ctx)
    out = cake_kda_det_kcp(q, k, v, g, beta, ctx)
    torch.autograd.backward([y, out], [local["dy"], local["dout"]])


def _step_local(problem: dict[str, Any]) -> None:
    local, weight = problem["local"], problem["weight"]
    x = local["x"].detach().requires_grad_(True)
    w = weight.detach().requires_grad_(True)
    q, k, v, g, beta = (
        local[n].detach().requires_grad_(True) for n in ("q", "k", "v", "g", "beta")
    )
    y = cake_causal_conv1d_det(x, w, local["cu_seqlens"])
    out = cake_kda_det(q, k, v, g, beta, local["cu_seqlens"])
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
            "workspace": cake_kda_det_train_kcp_workspace_bytes(
                ctx, H=args.heads, C=args.channels
            ),
            "saved": cake_kda_det_train_kcp_saved_bytes(
                ctx, H=args.heads, C=args.channels
            ),
            "comm": cake_kda_det_train_kcp_comm_bytes(
                ctx, H=args.heads, C=args.channels
            ),
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


# --------------------------------------------------------------------------- gate arms (#7816)


def _token_problem(
    args, lengths, inputs: dict, rank: int, world: int, device: torch.device
) -> dict[str, Any]:
    """This rank's shard of the whole batch's token tensors ``inputs`` (on the CPU, the same on every rank) on the
    device, and the context of the partition."""
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    part = cake_kda_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int32), world, args.layout
    )
    ctx = cake_kda_det_train_kcp_build_context(part, dist.group.WORLD, rank, device)
    fragments = part.fragments[rank]
    local = {
        name: _shard(inputs[name], cu, fragments).to(device)
        for name in ("q", "k", "v", "g", "beta", "dout")
    }
    return {
        "lengths": lengths,
        "ctx": ctx,
        "local": local,
        "tokens_local": int(local["q"].shape[0]),
        "tokens_total": args.seqlen,
    }


def make_gate_problem(
    args, rank: int, world: int, device: torch.device
) -> dict[str, Any]:
    """The raw-gate operands of the whole batch on the CPU (every rank draws the same tensors), this rank's shard of
    the token tensors on the device, ``A_log`` / ``dt_bias`` on every rank."""
    lengths = split_docs(args.seqlen, args.docs)
    inputs = make_raw_inputs(lengths, args.heads, device="cpu", seed=args.seed)
    problem = _token_problem(args, lengths, inputs, rank, world, device)
    problem["local"]["A_log"] = inputs["A_log"].to(device)
    problem["local"]["dt_bias"] = inputs["dt_bias"].to(device)
    return problem


def _torch_gate(raw, a_log, dt_bias):
    return raw_gate(raw, a_log, dt_bias, LOWER_BOUND, torch.float32)


def gate_steps(
    problem: dict[str, Any], arms, *, ckpt: str
) -> tuple[
    dict[str, Callable[[], object]], dict[str, Callable[[], object]], dict, dict
]:
    """Per arm a training step (forward + backward from fresh gradients + the cross-rank reduction of the parameter
    partial sums) and its forward alone, on the same operands and ``dout``; the arms left out with the reason, and
    the workspaces by gate mode."""
    local, ctx = problem["local"], problem["ctx"]
    dout = local["dout"]
    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
    workspaces = {
        "raw": CakeKDADetKcpWorkspace(use_gate_in_kernel=True),
        "g": CakeKDADetKcpWorkspace(),
    }
    gates: dict[str, Any] = {"E_T": _torch_gate}
    skipped: dict[str, str] = {}
    if "E_F" in arms:
        # the adapted external gate, value-checked against the torch FP32 gate on this rank's operands; the arm runs
        # only where every rank passes (the steps' collectives pair across the ranks)
        fla, info = fla_kda_gate(LOWER_BOUND)
        reason = info.get("unavailable")
        if fla is not None:
            reason = check_fla_gate(fla, local, LOWER_BOUND)
        ok = torch.tensor([int(reason is None)], dtype=torch.int32, device=dout.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item():
            gates["E_F"] = fla
        else:
            skipped["E_F"] = (
                f"flash-linear-attention's gate: {reason or 'refused on another rank'}"
            )

    def make(arm):
        leaves = [local[n].detach().clone().requires_grad_(True) for n in names]
        a_log, dt_bias = leaves[5], leaves[6]
        gate_fn = gates.get(arm)
        ws = workspaces["raw" if arm == "R" else "g"]

        def fn(q, k, v, raw, beta, a_log, dt_bias, *handle):
            kw = {"workspace": ws, "recompute_handle": handle[0] if handle else None}
            if arm == "R":
                return cake_kda_det_kcp(
                    q,
                    k,
                    v,
                    raw,
                    beta,
                    ctx,
                    use_gate_in_kernel=True,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    lower_bound=LOWER_BOUND,
                    **kw,
                )
            g = gate_fn(raw, a_log, dt_bias).contiguous()
            return cake_kda_det_kcp(q, k, v, g, beta, ctx, **kw)

        def forward():
            if ckpt == "off":
                return fn(*leaves)
            # one handle per step, an argument of the checkpointed function (the recompute binds it)
            handle = (CakeKDADetKcpRecomputeHandle(),) if ckpt == "handle" else ()
            return checkpoint(fn, *leaves, *handle, use_reentrant=False)

        def clear():
            for leaf in leaves:
                leaf.grad = None

        def step():
            clear()
            forward().backward(dout)
            # the same deterministic cross-rank sum of the parameter partials in every arm
            return cake_kda_det_train_kcp_reduce_gate_param_grads(
                ctx, a_log.grad, dt_bias.grad
            )

        step.clear = clear  # the memory pass drops the last step's gradients first
        return step, forward

    steps, forwards = {}, {}
    for arm in GATE_ARMS:
        if arm in arms and arm not in skipped:
            steps[arm], forwards[arm] = make(arm)
    return steps, forwards, skipped, workspaces


# --------------------------------------------------------------------------- Q/K L2 norm arms (#7817)


def make_l2n_problem(
    args, rank: int, world: int, device: torch.device
) -> dict[str, Any]:
    """The FP32-gate operands of the whole batch on the CPU with raw (not normalized) BF16 ``q`` / ``k`` rows, this
    rank's shard on the device."""
    lengths = split_docs(args.seqlen, args.docs)
    inputs = make_inputs(lengths, args.heads, device="cpu", seed=args.seed)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    for name in ("q", "k"):  # the norm is the arms' work
        inputs[name] = (
            torch.randn(args.seqlen, args.heads, D, generator=gen) * 2.0
        ).to(torch.bfloat16)
    return _token_problem(args, lengths, inputs, rank, world, device)


def _torch_l2norm(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + L2N_EPS)).to(x.dtype)


def fla_l2norm() -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, dict]:
    """flash-linear-attention's L2 norm with autograd (``fla.modules.l2norm.l2norm``, eps under the square root) and
    the package's provenance (as :func:`fla_kda_gate` records it), or ``None`` with the reason."""
    info = {k: v for k, v in fla_kda_gate(LOWER_BOUND)[1].items() if k != "unavailable"}
    try:
        from fla.modules.l2norm import l2norm  # type: ignore
    except Exception as exc:  # noqa: BLE001 -- the arm is optional
        info["unavailable"] = f"import: {exc!r}"
        return None, info
    return (lambda x: l2norm(x, eps=L2N_EPS)), info


def check_l2norm(norm, x: torch.Tensor) -> str | None:
    """``None`` when ``norm`` gives :func:`_torch_l2norm`'s BF16 rows on ``x`` within :data:`L2N_ATOL`, else why
    not."""
    try:
        with torch.no_grad():
            got, want = norm(x), _torch_l2norm(x)
    except Exception as exc:  # noqa: BLE001
        return f"call: {exc!r}"
    if got.shape != want.shape or got.dtype != want.dtype:
        return f"{got.dtype} {tuple(got.shape)} != {want.dtype} {tuple(want.shape)}"
    err = (got.float() - want.float()).abs().max().item() if x.numel() else 0.0
    if not err <= L2N_ATOL:
        return f"values differ from x / sqrt(sum x^2 + {L2N_EPS}) by {err}"
    return None


def l2n_steps(
    problem: dict[str, Any], arms, *, ckpt: str, packed: bool
) -> tuple[
    dict[str, Callable[[], object]], dict[str, Callable[[], object]], dict, dict
]:
    """Per arm a training step (forward + backward from fresh gradients) and its forward alone, on the same raw
    ``q`` / ``k`` rows and ``dout``; the arms left out with the reason, and the workspaces by mode.  ``packed``:
    ``q`` / ``k`` are the column halves of one ``[T_r, 2 * H * 128]`` tensor (``v`` stays contiguous; the flag-off
    route takes no strided operands)."""
    local, ctx = problem["local"], problem["ctx"]
    T, H = local["q"].shape[0], local["q"].shape[1]
    workspaces = {
        "in-kernel": CakeKDADetKcpWorkspace(use_qk_l2norm_in_kernel=True),
        "separate": CakeKDADetKcpWorkspace(),
    }
    norms: dict[str, Any] = {"S_T": _torch_l2norm}
    skipped: dict[str, str] = {}
    if "S_F" in arms:
        # the external norm, value-checked against the torch FP32 norm on this rank's rows; the arm runs only where
        # every rank passes (the steps' collectives pair across the ranks)
        fla, info = fla_l2norm()
        reason = info.get("unavailable")
        if fla is not None:
            reason = check_l2norm(fla, local["q"])
        ok = torch.tensor(
            [int(reason is None)], dtype=torch.int32, device=local["dout"].device
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item():
            norms["S_F"] = fla
        else:
            skipped["S_F"] = (
                f"flash-linear-attention's L2 norm: {reason or 'refused on another rank'}"
            )

    def make(arm):
        if packed:
            qk = torch.cat(
                (local["q"].reshape(T, H * D), local["k"].reshape(T, H * D)), 1
            )
            ops = [qk] + [local[n] for n in ("v", "g", "beta")]
        else:
            ops = [local[n] for n in ("q", "k", "v", "g", "beta")]
        leaves = [x.detach().clone().requires_grad_(True) for x in ops]
        ws = workspaces["in-kernel" if arm == "N" else "separate"]
        norm = norms.get(arm)

        def fn(*xs):
            handle = xs[len(leaves) :]
            if packed:
                qk, v, g, beta = xs[: len(leaves)]
                q = qk[:, : H * D].view(T, H, D)
                k = qk[:, H * D :].view(T, H, D)
            else:
                q, k, v, g, beta = xs[: len(leaves)]
            kw = {"workspace": ws, "recompute_handle": handle[0] if handle else None}
            if arm == "N":
                return cake_kda_det_kcp(
                    q, k, v, g, beta, ctx, use_qk_l2norm_in_kernel=True, **kw
                )
            # flash-linear-attention's norm reads contiguous rows (a no-op unless packed)
            if arm == "S_F":
                q, k = q.contiguous(), k.contiguous()
            return cake_kda_det_kcp(norm(q), norm(k), v, g, beta, ctx, **kw)

        def forward():
            if ckpt == "off":
                return fn(*leaves)
            handle = (CakeKDADetKcpRecomputeHandle(),) if ckpt == "handle" else ()
            return checkpoint(fn, *leaves, *handle, use_reentrant=False)

        def clear():
            for leaf in leaves:
                leaf.grad = None

        def step():
            clear()
            forward().backward(local["dout"])

        step.clear = clear
        return step, forward

    steps, forwards = {}, {}
    for arm in L2N_ARMS:
        if arm in arms and arm not in skipped:
            steps[arm], forwards[arm] = make(arm)
    return steps, forwards, skipped, workspaces


def _gather(obj) -> list:
    out: list = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _memory(step, forward, dout: torch.Tensor, device: torch.device) -> tuple[int, int]:
    """This rank's peak memory of one step above what is allocated before it (inputs, leaves, workspaces; no
    gradients), and what the forward keeps for the backward (beyond its output)."""
    step.clear()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    step.clear()
    base = torch.cuda.memory_allocated(device)
    out = forward()
    torch.cuda.synchronize(device)
    retained = (
        torch.cuda.memory_allocated(device) - base - out.untyped_storage().nbytes()
    )
    out.backward(dout)
    del out
    step.clear()
    return int(peak), int(retained)


def _stage_of_symbol() -> dict[str, tuple[str, str]]:
    """The kernel symbols of the local stages and of the KCP programs: symbol -> (record, stage)."""
    from flashinfer.experimental.cake.cake_kda_det_train import cake_jit
    from flashinfer.experimental.cake.cake_kda_det_train_kcp import cake_kcp_jit

    out = {}
    for default, modules in (
        ("local", cake_jit.MODULES),
        ("kcp", cake_kcp_jit.MODULES),
    ):
        for name, rec in modules.items():
            if rec.get("kernel_symbol"):
                out[rec["kernel_symbol"]] = (name, rec.get("stage") or default)
    return out


def kernel_times(
    step: Callable[[], object], steps: int, device: torch.device, trace: str | None
) -> dict:
    """Per-kernel GPU time of ``steps`` training steps of this rank under the torch profiler (CUPTI kernel
    activity): per kernel name its record and stage (``external`` for the kernels outside the Cake programs: the
    explicit gates' forward and backward, casts, reductions, collectives, memsets), its launches and time per
    step; the totals by stage.  ``trace``: write the chrome trace there."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    symbols = _stage_of_symbol()
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize(device)
    if trace is not None:
        prof.export_chrome_trace(trace)
    kernels: dict[str, dict] = {}
    for e in prof.events():
        if e.device_type != DeviceType.CUDA:
            continue
        rec = kernels.get(e.name)
        if rec is None:
            record, stage = next(
                (v for sym, v in symbols.items() if sym in e.name), (None, "external")
            )
            rec = kernels[e.name] = {
                "record": record,
                "stage": stage,
                "launches": 0,
                "us": 0.0,
            }
        rec["launches"] += 1
        rec["us"] += e.time_range.elapsed_us()
    by_stage: dict[str, dict] = {}
    for rec in kernels.values():
        rec["launches_per_step"] = rec["launches"] / steps
        rec["us_per_step"] = rec["us"] / steps
        agg = by_stage.setdefault(
            rec["stage"], {"launches_per_step": 0.0, "us_per_step": 0.0}
        )
        agg["launches_per_step"] += rec["launches_per_step"]
        agg["us_per_step"] += rec["us_per_step"]
    return {
        "steps": steps,
        "kernels": kernels,
        "stages": by_stage,
        "us_per_step": sum(r["us"] for r in kernels.values()) / steps,
    }


def large_allocations(
    step: Callable[[], object], device: torch.device, sizes: dict[str, int]
) -> dict:
    """This rank's allocations of one step through the caching allocator of at least the smallest of ``sizes``
    (name -> bytes, e.g. the BF16 and FP32 ``[T_r, H, 128]`` tensors), from its memory history: per allocation its
    bytes, the size name it equals and the innermost Python frames (none for the autograd engine's own nodes);
    the counts per size name."""
    step.clear()
    torch.cuda.synchronize(device)
    torch.cuda.memory._record_memory_history(max_entries=1 << 20, stacks="python")
    try:
        # the history this step adds (an earlier recording's entries stay in front)
        first = len(torch.cuda.memory._snapshot()["device_traces"][device.index])
        step()
        torch.cuda.synchronize(device)
        events = torch.cuda.memory._snapshot()["device_traces"][device.index][first:]
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    step.clear()
    named = {b: n for n, b in sizes.items()}
    allocs = []
    for ev in events:
        if ev["action"] != "alloc" or ev["size"] < min(sizes.values()):
            continue
        allocs.append(
            {
                "bytes": int(ev["size"]),
                "size": named.get(int(ev["size"]), "other"),
                "frames": [
                    f"{f['name']} ({Path(f['filename']).name}:{f['line']})"
                    for f in ev.get("frames", [])[:4]
                ],
            }
        )
    counts = {n: sum(a["size"] == n for a in allocs) for n in (*sizes, "other")}
    return {"sizes": sizes, "counts": counts, "allocations": allocs}


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    t = torch.tensor(xs, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return float(q[0]), float(q[1]), float(q[2])


def run_gate_row(
    args,
    problem: dict[str, Any],
    ckpt: str,
    flush: torch.Tensor,
    rank: int,
    device: torch.device,
) -> dict:
    """One ``--ckpt`` mode of the gate arms (``--l2n-arms``: of the Q/K L2 norm arms): the groups of interleaved
    visits, the medians and ratios to ``R`` (``N``), the control ratio and drift, per-rank peak and retained memory,
    the workspace bytes; optionally rank 0's kernel times and large allocations."""
    if args.l2n_arms is not None:
        ref, others = "N", ("S_T", "S_F")
        steps, forwards, skipped, workspaces = l2n_steps(
            problem, args.l2n_arms, ckpt=ckpt, packed=args.l2n_packed
        )
    else:
        ref, others = "R", ("E_T", "E_F")
        steps, forwards, skipped, workspaces = gate_steps(
            problem, args.gate_arms, ckpt=ckpt
        )
    steps["control"] = steps[ref]
    arms = [arm for arm in (ref, "control", *others) if arm in steps]
    for arm in arms:  # one untimed call per arm: the JIT builds, the first exchanges, the workspaces' sizing
        steps[arm]()
    torch.cuda.synchronize(device)
    dist.barrier()
    order = arms + arms[::-1]
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
    rows: dict[str, dict] = {}
    for arm in arms:
        q25, _, q75 = _quartiles([t for visit in raw[arm] for t in visit])
        rows[arm] = {
            "group_medians_ms": per_group[arm],
            "median_ms": statistics.median(per_group[arm]),
            "q25_ms": q25,
            "q75_ms": q75,
            "tokens_per_s": problem["tokens_total"]
            / (statistics.median(per_group[arm]) * 1e-3),
            "samples": raw[arm],
        }
    dout = problem["local"]["dout"]
    T, H = problem["tokens_local"], args.heads
    sizes = {"bf16_full": T * H * D * 2, "fp32_full": T * H * D * 4}
    for arm in steps:
        if arm == "control":
            continue
        peak, retained = _memory(steps[arm], forwards[arm], dout, device)
        mem = _gather((peak, retained))
        rows[arm]["peak_step_bytes_by_rank"] = [m[0] for m in mem]
        rows[arm]["retained_after_forward_bytes_by_rank"] = [m[1] for m in mem]
        rows[arm]["peak_step_bytes"] = max(m[0] for m in mem)
        rows[arm]["retained_after_forward_bytes"] = max(m[1] for m in mem)
        if args.kernel_times > 0:
            trace = None
            if args.trace_dir and rank == 0:
                Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
                trace = str(Path(args.trace_dir) / f"ckpt-{ckpt}-{arm}.json")
            rows[arm]["kernel_times"] = kernel_times(
                steps[arm], args.kernel_times, device, trace
            )
            steps[arm].clear()
        if args.alloc_trace:
            rows[arm]["large_allocations"] = large_allocations(
                steps[arm], device, sizes
            )
    r = rows[ref]["median_ms"]
    ratios = {
        f"{arm}/{ref}": {
            "median_ratio": rows[arm]["median_ms"] / r,
            "group_ratios": [
                e / b for e, b in zip(per_group[arm], per_group[ref], strict=True)
            ],
            "peak_bytes_delta": rows[arm]["peak_step_bytes"]
            - rows[ref]["peak_step_bytes"],
        }
        for arm in arms
        if arm not in (ref, "control")
    }
    control = rows["control"]["median_ms"] / r
    drift = abs(per_group[ref][-1] - per_group[ref][0]) / r
    return {
        "ckpt": ckpt,
        "ref": ref,
        "arms": rows,
        "ratios": ratios,
        "skipped": skipped,
        "workspace_bytes": {
            gate: max(_gather(ws.bytes)) for gate, ws in workspaces.items()
        },
        "control_ratio": control,
        "control_band": list(CONTROL_BAND),
        "control_ok": CONTROL_BAND[0] <= control <= CONTROL_BAND[1],
        "drift": drift,
        "drift_ok": drift <= 0.02,
    }


def run_gate(args, rank: int, world: int, device: torch.device) -> dict[str, Any]:
    l2n = args.l2n_arms is not None
    problem = (make_l2n_problem if l2n else make_gate_problem)(
        args, rank, world, device
    )
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    rows = [
        run_gate_row(args, problem, ckpt, flush, rank, device) for ckpt in args.ckpt
    ]
    if l2n:
        fla = fla_l2norm()[1] if "S_F" in args.l2n_arms else None
        family = {
            "family": "l2n",
            "l2n_arms": list(args.l2n_arms),
            "l2n_packed": args.l2n_packed,
            "eps": L2N_EPS,
        }
    else:
        fla = fla_kda_gate(LOWER_BOUND)[1] if "E_F" in args.gate_arms else None
        family = {
            "family": "gate",
            "gate_arms": list(args.gate_arms),
            "lower_bound": LOWER_BOUND,
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "config": {
            "seqlen": args.seqlen,
            "docs": args.docs,
            "lengths": list(problem["lengths"]),
            "heads": args.heads,
            "layout": args.layout,
            "world_size": world,
            "tokens_by_rank": _gather(problem["tokens_local"]),
            **family,
            "groups": args.groups,
            "samples": args.samples,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "rows": rows,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": props.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "sm_count": props.multi_processor_count,
            "ranks": world,
            "hosts": _gather(socket.gethostname()),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "fla": None
            if fla is None
            else {k: fla.get(k) for k in ("version", "git_commit", "commit_env")},
            "argv": sys.argv,
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


def _print_gate(report: dict[str, Any]) -> None:
    cfg = report["config"]
    packed = " packed q/k" if cfg.get("l2n_packed") else ""
    print(
        f"KCP x{cfg['world_size']} {cfg['family']} arms{packed}: S={cfg['seqlen']} docs={cfg['docs']} H={cfg['heads']} "
        f"T_r={cfg['tokens_by_rank']} layout={cfg['layout']} groups={cfg['groups']} samples={cfg['samples']}"
    )
    for row in report["rows"]:
        print(f"  ckpt={row['ckpt']}  workspace bytes {row['workspace_bytes']}")
        for arm, rec in row["arms"].items():
            groups = " ".join(f"{t:.3f}" for t in rec["group_medians_ms"])
            mem = (
                f"  peak {rec['peak_step_bytes'] / 2**20:8.1f} MiB kept {rec['retained_after_forward_bytes'] / 2**20:7.1f} MiB"
                if "peak_step_bytes" in rec
                else ""
            )
            print(
                f"    {arm:8s} {rec['median_ms']:9.3f} ms [{rec['q25_ms']:.3f}, {rec['q75_ms']:.3f}] groups [{groups}]{mem}"
            )
        for name, r in row["ratios"].items():
            groups = " ".join(f"{x:.3f}" for x in r["group_ratios"])
            print(f"    {name} x{r['median_ratio']:.3f} groups [{groups}]")
        for arm, why in row["skipped"].items():
            print(f"    {arm} skipped: {why}")
        ref = row["ref"]
        print(
            f"    control/{ref} = {row['control_ratio']:.4f} (band {row['control_band']}) -> {'ok' if row['control_ok'] else 'OUT OF BAND'}"
            f"; drift({ref}) = {row['drift']:.4f} -> {'ok' if row['drift_ok'] else 'ABOVE 2%'}"
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
    parser.add_argument(
        "--gate-arms",
        nargs="+",
        default=None,
        choices=GATE_ARMS,
        help="time the forget gate's raw mode (R, required) against the explicit gates instead (KDA step only)",
    )
    parser.add_argument(
        "--l2n-arms",
        nargs="+",
        default=None,
        choices=L2N_ARMS,
        help="time the in-kernel Q/K L2 norm (N, required) against the separate norms instead (KDA step only)",
    )
    parser.add_argument(
        "--l2n-packed",
        action="store_true",
        help="l2n arms: q / k are the column halves of one [T_r, 2 * H * 128] tensor",
    )
    parser.add_argument(
        "--ckpt",
        nargs="+",
        default=["off"],
        choices=CKPT_MODES,
        help="gate / l2n arms: plain step, checkpointed step, checkpointed step with a recompute handle",
    )
    parser.add_argument(
        "--kernel-times",
        type=int,
        default=0,
        metavar="STEPS",
        help="gate arms: profile STEPS more steps per arm for rank 0's per-kernel GPU time",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="gate arms with --kernel-times: rank 0 writes a chrome trace per row and arm here",
    )
    parser.add_argument(
        "--alloc-trace",
        action="store_true",
        help="gate arms: rank 0's allocations of one step of at least a BF16 [T_r, H, 128] tensor",
    )
    args = parser.parse_args()
    if args.groups < 3 or args.samples < 8:
        parser.error("the protocol takes at least 3 groups and 8 samples per visit")
    if args.gate_arms is not None and "R" not in args.gate_arms:
        parser.error("the gate arms take the R arm (the control repeats it)")
    if args.l2n_arms is not None and "N" not in args.l2n_arms:
        parser.error("the l2n arms take the N arm (the control repeats it)")
    if args.gate_arms is not None and args.l2n_arms is not None:
        parser.error("--gate-arms and --l2n-arms are separate runs")
    if args.l2n_packed and args.l2n_arms is None:
        parser.error("--l2n-packed applies to the l2n arms")
    if (args.gate_arms, args.l2n_arms) == (None, None) and (
        "cake" not in args.arms or "control" not in args.arms
    ):
        parser.error("the cake and control arms are required")
    rank, world, device = _init()
    if args.gate_arms is not None or args.l2n_arms is not None:
        report = run_gate(args, rank, world, device)
        ok = all(row["control_ok"] and row["drift_ok"] for row in report["rows"])
        show = _print_gate
    else:
        report = run(args, rank, world, device)
        ok = report["control_ok"] and report["drift_ok"]
        show = _print
    if rank == 0:
        show(report)
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
