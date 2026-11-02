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

Benchmark of the Cake packed KDA chunked training kernels
(``flashinfer.experimental.cake_kda_det_train``) on three boundaries of a
training step, against flash-linear-attention's ``chunk_kda`` when it is
installed:

* ``fwd``: the forward with the saved set written;
* ``bwd``: the backward from the saved set;
* ``ckpt``: ``torch.utils.checkpoint`` forward + replay + backward (one span).

Every row is one packed batch of ``--docs`` documents of ``--seqlen`` tokens
in total (unequal lengths with unaligned tails) and ``--heads`` heads; the
workspace is allocated once per row and reused, so the timed closures allocate
nothing.  Timing uses CUPTI kernel activity with a cold L2
(``flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)``);
the reported figure is the median over the timed calls of the summed GPU time
of a boundary's kernels, in milliseconds, with tokens per second beside it.

Example::

    python benchmarks/bench_cake_kda_det_train.py --seqlen 16384 65536 --docs 3 --heads 4 64
    python benchmarks/bench_cake_kda_det_train.py --seqlen 261393 --docs 4 --heads 4 --json out.json

The gate arms (``--gate-arms``) time one training step (forward + backward
through the autograd entry, CUDA events) of the forget gate's two forms on the
same raw operands (BF16 raw gate, FP32 ``A_log`` / ``dt_bias``, lower bound
-5), interleaved in one process (the arm order alternates per step):

* ``R``: the raw gate mode, ``chunk_kda_det(raw, use_gate_in_kernel=True, ...)``;
* ``E_T``: a torch FP32 gate with autograd, then ``chunk_kda_det(g)``;
* ``E_F``: flash-linear-attention's gate with autograd, then ``chunk_kda_det(g)``
  (where it imports).

The ``_strided`` arms (``R_strided``, ``E_T_strided``, ``E_F_strided``) take
the raw gate as a column slice of a fused projection (the leaf is the
projection, so each step also runs the slice's backward into it).

Per arm: the median step time and its interquartile range, the paired ratio
against each explicit arm with the upper end of its 95 % t-interval (over the
per-step log-ratios; ``matched`` when both arms read the raw gate in the same
layout: ``R`` against ``E_T`` / ``E_F``, ``R_strided`` against
``E_T_strided`` / ``E_F_strided``), the peak memory of one step above the inputs
(``max_memory_allocated`` after ``reset_peak_memory_stats``), the memory the
forward keeps for the backward, the arena bytes and the backend launch count (with
``--kernel-times STEPS``: the per-kernel and per-stage GPU time and the actual
launches per step of each arm, explicit gates' kernels included, from the torch
profiler's CUPTI kernel activity), under ``--ckpt`` (``torch.utils.checkpoint``), ``--save-mode`` and ``--arena``
(``default``, ``lean4``: four lean tail passes, ``batched``: groups of
``--batch-tokens`` tokens)::

    python benchmarks/bench_cake_kda_det_train.py --seqlen 16384 --docs 1 3 16 --heads 4 \
        --gate-arms R R_strided E_T E_F E_T_strided E_F_strided \
        --ckpt off on --save-mode light tape \
        --arena default lean4 batched --json gate.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import socket
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.cake_kda_det_train import cake_backend, cake_jit  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_kda_det_train_reference import (  # noqa: E402
    LOWER_BOUND,
    check_fla_gate,
    fla_kda_gate,
    make_inputs,
    make_raw_inputs,
    raw_gate,
    t_quantile,
)

BOUNDARIES = ("fwd", "bwd", "ckpt")
GATE_ARMS = ("R", "R_strided", "E_T", "E_F", "E_T_strided", "E_F_strided")
RAW_GATE_ARMS = ("R", "R_strided")
EXPLICIT_GATE_ARMS = ("E_T", "E_F", "E_T_strided", "E_F_strided")
STRIDED_SUFFIX = "_strided"
ARENAS = ("default", "lean4", "batched")


def gate_arm_kind(arm: str) -> tuple[str, bool]:
    """``(form, strided)`` of a gate arm: ``("E_F", True)`` for ``E_F_strided``."""
    if arm.endswith(STRIDED_SUFFIX):
        return arm[: -len(STRIDED_SUFFIX)], True
    return arm, False


def gate_ratio_pairs(arms) -> list[tuple[str, str, bool]]:
    """The ``(raw arm, explicit arm, matched)`` ratios of ``arms`` (in ``GATE_ARMS`` order): every raw arm against
    every explicit arm of the same layout (``matched``, the gated comparison), and ``R_strided`` also against the
    contiguous explicit arms (reported)."""
    present = [a for a in GATE_ARMS if a in arms]
    pairs = []
    for arm in present:
        if arm not in RAW_GATE_ARMS:
            continue
        strided = gate_arm_kind(arm)[1]
        for base in present:
            if base not in EXPLICIT_GATE_ARMS:
                continue
            matched = gate_arm_kind(base)[1] == strided
            if matched or strided:
                pairs.append((arm, base, matched))
    return pairs


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


def _median_ms(samples) -> float:
    return float(statistics.median(list(samples)))


def _time(fn: Callable[[], object]) -> float:
    return _median_ms(bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True))


def cake_boundaries(
    inputs: dict, ws: cake_backend.KdaDetWorkspace
) -> dict[str, Callable[[], None]]:
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )

    def fwd() -> None:
        cake_backend.chunk_kda_det_fwd(
            q, k, v, g, beta, cu, save_for_backward=True, workspace=ws
        )

    _, saved = cake_backend.chunk_kda_det_fwd(
        q, k, v, g, beta, cu, save_for_backward=True, workspace=ws
    )

    def bwd() -> None:
        cake_backend.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )

    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def ckpt() -> None:
        o = checkpoint(
            lambda *a: cake_backend.chunk_kda_det(*a, cu, workspace=ws),
            *leaves,
            use_reentrant=False,
        )
        torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt}


def fla_boundaries(inputs: dict) -> dict[str, Callable[[], None]] | None:
    """flash-linear-attention's ``chunk_kda`` on the same preactivated operands (no in-kernel L2 norm, gate or
    sigmoid), or ``None`` when the package is not installed."""
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda
    except Exception:
        return None
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )

    def call(q_, k_, v_, g_, beta_):
        o, _ = fla_chunk_kda(
            q_.unsqueeze(0),
            k_.unsqueeze(0),
            v_.unsqueeze(0),
            g_.unsqueeze(0),
            beta_.unsqueeze(0),
            scale=cake_backend.SCALE,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            safe_gate=True,
            cu_seqlens=cu,
        )
        return o.squeeze(0)

    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def fwd() -> None:
        call(*leaves)

    retained = call(*leaves)

    def bwd() -> None:
        torch.autograd.grad(retained, leaves, dout, retain_graph=True)

    def ckpt() -> None:
        o = checkpoint(call, *leaves, use_reentrant=False)
        torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt}


def run_row(
    seqlen: int, docs: int, heads: int, *, device: torch.device, baseline: bool
) -> dict:
    lengths = split_docs(seqlen, docs)
    inputs = make_inputs(lengths, heads, device)
    ws = cake_backend.KdaDetWorkspace.for_problem(seqlen, docs, heads, device)
    ws.load()
    row = {
        "seqlen": seqlen,
        "docs": docs,
        "lengths": list(lengths),
        "heads": heads,
        "arch": ws.arch,
        "records": ws.records(),
        "recurrence": ws.plan.recur,
        "segments_per_doc": ws.plan.P,
        "workspace_bytes": ws.bytes,
        "saved_bytes": cake_backend.saved_bytes(
            seqlen, heads, docs, sm_count=ws.plan.sm_count
        ),
        "cake_ms": {},
        "fla_ms": {},
    }
    cake = cake_boundaries(inputs, ws)
    for name in BOUNDARIES:
        cake[name]()
    torch.cuda.synchronize(device)
    for name in BOUNDARIES:
        row["cake_ms"][name] = _time(cake[name])
    if baseline:
        fla = fla_boundaries(inputs)
        if fla is not None:
            for name in BOUNDARIES:
                fla[name]()
            torch.cuda.synchronize(device)
            for name in BOUNDARIES:
                row["fla_ms"][name] = _time(fla[name])
    return row


def _torch_gate(raw, a_log, dt_bias):
    return raw_gate(raw, a_log, dt_bias, LOWER_BOUND, torch.float32)


def _arena_option(kind: str, batch_tokens: int):
    if kind == "default":
        return None
    if kind == "lean4":
        return cake_backend.KdaDetArena(lean=True, tail_passes=4)
    if kind == "batched":
        return cake_backend.KdaDetArena(batch_tokens=batch_tokens)
    raise ValueError(f"unknown arena {kind!r}; one of {ARENAS}")


def gate_steps(
    inputs: dict,
    arms: list[str],
    *,
    ckpt: bool,
    save_mode: str,
    arena,
    doc_lengths,
) -> tuple[dict[str, Callable[[], object]], dict[str, Callable[[], object]], dict]:
    """Per arm a training step (forward + backward from fresh gradients) and its forward alone (whose output the
    caller backpropagates or drops), on the same operands and ``dout``; the arms an environment cannot run are
    left out with the reason."""
    cu, dout = inputs["cu_seqlens"], inputs["dout"]
    S, H, D = inputs["g"].shape
    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
    kw = {"save_mode": save_mode, "arena": arena, "doc_lengths": doc_lengths}
    skipped: dict = {}
    # the raw gate as a column slice of a fused projection (the gate's columns after 8 others, 8 after it)
    proj = torch.zeros(S, 8 + H * D + 8, dtype=torch.bfloat16, device=dout.device)
    proj[:, 8 : 8 + H * D] = inputs["g"].reshape(S, H * D)
    fla = None
    fla_arms = [a for a in arms if gate_arm_kind(a)[0] == "E_F"]
    if fla_arms:
        # the adapted external gate, value-checked against the torch FP32 gate on this row's operands (the same
        # check as the accuracy protocol's): a mis-adapted call would time another computation
        fla, fla_info = fla_kda_gate(LOWER_BOUND)
        reason = fla_info.get("unavailable")
        if fla is not None:
            reason = check_fla_gate(fla, inputs, LOWER_BOUND)
        if reason is None and "E_F_strided" in arms:
            # the same check on the projection's column slice (a stride the adapter mishandles fails here)
            reason = check_fla_gate(
                fla, {**inputs, "g": proj[:, 8 : 8 + H * D].view(S, H, D)}, LOWER_BOUND
            )
            reason = None if reason is None else f"strided slice: {reason}"
        if reason is not None:
            fla = None
            for arm in fla_arms:
                skipped[arm] = f"flash-linear-attention's gate: {reason}"

    def make(arm):
        form, strided = gate_arm_kind(arm)
        leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]
        if strided:
            leaf = proj.detach().clone().requires_grad_(True)
            leaves[3] = leaf
        gate_fn = {"E_T": _torch_gate, "E_F": fla}.get(form)

        def forward():
            q, k, v, raw, beta, a_log, dt_bias = leaves
            if strided:
                raw = raw[:, 8 : 8 + H * D].view(S, H, D)

            def fn(q, k, v, raw, beta, a_log, dt_bias):
                if form == "R":
                    return cake_backend.chunk_kda_det(
                        q,
                        k,
                        v,
                        raw,
                        beta,
                        cu,
                        use_gate_in_kernel=True,
                        A_log=a_log,
                        dt_bias=dt_bias,
                        lower_bound=LOWER_BOUND,
                        **kw,
                    )
                g = gate_fn(raw, a_log, dt_bias).contiguous()
                return cake_backend.chunk_kda_det(q, k, v, g, beta, cu, **kw)

            if ckpt:
                return checkpoint(
                    fn, q, k, v, raw, beta, a_log, dt_bias, use_reentrant=False
                )
            return fn(q, k, v, raw, beta, a_log, dt_bias)

        def clear():
            for leaf in leaves:
                leaf.grad = None

        def step():
            clear()
            forward().backward(dout)

        step.clear = clear  # the memory pass drops the last step's gradients first
        return step, forward

    steps, forwards = {}, {}
    for arm in arms:
        if arm in skipped:
            continue
        try:
            step, forward = make(arm)
            step()  # the first call loads the programs (and refuses an option the row does not serve)
        except ValueError as exc:
            skipped[arm] = f"{exc}"
            continue
        except Exception as exc:  # the external gate's own failures leave its arm out
            if gate_arm_kind(arm)[0] != "E_F":
                raise
            skipped[arm] = f"{exc!r}"
            continue
        steps[arm], forwards[arm] = step, forward
    return steps, forwards, skipped


def _event_time_ms(fn: Callable[[], object]) -> float:
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _stage_of_symbol() -> dict[str, tuple[str, str]]:
    """The backend's kernel symbols: symbol -> (record, stage)."""
    return {
        rec["kernel_symbol"]: (name, rec.get("stage"))
        for name, rec in cake_jit.MODULES.items()
        if rec.get("kernel_symbol")
    }


def kernel_times(step: Callable[[], object], steps: int, device: torch.device) -> dict:
    """Per-kernel GPU time of ``steps`` training steps under the torch profiler (CUPTI kernel activity): per kernel
    name its backend record and stage (``external`` for the kernels outside the backend: the explicit gates'
    forward and backward, casts, reductions, memsets), its launches and total time per step; the totals by stage
    and the device launches per step."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    symbols = _stage_of_symbol()
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize(device)
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
        "launches_per_step": sum(r["launches"] for r in kernels.values()) / steps,
        "us_per_step": sum(r["us"] for r in kernels.values()) / steps,
    }


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    t = torch.tensor(xs, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return float(q[0]), float(q[1]), float(q[2])


def run_gate_row(
    seqlen: int,
    docs: int,
    heads: int,
    *,
    device: torch.device,
    arms: list[str],
    ckpt: bool,
    save_mode: str,
    arena_kind: str,
    batch_tokens: int,
    warmup: int,
    iters: int,
    kernel_steps: int = 0,
) -> dict:
    """One configuration of the gate arms: interleaved step times (the arm order alternates per step), the paired
    ratios, peak and retained memory, arena bytes and launch counts; with ``kernel_steps`` the per-kernel and
    per-stage GPU times and the actual launches of each arm (:func:`kernel_times`, after the timed loop)."""
    lengths = split_docs(seqlen, docs)
    inputs = make_raw_inputs(lengths, heads, device)
    arena = _arena_option(arena_kind, batch_tokens)
    doc_lengths = lengths if arena_kind == "batched" else None
    row: dict = {
        "seqlen": seqlen,
        "docs": docs,
        "lengths": list(lengths),
        "heads": heads,
        "ckpt": ckpt,
        "save_mode": save_mode,
        "arena": arena_kind,
        "batch_tokens": batch_tokens if arena_kind == "batched" else None,
        "warmup": warmup,
        "iters": iters,
        "arms": {},
    }
    steps, forwards, skipped = gate_steps(
        inputs,
        arms,
        ckpt=ckpt,
        save_mode=save_mode,
        arena=arena,
        doc_lengths=doc_lengths,
    )
    row["skipped"] = skipped
    if not steps:
        return row
    sm = cake_jit.sm_count(device.index)
    arch = cake_jit.device_arch(device.index)
    for gate in ("g", "raw"):
        p = cake_backend.plan(seqlen, docs, heads, sm, arch, gate=gate)
        info: dict = {
            "records_light": p.records[p.light_path],
            "launches_fwd": p.launch_count("fwd"),
            "launches_bwd": p.launch_count("bwd"),
        }
        try:
            info["workspace_bytes"] = cake_backend.workspace_bytes(
                seqlen,
                docs,
                heads,
                save_mode=save_mode,
                sm_count=sm,
                arch=arch,
                arena=arena,
                doc_lengths=doc_lengths,
                gate=gate,
            )
        except ValueError as exc:  # an arena option the row does not serve
            info["workspace_bytes"] = f"refused: {exc}"
        row[f"plan_{gate}"] = info
    for _ in range(warmup):
        for step in steps.values():
            step()
    torch.cuda.synchronize(device)
    times: dict[str, list[float]] = {arm: [] for arm in steps}
    order = list(steps)
    for i in range(iters):
        for arm in order if i % 2 == 0 else reversed(order):
            times[arm].append(_event_time_ms(steps[arm]))
    for arm, step in steps.items():
        q25, med, q75 = _quartiles(times[arm])
        # the peak of one step above what is allocated before it (inputs, leaves; no gradients)
        step.clear()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        step()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - base
        # what the forward keeps for the backward (beyond its output)
        base = torch.cuda.memory_allocated(device)
        out = forwards[arm]()
        torch.cuda.synchronize(device)
        retained = (
            torch.cuda.memory_allocated(device) - base - out.untyped_storage().nbytes()
        )
        out.backward(inputs["dout"])
        del out
        step.clear()
        row["arms"][arm] = {
            "times_ms": times[arm],
            "median_ms": med,
            "q25_ms": q25,
            "q75_ms": q75,
            "peak_step_bytes": int(peak),
            "retained_after_forward_bytes": int(retained),
        }
        if kernel_steps > 0:
            row["arms"][arm]["kernel_times"] = kernel_times(step, kernel_steps, device)
            step.clear()
    tq = t_quantile(0.975, iters - 1) if iters > 1 else float("nan")
    row["ratios"] = {}
    for arm, base_arm, matched in gate_ratio_pairs(steps):
        logs = [
            math.log(a / b) for a, b in zip(times[arm], times[base_arm], strict=True)
        ]
        mean = sum(logs) / len(logs)
        sd = (
            math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))
            if len(logs) > 1
            else float("nan")
        )
        row["ratios"][f"{arm}/{base_arm}"] = {
            "matched": matched,
            "median_ratio": statistics.median(
                a / b for a, b in zip(times[arm], times[base_arm], strict=True)
            ),
            "geomean_ratio": math.exp(mean),
            "ci95_upper": math.exp(mean + tq * sd / math.sqrt(len(logs))),
            "ci95_lower": math.exp(mean - tq * sd / math.sqrt(len(logs))),
            "peak_bytes_delta": row["arms"][arm]["peak_step_bytes"]
            - row["arms"][base_arm]["peak_step_bytes"],
            "retained_bytes_delta": row["arms"][arm]["retained_after_forward_bytes"]
            - row["arms"][base_arm]["retained_after_forward_bytes"],
        }
    return row


def _print_gate_row(row: dict) -> None:
    head = (
        f"S={row['seqlen']:>7} N={row['docs']} H={row['heads']:>2} ckpt={'on' if row['ckpt'] else 'off'} "
        f"{row['save_mode']} {row['arena']}"
    )
    cells = [head]
    for arm, rec in row["arms"].items():
        cells.append(
            f"{arm} {rec['median_ms']:8.3f} ms [{rec['q25_ms']:.3f}, {rec['q75_ms']:.3f}] "
            f"peak {rec['peak_step_bytes'] / 2**20:8.1f} MiB kept {rec['retained_after_forward_bytes'] / 2**20:7.1f} MiB"
        )
    for name, r in row.get("ratios", {}).items():
        cells.append(
            f"{name}{'' if r['matched'] else ' (layouts differ)'} x{r['median_ratio']:.3f} "
            f"(CI95 upper {r['ci95_upper']:.3f})"
        )
    for arm, why in row.get("skipped", {}).items():
        cells.append(f"{arm} skipped: {why}")
    print(" | ".join(cells))


def _print_row(row: dict) -> None:
    cells = [
        f"S={row['seqlen']:>7} N={row['docs']} H={row['heads']:>2} {row['arch']} "
        f"{row['recurrence']}{'/P' + str(row['segments_per_doc']) if row['recurrence'] == 'seg' else ''}"
    ]
    for name in BOUNDARIES:
        ms = row["cake_ms"][name]
        cell = f"{name} {ms:8.3f} ms ({row['seqlen'] * row['heads'] / ms / 1e3:8.1f} Mtok-head/s)"
        if name in row["fla_ms"]:
            cell += (
                f" fla {row['fla_ms'][name]:8.3f} ms x{row['fla_ms'][name] / ms:5.2f}"
            )
        cells.append(cell)
    print(" | ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--docs", type=int, nargs="+", default=[3])
    parser.add_argument("--heads", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip flash-linear-attention"
    )
    parser.add_argument("--json", type=str, default=None, help="write the rows here")
    parser.add_argument(
        "--gate-arms",
        nargs="*",
        default=None,
        choices=GATE_ARMS,
        help="time the forget gate's raw mode against the explicit gates instead",
    )
    parser.add_argument("--ckpt", nargs="+", default=["off"], choices=("off", "on"))
    parser.add_argument(
        "--save-mode", nargs="+", default=["light"], choices=("light", "tape")
    )
    parser.add_argument("--arena", nargs="+", default=["default"], choices=ARENAS)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--kernel-times",
        type=int,
        default=0,
        metavar="STEPS",
        help="gate arms: profile STEPS more steps per arm for per-kernel / per-stage GPU time and launches",
    )
    args = parser.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    arch = cake_jit.arch_for_capability(
        tuple(torch.cuda.get_device_capability(device.index))
    )
    if arch is None:
        raise SystemExit(
            f"{torch.cuda.get_device_name(device)} is not an exported target {cake_jit.ARCHES}"
        )
    rows = []
    for seqlen in args.seqlen:
        for docs in args.docs:
            for heads in args.heads:
                if args.gate_arms:
                    for ckpt, save_mode, arena in itertools.product(
                        args.ckpt, args.save_mode, args.arena
                    ):
                        row = run_gate_row(
                            seqlen,
                            docs,
                            heads,
                            device=device,
                            arms=list(args.gate_arms),
                            ckpt=ckpt == "on",
                            save_mode=save_mode,
                            arena_kind=arena,
                            batch_tokens=args.batch_tokens,
                            warmup=args.warmup,
                            iters=args.iters,
                            kernel_steps=args.kernel_times,
                        )
                        _print_gate_row(row)
                        rows.append(row)
                    continue
                row = run_row(
                    seqlen, docs, heads, device=device, baseline=not args.no_baseline
                )
                _print_row(row)
                rows.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "arch": arch,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "sm_count": cake_jit.sm_count(device.index),
                    "hostname": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "fla_commit": os.environ.get("FLA_COMMIT"),
                    "argv": sys.argv,
                    "rows": rows,
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
