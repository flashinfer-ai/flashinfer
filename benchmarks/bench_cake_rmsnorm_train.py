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

Benchmark of the Cake BF16 RMSNorm training kernels with the opt-in
non-deterministic weight gradient (``deterministic=False``) against the
deterministic default (``deterministic=True``: the deterministic package's main
kernel and its fixed-order ``dw`` reduction) wherever the deterministic package
serves the shape.

For every shape (hidden size, token count, residual flag, weight dtype) and
mode the benchmark records:

* GPU time of the call's own kernels for ``fwd``, ``bwd`` and ``step`` (two
  forwards and one backward: a training step with activation checkpointing).
  Each is captured once in a CUDA graph and every replay is traced with CUPTI
  after an L2 flush: ``span`` is the first kernel's start to the last kernel's
  end of one replay (no host launch gap), ``kernel_sum`` the sum of the
  kernels' durations; the kernel names and counts of a replay are kept.
* Host time of one eager call with the stream idle before it (the Python and
  launch work; an eager step that does not capture the call pays the larger of
  host and GPU time).
* Memory: the workspace bytes the mode documents for the shape; the peak and
  retained ``torch.cuda.memory_allocated`` deltas of a default call (no output
  buffers, no workspace), of a call with preallocated outputs and workspace,
  and of the autograd wrapper (what its forward keeps for the backward, and the
  backward's peak); and what a ``torch.utils.checkpoint(use_reentrant=False)``
  forward of the autograd wrapper keeps beyond its outputs.

Each sample set is summarized as median, p10 and p90.  ``--jsonl`` appends one
JSON object per (shape, mode) with the raw per-replay samples; run the
benchmark in two or more processes (sessions) and merge them with
``--report``, which pools the samples per shape and mode, prints the median and
p10 / p90 over all sessions next to each session's median, and writes a
markdown table.

Example::

    python benchmarks/bench_cake_rmsnorm_train.py --hidden 6144 --rows 256 4096 65536 \\
        --residual --w-dtype bf16 f32 --jsonl s1.jsonl --session s1
    python benchmarks/bench_cake_rmsnorm_train.py --report s1.jsonl s2.jsonl --markdown table.md
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime
import json
import os
import statistics
import time
from typing import Callable, Optional

import torch
import torch.utils.checkpoint

from flashinfer.cake_rmsnorm_train import (
    cake_rmsnorm,
    cake_rmsnorm_train_backward,
    cake_rmsnorm_train_backward_workspace,
    cake_rmsnorm_train_backward_workspace_bytes,
    cake_rmsnorm_train_forward,
    is_cake_rmsnorm_train_supported,
)
from flashinfer.cake_rmsnorm_train_backend import cake_jit

SCHEMA = "cake_rmsnorm_train_bench/1"
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-5),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
EPS = {hidden: eps for hidden, eps in WIDTHS}
ROWS = (256, 1024, 4096, 16384, 32768, 65536)
MODES = (("det", True), ("nondet", False))
W_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
PHASES = ("fwd", "bwd", "step")


# --------------------------------------------------------------------------- statistics


def _summary(values: list[float]) -> dict:
    """``{"median", "p10", "p90", "n"}`` of ``values`` (p10 / p90 by the
    inclusive method; the median itself for fewer than two values)."""

    if not values:
        return {"median": None, "p10": None, "p90": None, "n": 0}
    median = float(statistics.median(values))
    if len(values) < 2:
        return {"median": median, "p10": median, "p90": median, "n": len(values)}
    deciles = statistics.quantiles(values, n=10, method="inclusive")
    return {
        "median": median,
        "p10": float(deciles[0]),
        "p90": float(deciles[-1]),
        "n": len(values),
    }


# --------------------------------------------------------------------------- GPU time (CUPTI)


def _cupti_replays(
    runner: Callable[[], object], *, iters: int, warmup: int, device
) -> list[list[tuple[str, int, int]]]:
    """The kernels of each of ``iters`` calls of ``runner`` as ``[(name, start
    ns, end ns), ...]`` in start order, from CUPTI activity records matched to
    the call's own launches by correlation id; the L2 is flushed (a buffer of
    twice its size zeroed) and the device synchronized before every call."""

    from cupti import cupti

    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    flush = torch.empty(2 * l2_bytes, dtype=torch.int8, device=device)
    for _ in range(warmup):
        flush.zero_()
        runner()
    torch.cuda.synchronize()
    launches: list[tuple[int, int]] = []
    kernels: dict[int, list[tuple[str, int, int]]] = collections.defaultdict(list)
    copies = {
        cupti.ActivityKind.MEMSET: "MEMSET",
        cupti.ActivityKind.MEMCPY: "MEMCPY",
    }

    def requested():
        return 8 * 1024 * 1024, 0

    def completed(activities):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                kernels[activity.correlation_id].append(
                    (activity.name, activity.start, activity.end)
                )
            elif activity.kind in copies:
                kernels[activity.correlation_id].append(
                    (copies[activity.kind], activity.start, activity.end)
                )
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                launches.append((activity.start, activity.correlation_id))

    kinds = (
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    )
    for kind in kinds:
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(requested, completed)
    stamps = []
    try:
        for _ in range(iters):
            flush.zero_()
            torch.cuda.synchronize()
            begin = cupti.get_timestamp()
            runner()
            end = cupti.get_timestamp()
            torch.cuda.synchronize()
            stamps.append((begin, end))
        cupti.activity_flush_all(0)
    finally:
        for kind in kinds:
            cupti.activity_disable(kind)
        cupti.finalize()
    launches.sort()
    starts = [start for start, _corr in launches]
    replays = []
    for begin, end in stamps:
        lo, hi = bisect.bisect_left(starts, begin), bisect.bisect_right(starts, end)
        correlation = {launches[index][1] for index in range(lo, hi)}
        found = sorted(
            (k for corr in correlation for k in kernels.get(corr, ())),
            key=lambda k: k[1],
        )
        if not found:
            raise RuntimeError("CUPTI recorded no kernel of a replay")
        replays.append(found)
    return replays


def _graph(calls: list[Callable[[], object]]) -> torch.cuda.CUDAGraph:
    """One CUDA graph of ``calls`` in order (warmed up on a side stream first)."""

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    torch.cuda.synchronize()
    return graph


def _gpu_time(calls, *, iters: int, warmup: int, device) -> dict:
    graph = _graph(calls)
    replays = _cupti_replays(graph.replay, iters=iters, warmup=warmup, device=device)
    spans = [(max(k[2] for k in r) - min(k[1] for k in r)) / 1e3 for r in replays]
    sums = [sum(k[2] - k[1] for k in r) / 1e3 for r in replays]
    names = collections.Counter(k[0] for k in replays[0])
    return {
        "span_us": _summary(spans),
        "kernel_sum_us": _summary(sums),
        "kernels": sorted(names.items()),
        "samples_span_us": [round(v, 4) for v in spans],
    }


def _host_time(fn: Callable[[], object], runs: int) -> dict:
    """Wall time (us) of one ``fn`` call with the stream idle before it."""

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e6)
    torch.cuda.synchronize()
    return _summary(times)


# --------------------------------------------------------------------------- memory


def _memory(fn: Callable[[], object]) -> tuple[int, int]:
    """``(peak, retained)`` bytes of ``torch.cuda.memory_allocated`` above its
    value before ``fn``: the peak during the call, and what the call's result
    still holds after it."""

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    retained = torch.cuda.memory_allocated() - base
    del result
    return int(peak), int(retained)


# --------------------------------------------------------------------------- one shape


def _logical_bytes(rows: int, hidden: int, *, residual: bool, w_bytes: int) -> dict:
    fwd = 4 * rows * hidden + 4 * rows + w_bytes * hidden
    fwd += 4 * rows * hidden if residual else 0
    bwd = 6 * rows * hidden + 4 * rows + (w_bytes + 4) * hidden
    bwd += 2 * rows * hidden if residual else 0
    return {"fwd": fwd, "bwd": bwd, "step": 2 * fwd + bwd}


def _form(device, hidden: int, rows: int, residual: bool, w_token: str) -> str:
    """The backward form the non-deterministic call of ``rows`` tokens runs."""

    arch = cake_jit.device_arch(device.index or 0)
    names = cake_jit.selected_modules(
        arch, hidden, "bwd", residual, rows=rows, w_dtype=w_token
    )
    reduces = [n for n in names if cake_jit.MODULES[n]["role"] == "bwd_reduce"]
    return "det pair" if reduces else "single"


def _inputs(hidden: int, rows: int, residual: bool, w_dtype, device):
    generator = torch.Generator(device=device).manual_seed(8066 + rows + hidden)

    def randn(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=generator
        )

    x = randn(rows, hidden).to(torch.bfloat16)
    w = (1.0 + 0.1 * randn(hidden)).to(w_dtype)
    g = randn(rows, hidden).to(torch.bfloat16)
    u = randn(rows, hidden).to(torch.bfloat16) if residual else None
    g_h = randn(rows, hidden).to(torch.bfloat16) if residual else None
    return x, w, g, u, g_h


def bench_mode(
    hidden: int,
    rows: int,
    *,
    residual: bool,
    w_token: str,
    deterministic: bool,
    device,
    iters: int,
    warmup: int,
    host_runs: int,
) -> dict:
    """Every measurement of one (shape, mode); ``supported: False`` with the
    reason when the mode has no program for the shape."""

    eps = EPS[hidden]
    x, w, g, u, g_h = _inputs(hidden, rows, residual, W_DTYPES[w_token], device)
    kw = {"deterministic": deterministic}
    entry: dict = {"supported": True}
    try:
        workspace_bytes = cake_rmsnorm_train_backward_workspace_bytes(
            rows, hidden, device, residual=residual, **kw
        )
        cake_rmsnorm_train_forward(x, w, eps, residual=u, **kw)
    except (ValueError, RuntimeError) as exc:
        return {"supported": False, "reason": str(exc).splitlines()[0][:300]}
    y = torch.empty_like(x)
    rstd = torch.empty(rows, dtype=torch.float32, device=device)
    h_new = torch.empty_like(x) if residual else None
    dx = torch.empty_like(x)
    dw = torch.empty(hidden, dtype=torch.float32, device=device)
    workspace = cake_rmsnorm_train_backward_workspace(
        rows, hidden, device, residual=residual, **kw
    )
    normalized = h_new if residual else x

    def fwd():
        return cake_rmsnorm_train_forward(
            x, w, eps, residual=u, out=y, rstd=rstd, residual_out=h_new, **kw
        )

    def bwd():
        return cake_rmsnorm_train_backward(
            g,
            normalized,
            w,
            rstd,
            g_residual=g_h,
            workspace=workspace,
            dx=dx,
            dw=dw,
            **kw,
        )

    try:
        fwd()
        bwd()
        torch.cuda.synchronize()
    except (ValueError, RuntimeError) as exc:
        return {"supported": False, "reason": str(exc).splitlines()[0][:300]}
    entry["gpu"] = {
        "fwd": _gpu_time([fwd], iters=iters, warmup=warmup, device=device),
        "bwd": _gpu_time([bwd], iters=iters, warmup=warmup, device=device),
        "step": _gpu_time([fwd, fwd, bwd], iters=iters, warmup=warmup, device=device),
    }
    entry["host_us"] = {
        "fwd": _host_time(fwd, host_runs),
        "bwd": _host_time(bwd, host_runs),
    }
    fwd_peak, fwd_kept = _memory(
        lambda: cake_rmsnorm_train_forward(x, w, eps, residual=u, **kw)
    )
    bwd_peak, bwd_kept = _memory(
        lambda: cake_rmsnorm_train_backward(
            g, normalized, w, rstd, g_residual=g_h, **kw
        )
    )
    pre_peak, _pre_kept = _memory(bwd)
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    ua = u.clone().requires_grad_(True) if residual else None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    outputs = cake_rmsnorm(xa, wa, eps, residual=ua, **kw)
    torch.cuda.synchronize()
    outputs_bytes = sum(
        t.numel() * t.element_size() for t in (outputs if residual else (outputs,))
    )
    saved = torch.cuda.memory_allocated() - base - outputs_bytes

    tensors = outputs if residual else (outputs,)
    grads = (g, g_h) if residual else (g,)
    ag_peak, _ag_kept = _memory(lambda: torch.autograd.backward(tensors, grads))
    # What a ``torch.utils.checkpoint(use_reentrant=False)`` forward keeps beyond its
    # outputs (expected 0: the checkpoint saves its inputs, which the caller holds).
    xc = x.clone().requires_grad_(True)
    wc = w.clone().requires_grad_(True)
    uc = u.clone().requires_grad_(True) if residual else None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    checkpointed = torch.utils.checkpoint.checkpoint(
        lambda a, b, c: cake_rmsnorm(a, b, eps, residual=c, **kw),
        xc,
        wc,
        uc,
        use_reentrant=False,
    )
    torch.cuda.synchronize()
    checkpointed = checkpointed if residual else (checkpointed,)
    checkpoint_retained = (
        torch.cuda.memory_allocated()
        - base
        - sum(t.untyped_storage().nbytes() for t in checkpointed)
    )
    del checkpointed
    entry["memory"] = {
        "workspace_bytes": int(workspace_bytes),
        "fwd_default_peak_bytes": fwd_peak,
        "fwd_default_retained_bytes": fwd_kept,
        "bwd_default_peak_bytes": bwd_peak,
        "bwd_default_retained_bytes": bwd_kept,
        "bwd_default_temporary_bytes": bwd_peak - bwd_kept,
        "bwd_prealloc_peak_bytes": pre_peak,
        "autograd_saved_bytes": int(saved),
        "autograd_backward_peak_bytes": ag_peak,
        "checkpoint_retained_bytes": int(checkpoint_retained),
    }
    logical = _logical_bytes(
        rows, hidden, residual=residual, w_bytes=W_DTYPES[w_token].itemsize
    )
    entry["gbps"] = {
        phase: logical[phase] / (entry["gpu"][phase]["span_us"]["median"] * 1e-6) / 1e9
        for phase in PHASES
    }
    return entry


def _meta(device, session: str) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "schema": SCHEMA,
        "session": session,
        "time": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "revision": os.environ.get("BENCH_REVISION", ""),
    }


# --------------------------------------------------------------------------- report


def _cell(summary: Optional[dict], scale: float = 1.0, digits: int = 2) -> str:
    if not summary or summary.get("median") is None:
        return "-"
    fmt = f"{{:.{digits}f}}"
    return (
        fmt.format(summary["median"] * scale)
        + " ("
        + fmt.format(summary["p10"] * scale)
        + "-"
        + fmt.format(summary["p90"] * scale)
        + ")"
    )


def report(paths: list[str]) -> tuple[list[dict], str]:
    """Pool the per-replay samples of every (shape, mode) over the sessions of
    ``paths`` (JSONL files); ``(rows, markdown table)``."""

    groups: dict = collections.defaultdict(list)
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    key = (
                        record["hidden"],
                        record["rows"],
                        record["residual"],
                        record["w_dtype"],
                    )
                    groups[key].append(record)
    sessions = sorted({r["session"] for records in groups.values() for r in records})
    rows = []
    for key in sorted(groups):
        records = groups[key]
        row: dict = {
            "hidden": key[0],
            "rows": key[1],
            "residual": key[2],
            "w_dtype": key[3],
            "form": next((r["form"] for r in records if r["mode"] == "nondet"), "-"),
        }
        for mode, _flag in MODES:
            mine = [r for r in records if r["mode"] == mode and r["supported"]]
            if not mine:
                reasons = {r.get("reason", "") for r in records if r["mode"] == mode}
                row[mode] = {"supported": False, "reason": "; ".join(sorted(reasons))}
                continue
            pooled = {}
            for phase in PHASES:
                samples = [v for r in mine for v in r["gpu"][phase]["samples_span_us"]]
                pooled[phase] = _summary(samples)
                pooled[f"{phase}_sessions"] = {
                    r["session"]: r["gpu"][phase]["span_us"]["median"] for r in mine
                }
                pooled[f"{phase}_kernels"] = mine[0]["gpu"][phase]["kernels"]
            pooled["host_bwd"] = _summary([r["host_us"]["bwd"]["median"] for r in mine])
            pooled["memory"] = mine[0]["memory"]
            pooled["sessions"] = sorted({r["session"] for r in mine})
            row[mode] = pooled
        for phase in PHASES:
            det, nondet = row.get("det", {}), row.get("nondet", {})
            if det.get(phase) and nondet.get(phase):
                row[f"{phase}_det_over_nondet"] = (
                    det[phase]["median"] / nondet[phase]["median"]
                )
        rows.append(row)
    header = (
        "| H | T | res | w | bwd form | fwd us det | fwd us nondet | bwd us det | "
        "bwd us nondet | bwd det/nondet | 2xfwd+bwd us det | 2xfwd+bwd us nondet | "
        "step det/nondet | bwd nondet per session | bwd kernels nondet | "
        "host bwd us det / nondet | workspace KiB det / nondet | "
        "bwd temp KiB det / nondet | autograd saved KiB nondet | "
        "checkpoint retained KiB det / nondet |"
    )
    lines = [header, "|" + "|".join("---" for _ in range(header.count("|") - 1)) + "|"]
    for row in rows:
        det, nondet = row.get("det", {}), row.get("nondet", {})

        def ratio(phase):
            value = row.get(f"{phase}_det_over_nondet")
            return "-" if value is None else f"{value:.2f}x"

        def kib(entry, key):
            value = (entry.get("memory") or {}).get(key)
            return "-" if value is None else f"{value / 1024:.1f}"

        def host(entry):
            summary = entry.get("host_bwd")
            return "-" if not summary else f"{summary['median']:.1f}"

        per_session = " / ".join(
            f"{value:.2f}"
            for _s, value in sorted(nondet.get("bwd_sessions", {}).items())
        )
        kernels = ", ".join(
            f"{count}x{name[-20:]}" for name, count in nondet.get("bwd_kernels", [])
        )
        lines.append(
            f"| {row['hidden']} | {row['rows']} | {'y' if row['residual'] else 'n'} | "
            f"{row['w_dtype']} | {row['form']} | {_cell(det.get('fwd'))} | "
            f"{_cell(nondet.get('fwd'))} | {_cell(det.get('bwd'))} | "
            f"{_cell(nondet.get('bwd'))} | {ratio('bwd')} | {_cell(det.get('step'))} | "
            f"{_cell(nondet.get('step'))} | {ratio('step')} | {per_session or '-'} | "
            f"{kernels or '-'} | {host(det)} / {host(nondet)} | "
            f"{kib(det, 'workspace_bytes')} / {kib(nondet, 'workspace_bytes')} | "
            f"{kib(det, 'bwd_default_temporary_bytes')} / "
            f"{kib(nondet, 'bwd_default_temporary_bytes')} | "
            f"{kib(nondet, 'autograd_saved_bytes')} | "
            f"{kib(det, 'checkpoint_retained_bytes')} / "
            f"{kib(nondet, 'checkpoint_retained_bytes')} |"
        )
    caption = (
        f"GPU us: median (p10-p90) of the CUPTI span of CUDA-graph replays with a "
        f"cold L2, pooled over sessions {', '.join(sessions)}; det = deterministic "
        "package ('-' where it has no program for the shape); ratio = det / nondet "
        "medians (> 1: the non-deterministic call is faster)."
    )
    return rows, caption + "\n\n" + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hidden", type=int, nargs="*", default=[h for h, _ in WIDTHS])
    parser.add_argument("--rows", type=int, nargs="*", default=list(ROWS))
    parser.add_argument(
        "--residual",
        action="store_true",
        help="also run the fused residual-add rows (H=6144)",
    )
    parser.add_argument(
        "--w-dtype",
        nargs="*",
        choices=tuple(W_DTYPES),
        default=["bf16"],
        help="weight dtypes (f32: the FP32-weight routes)",
    )
    parser.add_argument(
        "--iters", type=int, default=200, help="CUPTI replays per phase"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--host-runs", type=int, default=50)
    parser.add_argument("--session", type=str, default="s1", help="session tag")
    parser.add_argument("--jsonl", type=str, default=None, help="append records here")
    parser.add_argument(
        "--report",
        type=str,
        nargs="+",
        default=None,
        help="merge these JSONL files (sessions) instead of measuring",
    )
    parser.add_argument("--markdown", type=str, default=None)
    args = parser.parse_args(argv)

    if args.report:
        _rows, table = report(args.report)
        print(table)
        if args.markdown:
            with open(args.markdown, "w") as handle:
                handle.write(table)
        return 0

    device = torch.device("cuda", torch.cuda.current_device())
    if not is_cake_rmsnorm_train_supported(device, deterministic=False):
        raise SystemExit(
            "cake_rmsnorm_train(deterministic=False) has no exported route on this device"
        )
    meta = _meta(device, args.session)
    plans = [
        (hidden, rows, False, w_token)
        for w_token in args.w_dtype
        for hidden in args.hidden
        for rows in args.rows
    ]
    if args.residual and 6144 in args.hidden:
        plans += [(6144, rows, True, w) for w in args.w_dtype for rows in args.rows]
    print(
        "H T res w form | mode: fwd / bwd / 2xfwd+bwd GPU us (median, CUPTI span) | "
        "host bwd us | workspace KiB | kernels of one bwd"
    )
    for hidden, rows, residual, w_token in plans:
        if not is_cake_rmsnorm_train_supported(device, hidden, deterministic=False):
            print(f"H={hidden}: no exported non-deterministic route, skipped")
            continue
        form = _form(device, hidden, rows, residual, w_token)
        for mode, deterministic in MODES:
            entry = bench_mode(
                hidden,
                rows,
                residual=residual,
                w_token=w_token,
                deterministic=deterministic,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
                host_runs=args.host_runs,
            )
            record = {
                **meta,
                "hidden": hidden,
                "rows": rows,
                "residual": residual,
                "w_dtype": w_token,
                "eps": EPS[hidden],
                "mode": mode,
                "form": form if mode == "nondet" else "det pair",
                **entry,
            }
            if args.jsonl:
                with open(args.jsonl, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
            tag = f"H={hidden} T={rows} res={int(residual)} w={w_token} {form}"
            if not entry["supported"]:
                print(f"{tag} | {mode}: unsupported ({entry['reason'][:80]})")
                continue
            gpu = entry["gpu"]
            print(
                f"{tag} | {mode}: "
                + " / ".join(f"{gpu[p]['span_us']['median']:.2f}" for p in PHASES)
                + f" | host {entry['host_us']['bwd']['median']:.1f}"
                + f" | ws {entry['memory']['workspace_bytes'] / 1024:.1f}"
                + " | "
                + ", ".join(f"{c}x{n[-24:]}" for n, c in gpu["bwd"]["kernels"]),
                flush=True,
            )
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
