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

Benchmark of the Cake packed KDA chunked training kernels with a selectable
numerical mode: the non-deterministic package
(``flashinfer.experimental.cake.cake_kda_train``, ``deterministic=False``)
against the deterministic package
(``flashinfer.experimental.cake_kda_det_train``) and flash-linear-attention's
``chunk_kda`` (when it is installed), on the same inputs, on the boundaries of
a training step:

* ``fwd``: the forward with the saved set written;
* ``bwd``: the backward from the saved set;
* ``ckpt``: ``torch.utils.checkpoint`` forward + replay + backward (one span:
  two forwards and the backward);
* ``replay`` (optional, Cake arms only): the backward without a saved set
  (``saved=None``), which recomputes it.

Every row is one packed batch of ``--docs`` documents of ``--seqlen`` tokens
in total (unequal lengths with unaligned tails), or one ``--lengths`` packing
(explicit comma-separated document lengths), and ``--heads`` heads; each
Cake arm allocates its workspace once per row and reuses it, so the timed
closures allocate nothing.  Timing uses CUPTI kernel activity with a cold L2
(``flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True)``):
the median over the timed calls of the summed GPU time of a boundary's
kernels, in milliseconds.  The arms run interleaved: ``--rounds`` rounds, each
timing every arm and boundary once, the arm order reversed on every other
round; a row reports the median of the round medians of each arm and boundary
with their spread (min, max), the ratios ``det / nd`` and ``fla / nd`` of those
medians (above 1: the non-deterministic arm is faster), the records each Cake
arm launches, and the non-deterministic results against the deterministic ones
(``out``, ``dq``, ``dk``, ``dv`` and ``dbeta`` are bitwise; ``dg`` differs by FP32
rounding on the rows whose backward runs the S1 pair).  The JSON records the
device, architecture, SM count, torch, CUDA and flash-linear-attention versions
(the distribution's and ``fla.__version__``).

Memory: per row, arm and boundary one untimed call after the warm-up, between
``torch.cuda.reset_peak_memory_stats()`` and the allocator's counters, gives
``peak_bytes`` (the peak allocation of the call beyond what was allocated
before it) and ``retained_bytes`` (what is still allocated after it while its
results are held: after ``fwd`` the output and the saved set -- for FLA its
autograd graph --, after the other boundaries the gradients).  A Cake arm's
caller-owned workspace (``workspace_bytes``) is allocated before and counts in
neither.

The flash-linear-attention arm fails soft: an exception from its first call,
its warm-up, its memory call or its timing records ``fla_error`` (the exception
type and message) in the row, drops the arm, and the Cake arms go on (an error
that corrupts the CUDA context still stops the run).

``--side-stream`` adds, per row, Cake arm and boundary, the wall latency of the
call on its stream (CUDA events; the median of seven calls after three
warm-ups) alone and with a one-thread kernel (``torch.cuda._sleep``) resident
on another stream, alternating, and their ratio (the concurrent-launch
measurement of FlashInfer issue #7038, after the merged causal conv1d and
RMSNorm SiLU-gate benchmarks).  The side kernel is launched first, the call
after a short sleep on its own stream, and it spins for at least twice the
call's warm-up latency (``side_covered``: it was still running when every
call beside it ended).  Information, not a gate.

Example::

    python benchmarks/bench_cake_kda_train.py --seqlen 16384 65536 --docs 3 --heads 4 64
    python benchmarks/bench_cake_kda_train.py --seqlen 261393 --docs 4 --heads 4 --rounds 5 --json out.json
    python benchmarks/bench_cake_kda_train.py --seqlen 65536 --docs 3 --heads 16 --side-stream
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.cake.cake_kda_train import (  # noqa: E402
    cake_backend as nd_backend,
)
from flashinfer.experimental.cake.cake_kda_train import cake_jit  # noqa: E402
from flashinfer.experimental.cake_kda_det_train import (  # noqa: E402
    cake_backend as det_backend,
)
from flashinfer.testing import bench_gpu_time  # noqa: E402
from tests.test_helpers.cake_kda_det_train_reference import make_inputs  # noqa: E402

BOUNDARIES = ("fwd", "bwd", "ckpt", "replay")
DEFAULT_BOUNDARIES = ("fwd", "bwd", "ckpt")
ARMS = ("det", "nd", "fla")
GRAD_ORDER = ("dq", "dk", "dv", "dbeta", "dg")  # the order the backward returns
# ``--side-stream``: a one-thread kernel resident on another stream while a Cake call runs.  It spins for at least
# SIDE_MIN_CYCLES (about 5 ms at 2 GHz) and SIDE_CYCLES_PER_MS per ms of the call's slowest warm-up (twice the call at
# 2 GHz); the call starts after PRE_SLEEP_CYCLES on its own stream, so the side kernel is resident first.
SIDE_MIN_CYCLES = 10_000_000
SIDE_CYCLES_PER_MS = 4_000_000
PRE_SLEEP_CYCLES = 1_000_000
SIDE_WARMUPS = 3
SIDE_SAMPLES = 7


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


def _timer() -> str:
    """What ``bench_gpu_time(enable_cupti=True)`` measures here: CUPTI kernel time
    when ``cupti-python`` >= 13 imports, else its fallback, CUDA-event time."""
    try:
        from importlib.metadata import version

        from cupti import cupti  # noqa: F401

        cupti_version = version("cupti-python")
    except Exception:
        return "CUDA events (cupti-python unavailable)"
    if int(cupti_version.split(".")[0]) < 13:
        return f"CUDA events (cupti-python {cupti_version} < 13)"
    return f"CUPTI kernel time (cupti-python {cupti_version})"


def _fla_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("flash-linear-attention")
    except Exception:
        try:
            import fla

            return getattr(fla, "__version__", "unknown")
        except Exception:
            return None


def _fla_module_version() -> str | None:
    """``fla.__version__`` when the module imports (``None`` otherwise)."""
    try:
        import fla
    except Exception:
        return None
    return str(getattr(fla, "__version__", "unknown"))


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def cake_boundaries(
    inputs: dict, ws, *, deterministic: bool
) -> dict[str, Callable[[], object]]:
    """The boundaries of one Cake arm on its workspace: the deterministic package's entries, or this package's
    with ``deterministic=False``.  Each closure returns its results (the memory call holds them)."""
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    if deterministic:
        fwd_fn = det_backend.chunk_kda_det_fwd
        bwd_fn = det_backend.chunk_kda_det_bwd
        autograd_fn = det_backend.chunk_kda_det
        mode = {}
    else:
        fwd_fn = nd_backend.chunk_kda_fwd
        bwd_fn = nd_backend.chunk_kda_bwd
        autograd_fn = nd_backend.chunk_kda
        mode = {"deterministic": False}

    def fwd() -> object:
        return fwd_fn(
            q, k, v, g, beta, cu, save_for_backward=True, workspace=ws, **mode
        )

    _, saved = fwd_fn(
        q, k, v, g, beta, cu, save_for_backward=True, workspace=ws, **mode
    )

    def bwd() -> object:
        return bwd_fn(q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, **mode)

    def replay() -> object:
        return bwd_fn(q, k, v, g, beta, cu, dout, saved=None, workspace=ws, **mode)

    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def ckpt() -> object:
        o = checkpoint(
            lambda *a: autograd_fn(*a, cu, workspace=ws, **mode),
            *leaves,
            use_reentrant=False,
        )
        return torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt, "replay": replay}


def fla_boundaries(inputs: dict) -> dict[str, Callable[[], object]] | None:
    """flash-linear-attention's ``chunk_kda`` on the same preactivated operands (no in-kernel L2 norm, gate or
    sigmoid), or ``None`` when the package is not installed.  The first call runs here: an exception from it
    propagates to the caller (:func:`run_row` records it as ``fla_error``)."""
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
            scale=nd_backend.SCALE,
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

    def fwd() -> object:
        return call(*leaves)

    retained = call(*leaves)

    def bwd() -> object:
        return torch.autograd.grad(retained, leaves, dout, retain_graph=True)

    def ckpt() -> object:
        o = checkpoint(call, *leaves, use_reentrant=False)
        return torch.autograd.grad(o, leaves, dout)

    return {"fwd": fwd, "bwd": bwd, "ckpt": ckpt}


def _results(inputs: dict, ws, *, deterministic: bool, replay: bool) -> dict:
    """``out`` and the gradients of one Cake arm (from the saved set, or the replay's)."""
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    if deterministic:
        out, saved = det_backend.chunk_kda_det_fwd(q, k, v, g, beta, cu, workspace=ws)
        grads = det_backend.chunk_kda_det_bwd(
            q, k, v, g, beta, cu, dout, saved=None if replay else saved, workspace=ws
        )
    else:
        out, saved = nd_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, deterministic=False
        )
        grads = nd_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=None if replay else saved,
            workspace=ws,
            deterministic=False,
        )
    return {
        "out": out.clone(),
        **{n: t.clone() for n, t in zip(GRAD_ORDER, grads, strict=True)},
    }


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _nd_vs_det(inputs: dict, det_ws, nd_ws) -> dict:
    """The non-deterministic results against the deterministic ones on the same inputs, from the saved set and from
    the replay: the outputs that are bitwise, and ``dg``'s differing elements and largest difference."""
    report = {}
    for label, replay in (("saved", False), ("replay", True)):
        det = _results(inputs, det_ws, deterministic=True, replay=replay)
        nd = _results(inputs, nd_ws, deterministic=False, replay=replay)
        diff = (nd["dg"].double() - det["dg"].double()).abs()
        report[label] = {
            "bitwise": [name for name in det if _bits_equal(nd[name], det[name])],
            "dg_differing_elements": int((diff > 0).sum().item()),
            "dg_max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
            "dg_max_abs": float(det["dg"].abs().max().item()) if diff.numel() else 0.0,
        }
    return report


def _nd_vs_det_or_error(inputs: dict, det_ws, nd_ws) -> dict:
    """:func:`_nd_vs_det`, or its error when the two arms' results do not fit in device memory beside the inputs."""
    try:
        return _nd_vs_det(inputs, det_ws, nd_ws)
    except torch.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": _error(exc)}


def _s1_paths(ws) -> list[str]:
    """The backward paths of a workspace whose fused gradient runs an S1 record."""
    return [
        path
        for path, by_stage in ws.records().items()
        if by_stage.get("dqkg") in cake_jit.S1_RECORDS
    ]


def _memory(fn: Callable[[], object], device: torch.device) -> dict[str, int]:
    """``peak_bytes`` / ``retained_bytes`` of one untimed call of ``fn``: the peak allocation beyond what was
    allocated before the call, and what stays allocated after it while its results are held."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    results = fn()
    torch.cuda.synchronize(device)
    report = {
        "peak_bytes": torch.cuda.max_memory_allocated(device) - base,
        "retained_bytes": torch.cuda.memory_allocated(device) - base,
    }
    del results
    return report


def _latency_beside(fn: Callable[[], object], device: torch.device) -> dict:
    """The wall latency of ``fn`` on the current stream (CUDA events; median of :data:`SIDE_SAMPLES` calls after
    :data:`SIDE_WARMUPS`) alone and with a one-thread kernel resident on another stream, alternating.  The side
    kernel is launched first and spins for at least twice the slowest warm-up (at 2 GHz); ``side_covered`` tells
    whether it was still running when every call beside it ended."""
    side = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    side_end = torch.cuda.Event(enable_timing=True)

    def timed(cycles: int) -> tuple[float, bool]:
        torch.cuda.synchronize(device)
        if cycles:
            with torch.cuda.stream(side):
                torch.cuda._sleep(cycles)
            side_end.record(side)
        torch.cuda._sleep(
            PRE_SLEEP_CYCLES
        )  # the side kernel becomes resident before fn's launch
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        ms = start.elapsed_time(end)
        return ms, not cycles or start.elapsed_time(side_end) >= ms

    warm = [timed(0)[0] for _ in range(SIDE_WARMUPS)]
    cycles = max(SIDE_MIN_CYCLES, math.ceil(max(warm) * SIDE_CYCLES_PER_MS))
    alone, beside, covered = [], [], True
    for _ in range(SIDE_SAMPLES):
        alone.append(timed(0)[0])
        ms, ok = timed(cycles)
        beside.append(ms)
        covered = covered and ok
    alone_ms, beside_ms = _median_ms(alone), _median_ms(beside)
    return {
        "alone_ms": alone_ms,
        "beside_ms": beside_ms,
        "ratio_beside_over_alone": beside_ms / alone_ms,
        "side_cycles": cycles,
        "side_covered": covered,
    }


def run_row(
    seqlen: int,
    docs: int,
    heads: int,
    *,
    device: torch.device,
    baseline: bool,
    boundaries: tuple[str, ...],
    rounds: int,
    side_stream: bool = False,
    lengths: tuple[int, ...] | None = None,
    fla_alone: bool = False,
    arms_alone: bool = False,
) -> dict:
    lengths = split_docs(seqlen, docs) if lengths is None else lengths
    inputs = make_inputs(lengths, heads, device)

    def det_workspace():
        ws = det_backend.KdaDetWorkspace.for_problem(seqlen, docs, heads, device)
        ws.load()
        return ws

    def nd_workspace():
        ws = nd_backend.KdaWorkspace.for_problem(
            seqlen, docs, heads, device, deterministic=False
        )
        ws.load()
        return ws

    det_ws, nd_ws = det_workspace(), nd_workspace()
    row = {
        "seqlen": seqlen,
        "docs": docs,
        "lengths": list(lengths),
        "heads": heads,
        "arch": nd_ws.arch,
        "recurrence": nd_ws.plan.recur,
        "segments_per_doc": nd_ws.plan.P,
        "records": {"det": det_ws.records(), "nd": nd_ws.records()},
        "s1_paths": _s1_paths(nd_ws),
        "workspace_bytes": {"det": det_ws.bytes, "nd": nd_ws.bytes},
        "saved_bytes": nd_backend.saved_bytes(
            seqlen, heads, docs, sm_count=nd_ws.plan.sm_count
        ),
        "nd_vs_det": {"skipped": "--arms-alone"}
        if arms_alone
        else _nd_vs_det_or_error(inputs, det_ws, nd_ws),
        "rounds": rounds,
        "arm_order": [],
        "ms_rounds": {},
        "ms": {},
        "ms_min": {},
        "ms_max": {},
        "ratio_det_over_nd": {},
        "ratio_fla_over_nd": {},
        "memory": {},
    }
    arms: dict[str, dict[str, Callable[[], object]]] = (
        {}
        if arms_alone
        else {
            "det": cake_boundaries(inputs, det_ws, deterministic=True),
            "nd": cake_boundaries(inputs, nd_ws, deterministic=False),
        }
    )
    samples: dict[str, dict[str, list[float]]] = {}

    def fla_failed(exc: Exception) -> None:
        # the FLA arm fails soft: its error goes to the row, the arm (and its samples) away, the Cake arms go on
        row["fla_error"] = _error(exc)
        arms.pop("fla", None)
        samples.pop("fla", None)
        row["memory"].pop("fla", None)
        torch.cuda.synchronize(device)

    if baseline and not fla_alone and not arms_alone:
        try:
            fla = fla_boundaries(inputs)
        except Exception as exc:
            fla = None
            fla_failed(exc)
        if fla is not None:
            arms["fla"] = fla
    # warm every closure once (libraries loaded, the allocator primed) before the first timed round
    for arm in list(arms):
        try:
            for name in boundaries:
                if name in arms[arm]:
                    arms[arm][name]()
        except Exception as exc:
            if arm != "fla":
                raise
            fla_failed(exc)
    torch.cuda.synchronize(device)
    # peak and retained memory of one untimed call per arm and boundary
    for arm in list(arms):
        try:
            row["memory"][arm] = {
                name: _memory(arms[arm][name], device)
                for name in boundaries
                if name in arms[arm]
            }
        except Exception as exc:
            if arm != "fla":
                raise
            fla_failed(exc)
    samples.update(
        {arm: {name: [] for name in boundaries if name in arms[arm]} for arm in arms}
    )
    for r in range(rounds):
        order = [arm for arm in ARMS if arm in arms]
        round_order = order if r % 2 == 0 else order[::-1]
        row["arm_order"].append(round_order)
        for arm in round_order:
            if arm not in arms:
                continue
            try:
                for name in boundaries:
                    if name in arms[arm]:
                        samples[arm][name].append(_time(arms[arm][name]))
            except Exception as exc:
                if arm != "fla":
                    raise
                fla_failed(exc)
    for arm, by_name in samples.items():
        row["ms_rounds"][arm] = by_name
        row["ms"][arm] = {n: _median_ms(v) for n, v in by_name.items()}
        row["ms_min"][arm] = {n: min(v) for n, v in by_name.items()}
        row["ms_max"][arm] = {n: max(v) for n, v in by_name.items()}
    if arms_alone:
        # every arm on its own, one after another, its closures released before the next (rows whose Cake arms do
        # not fit in device memory together): the same inputs, ``rounds`` timed calls per boundary, not interleaved
        row["arms_alone"] = True
        # each Cake arm on a workspace of its own (the two workspaces of a large row need not fit together)
        det_ws = nd_ws = None
        gc.collect()
        torch.cuda.empty_cache()
        builders = {
            "det": lambda: cake_boundaries(inputs, det_workspace(), deterministic=True),
            "nd": lambda: cake_boundaries(inputs, nd_workspace(), deterministic=False),
        }
        if baseline:
            builders["fla"] = lambda: fla_boundaries(inputs)
        for arm, build in builders.items():
            fns = None
            try:
                fns = build()
                for name in boundaries:
                    if name in fns:
                        fns[name]()
                torch.cuda.synchronize(device)
                row["memory"][arm] = {
                    name: _memory(fns[name], device)
                    for name in boundaries
                    if name in fns
                }
                by_name = {
                    name: [_time(fns[name]) for _ in range(rounds)]
                    for name in boundaries
                    if name in fns
                }
                row["ms_rounds"][arm] = by_name
                row["ms"][arm] = {n: _median_ms(v) for n, v in by_name.items()}
                row["ms_min"][arm] = {n: min(v) for n, v in by_name.items()}
                row["ms_max"][arm] = {n: max(v) for n, v in by_name.items()}
            except Exception as exc:
                if arm != "fla":
                    raise
                row["fla_error"] = _error(exc)
                row["memory"].pop("fla", None)
            fns = None
            gc.collect()
            torch.cuda.empty_cache()
    if baseline and fla_alone and not arms_alone:
        # the FLA arm on its own, after the Cake arms and workspaces are released (rows whose three arms do not fit
        # in device memory at once): the same inputs, ``rounds`` timed calls per boundary, not interleaved
        row["fla_alone"] = True
        arms.clear()
        det_ws = nd_ws = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            fla = fla_boundaries(inputs)
            for name in boundaries:
                if name in fla:
                    fla[name]()
            torch.cuda.synchronize(device)
            row["memory"]["fla"] = {
                name: _memory(fla[name], device) for name in boundaries if name in fla
            }
            by_name = {
                name: [_time(fla[name]) for _ in range(rounds)]
                for name in boundaries
                if name in fla
            }
            row["ms_rounds"]["fla"] = by_name
            row["ms"]["fla"] = {n: _median_ms(v) for n, v in by_name.items()}
            row["ms_min"]["fla"] = {n: min(v) for n, v in by_name.items()}
            row["ms_max"]["fla"] = {n: max(v) for n, v in by_name.items()}
        except Exception as exc:
            row["fla_error"] = _error(exc)
            row["memory"].pop("fla", None)
            torch.cuda.synchronize(device)
        fla = None
        gc.collect()
        torch.cuda.empty_cache()
    for name in boundaries:
        nd_ms = row["ms"]["nd"].get(name)
        if nd_ms:
            row["ratio_det_over_nd"][name] = row["ms"]["det"][name] / nd_ms
            if "fla" in row["ms"] and name in row["ms"]["fla"]:
                row["ratio_fla_over_nd"][name] = row["ms"]["fla"][name] / nd_ms
    if side_stream:
        row["side_stream"] = {
            arm: {
                name: _latency_beside(arms[arm][name], device)
                for name in boundaries
                if name in arms[arm]
            }
            for arm in ("det", "nd")
        }
        row["side_stream_ratio_det_over_nd"] = {
            name: row["side_stream"]["det"][name]["beside_ms"]
            / row["side_stream"]["nd"][name]["beside_ms"]
            for name in row["side_stream"]["nd"]
        }
    return row


def _print_row(row: dict, boundaries: tuple[str, ...]) -> None:
    cells = [
        f"S={row['seqlen']:>7} N={row['docs']} H={row['heads']:>2} {row['arch']} "
        f"{row['recurrence']}{'/P' + str(row['segments_per_doc']) if row['recurrence'] == 'seg' else ''} "
        f"S1[{','.join(row['s1_paths']) or '-'}]"
    ]
    for name in boundaries:
        if name not in row["ms"]["nd"]:
            continue
        nd_ms, det_ms = row["ms"]["nd"][name], row["ms"]["det"][name]
        cell = (
            f"{name} det {det_ms:8.3f} nd {nd_ms:8.3f} ms det/nd x{det_ms / nd_ms:6.4f}"
        )
        if name in row["ratio_fla_over_nd"]:
            cell += f" fla {row['ms']['fla'][name]:8.3f} fla/nd x{row['ratio_fla_over_nd'][name]:5.2f}"
        cells.append(cell)
    if "fla_error" in row:
        cells.append(f"fla error: {row['fla_error'][:120]}")
    print(" | ".join(cells))
    for name, ratio in row.get("side_stream_ratio_det_over_nd", {}).items():
        det, nd = row["side_stream"]["det"][name], row["side_stream"]["nd"][name]
        print(
            f"    {name} beside a resident kernel: det {det['alone_ms']:8.3f} -> {det['beside_ms']:8.3f} ms, "
            f"nd {nd['alone_ms']:8.3f} -> {nd['beside_ms']:8.3f} ms, det/nd x{ratio:6.4f}"
            + (
                ""
                if det["side_covered"] and nd["side_covered"]
                else " (side kernel ended early)"
            )
        )


def _write_json(
    path, rows, boundaries, device, arch, timer, fla_version, fla_module_version, args
) -> None:
    """The run's rows so far with its environment and timing method (rewritten after every row)."""
    with open(path, "w") as f:
        json.dump(
            {
                "device": torch.cuda.get_device_name(device),
                "arch": arch,
                "sm_count": cake_jit.sm_count(device.index),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "fla": fla_version,
                "fla_module_version": fla_module_version,
                "timing": f"{timer}, cold L2, median of the timed calls; per arm and boundary the "
                "median of the round medians",
                "memory": "per arm and boundary, one untimed call: peak_bytes = max_memory_allocated - "
                "the allocation before it, retained_bytes = memory_allocated after it while its results are "
                "held - the allocation before it",
                "side_stream": "CUDA-event wall latency, median of "
                f"{SIDE_SAMPLES} calls alone and beside a one-thread resident kernel"
                if args.side_stream
                else None,
                "boundaries": list(boundaries),
                "rows": rows,
            },
            f,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--docs", type=int, nargs="+", default=[3])
    parser.add_argument("--heads", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--lengths",
        type=str,
        nargs="+",
        default=None,
        help="explicit packings, each comma-separated document lengths (replaces --seqlen / --docs)",
    )
    parser.add_argument(
        "--boundaries",
        type=str,
        default=",".join(DEFAULT_BOUNDARIES),
        help=f"comma-separated subset of {','.join(BOUNDARIES)}",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="interleaved rounds per row (the arm order reversed every other round)",
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip flash-linear-attention"
    )
    parser.add_argument(
        "--side-stream",
        action="store_true",
        help="also time the Cake arms' wall latency alone and beside a kernel resident on another stream",
    )
    parser.add_argument("--json", type=str, default=None, help="write the rows here")
    parser.add_argument(
        "--arms-alone",
        action="store_true",
        help="time det, nd and flash-linear-attention one after another, each released before the next",
    )
    parser.add_argument(
        "--fla-alone",
        action="store_true",
        help="time the flash-linear-attention arm after releasing the Cake arms (rows whose three arms do not fit)",
    )
    parser.add_argument(
        "--require-cupti",
        action="store_true",
        help="refuse to run when the timer would fall back to CUDA events",
    )
    args = parser.parse_args()
    timer = _timer()
    if (args.fla_alone or args.arms_alone) and args.side_stream:
        parser.error("--fla-alone releases the Cake arms the --side-stream rows time")
    if args.require_cupti and not timer.startswith("CUPTI"):
        parser.error(f"--require-cupti: the timer is {timer}")
    print(f"# timer: {timer}")
    boundaries = tuple(b.strip() for b in args.boundaries.split(",") if b.strip())
    unknown = [b for b in boundaries if b not in BOUNDARIES]
    if unknown or not boundaries:
        parser.error(f"unknown boundaries {unknown}; choose from {BOUNDARIES}")
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    device = torch.device("cuda", torch.cuda.current_device())
    arch = cake_jit.arch_for_capability(
        tuple(torch.cuda.get_device_capability(device.index))
    )
    if arch is None:
        raise SystemExit(
            f"{torch.cuda.get_device_name(device)} is not an exported target {cake_jit.ARCHES}"
        )
    fla_version = None if args.no_baseline else _fla_version()
    fla_module_version = None if args.no_baseline else _fla_module_version()
    if args.lengths:
        packs = [tuple(int(x) for x in p.split(",")) for p in args.lengths]
    else:
        packs = [split_docs(s, d) for s in args.seqlen for d in args.docs]
    rows = []
    for lengths in packs:
        for heads in args.heads:
            row = run_row(
                sum(lengths),
                len(lengths),
                heads,
                device=device,
                baseline=not args.no_baseline,
                boundaries=boundaries,
                rounds=args.rounds,
                side_stream=args.side_stream,
                lengths=lengths,
                fla_alone=args.fla_alone,
                arms_alone=args.arms_alone,
            )
            _print_row(row, boundaries)
            rows.append(row)
            if args.json:
                _write_json(
                    args.json,
                    rows,
                    boundaries,
                    device,
                    arch,
                    timer,
                    fla_version,
                    fla_module_version,
                    args,
                )
    for name in boundaries:
        ratios = [
            r["ratio_det_over_nd"][name] for r in rows if name in r["ratio_det_over_nd"]
        ]
        if ratios:
            geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            slower = sum(x < 1.0 for x in ratios)
            print(
                f"# {name}: det / nd geomean x{geo:.4f} over {len(ratios)} rows; nd slower on {slower} rows"
            )
    if args.json:
        _write_json(
            args.json,
            rows,
            boundaries,
            device,
            arch,
            timer,
            fla_version,
            fla_module_version,
            args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
