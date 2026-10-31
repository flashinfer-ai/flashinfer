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

Benchmark of the context-parallel (KCP) Cake packed width-4 depthwise causal convolution + SiLU training step in its
two numerical modes, ``deterministic=True`` (the deterministic KCP backward: a cooperative launch whose grid reduces
the ``dweight`` partial in a fixed order) against ``deterministic=False`` (one ordinary launch with unordered
``dweight`` accumulators), both through ``flashinfer.cake_causal_conv1d_train_kcp``: ``cake_causal_conv1d_kcp``, or
``cake_causal_conv1d_multi_kcp`` over ``--tensors`` convolutions of one layer (one halo exchange per direction).
One process per rank: ``torchrun --nproc_per_node W`` (W = 2 / 4 / 8), or a plain ``python`` launch for a one-rank
group (W = 1).

Rows: the single-rank benchmark's performance matrix (``C = 512`` over eight packed token counts and ``S = 250340``
over the channel counts), each the same four-document pack (0.37 / 0.29 / 0.21 / rest of ``S``, nudged off a multiple
of 64), token-sharded over the W ranks by ``flashinfer.kda_train.kcp_partition`` (``--layout``; ``contiguous``: equal
token shares in order, documents cut at the shard boundaries, whose first tokens read their history rows through
the halo exchange).  ``S`` is the whole pack; a rank holds about ``S / W`` tokens.  Every rank draws the whole pack
from the same seed and keeps its fragments.

Phases per row and mode, each through the autograd entry with gradient-requiring leaves:

* ``fwd``: the forward (the forward halo exchange, the saved set written);
* ``bwd``: the backward of a recorded forward (``retain_graph=True``, the gradients reset before every call): the
  head program, the backward halo exchange and the main backward;
* ``ckpt``: ``torch.utils.checkpoint(..., use_reentrant=False)`` around the entry, then its backward (two forwards,
  two forward exchanges, one backward).

Measures (on every rank; a row reports the slowest rank and the JSON keeps every rank's value):

* ``gpu_us``: the CUPTI span of the phase's kernels on the rank (``flashinfer.testing.bench_gpu_time``, cold L2, the
  same iteration counts on every rank).  The exchange's NCCL kernels are among them, so a rank's span includes its
  wait for its peers; a host-bound phase (the autograd host time above its kernels' time) also puts host gaps in it;
* ``ahead_us``: the GPU time with the host enqueue hidden: after a barrier and an L2 flush, a spin kernel of the same
  length on every rank runs while the host enqueues the phase behind it; CUDA events around the phase
  (``ahead_late`` counts the samples whose enqueue outlasted the spin; 0 expected);
* ``wall_us``: the synchronized wall time of one call (barrier, ``perf_counter``, the call, device synchronize; the
  slowest rank per sample, the halo exchange included) and ``host_us``: the time to enqueue it;
* ``first`` / ``steady``: the first call of the phase after the packages' workspace caches are cleared (workspace
  allocation and zeroing included) and the next call, one sample each: the slowest rank's wall time, the allocator's
  peak above the bytes held before the call (``peak_temp_bytes``) and the net bytes still held after it
  (``retained_bytes``; ``fwd``: its outputs and saved set; ``bwd``: its gradients less the saved set it frees;
  ``ckpt``: with the leaves' gradients dropped again);
* ``halo``: per rank, the bytes one call sends and receives in each halo exchange (``ckpt`` runs the forward
  exchange twice).

The timings interleave the modes: ``--rounds`` rounds, each timing every phase of both modes (det first in even
rounds, nondet first in odd ones).  Each value is the median over the rounds of the round's value, with its
``[min, max]`` over the rounds; ``speedup`` is ``det / nondet`` of the medians (above 1: the non-deterministic mode
is faster), ``speedup_range`` the ``[min, max]`` of the per-round ratios.  A check per row on every rank (exit
status 1 when it fails): the non-deterministic ``y`` and ``dx`` are bitwise the deterministic ones and each
``dweight`` partial is within ``5e-2 + 1e-4 |det|`` of the deterministic partial.  Rank 0 prints the rows and writes
``--json``.

Example::

    python benchmarks/bench_cake_causal_conv1d_train_kcp.py --tokens 250340 --channels 512 1280
    torchrun --nproc_per_node 4 benchmarks/bench_cake_causal_conv1d_train_kcp.py --json kcp_w4.json
    torchrun --nproc_per_node 8 benchmarks/bench_cake_causal_conv1d_train_kcp.py --tensors 3 --layout zigzag
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flashinfer.cake_causal_conv1d_det_train_backend as det_backend  # noqa: E402
import flashinfer.cake_causal_conv1d_train_kcp as kcp_api  # noqa: E402
import flashinfer.experimental.cake.cake_causal_conv1d_train as conv  # noqa: E402
import flashinfer.experimental.cake.cake_causal_conv1d_train_backend as nondet_backend  # noqa: E402
from bench_cake_causal_conv1d_train import (  # noqa: E402
    P1_CHANNELS,
    P2_TOKENS,
    doc_lengths,
    perf_rows,
)
from flashinfer.api_logging import ExperimentalWarning  # noqa: E402
from flashinfer.kda_train import build_kcp_context, kcp_partition  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402

WIDTH = 4
SEED = 0
LAYOUTS = ("contiguous", "zigzag", "zigzag_per_document")
MODES = {"det": True, "nondet": False}
PHASES = ("fwd", "bwd", "ckpt")
KINDS = ("gpu", "ahead", "wall", "host")
ROUNDS = 7
GPU_WARMUPS = 3
GPU_SAMPLES = 10
WALL_WARMUPS = 3
WALL_SAMPLES = 10
AHEAD_WARMUPS = 2
AHEAD_SAMPLES = 10
# The spin kernel ahead of a sample lasts at least AHEAD_MIN_MS and AHEAD_ENQUEUE_FACTOR times the longest enqueue
# any rank saw while warming up.
AHEAD_MIN_MS = 2.0
AHEAD_ENQUEUE_FACTOR = 3.0
# The deterministic KCP suite's dweight budget.
DW_ATOL, DW_RTOL = 5e-2, 1e-4


def _median(samples) -> float:
    return float(statistics.median(list(samples)))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _init() -> tuple[int, int, torch.device]:
    """The process group: ``torchrun``'s, or a one-rank group for a plain ``python`` launch."""

    if "RANK" not in os.environ:
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def _gather(value) -> list:
    """Every rank's ``value``, in rank order."""

    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, value)
    return out


def _max_over_ranks(values: list[float], device: torch.device) -> list[float]:
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def _gpu_ms(fn: Callable[[], object]) -> list[float]:
    """Every rank's median CUPTI span (ms) of ``fn``'s kernels, cold L2; every rank runs the same iteration counts."""

    times = bench_gpu_time(
        fn,
        enable_cupti=True,
        cold_l2_cache=True,
        dry_run_iters=GPU_WARMUPS,
        repeat_iters=GPU_SAMPLES,
        aggregate_op=list,
    )
    if dist.get_world_size() == 1:
        return [_median(times)]
    # One list of every rank's value per iteration.
    return [_median(per_rank) for per_rank in zip(*times, strict=True)]


def _wall_ms(fn: Callable[[], object], device: torch.device) -> tuple[float, float]:
    """``(wall_ms, host_ms)``: the medians over the samples of the slowest rank's synchronized wall time of one call
    (a barrier first) and of its enqueue time."""

    walls, hosts = [], []
    for i in range(WALL_WARMUPS + WALL_SAMPLES):
        torch.cuda.synchronize(device)
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        wall, host = _max_over_ranks([(t2 - t0) * 1e3, (t1 - t0) * 1e3], device)
        if i >= WALL_WARMUPS:
            walls.append(wall)
            hosts.append(host)
    return _median(walls), _median(hosts)


class _GpuAhead:
    """GPU time (ms) of a call with its host enqueue hidden: after a barrier, an L2 flush and then a spin kernel (the
    same length on every rank) long enough for the host to enqueue the whole call behind it; CUDA events around the
    call."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        self.flush = torch.empty(2 * l2_bytes, dtype=torch.int8, device=device)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        probe = 1_000_000
        spins = []
        for _ in range(3):
            torch.cuda.synchronize(device)
            self.start.record()
            torch.cuda._sleep(probe)
            self.end.record()
            torch.cuda.synchronize(device)
            spins.append(self.start.elapsed_time(self.end))
        self.cycles_per_ms = probe / _median(spins)

    def _sample(self, fn: Callable[[], object], spin_ms: float) -> tuple[float, float]:
        torch.cuda.synchronize(self.device)
        dist.barrier()
        self.flush.zero_()
        torch.cuda._sleep(int(spin_ms * self.cycles_per_ms))
        self.start.record()
        t0 = time.perf_counter()
        fn()
        enqueue_ms = (time.perf_counter() - t0) * 1e3
        self.end.record()
        torch.cuda.synchronize(self.device)
        return self.start.elapsed_time(self.end), enqueue_ms

    def __call__(self, fn: Callable[[], object]) -> tuple[float, int]:
        """``(median_ms, late)`` of this rank: ``late`` counts the samples whose enqueue outlasted the spin."""

        enqueues = [self._sample(fn, AHEAD_MIN_MS)[1] for _ in range(AHEAD_WARMUPS)]
        (spin_ms,) = _max_over_ranks(
            [max(AHEAD_MIN_MS, AHEAD_ENQUEUE_FACTOR * max(enqueues))], self.device
        )
        times, late = [], 0
        for _ in range(AHEAD_SAMPLES):
            elapsed, enqueue_ms = self._sample(fn, spin_ms)
            times.append(elapsed)
            late += enqueue_ms >= spin_ms
        return _median(times), late


def _clear_package_caches() -> None:
    for module in (det_backend, nondet_backend):
        for name in ("_FORWARD_WORKSPACES", "_BACKWARD_WORKSPACES"):
            cache = getattr(module, name, None)
            if isinstance(cache, dict):
                cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _probe(fn: Callable[[], object], device: torch.device) -> dict:
    """One call of ``fn``: the slowest rank's wall time and the allocator's peak above / bytes held after, per rank."""

    torch.cuda.synchronize(device)
    # Collect before the barrier: the ranks leave it together and start the call at once.
    gc.collect()
    dist.barrier()
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - t0) * 1e3
    peak = torch.cuda.max_memory_allocated(device) - before
    retained = torch.cuda.memory_allocated(device) - before
    del result
    ranks = _gather((wall_ms * 1e3, int(peak), int(retained)))
    return {
        "wall_us": max(r[0] for r in ranks),
        "peak_temp_bytes": max(r[1] for r in ranks),
        "retained_bytes": max(r[2] for r in ranks),
        "per_rank": [
            {"wall_us": w, "peak_temp_bytes": p, "retained_bytes": r}
            for w, p, r in ranks
        ],
    }


def _bits(t: torch.Tensor) -> torch.Tensor:
    return (
        t.detach()
        .contiguous()
        .view(torch.int16 if t.element_size() == 2 else torch.int32)
    )


class Problem:
    """One row: the whole pack drawn on every rank from the same seed, this rank's fragments kept."""

    def __init__(
        self,
        tokens: int,
        channels: int,
        tensors: int,
        layout: str,
        rank: int,
        world: int,
        device: torch.device,
    ) -> None:
        self.lens = doc_lengths(tokens)
        cu = [0]
        for n in self.lens:
            cu.append(cu[-1] + n)
        self.part = kcp_partition(cu, world, layout)
        self.ctx = build_kcp_context(
            self.part, dist.group.WORLD, rank, device, conv_kernel_size=WIDTH
        )
        fragments = [f for f in self.part.fragments[rank] if f.length > 0]

        def shard(t: torch.Tensor) -> torch.Tensor:
            pieces = [
                t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length] for f in fragments
            ]
            return torch.cat(pieces, 0).contiguous() if pieces else t[:0].clone()

        generator = torch.Generator(device=device).manual_seed(SEED)
        self.xs, self.ws, self.dys = [], [], []
        for _ in range(tensors):
            x = torch.randn(
                tokens,
                channels,
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            )
            # N(0, 0.25) taps, as the single-rank benchmark's.
            w = torch.randn(channels, WIDTH, device=device, generator=generator) * 0.5
            dy = torch.randn(
                tokens,
                channels,
                device=device,
                generator=generator,
                dtype=torch.bfloat16,
            )
            self.xs.append(shard(x))
            self.ws.append(w.contiguous())
            self.dys.append(shard(dy))
            del x, dy
        self.tokens_local = int(self.xs[0].shape[0])
        # The halo exchange per call and direction (rows: BF16 history rows forward, FP32 derivative rows backward).
        halo = self.ctx.halo
        self.halo = {
            "fwd_send_bytes": int(sum(halo.fwd_send_rows)) * channels * 2 * tensors,
            "fwd_recv_bytes": int(sum(halo.fwd_recv_rows)) * channels * 2 * tensors,
            "bwd_send_bytes": int(sum(halo.bwd_send_rows)) * channels * 4 * tensors,
            "bwd_recv_bytes": int(sum(halo.bwd_recv_rows)) * channels * 4 * tensors,
        }
        torch.cuda.empty_cache()


def _calls(p: Problem, deterministic: bool) -> dict[str, Callable[[], object]]:
    """``{name: fn}`` of one mode on its own leaves: the timed phases ``fwd`` / ``bwd`` / ``ckpt``, ``record`` (the
    forward the timed ``bwd`` replays the backward of), the one-shot ``bwd_once`` and ``ckpt_memory`` probes."""

    xs = [x.detach().clone().requires_grad_(True) for x in p.xs]
    ws = [w.detach().clone().requires_grad_(True) for w in p.ws]
    n = len(xs)
    state: dict[str, tuple] = {}

    def layer(*args) -> tuple:
        if n == 1:
            return (
                kcp_api.cake_causal_conv1d_kcp(
                    args[0], args[1], p.ctx, deterministic=deterministic
                ),
            )
        return tuple(
            kcp_api.cake_causal_conv1d_multi_kcp(
                args[:n], args[n:], p.ctx, deterministic=deterministic
            )
        )

    def reset() -> None:
        for leaf in (*xs, *ws):
            leaf.grad = None

    def grads() -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(x.grad, w.grad) for x, w in zip(xs, ws, strict=True)]

    def fwd():
        return layer(*xs, *ws)

    def record():
        state["ys"] = layer(*xs, *ws)
        return state["ys"]

    def bwd():
        reset()
        torch.autograd.backward(list(state["ys"]), list(p.dys), retain_graph=True)
        return grads()

    def bwd_once():
        reset()
        torch.autograd.backward(list(state.pop("ys")), list(p.dys))
        return grads()

    def ckpt():
        reset()
        ys = checkpoint(layer, *xs, *ws, use_reentrant=False)
        torch.autograd.backward(list(ys), list(p.dys))
        return grads()

    def ckpt_memory() -> None:
        # The leaves' gradients are dropped again, so the retained bytes are the package's.
        ckpt()
        reset()

    return {
        "fwd": fwd,
        "record": record,
        "bwd": bwd,
        "bwd_once": bwd_once,
        "ckpt": ckpt,
        "ckpt_memory": ckpt_memory,
    }


def _check(p: Problem, device: torch.device) -> dict:
    """Per rank: nondet ``y`` / ``dx`` bitwise the det ones, each ``dweight`` partial within the budget of det's."""

    outs = {}
    for mode, deterministic in MODES.items():
        calls = _calls(p, deterministic)
        ys = calls["record"]()
        g = calls["bwd"]()
        torch.cuda.synchronize(device)
        outs[mode] = (
            [y.detach().clone() for y in ys],
            [(dx.clone(), dw.clone()) for dx, dw in g],
        )
    (ys_d, g_d), (ys_n, g_n) = outs["det"], outs["nondet"]
    worst_ratio, worst_rel = 0.0, 0.0
    for (_, dw_d), (_, dw_n) in zip(g_d, g_n, strict=True):
        diff = (dw_n.double() - dw_d.double()).abs()
        worst_ratio = max(
            worst_ratio, float((diff / (DW_ATOL + DW_RTOL * dw_d.double().abs())).max())
        )
        worst_rel = max(
            worst_rel, float(diff.max() / dw_d.double().abs().max().clamp_min(1e-30))
        )
    flags = {
        "y_bitwise": all(
            torch.equal(_bits(a), _bits(b)) for a, b in zip(ys_d, ys_n, strict=True)
        ),
        "dx_bitwise": all(
            torch.equal(_bits(a[0]), _bits(b[0])) for a, b in zip(g_d, g_n, strict=True)
        ),
        "dweight_within_budget": worst_ratio <= 1.0,
    }
    ranks = _gather(
        {
            **flags,
            "dweight_ratio_to_budget": worst_ratio,
            "dweight_max_rel_diff": worst_rel,
        }
    )
    return {
        **{key: all(r[key] for r in ranks) for key in flags},
        "dweight_ratio_to_budget": max(r["dweight_ratio_to_budget"] for r in ranks),
        "dweight_max_rel_diff": max(r["dweight_max_rel_diff"] for r in ranks),
        "ok": all(all(r[key] for key in flags) for r in ranks),
    }


def bench_row(
    tokens: int,
    channels: int,
    *,
    args: argparse.Namespace,
    rank: int,
    world: int,
    device: torch.device,
    ahead: _GpuAhead,
) -> dict:
    p = Problem(tokens, channels, args.tensors, args.layout, rank, world, device)
    result: dict = {
        "tokens": tokens,
        "channels": channels,
        "tensors": args.tensors,
        "documents": list(p.lens),
        "tokens_local": _gather(p.tokens_local),
        "halo": _gather(p.halo),
        "rounds": args.rounds,
        "modes": {},
    }
    # One untimed pass per mode: the JIT loads and the first exchanges.
    for deterministic in MODES.values():
        calls = _calls(p, deterministic)
        calls["fwd"]()
        calls["record"]()
        calls["bwd"]()
        calls["ckpt"]()
    torch.cuda.synchronize(device)
    entries: dict[str, dict] = {mode: {} for mode in MODES}
    # Initialization and memory: the first call of each phase after clearing the workspace caches, then a steady one.
    for mode, deterministic in MODES.items():
        calls = _calls(p, deterministic)
        for phase in PHASES:
            _clear_package_caches()
            for when in ("first", "steady"):
                if phase == "bwd":
                    calls["record"]()
                probe = {
                    "fwd": calls["fwd"],
                    "bwd": calls["bwd_once"],
                    "ckpt": calls["ckpt_memory"],
                }[phase]
                entries[mode][f"{phase}_{when}"] = _probe(probe, device)
    # Interleaved rounds: every phase of both modes per round, the first mode alternating.
    timed = {mode: _calls(p, deterministic) for mode, deterministic in MODES.items()}
    for calls in timed.values():
        calls["record"]()
    samples = {
        mode: {f"{phase}_{kind}": [] for phase in PHASES for kind in KINDS}
        for mode in MODES
    }
    per_rank = {
        mode: {f"{phase}_{kind}": [] for phase in PHASES for kind in ("gpu", "ahead")}
        for mode in MODES
    }
    late = {mode: 0 for mode in MODES}
    order = list(MODES)
    for index in range(args.rounds):
        for mode in order if index % 2 == 0 else order[::-1]:
            calls = timed[mode]
            for phase in PHASES:
                gpu = _gpu_ms(calls[phase])
                ahead_ms, ahead_late = ahead(calls[phase])
                aheads = _gather(ahead_ms)
                late[mode] += sum(_gather(ahead_late))
                wall, host = _wall_ms(calls[phase], device)
                per_rank[mode][f"{phase}_gpu"].append([t * 1e3 for t in gpu])
                per_rank[mode][f"{phase}_ahead"].append([t * 1e3 for t in aheads])
                for kind, value in (
                    ("gpu", max(gpu)),
                    ("ahead", max(aheads)),
                    ("wall", wall),
                    ("host", host),
                ):
                    samples[mode][f"{phase}_{kind}"].append(value * 1e3)
    for mode in MODES:
        entry = entries[mode]
        for key, values in samples[mode].items():
            entry[f"{key}_us"] = _median(values)
            entry[f"{key}_us_range"] = [min(values), max(values)]
        entry["per_rank_us"] = per_rank[mode]
        entry["ahead_late"] = late[mode]
    timed.clear()
    result["modes"] = entries
    keys = [f"{phase}_{kind}" for phase in PHASES for kind in KINDS]
    result["speedup"] = {
        key: entries["det"][f"{key}_us"] / entries["nondet"][f"{key}_us"]
        for key in keys
    }
    result["speedup_range"] = {}
    for key in keys:
        ratios = [
            d / n
            for d, n in zip(samples["det"][key], samples["nondet"][key], strict=True)
        ]
        result["speedup_range"][key] = [min(ratios), max(ratios)]
    result["check"] = _check(p, device)
    del p
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _format(result: dict, world: int) -> list[str]:
    lines = []
    head = f"W={world} S={result['tokens']:6d} C={result['channels']:5d}"
    if result["tensors"] > 1:
        head += f" x{result['tensors']}"
    det, nondet = result["modes"]["det"], result["modes"]["nondet"]
    for phase in PHASES:
        cells = []
        for entry in (det, nondet):
            lo, hi = entry[f"{phase}_ahead_us_range"]
            cells.append(
                f"{entry[f'{phase}_gpu_us']:9.2f} gpu {entry[f'{phase}_ahead_us']:9.2f} ahead "
                f"[{lo:9.2f}, {hi:9.2f}] {entry[f'{phase}_wall_us']:9.2f} wall"
            )
        ratios = " ".join(
            f"{kind} {result['speedup'][f'{phase}_{kind}']:5.3f}x "
            f"[{result['speedup_range'][f'{phase}_{kind}'][0]:5.3f}, {result['speedup_range'][f'{phase}_{kind}'][1]:5.3f}]"
            for kind in ("gpu", "ahead", "wall")
        )
        lines.append(
            f"{head} {phase:4s} | det {cells[0]} | nondet {cells[1]} | det/nondet {ratios}"
        )
    for mode, entry in (("det", det), ("nondet", nondet)):
        first, steady = entry["bwd_first"], entry["bwd_steady"]
        lines.append(
            f"{head} {mode} bwd first call {first['wall_us']:9.2f} us peak temp {first['peak_temp_bytes']} B "
            f"retained {first['retained_bytes']} B | steady {steady['wall_us']:9.2f} us peak temp "
            f"{steady['peak_temp_bytes']} B | ckpt steady peak temp {entry['ckpt_steady']['peak_temp_bytes']} B "
            f"retained {entry['ckpt_steady']['retained_bytes']} B | ahead_late {entry['ahead_late']}"
        )
    lines.append(
        f"{head} tokens per rank {result['tokens_local']} check {result['check']}"
    )
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=None,
        help="whole-pack token counts S; with --channels the rows are their product (default: the matrix)",
    )
    parser.add_argument(
        "--channels", type=int, nargs="*", default=None, help="channel counts C"
    )
    parser.add_argument(
        "--tensors",
        type=int,
        default=1,
        help="convolutions per layer: 1 runs cake_causal_conv1d_kcp, more run cake_causal_conv1d_multi_kcp",
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUTS,
        default="contiguous",
        help="the KCP partition layout",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="interleaved det / nondet timing rounds (each value: the median over the rounds, with its range)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="rank 0 writes all rows to this JSON file",
    )
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.tensors < 1:
        parser.error("--rounds and --tensors must be at least 1")
    warnings.simplefilter("ignore", ExperimentalWarning)
    rank, world, device = _init()
    for mode, deterministic in MODES.items():
        if not conv.is_cake_causal_conv1d_train_supported(
            device, deterministic=deterministic
        ):
            raise SystemExit(
                f"the {mode} package has no exported route on this device (or is not rendered)"
            )
    if torch.are_deterministic_algorithms_enabled():
        raise SystemExit(
            "torch.use_deterministic_algorithms(True) would run both modes deterministically"
        )
    if rank == 0:
        print(
            f"device {torch.cuda.get_device_name(device)} CUDA {torch.version.cuda} torch {torch.__version__} "
            f"W={world} layout={args.layout} tensors={args.tensors} rounds={args.rounds}"
        )
    if args.tokens is None and args.channels is None:
        rows = perf_rows()
    else:
        rows = [
            (s, c)
            for s in (args.tokens or [P2_TOKENS])
            for c in (args.channels or [P1_CHANNELS])
        ]
    ahead = _GpuAhead(device)
    results = []
    for tokens, channels in rows:
        result = bench_row(
            tokens,
            channels,
            args=args,
            rank=rank,
            world=world,
            device=device,
            ahead=ahead,
        )
        results.append(result)
        if rank == 0:
            for line in _format(result, world):
                print(line, flush=True)
    ok = all(result["check"]["ok"] for result in results)
    if rank == 0:
        if not ok:
            print("CHECK FAILED: " + ", ".join(
                f"S={r['tokens']} C={r['channels']}" for r in results if not r["check"]["ok"]
            ))  # fmt: skip
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(
                    {
                        "device": torch.cuda.get_device_name(device),
                        "capability": list(torch.cuda.get_device_capability(device)),
                        "sm_count": torch.cuda.get_device_properties(
                            device
                        ).multi_processor_count,
                        "cuda": torch.version.cuda,
                        "torch": torch.__version__,
                        "world": world,
                        "layout": args.layout,
                        "tensors": args.tensors,
                        "rows": results,
                    },
                    handle,
                    indent=1,
                )
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
