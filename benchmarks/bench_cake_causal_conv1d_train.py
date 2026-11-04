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

Benchmark of the Cake packed width-4 depthwise causal convolution + SiLU training kernels in their two numerical
modes: ``deterministic=True`` (the deterministic package: a main backward program plus a fixed-order reduce) against
``deterministic=False`` (one backward launch with unordered ``dweight`` / ``dbias`` reductions), both through the
dispatching entry points of ``flashinfer.experimental.cake.cake_causal_conv1d_train``.

The rows are the deterministic benchmark's performance matrix: ``C = 512`` over eight packed token counts and
``S = 250340`` over the channel counts (the multi-sub-tile tiles C = 1280 / 1536 and the partial last tiles
C = 1160 / 1344 included), each packed as four unequal documents of 0.37 / 0.29 / 0.21 / rest of ``S`` tokens (each
nudged off a multiple of 64).  Per row and mode:

* ``fwd``, ``bwd`` and ``step`` (``fwd + bwd``): the explicit calls with caller-owned outputs and workspaces;
* ``ckpt``: ``torch.utils.checkpoint(..., use_reentrant=False)`` around the autograd entry, then its backward (two
  forwards and one backward, the autograd overhead included);
* ``gpu_us``: the GPU time of the phase, cold L2.  ``fwd`` / ``bwd`` / ``step``: the span of the phase's kernels
  (CUPTI activity; ``flashinfer.testing.bench_gpu_time(enable_cupti=True)``).  ``ckpt``: the autograd host time
  exceeds its kernels' time, so a CUPTI span measures host gaps; instead the whole step is enqueued behind a spin
  kernel (the host finishes enqueuing before the GPU starts the step) and timed with CUDA events around it
  (``ckpt_ahead_late`` counts the samples whose enqueue outlasted the spin; 0 expected);
* ``wall_us``: the median host wall time of one synchronized call (``perf_counter`` between two device
  synchronizations, warm L2) and ``host_us``: the median time to enqueue it (the call's return, no synchronization);
* ``peak_temp_bytes`` / ``retained_bytes``: the allocator's peak above the bytes held before the call and the bytes
  still held after it, on the first call with the package's own workspaces (no caller workspace; caches cleared)
  and on a steady-state call; plus the public backward workspace size of each mode.

The timings interleave the modes: ``--rounds`` rounds, each timing every phase of both modes (det first in even
rounds, nondet first in odd ones).  Each value is the median over the rounds of the round's median, with its
``[min, max]`` over the rounds (``*_us_range``).  ``speedup`` is ``det / nondet`` of the medians (above 1: the
non-deterministic mode is faster), ``speedup_range`` the ``[min, max]`` of the per-round ratios.  A sanity check per
row (not a gate) records whether ``dx`` is bitwise equal between the modes and the largest relative ``dweight``
difference.  The route is chosen as in the deterministic benchmark (``--weight-dtype``, ``--bias``,
``--silu-grad-preactivation``); ``--side-kernel`` adds the backward's wall latency beside a kernel resident on another
stream (FlashInfer issues #7038 / #7548).

Non-Cake baselines (``--sota``, default ``all``; ``cake_causal_conv1d_baselines.py``): flash-linear-attention's
``causal_conv1d`` on its Triton (``fla``) and opt-in Gluon (``fla_gluon``) backends, Dao-AILab ``causal_conv1d_fn``
with its atomic (``dao``) and deterministic (``dao_det``) weight gradients, and the cuDNN frontend's
``cudnn.ops.causal_conv1d`` (``cudnn_fe``).  They run through autograd on the row's tensors, interleaved with the
two Cake modes in the same rounds and timed the same way (``fwd``: the training forward with its graph recorded;
``bwd``: the backward of a recorded forward; ``step``: both; ``ckpt``: as above), and are reported per row under
``sota`` with the library version, determinism, feature match and a numerics check against the Cake
non-deterministic result, or as unsupported with the reason.  ``best_sota`` names the fastest eligible arm per phase
and measure, ``speedup_vs_sota`` its time over the non-deterministic time (above 1: Cake faster), and every arm's
``speedup`` its own ratio.  ``comparison`` is the row's two-baseline comparison on one column, the ``gpu`` time
(``COMPARISON_COLUMN``): per phase the deterministic and non-deterministic Cake times, the best eligible arm with
its library, determinism and match (or ``unsupported`` with every arm's reason; a feature- or precision-mismatched
arm is listed but never the best), ``det_over_nondet`` and ``sota_over_nondet``; the ``compare`` lines print it.

Example::

    python benchmarks/bench_cake_causal_conv1d_train.py
    python benchmarks/bench_cake_causal_conv1d_train.py --channels 512 1280 --tokens 250340 --json out.json
    python benchmarks/bench_cake_causal_conv1d_train.py --silu-grad-preactivation fp32 --bias float32
    python benchmarks/bench_cake_causal_conv1d_train.py --channels 512 --tokens 250340 --sota fla_gluon dao
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

import flashinfer.cake_causal_conv1d_det_train_backend as det_backend
import flashinfer.experimental.cake.cake_causal_conv1d_train as conv
import flashinfer.experimental.cake.cake_causal_conv1d_train_backend as nondet_backend
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
    cake_jit as loader,
)
from flashinfer.testing import bench_gpu_time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cake_causal_conv1d_baselines as baselines  # noqa: E402

WIDTH = 4
P1_TOKENS = (239718, 245028, 249866, 250340, 258574, 259042, 259584, 261393)
P1_CHANNELS = 512
P2_TOKENS = 250340
P2_CHANNELS = (128, 256, 1024, 1160, 1280, 1344, 1536, 2048, 4096, 8192)
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 0
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
MODES = {"det": True, "nondet": False}
PHASES = ("fwd", "bwd", "step", "ckpt")
KINDS = ("gpu", "wall", "host")
ROUNDS = 7
WALL_WARMUPS = 3
WALL_SAMPLES = 50
AHEAD_WARMUPS = 2
AHEAD_SAMPLES = 10
# The spin kernel ahead of a ckpt sample lasts at least AHEAD_MIN_MS and AHEAD_ENQUEUE_FACTOR times the longest
# enqueue seen while warming up.
AHEAD_MIN_MS = 2.0
AHEAD_ENQUEUE_FACTOR = 3.0
SIDE_KERNEL_CYCLES = 10_000_000
SIDE_KERNEL_WARMUPS = 3
SIDE_KERNEL_SAMPLES = 7
# The two-baseline comparison of a row (det Cake, nondet Cake, best eligible non-Cake arm) uses one timing column.
COMPARISON_COLUMN = "gpu"
COMPARISON_RULE = (
    "single rank: every arm compared on the CUPTI cold-L2 span (gpu; ckpt: the host-hidden span); "
    "feature- or precision-mismatched arms are listed, never the best"
)


@dataclass(frozen=True)
class Route:
    """The route of a run: the taps' dtype (and the ``dweight``'s), the bias dtype (``None``: no bias) and the
    SiLU' policy of the weight gradient."""

    weight_dtype: torch.dtype = torch.float32
    bias_dtype: Optional[torch.dtype] = None
    silu_grad_preactivation: str = "bf16"

    def backward_route(self) -> str:
        return loader.backward_route(
            bias=self.bias_dtype is not None,
            fp32_grad=self.silu_grad_preactivation == "fp32",
            bf16_taps_or_dweight=self.weight_dtype == torch.bfloat16,
        )

    def describe(self) -> dict:
        return {
            "weight_dtype": str(self.weight_dtype).removeprefix("torch."),
            "bias_dtype": None
            if self.bias_dtype is None
            else str(self.bias_dtype).removeprefix("torch."),
            "silu_grad_preactivation": self.silu_grad_preactivation,
            "backward_route": self.backward_route(),
        }


def perf_rows() -> list[tuple[int, int]]:
    return [(tokens, P1_CHANNELS) for tokens in P1_TOKENS] + [
        (P2_TOKENS, c) for c in P2_CHANNELS
    ]


def doc_lengths(tokens: int) -> tuple[int, ...]:
    lens = []
    for fraction in PERF_FRACTIONS:
        n = int(fraction * tokens)
        if n % 64 == 0:
            n -= 1
        lens.append(n)
    lens.append(tokens - sum(lens))
    if not all(n > 0 for n in lens):
        raise ValueError(f"S={tokens} is too short for the perf packing")
    return tuple(lens)


def _median(samples) -> float:
    return float(statistics.median(list(samples)))


def _gpu_ms(fn: Callable[[], object]) -> float:
    """Median GPU time (ms) of ``fn``'s kernels, cold L2, CUPTI."""

    return _median(bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True))


def _host_ms(fn: Callable[[], object], device: torch.device) -> tuple[float, float]:
    """``(wall_ms, host_ms)``: the median synchronized wall time of one call and the median time to enqueue it."""

    walls, hosts = [], []
    for i in range(WALL_WARMUPS + WALL_SAMPLES):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        if i >= WALL_WARMUPS:
            hosts.append((t1 - t0) * 1e3)
            walls.append((t2 - t0) * 1e3)
    return _median(walls), _median(hosts)


class _GpuAhead:
    """GPU time (ms) of a call whose host enqueue outlasts its kernels: an L2 flush, then a spin kernel long enough
    for the host to enqueue the whole call behind it, then CUDA events around the call (no host gap enters)."""

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
        """``(median_ms, late)``: ``late`` counts the samples whose enqueue outlasted the spin."""

        enqueues = [self._sample(fn, AHEAD_MIN_MS)[1] for _ in range(AHEAD_WARMUPS)]
        spin_ms = max(AHEAD_MIN_MS, AHEAD_ENQUEUE_FACTOR * max(enqueues))
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


def _memory(fn: Callable[[], object], device: torch.device) -> dict:
    """The allocator's peak above the bytes held before ``fn()`` and the bytes still held after it."""

    torch.cuda.synchronize(device)
    gc.collect()
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) - before
    retained = torch.cuda.memory_allocated(device) - before
    del result
    return {"peak_temp_bytes": int(peak), "retained_bytes": int(retained)}


def _latency_beside(
    fn: Callable[[], object], device: torch.device
) -> tuple[float, float]:
    side = torch.cuda.Stream(device=device)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    def timed(beside: bool) -> float:
        torch.cuda.synchronize(device)
        if beside:
            with torch.cuda.stream(side):
                torch.cuda._sleep(SIDE_KERNEL_CYCLES)
        torch.cuda._sleep(1_000_000)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(SIDE_KERNEL_WARMUPS):
        timed(False)
    alone = _median(timed(False) for _ in range(SIDE_KERNEL_SAMPLES))
    beside = _median(timed(True) for _ in range(SIDE_KERNEL_SAMPLES))
    return alone, beside


class Problem:
    def __init__(
        self, tokens: int, channels: int, route: Route, device: torch.device
    ) -> None:
        generator = torch.Generator(device=device).manual_seed(SEED)
        self.lens = doc_lengths(tokens)
        self.cu = torch.tensor(
            [0, *torch.tensor(self.lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        self.x = torch.randn(
            tokens, channels, device=device, generator=generator, dtype=torch.bfloat16
        )
        # N(0, 0.25): variance 0.25, standard deviation 0.5 (the bias too).
        self.w = (
            torch.randn(channels, WIDTH, device=device, generator=generator) * 0.5
        ).to(route.weight_dtype)
        self.b = (
            None
            if route.bias_dtype is None
            else (torch.randn(channels, device=device, generator=generator) * 0.5).to(
                route.bias_dtype
            )
        )
        self.dy = torch.randn(
            tokens, channels, device=device, generator=generator, dtype=torch.bfloat16
        )


def _calls(p: Problem, route: Route, deterministic: bool, *, own_workspace: bool):
    """``{phase: fn}`` of one mode: the explicit calls into preallocated outputs (``own_workspace``: caller-owned
    forward / backward workspaces of the mode) and the checkpointed autograd step."""

    from torch.utils.checkpoint import checkpoint

    tokens, channels = p.x.shape
    device = p.x.device
    y = torch.empty_like(p.x)
    dx = torch.empty_like(p.x)
    dw = torch.empty(channels, WIDTH, dtype=p.w.dtype, device=device)
    db = None if p.b is None else torch.empty(channels, dtype=p.b.dtype, device=device)
    fws = bws = None
    if own_workspace:
        fws = conv.cake_causal_conv1d_train_forward_workspace(
            tokens, channels, device, deterministic=deterministic
        )
        bws = conv.cake_causal_conv1d_train_backward_workspace(
            tokens, channels, device, deterministic=deterministic
        )

    def fwd():
        return conv.cake_causal_conv1d_train_forward(
            p.x, p.w, p.cu, bias=p.b, out=y, workspace=fws, deterministic=deterministic
        )

    def bwd():
        return conv.cake_causal_conv1d_train_backward(
            p.x, p.w, p.cu, p.dy, bias=p.b, silu_grad_preactivation=route.silu_grad_preactivation,
            workspace=bws, dx=dx, dweight=dw, dbias=db, deterministic=deterministic,
        )  # fmt: skip

    def step():
        fwd()
        return bwd()

    xl = p.x.detach().clone().requires_grad_(True)
    wl = p.w.detach().clone().requires_grad_(True)
    bl = None if p.b is None else p.b.detach().clone().requires_grad_(True)
    leaves = [leaf for leaf in (xl, wl, bl) if leaf is not None]

    def layer(x_, w_, b_):
        return conv.cake_causal_conv1d(
            x_,
            w_,
            p.cu,
            b_,
            silu_grad_preactivation=route.silu_grad_preactivation,
            deterministic=deterministic,
        )

    def ckpt():
        for leaf in leaves:
            leaf.grad = None
        out = checkpoint(layer, xl, wl, bl, use_reentrant=False)
        out.backward(p.dy)
        return tuple(leaf.grad for leaf in leaves)

    def ckpt_dropping_grads():
        # The memory probe: the leaves' gradients are dropped again, so the retained bytes are the package's.
        ckpt()
        for leaf in leaves:
            leaf.grad = None

    return {
        "fwd": fwd,
        "bwd": bwd,
        "step": step,
        "ckpt": ckpt,
        "ckpt_memory": ckpt_dropping_grads,
    }


def _cake_reference(p: Problem, route: Route) -> tuple:
    """The non-deterministic Cake ``(y, dx, dweight, dbias)`` of the row through autograd (the baselines' check)."""

    leaves = [
        t.detach().clone().requires_grad_(True)
        for t in (p.x, p.w, p.b)
        if t is not None
    ]
    y = conv.cake_causal_conv1d(
        leaves[0],
        leaves[1],
        p.cu,
        leaves[2] if len(leaves) > 2 else None,
        silu_grad_preactivation=route.silu_grad_preactivation,
        deterministic=False,
    )
    y.backward(p.dy)
    return (y.detach(), *(leaf.grad for leaf in leaves))


def _baselines(
    p: Problem, route: Route, sota: Sequence[str], device: torch.device
) -> tuple[dict[str, dict], dict[str, dict]]:
    """``(entries, calls)`` of the requested non-Cake arms on the row: each built, checked against the Cake result
    and warmed up (JIT compilation and autotuning happen here, outside the timing)."""

    entries: dict[str, dict] = {}
    calls: dict[str, dict] = {}
    if not sota:
        return entries, calls
    reference = _cake_reference(p, route)
    for name in sota:
        entry, arm = baselines.single_rank_arm(
            name, x=p.x, w=p.w, b=p.b, dy=p.dy, cu=p.cu, lens=p.lens,
            cake_preactivation=route.silu_grad_preactivation, reference=reference,
        )  # fmt: skip
        entries[name] = entry
        if arm is None:
            continue
        try:
            with baselines.environment(entry["env"]):
                for phase in PHASES:
                    arm[phase]()
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 -- a failing library is reported, the row goes on
            entry["unsupported"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            entry["eligible"] = False
            continue
        entry["ckpt_ahead_late"] = 0
        calls[name] = arm
    del reference
    torch.cuda.empty_cache()
    return entries, calls


def bench_row(
    tokens: int,
    channels: int,
    *,
    route: Route,
    device: torch.device,
    side_kernel: bool,
    rounds: int,
    ahead: _GpuAhead,
    sota: Sequence[str] = (),
) -> dict:
    p = Problem(tokens, channels, route, device)
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    result: dict = {
        "tokens": tokens,
        "channels": channels,
        "documents": list(p.lens),
        "nondet_program": loader.selected_modules(
            arch, route.backward_route(), C=channels, S=tokens
        )[0],
        **route.describe(),
        "rounds": rounds,
        "modes": {},
    }
    entries: dict[str, dict] = {}
    timed: dict[str, dict] = {}
    for mode, deterministic in MODES.items():
        entry: dict = {
            "backward_workspace_bytes": conv.cake_causal_conv1d_train_backward_workspace_bytes(
                tokens, channels, device, deterministic=deterministic
            )
        }
        entries[mode] = entry
        try:
            # Memory: the package's own workspaces, first call after clearing the caches, then a steady-state call.
            _clear_package_caches()
            calls = _calls(p, route, deterministic, own_workspace=False)
            for phase in ("fwd", "bwd", "ckpt"):
                probe = calls["ckpt_memory" if phase == "ckpt" else phase]
                entry[f"{phase}_memory_first"] = _memory(probe, device)
                entry[f"{phase}_memory_steady"] = _memory(probe, device)
            # Timing: caller-owned workspaces.
            calls = _calls(p, route, deterministic, own_workspace=True)
            for phase in PHASES:
                calls[phase]()
            torch.cuda.synchronize(device)
            timed[mode] = calls
            entry["ckpt_ahead_late"] = 0
        except Exception as exc:  # keep the other mode's row
            entry["error"] = repr(exc)[:300]
    sota_entries, sota_calls = _baselines(p, route, sota, device)
    # Interleaved rounds: every phase of both modes and of every baseline per round, the order alternating.
    runners = {
        **{mode: (calls, {}, entries[mode]) for mode, calls in timed.items()},
        **{
            name: (calls, sota_entries[name]["env"], sota_entries[name])
            for name, calls in sota_calls.items()
        },
    }
    samples = {
        name: {f"{phase}_{kind}": [] for phase in PHASES for kind in KINDS}
        for name in runners
    }
    order = list(runners)
    for index in range(rounds):
        for name in order if index % 2 == 0 else order[::-1]:
            calls, env, record = runners[name]
            if "error" in record or "unsupported" in record:
                continue
            try:
                values = {}
                with baselines.environment(env):
                    for phase in PHASES:
                        if phase == "ckpt":
                            gpu, late = ahead(calls[phase])
                            record["ckpt_ahead_late"] += late
                        else:
                            gpu = _gpu_ms(calls[phase])
                        wall, host = _host_ms(calls[phase], device)
                        values.update(
                            {
                                f"{phase}_gpu": gpu * 1e3,
                                f"{phase}_wall": wall * 1e3,
                                f"{phase}_host": host * 1e3,
                            }
                        )
            except Exception as exc:  # keep the other rows
                if name in MODES:
                    record["error"] = repr(exc)[:300]
                else:
                    record["unsupported"] = repr(exc)[:300]
                    record["eligible"] = False
                continue
            for key, value in values.items():
                samples[name][key].append(value)
    for name, per_key in samples.items():
        record = runners[name][2]
        if "error" in record or "unsupported" in record:
            continue
        for key, values in per_key.items():
            record[f"{key}_us"] = _median(values)
            record[f"{key}_us_range"] = [min(values), max(values)]
    sota_calls.clear()
    grads = {}
    for mode, calls in timed.items():
        entry = entries[mode]
        if "error" in entry:
            continue
        try:
            if side_kernel:
                alone, beside = _latency_beside(calls["bwd"], device)
                entry["bwd_alone_us"], entry["bwd_beside_us"] = (
                    alone * 1e3,
                    beside * 1e3,
                )
                entry["bwd_beside_ratio"] = beside / alone
            out = calls["bwd"]()
            torch.cuda.synchronize(device)
            grads[mode] = tuple(t.clone() for t in out)
        except Exception as exc:  # keep the other mode's row
            entry["error"] = repr(exc)[:300]
    result["modes"] = entries
    det, nondet = entries["det"], entries["nondet"]
    if sota:
        result["sota"] = sota_entries
        result["best_sota"], result["speedup_vs_sota"] = baselines.best_baseline(
            sota_entries,
            nondet,
            [f"{phase}_{kind}" for phase in PHASES for kind in KINDS],
        )
        result["comparison"] = baselines.comparison(
            result, PHASES, COMPARISON_COLUMN, COMPARISON_RULE
        )
    keys = [
        f"{phase}_{kind}"
        for phase in PHASES
        for kind in KINDS
        if f"{phase}_{kind}_us" in det and f"{phase}_{kind}_us" in nondet
    ]
    result["speedup"] = {key: det[f"{key}_us"] / nondet[f"{key}_us"] for key in keys}
    result["speedup_range"] = {}
    for key in keys:
        ratios = [
            d / n
            for d, n in zip(samples["det"][key], samples["nondet"][key], strict=True)
        ]
        result["speedup_range"][key] = [min(ratios), max(ratios)]
    if "det" in grads and "nondet" in grads:
        (dx_d, dw_d, *_), (dx_n, dw_n, *_) = grads["det"], grads["nondet"]
        result["check"] = {
            "dx_bitwise_equal": bool(torch.equal(dx_d, dx_n)),
            "dweight_max_rel_diff": float(
                (
                    (dw_n.double() - dw_d.double()).abs().max()
                    / dw_d.double().abs().max().clamp_min(1e-30)
                ).item()
            ),
        }
    return result


def _format(result: dict) -> list[str]:
    lines = []
    head = f"S={result['tokens']:6d} C={result['channels']:5d}"
    det, nondet = result["modes"]["det"], result["modes"]["nondet"]
    for phase in PHASES:
        cells = []
        for entry in (det, nondet):
            gpu, wall = entry.get(f"{phase}_gpu_us"), entry.get(f"{phase}_wall_us")
            if gpu is None:
                cells.append("-")
                continue
            lo, hi = entry[f"{phase}_gpu_us_range"]
            cells.append(f"{gpu:9.2f} gpu [{lo:9.2f}, {hi:9.2f}] {wall:9.2f} wall")
        ratio = result["speedup"].get(f"{phase}_gpu")
        spread = result.get("speedup_range", {}).get(f"{phase}_gpu")
        lines.append(
            f"{head} {phase:4s} | det {cells[0]} | nondet {cells[1]} | det/nondet gpu "
            + (
                "-"
                if ratio is None
                else f"{ratio:5.3f}x [{spread[0]:5.3f}, {spread[1]:5.3f}]"
            )
        )
    for mode, entry in (("det", det), ("nondet", nondet)):
        if "error" in entry:
            lines.append(f"{head} {mode}: {entry['error']}")
            continue
        mem = entry.get("bwd_memory_first", {})
        lines.append(
            f"{head} {mode} bwd workspace {entry['backward_workspace_bytes']} B, first call peak temp "
            f"{mem.get('peak_temp_bytes')} B retained {mem.get('retained_bytes')} B"
        )
        if "bwd_beside_us" in entry:
            lines.append(
                f"{head} {mode} bwd beside a resident kernel: alone {entry['bwd_alone_us']:9.2f} us, "
                f"beside {entry['bwd_beside_us']:9.2f} us, {entry['bwd_beside_ratio']:.2f}x"
            )
    if "check" in result:
        lines.append(f"{head} check {result['check']}")
    lines.extend(baselines.format_sota(result, head, PHASES, COMPARISON_COLUMN))
    lines.extend(baselines.format_comparison(result, head))
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
        help="packed token counts S; with --channels the rows are their product (default: the matrix)",
    )
    parser.add_argument(
        "--channels", type=int, nargs="*", default=None, help="channel counts C"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="write all rows to this JSON file"
    )
    parser.add_argument(
        "--weight-dtype",
        choices=sorted(DTYPES),
        default="float32",
        help="dtype of the taps and of dweight (bfloat16: the _dt routes)",
    )
    parser.add_argument(
        "--bias",
        choices=("none", *sorted(DTYPES)),
        default="none",
        help="add a per-channel bias of this dtype (the _bias routes)",
    )
    parser.add_argument(
        "--silu-grad-preactivation",
        choices=("bf16", "fp32"),
        default="bf16",
        help="the weight gradient's SiLU' policy (fp32: the _fp32g routes)",
    )
    parser.add_argument(
        "--side-kernel",
        action="store_true",
        help="also time the backward's wall latency alone and beside a 1-thread kernel on another stream",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="interleaved det / nondet timing rounds (each value: the median over the rounds, with its range)",
    )
    parser.add_argument(
        "--sota",
        nargs="*",
        default=["all"],
        help=f"non-Cake baselines: all (default), none, or names from {list(baselines.SINGLE_RANK_ARMS)}",
    )
    args = parser.parse_args(argv)  # fmt: skip
    sota = baselines.parse_arms(args.sota, baselines.SINGLE_RANK_ARMS)
    route = Route(
        weight_dtype=DTYPES[args.weight_dtype],
        bias_dtype=None if args.bias == "none" else DTYPES[args.bias],
        silu_grad_preactivation=args.silu_grad_preactivation,
    )
    warnings.simplefilter("ignore", ExperimentalWarning)
    device = torch.device("cuda", torch.cuda.current_device())
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
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    if route.backward_route() not in loader.ROUTES[arch]:
        raise SystemExit(
            f"the {route.backward_route()} route is not exported for {arch}"
        )
    print(
        f"device {torch.cuda.get_device_name(device)} {arch} CUDA {torch.version.cuda} torch {torch.__version__}"
    )
    print(f"route: {route.describe()}")
    if args.tokens is None and args.channels is None:
        rows = perf_rows()
    else:
        rows = [
            (s, c)
            for s in (args.tokens or [P2_TOKENS])
            for c in (args.channels or [P1_CHANNELS])
        ]
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    ahead = _GpuAhead(device)
    results = []
    for tokens, channels in rows:
        result = bench_row(
            tokens,
            channels,
            route=route,
            device=device,
            side_kernel=args.side_kernel,
            rounds=args.rounds,
            ahead=ahead,
            sota=sota,
        )
        results.append(result)
        torch.cuda.empty_cache()
        for line in _format(result):
            print(line, flush=True)
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
                    "route": route.describe(),
                    "sota_arms": sota,
                    "rows": results,
                },
                handle,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
