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
* ``gpu_us``: the median GPU time of the phase's kernels (CUPTI activity, cold L2;
  ``flashinfer.testing.bench_gpu_time(enable_cupti=True)``);
* ``wall_us``: the median host wall time of one synchronized call (``perf_counter`` between two device
  synchronizations, warm L2) and ``host_us``: the median time to enqueue it (the call's return, no synchronization);
* ``peak_temp_bytes`` / ``retained_bytes``: the allocator's peak above the bytes held before the call and the bytes
  still held after it, on the first call with the package's own workspaces (no caller workspace; caches cleared)
  and on a steady-state call; plus the public backward workspace size of each mode.

``speedup`` is ``det / nondet`` of each timing (above 1: the non-deterministic mode is faster).  A sanity check per
row (not a gate) records whether ``dx`` is bitwise equal between the modes and the largest relative ``dweight``
difference.  The route is chosen as in the deterministic benchmark (``--weight-dtype``, ``--bias``,
``--silu-grad-preactivation``); ``--side-kernel`` adds the backward's wall latency beside a kernel resident on another
stream (FlashInfer issues #7038 / #7548).

Example::

    python benchmarks/bench_cake_causal_conv1d_train.py
    python benchmarks/bench_cake_causal_conv1d_train.py --channels 512 1280 --tokens 250340 --json out.json
    python benchmarks/bench_cake_causal_conv1d_train.py --silu-grad-preactivation fp32 --bias float32
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import torch

import flashinfer.cake_causal_conv1d_det_train_backend as det_backend
import flashinfer.experimental.cake.cake_causal_conv1d_train as conv
import flashinfer.experimental.cake.cake_causal_conv1d_train_backend as nondet_backend
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
    cake_jit as loader,
)
from flashinfer.testing import bench_gpu_time

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
WALL_WARMUPS = 3
WALL_SAMPLES = 20
SIDE_KERNEL_CYCLES = 10_000_000
SIDE_KERNEL_WARMUPS = 3
SIDE_KERNEL_SAMPLES = 7


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
            "bias_dtype": None if self.bias_dtype is None else str(self.bias_dtype).removeprefix("torch."),
            "silu_grad_preactivation": self.silu_grad_preactivation,
            "backward_route": self.backward_route(),
        }


def perf_rows() -> list[tuple[int, int]]:
    return [(tokens, P1_CHANNELS) for tokens in P1_TOKENS] + [(P2_TOKENS, c) for c in P2_CHANNELS]


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


def _latency_beside(fn: Callable[[], object], device: torch.device) -> tuple[float, float]:
    side = torch.cuda.Stream(device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

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
    def __init__(self, tokens: int, channels: int, route: Route, device: torch.device) -> None:
        generator = torch.Generator(device=device).manual_seed(SEED)
        self.lens = doc_lengths(tokens)
        self.cu = torch.tensor([0, *torch.tensor(self.lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
        self.x = torch.randn(tokens, channels, device=device, generator=generator, dtype=torch.bfloat16)
        # N(0, 0.25): variance 0.25, standard deviation 0.5 (the bias too).
        self.w = (torch.randn(channels, WIDTH, device=device, generator=generator) * 0.5).to(route.weight_dtype)
        self.b = (
            None
            if route.bias_dtype is None
            else (torch.randn(channels, device=device, generator=generator) * 0.5).to(route.bias_dtype)
        )
        self.dy = torch.randn(tokens, channels, device=device, generator=generator, dtype=torch.bfloat16)


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
        fws = conv.cake_causal_conv1d_train_forward_workspace(tokens, channels, device, deterministic=deterministic)
        bws = conv.cake_causal_conv1d_train_backward_workspace(tokens, channels, device, deterministic=deterministic)

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
            x_, w_, p.cu, b_, silu_grad_preactivation=route.silu_grad_preactivation, deterministic=deterministic
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

    return {"fwd": fwd, "bwd": bwd, "step": step, "ckpt": ckpt, "ckpt_memory": ckpt_dropping_grads}


def bench_row(tokens: int, channels: int, *, route: Route, device: torch.device, side_kernel: bool) -> dict:
    p = Problem(tokens, channels, route, device)
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    result: dict = {
        "tokens": tokens,
        "channels": channels,
        "documents": list(p.lens),
        "nondet_program": loader.selected_modules(arch, route.backward_route(), C=channels, S=tokens)[0],
        **route.describe(),
        "modes": {},
    }
    grads = {}
    for mode, deterministic in MODES.items():
        entry: dict = {
            "backward_workspace_bytes": conv.cake_causal_conv1d_train_backward_workspace_bytes(
                tokens, channels, device, deterministic=deterministic
            )
        }
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
            for phase in PHASES:
                entry[f"{phase}_gpu_us"] = _gpu_ms(calls[phase]) * 1e3
                wall, host = _host_ms(calls[phase], device)
                entry[f"{phase}_wall_us"], entry[f"{phase}_host_us"] = wall * 1e3, host * 1e3
            if side_kernel:
                alone, beside = _latency_beside(calls["bwd"], device)
                entry["bwd_alone_us"], entry["bwd_beside_us"] = alone * 1e3, beside * 1e3
                entry["bwd_beside_ratio"] = beside / alone
            out = calls["bwd"]()
            torch.cuda.synchronize(device)
            grads[mode] = tuple(t.clone() for t in out)
        except Exception as exc:  # keep the other mode's row
            entry["error"] = repr(exc)[:300]
        result["modes"][mode] = entry
        torch.cuda.empty_cache()
    det, nondet = result["modes"]["det"], result["modes"]["nondet"]
    result["speedup"] = {
        f"{phase}_{kind}": det[f"{phase}_{kind}_us"] / nondet[f"{phase}_{kind}_us"]
        for phase in PHASES
        for kind in ("gpu", "wall", "host")
        if f"{phase}_{kind}_us" in det and f"{phase}_{kind}_us" in nondet
    }
    if "det" in grads and "nondet" in grads:
        (dx_d, dw_d, *_), (dx_n, dw_n, *_) = grads["det"], grads["nondet"]
        result["check"] = {
            "dx_bitwise_equal": bool(torch.equal(dx_d, dx_n)),
            "dweight_max_rel_diff": float(
                ((dw_n.double() - dw_d.double()).abs().max() / dw_d.double().abs().max().clamp_min(1e-30)).item()
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
            cells.append("-" if gpu is None else f"{gpu:9.2f} gpu {wall:9.2f} wall")
        ratio = result["speedup"].get(f"{phase}_gpu")
        lines.append(
            f"{head} {phase:4s} | det {cells[0]} | nondet {cells[1]} | det/nondet gpu "
            + ("-" if ratio is None else f"{ratio:5.3f}x")
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
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="*", default=None,
                        help="packed token counts S; with --channels the rows are their product (default: the matrix)")
    parser.add_argument("--channels", type=int, nargs="*", default=None, help="channel counts C")
    parser.add_argument("--json", type=str, default=None, help="write all rows to this JSON file")
    parser.add_argument("--weight-dtype", choices=sorted(DTYPES), default="float32",
                        help="dtype of the taps and of dweight (bfloat16: the _dt routes)")
    parser.add_argument("--bias", choices=("none", *sorted(DTYPES)), default="none",
                        help="add a per-channel bias of this dtype (the _bias routes)")
    parser.add_argument("--silu-grad-preactivation", choices=("bf16", "fp32"), default="bf16",
                        help="the weight gradient's SiLU' policy (fp32: the _fp32g routes)")
    parser.add_argument("--side-kernel", action="store_true",
                        help="also time the backward's wall latency alone and beside a 1-thread kernel on another stream")
    args = parser.parse_args(argv)  # fmt: skip
    route = Route(
        weight_dtype=DTYPES[args.weight_dtype],
        bias_dtype=None if args.bias == "none" else DTYPES[args.bias],
        silu_grad_preactivation=args.silu_grad_preactivation,
    )
    warnings.simplefilter("ignore", ExperimentalWarning)
    device = torch.device("cuda", torch.cuda.current_device())
    for mode, deterministic in MODES.items():
        if not conv.is_cake_causal_conv1d_train_supported(device, deterministic=deterministic):
            raise SystemExit(f"the {mode} package has no exported route on this device (or is not rendered)")
    if torch.are_deterministic_algorithms_enabled():
        raise SystemExit("torch.use_deterministic_algorithms(True) would run both modes deterministically")
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    if route.backward_route() not in loader.ROUTES[arch]:
        raise SystemExit(f"the {route.backward_route()} route is not exported for {arch}")
    print(f"device {torch.cuda.get_device_name(device)} {arch} CUDA {torch.version.cuda} torch {torch.__version__}")
    print(f"route: {route.describe()}")
    if args.tokens is None and args.channels is None:
        rows = perf_rows()
    else:
        rows = [(s, c) for s in (args.tokens or [P2_TOKENS]) for c in (args.channels or [P1_CHANNELS])]
    results = []
    for tokens, channels in rows:
        result = bench_row(tokens, channels, route=route, device=device, side_kernel=args.side_kernel)
        results.append(result)
        for line in _format(result):
            print(line, flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
                    "cuda": torch.version.cuda,
                    "torch": torch.__version__,
                    "route": route.describe(),
                    "rows": results,
                },
                handle,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
