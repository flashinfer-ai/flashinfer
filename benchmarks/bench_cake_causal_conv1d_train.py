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

Benchmark of the Cake packed width-4 depthwise causal convolution + SiLU
training kernels against the training-side comparators: flash-linear-attention's
Triton ``causal_conv1d`` (when installed; the baseline of the kernels' claim)
and eager FP32 autograd of the document-local formula.

The rows are the kernels' performance matrix: ``C = 512`` over eight packed
token counts and ``S = 250340`` over six channel counts, each packed as four
unequal documents of 0.37 / 0.29 / 0.21 / rest of ``S`` tokens (each nudged off
a multiple of 64).  Every row reports forward, backward (``dx`` and the
deterministic FP32 ``dweight``) and their sum in microseconds plus the achieved
GB/s over the logical byte volume (forward ``4SC + 16C``, backward ``6SC + 32C
+ 32 C rows`` (``rows`` the FP32 partial rows of the grid of the backward
program the row's channel count selects; the JSON names that program) for the
partial workspace's round trip).  Timing uses CUPTI kernel activity with a
cold L2 (``flashinfer.testing.bench_gpu_time(enable_cupti=True)``);
multi-kernel comparators report the sum of their kernels' GPU time.

``--side-kernel`` adds, per row and comparator, the backward's wall latency on
its stream (CUDA events, median of seven calls after three warm-ups) alone and
with a kernel resident on another stream, and their ratio (FlashInfer issue
#7038: a cooperative launch waits for such a kernel to finish; the Cake
backward is two ordinary launches).  ``one-thread`` is the issue's recipe,
one CTA of one thread (``torch.cuda._sleep``); ``nccl-like`` a spin kernel of
8 CTAs x 512 threads, the footprint of a communication kernel, which takes
slots from eight SMs so the displaced CTAs run in a later wave.  Information,
not a gate: the kernel's own GPU time is unchanged either way.

Example::

    python benchmarks/bench_cake_causal_conv1d_train.py
    python benchmarks/bench_cake_causal_conv1d_train.py --channels 512 --tokens 250340 --json out.json
    python benchmarks/bench_cake_causal_conv1d_train.py --channels 512 --tokens 261393 --side-kernel nccl-like
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable, Optional

import torch

from flashinfer.cake_causal_conv1d_train import (
    cake_causal_conv1d_train_backward,
    cake_causal_conv1d_train_backward_workspace,
    cake_causal_conv1d_train_forward,
    is_cake_causal_conv1d_train_supported,
)
from flashinfer.cake_causal_conv1d_train_backend import cake_jit as loader
from flashinfer.testing import bench_gpu_time

WIDTH = 4
# The performance matrix: P1 (C = 512 over the packed token counts) and P2 (S = 250340 over the channel counts).
P1_TOKENS = (239718, 245028, 249866, 250340, 258574, 259042, 259584, 261393)
P1_CHANNELS = 512
P2_TOKENS = 250340
P2_CHANNELS = (128, 256, 1024, 2048, 4096, 8192)
# Four unequal documents: floor(fraction x S) tokens each, nudged off a multiple of 64, the rest last.
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 0
# The kernels resident on another stream while the backward is timed (``--side-kernel``): CTAs x threads.
SIDE_KERNELS = {"one-thread": (1, 1), "nccl-like": (8, 512)}
# About 5 ms at 2 GHz: resident for the whole timed call.
SIDE_KERNEL_CYCLES = 10_000_000
SIDE_KERNEL_WARMUPS = 3
SIDE_KERNEL_SAMPLES = 7

# The ``nccl-like`` side kernel: every thread spins on the SM clock for ``cycles``.
_SPIN_DECLARATION = (
    "#include <cstdint>\nvoid spin(int64_t ctas, int64_t threads, int64_t cycles);"
)
_SPIN_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void spin_kernel(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

void spin(int64_t ctas, int64_t threads, int64_t cycles) {
  spin_kernel<<<static_cast<unsigned>(ctas), static_cast<unsigned>(threads), 0,
                at::cuda::getCurrentCUDAStream()>>>(cycles);
}
"""


def perf_rows() -> list[tuple[int, int]]:
    rows = [(tokens, P1_CHANNELS) for tokens in P1_TOKENS]
    rows += [(P2_TOKENS, channels) for channels in P2_CHANNELS]
    return rows


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


def _median_ms(samples) -> float:
    return float(statistics.median(list(samples)))


def _time(fn: Callable[[], object]) -> float:
    """Median GPU time (ms) of ``fn`` over its kernels, cold L2, CUPTI."""

    times = bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=True)
    return _median_ms(times)


def side_kernel_launcher(name: str) -> Callable[[], None]:
    """A launcher of the resident side kernel ``name`` (:data:`SIDE_KERNELS`) on the current stream."""

    ctas, threads = SIDE_KERNELS[name]
    if (ctas, threads) == (1, 1):
        return lambda: torch.cuda._sleep(SIDE_KERNEL_CYCLES)
    from torch.utils.cpp_extension import load_inline

    module = load_inline(
        name="cake_causal_conv1d_bench_spin",
        cpp_sources=_SPIN_DECLARATION,
        cuda_sources=_SPIN_SOURCE,
        functions=["spin"],
    )
    return lambda: module.spin(ctas, threads, SIDE_KERNEL_CYCLES)


def _latency_beside(
    fn: Callable[[], object], side_kernel: Callable[[], None], device: torch.device
) -> tuple[float, float]:
    """``(alone_ms, beside_ms)``: the median wall latency of ``fn`` on the current stream (CUDA events) over
    :data:`SIDE_KERNEL_SAMPLES` calls after :data:`SIDE_KERNEL_WARMUPS`, alone and with ``side_kernel``
    resident on another stream (launched there first, then given time to become resident)."""

    side = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(beside: bool) -> float:
        torch.cuda.synchronize(device)
        if beside:
            with torch.cuda.stream(side):
                side_kernel()
        torch.cuda._sleep(
            1_000_000
        )  # let the side kernel become resident before fn's launch
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(SIDE_KERNEL_WARMUPS):
        timed(False)
    alone = _median_ms(timed(False) for _ in range(SIDE_KERNEL_SAMPLES))
    beside = _median_ms(timed(True) for _ in range(SIDE_KERNEL_SAMPLES))
    return alone, beside


def _doc_ids(tokens: int, cu: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(
        torch.arange(tokens, device=cu.device), cu[1:].long(), right=True
    )


def _shift_down(v: torch.Tensor, ids: torch.Tensor, d: int) -> torch.Tensor:
    out = torch.zeros_like(v)
    if d == 0:
        return v
    if d < v.shape[0]:
        same = (ids[d:] == ids[:-d]).unsqueeze(1)
        out[d:] = torch.where(same, v[:-d], 0.0)
    return out


def _causal_conv1d_silu_fp32(
    x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """The document-local formula in FP32 with one BF16 rounding (autograd gives the FP32 backward)."""

    xf = x.float()
    pre = torch.zeros_like(xf)
    for k in range(WIDTH):
        pre = pre + w[:, k] * _shift_down(xf, ids, WIDTH - 1 - k)
    return (pre * torch.sigmoid(pre)).to(torch.bfloat16)


class Comparator:
    name = "base"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def make(self, x, w, cu):
        """Return ``(fwd, bwd)``: ``fwd()`` returns ``y`` (keeping a graph when needed), ``bwd(dy)`` returns
        ``(dx, dweight)``."""

        raise NotImplementedError


class Cake(Comparator):
    name = "cake"

    def make(self, x, w, cu):
        tokens, channels = x.shape
        y = torch.empty_like(x)
        dx = torch.empty_like(x)
        dw = torch.empty(channels, WIDTH, dtype=torch.float32, device=x.device)
        workspace = cake_causal_conv1d_train_backward_workspace(
            tokens, channels, x.device
        )

        def fwd():
            cake_causal_conv1d_train_forward(x, w, cu, out=y)
            return y

        def bwd(dy):
            cake_causal_conv1d_train_backward(
                x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw
            )
            return dx, dw

        return fwd, bwd


class _AutogradComparator(Comparator):
    def _forward(self, x, w, cu):
        raise NotImplementedError

    def make(self, x, w, cu):
        xl = x.detach().clone().requires_grad_(True)
        wl = w.detach().clone().requires_grad_(True)
        state: dict = {}

        def fwd():
            xl.grad = wl.grad = None
            y = self._forward(xl, wl, cu)
            state["y"] = y
            return y

        def bwd(dy):
            xl.grad = wl.grad = None
            torch.autograd.backward(
                [state["y"]], [dy.view_as(state["y"])], retain_graph=True
            )
            return xl.grad, wl.grad

        return fwd, bwd


class EagerFp32(_AutogradComparator):
    name = "eager_fp32"

    def make(self, x, w, cu):
        self._ids = _doc_ids(x.shape[0], cu)
        return super().make(x, w, cu)

    def _forward(self, x, w, cu):
        return _causal_conv1d_silu_fp32(x, w, self._ids)


class FlashLinearAttention(_AutogradComparator):
    """flash-linear-attention's Triton ``causal_conv1d`` with ``activation="silu"`` over ``cu_seqlens``."""

    name = "fla"

    def available(self):
        try:
            import fla
            from fla.modules.conv import causal_conv1d  # noqa: F401

            return True, f"flash-linear-attention {getattr(fla, '__version__', '?')}"
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, repr(exc)

    def _forward(self, x, w, cu):
        from fla.modules.conv import causal_conv1d

        # FLA takes [B, T, D] packed activations with cu_seqlens and a [D, W] weight.
        result = causal_conv1d(
            x.unsqueeze(0), w, bias=None, activation="silu", cu_seqlens=cu
        )
        y = result[0] if isinstance(result, tuple) else result
        return y


COMPARATORS = (Cake(), FlashLinearAttention(), EagerFp32())


def _bytes(mode: str, tokens: int, channels: int, *, partial_rows: int) -> int:
    # Forward: x read, y written, w read.  Backward: x and dy read (twice for x: the recompute and the dweight
    # taps are one pass), dx written, w read, dw written, and the FP32 [rows, C, 4] partials written then read.
    fwd = 4 * tokens * channels + 16 * channels
    bwd = 6 * tokens * channels + 32 * channels + 32 * channels * partial_rows
    return {"fwd": fwd, "bwd": bwd, "step": fwd + bwd}[mode]


def _selected_program(arch: str, tokens: int, channels: int) -> str:
    """The backward program the registry selects for the row (by channel count; ``tokens`` never enters)."""

    return loader.selected_modules(arch, "bwd", C=channels, S=tokens)[0]


def _partial_rows(program: str, channels: int, *, device_index: int) -> int:
    """The FP32 partial rows of ``program``'s own grid (the public workspace is sized for the largest grid)."""

    record = loader.MODULES[program]
    return loader.workspace_layout(record, channels, device_index=device_index)["rows"]


def bench_row(
    tokens: int, channels: int, *, comparators, device, side_kernel=None
) -> dict:
    generator = torch.Generator(device=device).manual_seed(SEED)
    lens = doc_lengths(tokens)
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    x = torch.randn(
        tokens, channels, device=device, generator=generator, dtype=torch.bfloat16
    )
    # N(0, 0.25): variance 0.25, standard deviation 0.5.
    w = torch.randn(channels, WIDTH, device=device, generator=generator) * 0.5
    dy = torch.randn(
        tokens, channels, device=device, generator=generator, dtype=torch.bfloat16
    )
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    program = _selected_program(arch, tokens, channels)
    partial_rows = _partial_rows(program, channels, device_index=device.index)
    result = {
        "tokens": tokens,
        "channels": channels,
        "documents": list(lens),
        "program": program,
        "partial_rows": partial_rows,
        "comparators": {},
    }
    for comparator in comparators:
        entry: dict = {}
        try:
            fwd, bwd = comparator.make(x, w, cu)
            fwd()
            bwd(dy)
            torch.cuda.synchronize()
            entry["fwd_ms"] = _time(fwd)
            fwd()
            torch.cuda.synchronize()
            entry["bwd_ms"] = _time(lambda: bwd(dy))
            entry["step_ms"] = entry["fwd_ms"] + entry["bwd_ms"]
            for mode in ("fwd", "bwd", "step"):
                nbytes = _bytes(mode, tokens, channels, partial_rows=partial_rows)
                entry[f"{mode}_gbps"] = nbytes / (entry[f"{mode}_ms"] * 1e-3) / 1e9
            if side_kernel is not None:
                alone, beside = _latency_beside(lambda: bwd(dy), side_kernel, device)
                entry["bwd_alone_ms"] = alone
                entry["bwd_beside_ms"] = beside
                entry["bwd_beside_ratio"] = beside / alone
        except Exception as exc:  # keep the other comparators' rows
            entry["error"] = repr(exc)[:200]
        result["comparators"][comparator.name] = entry
        torch.cuda.empty_cache()
    return result


def _format(result: dict, names) -> list[str]:
    lines = []
    for mode in ("fwd", "bwd", "step"):
        cells = []
        for name in names:
            entry = result["comparators"].get(name, {})
            value = entry.get(f"{mode}_ms")
            cells.append("-" if value is None else f"{value * 1e3:9.2f}")
        line = (
            f"S={result['tokens']:6d} C={result['channels']:5d} {mode:4s} | "
            + " | ".join(cells)
        )
        gbps = result["comparators"].get("cake", {}).get(f"{mode}_gbps")
        if gbps is not None:
            line += f" | cake {gbps:8.0f} GB/s"
        lines.append(line)
    for name in names:
        entry = result["comparators"].get(name, {})
        if "bwd_beside_ms" in entry:
            lines.append(
                f"S={result['tokens']:6d} C={result['channels']:5d} bwd beside the side kernel | {name}: "
                f"alone {entry['bwd_alone_ms'] * 1e3:9.2f} us, beside {entry['bwd_beside_ms'] * 1e3:9.2f} us, "
                f"{entry['bwd_beside_ratio']:.2f}x"
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
        help="packed token counts S; with --channels the rows are their product (default: the perf matrix)",
    )
    parser.add_argument(
        "--channels", type=int, nargs="*", default=None, help="channel counts C"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], help="comparator names to skip"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="write all rows to this JSON file"
    )
    parser.add_argument(
        "--side-kernel",
        choices=sorted(SIDE_KERNELS),
        default=None,
        help="also time the backward's wall latency alone and beside this kernel resident on another stream",
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda", torch.cuda.current_device())
    if not is_cake_causal_conv1d_train_supported(device):
        raise SystemExit(
            "cake_causal_conv1d_train has no exported route on this device"
        )
    comparators = []
    for comparator in COMPARATORS:
        if comparator.name in args.skip:
            continue
        ok, note = comparator.available()
        print(f"{comparator.name}: {'available' if ok else 'unavailable'} {note}")
        if ok:
            comparators.append(comparator)
    names = [c.name for c in comparators]
    print("columns (us): " + " | ".join(names))
    if args.tokens is None and args.channels is None:
        rows = perf_rows()
    else:
        tokens = args.tokens or [P2_TOKENS]
        channels = args.channels or [P1_CHANNELS]
        rows = [(s, c) for s in tokens for c in channels]
    side_kernel = (
        None if args.side_kernel is None else side_kernel_launcher(args.side_kernel)
    )
    results = []
    for tokens, channels in rows:
        result = bench_row(
            tokens,
            channels,
            comparators=comparators,
            device=device,
            side_kernel=side_kernel,
        )
        results.append(result)
        for line in _format(result, names):
            print(line, flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "torch": torch.__version__,
                    "side_kernel": args.side_kernel,
                    "rows": results,
                },
                handle,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
