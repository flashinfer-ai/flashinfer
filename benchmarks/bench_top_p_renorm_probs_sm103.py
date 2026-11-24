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
"""

"""Benchmark ``top_p_renorm_probs`` on compute capability 10.3 (B300 / GB300): the SM103 cluster
kernel against the stock AIR top-p path, on the same inputs, back to back.

Kernel time (``--sections sweep,scan``): ``flashinfer.testing.bench_gpu_time`` with CUPTI
(cupti-python >= 13) and a cold L2 (an L2 flush before every iteration). A sample is the GPU span
of one call, from its first kernel's start to its last kernel's end; the median is reported.

- ``--modes graph``: the call is captured in a CUDA graph and replayed, so the span is GPU time
  only. This is the kernel-level comparison.
- ``--modes eager``: the call is launched from Python after a device synchronize. For a call that
  launches several kernels (the AIR path launches five) the span also holds the host launch gaps
  between them, which a serving loop that runs ahead of the GPU hides.

The two paths:

- ``specialized``: the SM103 kernel. Through ``top_p_renorm_probs`` for a ``(batch_size,
  vocab_size)`` on its allowlist (``via=api``, the dispatch is checked); through the private
  module for any other shape (``via=module``), so ``scan`` can time batch sizes it does not serve.
- ``stock``: ``top_p_renorm_probs`` with ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``.

Inputs (``--dists``): float32 rows that sum to 1, ``top_p`` a per-row float32 tensor.

- ``topk_truncated``: 20 values (softmax of N(0, 16) logits) at random positions, exact zeros
  elsewhere: the rows of a sampler that applies ``top_k_renorm_probs`` (top-k 20) before
  ``top_p_renorm_probs``. The AIR histogram serializes its atomics on the zero bucket here.
- ``peaked``: one token holding 0.97 over a log-normal bulk, no zeros (nucleus of one token).
- ``softmax_randn2``: softmax of N(0, 4) logits: flat, the nucleus spans thousands of entries.
- ``rand_pow8``: ``rand ** 8``, normalised: flat, no entry reaches the kernel's 2^-8 fast path.
- ``uniform``: ``torch.rand``, normalised (the input of ``flashinfer_benchmark.py``'s routine).

Host time (``--sections host``): the host cost of the ``top_p_renorm_probs`` calls the SM103 kernel
never serves (scalar ``top_p``, a shape off the allowlist, any other GPU): per call, the
specialized guard alone, and the whole call with and without the guard (the stock path as it was
before the guard existed), timed while a sleep kernel keeps the GPU busy so that launches only
enqueue.

Examples::

    python benchmarks/bench_top_p_renorm_probs_sm103.py
    python benchmarks/bench_top_p_renorm_probs_sm103.py --sections scan --scan-batches 1-72
    python benchmarks/bench_top_p_renorm_probs_sm103.py --sections host
"""

import argparse
import json
import os
import statistics
import time
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

import flashinfer
import flashinfer.sampling as sampling
from flashinfer import _top_p_renorm_specialized as specialized
from flashinfer.testing import bench_gpu_time

KILL_SWITCH = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Default (batch_size, vocab_size) shapes for the sweep, whether allowlisted or not.
SERVING_SHAPES: Tuple[Tuple[int, int], ...] = (
    (5, 154880),
    (6, 154880),
    (12, 154880),
    (18, 154880),
    (24, 154880),
    (30, 154880),
    (36, 154880),
    (54, 154880),
    (4, 248320),
    (7, 248320),
    (12, 248320),
    (16, 248320),
    (20, 248320),
    (24, 248320),
    (28, 248320),
    (32, 248320),
    (68, 248320),
)


def _normalise(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True)


def _topk_truncated(b: int, v: int, g: torch.Generator, dev: str) -> torch.Tensor:
    vals = torch.softmax(torch.randn(b, 20, generator=g, device=dev) * 4.0, dim=-1)
    pos = torch.rand(b, v, generator=g, device=dev).topk(20, dim=-1).indices
    return torch.zeros(b, v, device=dev).scatter_(1, pos, vals)


def _peaked(b: int, v: int, g: torch.Generator, dev: str) -> torch.Tensor:
    probs = _normalise(torch.exp(torch.randn(b, v, generator=g, device=dev) * 3.0))
    probs *= 0.03
    top = torch.randint(0, v, (b, 1), generator=g, device=dev)
    return probs.scatter_add_(1, top, torch.full((b, 1), 0.97, device=dev))


DISTS: Dict[str, Callable[[int, int, torch.Generator, str], torch.Tensor]] = {
    "topk_truncated": _topk_truncated,
    "peaked": _peaked,
    "softmax_randn2": lambda b, v, g, dev: torch.softmax(
        torch.randn(b, v, generator=g, device=dev) * 2.0, dim=-1
    ),
    "rand_pow8": lambda b, v, g, dev: _normalise(
        torch.rand(b, v, generator=g, device=dev) ** 8
    ),
    "uniform": lambda b, v, g, dev: _normalise(
        torch.rand(b, v, generator=g, device=dev)
    ),
}


def make_inputs(
    dist: str, batch_size: int, vocab_size: int, top_p: float, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    probs = DISTS[dist](batch_size, vocab_size, g, "cuda").float().contiguous()
    return probs, torch.full((batch_size,), top_p, device="cuda")


@contextmanager
def kill_switch(on: bool) -> Iterator[None]:
    old = os.environ.get(KILL_SWITCH)
    if on:
        os.environ[KILL_SWITCH] = "1"
    else:
        os.environ.pop(KILL_SWITCH, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(KILL_SWITCH, None)
        else:
            os.environ[KILL_SWITCH] = old


def _dispatch_count() -> int:
    return int(specialized._top_p_renorm_probs_specialized_stats()["dispatch_count"])


def _cupti_timer() -> str:
    try:
        from importlib.metadata import version

        from cupti import cupti  # noqa: F401

        if int(version("cupti-python").split(".")[0]) >= 13:
            return "cupti"
    except Exception:  # noqa: BLE001 - only names the timer bench_gpu_time falls back to
        pass
    return "cuda_event (no cupti-python >= 13)"


def time_us(
    fn: Callable[..., Any],
    args: Sequence[Any],
    graph: bool,
    repeat_ms: float = 100.0,
    warmup_ms: float = 25.0,
) -> Dict[str, float]:
    """Median / p10 / p90 GPU span of one call, in microseconds (CUPTI, cold L2)."""
    times = bench_gpu_time(
        fn,
        enable_cupti=True,
        use_cuda_graph=graph,
        cold_l2_cache=True,
        input_args=tuple(args),
        dry_run_time_ms=warmup_ms,
        repeat_time_ms=repeat_ms,
    )
    us = sorted(1000.0 * t for t in times)
    return {
        "median_us": statistics.median(us),
        "p10_us": us[int(0.1 * (len(us) - 1))],
        "p90_us": us[int(0.9 * (len(us) - 1))],
        "iters": len(us),
    }


def _specialized_fn(
    batch_size: int, vocab_size: int, probs: torch.Tensor
) -> Tuple[Callable[..., Any], str]:
    """The SM103 kernel for this shape: through the API when the shape is allowlisted (the
    dispatch is checked), else through the private module."""
    if (batch_size, vocab_size) in specialized._load_allowlist():
        return sampling.top_p_renorm_probs, "api"
    module = specialized._get_module()
    if module is None:
        raise RuntimeError(
            f"specialized module unavailable: {specialized._module_error}"
        )
    out = torch.empty_like(probs)

    def run(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        module.top_p_renorm_probs_specialized(p, t, out)
        return out

    return run, "module"


def compare_paths(
    batch_size: int,
    vocab_size: int,
    dist: str,
    modes: Sequence[str],
    top_p: float = 0.95,
    seed: int = 0,
    repeat_ms: float = 100.0,
    warmup_ms: float = 25.0,
) -> List[Dict[str, Any]]:
    """One row per timer mode: the specialized kernel and the stock path on the same inputs."""
    probs, tp = make_inputs(dist, batch_size, vocab_size, top_p, seed)
    spec_fn, via = _specialized_fn(batch_size, vocab_size, probs)
    with kill_switch(False):
        before = _dispatch_count()
        out_spec = spec_fn(probs, tp).clone()
        dispatched = via == "module" or _dispatch_count() == before + 1
    with kill_switch(True):
        out_stock = sampling.top_p_renorm_probs(probs, tp)
    torch.cuda.synchronize()
    agreement = {
        "max_abs_vs_stock": float((out_spec - out_stock).abs().max()),
        "kept_diff_vs_stock": int(((out_spec > 0) != (out_stock > 0)).sum()),
        "kept_per_row": float((out_spec > 0).sum(dim=-1).float().mean()),
        "spec_nan": bool(torch.isnan(out_spec).any()),
    }
    rows = []
    for mode in modes:
        graph = mode == "graph"
        with kill_switch(False):
            spec = time_us(spec_fn, (probs, tp), graph, repeat_ms, warmup_ms)
        with kill_switch(True):
            stock = time_us(
                sampling.top_p_renorm_probs, (probs, tp), graph, repeat_ms, warmup_ms
            )
        rows.append(
            {
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "dist": dist,
                "top_p": top_p,
                "mode": mode,
                "via": via,
                "dispatched": dispatched,
                "allowlisted": (batch_size, vocab_size)
                in specialized._load_allowlist(),
                "spec_us": spec["median_us"],
                "spec_p10_us": spec["p10_us"],
                "spec_p90_us": spec["p90_us"],
                "stock_us": stock["median_us"],
                "stock_p10_us": stock["p10_us"],
                "stock_p90_us": stock["p90_us"],
                "speedup": stock["median_us"] / spec["median_us"],
                **agreement,
            }
        )
    return rows


def _emit(row: Dict[str, Any], json_out: Optional[Any]) -> None:
    if json_out is not None:
        json_out.write(json.dumps(row) + "\n")
        json_out.flush()


def run_sweep(
    shapes: Sequence[Tuple[int, int]] = SERVING_SHAPES,
    dists: Sequence[str] = tuple(DISTS),
    modes: Sequence[str] = ("graph", "eager"),
    top_p: float = 0.95,
    repeat_ms: float = 100.0,
    warmup_ms: float = 25.0,
    json_out: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    print(
        f"{'B':>4} {'V':>7} {'dist':>15} {'mode':>5} {'via':>6} {'spec us':>9} "
        f"{'stock us':>9} {'speedup':>8} {'kept/row':>9} {'max|d|':>8} {'kept!=':>6}",
        flush=True,
    )
    rows = []
    for batch_size, vocab_size in shapes:
        for dist in dists:
            for row in compare_paths(
                batch_size, vocab_size, dist, modes, top_p, 0, repeat_ms, warmup_ms
            ):
                row["section"] = "sweep"
                print(
                    f"{batch_size:>4} {vocab_size:>7} {dist:>15} {row['mode']:>5} "
                    f"{row['via'] if row['dispatched'] else 'NOT':>6} {row['spec_us']:>9.2f} "
                    f"{row['stock_us']:>9.2f} {row['speedup']:>7.2f}x "
                    f"{row['kept_per_row']:>9.1f} {row['max_abs_vs_stock']:>8.1e} "
                    f"{row['kept_diff_vs_stock']:>6}",
                    flush=True,
                )
                _emit(row, json_out)
                rows.append(row)
    return rows


def run_batch_scan(
    batches: Sequence[int] = tuple(range(1, 73)),
    vocabs: Sequence[int] = (154880, 248320),
    dists: Sequence[str] = ("peaked", "softmax_randn2", "rand_pow8"),
    modes: Sequence[str] = ("graph",),
    top_p: float = 0.95,
    repeat_ms: float = 50.0,
    warmup_ms: float = 10.0,
    json_out: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Every batch size of each vocab (allowlisted or not): where the kernel is not slower."""
    print(
        f"{'V':>7} {'B':>4} {'mode':>5} "
        + " ".join(f"{d + ' spec/stock us':>34}" for d in dists),
        flush=True,
    )
    rows = []
    for vocab_size in vocabs:
        for batch_size in batches:
            per_mode: Dict[str, List[str]] = {m: [] for m in modes}
            for dist in dists:
                for row in compare_paths(
                    batch_size, vocab_size, dist, modes, top_p, 0, repeat_ms, warmup_ms
                ):
                    row["section"] = "scan"
                    per_mode[row["mode"]].append(
                        f"{row['spec_us']:>9.2f} / {row['stock_us']:>8.2f} "
                        f"({row['speedup']:>5.2f}x)"
                    )
                    _emit(row, json_out)
                    rows.append(row)
            for mode, cells in per_mode.items():
                print(
                    f"{vocab_size:>7} {batch_size:>4} {mode:>5} "
                    + " ".join(f"{c:>34}" for c in cells),
                    flush=True,
                )
    return rows


def _host_us_per_call(
    fn: Callable[[], Any], calls: int, reps: int, stall_gpu: bool
) -> List[float]:
    """Host time per call (us) of ``calls`` back-to-back calls, one sample per rep."""
    out = []
    for _ in range(reps):
        if stall_gpu:
            # keep the GPU busy so that the calls' launches only enqueue (no back pressure)
            torch.cuda._sleep(50_000_000)
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        out.append((t1 - t0) / calls * 1e6)
    return out


def _host_cases() -> List[Tuple[str, torch.Tensor, Any]]:
    """Calls the SM103 kernel never serves (and, on SM103, one it serves, for reference)."""
    g = torch.Generator(device="cuda").manual_seed(0)
    cases = []
    for label, (b, v), scalar in (
        ("scalar_top_p", (7, 248320), True),
        ("tensor_top_p_unlisted_shape", (8, 248320), False),
        ("tensor_top_p_allowlisted_shape", (7, 248320), False),
    ):
        probs = _peaked(b, v, g, "cuda")
        top_p: Any = 0.95 if scalar else torch.full((b,), 0.95, device="cuda")
        cases.append((label, probs, top_p))
    return cases


def run_host_overhead(
    calls: int = 64, reps: int = 30, json_out: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Host cost of the specialized guard on ``top_p_renorm_probs`` calls (any GPU)."""
    on_sm103 = torch.cuda.get_device_capability() == (10, 3)
    guard = specialized.top_p_renorm_probs_specialized
    real_guard = sampling._top_p_renorm_probs_specialized
    rows = []
    print(
        f"{'case':>31} {'served':>6} {'guard ns':>9} {'call us':>8} "
        f"{'no-guard us':>11} {'delta us':>8}",
        flush=True,
    )
    for label, probs, top_p in _host_cases():
        served = (
            on_sm103
            and isinstance(top_p, torch.Tensor)
            and tuple(probs.shape) in specialized._load_allowlist()
        )

        def call(probs=probs, top_p=top_p):
            return sampling.top_p_renorm_probs(probs, top_p)

        def call_guard(probs=probs, top_p=top_p):
            return guard(probs, top_p, False)

        # warm every path (JIT builds, allocator pools, one-time markers)
        for _ in range(3):
            call()
            sampling._top_p_renorm_probs_specialized = lambda p, t, d: None
            try:
                call()
            finally:
                sampling._top_p_renorm_probs_specialized = real_guard
        torch.cuda.synchronize()
        guard_ns = None
        if not served:
            n = 20000
            samples = []
            for _ in range(7):
                t0 = time.perf_counter()
                for _ in range(n):
                    call_guard()
                samples.append((time.perf_counter() - t0) / n * 1e9)
            guard_ns = statistics.median(samples)
        with_guard, without_guard = [], []
        for _ in range(reps):
            with_guard += _host_us_per_call(call, calls, 1, True)
            sampling._top_p_renorm_probs_specialized = lambda p, t, d: None
            try:
                without_guard += _host_us_per_call(call, calls, 1, True)
            finally:
                sampling._top_p_renorm_probs_specialized = real_guard
        row = {
            "section": "host",
            "case": label,
            "shape": list(probs.shape),
            "served": served,
            "guard_ns": guard_ns,
            "call_us": statistics.median(with_guard),
            "call_no_guard_us": statistics.median(without_guard),
        }
        row["delta_us"] = row["call_us"] - row["call_no_guard_us"]
        guard_cell = "-" if guard_ns is None else f"{guard_ns:.0f}"
        print(
            f"{label:>31} {str(served):>6} {guard_cell:>9} {row['call_us']:>8.2f} "
            f"{row['call_no_guard_us']:>11.2f} {row['delta_us']:>8.2f}",
            flush=True,
        )
        _emit(row, json_out)
        rows.append(row)
    return rows


def describe_environment() -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    env = {
        "section": "env",
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "timer": _cupti_timer(),
        "allowlist": sorted(specialized._load_allowlist()),
    }
    print(" ".join(f"{k}={v}" for k, v in env.items() if k != "allowlist"), flush=True)
    return env


class _Lines:
    """``json_out`` that prints each JSON row as a ``TOPP_BENCH`` line on stdout."""

    def write(self, s: str) -> None:
        for line in s.splitlines():
            print(f"TOPP_BENCH {line}", flush=True)

    def flush(self) -> None:
        pass


def run_default_suite() -> None:
    """Every section with its default settings, rows also as ``TOPP_BENCH {json}`` lines on
    stdout (callable without arguments, e.g. by a test runner on a remote GPU node): the host
    overhead on any GPU, then the sweep and the batch scan on compute capability 10.3."""
    out = _Lines()
    _emit(describe_environment(), out)
    run_host_overhead(json_out=out)
    if torch.cuda.get_device_capability() == (10, 3):
        run_sweep(json_out=out)
        run_batch_scan(json_out=out)


def _parse_batches(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sections", default="sweep", help="comma list of sweep,scan,host")
    ap.add_argument(
        "--shapes",
        default=",".join(f"{b}x{v}" for b, v in SERVING_SHAPES),
        help="sweep shapes, BxV comma list",
    )
    ap.add_argument("--dists", default=",".join(DISTS), help="sweep input rows")
    ap.add_argument("--modes", default="graph,eager", help="sweep timer modes")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--scan-batches", default="1-72")
    ap.add_argument("--scan-vocabs", default="154880,248320")
    ap.add_argument("--scan-dists", default="peaked,softmax_randn2,rand_pow8")
    ap.add_argument("--scan-modes", default="graph")
    ap.add_argument("--repeat-ms", type=float, default=100.0)
    ap.add_argument("--warmup-ms", type=float, default=25.0)
    ap.add_argument("--json", help="also write every row as a JSON line to this file")
    args = ap.parse_args()

    sections = set(args.sections.split(","))
    with ExitStack() as stack:
        json_out = stack.enter_context(open(args.json, "w")) if args.json else None
        _emit(describe_environment(), json_out)
        on_sm103 = torch.cuda.get_device_capability() == (10, 3)
        if "host" in sections:
            run_host_overhead(json_out=json_out)
        if (sections & {"sweep", "scan"}) and not on_sm103:
            print(
                "sweep / scan: the specialized kernel runs on compute capability 10.3 only"
            )
            return
        if "sweep" in sections:
            shapes = [
                (int(b), int(v))
                for b, v in (s.split("x") for s in args.shapes.split(","))
            ]
            run_sweep(
                shapes,
                args.dists.split(","),
                args.modes.split(","),
                args.top_p,
                args.repeat_ms,
                args.warmup_ms,
                json_out,
            )
        if "scan" in sections:
            run_batch_scan(
                _parse_batches(args.scan_batches),
                [int(v) for v in args.scan_vocabs.split(",")],
                args.scan_dists.split(","),
                args.scan_modes.split(","),
                args.top_p,
                json_out=json_out,
            )


if __name__ == "__main__":
    main()
