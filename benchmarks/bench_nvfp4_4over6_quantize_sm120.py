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

NVFP4 4over6 activation quantization on SM12x: nvfp4_4over6_quantize_sm120 / nvfp4_4over6_dynamic_quantize_sm120
against fp4_quantize with the same recipe (CUDA and CuTe-DSL backends). Each op is captured CALLS times in a
CUDA graph, so programmatic dependent launch overlaps consecutive calls as it would in a serving graph; the
time per call is the median over REPS replays. --producer precedes every call with x.copy_(src) (x written
front to back, as by a norm or activation kernel) and subtracts the copy's own time.

    python benchmarks/bench_nvfp4_4over6_quantize_sm120.py [--m 1 16 ...] [--k 2048 ...] [--producer]
"""

import argparse
import statistics

import torch

from flashinfer import (
    NVFP44Over6Config,
    fp4_quantize,
    nvfp4_4over6_dynamic_quantize_sm120,
    nvfp4_4over6_quantize_sm120,
)

CONFIG = NVFP44Over6Config(e4m3_max=256, err_mode="MSE")


def time_us(fn, calls: int, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / calls)
    return statistics.median(times)


def bench_shape(m: int, k: int, args, has_cute: bool) -> str:
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device).bfloat16()
    src = x.clone()
    gs = (6.0 * CONFIG.e4m3_max / x.abs().amax().float()).reshape(1)
    inv = torch.empty(1, device=device)
    alpha = torch.empty(1, device=device)
    wgs = torch.ones(1, device=device)

    def fi_dynamic():
        torch.div(
            6.0 * CONFIG.e4m3_max, x.abs().amax().float().clamp(min=1e-12), out=inv
        )
        return fp4_quantize(x, inv, 16, False, True, nvfp4_4over6=CONFIG)

    ops = {
        "fi": lambda: fp4_quantize(x, gs, 16, False, True, nvfp4_4over6=CONFIG),
        "cute": (
            lambda: fp4_quantize(
                x, gs, 16, False, True, backend="cute-dsl", nvfp4_4over6=CONFIG
            )
        )
        if has_cute
        else None,
        "new": lambda: nvfp4_4over6_quantize_sm120(x, gs, CONFIG),
        "fi_dyn": fi_dynamic,
        "new_dyn": lambda: nvfp4_4over6_dynamic_quantize_sm120(
            x, wgs, CONFIG, global_scale_out=inv, alpha_out=alpha
        ),
    }
    base = (
        time_us(lambda: x.copy_(src), args.calls, args.reps) if args.producer else 0.0
    )
    t = {}
    for name, fn in ops.items():
        if fn is None:
            t[name] = float("nan")
            continue
        run = (lambda fn=fn: (x.copy_(src), fn())) if args.producer else fn
        try:
            t[name] = time_us(run, args.calls, args.reps) - base
        except Exception as error:  # e.g. a backend without this recipe
            print(f"# {name} failed at M={m} K={k}: {type(error).__name__}: {error}")
            t[name] = float("nan")
    return (
        f"{m:>6} {k:>6} | {t['fi']:>12.2f} {t['cute']:>9.2f} {t['new']:>8.2f} {t['fi'] / t['new']:>6.2f}x | "
        f"{t['fi_dyn']:>10.2f} {t['new_dyn']:>9.2f} {t['fi_dyn'] / t['new_dyn']:>6.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m", type=int, nargs="+", default=[1, 16, 64, 256, 1024, 4096, 16384]
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[2048, 4096, 5120, 12288, 17408]
    )
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--producer", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    print(
        f"# {torch.cuda.get_device_name(device)}, recipe {CONFIG!r}, us per call"
        f"{', producer x.copy_(src) subtracted' if args.producer else ''}"
    )
    print(
        f"{'M':>6} {'K':>6} | {'fp4_quantize':>12} {'cute-dsl':>9} {'sm120':>8} {'speedup':>7} | "
        f"{'amax+fp4_q':>10} {'dyn sm120':>9} {'speedup':>7}"
    )
    try:
        from flashinfer.quantization import nvfp4_quantize_cute_dsl  # noqa: F401

        has_cute = True
    except ImportError:
        has_cute = False
    for k in args.k:
        for m in args.m:
            print(bench_shape(m, k, args, has_cute))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
