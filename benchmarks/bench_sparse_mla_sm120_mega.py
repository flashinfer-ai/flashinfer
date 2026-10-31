"""Mega fusion gate: compiled unfused transforms + core, timed end to end.

Run from the repository with PYTHONPATH=. Reports each stage and the sequential
CUDA graph pipeline. No Python timing, allocations, or fixture packing is timed.
"""

import argparse
from functools import partial
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import torch
from tests.attention.mega_mla_utils import fixture, fused_call


def capture(fn, calls):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(g):
        start.record()
        for _ in range(calls):
            fn()
        end.record()
    return g, start, end


def measure(g, calls, flush=None):
    if flush is not None:
        flush.zero_()
    graph, start, end = g
    graph.replay()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / calls


def use_baseline_module(f, module):
    """Compare an untouched build in-process; the execution-plan ABI must match."""
    from flashinfer.mla._sparse_mla_sm120._prepared import _wrapper_scratch

    f["baseline"]()
    prepared = next(iter(f["wrapper"]._prepared_calls.values()))
    prepared, mid, mlse = _wrapper_scratch(
        f["wrapper"], prepared, None, None, f["q"].device, False
    )
    execute = (
        module.dsv4_nvfp4_execute_attention
        if f["cache_format"] == "nvfp4"
        else module.execute_attention
    )

    def core():
        execute(
            prepared.plan,
            f["rotated"],
            f["caches"][0],
            f["indices"][0],
            mid,
            mlse,
            f["context"],
            prepared.lse,
            512**-0.5,
            f["lengths"][0],
            f["sink"],
            f["caches"][1],
            f["indices"][1],
            f["lengths"][1],
            math.log(2),
        )
        return prepared.lse

    def baseline():
        f["prologue"]()
        lse = core()
        f["post"]()
        return f["out"], f["scales"], lse

    f["core"], f["baseline"] = core, baseline


def previous_mega_call(f, module):
    from flashinfer.mla._sparse_mla_sm120._prepared import _wrapper_scratch

    fused_call(f)
    p = next(iter(f["wrapper"]._prepared_calls.values()))
    p, mid, mlse = _wrapper_scratch(f["wrapper"], p, None, None, f["q"].device, False)
    execute = (
        module.dsv4_nvfp4_execute_mega_attention
        if f["cache_format"] == "nvfp4"
        else module.execute_mega_attention
    )

    def run():
        execute(
            p.plan,
            f["q"],
            f["caches"][0],
            f["indices"][0],
            mid,
            mlse,
            f["q"],
            p.lse,
            512**-0.5,
            f["lengths"][0],
            f["sink"],
            f["caches"][1],
            f["indices"][1],
            f["lengths"][1],
            math.log(2),
            f["positions"],
            f["cs"],
            f["out"].view(torch.uint8),
            f["scales"],
        )
        return p.lse

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,4,16,64,128,512,1024")
    ap.add_argument("--heads", default="16,64")
    ap.add_argument("--formats", default="fp8,nvfp4")
    ap.add_argument("--pool-tokens", type=int, default=1048576)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cases", help="JSON workload list with fixture arguments")
    ap.add_argument("--mega-baseline-module", help="Previously committed mega .so")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--warmup-seconds", type=float, default=0.0)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument(
        "--baseline-module", help="Untouched .so with matching execution-plan ABI"
    )
    a = ap.parse_args()
    if a.mega_baseline_module and not a.fused:
        ap.error("--mega-baseline-module requires --fused")
    if a.calls < 1 or a.samples < 1 or a.warmup_seconds < 0:
        ap.error("calls/samples must be positive and warmup must be nonnegative")
    baseline_module = None
    if a.baseline_module:
        import tvm_ffi

        baseline_module = tvm_ffi.load_module(a.baseline_module)
    mega_baseline_module = None
    if a.mega_baseline_module:
        import tvm_ffi

        mega_baseline_module = tvm_ffi.load_module(a.mega_baseline_module)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    calls = 1 if a.cold else a.calls
    # A full eviction before EACH timed single invocation, outside the events.
    flush = (
        torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        if a.cold
        else None
    )
    cases = (
        json.loads(Path(a.cases).read_text())
        if a.cases
        else [
            dict(cache_format=fmt, tokens=t, heads=h, pool_tokens=a.pool_tokens)
            for fmt in a.formats.split(",")
            for h in map(int, a.heads.split(","))
            for t in map(int, a.tokens.split(","))
        ]
    )
    for case in cases:
        args = {k: v for k, v in case.items() if k not in ("name", "source")}
        args.setdefault("pool_tokens", a.pool_tokens)
        args["keep_decoded"] = False
        f = fixture(**args)
        fmt, t, h = f["cache_format"], f["tokens"], f["heads"]
        if baseline_module is not None:
            use_baseline_module(f, baseline_module)
        reference_out, reference_scales, reference_lse = f["baseline"]()
        if a.fused:
            expected = (
                reference_out.float().clone(),
                reference_scales.clone(),
                reference_lse.clone(),
            )
            actual_lse = fused_call(f)
            for actual, target in zip(
                (f["out"].float(), f["scales"], actual_lse),
                expected,
                strict=True,
            ):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
        functions = {name: f[name] for name in ["prologue", "core", "post", "baseline"]}
        if a.fused:
            functions["fused"] = partial(fused_call, f)
        if mega_baseline_module is not None:
            functions["previous_mega"] = previous_mega_call(f, mega_baseline_module)
            previous_lse = functions["previous_mega"]()
            for actual, target in zip(
                (f["out"].float(), f["scales"], previous_lse), expected, strict=True
            ):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
        graphs = {name: capture(fn, calls) for name, fn in functions.items()}
        # Optional sustained warmup to settle clocks before sampling. Keep
        # cold-cache eviction outside every measured invocation below.
        deadline = time.monotonic() + a.warmup_seconds
        while time.monotonic() < deadline:
            for graph in graphs.values():
                graph[0].replay()
            torch.cuda.synchronize()
        samples = {name: [] for name in graphs}
        for repeat in range(a.samples):
            # Rotate the starting arm and reverse each complete rotation. In
            # warm mode, prime each arm separately so preceding stages cannot
            # systematically warm the old/new comparison differently.
            order = list(graphs)
            offset = repeat % len(order)
            order = order[offset:] + order[:offset]
            if (repeat // len(order)) % 2:
                order.reverse()
            for name in order:
                if flush is None:
                    graphs[name][0].replay()
                samples[name].append(measure(graphs[name], calls, flush))
        medians = {name: statistics.median(v) for name, v in samples.items()}
        row = dict(
            case=case,
            cache_format=fmt,
            mega_baseline_module=a.mega_baseline_module,
            baseline_module=a.baseline_module,
            warmup_seconds=a.warmup_seconds,
            tokens=t,
            heads=h,
            pool_tokens=f["pool_tokens"],
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            git=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            plans=[
                dict(p.plan.inspect()) for p in f["wrapper"]._prepared_calls.values()
            ],
            median_us=medians,
            samples_us=samples,
            stages_sum_us=sum(medians[n] for n in ["prologue", "core", "post"]),
            timing=f"CUDA graph with captured timing events: {calls} calls/sample, {a.samples} samples, rotating/reversed order; {'cold, 256MiB eviction before every call' if a.cold else 'warm/reused input, each arm primed separately'}",
            dirty=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        )
        if a.fused:
            row["speedup_pipeline"] = medians["baseline"] / medians["fused"]
            row["speedup_stage_sum"] = row["stages_sum_us"] / medians["fused"]
            row["performance_gate"] = medians["fused"] <= min(
                medians["baseline"], row["stages_sum_us"]
            )
        if mega_baseline_module is not None:
            row["speedup_previous_mega"] = medians["previous_mega"] / medians["fused"]
        with open(a.output, "a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        del graphs, f


if __name__ == "__main__":
    main()
