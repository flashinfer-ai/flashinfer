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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,4,16,64,128,512,1024")
    ap.add_argument("--heads", default="16,64")
    ap.add_argument("--formats", default="fp8,nvfp4")
    ap.add_argument("--pool-tokens", type=int, default=1048576)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--warmup-seconds", type=float, default=0.0)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument(
        "--baseline-module", help="Untouched .so with matching execution-plan ABI"
    )
    a = ap.parse_args()
    baseline_module = None
    if a.baseline_module:
        import tvm_ffi

        baseline_module = tvm_ffi.load_module(a.baseline_module)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    calls = 1 if a.cold else 20
    # A full eviction before EACH timed single invocation, outside the events.
    flush = (
        torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        if a.cold
        else None
    )
    for fmt in a.formats.split(","):
        for h in map(int, a.heads.split(",")):
            for t in map(int, a.tokens.split(",")):
                f = fixture(t, h, fmt, a.pool_tokens)
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
                functions = {
                    name: f[name] for name in ["prologue", "core", "post", "baseline"]
                }
                if a.fused:
                    functions["fused"] = partial(fused_call, f)
                graphs = {name: capture(fn, calls) for name, fn in functions.items()}
                # Optional sustained warmup to settle clocks before sampling. Keep
                # cold-cache eviction outside every measured invocation below.
                deadline = time.monotonic() + a.warmup_seconds
                while time.monotonic() < deadline:
                    for graph in graphs.values():
                        graph[0].replay()
                    torch.cuda.synchronize()
                samples = {name: [] for name in graphs}
                for repeat in range(15):
                    order = list(graphs) if repeat % 2 == 0 else list(graphs)[::-1]
                    for name in order:
                        samples[name].append(measure(graphs[name], calls, flush))
                medians = {name: statistics.median(v) for name, v in samples.items()}
                row = dict(
                    cache_format=fmt,
                    baseline_module=a.baseline_module,
                    warmup_seconds=a.warmup_seconds,
                    tokens=t,
                    heads=h,
                    pool_tokens=a.pool_tokens,
                    gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__,
                    git=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    plans=[
                        dict(p.plan.inspect())
                        for p in f["wrapper"]._prepared_calls.values()
                    ],
                    median_us=medians,
                    samples_us=samples,
                    stages_sum_us=sum(medians[n] for n in ["prologue", "core", "post"]),
                    timing=f"CUDA graph with captured timing events: {calls} calls/sample, 15 samples, alternating order; {'cold, 256MiB eviction before every call' if a.cold else 'warm/reused input'}",
                    dirty=subprocess.check_output(
                        ["git", "status", "--porcelain"], text=True
                    ),
                )
                if a.fused:
                    row["speedup_pipeline"] = medians["baseline"] / medians["fused"]
                    row["speedup_stage_sum"] = row["stages_sum_us"] / medians["fused"]
                    row["performance_gate"] = medians["fused"] <= min(
                        medians["baseline"], row["stages_sum_us"]
                    )
                with open(a.output, "a") as out:
                    out.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                del graphs, f


if __name__ == "__main__":
    main()
