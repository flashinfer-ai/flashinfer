"""Compare explicit SM120 SVDQuant backends with identical inputs on one GPU.

Examples (run from the repository root):
    python benchmarks/bench_svdquant_sm120_backends.py --output small.jsonl
    python benchmarks/bench_svdquant_sm120_backends.py --all-shapes --output all.jsonl
    python benchmarks/bench_svdquant_sm120_backends.py --shape 537x5376x7168 --cold-l2

Timings are CUDA-event measurements of graph device work, with five alternating
backend-order trials by default. GEMM uses a preallocated output; linear covers
the public chain, including operations captured while allocating its output.
Warm samples report a graph batch's mean time per operation. Cold samples time
one operation per graph replay, after zeroing a 2x-L2 buffer before the start
event. The flush is excluded; device gaps between events and graph work remain
part of the elapsed time. A speedup above one means CUTLASS is faster. An exact
output match records a null SQNR.
--tuning-l2 selects the tuning policy independently of the measurement's
--cold-l2 switch; operator preserves each operator's default tuning behavior.
--tuning-repeat and --tuning-replays optionally control profiling precision for
both backends; omitted values preserve the autotuner/operator defaults.
For sustained warm profiling, use --tuning-l2 warm --tuning-repeat 64
--tuning-replays 5. tuning_time_s includes first-use compilation, profiling,
and synchronization, making the cost of additional profiling visible.
"""

import gc
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import ContextManager, TextIO

import torch

from flashinfer.testing.svdq_model_shapes import ShapeCase
from flashinfer.testing.utils import (
    bench_gpu_time_with_cuda_event,
    bench_gpu_time_with_cudagraph,
    get_l2_cache_size,
)
from svdquant_sm120_benchmark import (
    BACKENDS,
    Backend,
    Call,
    Case,
    Environment,
    Operation,
    build_case,
    compare_outputs,
    environment,
    graph_call_count,
    make_call,
    snapshot_winners,
    tuning_context,
    tuning_policy,
    tuning_precision,
)
from svdquant_sm120_cli import Options, parse_options


def _measure_cold_graph(call: Call, warmup_ms: int, repeat_ms: int) -> list[float]:
    """Capture one operation and flush L2 outside every replay's event window."""
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return bench_gpu_time_with_cuda_event(
        graph.replay,
        dry_run_time_ms=warmup_ms,
        repeat_time_ms=repeat_ms,
        cold_l2_cache=True,
    )


def measure(
    call: Call, graph_calls: int, cold_l2: bool, warmup_ms: int, repeat_ms: int
) -> list[float]:
    if cold_l2:
        samples = _measure_cold_graph(call, warmup_ms, repeat_ms)
    else:
        samples = bench_gpu_time_with_cudagraph(
            call.function,
            input_args=call.operands,
            dry_run_time_ms=warmup_ms,
            repeat_time_ms=repeat_ms,
            num_iters_within_graph=graph_calls,
            cold_l2_cache=False,
        )
    samples_us = [float(value) * 1000.0 for value in samples]
    assert samples_us and all(
        math.isfinite(value) and value > 0 for value in samples_us
    )
    return samples_us


def run_operation(
    case: Case,
    shape: ShapeCase,
    operation: Operation,
    options: Options,
    env: Environment,
    stream: TextIO,
) -> None:
    calls = {
        backend: make_call(case, operation, backend, options.enable_pdl)
        for backend in BACKENDS
    }
    policy = tuning_policy(options.tuning_l2)
    tuning_time_s: dict[Backend, float] = {}
    torch.cuda.synchronize()
    for backend, call in calls.items():
        print(f"TUNE {shape} {operation}/{backend}", file=sys.stderr, flush=True)
        tuning_started = time.perf_counter()
        with tuning_context(policy, tuning=True):
            call()
        torch.cuda.synchronize()
        tuning_time_s[backend] = time.perf_counter() - tuning_started
    with tuning_context(policy, tuning=False):
        accuracy = compare_outputs(calls["cute-dsl"](), calls["cutlass-sm120"]())
        graph_calls = (
            1
            if options.cold_l2
            else graph_call_count(
                shape, options.graph_calls, options.graph_output_budget_mib
            )
        )
        samples: dict[Backend, list[list[float]]] = {
            backend: [] for backend in BACKENDS
        }
        orders = []
        trial_windows = []
        for trial in range(options.trials):
            order = BACKENDS if trial % 2 == 0 else tuple(reversed(BACKENDS))
            orders.append(order)
            windows: dict[Backend, dict[str, int]] = {}
            trial_windows.append(windows)
            for backend in order:
                started = time.time_ns()
                samples[backend].append(
                    measure(
                        calls[backend],
                        graph_calls,
                        options.cold_l2,
                        options.warmup_ms,
                        options.repeat_ms,
                    )
                )
                windows[backend] = {
                    "start_unix_ns": started,
                    "end_unix_ns": time.time_ns(),
                }
        trial_medians = {
            backend: [statistics.median(trial) for trial in trials]
            for backend, trials in samples.items()
        }
        medians = {
            backend: statistics.median(trials)
            for backend, trials in trial_medians.items()
        }
        record = {
            "schema_version": 2,
            "environment": asdict(env),
            "shape": shape,
            "operation": operation,
            "rank": 32,
            "bias": options.bias,
            "seed": options.seed,
            "enable_pdl": options.enable_pdl,
            "inference_mode": options.inference_mode,
            "timer": "cuda-event-graph",
            "sample_observable": (
                "isolated_graph_replay_elapsed_us"
                if options.cold_l2
                else "graph_replay_elapsed_us_divided_by_graph_calls"
            ),
            "cold_l2": options.cold_l2,
            "l2_policy": "flush_before_each_replay"
            if options.cold_l2
            else "reuse_operands",
            "l2_flush_bytes": (
                (2 * get_l2_cache_size() // (1 << 20)) * (1 << 20)
                if options.cold_l2
                else 0
            ),
            "l2_flush_in_timed_region": False,
            "tuning_l2": options.tuning_l2,
            "tuning_policy": asdict(policy) if policy is not None else None,
            "tuning_repeat": options.tuning_repeat,
            "tuning_replays": options.tuning_replays,
            "tuning_time_s": tuning_time_s,
            "winning_tactics": snapshot_winners(),
            "graph_calls": graph_calls,
            "warmup_ms": options.warmup_ms,
            "repeat_ms": options.repeat_ms,
            "trial_order": orders,
            "trial_windows": trial_windows,
            "accuracy": asdict(accuracy),
            "samples_us": samples,
            "trial_medians_us": trial_medians,
            "median_us": medians,
            "cutlass_speedup": medians["cute-dsl"] / medians["cutlass-sm120"],
        }
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        if options.output:
            options.output.with_suffix(".tactics.json").write_text(
                json.dumps(
                    {
                        "tuning_l2": record["tuning_l2"],
                        "tuning_policy": record["tuning_policy"],
                        "tuning_repeat": record["tuning_repeat"],
                        "tuning_replays": record["tuning_replays"],
                        "winners": record["winning_tactics"],
                    }
                ),
                encoding="utf-8",
            )
        print(
            f"RESULT {shape} {operation}: {record['cutlass_speedup']:.3f}x CUTLASS speedup",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    options = parse_options(description=__doc__)
    env = environment()
    context = torch.inference_mode if options.inference_mode else torch.no_grad
    output_context: ContextManager[TextIO]
    if options.output:
        output_context = options.output.open("w", encoding="utf-8")
    else:
        output_context = nullcontext(sys.stdout)
    with (
        tuning_precision(repeat=options.tuning_repeat, replays=options.tuning_replays),
        output_context as stream,
    ):
        for shape in options.shapes:
            with context():
                case = build_case(shape, options.seed, options.bias)
                for operation in options.operations:
                    run_operation(case, shape, operation, options, env, stream)
                del case
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
