"""Balanced end-to-end races for GDN prefill production plans."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import statistics
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict

import torch

from b12x.testing.delta_prefill_cases import (
    GDN_PREFILL_CASES,
    check_binding,
    make_inputs,
    oracle,
    prepared_binding,
    run_binding,
)
from benchmarks.experimental.b12x.common import (
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
)


def select_cases(selection):
    if selection == "all":
        return GDN_PREFILL_CASES
    names = selection.split(",")
    by_name = {case.name: case for case in GDN_PREFILL_CASES}
    missing = set(names) - set(by_name)
    if missing:
        raise ValueError(
            f"unknown prefill cases: {sorted(missing)}; available: {list(by_name)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("duplicate prefill cases")
    return tuple(by_name[name] for name in names)


def balanced_order(arms, iteration):
    """Alternate the complete order and its reverse for paired measurements."""
    return arms if iteration % 2 == 0 else tuple(reversed(arms))


def _summary(samples):
    return {
        "median_us": statistics.median(samples),
        "minimum_us": min(samples),
        "samples_us": samples,
    }


def _source(path):
    path = pathlib.Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _versions():
    result = {"torch": str(torch.__version__), "cuda": torch.version.cuda}
    for name in ("nvidia-cutlass-dsl", "flashinfer-python", "cuda-python"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def benchmark_case(
    case,
    *,
    device,
    seed,
    warmup,
    iterations,
    mode,
    flush,
    max_tokens=None,
    max_seqs=None,
    profile_replays=0,
):
    tensors = make_inputs(
        case, device=device, seed=seed, max_tokens=max_tokens, max_seqs=max_seqs
    )
    initial = tensors["recurrent_state"].clone()
    expected, expected_pool = oracle(case, tensors)
    immutable = {
        k: v.clone()
        for k, v in tensors.items()
        if k not in ("recurrent_state", "output")
    }
    arms, reports = [], {}
    factories = ["b12x"]
    prepared_scopes = ExitStack()
    for name in factories:
        report = {"status": "failed", "timings": {}}
        reports[name] = report
        try:
            binding = prepared_scopes.enter_context(
                prepared_binding(
                    case, tensors, max_tokens=max_tokens, max_seqs=max_seqs
                )
            )
            fn = lambda: run_binding("gdn", binding)
            buffers = (binding.scratch,)
            poison = lambda: binding.scratch.fill_(0xFF)
            report["config"] = asdict(binding.execution.selection.config)
            for _ in range(warmup):
                tensors["recurrent_state"].copy_(initial)
                fn()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            tensors["recurrent_state"].copy_(initial)
            from b12x._lib.runtime_control import kernel_resolution_guard

            guard = kernel_resolution_guard("GDN prefill capture")
            with guard, torch.cuda.graph(graph):
                fn()
            tensors["recurrent_state"].copy_(initial)
            tensors["output"].fill_(float("nan"))
            poison()
            bound = (*tensors.values(), *buffers)
            addresses = tuple(t.data_ptr() for t in bound)
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            allocation_delta = torch.cuda.memory_allocated(device) - allocated
            if allocation_delta != 0 or addresses != tuple(t.data_ptr() for t in bound):
                raise AssertionError(
                    f"unstable graph replay: allocation_delta={allocation_delta}"
                )
            correctness = check_binding(case, binding, expected, expected_pool, initial)
            for key, saved in immutable.items():
                torch.testing.assert_close(tensors[key], saved, rtol=0, atol=0)
            torch.testing.assert_close(
                tensors["output"][case.tokens :],
                expected[case.tokens :],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            report.update(
                status="qualified",
                correctness=correctness,
                stable_addresses=True,
                replay_allocation_bytes=allocation_delta,
                input_immutability=True,
                graph_replay_after_output_poison=True,
                graph_replay_after_scratch_poison=True,
            )
            arms.append((name, fn, graph, buffers))
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            if (
                "illegal memory access" in str(exc).lower()
                or "device-side assert" in str(exc).lower()
            ):
                prepared_scopes.close()
                return {"case": asdict(case), "arms": reports, "fatal_cuda_error": True}
    from b12x.testing.benchmark import measure_calls
    from b12x.preparation import PreparedCall

    for temperature in ("warm", "l2_flushed_before_restore"):
        calls = {
            name: PreparedCall(
                run=fn,
                produce=lambda: tensors["recurrent_state"].copy_(initial),
                owners=(buffers, graph),
            )
            for name, fn, graph, buffers in arms
        }
        if not calls:
            break
        calls["restore"] = PreparedCall(
            run=lambda: tensors["recurrent_state"].copy_(initial), produce=lambda: None
        )
        before = nvidia_smi_gpu_mode_snapshot()
        measured = measure_calls(
            calls,
            samples=iterations,
            warmup=0,
            eviction=flush if temperature != "warm" else (lambda: None),
        )
        after = nvidia_smi_gpu_mode_snapshot()
        restore = measured.raw_samples("restore")
        for name, *_ in arms:
            raw = measured.raw_samples(name)
            reports[name]["timings"][f"stream_gated_{temperature}"] = {
                "samples_us": raw,
                "restore_samples_us": restore,
                **_summary(raw),
                "restore_median_us": statistics.median(restore),
                "method": measured.method,
                "gpu_mode_before": before,
                "gpu_mode_after": after,
            }
    if profile_replays and arms:
        torch.cuda.synchronize(device)
        torch.cuda.profiler.start()
        try:
            for iteration in range(profile_replays):
                for name, _fn, graph, _buffers in balanced_order(arms, iteration):
                    tensors["recurrent_state"].copy_(initial)
                    torch.cuda.synchronize(device)
                    with torch.cuda.nvtx.range(
                        f"{case.name}/{name}/replay-{iteration}"
                    ):
                        graph.replay()
                        torch.cuda.synchronize(device)
        finally:
            torch.cuda.profiler.stop()
    prepared_scopes.close()
    return {"case": asdict(case), "arms": reports}


def main(args, argv, parser):
    from benchmarks.experimental.b12x.benchmark_gdn_decode import (
        _device_provenance,
        _git_provenance,
    )

    if args.capacity_columns is not None:
        parser.error(
            "--capacity-columns applies only to decode; prefill uses --capacity-tokens"
        )
    if args.capacity_seqs is not None and args.capacity_seqs > 4096:
        parser.error("prefill --capacity-seqs must be at most 4096")
    try:
        cases = select_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))
    for case in cases:
        if (
            args.capacity_tokens is not None and case.tokens > args.capacity_tokens
        ) or (
            args.capacity_seqs is not None and len(case.lengths) > args.capacity_seqs
        ):
            parser.error(f"{case.name} exceeds the requested prefill capacity")
    device = require_sm120()
    if torch.cuda.get_device_capability(device)[0] != 12:
        parser.error("GDN prefill requires a compute-capability 12.x GPU")
    flush = make_l2_flush_fn(True, args.l2_flush_bytes)
    root = pathlib.Path(__file__).resolve().parents[3]
    sources = [
        p
        for directory in (
            root / "flashinfer/experimental/b12x/sequence/_shared/delta_prefill",
            root / "flashinfer/experimental/b12x/sequence/gdn_prefill",
        )
        for p in directory.glob("*.py")
    ]
    sources += [
        pathlib.Path(__file__),
        root / "flashinfer/experimental/b12x/testing/delta_prefill_cases.py",
    ]
    provenance = {
        "command": [
            sys.executable,
            str(root / "benchmarks/experimental/b12x/benchmark_gdn_decode.py"),
            *argv,
        ],
        "cwd": os.getcwd(),
        "git": _git_provenance(),
        "device": _device_provenance(device),
        "toolchain": _versions(),
        "source_files": [_source(p) for p in sorted(sources)],
        "gpu_mode_before": nvidia_smi_gpu_mode_snapshot(),
        "timestamp_unix": time.time(),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timed_path": "raw Q/K/V, a/b, pooled initial state to recurrence output and pooled final state",
        "checkpoint_export": False,
        "restoration": "identical full pool before every invocation; measured separately; L2 flush precedes restore",
        "metric_direction": "latency_us; lower is better",
        "sampling": "alternating complete arm order and reverse; CUDA events per invocation",
        "reference_timed": False,
        "profile_replays": args.profile_replays,
        "profile_capture": "qualified graph replays after timing; state restored outside each NVTX range",
    }
    reports = []
    print(json.dumps(provenance, sort_keys=True), flush=True)
    try:
        for case in cases:
            report = benchmark_case(
                case,
                device=device,
                seed=args.seed + GDN_PREFILL_CASES.index(case),
                warmup=args.warmup,
                iterations=args.iterations,
                mode=args.mode,
                flush=flush,
                max_tokens=args.capacity_tokens,
                max_seqs=args.capacity_seqs,
                profile_replays=args.profile_replays,
            )
            reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)
            if report.get("fatal_cuda_error"):
                break
    finally:
        provenance["gpu_mode_after"] = nvidia_smi_gpu_mode_snapshot()
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("x", encoding="utf-8") as output:
                json.dump(
                    {"provenance": provenance, "reports": reports},
                    output,
                    indent=2,
                    sort_keys=True,
                )
                output.write("\n")
    return int(
        len(reports) != len(cases)
        or any(
            arm["status"] != "qualified"
            for report in reports
            for arm in report["arms"].values()
        )
    )
