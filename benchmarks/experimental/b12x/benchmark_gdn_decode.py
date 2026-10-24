#!/usr/bin/env python3
"""Benchmark public Qwen GDN decode or prefill transactions.

The benchmark restores the recurrent state before every measured invocation.
State restoration is reported separately from CUDA-graph replay latency and is
never included in the kernel result. The GLM/Kimi KDA API is intentionally
outside this benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from b12x.sequence import gdn_decode as gdn
from benchmarks.experimental.b12x.common import (
    bench_cuda_graph,
    capture_cuda_graph,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    query_lengths: tuple[int, ...]
    key_heads: int
    value_heads: int
    state_dtype: torch.dtype = torch.float32

    @property
    def tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def sequences(self) -> int:
        return len(self.query_lengths)

    @property
    def columns(self) -> int:
        return max(self.query_lengths)


QWEN38_GDN_CASES = (
    BenchmarkCase("qk16-v48-decode-bs1", (1,), 16, 48),
    BenchmarkCase("qk8-v24-decode-bs1", (1,), 8, 24),
    BenchmarkCase("qk8-v24-decode-bs4", (1, 1, 1, 1), 8, 24),
    BenchmarkCase("qk8-v24-spec2-bs4", (2, 2, 2, 2), 8, 24),
    BenchmarkCase("qk8-v24-spec4-bs1", (4,), 8, 24),
    BenchmarkCase("qk8-v24-spec4-uneven", (4, 2, 1, 3), 8, 24),
    BenchmarkCase("qk8-v24-spec4-bs4", (4, 4, 4, 4), 8, 24),
    BenchmarkCase("qk4-v12-decode-bs1", (1,), 4, 12),
    BenchmarkCase("qk2-v6-decode-bs1", (1,), 2, 6),
)


@dataclass
class CaseBuffers:
    binding: gdn.Binding
    initial_state: torch.Tensor


def resolve_capacity(
    case: BenchmarkCase,
    *,
    capacity_seqs: int | None,
    capacity_columns: int | None,
) -> tuple[int, int, int]:
    max_seqs = max(4, case.sequences) if capacity_seqs is None else int(capacity_seqs)
    columns = (
        max(4, case.columns) if capacity_columns is None else int(capacity_columns)
    )
    if (
        case.sequences > max_seqs
        or case.columns > columns
        or case.tokens > max_seqs * columns
    ):
        raise ValueError(
            f"case {case.name} exceeds planned capacity "
            f"capacity_seqs={max_seqs},capacity_columns={columns}"
        )
    return max_seqs, columns, max_seqs * columns


@dataclass(frozen=True)
class Correctness:
    output_max_abs: float
    state_max_abs: float
    output_nonzero: int


@dataclass(frozen=True)
class Timing:
    median_us: float
    p90_us: float
    minimum_us: float
    samples_us: tuple[float, ...]
    restore_median_us: float | None = None


@dataclass(frozen=True)
class CaseReport:
    case: BenchmarkCase
    correctness: Correctness
    eager: Timing | None
    graph: Timing | None
    graph_correctness: Correctness | None
    graph_replay_after_output_poison: bool
    stable_addresses: bool
    replay_allocation_bytes: int


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator,
    dtype: torch.dtype = torch.bfloat16,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, generator=generator, dtype=torch.float32)
        .mul_(scale)
        .to(device=device, dtype=dtype)
        .contiguous()
    )


def build_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int,
    capacity_seqs: int | None = None,
    capacity_columns: int | None = None,
) -> CaseBuffers:
    if case.value_heads != 3 * case.key_heads:
        raise ValueError(f"Qwen GDN requires value_heads=3*key_heads: {case}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    live_tokens = case.tokens
    live_seqs = case.sequences
    max_seqs, columns, max_tokens = resolve_capacity(
        case,
        capacity_seqs=capacity_seqs,
        capacity_columns=capacity_columns,
    )
    state_slots = max_tokens + 1
    caps = gdn.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        max_state_slots=state_slots,
        key_heads=case.key_heads,
        value_heads=case.value_heads,
        state_index_columns=columns,
        state_dtype=case.state_dtype,
        gate_activation="sigmoid",
        qk_l2norm=True,
    )
    planned = gdn.plan(caps)
    (scratch_spec,) = planned.scratch_specs()
    query_start_loc = torch.full(
        (max_seqs + 1,), live_tokens, dtype=torch.int32, device=device
    )
    query_start_loc[0] = 0
    query_start_loc[1 : live_seqs + 1].copy_(
        torch.tensor(case.query_lengths, dtype=torch.int32, device=device).cumsum(0)
    )
    state_indices = torch.arange(
        max_seqs * columns, dtype=torch.int32, device=device
    ).view(max_seqs, columns)
    tensors = {
        "scratch": torch.empty(
            scratch_spec.shape, dtype=scratch_spec.dtype, device=device
        ),
        "mixed_qkv": _randn(
            (max_tokens, caps.packed_qkv_width),
            device=device,
            generator=generator,
        ),
        "a": _randn((max_tokens, case.value_heads), device=device, generator=generator),
        "b": _randn((max_tokens, case.value_heads), device=device, generator=generator),
        "z": _randn(
            (max_tokens, case.value_heads, 128), device=device, generator=generator
        ),
        "A_log": _randn(
            (case.value_heads,),
            device=device,
            generator=generator,
            dtype=torch.float32,
            scale=0.1,
        ),
        "dt_bias": _randn(
            (case.value_heads,),
            device=device,
            generator=generator,
            dtype=torch.float32,
            scale=0.1,
        ),
        "norm_weight": (
            1.0
            + _randn(
                (128,),
                device=device,
                generator=generator,
                scale=0.05,
            )
        ).contiguous(),
        "recurrent_state": _randn(
            (state_slots, case.value_heads, 128, 128),
            device=device,
            generator=generator,
            dtype=case.state_dtype,
            scale=0.1,
        ),
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": torch.ones(max_seqs, dtype=torch.int32, device=device),
        "state_indices": state_indices,
        "num_seqs": torch.tensor([live_seqs], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([live_tokens], dtype=torch.int32, device=device),
        "output": torch.empty(
            (max_tokens, case.value_heads, 128),
            dtype=torch.bfloat16,
            device=device,
        ),
    }
    binding = gdn.bind(planned, **tensors)
    return CaseBuffers(binding=binding, initial_state=binding.recurrent_state.clone())


def _reference(buffers: CaseBuffers) -> tuple[torch.Tensor, torch.Tensor]:
    binding = buffers.binding
    caps = binding.plan.caps
    state = buffers.initial_state.clone()
    output = gdn.reference.decode(
        binding.mixed_qkv,
        binding.a,
        binding.b,
        binding.z,
        binding.A_log,
        binding.dt_bias,
        binding.norm_weight,
        state,
        binding.query_start_loc,
        binding.num_accepted_tokens,
        binding.state_indices,
        binding.num_seqs,
        binding.num_tokens,
        key_heads=caps.key_heads,
        value_heads=caps.value_heads,
        gate_activation="sigmoid",
        qk_l2norm=True,
    )
    return output, state


def check_correctness(buffers: CaseBuffers) -> Correctness:
    expected_output, expected_state = _reference(buffers)
    buffers.binding.recurrent_state.copy_(buffers.initial_state)
    actual = gdn.run(buffers.binding)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(actual).all().item()):
        raise RuntimeError("GDN output contains non-finite values")
    nonzero = int(torch.count_nonzero(actual).item())
    if nonzero == 0:
        raise RuntimeError("GDN output is all zero")
    torch.testing.assert_close(actual, expected_output, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        buffers.binding.recurrent_state,
        expected_state,
        rtol=1e-2 if buffers.binding.recurrent_state.dtype == torch.bfloat16 else 1e-5,
        atol=8e-3 if buffers.binding.recurrent_state.dtype == torch.bfloat16 else 2e-5,
    )
    output_max_abs = float((actual.float() - expected_output.float()).abs().max())
    state_max_abs = float(
        (buffers.binding.recurrent_state.float() - expected_state.float()).abs().max()
    )
    buffers.binding.recurrent_state.copy_(buffers.initial_state)
    return Correctness(output_max_abs, state_max_abs, nonzero)


def _check_current_result(
    buffers: CaseBuffers,
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
) -> Correctness:
    actual = buffers.binding.output
    if not bool(torch.isfinite(actual).all().item()):
        raise RuntimeError("GDN graph output contains non-finite values")
    nonzero = int(torch.count_nonzero(actual).item())
    if nonzero == 0:
        raise RuntimeError("GDN graph output is all zero")
    torch.testing.assert_close(actual, expected_output, rtol=1e-2, atol=2e-2)
    state = buffers.binding.recurrent_state
    torch.testing.assert_close(
        state,
        expected_state,
        rtol=1e-2 if state.dtype == torch.bfloat16 else 1e-5,
        atol=8e-3 if state.dtype == torch.bfloat16 else 2e-5,
    )
    return Correctness(
        float((actual.float() - expected_output.float()).abs().max()),
        float((state.float() - expected_state.float()).abs().max()),
        nonzero,
    )


def _percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        raise ValueError("timing samples must not be empty")
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999))
    return ordered[index]


def _timing(samples: list[float], restore: list[float] | None = None) -> Timing:
    return Timing(
        median_us=statistics.median(samples),
        p90_us=_percentile(samples, 0.9),
        minimum_us=min(samples),
        samples_us=tuple(samples),
        restore_median_us=None if restore is None else statistics.median(restore),
    )


def _bench_eager(buffers, *, warmup, iterations, l2_flush):
    from b12x.testing.benchmark import measure_call

    binding = buffers.binding
    result = measure_call(
        lambda: gdn.run(binding),
        warmup=warmup,
        samples=iterations,
        reset=lambda: binding.recurrent_state.copy_(buffers.initial_state),
        eviction=l2_flush or (lambda: None),
    )
    return _timing(result.raw_samples("workload"))


def _bench_graph(
    buffers: CaseBuffers,
    *,
    warmup: int,
    iterations: int,
    l2_flush,
) -> tuple[Timing, Correctness, bool, int]:
    binding = buffers.binding
    expected_output, expected_state = _reference(buffers)

    def restore() -> None:
        binding.recurrent_state.copy_(buffers.initial_state)

    def launch() -> torch.Tensor:
        return gdn.run(binding)

    graph = capture_cuda_graph(launch, warmup=warmup, prepare=restore)
    addresses = (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    restore()
    binding.output.fill_(float("nan"))
    torch.cuda.synchronize(binding.output.device)
    allocated_before = torch.cuda.memory_allocated(binding.output.device)
    graph.replay()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated(binding.output.device)
    replay_correctness = _check_current_result(
        buffers,
        expected_output,
        expected_state,
    )
    stats = bench_cuda_graph(
        graph,
        replays=iterations,
        prepare=restore,
        l2_flush=l2_flush,
    )
    stable = addresses == (
        binding.recurrent_state.data_ptr(),
        binding.output.data_ptr(),
        binding.scratch.data_ptr(),
    )
    return (
        _timing(stats["replay_us"], stats["metadata_us"]),
        replay_correctness,
        stable,
        allocated_after - allocated_before,
    )


def benchmark_case(
    case: BenchmarkCase,
    *,
    device: torch.device,
    seed: int,
    warmup: int,
    iterations: int,
    mode: str,
    l2_flush,
    capacity_seqs: int | None = None,
    capacity_columns: int | None = None,
) -> CaseReport:
    buffers = build_case(
        case,
        device=device,
        seed=seed,
        capacity_seqs=capacity_seqs,
        capacity_columns=capacity_columns,
    )
    correctness = check_correctness(buffers)
    eager = (
        _bench_eager(buffers, warmup=warmup, iterations=iterations, l2_flush=l2_flush)
        if mode in ("eager", "both")
        else None
    )
    graph = None
    graph_correctness = None
    stable_addresses = True
    replay_allocation_bytes = 0
    if mode in ("graph", "both"):
        (
            graph,
            graph_correctness,
            stable_addresses,
            replay_allocation_bytes,
        ) = _bench_graph(
            buffers,
            warmup=warmup,
            iterations=iterations,
            l2_flush=l2_flush,
        )
        if not stable_addresses:
            raise RuntimeError("GDN graph replay changed a bound tensor address")
        if replay_allocation_bytes != 0:
            raise RuntimeError(
                f"GDN graph replay allocated {replay_allocation_bytes} bytes"
            )
    return CaseReport(
        case=case,
        correctness=correctness,
        eager=eager,
        graph=graph,
        graph_correctness=graph_correctness,
        graph_replay_after_output_poison=graph_correctness is not None,
        stable_addresses=stable_addresses,
        replay_allocation_bytes=replay_allocation_bytes,
    )


def select_cases(raw: str) -> tuple[BenchmarkCase, ...]:
    if raw.strip().lower() in ("", "all"):
        return QWEN38_GDN_CASES
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    by_name = {case.name: case for case in QWEN38_GDN_CASES}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"unknown cases {missing}; choices are {sorted(by_name)}")
    return tuple(by_name[name] for name in requested)


def _git_provenance() -> dict[str, object]:
    worktree = pathlib.Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
                cwd=worktree,
            )
        except OSError:
            return "unknown"
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "worktree": str(worktree),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty_paths": git("status", "--short").splitlines(),
    }


def _device_provenance(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "logical_device": device.index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unknown")),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
    }


def _timing_text(timing: Timing | None) -> str:
    if timing is None:
        return "-"
    return (
        f"median={timing.median_us:.2f}us p90={timing.p90_us:.2f}us "
        f"min={timing.minimum_us:.2f}us"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("decode", "prefill"), default="decode")
    parser.add_argument(
        "--capacity-tokens", type=int, help="planned prefill token capacity"
    )
    parser.add_argument(
        "--policy-profile",
        type=pathlib.Path,
        help="prefill profile artifact; require measured coverage for every case",
    )
    parser.add_argument("--cases", default="all")
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--profile-replays",
        type=int,
        default=0,
        help="prefill diagnostic graph replays inside a CUDA profiler capture range",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument(
        "--capacity-seqs",
        type=int,
        help="planned max_num_seqs; defaults to at least 4",
    )
    parser.add_argument(
        "--capacity-columns",
        type=int,
        help="planned state-index columns; defaults to at least 4",
    )
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")
    if args.profile_replays < 0:
        parser.error("--profile-replays must be nonnegative")
    if args.capacity_tokens is not None and args.capacity_tokens < 1:
        parser.error("--capacity-tokens must be positive")
    if args.capacity_seqs is not None and args.capacity_seqs < 1:
        parser.error("--capacity-seqs must be positive")
    if args.capacity_columns is not None and not 1 <= args.capacity_columns <= 8:
        parser.error("--capacity-columns must be in [1, 8]")
    if args.json is not None:
        args.json = args.json.expanduser().resolve()
        if args.json.exists():
            parser.error(f"refusing to overwrite existing output: {args.json}")
    if args.operation == "prefill":
        from benchmarks.experimental.b12x._gdn_prefill_race import main as prefill_main

        return prefill_main(args, list(sys.argv[1:] if argv is None else argv), parser)
    if (
        args.capacity_tokens is not None
        or args.policy_profile is not None
        or args.profile_replays
    ):
        parser.error(
            "--capacity-tokens, --policy-profile, and --profile-replays require --operation prefill"
        )
    try:
        cases = select_cases(args.cases)
    except ValueError as error:
        parser.error(str(error))

    device = require_sm120()
    l2_flush = make_l2_flush_fn(args.l2_flush, args.l2_flush_bytes)
    command_argv = list(sys.argv[1:] if argv is None else argv)
    gpu_mode_before = nvidia_smi_gpu_mode_snapshot()
    provenance = {
        "command": [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            *command_argv,
        ],
        "cwd": os.getcwd(),
        "git": _git_provenance(),
        "device": _device_provenance(device),
        "gpu_mode_before": gpu_mode_before,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "timed_path": "b12x.sequence.gdn_decode public Qwen decode transaction",
        "recurrence_backend": "cutedsl",
        "triton_role": "gated_rmsnorm_auxiliary",
        "reference_timed": False,
        "metric_direction": "lower_is_better",
        "mode": args.mode,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "l2_flush": bool(args.l2_flush),
        "capacity_policy": {
            "requested_max_seqs": args.capacity_seqs,
            "requested_state_index_columns": args.capacity_columns,
            "default_minimum_max_seqs": 4,
            "default_minimum_state_index_columns": 4,
        },
    }
    print(json.dumps(_jsonable(provenance), sort_keys=True))
    reports: list[CaseReport] = []
    for index, case in enumerate(cases):
        report = benchmark_case(
            case,
            device=device,
            seed=args.seed + index,
            warmup=args.warmup,
            iterations=args.iterations,
            mode=args.mode,
            l2_flush=l2_flush,
            capacity_seqs=args.capacity_seqs,
            capacity_columns=args.capacity_columns,
        )
        reports.append(report)
        print(
            f"{case.name}: eager[{_timing_text(report.eager)}] "
            f"graph[{_timing_text(report.graph)}] "
            f"output_max_abs={report.correctness.output_max_abs:.6g} "
            f"state_max_abs={report.correctness.state_max_abs:.6g}"
        )
    gpu_mode_after = nvidia_smi_gpu_mode_snapshot()
    provenance["gpu_mode_after"] = gpu_mode_after
    if args.json is not None:
        payload = {
            "provenance": provenance,
            "reports": [asdict(report) for report in reports],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
            )
    print(
        json.dumps(
            _jsonable(
                {
                    "status": "passed",
                    "gpu_mode_before": gpu_mode_before,
                    "gpu_mode_after": gpu_mode_after,
                }
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    raise SystemExit(main())
