# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibrate and sweep Prims-TS MoE kernel configurations.

Run ``calibrate`` once for a GPU/software environment, then run separate
``sweep --fc fc1`` and ``sweep --fc fc2`` commands. Pass ``--resume`` with the
same output directory after interruption; code hotfixes start a new recorded
segment unless ``--strict-resume`` is used. Each sweep writes ``results.jsonl``,
``state.json``, ``run_manifest.json``, ``summary.json``, and ``report.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import warnings

from .calibration import (
    CalibrationData,
    CompileScalingPoint,
    EnvironmentFingerprint,
    MAX_CALIBRATION_WORKERS,
    ShapeGPUCost,
    WorkerGPUFootprint,
    derive_noise_band,
    select_compile_scaling_knee,
)
from .defaults import (
    CALIBRATION_REPEAT_ITERS,
    CALIBRATION_WARMUP_ITERS,
    CALIBRATION_WORKER_COUNTS,
    CANDIDATE_TIMEOUT_SECONDS,
    COMPILE_TIMEOUT_SECONDS,
    DEFAULT_BUDGET_MINUTES,
    DEFAULT_CONFIRM_FRACTION,
    DEFAULT_GPU,
    DEFAULT_MAX_ACCUMULATOR_STAGES,
    DEFAULT_MAX_C_SMEM_STAGES,
    DEFAULT_MAX_INPUT_STAGES,
    DEFAULT_MAX_NEAR_BEST_PER_SHAPE,
    DEFAULT_MAX_ROUTING_REGRET,
    DEFAULT_MAX_SF_RING_RATIO,
    DEFAULT_MAX_WORKID_STAGES,
    DEFAULT_MENU_SIZE,
    DEFAULT_REFERENCE_RESERVE_MINUTES,
    DEFAULT_RETIME_ROUTINGS,
    DEFAULT_SEARCH_ROUTING,
    DEFAULT_TAIL_MINUTES,
    DEFAULT_UNCALIBRATED_WORKERS,
    MAX_SEEDS,
    PROGRESS_INTERVAL_SECONDS,
    RANDOM_SEED,
    ROUTING_SEED,
    SEARCH_REPEAT_ITERS,
    SEARCH_TOKENS,
    SEARCH_WARMUP_ITERS,
    SEED_LIMIT,
    SHAPE_TIMEOUT_SECONDS,
    WORKER_RESTART_CONFIGS,
)
from .engine import (
    SweepEngine,
    SweepOptions,
    run_worker_loop,
)
from .measure import TimingProtocol
from .menu_config import load_menu_target
from .menu_emit import emit_menu_prefix
from .menu_report import write_menu_report
from .menu_retime import collect_menu_candidates, run_menu_retime
from .menu_select import load_joint_inputs, select_joint_menu
from .shapes import (
    MODEL_PRESETS,
    default_global_num_tokens,
    resolve_activation_params,
)
from .utils import (
    available_cpu_count,
    git_output,
    nvml_driver_version,
    resolve_visible_gpu,
)

_CUDA_ARCH_ENV = {
    100: "10.0a",
    103: "10.3a",
    107: "10.7a",
}


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)


def _float_or_none(value: str) -> float | str:
    if value.lower() == "none":
        return "none"
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a floating-point value or 'none', got {value!r}"
        ) from exc


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", choices=sorted(MODEL_PRESETS), help="Built-in model preset."
    )
    group.add_argument(
        "--dtype", help="Custom model input/weight dtype or quantization mode."
    )
    parser.add_argument("--hidden", type=int, help="Custom model hidden size.")
    parser.add_argument(
        "--intermediate", type=int, help="Custom model intermediate size."
    )
    parser.add_argument(
        "--num-experts", type=int, help="Custom model global expert count."
    )
    parser.add_argument("--top-k", type=int, help="Custom model routed top-k.")
    parser.add_argument("--activation", help="Custom model activation name.")
    parser.add_argument(
        "--bias",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable FC1/FC2 bias (default: use the model preset).",
    )
    parser.add_argument(
        "--gemm1-alpha",
        type=_float_or_none,
        default=None,
        metavar="FLOAT|none",
        help="FC1 per-expert activation alpha (default: use the model preset).",
    )
    parser.add_argument(
        "--gemm1-beta",
        type=_float_or_none,
        default=None,
        metavar="FLOAT|none",
        help="FC1 per-expert activation beta (default: use the model preset).",
    )
    parser.add_argument(
        "--gemm1-clamp-limit",
        type=_float_or_none,
        default=None,
        metavar="FLOAT|none",
        help="FC1 per-expert clamp limit (default: use the model preset).",
    )


def _validate_model_args(parser: argparse.ArgumentParser, args) -> None:
    explicit = (
        args.hidden,
        args.intermediate,
        args.num_experts,
        args.top_k,
        args.activation,
    )
    if args.model is None and any(value is None for value in explicit):
        parser.error(
            "--dtype requires --hidden, --intermediate, --num-experts, "
            "--top-k, and --activation"
        )
    if args.model is not None and any(value is not None for value in explicit):
        parser.error("--model cannot be combined with explicit model fields")


def _tp_ep_pair(value: str) -> tuple[int, int]:
    try:
        tp_text, ep_text = value.split(":", maxsplit=1)
        tp, ep = int(tp_text), int(ep_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid TP/EP pair {value!r}; expected TP:EP, for example 4:4"
        ) from exc
    if tp <= 0 or ep <= 0:
        raise argparse.ArgumentTypeError("TP and EP values must be positive")
    return tp, ep


def _add_shape_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument("--tp", type=int, nargs="+", help="Tensor parallel degrees.")
    parser.add_argument("--ep", type=int, nargs="+", help="Expert parallel degrees.")
    parser.add_argument(
        "--tp-ep",
        type=_tp_ep_pair,
        nargs="+",
        metavar="TP:EP",
        help="Exact TP/EP pairs, e.g. --tp-ep 2:2 4:4 8:8.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        help="Global token counts; uses the standard search grid when omitted.",
    )


def _validate_parallelism_args(parser: argparse.ArgumentParser, args) -> None:
    if args.tp_ep is not None:
        if args.tp is not None or args.ep is not None:
            parser.error("--tp-ep cannot be combined with --tp or --ep")
        return
    if args.tp is None or args.ep is None:
        parser.error("provide either --tp-ep TP:EP [...] or both --tp and --ep")


def _tokens(args) -> tuple[int, ...]:
    return tuple(args.tokens or SEARCH_TOKENS)


def _final_tokens(args) -> tuple[int, ...]:
    return tuple(
        getattr(args, "final_tokens", None)
        or (*default_global_num_tokens(8192), 12288, 16384)
    )


def _parallelism_args(
    args,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...] | None]:
    if args.tp_ep is not None:
        pairs = tuple(dict.fromkeys(args.tp_ep))
        return (
            tuple(dict.fromkeys(tp for tp, _ in pairs)),
            tuple(dict.fromkeys(ep for _, ep in pairs)),
            pairs,
        )
    return tuple(args.tp), tuple(args.ep), None


def _sweep_options(args, *, out_dir: Path, fc: str, workers: int) -> SweepOptions:
    tp, ep, tp_ep = _parallelism_args(args)
    activation_params = resolve_activation_params(
        args.model,
        has_bias=args.bias,
        gemm1_alpha=args.gemm1_alpha,
        gemm1_beta=args.gemm1_beta,
        gemm1_clamp_limit=args.gemm1_clamp_limit,
    )
    return SweepOptions(
        out_dir=out_dir,
        fc=fc,
        model=args.model,
        dtype=args.dtype,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_experts=args.num_experts,
        top_k=args.top_k,
        activation=args.activation,
        has_bias=activation_params.has_bias,
        gemm1_alpha=activation_params.gemm1_alpha,
        gemm1_beta=activation_params.gemm1_beta,
        gemm1_clamp_limit=activation_params.gemm1_clamp_limit,
        sm_arch=args.sm_arch,
        tp=tp,
        ep=ep,
        tp_ep=tp_ep,
        tokens=_tokens(args),
        final_tokens=_final_tokens(args),
        budget_minutes=float(getattr(args, "budget_min", DEFAULT_BUDGET_MINUTES)),
        tail_minutes=float(getattr(args, "tail_min", DEFAULT_TAIL_MINUTES)),
        menu_size=int(getattr(args, "menu_size", DEFAULT_MENU_SIZE)),
        seed_limit=SEED_LIMIT,
        max_seeds=MAX_SEEDS,
        max_near_best_per_shape=int(
            getattr(
                args,
                "max_near_best_per_shape",
                DEFAULT_MAX_NEAR_BEST_PER_SHAPE,
            )
        ),
        reference_reserve_minutes=float(
            getattr(
                args,
                "reference_reserve_min",
                DEFAULT_REFERENCE_RESERVE_MINUTES,
            )
        ),
        seed=RANDOM_SEED,
        worker_count=workers,
        candidate_timeout_seconds=CANDIDATE_TIMEOUT_SECONDS,
        gpu=str(getattr(args, "gpu", 0)),
        dry_run=bool(getattr(args, "dry_run", False)),
        confirm_fraction=float(
            getattr(args, "confirm_fraction", DEFAULT_CONFIRM_FRACTION)
        ),
        warmup_iters=SEARCH_WARMUP_ITERS,
        repeat_iters=SEARCH_REPEAT_ITERS,
        resume=bool(getattr(args, "resume", False)),
        strict_resume=bool(getattr(args, "strict_resume", False)),
        max_input_stages=int(getattr(args, "max_input_stages", 16)),
        max_accumulator_stages=int(getattr(args, "max_accumulator_stages", 10)),
        max_c_smem_stages=int(getattr(args, "max_c_smem_stages", 4)),
        max_workid_stages=int(getattr(args, "max_workid_stages", 8)),
        max_sf_ring_ratio=int(getattr(args, "max_sf_ring_ratio", 4)),
        compile_timeout_seconds=COMPILE_TIMEOUT_SECONDS,
        shape_timeout_seconds=SHAPE_TIMEOUT_SECONDS,
        worker_restart_configs=WORKER_RESTART_CONFIGS,
        progress_interval_seconds=PROGRESS_INTERVAL_SECONDS,
        search_routing=str(getattr(args, "search_routing", "uniform")),
        retime_routings=tuple(
            getattr(
                args,
                "retime_routings",
                ("uniform", "zipf:0.8", "balanced_round_robin"),
            )
        ),
        selection_routings=(
            None
            if getattr(args, "selection_routings", None) is None
            else tuple(args.selection_routings)
        ),
        routing_seed=ROUTING_SEED,
        max_routing_regret=float(
            getattr(args, "max_routing_regret", DEFAULT_MAX_ROUTING_REGRET)
        ),
        enable_cluster_y=bool(getattr(args, "enable_cluster_y", False)),
        retry_failed=bool(getattr(args, "retry_failed", False)),
    )


def _environment_fingerprint(gpu: str = "0") -> EnvironmentFingerprint:
    import torch

    visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    resolved_gpu = resolve_visible_gpu(gpu, visible_spec)
    visible_devices = (
        []
        if not visible_spec
        else [value.strip() for value in visible_spec.split(",") if value.strip()]
    )
    if visible_devices:
        device_index = visible_devices.index(resolved_gpu)
    else:
        try:
            device_index = int(resolved_gpu)
        except ValueError:
            device_index = next(
                (
                    index
                    for index in range(torch.cuda.device_count())
                    if str(getattr(torch.cuda.get_device_properties(index), "uuid", ""))
                    == resolved_gpu
                ),
                torch.cuda.current_device(),
            )
    driver = nvml_driver_version()
    if driver is None:
        try:
            driver = str(torch._C._cuda_getDriverVersion())
        except (AttributeError, RuntimeError):
            driver = "unknown"
    try:
        import cutlass

        cute_dsl = str(getattr(cutlass, "__version__", "unknown"))
    except ImportError:
        cute_dsl = "unknown"
    sha = git_output("rev-parse", "HEAD") or "unknown"
    properties = torch.cuda.get_device_properties(device_index)
    return EnvironmentFingerprint(
        gpu=properties.name,
        driver=driver,
        cuda=str(torch.version.cuda),
        cute_dsl=cute_dsl,
        cpu=platform.processor() or platform.machine(),
        flashinfer_sha=sha,
        sm_count=int(getattr(properties, "multi_processor_count", 0)) or None,
        clock_rate_khz=(int(getattr(properties, "clock_rate", 0)) or None),
        memory_clock_rate_khz=(
            int(getattr(properties, "memory_clock_rate", 0)) or None
        ),
        compute_capability=(f"{int(properties.major)}.{int(properties.minor)}"),
    )


def _measurement_samples(result: dict[str, object]) -> list[float]:
    samples = []
    for measurement in result.get("measurements", ()):
        if measurement.get("status") != "ok":
            continue
        raw = measurement.get("samples_us") or (measurement.get("median_us"),)
        samples.extend(float(value) for value in raw if value is not None)
    return samples


def _calibration_failure(results: list[dict[str, object]]) -> str:
    details = []
    for result in results:
        status = str(result.get("status", "unknown"))
        failure = result.get("failure")
        error = str(result.get("error") or "").strip()
        prefix = status if failure is None else f"{status} ({failure})"
        details.append(f"{prefix}: {error}" if error else prefix)
    return "\n".join(dict.fromkeys(details)) or "workers returned no diagnostics"


def _calibration_worker_counts(
    requested: Sequence[int], *, cpu_count: int | None = None
) -> tuple[int, ...]:
    if cpu_count is None:
        cpu_count = available_cpu_count()
    available = max(1, cpu_count or 1)
    return tuple(
        sorted(
            {
                max(
                    1,
                    min(available, MAX_CALIBRATION_WORKERS, int(value)),
                )
                for value in requested
            }
        )
    )


def _fit_calibration_worker_counts(
    requested: Sequence[int], *, available_jobs: int
) -> tuple[int, ...]:
    fitted = []
    consumed = 0
    for count in requested:
        job_count = max(2, count)
        if consumed + job_count > available_jobs:
            break
        fitted.append(count)
        consumed += job_count
    if not fitted:
        raise RuntimeError(
            "calibration needs at least two distinct measured candidates after "
            "reserving one FC1 and FC2 warmup candidate"
        )
    return tuple(fitted)


def _dispatch_calibration_jobs(
    jobs,
    *,
    worker_count: int,
    protocol: TimingProtocol,
    shapes_per_candidate: int,
) -> list[dict[str, object]]:
    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(
                engine.dispatch,
                candidate,
                [
                    shape
                    for shape in engine.shapes
                    if candidate.tile_n in shape.selected_tile_ns
                ][:shapes_per_candidate],
                protocol=protocol,
                action="compile",
            )
            for engine, candidate in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_calibration(args) -> CalibrationData:
    worker_counts = _calibration_worker_counts(CALIBRATION_WORKER_COUNTS)
    points: list[CompileScalingPoint] = []
    shape_costs: list[ShapeGPUCost] = []
    peak_gpu_memory_bytes = 0
    with (
        tempfile.TemporaryDirectory(prefix="flashinfer-sweep-calibrate-") as root,
        ExitStack() as cleanup,
    ):
        # Compile jobs alternate FC1 and FC2, so each engine needs at most half
        # of the largest aggregate concurrency under test.
        per_fc_workers = max(1, (max(worker_counts) + 1) // 2)
        engines = []
        for fc in ("fc1", "fc2"):
            engine = SweepEngine(
                _sweep_options(
                    args,
                    out_dir=Path(root) / fc,
                    fc=fc,
                    workers=per_fc_workers,
                )
            )
            cleanup.callback(engine.close)
            engines.append(engine)
        candidate_sets = [list(engine.candidates.values()) for engine in engines]
        if any(not candidates for candidates in candidate_sets):
            raise RuntimeError("calibration found no FC1 or FC2 candidates")
        # Compilation caches are process-local.  Warm every persistent worker
        # with one reserved candidate per FC, repeating that candidate across
        # the workers for that FC.  The measured probes remain globally
        # distinct from the warmup candidates without consuming one unique
        # candidate per warmup worker.
        warmup_candidates = [
            (engine, candidates[-1])
            for engine, candidates in zip(engines, candidate_sets, strict=True)
        ]
        warmup_jobs = [job for _ in range(per_fc_workers) for job in warmup_candidates]
        measured_candidate_sets = [candidates[:-1] for candidates in candidate_sets]
        jobs = [
            (engine, candidates[index])
            for index in range(max(len(values) for values in measured_candidate_sets))
            for engine, candidates in zip(engines, measured_candidate_sets, strict=True)
            if index < len(candidates)
        ]
        fitted_worker_counts = _fit_calibration_worker_counts(
            worker_counts, available_jobs=len(jobs)
        )
        if fitted_worker_counts != worker_counts:
            print(
                "Calibration has "
                f"{len(jobs)} distinct measured candidates; reducing worker "
                f"probes from {worker_counts} to {fitted_worker_counts}.",
                file=sys.stderr,
                flush=True,
            )
            worker_counts = fitted_worker_counts
        protocol = TimingProtocol(
            warmup_iters=CALIBRATION_WARMUP_ITERS,
            repeat_iters=CALIBRATION_REPEAT_ITERS,
        )
        # Exercise every persistent process before timing.  The jobs alternate
        # FC family, draining every worker from both pools.
        warmup_results = _dispatch_calibration_jobs(
            warmup_jobs,
            worker_count=len(warmup_jobs),
            protocol=protocol,
            shapes_per_candidate=1,
        )
        warmup_failed = [
            result for result in warmup_results if result.get("status") != "ok"
        ]
        if warmup_failed:
            raise RuntimeError(
                "calibration worker warmup failed:\n"
                f"{_calibration_failure(warmup_failed)}"
            )
        job_offset = 0
        for point_index, count in enumerate(worker_counts):
            print(
                f"Calibration {point_index + 1}/{len(worker_counts)}: "
                f"testing {count} compiler worker(s)",
                flush=True,
            )
            # Keep GPU allocation and timing out of this interval: I/O setup
            # can take tens of seconds and would otherwise dominate what is
            # intended to be a compiler-throughput measurement.  Two jobs at
            # counts 1 and 2 ensure both FC families are represented.
            job_count = max(2, count)
            selected = jobs[job_offset : job_offset + job_count]
            job_offset += job_count
            started = time.perf_counter()
            results = _dispatch_calibration_jobs(
                selected,
                worker_count=count,
                protocol=protocol,
                shapes_per_candidate=1,
            )
            elapsed = time.perf_counter() - started
            successful = [result for result in results if result.get("status") == "ok"]
            failed = [result for result in results if result.get("status") != "ok"]
            failure = None if not failed else _calibration_failure(failed)
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes,
                max(
                    (
                        int(result.get("peak_gpu_memory_bytes") or 0)
                        for result in successful
                    ),
                    default=0,
                ),
            )
            points.append(
                CompileScalingPoint(
                    workers=count,
                    throughput_per_minute=60.0 * len(successful) / max(elapsed, 1e-9),
                    peak_rss_bytes=max(
                        (
                            int(result.get("peak_rss_bytes") or 0)
                            for result in successful
                        ),
                        default=0,
                    ),
                    failed=bool(failed),
                    failure=failure,
                    worker_gpu_footprints=tuple(
                        WorkerGPUFootprint.from_dict(footprint)
                        for result in successful
                        if isinstance(
                            (footprint := result.get("gpu_memory_footprint")),
                            dict,
                        )
                    ),
                )
            )
            print(
                f"  throughput={points[-1].throughput_per_minute:.2f}/minute, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            if failed:
                if not any(not point.failed for point in points[:-1]):
                    raise RuntimeError(
                        f"calibration failed with {count} worker(s):\n{failure}"
                    )
                print(
                    f"  stopping worker-count probes at the last good point; "
                    f"{count} worker(s) failed:\n{failure}",
                    file=sys.stderr,
                    flush=True,
                )
                break
        print("Calibration timing probes: measuring representative FC1/FC2 shapes")
        quiet_samples: list[float] = []
        evaluate_footprints: list[WorkerGPUFootprint] = []
        evaluate_footprints_supported = True
        for engine in engines:
            shape = engine.shapes[len(engine.shapes) // 2]
            candidate = next(
                candidate
                for candidate in engine.candidates.values()
                if candidate.tile_n in shape.selected_tile_ns
            )
            result = engine.dispatch(
                candidate,
                [shape],
                protocol=protocol,
                cache_tag=f"timing-{engine.options.fc}",
            )
            if result.get("status") != "ok":
                raise RuntimeError(
                    f"{engine.options.fc} timing probe failed:\n"
                    f"{_calibration_failure([result])}"
                )
            samples = _measurement_samples(result)
            if not quiet_samples:
                quiet_samples = samples
            peak_gpu_memory_bytes = max(
                peak_gpu_memory_bytes,
                int(result.get("peak_gpu_memory_bytes") or 0),
            )
            footprint = result.get("gpu_memory_footprint")
            if isinstance(footprint, dict):
                evaluate_footprints.append(WorkerGPUFootprint.from_dict(footprint))
            else:
                evaluate_footprints_supported = False
            for measurement in result.get("measurements", ()):
                if measurement.get("status") != "ok":
                    continue
                shape_costs.append(
                    ShapeGPUCost(
                        shape_key=f"{engine.options.fc}:{measurement['shape_id']}",
                        gpu_lock_seconds=(
                            float(result.get("lock_hold_ms") or 0.0) / 1000.0
                        ),
                    )
                )
    quiet_samples = quiet_samples or [1.0]
    noise = derive_noise_band(quiet_samples, relative_floor=0.005)
    chosen = select_compile_scaling_knee(points)
    chosen_index = next(
        index for index, point in enumerate(points) if point.workers == chosen
    )
    chosen_point = points[chosen_index]
    if evaluate_footprints_supported and evaluate_footprints:
        # Evaluation materializes representative IO and therefore captures the
        # memory a real sweep worker retains.  Older worker implementations may
        # omit this record; in that case retain the compile-only fallback.
        chosen_point = replace(
            chosen_point,
            worker_gpu_footprints=tuple(evaluate_footprints),
        )
        points[chosen_index] = chosen_point
    footprints = chosen_point.worker_gpu_footprints
    total_gpu_memory_bytes = (
        max(
            (int(footprint.total_device_bytes or 0) for footprint in footprints),
            default=0,
        )
        or None
    )
    max_worker_bytes = max(
        (footprint.effective_bytes for footprint in footprints),
        default=peak_gpu_memory_bytes,
    )
    calibration = CalibrationData(
        environment=_environment_fingerprint(str(args.gpu)),
        compile_scaling=points,
        worker_count=chosen,
        noise=noise,
        shape_costs=shape_costs,
        total_gpu_memory_bytes=total_gpu_memory_bytes,
        reference_reserve_bytes=2 * max_worker_bytes,
        gpu_memory_margin_bytes=(
            int(total_gpu_memory_bytes * 0.10)
            if total_gpu_memory_bytes is not None
            else 0
        ),
        metadata={"peak_gpu_memory_bytes": peak_gpu_memory_bytes},
    )
    return calibration


def _cmd_calibrate(args) -> int:
    calibration = _run_calibration(args)
    calibration.save(args.calibration_file)
    print(json.dumps(calibration.to_dict(), indent=2, sort_keys=True))
    return 0


def _load_or_calibrate(args) -> CalibrationData | None:
    if args.calibration_file is None:
        return None
    else:
        path = Path(args.calibration_file)
        if not path.exists():
            print(
                f"WARNING: calibration file {path} does not exist; "
                "continuing without calibration",
                file=sys.stderr,
            )
            return None
        calibration = CalibrationData.load(path)
    mismatches = calibration.validate_compatibility(
        _environment_fingerprint(str(args.gpu)), force=args.force_calibration
    )
    for mismatch in mismatches:
        print(f"WARNING: forced calibration mismatch: {mismatch}", file=sys.stderr)
    return calibration


def _cmd_sweep(args) -> int:
    if args.dry_run:
        calibration = None
        workers = args.workers or 1
    else:
        calibration = _load_or_calibrate(args)
        workers = args.workers or (
            calibration.recommended_worker_count()
            if calibration is not None
            else min(DEFAULT_UNCALIBRATED_WORKERS, available_cpu_count())
        )
        if calibration is None:
            print(
                f"Running without calibration using {workers} compiler worker(s).",
                file=sys.stderr,
            )
    options = _sweep_options(args, out_dir=args.out, fc=args.fc, workers=workers)
    projection = (
        None
        if calibration is None
        else next(
            (
                point.throughput_per_minute
                for point in calibration.compile_scaling
                if point.workers == workers
            ),
            None,
        )
    )
    if projection is not None:
        gpu_lock_seconds = sum(
            cost.gpu_lock_seconds
            for cost in calibration.shape_costs
            if cost.shape_key.startswith(f"{args.fc}:")
        )
        gpu_throughput = None if gpu_lock_seconds <= 0 else 60.0 / gpu_lock_seconds
        print(
            f"Projected compile throughput: {projection:.2f}/minute "
            f"with {workers} workers"
        )
        if gpu_throughput is not None:
            bottleneck = "GPU timing" if gpu_throughput < projection else "compilation"
            print(
                f"Projected GPU-lock cost: {gpu_lock_seconds:.3f}s/config "
                f"({gpu_throughput:.2f}/minute); bottleneck: {bottleneck}"
            )
    engine = SweepEngine(options, calibration=calibration)
    try:
        engine.run()
    finally:
        engine.close()
    return 0


def _cmd_worker_loop(_args) -> int:
    run_worker_loop()
    return 0


def _cmd_menu_retime(args) -> int:
    target = load_menu_target(args.config)
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", _CUDA_ARCH_ENV[target.sm_arch])
    calibration = _load_or_calibrate(args)
    workers = args.workers or (
        calibration.recommended_worker_count()
        if calibration is not None
        else min(DEFAULT_UNCALIBRATED_WORKERS, available_cpu_count())
    )
    candidates = collect_menu_candidates(
        target,
        args.runs,
        candidate_source=args.candidate_source,
    )
    matrix = run_menu_retime(
        target,
        candidates,
        out_dir=args.out,
        worker_count=workers,
        gpu=args.gpu,
        resume=args.resume,
        calibration=calibration,
        budget_minutes=args.budget_min,
    )
    print(matrix)
    return 0


def _cmd_menu_select(args) -> int:
    candidates, shapes, timings, cost, deployments = load_joint_inputs(args.retime_dir)
    manifest_path = args.retime_dir / "menu_manifest.json"
    input_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    selection = select_joint_menu(
        candidates,
        shapes,
        timings,
        cost=cost,
        deployments=deployments,
        order_by=args.order_by,
        floor_tile_ns=args.floor_tile_n,
        floor_plausible_ratio=args.floor_plausible_ratio,
    )
    order, report = write_menu_report(
        selection,
        out_dir=args.out or args.retime_dir,
        cost=cost,
        order_by=args.order_by,
        input_manifest=input_manifest,
    )
    print(order)
    print(report)
    return 0


def _cmd_menu_emit(args) -> int:
    print(
        emit_menu_prefix(
            args.order,
            n=args.n,
            base_path=args.base,
            out_path=args.out,
            dtype=args.dtype,
            activations=args.activations,
            in_place=args.in_place,
            dry_run=args.dry_run,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    _add_shape_args(calibrate)
    calibrate.add_argument(
        "--sm-arch",
        type=int,
        choices=(100, 103, 107),
        required=True,
        help="Target SM architecture.",
    )
    calibrate.add_argument(
        "--calibration-file",
        type=Path,
        required=True,
        help="Output calibration JSON path.",
    )
    calibrate.add_argument(
        "--gpu", default=DEFAULT_GPU, help="GPU index within CUDA_VISIBLE_DEVICES."
    )
    calibrate.set_defaults(func=_cmd_calibrate)

    sweep = subparsers.add_parser("sweep")
    _add_shape_args(sweep)
    sweep.add_argument(
        "--fc", choices=("fc1", "fc2"), required=True, help="FC layer to tune."
    )
    sweep.add_argument(
        "--sm-arch",
        type=int,
        choices=(100, 103, 107),
        required=True,
        help="Target SM architecture.",
    )
    sweep.add_argument(
        "--calibration-file",
        type=Path,
        help=(
            "Optional saved calibration. A missing file is not generated "
            "automatically; use the calibrate subcommand."
        ),
    )
    sweep.add_argument(
        "--out", type=Path, required=True, help="Sweep output directory."
    )
    sweep.add_argument(
        "--budget-min",
        type=float,
        default=DEFAULT_BUDGET_MINUTES,
        help="Total hard sweep budget in minutes.",
    )
    sweep.add_argument(
        "--tail-min",
        type=float,
        default=DEFAULT_TAIL_MINUTES,
        help="Minutes reserved for final retiming and verification.",
    )
    sweep.add_argument(
        "--menu-size",
        type=int,
        default=DEFAULT_MENU_SIZE,
        help="Maximum number of selected tactics.",
    )
    sweep.add_argument(
        "--max-near-best-per-shape",
        type=int,
        default=DEFAULT_MAX_NEAR_BEST_PER_SHAPE,
        help="Maximum 3%%-near-best candidates retained per shape for final re-timing.",
    )
    sweep.add_argument(
        "--reference-reserve-min",
        type=float,
        default=DEFAULT_REFERENCE_RESERVE_MINUTES,
        help="Minutes reserved at the end of the sweep for correctness checks.",
    )
    sweep.add_argument(
        "--workers",
        type=int,
        help="Compiler workers (default: calibrated value, otherwise min(8, nproc)).",
    )
    sweep.add_argument(
        "--gpu", default=DEFAULT_GPU, help="GPU index within CUDA_VISIBLE_DEVICES."
    )
    sweep.add_argument(
        "--final-tokens",
        type=int,
        nargs="+",
        help=(
            "Token counts used for final re-timing, verification and menu "
            "selection; defaults to the complete <=8192 list plus 12288/16384."
        ),
    )
    sweep.add_argument(
        "--search-routing",
        default=DEFAULT_SEARCH_ROUTING,
        help="Deterministic routing distribution used during adaptive search.",
    )
    sweep.add_argument(
        "--retime-routings",
        nargs="+",
        default=DEFAULT_RETIME_ROUTINGS,
        help="Routing distributions used for final re-timing.",
    )
    sweep.add_argument(
        "--selection-routings",
        nargs="+",
        help=(
            "Routing distributions used to select the final menu; defaults to "
            "all --retime-routings and must include --search-routing."
        ),
    )
    sweep.add_argument(
        "--max-routing-regret",
        type=float,
        default=DEFAULT_MAX_ROUTING_REGRET,
        help=(
            "Maximum allowed finalist-menu regret on any retime routing "
            "before the summary is marked unverified."
        ),
    )
    sweep.add_argument(
        "--enable-cluster-y",
        action="store_true",
        help=(
            "Explore experimental 2-CTA token-direction CGAs for weight "
            "multicast; intended for large or skewed routes."
        ),
    )
    sweep.add_argument(
        "--force-calibration",
        action="store_true",
        help="Use a calibration file despite environment mismatches.",
    )
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate generated candidates without compiling or timing.",
    )
    sweep.add_argument(
        "--resume",
        action="store_true",
        help="Continue from state in the output directory.",
    )
    sweep.add_argument(
        "--strict-resume",
        action="store_true",
        help="Reject resume when the Prims-TS source fingerprint changed.",
    )
    sweep.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry the latest timeout, worker error, and CUDA OOM search records.",
    )
    sweep.add_argument(
        "--confirm-fraction",
        type=float,
        default=DEFAULT_CONFIRM_FRACTION,
        help=(
            "Re-time search results that improve a shape by more than this "
            "fraction before using them for selection."
        ),
    )
    sweep.add_argument(
        "--max-input-stages",
        type=int,
        default=DEFAULT_MAX_INPUT_STAGES,
        help="Upper bound for input and input-scale pipeline stages.",
    )
    sweep.add_argument(
        "--max-accumulator-stages",
        type=int,
        default=DEFAULT_MAX_ACCUMULATOR_STAGES,
        help="Upper bound for accumulator TMEM stages.",
    )
    sweep.add_argument(
        "--max-c-smem-stages",
        type=int,
        default=DEFAULT_MAX_C_SMEM_STAGES,
        help="Upper bound for output shared-memory stages.",
    )
    sweep.add_argument(
        "--max-workid-stages",
        type=int,
        default=DEFAULT_MAX_WORKID_STAGES,
        help="Upper bound for persistent scheduler work-ID stages.",
    )
    sweep.add_argument(
        "--max-sf-ring-ratio",
        type=int,
        default=DEFAULT_MAX_SF_RING_RATIO,
        help="Maximum scale-factor ring depth relative to operand stages.",
    )
    sweep.set_defaults(func=_cmd_sweep)

    menu = subparsers.add_parser(
        "menu", help="Re-time, select, and emit a cross-model tactic menu."
    )
    menu_commands = menu.add_subparsers(dest="menu_command", required=True)

    menu_retime = menu_commands.add_parser("retime")
    menu_retime.add_argument("--config", type=Path, required=True)
    menu_retime.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "FC1/FC2 sweep output directories contributing candidates; omit to "
            "re-time only the shipped incumbents."
        ),
    )
    menu_retime.add_argument("--out", type=Path, required=True)
    menu_retime.add_argument(
        "--candidate-source",
        choices=("selected", "pool"),
        default="selected",
        help="Use each sweep's selected menu or its complete final re-time pool.",
    )
    menu_retime.add_argument("--workers", type=int)
    menu_retime.add_argument(
        "--gpu", default=DEFAULT_GPU, help="GPU index within CUDA_VISIBLE_DEVICES."
    )
    menu_retime.add_argument("--calibration-file", type=Path)
    menu_retime.add_argument("--force-calibration", action="store_true")
    menu_retime.add_argument("--resume", action="store_true")
    menu_retime.add_argument(
        "--budget-min",
        type=float,
        default=1440.0,
        help="Hard budget for each model/FC sub-run.",
    )
    menu_retime.set_defaults(func=_cmd_menu_retime)

    menu_select = menu_commands.add_parser("select")
    menu_select.add_argument("--retime-dir", type=Path, required=True)
    menu_select.add_argument("--out", type=Path)
    menu_select.add_argument(
        "--order-by",
        choices=("cost", "gain"),
        default="cost",
        help="Rank marginal objective gain per cost or by absolute gain.",
    )
    menu_select.add_argument(
        "--floor-tile-n",
        type=int,
        action="append",
        default=[],
        help=(
            "Guarantee the best plausible FC1+FC2 pair for this tile_n in the "
            "mandatory floor; repeat for multiple tiles."
        ),
    )
    menu_select.add_argument(
        "--floor-plausible-ratio",
        type=float,
        default=1.25,
        help=(
            "Score extra floor pairs on shapes where that tile is within this "
            "ratio of the globally best pair."
        ),
    )
    menu_select.set_defaults(func=_cmd_menu_select)

    menu_emit = menu_commands.add_parser("emit")
    menu_emit.add_argument("--order", type=Path, required=True)
    menu_emit.add_argument("--n", type=int, required=True)
    menu_emit.add_argument("--base", type=Path, required=True)
    emit_destination = menu_emit.add_mutually_exclusive_group(required=True)
    emit_destination.add_argument("--out", type=Path)
    emit_destination.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace --base after the emitted menu verifies.",
    )
    menu_emit.add_argument("--dtype", required=True)
    menu_emit.add_argument(
        "--activations",
        choices=("auto", "per-class", "shared"),
        default="auto",
        help=(
            "Apply FC1 geometries per measured activation class or to all "
            "base activations. Auto uses per-class when the prefix contains "
            "elementwise entries, otherwise shared."
        ),
    )
    menu_emit.add_argument(
        "--dry-run",
        action="store_true",
        help="Build, verify, and summarize the menu without writing it.",
    )
    menu_emit.set_defaults(func=_cmd_menu_emit)

    worker_loop = subparsers.add_parser("worker-loop", help=argparse.SUPPRESS)
    worker_loop.set_defaults(func=_cmd_worker_loop)
    return parser


def main(argv: list[str] | None = None) -> int:
    # The task-scheduling builder emits this for every candidate; in worker
    # logs it buries the actual failure in timeout and crash records.
    warnings.filterwarnings(
        "ignore",
        message=r".*is never routed to a downstream work call",
        category=UserWarning,
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("calibrate", "sweep"):
        _validate_model_args(parser, args)
        _validate_parallelism_args(parser, args)
    if args.command in ("calibrate", "sweep"):
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", _CUDA_ARCH_ENV[args.sm_arch])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
