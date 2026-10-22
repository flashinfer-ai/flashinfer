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

"""Compile-once, measure-many execution for Prims-TS MoE sweep candidates."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import statistics
import time
from typing import Any, Callable, Iterable, Mapping

from .defaults import ROUTING_SEED, SEARCH_REPEAT_ITERS, SEARCH_WARMUP_ITERS
from .space import ConfigContext, MaterializedCandidate, is_applicable
from .utils import nvml_handle_for_uuid


_NVML_MODULE = None
_NVML_HANDLE = None
_NVML_UNAVAILABLE = False


@dataclass(frozen=True)
class TimingProtocol:
    warmup_iters: int = SEARCH_WARMUP_ITERS
    repeat_iters: int = SEARCH_REPEAT_ITERS
    enable_cupti: bool = True
    use_cuda_graph: bool = True
    cold_l2_cache: bool = True
    num_rotated_buffers: int | None = None
    timing_backend: str | None = None
    protocol_version: int = 1


def resolve_timing_backend(
    *, enable_cupti: bool = True, use_cuda_graph: bool = True
) -> str:
    """Resolve one timing backend for the entire run."""
    if enable_cupti:
        try:
            from cupti import cupti as _cupti  # noqa: F401

            if int(version("cupti-python").split(".")[0]) >= 13:
                return "cupti"
        except (ImportError, PackageNotFoundError, RuntimeError, ValueError):
            pass
    return "cutlass_testing_graph" if use_cuda_graph else "cutlass_testing_event"


def final_timing_protocol(search: TimingProtocol) -> TimingProtocol:
    """Return the protocol used to rank the final menu.

    Doubles the search sample count, with a floor of 60. That is enough for a
    stable median. A 200-iteration cold-L2 pass holds the single GPU lock for
    too long to finish the retime pool inside a normal tail budget.
    """

    return TimingProtocol(
        warmup_iters=search.warmup_iters,
        repeat_iters=max(60, search.repeat_iters * 2),
        enable_cupti=search.enable_cupti,
        use_cuda_graph=search.use_cuda_graph,
        cold_l2_cache=search.cold_l2_cache,
        num_rotated_buffers=search.num_rotated_buffers,
        timing_backend=search.timing_backend,
        protocol_version=search.protocol_version,
    )


@dataclass(frozen=True)
class ShapeMeasurement:
    shape_id: str
    status: str
    median_us: float | None = None
    mad_us: float | None = None
    min_us: float | None = None
    max_us: float | None = None
    samples_us: tuple[float, ...] = ()
    timing_backend: str | None = None
    resolved_scale_c: float | None = None
    error: str | None = None
    io_build_ms: float | None = None
    sm_clock_mhz: int | None = None
    sm_clock_max_mhz: int | None = None
    clock_event_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    candidate_id: str
    status: str
    compile_ms: float
    lock_wait_ms: float
    lock_hold_ms: float
    measurements: tuple[ShapeMeasurement, ...]
    error: str | None
    protocol: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "compile_ms": self.compile_ms,
            "lock_wait_ms": self.lock_wait_ms,
            "lock_hold_ms": self.lock_hold_ms,
            "measurements": [asdict(value) for value in self.measurements],
            "error": self.error,
            "measurement_protocol": dict(self.protocol),
        }


def _shape_id(shape: object) -> str:
    value = getattr(shape, "shape_id", None)
    if value is not None:
        return str(value)
    value = getattr(shape, "id", None)
    if value is not None:
        return str(value)
    return repr(shape)


def _problem_dimension(shape: object, fc: str, dimension: str) -> int:
    return int(getattr(shape, f"{fc}_{dimension}"))


def _routing_kwargs(shape: object) -> dict[str, object]:
    return {
        "routing": str(getattr(shape, "routing", "balanced_round_robin")),
        "routing_seed": int(getattr(shape, "routing_seed", ROUTING_SEED)),
    }


def _activation_value_kwargs(context: ConfigContext) -> dict[str, float]:
    """Return only enabled activation values, preserving harness defaults."""

    return {
        name: float(value)
        for name, value in (
            ("gemm1_alpha_value", context.gemm1_alpha_value),
            ("gemm1_beta_value", context.gemm1_beta_value),
            ("gemm1_clamp_limit_value", context.gemm1_clamp_limit_value),
        )
        if value is not None
    }


def _mad(samples: Iterable[float], median: float) -> float:
    values = [abs(value - median) for value in samples]
    return statistics.median(values) if values else 0.0


def _gpu_clock_sample() -> tuple[int | None, int | None, tuple[str, ...]]:
    """Read SM clocks from NVML, caching the device handle per worker."""

    global _NVML_HANDLE, _NVML_MODULE, _NVML_UNAVAILABLE
    if _NVML_UNAVAILABLE:
        return None, None, ()
    try:
        if _NVML_MODULE is None:
            import pynvml
            import torch

            pynvml.nvmlInit()
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            uuid = str(properties.uuid)
            if not uuid.startswith("GPU-"):
                uuid = f"GPU-{uuid}"
            _NVML_HANDLE = nvml_handle_for_uuid(pynvml, uuid)
            _NVML_MODULE = pynvml
        assert _NVML_MODULE is not None and _NVML_HANDLE is not None
        nvml = _NVML_MODULE
        current = int(nvml.nvmlDeviceGetClockInfo(_NVML_HANDLE, nvml.NVML_CLOCK_SM))
        maximum = int(nvml.nvmlDeviceGetMaxClockInfo(_NVML_HANDLE, nvml.NVML_CLOCK_SM))
        reason_bits = 0
        for getter_name in (
            "nvmlDeviceGetCurrentClocksEventReasons",
            "nvmlDeviceGetCurrentClocksThrottleReasons",
        ):
            getter = getattr(nvml, getter_name, None)
            if getter is not None:
                reason_bits = int(getter(_NVML_HANDLE))
                break
        reasons = []
        for name in dir(nvml):
            if not (
                name.startswith("nvmlClocksEventReason")
                or name.startswith("nvmlClocksThrottleReason")
            ):
                continue
            value = getattr(nvml, name)
            if isinstance(value, int) and value and reason_bits & value:
                reasons.append(name)
        return current, maximum, tuple(sorted(set(reasons)))
    except Exception:
        _NVML_UNAVAILABLE = True
        return None, None, ()


def _time_with_flashinfer(
    compiled_fn,
    io: dict,
    stream,
    protocol: TimingProtocol,
) -> tuple[list[float], str]:
    """Prefer CUPTI/CUDA events; fall back to the established CuTe timer."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _launch_arg_tuple,
        benchmark_compiled,
    )

    backend = protocol.timing_backend or resolve_timing_backend(
        enable_cupti=protocol.enable_cupti,
        use_cuda_graph=protocol.use_cuda_graph,
    )
    if backend == "cupti":
        import cuda.bindings.driver as cuda_drv
        import torch

        from flashinfer.testing import bench_gpu_time

        def launch() -> None:
            # CUPTI does not need graph replay to remove CPU launch overhead.
            # Constructing the driver stream here also keeps the launch on
            # the stream that the timing utility made current.
            current_stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled_fn(*_launch_arg_tuple(io, current_stream))

        samples_ms = bench_gpu_time(
            launch,
            dry_run_iters=protocol.warmup_iters,
            repeat_iters=protocol.repeat_iters,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=protocol.cold_l2_cache,
        )
        return [float(value) * 1000.0 for value in samples_ms], "cupti"
    if backend not in ("cutlass_testing_graph", "cutlass_testing_event"):
        raise ValueError(f"unknown timing backend {backend!r}")

    batch_count = min(5, protocol.repeat_iters)
    base_iters, extra = divmod(protocol.repeat_iters, batch_count)
    elapsed_samples = [
        benchmark_compiled(
            compiled_fn,
            io,
            stream,
            warmup_iters=protocol.warmup_iters if batch_index == 0 else 1,
            bench_iters=base_iters + (batch_index < extra),
            num_rotated_buffers=(
                protocol.num_rotated_buffers if protocol.cold_l2_cache else 0
            ),
            use_cuda_graphs=protocol.use_cuda_graph,
        )
        for batch_index in range(batch_count)
    ]
    observed_backend = (
        "cutlass_testing_graph" if protocol.use_cuda_graph else "cutlass_testing_event"
    )
    if observed_backend != backend:
        raise RuntimeError(
            f"timing backend changed: expected {backend}, got {observed_backend}"
        )
    return [float(value) for value in elapsed_samples], observed_backend


def evaluate_candidate(
    candidate: MaterializedCandidate,
    context: ConfigContext,
    shapes: Iterable[object],
    protocol: TimingProtocol,
    *,
    weight_cache: dict[tuple[object, ...], tuple[object, object]] | None = None,
    gpu_lock_identity: str | int | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> Evaluation:
    """Compile each resolved K signature once and time every applicable shape."""
    import cuda.bindings.driver as cuda_drv
    import torch

    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _build_launch_io
    from flashinfer.prims_ts.moe.compile_cache import (
        get_compiled_gemm,
        stable_config_hash,
    )

    shape_list = list(shapes)
    measurements: list[ShapeMeasurement] = []
    compile_ms = 0.0
    lock_wait_ms = 0.0
    lock_hold_ms = 0.0
    resolved_lock_identity = gpu_lock_identity
    if gpu_lock_identity is not None:
        from .workers import resolve_gpu_lock_identity

        resolved_lock_identity = resolve_gpu_lock_identity(gpu_lock_identity)
    groups: dict[int, list[object]] = {}
    for shape in shape_list:
        applicable, reason = is_applicable(candidate, context, shape)
        if not applicable:
            measurements.append(
                ShapeMeasurement(
                    shape_id=_shape_id(shape),
                    status="not_applicable",
                    error=reason,
                )
            )
            continue
        problem_k = _problem_dimension(shape, context.fc, "k")
        groups.setdefault(problem_k, []).append(shape)

    timed_shape_count = sum(len(group_shapes) for group_shapes in groups.values())
    timed_shape_index = 0
    torch_stream = torch.cuda.Stream()
    stream = cuda_drv.CUstream(torch_stream.cuda_stream)
    try:
        with torch.cuda.stream(torch_stream):
            for problem_k, group_shapes in groups.items():
                if progress is not None:
                    progress("compile", f"K={problem_k}")
                expected_hash = candidate.resolved_hashes[problem_k]
                first = group_shapes[0]
                prep_lock = nullcontext()
                if resolved_lock_identity is not None:
                    from .workers import gpu_lock

                    prep_lock = gpu_lock(resolved_lock_identity)
                with prep_lock:
                    small_io = _build_launch_io(
                        num_experts=max(
                            context.top_k, min(context.num_local_experts, 2)
                        ),
                        num_tokens=max(context.top_k, 4),
                        top_k=context.top_k,
                        problem_n=_problem_dimension(first, context.fc, "n"),
                        problem_k=problem_k,
                        **_activation_value_kwargs(context),
                        **dict(candidate.kwargs),
                    )
                actual_hash = stable_config_hash(small_io["cfg"])
                if actual_hash != expected_hash:
                    raise AssertionError(
                        "compile signature mismatch for K="
                        f"{problem_k}: expected {expected_hash}, got {actual_hash}"
                    )
                started = time.perf_counter()
                compiled_fn = get_compiled_gemm(
                    expected_hash, context.fc, small_io, stream
                )
                torch.cuda.synchronize()
                compile_ms += (time.perf_counter() - started) * 1000.0

                lock = nullcontext()
                if resolved_lock_identity is not None:
                    from .workers import gpu_lock

                    lock = gpu_lock(resolved_lock_identity)
                if progress is not None:
                    progress("lock_wait", f"K={problem_k}")
                lock_started = time.perf_counter()
                with lock:
                    lock_wait_ms += (time.perf_counter() - lock_started) * 1000.0
                    lock_hold_started = time.perf_counter()
                    for shape in group_shapes:
                        shape_attrs: Any = shape
                        shape_id = _shape_id(shape)
                        timed_shape_index += 1
                        if progress is not None:
                            progress(
                                "run",
                                f"{timed_shape_index}/{timed_shape_count} {shape_id}",
                            )
                        started = time.perf_counter()
                        io = _build_launch_io(
                            num_experts=int(shape_attrs.local_num_experts),
                            num_tokens=int(shape_attrs.timing_num_tokens),
                            top_k=int(shape_attrs.top_k),
                            problem_n=_problem_dimension(shape, context.fc, "n"),
                            problem_k=problem_k,
                            **_routing_kwargs(shape),
                            **_activation_value_kwargs(context),
                            weight_cache=weight_cache,
                            **dict(candidate.kwargs),
                        )
                        io_build_ms = (time.perf_counter() - started) * 1000.0
                        if stable_config_hash(io["cfg"]) != expected_hash:
                            raise AssertionError(
                                f"shape {shape_id} resolved to a different "
                                "compile signature"
                            )
                        try:
                            samples, backend = _time_with_flashinfer(
                                compiled_fn, io, stream, protocol
                            )
                            sm_clock, sm_clock_max, clock_reasons = _gpu_clock_sample()
                            median = statistics.median(samples)
                            measurements.append(
                                ShapeMeasurement(
                                    shape_id=shape_id,
                                    status="ok",
                                    median_us=median,
                                    mad_us=_mad(samples, median),
                                    min_us=min(samples),
                                    max_us=max(samples),
                                    samples_us=tuple(samples),
                                    timing_backend=backend,
                                    resolved_scale_c=float(io["resolved_scale_c"]),
                                    io_build_ms=io_build_ms,
                                    sm_clock_mhz=sm_clock,
                                    sm_clock_max_mhz=sm_clock_max,
                                    clock_event_reasons=clock_reasons,
                                )
                            )
                        except Exception as exc:
                            measurements.append(
                                ShapeMeasurement(
                                    shape_id=shape_id,
                                    status="run_error",
                                    error=f"{type(exc).__name__}: {exc}",
                                    io_build_ms=io_build_ms,
                                )
                            )
                    lock_hold_ms += (time.perf_counter() - lock_hold_started) * 1000.0
    except Exception as exc:
        return Evaluation(
            candidate_id=candidate.candidate.candidate_id,
            status="error",
            compile_ms=compile_ms,
            lock_wait_ms=lock_wait_ms,
            lock_hold_ms=lock_hold_ms,
            measurements=tuple(measurements),
            error=f"{type(exc).__name__}: {exc}",
            protocol=asdict(protocol),
        )

    status = (
        "ok"
        if any(measurement.status == "ok" for measurement in measurements)
        else "error"
    )
    return Evaluation(
        candidate_id=candidate.candidate.candidate_id,
        status=status,
        compile_ms=compile_ms,
        lock_wait_ms=lock_wait_ms,
        lock_hold_ms=lock_hold_ms,
        measurements=tuple(measurements),
        error=None,
        protocol=asdict(protocol),
    )


def compile_candidate(
    candidate: MaterializedCandidate,
    context: ConfigContext,
    shapes: Iterable[object],
    *,
    gpu_lock_identity: str | int | None = None,
) -> float:
    """Compile every resolved K specialization using minimal launch IO."""
    import cuda.bindings.driver as cuda_drv
    import torch

    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _build_launch_io
    from flashinfer.prims_ts.moe.compile_cache import (
        get_compiled_gemm,
        stable_config_hash,
    )

    first_by_k: dict[int, object] = {}
    for shape in shapes:
        applicable, _ = is_applicable(candidate, context, shape)
        if applicable:
            first_by_k.setdefault(_problem_dimension(shape, context.fc, "k"), shape)
    torch_stream = torch.cuda.Stream()
    stream = cuda_drv.CUstream(torch_stream.cuda_stream)
    compile_ms = 0.0
    resolved_lock_identity = gpu_lock_identity
    if gpu_lock_identity is not None:
        from .workers import resolve_gpu_lock_identity

        resolved_lock_identity = resolve_gpu_lock_identity(gpu_lock_identity)
    with torch.cuda.stream(torch_stream):
        for problem_k, shape in first_by_k.items():
            expected_hash = candidate.resolved_hashes[problem_k]
            prep_lock = nullcontext()
            if resolved_lock_identity is not None:
                from .workers import gpu_lock

                prep_lock = gpu_lock(resolved_lock_identity)
            with prep_lock:
                small_io = _build_launch_io(
                    num_experts=max(context.top_k, min(context.num_local_experts, 2)),
                    num_tokens=max(context.top_k, 4),
                    top_k=context.top_k,
                    problem_n=_problem_dimension(shape, context.fc, "n"),
                    problem_k=problem_k,
                    **_activation_value_kwargs(context),
                    **dict(candidate.kwargs),
                )
            if stable_config_hash(small_io["cfg"]) != expected_hash:
                raise AssertionError(f"compile signature mismatch for K={problem_k}")
            started = time.perf_counter()
            get_compiled_gemm(expected_hash, context.fc, small_io, stream)
            torch.cuda.synchronize()
            compile_ms += (time.perf_counter() - started) * 1000.0
    return compile_ms


def reference_check_candidate(
    candidate: MaterializedCandidate,
    context: ConfigContext,
    shape: object,
) -> object:
    """Run the authoritative randomized-scale-factor reference check."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    applicable, _ = is_applicable(candidate, context, shape)
    if not applicable:
        return False
    shape_attrs: Any = shape
    return reference_check(
        num_experts=int(shape_attrs.local_num_experts),
        num_tokens=int(shape_attrs.timing_num_tokens),
        top_k=int(shape_attrs.top_k),
        problem_n=_problem_dimension(shape, context.fc, "n"),
        problem_k=_problem_dimension(shape, context.fc, "k"),
        **_routing_kwargs(shape),
        **_activation_value_kwargs(context),
        **dict(candidate.kwargs),
    )
