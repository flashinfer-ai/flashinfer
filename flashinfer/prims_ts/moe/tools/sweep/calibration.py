# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Versioned, host-safe calibration records for Prims-TS MoE sweeps."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .db import atomic_write_json, canonical_json, deterministic_id
from .utils import nvml_handle_for_uuid


CALIBRATION_SCHEMA_VERSION = 3
CALIBRATION_PROTOCOL_VERSION = "prims-ts-moe-sweep-v1"
MAX_CALIBRATION_WORKERS = 16
_RESULTS_V5_SCHEMA_VERSION = 2
_RESULTS_V5_SCHEMA_FINGERPRINT = (
    "b044e7780d7a32567cfc8c1049294ccf99a75d80bdf6c761fffd680457bb3ffd"
)
_OPTIONAL_ENVIRONMENT_FIELDS = frozenset(
    {
        "sm_count",
        "clock_rate_khz",
        "memory_clock_rate_khz",
        "compute_capability",
    }
)
_SCHEMA_FIELDS = {
    "environment": (
        "gpu",
        "driver",
        "cuda",
        "cute_dsl",
        "cpu",
        "flashinfer_sha",
        "protocol",
        "sm_count",
        "clock_rate_khz",
        "memory_clock_rate_khz",
        "compute_capability",
    ),
    "compile_scaling": (
        "workers",
        "throughput_per_minute",
        "peak_rss_bytes",
        "failed",
        "failure",
        "worker_gpu_footprints",
    ),
    "noise_band": (
        "median",
        "mad",
        "lower",
        "upper",
        "absolute",
        "relative",
        "sample_count",
    ),
    "shape_cost": (
        "shape_key",
        "gpu_lock_seconds",
    ),
    "worker_gpu_footprint": (
        "allocated_bytes",
        "reserved_bytes",
        "nvml_process_bytes",
        "total_device_bytes",
    ),
    "calibration": (
        "total_gpu_memory_bytes",
        "reference_reserve_bytes",
        "gpu_memory_margin_bytes",
    ),
}
CALIBRATION_SCHEMA_FINGERPRINT = deterministic_id(
    {"version": CALIBRATION_SCHEMA_VERSION, "fields": _SCHEMA_FIELDS}
)
# Kept in the record for provenance. A later checkout on the same host
# still reuses the worker-count and noise measurements.
_UNCHECKED_ENVIRONMENT_FIELDS = frozenset({"flashinfer_sha"})


class CalibrationCompatibilityError(ValueError):
    """Raised when calibration was measured in an incompatible environment."""

    def __init__(self, mismatches: Sequence[str]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__(
            "incompatible calibration: " + "; ".join(self.mismatches)
            if self.mismatches
            else "incompatible calibration"
        )


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Inputs that can change compilation, legality, or measured performance."""

    gpu: str
    driver: str
    cuda: str
    cute_dsl: str
    cpu: str
    flashinfer_sha: str
    protocol: str = CALIBRATION_PROTOCOL_VERSION
    sm_count: int | None = None
    clock_rate_khz: int | None = None
    memory_clock_rate_khz: int | None = None
    compute_capability: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnvironmentFingerprint:
        missing = [
            name
            for name in _SCHEMA_FIELDS["environment"]
            if name not in _OPTIONAL_ENVIRONMENT_FIELDS and name not in value
        ]
        if missing:
            raise ValueError(
                "calibration environment is missing: " + ", ".join(missing)
            )
        return cls(
            gpu=str(value["gpu"]),
            driver=str(value["driver"]),
            cuda=str(value["cuda"]),
            cute_dsl=str(value["cute_dsl"]),
            cpu=str(value["cpu"]),
            flashinfer_sha=str(value["flashinfer_sha"]),
            protocol=str(value["protocol"]),
            sm_count=(
                None if value.get("sm_count") is None else int(value["sm_count"])
            ),
            clock_rate_khz=(
                None
                if value.get("clock_rate_khz") is None
                else int(value["clock_rate_khz"])
            ),
            memory_clock_rate_khz=(
                None
                if value.get("memory_clock_rate_khz") is None
                else int(value["memory_clock_rate_khz"])
            ),
            compute_capability=(
                None
                if value.get("compute_capability") is None
                else str(value["compute_capability"])
            ),
        )

    @property
    def digest(self) -> str:
        return deterministic_id(self.to_dict())


@dataclass(frozen=True)
class WorkerGPUFootprint:
    """Observed memory held by one worker after building its largest shape."""

    allocated_bytes: int
    reserved_bytes: int
    nvml_process_bytes: int | None = None
    total_device_bytes: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.allocated_bytes,
            self.reserved_bytes,
            self.nvml_process_bytes,
            self.total_device_bytes,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("GPU footprint values must be non-negative")

    @property
    def effective_bytes(self) -> int:
        """Conservative non-double-counted worker footprint."""

        return max(
            self.allocated_bytes,
            self.reserved_bytes,
            self.nvml_process_bytes or 0,
        )

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerGPUFootprint:
        return cls(
            allocated_bytes=int(value.get("allocated_bytes", 0)),
            reserved_bytes=int(value.get("reserved_bytes", 0)),
            nvml_process_bytes=(
                None
                if value.get("nvml_process_bytes") is None
                else int(value["nvml_process_bytes"])
            ),
            total_device_bytes=(
                None
                if value.get("total_device_bytes") is None
                else int(value["total_device_bytes"])
            ),
        )


def measure_worker_gpu_footprint(
    device: object = None,
    *,
    torch_module: Any | None = None,
    nvml_module: Any | None = None,
    pid: int | None = None,
) -> WorkerGPUFootprint:
    """Measure allocator and process-visible GPU memory for this worker.

    NVML is optional. Its process usage includes allocator reservations, so
    callers should use :attr:`WorkerGPUFootprint.effective_bytes` rather than
    adding the fields and double-counting the same memory.
    """

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to measure GPU memory") from exc
    cuda = torch_module.cuda
    resolved_device = cuda.current_device() if device is None else device
    allocated = int(cuda.memory_allocated(resolved_device))
    reserved = int(cuda.memory_reserved(resolved_device))
    properties = cuda.get_device_properties(resolved_device)
    total = int(properties.total_memory)
    process_bytes = _nvml_process_memory(
        resolved_device,
        properties,
        nvml_module=nvml_module,
        pid=os.getpid() if pid is None else pid,
    )
    return WorkerGPUFootprint(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        nvml_process_bytes=process_bytes,
        total_device_bytes=total,
    )


def _nvml_process_memory(
    device: object,
    properties: object,
    *,
    nvml_module: Any | None,
    pid: int,
) -> int | None:
    if nvml_module is None:
        try:
            import pynvml as nvml_module
        except ImportError:
            return None
    initialized = False
    try:
        nvml_module.nvmlInit()
        initialized = True
        uuid = getattr(properties, "uuid", None)
        handle = nvml_handle_for_uuid(
            nvml_module, "" if uuid is None else uuid, index=int(device)
        )
        processes: list[Any] = []
        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            getter = getattr(nvml_module, getter_name, None)
            if getter is None:
                continue
            try:
                processes.extend(getter(handle))
            except Exception:
                continue
        usages = {
            int(process.usedGpuMemory)
            for process in processes
            if int(getattr(process, "pid", -1)) == pid
            and isinstance(getattr(process, "usedGpuMemory", None), int)
            and int(process.usedGpuMemory) >= 0
            and int(process.usedGpuMemory) < (1 << 63)
        }
        return max(usages) if usages else None
    except Exception:
        return None
    finally:
        if initialized:
            with suppress(Exception):
                nvml_module.nvmlShutdown()


def derive_safe_worker_cap(
    total_gpu_memory_bytes: int,
    footprints: Iterable[WorkerGPUFootprint | int],
    *,
    reference_reserve_bytes: int,
    margin_bytes: int,
    maximum_workers: int | None = None,
) -> int:
    """Derive a worker cap from usable device memory and measured workers."""

    if min(total_gpu_memory_bytes, reference_reserve_bytes, margin_bytes) < 0:
        raise ValueError("GPU memory inputs must be non-negative")
    footprint_bytes = [
        (
            footprint.effective_bytes
            if isinstance(footprint, WorkerGPUFootprint)
            else int(footprint)
        )
        for footprint in footprints
    ]
    if not footprint_bytes or any(value <= 0 for value in footprint_bytes):
        raise ValueError("at least one positive worker footprint is required")
    usable = total_gpu_memory_bytes - reference_reserve_bytes - margin_bytes
    cap = usable // max(footprint_bytes)
    if maximum_workers is not None:
        if maximum_workers < 1:
            raise ValueError("maximum_workers must be positive")
        cap = min(cap, maximum_workers)
    if cap < 1:
        raise ValueError("GPU reserve and margin leave no room for one worker")
    return int(cap)


@dataclass(frozen=True)
class CompileScalingPoint:
    workers: int
    throughput_per_minute: float
    peak_rss_bytes: int | None = None
    failed: bool = False
    failure: str | None = None
    worker_gpu_footprints: tuple[WorkerGPUFootprint, ...] = ()

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.throughput_per_minute < 0 or not math.isfinite(
            self.throughput_per_minute
        ):
            raise ValueError("throughput_per_minute must be finite and non-negative")
        if self.peak_rss_bytes is not None and self.peak_rss_bytes < 0:
            raise ValueError("peak_rss_bytes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["worker_gpu_footprints"] = [
            footprint.to_dict() for footprint in self.worker_gpu_footprints
        ]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompileScalingPoint:
        return cls(
            workers=int(value["workers"]),
            throughput_per_minute=float(value["throughput_per_minute"]),
            peak_rss_bytes=(
                None
                if value.get("peak_rss_bytes") is None
                else int(value["peak_rss_bytes"])
            ),
            failed=bool(value.get("failed", False)),
            failure=(None if value.get("failure") is None else str(value["failure"])),
            worker_gpu_footprints=tuple(
                WorkerGPUFootprint.from_dict(footprint)
                for footprint in value.get("worker_gpu_footprints", ())
            ),
        )


def select_compile_scaling_knee(
    points: Iterable[CompileScalingPoint],
    *,
    minimum_gain: float = 0.10,
    max_peak_rss_bytes: int | None = None,
) -> int:
    """Select the smallest worker count near the best measured throughput.

    The valid curve ends at the first failed or memory-exceeding point.  Within
    that curve, select the smallest worker count whose aggregate throughput is
    at least ``1 - minimum_gain`` times the best measured throughput.  This
    avoids letting one noisy intermediate measurement terminate calibration
    early.  Measurements need not be powers of two, though calibration
    normally uses 1, 2, 4, ... workers.
    """

    if minimum_gain < 0:
        raise ValueError("minimum_gain must be non-negative")
    ordered = sorted(points, key=lambda point: point.workers)
    if not ordered:
        raise ValueError("at least one compile scaling point is required")
    if len({point.workers for point in ordered}) != len(ordered):
        raise ValueError("compile scaling worker counts must be unique")

    usable: list[CompileScalingPoint] = []
    for point in ordered:
        memory_exceeded = (
            max_peak_rss_bytes is not None
            and point.peak_rss_bytes is not None
            and point.peak_rss_bytes > max_peak_rss_bytes
        )
        if point.failed or memory_exceeded:
            if not usable:
                reason = point.failure or "memory limit exceeded"
                raise ValueError(f"the first compile scaling point failed: {reason}")
            break
        usable.append(point)

    best_throughput = max(point.throughput_per_minute for point in usable)
    threshold = (1.0 - minimum_gain) * best_throughput
    return next(
        point.workers for point in usable if point.throughput_per_minute >= threshold
    )


@dataclass(frozen=True)
class NoiseBand:
    median: float
    mad: float
    lower: float
    upper: float
    absolute: float
    relative: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NoiseBand:
        return cls(
            median=float(value["median"]),
            mad=float(value["mad"]),
            lower=float(value["lower"]),
            upper=float(value["upper"]),
            absolute=float(value["absolute"]),
            relative=float(value["relative"]),
            sample_count=int(value["sample_count"]),
        )


def _linear_percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def derive_noise_band(
    samples: Iterable[float],
    *,
    confidence: float = 0.90,
    mad_multiplier: float = 3.0,
    relative_floor: float = 0.0,
) -> NoiseBand:
    """Derive a robust symmetric comparison band from repeated timings.

    The half-width is the larger of a trimmed central percentile envelope,
    scaled MADs, and an optional relative floor.  Trimming one observation
    from each tail keeps a single small-sample outlier from setting the whole
    band; MAD and the relative floor retain useful width after trimming.
    """

    values = sorted(float(sample) for sample in samples)
    if not values:
        raise ValueError("at least one timing sample is required")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timing samples must be finite and positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if mad_multiplier < 0 or relative_floor < 0:
        raise ValueError("noise band multipliers must be non-negative")

    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    # A linearly interpolated high percentile still gives almost all of its
    # weight to an isolated maximum when the sample is small (for example,
    # p95 over five observations).  Trim one observation from each tail for
    # small samples; with at least 20 observations the percentile itself has
    # enough support.  Do not trim pairs: two samples contain no interior
    # observations.
    percentile_values = values[1:-1] if 3 <= len(values) < 20 else values
    tail = (1.0 - confidence) / 2.0
    low_quantile = _linear_percentile(percentile_values, tail)
    high_quantile = _linear_percentile(percentile_values, 1.0 - tail)
    percentile_width = max(median - low_quantile, high_quantile - median)
    robust_width = mad_multiplier * 1.4826 * mad
    absolute = max(percentile_width, robust_width, relative_floor * median)
    return NoiseBand(
        median=median,
        mad=mad,
        lower=max(0.0, median - absolute),
        upper=median + absolute,
        absolute=absolute,
        relative=absolute / median,
        sample_count=len(values),
    )


@dataclass(frozen=True)
class ShapeGPUCost:
    shape_key: str
    gpu_lock_seconds: float

    @property
    def full_gpu_lock_seconds(self) -> float:
        """Wall time for the complete lock-held region."""

        return self.gpu_lock_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShapeGPUCost:
        result = cls(
            shape_key=str(value["shape_key"]),
            gpu_lock_seconds=float(value["gpu_lock_seconds"]),
        )
        if result.gpu_lock_seconds < 0:
            raise ValueError("GPU costs must be non-negative")
        return result


@dataclass
class CalibrationData:
    environment: EnvironmentFingerprint
    compile_scaling: list[CompileScalingPoint]
    worker_count: int
    noise: NoiseBand
    shape_costs: list[ShapeGPUCost] = field(default_factory=list)
    total_gpu_memory_bytes: int | None = None
    reference_reserve_bytes: int = 0
    gpu_memory_margin_bytes: int = 0
    schema_version: int = CALIBRATION_SCHEMA_VERSION
    schema_fingerprint: str = CALIBRATION_SCHEMA_FINGERPRINT
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "schema_fingerprint": self.schema_fingerprint,
            "environment": self.environment.to_dict(),
            "compile_scaling": [point.to_dict() for point in self.compile_scaling],
            "worker_count": self.worker_count,
            "noise": self.noise.to_dict(),
            "shape_costs": [cost.to_dict() for cost in self.shape_costs],
            "total_gpu_memory_bytes": self.total_gpu_memory_bytes,
            "reference_reserve_bytes": self.reference_reserve_bytes,
            "gpu_memory_margin_bytes": self.gpu_memory_margin_bytes,
            "metadata": self.metadata,
        }
        canonical_json(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationData:
        schema_version = int(value.get("schema_version", -1))
        schema_fingerprint = str(value.get("schema_fingerprint", ""))
        results_v5 = (
            schema_version == _RESULTS_V5_SCHEMA_VERSION
            and schema_fingerprint == _RESULTS_V5_SCHEMA_FINGERPRINT
        )
        if schema_version != CALIBRATION_SCHEMA_VERSION and not results_v5:
            raise ValueError(
                f"unsupported calibration schema {schema_version}; "
                f"expected {_RESULTS_V5_SCHEMA_VERSION} or "
                f"{CALIBRATION_SCHEMA_VERSION}"
            )
        if (
            schema_version == CALIBRATION_SCHEMA_VERSION
            and schema_fingerprint != CALIBRATION_SCHEMA_FINGERPRINT
        ):
            raise ValueError(
                "calibration schema fingerprint does not match this implementation"
            )
        environment = value.get("environment")
        if not isinstance(environment, Mapping):
            raise ValueError("calibration environment is missing or invalid")
        result = cls(
            schema_version=CALIBRATION_SCHEMA_VERSION,
            schema_fingerprint=CALIBRATION_SCHEMA_FINGERPRINT,
            environment=EnvironmentFingerprint.from_dict(environment),
            compile_scaling=[
                CompileScalingPoint.from_dict(point)
                for point in value.get("compile_scaling", [])
            ],
            worker_count=int(value["worker_count"]),
            noise=NoiseBand.from_dict(
                value["loaded_noise"] if results_v5 else value["noise"]
            ),
            shape_costs=[
                ShapeGPUCost.from_dict(cost) for cost in value.get("shape_costs", [])
            ],
            total_gpu_memory_bytes=(
                None
                if value.get("total_gpu_memory_bytes") is None
                else int(value["total_gpu_memory_bytes"])
            ),
            reference_reserve_bytes=int(value.get("reference_reserve_bytes", 0)),
            gpu_memory_margin_bytes=int(value.get("gpu_memory_margin_bytes", 0)),
            metadata=dict(value.get("metadata", {})),
        )
        if result.worker_count < 1:
            raise ValueError("calibration worker_count must be positive")
        if min(result.reference_reserve_bytes, result.gpu_memory_margin_bytes) < 0:
            raise ValueError("calibration GPU reserves must be non-negative")
        if (
            result.total_gpu_memory_bytes is not None
            and result.total_gpu_memory_bytes < 0
        ):
            raise ValueError("total_gpu_memory_bytes must be non-negative")
        canonical_json(result.to_dict())
        return result

    def safe_worker_cap(self, *, maximum_workers: int | None = None) -> int | None:
        if self.total_gpu_memory_bytes is None:
            return None
        selected = next(
            (
                point
                for point in self.compile_scaling
                if point.workers == self.worker_count and point.worker_gpu_footprints
            ),
            None,
        )
        if selected is None:
            populated = [
                point for point in self.compile_scaling if point.worker_gpu_footprints
            ]
            selected = populated[-1] if populated else None
        if selected is None:
            return None
        return derive_safe_worker_cap(
            self.total_gpu_memory_bytes,
            selected.worker_gpu_footprints,
            reference_reserve_bytes=self.reference_reserve_bytes,
            margin_bytes=self.gpu_memory_margin_bytes,
            maximum_workers=maximum_workers,
        )

    def recommended_worker_count(self) -> int:
        selected = min(self.worker_count, MAX_CALIBRATION_WORKERS)
        cap = self.safe_worker_cap(maximum_workers=selected)
        return selected if cap is None else min(selected, cap)

    def save(self, path: str | os.PathLike[str], *, fsync: bool = True) -> None:
        atomic_write_json(path, self.to_dict(), fsync=fsync)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> CalibrationData:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("calibration root must be an object")
        return cls.from_dict(value)

    def validate_compatibility(
        self,
        current: EnvironmentFingerprint,
        *,
        force: bool = False,
    ) -> list[str]:
        return validate_compatibility(self, current, force=force)


def compatibility_mismatches(
    calibration: CalibrationData,
    current: EnvironmentFingerprint,
) -> list[str]:
    """Return stable, human-readable environment differences."""

    mismatches: list[str] = []
    if calibration.schema_version != CALIBRATION_SCHEMA_VERSION:
        mismatches.append(
            f"schema_version: saved={calibration.schema_version!r}, "
            f"current={CALIBRATION_SCHEMA_VERSION!r}"
        )
    if calibration.schema_fingerprint != CALIBRATION_SCHEMA_FINGERPRINT:
        mismatches.append(
            "schema_fingerprint: "
            f"saved={calibration.schema_fingerprint!r}, "
            f"current={CALIBRATION_SCHEMA_FINGERPRINT!r}"
        )
    saved = calibration.environment.to_dict()
    observed = current.to_dict()
    for field_name in _SCHEMA_FIELDS["environment"]:
        if field_name in _UNCHECKED_ENVIRONMENT_FIELDS:
            continue
        if field_name in _OPTIONAL_ENVIRONMENT_FIELDS and (
            saved[field_name] is None or observed[field_name] is None
        ):
            continue
        if field_name == "driver" and saved[field_name] == "unknown":
            continue
        if saved[field_name] != observed[field_name]:
            mismatches.append(
                f"{field_name}: saved={saved[field_name]!r}, "
                f"current={observed[field_name]!r}"
            )
    return mismatches


def validate_compatibility(
    calibration: CalibrationData,
    current: EnvironmentFingerprint,
    *,
    force: bool = False,
) -> list[str]:
    """Validate a calibration, returning mismatches only under force override."""

    mismatches = compatibility_mismatches(calibration, current)
    if mismatches and not force:
        raise CalibrationCompatibilityError(mismatches)
    return mismatches
