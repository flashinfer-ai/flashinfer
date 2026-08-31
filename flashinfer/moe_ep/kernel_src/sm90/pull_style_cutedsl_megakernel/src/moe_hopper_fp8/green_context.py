# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-side CUDA Green Context helpers for split Hopper MegaMoE.

The module deliberately imports neither CUDA Python nor PyTorch at import time.
That keeps CPU-only contract tests usable and avoids initializing CUDA before a
runner has selected its device.  CUDA handles are exposed unchanged; PyTorch
``ExternalStream`` wrappers are constructed only when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Any, Callable, Optional, Sequence


class GreenContextError(RuntimeError):
    """Base error for Green Context setup, use, and teardown."""


class GreenContextUnavailableError(GreenContextError):
    """Raised when Green Contexts are unavailable on the selected device."""


class GreenContextConfigurationError(GreenContextError, ValueError):
    """Raised for an invalid SM partition request or driver response."""


class GreenContextCleanupError(GreenContextError):
    """Raised after teardown attempted every outstanding cleanup action."""


@dataclass(frozen=True)
class GreenContextSupport:
    """Result of a non-throwing Green Context capability check."""

    supported: bool
    reason: Optional[str]
    device_ordinal: int
    compute_capability: Optional[tuple[int, int]] = None
    total_sms: Optional[int] = None
    min_sm_partition_size: Optional[int] = None
    sm_coscheduled_alignment: Optional[int] = None


@dataclass(frozen=True)
class _DeviceInfo:
    device_ordinal: int
    device: Any
    compute_capability: tuple[int, int]
    total_sms: int
    min_sm_partition_size: int
    sm_coscheduled_alignment: int
    full_sm_resource: Any


@dataclass
class GreenContextPartition:
    """One driver-created SM partition and its stream/context handles."""

    name: str
    device_ordinal: int
    sm_count: int
    resource: Any
    descriptor: Any
    green_context: Any
    cuda_context: Any
    stream: Any
    _closed: bool = field(default=False, init=False, repr=False)
    _torch_stream: Any = field(default=None, init=False, repr=False)

    @property
    def raw_stream(self) -> Any:
        """Return the raw ``CUstream`` handle."""

        return self.stream

    @property
    def raw_green_context(self) -> Any:
        """Return the raw ``CUgreenCtx`` handle."""

        return self.green_context

    @property
    def raw_cuda_context(self) -> Any:
        """Return the ``CUcontext`` view derived from the green context."""

        return self.cuda_context

    @property
    def closed(self) -> bool:
        return self._closed

    def as_torch_stream(self, torch_module: Any = None) -> Any:
        """Lazily wrap the raw stream in ``torch.cuda.ExternalStream``.

        ``torch_module`` exists primarily for CPU contract tests.  Production
        callers normally omit it, in which case PyTorch is imported on demand.
        """

        if self._closed:
            raise GreenContextError(
                f"Green Context partition {self.name!r} is already closed"
            )
        if self._torch_stream is not None:
            return self._torch_stream
        if torch_module is None:
            try:
                import torch as torch_module  # type: ignore[no-redef]
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise GreenContextUnavailableError(
                    "PyTorch is required to create a CUDA ExternalStream"
                ) from exc
        try:
            external_stream = torch_module.cuda.ExternalStream
        except AttributeError as exc:
            raise GreenContextUnavailableError(
                "torch.cuda.ExternalStream is unavailable in this PyTorch build"
            ) from exc
        self._torch_stream = external_stream(
            int(self.stream), device=self.device_ordinal
        )
        return self._torch_stream


_CleanupAction = tuple[str, Callable[[Any], Any], Any]


class GreenContextSplit:
    """RAII owner for a primary SM partition and the driver's remainder.

    Use :meth:`create` as a context manager whenever possible.  The requested
    primary count must satisfy the device's co-scheduling alignment.  The
    remainder is accepted exactly as returned by the driver: CUDA explicitly
    does not give remainder resources the same alignment/performance guarantee
    as regular split groups (for example, H200 returns 80 + 52 from 132 SMs).
    """

    def __init__(
        self,
        *,
        support: GreenContextSupport,
        primary: GreenContextPartition,
        remainder: GreenContextPartition,
        cleanup_actions: Sequence[_CleanupAction],
        driver: Any,
    ) -> None:
        self.support = support
        self.primary = primary
        self.remainder = remainder
        # K1/K2 aliases make the intended split-MegaMoE ownership explicit.
        self.k1 = primary
        self.k2 = remainder
        self._cleanup_actions = list(cleanup_actions)
        self._driver = driver
        self._closed = False

    @classmethod
    def create(
        cls,
        primary_sm_count: int,
        *,
        device_ordinal: int = 0,
        stream_priority: int = 0,
        driver: Any = None,
    ) -> "GreenContextSplit":
        """Create two disjoint Green Contexts from one device's SM resource."""

        driver = _load_driver() if driver is None else driver
        info = _query_device_info(device_ordinal, driver)
        _validate_partition_request(primary_sm_count, info)

        split_result = _checked_call(
            driver,
            "cuDevSmResourceSplitByCount",
            1,
            info.full_sm_resource,
            0,
            primary_sm_count,
        )
        if len(split_result) != 3:
            raise GreenContextError(
                "cuDevSmResourceSplitByCount returned an unexpected payload: "
                f"expected 3 values after CUresult, got {len(split_result)}"
            )
        parts, actual_group_count, remainder_resource = split_result
        if actual_group_count != 1 or not isinstance(parts, (list, tuple)):
            raise GreenContextConfigurationError(
                "CUDA did not produce the requested single primary SM group: "
                f"actual_group_count={actual_group_count!r}, parts={parts!r}"
            )
        if len(parts) != 1:
            raise GreenContextConfigurationError(
                f"CUDA returned {len(parts)} primary resources; expected exactly 1"
            )

        primary_resource = parts[0]
        actual_primary_sms = _resource_sm_count(primary_resource, "primary")
        actual_remainder_sms = _resource_sm_count(
            remainder_resource, "remainder"
        )
        _validate_driver_partition(
            requested_sms=primary_sm_count,
            primary_sms=actual_primary_sms,
            remainder_sms=actual_remainder_sms,
            info=info,
        )

        cleanup_actions: list[_CleanupAction] = []
        try:
            primary = _create_partition(
                name="k1",
                resource=primary_resource,
                sm_count=actual_primary_sms,
                device_ordinal=device_ordinal,
                device=info.device,
                stream_priority=stream_priority,
                driver=driver,
                cleanup_actions=cleanup_actions,
            )
            remainder = _create_partition(
                name="k2",
                resource=remainder_resource,
                sm_count=actual_remainder_sms,
                device_ordinal=device_ordinal,
                device=info.device,
                stream_priority=stream_priority,
                driver=driver,
                cleanup_actions=cleanup_actions,
            )
        except BaseException as exc:
            cleanup_errors = _run_cleanup(cleanup_actions, driver)
            if cleanup_errors:
                detail = "; ".join(cleanup_errors)
                raise GreenContextError(
                    f"Green Context creation failed ({exc}); partial cleanup "
                    f"also failed: {detail}"
                ) from exc
            raise

        support = _support_from_info(info)
        return cls(
            support=support,
            primary=primary,
            remainder=remainder,
            cleanup_actions=cleanup_actions,
            driver=driver,
        )

    @property
    def partitions(self) -> tuple[GreenContextPartition, GreenContextPartition]:
        return self.primary, self.remainder

    @property
    def sm_counts(self) -> tuple[int, int]:
        """Actual driver-selected SM counts, not merely the requested count."""

        return self.primary.sm_count, self.remainder.sm_count

    @property
    def raw_streams(self) -> tuple[Any, Any]:
        return self.primary.stream, self.remainder.stream

    @property
    def raw_green_contexts(self) -> tuple[Any, Any]:
        return self.primary.green_context, self.remainder.green_context

    @property
    def raw_cuda_contexts(self) -> tuple[Any, Any]:
        return self.primary.cuda_context, self.remainder.cuda_context

    @property
    def closed(self) -> bool:
        return self._closed

    def torch_streams(self, torch_module: Any = None) -> tuple[Any, Any]:
        """Lazily construct both PyTorch ``ExternalStream`` wrappers."""

        return (
            self.primary.as_torch_stream(torch_module),
            self.remainder.as_torch_stream(torch_module),
        )

    def close(self) -> None:
        """Destroy streams then contexts in global reverse creation order."""

        if self._closed:
            return
        self._closed = True
        self.primary._closed = True
        self.remainder._closed = True
        errors = _run_cleanup(self._cleanup_actions, self._driver)
        if errors:
            raise GreenContextCleanupError(
                "Green Context cleanup failed after attempting all resources: "
                + "; ".join(errors)
            )

    def __enter__(self) -> "GreenContextSplit":
        if self._closed:
            raise GreenContextError("cannot re-enter a closed GreenContextSplit")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except GreenContextCleanupError as cleanup_exc:
            if exc is None:
                raise
            warnings.warn(
                f"{cleanup_exc}; preserving the active exception {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        return False

    def __del__(self) -> None:  # pragma: no cover - nondeterministic fallback
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            # Destructors cannot report failures reliably.  Explicit ``close``
            # or a context manager is required when teardown errors matter.
            pass


def check_green_context_support(
    device_ordinal: int = 0, *, driver: Any = None
) -> GreenContextSupport:
    """Return Green Context support metadata without raising expected failures."""

    try:
        driver = _load_driver() if driver is None else driver
        return _support_from_info(_query_device_info(device_ordinal, driver))
    except (GreenContextError, ImportError) as exc:
        return GreenContextSupport(
            supported=False,
            reason=str(exc),
            device_ordinal=device_ordinal,
        )


def _load_driver() -> Any:
    try:
        import cuda.bindings.driver as driver
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise GreenContextUnavailableError(
            "cuda.bindings.driver is required for CUDA Green Contexts"
        ) from exc
    return driver


def _query_device_info(device_ordinal: int, driver: Any) -> _DeviceInfo:
    if not isinstance(device_ordinal, int) or isinstance(device_ordinal, bool):
        raise GreenContextConfigurationError("device_ordinal must be an integer")
    if device_ordinal < 0:
        raise GreenContextConfigurationError("device_ordinal must be non-negative")

    required = (
        "cuInit",
        "cuDeviceGet",
        "cuDeviceGetAttribute",
        "cuDeviceGetDevResource",
        "cuDevSmResourceSplitByCount",
        "cuDevResourceGenerateDesc",
        "cuGreenCtxCreate",
        "cuCtxFromGreenCtx",
        "cuGreenCtxStreamCreate",
        "cuStreamDestroy",
        "cuGreenCtxDestroy",
    )
    missing = [name for name in required if not callable(getattr(driver, name, None))]
    if missing:
        raise GreenContextUnavailableError(
            "CUDA Python/driver lacks required Green Context APIs: "
            + ", ".join(missing)
        )

    _checked_call(driver, "cuInit", 0)
    (device,) = _checked_call(driver, "cuDeviceGet", device_ordinal)
    attrs = _enum_container(driver, "CUdevice_attribute")
    (major,) = _checked_call(
        driver,
        "cuDeviceGetAttribute",
        _enum_value(attrs, "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"),
        device,
    )
    (minor,) = _checked_call(
        driver,
        "cuDeviceGetAttribute",
        _enum_value(attrs, "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR"),
        device,
    )
    (attribute_sm_count,) = _checked_call(
        driver,
        "cuDeviceGetAttribute",
        _enum_value(attrs, "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"),
        device,
    )
    compute_capability = (int(major), int(minor))
    if compute_capability < (9, 0):
        raise GreenContextUnavailableError(
            "CUDA Green Context SM partitioning requires compute capability "
            f"9.0 or newer; device {device_ordinal} is sm_{major}{minor}"
        )

    resource_types = _enum_container(driver, "CUdevResourceType")
    (full_resource,) = _checked_call(
        driver,
        "cuDeviceGetDevResource",
        device,
        _enum_value(resource_types, "CU_DEV_RESOURCE_TYPE_SM"),
    )
    total_sms = _resource_sm_count(full_resource, "full device")
    if total_sms != int(attribute_sm_count):
        raise GreenContextConfigurationError(
            "CUDA reported inconsistent SM counts: "
            f"device attribute={attribute_sm_count}, SM resource={total_sms}"
        )
    sm = getattr(full_resource, "sm", None)
    min_partition = int(getattr(sm, "minSmPartitionSize", 0))
    alignment = int(getattr(sm, "smCoscheduledAlignment", 0))
    if min_partition <= 0 or alignment <= 0:
        raise GreenContextUnavailableError(
            "CUDA returned invalid Green Context SM constraints: "
            f"minSmPartitionSize={min_partition}, "
            f"smCoscheduledAlignment={alignment}"
        )
    return _DeviceInfo(
        device_ordinal=device_ordinal,
        device=device,
        compute_capability=compute_capability,
        total_sms=total_sms,
        min_sm_partition_size=min_partition,
        sm_coscheduled_alignment=alignment,
        full_sm_resource=full_resource,
    )


def _support_from_info(info: _DeviceInfo) -> GreenContextSupport:
    return GreenContextSupport(
        supported=True,
        reason=None,
        device_ordinal=info.device_ordinal,
        compute_capability=info.compute_capability,
        total_sms=info.total_sms,
        min_sm_partition_size=info.min_sm_partition_size,
        sm_coscheduled_alignment=info.sm_coscheduled_alignment,
    )


def _validate_partition_request(primary_sm_count: int, info: _DeviceInfo) -> None:
    if not isinstance(primary_sm_count, int) or isinstance(primary_sm_count, bool):
        raise GreenContextConfigurationError(
            "primary_sm_count must be an integer"
        )
    if primary_sm_count < info.min_sm_partition_size:
        raise GreenContextConfigurationError(
            f"primary_sm_count={primary_sm_count} is below the device minimum "
            f"{info.min_sm_partition_size}"
        )
    if primary_sm_count >= info.total_sms:
        raise GreenContextConfigurationError(
            f"primary_sm_count={primary_sm_count} must leave a non-empty "
            f"remainder from {info.total_sms} SMs"
        )
    if primary_sm_count % info.sm_coscheduled_alignment != 0:
        raise GreenContextConfigurationError(
            f"primary_sm_count={primary_sm_count} is not aligned to the "
            f"device's {info.sm_coscheduled_alignment}-SM co-scheduling unit"
        )


def _validate_driver_partition(
    *,
    requested_sms: int,
    primary_sms: int,
    remainder_sms: int,
    info: _DeviceInfo,
) -> None:
    if primary_sms < requested_sms:
        raise GreenContextConfigurationError(
            f"CUDA returned only {primary_sms} primary SMs for minimum request "
            f"{requested_sms}"
        )
    if primary_sms % info.sm_coscheduled_alignment != 0:
        raise GreenContextConfigurationError(
            f"CUDA returned a primary partition of {primary_sms} SMs, not "
            f"aligned to {info.sm_coscheduled_alignment}"
        )
    if remainder_sms <= 0:
        raise GreenContextConfigurationError(
            f"CUDA returned an empty remainder ({remainder_sms} SMs)"
        )
    if primary_sms + remainder_sms != info.total_sms:
        raise GreenContextConfigurationError(
            "CUDA split does not cover the full SM resource: "
            f"{primary_sms} + {remainder_sms} != {info.total_sms}"
        )


def _create_partition(
    *,
    name: str,
    resource: Any,
    sm_count: int,
    device_ordinal: int,
    device: Any,
    stream_priority: int,
    driver: Any,
    cleanup_actions: list[_CleanupAction],
) -> GreenContextPartition:
    (descriptor,) = _checked_call(
        driver, "cuDevResourceGenerateDesc", [resource], 1
    )
    create_flags = _enum_container(driver, "CUgreenCtxCreate_flags")
    (green_context,) = _checked_call(
        driver,
        "cuGreenCtxCreate",
        descriptor,
        device,
        _enum_value(create_flags, "CU_GREEN_CTX_DEFAULT_STREAM"),
    )
    cleanup_actions.append(
        (f"cuGreenCtxDestroy({name})", driver.cuGreenCtxDestroy, green_context)
    )
    (cuda_context,) = _checked_call(
        driver, "cuCtxFromGreenCtx", green_context
    )
    stream_flags = _enum_container(driver, "CUstream_flags")
    (stream,) = _checked_call(
        driver,
        "cuGreenCtxStreamCreate",
        green_context,
        _enum_value(stream_flags, "CU_STREAM_NON_BLOCKING"),
        stream_priority,
    )
    cleanup_actions.append(
        (f"cuStreamDestroy({name})", driver.cuStreamDestroy, stream)
    )
    return GreenContextPartition(
        name=name,
        device_ordinal=device_ordinal,
        sm_count=sm_count,
        resource=resource,
        descriptor=descriptor,
        green_context=green_context,
        cuda_context=cuda_context,
        stream=stream,
    )


def _run_cleanup(actions: list[_CleanupAction], driver: Any) -> list[str]:
    errors: list[str] = []
    while actions:
        label, function, handle = actions.pop()
        try:
            result = function(handle)
            _checked_result(driver, label, result)
        except BaseException as exc:
            errors.append(f"{label}: {exc}")
    return errors


def _checked_call(driver: Any, name: str, *args: Any) -> tuple[Any, ...]:
    function = getattr(driver, name, None)
    if not callable(function):
        raise GreenContextUnavailableError(f"CUDA API {name} is unavailable")
    try:
        result = function(*args)
    except Exception as exc:
        raise GreenContextError(f"CUDA API {name} raised: {exc}") from exc
    return _checked_result(driver, name, result)


def _checked_result(driver: Any, name: str, result: Any) -> tuple[Any, ...]:
    if not isinstance(result, tuple) or not result:
        raise GreenContextError(
            f"CUDA API {name} returned an unexpected binding result {result!r}"
        )
    status = result[0]
    try:
        status_code = int(status)
    except (TypeError, ValueError) as exc:
        raise GreenContextError(
            f"CUDA API {name} returned an invalid CUresult {status!r}"
        ) from exc
    if status_code != 0:
        status_name = getattr(status, "name", str(status))
        detail = _cuda_error_string(driver, status)
        suffix = f": {detail}" if detail else ""
        raise GreenContextError(
            f"CUDA API {name} failed with {status_name} ({status_code}){suffix}"
        )
    return tuple(result[1:])


def _cuda_error_string(driver: Any, status: Any) -> Optional[str]:
    function = getattr(driver, "cuGetErrorString", None)
    if not callable(function):
        return None
    try:
        result = function(status)
    except Exception:
        return None
    if not isinstance(result, tuple) or len(result) < 2:
        return None
    try:
        if int(result[0]) != 0:
            return None
    except (TypeError, ValueError):
        return None
    value = result[1]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resource_sm_count(resource: Any, label: str) -> int:
    sm = getattr(resource, "sm", None)
    try:
        sm_count = int(sm.smCount)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GreenContextConfigurationError(
            f"CUDA {label} resource has no valid sm.smCount"
        ) from exc
    if sm_count <= 0:
        raise GreenContextConfigurationError(
            f"CUDA {label} resource has invalid smCount={sm_count}"
        )
    return sm_count


def _enum_container(driver: Any, name: str) -> Any:
    value = getattr(driver, name, None)
    if value is None:
        raise GreenContextUnavailableError(
            f"CUDA Python binding lacks enum container {name}"
        )
    return value


def _enum_value(container: Any, name: str) -> Any:
    value = getattr(container, name, None)
    if value is None:
        raise GreenContextUnavailableError(
            f"CUDA Python binding lacks required enum {name}"
        )
    return value


__all__ = [
    "GreenContextCleanupError",
    "GreenContextConfigurationError",
    "GreenContextError",
    "GreenContextPartition",
    "GreenContextSplit",
    "GreenContextSupport",
    "GreenContextUnavailableError",
    "check_green_context_support",
]
