# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CUDA graph ownership for the Green-Context split MegaMoE pipeline.

The helper deliberately imports neither CUDA Python nor PyTorch at module import
time. It captures one reset child plus caller-owned K1, K2, and K3 children
streams, then embeds them in a parent graph with this topology::

             /--> K1 --\
    K0 reset             +--> K3
             +--> K2 --/

K1 and K2 therefore have no graph dependency and may overlap on disjoint Green
Context SM partitions.  K3 has full dependencies on both children and normally
belongs to the device primary context.  Streams and execution contexts remain
owned by the caller; this object owns only the captured and parent graph handles.

For steady-state pipelines whose K3 child prepares the next epoch at its tail,
``capture_steady`` omits K0 and builds the three-child fork/join topology
``{K1, K2} -> K3``.  The four-child ``capture`` ABI and behavior remain the
default.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Callable, Optional


class GreenGraphError(RuntimeError):
    """Base error for split Green-Context graph construction and replay."""


class GreenGraphUnavailableError(GreenGraphError):
    """Raised when the installed CUDA binding lacks a required API."""


class GreenGraphCaptureError(GreenGraphError):
    """Raised when a child capture or parent graph construction fails."""


class GreenGraphCleanupError(GreenGraphError):
    """Raised after graph teardown attempted every outstanding action."""


LaunchCallable = Callable[[Any], None]
_CleanupAction = tuple[str, Callable[[Any], Any], Any]


@dataclass(frozen=True)
class GreenGraphTopology:
    """Handles useful for contract checks and optional graph inspection."""

    k0_child: Any
    k1_child: Any
    k2_child: Any
    k3_child: Any
    parent: Any
    k0_node: Any
    k1_node: Any
    k2_node: Any
    k3_node: Any
    executable: Any


class GreenGraph:
    """RAII owner for a mixed-execution-context CUDA graph."""

    def __init__(
        self,
        *,
        topology: GreenGraphTopology,
        cleanup_actions: list[_CleanupAction],
        driver: Any,
    ) -> None:
        self.topology = topology
        self._cleanup_actions = cleanup_actions
        self._driver = driver
        self._last_launch_stream: Any = None
        self._closed = False

    @classmethod
    def capture(
        cls,
        *,
        k0_stream: Any,
        k0_launch: LaunchCallable,
        k1_stream: Any,
        k1_launch: LaunchCallable,
        k2_stream: Any,
        k2_launch: LaunchCallable,
        k3_stream: Any,
        k3_launch: LaunchCallable,
        driver: Any = None,
        capture_mode: Any = None,
        synchronize_before_capture: bool = True,
    ) -> "GreenGraph":
        """Capture K0/K1/K2/K3 children and instantiate their parent graph.

        Each launch callable receives its corresponding raw CUDA stream.  The
        callable must only enqueue capture-safe, fixed-address work; JIT,
        allocation, and eager warmup belong before this method.

        K1/K2 streams should belong to disjoint Green Contexts.  K3 should use a
        non-blocking stream in the primary context so that the post-join reduce
        can use the full device.  The caller must keep all three streams and
        their contexts alive until this object is closed.
        """

        driver = _load_driver() if driver is None else driver
        if capture_mode is None:
            capture_mode = _default_capture_mode(driver)
        _require_callable(k0_launch, "k0_launch")
        _require_callable(k1_launch, "k1_launch")
        _require_callable(k2_launch, "k2_launch")
        _require_callable(k3_launch, "k3_launch")

        cleanup_actions: list[_CleanupAction] = []
        try:
            k0_child = _capture_child(
                role="K0",
                stream=k0_stream,
                launch=k0_launch,
                capture_mode=capture_mode,
                synchronize_before_capture=synchronize_before_capture,
                driver=driver,
            )
            cleanup_actions.append(
                ("cuGraphDestroy(K0 child)", driver.cuGraphDestroy, k0_child)
            )

            k1_child = _capture_child(
                role="K1",
                stream=k1_stream,
                launch=k1_launch,
                capture_mode=capture_mode,
                synchronize_before_capture=synchronize_before_capture,
                driver=driver,
            )
            cleanup_actions.append(
                ("cuGraphDestroy(K1 child)", driver.cuGraphDestroy, k1_child)
            )

            k2_child = _capture_child(
                role="K2",
                stream=k2_stream,
                launch=k2_launch,
                capture_mode=capture_mode,
                synchronize_before_capture=synchronize_before_capture,
                driver=driver,
            )
            cleanup_actions.append(
                ("cuGraphDestroy(K2 child)", driver.cuGraphDestroy, k2_child)
            )

            k3_child = _capture_child(
                role="K3",
                stream=k3_stream,
                launch=k3_launch,
                capture_mode=capture_mode,
                synchronize_before_capture=synchronize_before_capture,
                driver=driver,
            )
            cleanup_actions.append(
                ("cuGraphDestroy(K3 child)", driver.cuGraphDestroy, k3_child)
            )

            (parent,) = _checked_call(driver, "cuGraphCreate", 0)
            cleanup_actions.append(
                ("cuGraphDestroy(parent)", driver.cuGraphDestroy, parent)
            )

            (k0_node,) = _checked_call(
                driver,
                "cuGraphAddChildGraphNode",
                parent,
                None,
                0,
                k0_child,
            )
            (k1_node,) = _checked_call(
                driver,
                "cuGraphAddChildGraphNode",
                parent,
                [k0_node],
                1,
                k1_child,
            )
            (k2_node,) = _checked_call(
                driver,
                "cuGraphAddChildGraphNode",
                parent,
                [k0_node],
                1,
                k2_child,
            )
            (k3_node,) = _checked_call(
                driver,
                "cuGraphAddChildGraphNode",
                parent,
                [k1_node, k2_node],
                2,
                k3_child,
            )
            (executable,) = _checked_call(
                driver, "cuGraphInstantiate", parent, 0
            )
            cleanup_actions.append(
                (
                    "cuGraphExecDestroy",
                    driver.cuGraphExecDestroy,
                    executable,
                )
            )
        except BaseException as exc:
            cleanup_errors = _run_cleanup(cleanup_actions, driver)
            if isinstance(exc, GreenGraphCaptureError) and not cleanup_errors:
                raise
            detail = f"; cleanup also failed: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
            raise GreenGraphCaptureError(
                f"split Green graph construction failed: {exc}{detail}"
            ) from exc

        return cls(
            topology=GreenGraphTopology(
                k0_child=k0_child,
                k1_child=k1_child,
                k2_child=k2_child,
                k3_child=k3_child,
                parent=parent,
                k0_node=k0_node,
                k1_node=k1_node,
                k2_node=k2_node,
                k3_node=k3_node,
                executable=executable,
            ),
            cleanup_actions=cleanup_actions,
            driver=driver,
        )

    @classmethod
    def capture_steady(
        cls,
        *,
        k1_stream: Any,
        k1_launch: LaunchCallable,
        k2_stream: Any,
        k2_launch: LaunchCallable,
        k3_stream: Any,
        k3_launch: LaunchCallable,
        driver: Any = None,
        capture_mode: Any = None,
        synchronize_before_capture: bool = True,
    ) -> "GreenGraph":
        """Capture a steady-state ``{K1, K2} -> K3`` parent graph.

        K3 must include every operation needed to prepare the next epoch before
        it returns. In particular, callers that reuse accumulating counters or
        peer-written staging buffers must enqueue their tail reset and cross-rank
        publication barrier in ``k3_launch``. The first epoch must already be
        initialized before this graph is launched.

        This is a separate opt-in entry point so the existing four-child
        ``capture`` ABI and reset-before-fork topology remain unchanged.
        """

        driver = _load_driver() if driver is None else driver
        if capture_mode is None:
            capture_mode = _default_capture_mode(driver)
        _require_callable(k1_launch, "k1_launch")
        _require_callable(k2_launch, "k2_launch")
        _require_callable(k3_launch, "k3_launch")

        cleanup_actions: list[_CleanupAction] = []
        try:
            k1_child = _capture_child(
                role="K1",
                stream=k1_stream,
                launch=k1_launch,
                capture_mode=capture_mode,
                synchronize_before_capture=synchronize_before_capture,
                driver=driver,
            )
            cleanup_actions.append(
                ("cuGraphDestroy(K1 child)", driver.cuGraphDestroy, k1_child)
            )

            k2_child = _capture_child(
                role="K2",
                stream=k2_stream,
                launch=k2_launch,
                capture_mode=capture_mode,
                synchronize_before_capture=synchronize_before_capture,
                driver=driver,
            )
            cleanup_actions.append(
                ("cuGraphDestroy(K2 child)", driver.cuGraphDestroy, k2_child)
            )

            k3_child = _capture_child(
                role="K3",
                stream=k3_stream,
                launch=k3_launch,
                capture_mode=capture_mode,
                synchronize_before_capture=synchronize_before_capture,
                driver=driver,
            )
            cleanup_actions.append(
                ("cuGraphDestroy(K3 child)", driver.cuGraphDestroy, k3_child)
            )

            (parent,) = _checked_call(driver, "cuGraphCreate", 0)
            cleanup_actions.append(
                ("cuGraphDestroy(parent)", driver.cuGraphDestroy, parent)
            )

            (k1_node,) = _checked_call(
                driver,
                "cuGraphAddChildGraphNode",
                parent,
                None,
                0,
                k1_child,
            )
            (k2_node,) = _checked_call(
                driver,
                "cuGraphAddChildGraphNode",
                parent,
                None,
                0,
                k2_child,
            )
            (k3_node,) = _checked_call(
                driver,
                "cuGraphAddChildGraphNode",
                parent,
                [k1_node, k2_node],
                2,
                k3_child,
            )
            (executable,) = _checked_call(
                driver, "cuGraphInstantiate", parent, 0
            )
            cleanup_actions.append(
                (
                    "cuGraphExecDestroy",
                    driver.cuGraphExecDestroy,
                    executable,
                )
            )
        except BaseException as exc:
            cleanup_errors = _run_cleanup(cleanup_actions, driver)
            if isinstance(exc, GreenGraphCaptureError) and not cleanup_errors:
                raise
            detail = (
                f"; cleanup also failed: {'; '.join(cleanup_errors)}"
                if cleanup_errors
                else ""
            )
            raise GreenGraphCaptureError(
                f"steady split Green graph construction failed: {exc}{detail}"
            ) from exc

        return cls(
            topology=GreenGraphTopology(
                k0_child=None,
                k1_child=k1_child,
                k2_child=k2_child,
                k3_child=k3_child,
                parent=parent,
                k0_node=None,
                k1_node=k1_node,
                k2_node=k2_node,
                k3_node=k3_node,
                executable=executable,
            ),
            cleanup_actions=cleanup_actions,
            driver=driver,
        )

    @property
    def executable(self) -> Any:
        return self.topology.executable

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_launch_stream(self) -> Any:
        return self._last_launch_stream

    def launch(self, stream: Any) -> None:
        """Launch the parent graph on a caller-owned dependency stream."""

        self._ensure_open()
        _checked_call(
            self._driver,
            "cuGraphLaunch",
            self.topology.executable,
            stream,
        )
        self._last_launch_stream = stream

    def synchronize(self) -> None:
        """Wait for the most recent parent launch, if any, to complete."""

        self._ensure_open()
        if self._last_launch_stream is not None:
            _checked_call(
                self._driver,
                "cuStreamSynchronize",
                self._last_launch_stream,
            )

    def close(self, *, synchronize: bool = True) -> None:
        """Synchronize the last replay, then destroy graphs in reverse order."""

        if self._closed:
            return
        self._closed = True

        errors: list[str] = []
        if synchronize and self._last_launch_stream is not None:
            try:
                _checked_call(
                    self._driver,
                    "cuStreamSynchronize",
                    self._last_launch_stream,
                )
            except BaseException as exc:
                errors.append(f"cuStreamSynchronize(last launch): {exc}")
        errors.extend(_run_cleanup(self._cleanup_actions, self._driver))
        if errors:
            raise GreenGraphCleanupError(
                "Green graph cleanup failed after attempting every handle: "
                + "; ".join(errors)
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise GreenGraphError("GreenGraph is already closed")

    def __enter__(self) -> "GreenGraph":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except GreenGraphCleanupError as cleanup_exc:
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
            pass


def _capture_child(
    *,
    role: str,
    stream: Any,
    launch: LaunchCallable,
    capture_mode: Any,
    synchronize_before_capture: bool,
    driver: Any,
) -> Any:
    if synchronize_before_capture:
        _checked_call(driver, "cuStreamSynchronize", stream)

    _checked_call(driver, "cuStreamBeginCapture", stream, capture_mode)
    launch_error: Optional[BaseException] = None
    try:
        launch(stream)
    except BaseException as exc:
        launch_error = exc

    end_result: Any
    try:
        end_result = _call(driver, "cuStreamEndCapture", stream)
    except BaseException as exc:
        if launch_error is not None:
            raise GreenGraphCaptureError(
                f"{role} launch failed ({launch_error}); cuStreamEndCapture "
                f"also raised while restoring the stream: {exc}"
            ) from launch_error
        raise GreenGraphCaptureError(
            f"{role} cuStreamEndCapture raised while restoring the stream: {exc}"
        ) from exc

    end_error: Optional[BaseException] = None
    child: Any = None
    try:
        payload = _checked_result(driver, "cuStreamEndCapture", end_result)
        if len(payload) != 1:
            raise GreenGraphError(
                "cuStreamEndCapture returned an unexpected payload: "
                f"{payload!r}"
            )
        child = payload[0]
        if _is_null_handle(child):
            raise GreenGraphError("cuStreamEndCapture returned a null graph")
    except BaseException as exc:
        end_error = exc

    if launch_error is not None:
        cleanup_error = _destroy_captured_child(child, role, driver)
        suffix = f"; child cleanup failed: {cleanup_error}" if cleanup_error else ""
        if end_error is not None:
            suffix += f"; cuStreamEndCapture failed: {end_error}"
        raise GreenGraphCaptureError(
            f"{role} capture launch failed: {launch_error}{suffix}"
        ) from launch_error

    if end_error is not None:
        cleanup_error = _destroy_captured_child(child, role, driver)
        suffix = f"; child cleanup failed: {cleanup_error}" if cleanup_error else ""
        raise GreenGraphCaptureError(
            f"{role} capture could not produce a valid graph: {end_error}{suffix}"
        ) from end_error

    return child


def _destroy_captured_child(
    child: Any, role: str, driver: Any
) -> Optional[str]:
    if child is None or _is_null_handle(child):
        return None
    try:
        _checked_call(driver, "cuGraphDestroy", child)
    except BaseException as exc:
        return f"cuGraphDestroy({role} partial child): {exc}"
    return None


def _load_driver() -> Any:
    try:
        from cuda.bindings import driver
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise GreenGraphUnavailableError(
            "cuda.bindings.driver is required for GreenGraph"
        ) from exc
    return driver


def _default_capture_mode(driver: Any) -> Any:
    container = getattr(driver, "CUstreamCaptureMode", None)
    if container is None:
        raise GreenGraphUnavailableError(
            "CUDA binding lacks CUstreamCaptureMode"
        )
    mode = getattr(container, "CU_STREAM_CAPTURE_MODE_THREAD_LOCAL", None)
    if mode is None:
        raise GreenGraphUnavailableError(
            "CUDA binding lacks CU_STREAM_CAPTURE_MODE_THREAD_LOCAL"
        )
    return mode


def _require_callable(value: Any, name: str) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def _call(driver: Any, name: str, *args: Any) -> Any:
    function = getattr(driver, name, None)
    if not callable(function):
        raise GreenGraphUnavailableError(f"CUDA API {name} is unavailable")
    try:
        return function(*args)
    except Exception as exc:
        raise GreenGraphError(f"CUDA API {name} raised: {exc}") from exc


def _checked_call(driver: Any, name: str, *args: Any) -> tuple[Any, ...]:
    return _checked_result(driver, name, _call(driver, name, *args))


def _checked_result(driver: Any, name: str, result: Any) -> tuple[Any, ...]:
    if not isinstance(result, tuple) or not result:
        raise GreenGraphError(
            f"CUDA API {name} returned an unexpected binding result {result!r}"
        )
    status = result[0]
    try:
        status_code = int(status)
    except (TypeError, ValueError) as exc:
        raise GreenGraphError(
            f"CUDA API {name} returned an invalid CUresult {status!r}"
        ) from exc
    if status_code != 0:
        status_name = getattr(status, "name", str(status))
        raise GreenGraphError(
            f"CUDA API {name} failed with {status_name} ({status_code})"
        )
    return tuple(result[1:])


def _run_cleanup(
    actions: list[_CleanupAction], driver: Any
) -> list[str]:
    del driver  # The bound functions in actions already retain their owner.
    errors: list[str] = []
    while actions:
        label, function, handle = actions.pop()
        try:
            result = function(handle)
            if not isinstance(result, tuple) or not result:
                raise GreenGraphError(
                    f"unexpected binding result {result!r}"
                )
            if int(result[0]) != 0:
                status = result[0]
                status_name = getattr(status, "name", str(status))
                raise GreenGraphError(
                    f"failed with {status_name} ({int(status)})"
                )
        except BaseException as exc:
            errors.append(f"{label}: {exc}")
    return errors


def _is_null_handle(handle: Any) -> bool:
    if handle is None:
        return True
    try:
        return int(handle) == 0
    except (TypeError, ValueError):
        return False


__all__ = [
    "GreenGraph",
    "GreenGraphCaptureError",
    "GreenGraphCleanupError",
    "GreenGraphError",
    "GreenGraphTopology",
    "GreenGraphUnavailableError",
]
