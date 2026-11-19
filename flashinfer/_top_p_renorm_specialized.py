"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Specialized top-p renormalisation on compute capability 10.3 (B300 / GB300).

Private implementation module, not public API: its only caller is
:func:`flashinfer.sampling.top_p_renorm_probs`, which calls :func:`top_p_renorm_probs_specialized`
first and keeps its stock path whenever that returns ``None``.

:func:`flashinfer.sampling.top_p_renorm_probs` calls with float32 ``probs`` of shape
``(batch_size, vocab_size)``, a per-row float32 ``top_p`` tensor and ``is_deterministic=False``
are served by a single-launch thread-block-cluster kernel
(``include/flashinfer/top_p_renorm_specialized.cuh``) for the exact ``(batch_size, vocab_size)``
rows of ``top_p_renorm_probs_specialized_workloads.json``. Every other call keeps the stock AIR
top-p kernels. The stock path's rule, evaluated with 64-bit fixed-point masses: the nucleus is
the shortest descending prefix whose mass reaches ``top_p`` times the row's own mass (AIR's
target, so a row that does not sum to 1 keeps the same entries), the entries ``>=`` its threshold
are kept (ties kept) and renormalised to sum to 1 (a kept mass ``<= 1e-8`` is left as it is); a NaN
entry has no mass and is never kept. Each entry is a whole number of units of ``2**-q`` of the
row's scale (exact for entries of at least ``2**-35`` of the row's mass, at most half a unit off
for smaller ones), sums are exact integer sums and the target is rounded up to a whole unit, so
the kept set is what exact arithmetic keeps unless a prefix mass lies within ``V / 2`` units
(below ``2**-41`` of the row's mass for ``V <= 2**18``) of the target, and it does not depend on a
summation order. AIR sums in float32 in an order that depends on its launch, so where a prefix
mass lies within that rounding of the target (e.g. float32 ``[0.5, 0.3, 0.2]`` at ``top_p=0.5``,
whose exact mass is ``1 + 2**-26``) the two can keep different entries.

The decision is made at call time. The calls the kernel can never serve return ``None`` first,
before the compile check, the environment, any counter or any diagnostics, so the stock path's
host cost is unchanged for them: ``is_deterministic=True``, a scalar ``top_p``, non-float32
inputs, a device other than a CUDA device of compute capability exactly 10.3 (a flag cached per
device on its first call) and, after the ``torch.compile`` check below, a ``(batch_size,
vocab_size)`` off the allowlist (checked after it because under dynamo tracing that check would
specialize the batch size: one recompile per batch size). These calls are not counted and write
nothing. The other calls are checked in this order (semantics, then shape/layout, device last, so
the correctness gates are exercisable without a GPU; :func:`_reject_reason` repeats the checks
above):

- inside ``torch.compile`` tracing the stock (registered custom op) path is kept, and nothing
  else of this module is traced;
- ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` forces the stock path (read on every call);
- 2-D ``probs`` / 1-D ``top_p`` of matching length, both contiguous, 16-byte aligned ``probs``;
- ``top_p`` on the device of ``probs``;
- never built inside a CUDA-graph capture: a capture before the first eager dispatch takes the
  (capture-safe) stock path. The capture state is read for the current stream of ``probs``'s
  device (the stream the kernel launches on), not of the current device, which may differ.
  After the first eager dispatch the kernel is capture-safe (no host sync, no allocation besides
  the output, launches on the current stream of ``probs``'s device).

A failed JIT build or a rejected launch falls back to the stock path (the launch failure
disables the kernel on that device for the process) and is reported by
:func:`_top_p_renorm_probs_specialized_stats`.

Markers, once per process each: :data:`DISPATCH_MARKER` on the first specialized dispatch,
:data:`KILL_SWITCH_MARKER` on the first kill-switched call the kernel would have served, and
``<FALLBACK_MARKER>: <reason> (<context>)`` on the first fallback for each reason of such a call
(layout, alignment, capture before the build, build or launch error, an unavailable allowlist),
so a silent non-dispatch of a served shape is visible in serving logs. They are written directly
to ``sys.stderr`` (serving frameworks may route or drop library loggers) and to the
``flashinfer.jit`` logger only for calls whose ``probs`` live on a compute capability 10.3 device
(the dispatch marker always does). Calls on any other device write nothing to stderr: the lines
of the few that reach the full guard (one whose arguments the first checks cannot read) go once
each to the ``flashinfer.jit`` logger at DEBUG level. That gate's capability comes from a
per-device cache that it never fills inside a CUDA-graph capture (a device not seen before the
capture counts as another device).
"""

import functools
import json
import os
import sys
import threading
from importlib import resources
from typing import Any, Callable, Dict, FrozenSet, Optional, Set, Tuple, Union

import torch

from .jit.core import logger
from .jit.top_p_renorm_specialized import (
    TOP_P_RENORM_SPECIALIZED_MODULE_NAME,
    gen_top_p_renorm_specialized_module,
)
from .utils import get_compute_capability

# Kill switch shared by FlashInfer's specialized kernels: read at CALL time.
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Dispatch surface (package data; see the file's note).
_WORKLOAD_FILE = "top_p_renorm_probs_specialized_workloads.json"
_WORKLOAD_FIELDS = ("batch_size", "vocab_size")
_COMPUTE_CAPABILITY = (10, 3)

# Mirror of ``kLaunchConfigs`` / ``plan_launch`` in the kernel header: (cluster size, threads per
# CTA, minimum CTAs per SM, float4 loads in flight per thread). Every entry is compiled into the
# module twice (float4 and scalar rows).
_LAUNCH_CONFIGS: Tuple[Tuple[int, int, int, int], ...] = (
    (8, 256, 6, 8),  # short rows (< 64K entries)
    (16, 1024, 1, 4),  # B <= 9,  V > 200000
    (16, 1024, 1, 3),  # B <= 9,  V <= 200000
    (16, 512, 3, 8),  # B <= 20, V > 200000
    (16, 512, 3, 5),  # B <= 20, V <= 200000
    (16, 256, 4, 10),  # B <= 30, V <= 200000
    (8, 512, 3, 8),  # B > 30,  V <= 200000
    (8, 512, 2, 16),  # B <= 40, V > 200000
    (8, 512, 4, 8),  # B > 40,  V > 200000
)
_KERNEL_INSTANTIATIONS = 2 * len(_LAUNCH_CONFIGS)

DISPATCH_MARKER = "top_p_renorm_probs: specialized SM103 cluster kernel dispatched"
KILL_SWITCH_MARKER = (
    "top_p_renorm_probs: specialized SM103 cluster kernel disabled by "
    "FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1, using the AIR top-p path"
)
FALLBACK_MARKER = "top_p_renorm_probs: specialized SM103 path fallback"

_module: Optional[Any] = None
_module_error: Optional[str] = None
_module_lock = threading.Lock()
_launch_failed_devices: Set[int] = set()
_stats: Dict[str, Any] = {
    "calls": 0,
    "dispatch_count": 0,
    "fallbacks": 0,
    "fallback_reasons": {},
    "launch_errors": 0,
    "launch_error": None,
    "guard_error": None,
    "module_loaded": False,
    "module_error": None,
}
# Keys of the one-time markers already written to stderr by this process.
_markers_emitted: Set[str] = set()
# Keys of the kill-switch / fallback lines already logged at DEBUG level for calls on other
# devices (they never reach stderr).
_markers_logged_debug: Set[str] = set()
# Compute capability per CUDA device index, recorded by the guard and by the marker gate; the
# marker gate never queries a device inside a CUDA-graph capture.
_device_capability: Dict[int, Tuple[int, int]] = {}
# Whether the kernel can run on a device (CUDA, compute capability 10.3), per ``torch.device``:
# recorded on the first call on that device, read before any other work of the call.
_device_supported: Dict[torch.device, bool] = {}
# Error of a failed allowlist load (every call then falls back as ``shape_not_allowlisted``);
# reported with the first such fallback.
_allowlist_error: Optional[str] = None


def _emit_once(key: str, line: str) -> None:
    """Write ``line`` once per process to stderr and to the ``flashinfer.jit`` logger (INFO)."""
    if key in _markers_emitted:
        return
    _markers_emitted.add(key)
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - diagnostics must never break the call
        pass
    logger.info(line)


def _is_capturing(device: torch.device) -> bool:
    """Whether the current stream of CUDA ``device`` is capturing a CUDA graph.
    ``torch.cuda.is_current_stream_capturing()`` alone reads the current device's stream, which
    misses a capture on ``device`` when another device is current."""
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _is_target_device(device: Optional[torch.device]) -> bool:
    """Whether ``device`` is a CUDA device of compute capability 10.3 (the device whose calls
    write markers to stderr). Inside a CUDA-graph capture only a capability recorded before is
    used: a device not seen yet counts as another device. Never raises."""
    try:
        if device is None or device.type != "cuda" or device.index is None:
            return False
        capability = _device_capability.get(device.index)
        if capability is None:
            if _is_capturing(device):
                return False
            capability = get_compute_capability(device)
            _device_capability[device.index] = capability
        return capability == _COMPUTE_CAPABILITY
    except Exception:  # noqa: BLE001 - diagnostics must never break the call
        return False


def _report_once(
    key: str,
    make_line: Callable[[], str],
    device: Optional[torch.device],
    warning: Optional[str] = None,
) -> None:
    """Report a kill-switch / fallback line once per process: for a call on a compute capability
    10.3 device on stderr and the ``flashinfer.jit`` logger (after ``warning``, if any, at
    WARNING level), for a call on any other device at DEBUG level on that logger only."""
    if key in _markers_emitted:
        return
    if _is_target_device(device):
        if warning is not None:
            logger.warning(warning)
        _emit_once(key, make_line())
    elif key not in _markers_logged_debug:
        _markers_logged_debug.add(key)
        logger.debug(make_line())


@functools.cache
def _load_allowlist() -> FrozenSet[Tuple[int, int]]:
    """The allowlisted ``(batch_size, vocab_size)`` rows; empty => never dispatch (the error is
    kept in ``_allowlist_error`` and reported with the first ``shape_not_allowlisted``
    fallback)."""
    global _allowlist_error
    try:
        payload = json.loads(
            resources.files("flashinfer").joinpath(_WORKLOAD_FILE).read_text()
        )
        fields = tuple(payload["fields"])
        if fields != _WORKLOAD_FIELDS:
            raise ValueError(f"unexpected workload fields: {fields}")
        capability = tuple(int(v) for v in payload["compute_capability"])
        if capability != _COMPUTE_CAPABILITY:
            raise ValueError(f"unexpected compute capability: {capability}")
        rows = frozenset((int(b), int(v)) for b, v in payload["workloads"])
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        _allowlist_error = f"{type(exc).__name__}: {exc}"[:300]
        return frozenset()
    _allowlist_error = None
    return rows


def _get_module() -> Optional[Any]:
    """Build/load the JIT module once (never called inside a CUDA-graph capture)."""
    global _module, _module_error
    if _module is not None:
        return _module
    with _module_lock:
        if _module is None and _module_error is None:
            try:
                _module = gen_top_p_renorm_specialized_module().build_and_load()
                _stats["module_loaded"] = True
            except Exception as exc:  # noqa: BLE001 - a failed build must never break the stock path
                _module_error = f"{type(exc).__name__}: {exc}"[:500]
                _stats["module_error"] = _module_error
                logger.warning(
                    "top_p_renorm_probs: JIT build of the specialized kernel failed, "
                    "using the AIR top-p path: %s",
                    _module_error,
                )
    return _module


def _reject_reason(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    is_deterministic: bool,
) -> Optional[str]:
    """``None`` when the specialized kernel serves the call, else the fallback reason."""
    if os.environ.get(_DISABLE_ENV) == "1":
        return "kill_switch"
    # ---- semantics ----
    if is_deterministic:
        return "deterministic"
    if not isinstance(top_p, torch.Tensor):
        return "scalar_top_p"
    if probs.dtype != torch.float32 or top_p.dtype != torch.float32:
        return "dtype"
    # ---- shape / layout ----
    if probs.dim() != 2 or top_p.dim() != 1 or top_p.shape[0] != probs.shape[0]:
        return "shape"
    if not (probs.is_contiguous() and top_p.is_contiguous()):
        return "non_contiguous"
    if (int(probs.shape[0]), int(probs.shape[1])) not in _load_allowlist():
        return "shape_not_allowlisted"
    # float4 loads: 16-byte aligned rows (torch allocations are; an offset view may not be).
    if probs.data_ptr() % 16:
        return "alignment"
    # ---- device (last) ----
    if probs.device.type != "cuda" or top_p.device != probs.device:
        return "device"
    capability = get_compute_capability(probs.device)
    _device_capability[probs.device.index] = capability
    if capability != _COMPUTE_CAPABILITY:
        return "compute_capability"
    if probs.device.index in _launch_failed_devices:
        return "launch_error"
    if _module is None and _is_capturing(probs.device):
        # Never build/load inside a capture on the launch stream; the stock path is capture-safe.
        return "capturing_before_jit"
    return None


def _fallback_context(reason: str, probs: Any) -> str:
    """Short context for the first fallback of ``reason``; never raises."""
    try:
        parts = []
        if isinstance(probs, torch.Tensor):
            parts.append(
                f"shape={tuple(probs.shape)}, dtype={probs.dtype}, device={probs.device}"
            )
            if reason == "compute_capability":
                capability = _device_capability.get(probs.device.index)
                parts.append(f"capability={capability}")
        error = {
            "jit_unavailable": _stats["module_error"],
            "launch_error": _stats["launch_error"],
            "guard_error": _stats["guard_error"],
        }.get(reason)
        if error:
            parts.append(str(error))
        return "; ".join(parts)[:600]
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the call
        return f"context unavailable: {type(exc).__name__}"


def _bump_fallback(reason: str, probs: Any) -> None:
    _stats["fallbacks"] += 1
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    device = probs.device if isinstance(probs, torch.Tensor) else None
    if reason == "kill_switch":
        _report_once("kill_switch", lambda: KILL_SWITCH_MARKER, device)
        return
    _report_once(
        f"fallback:{reason}",
        lambda: f"{FALLBACK_MARKER}: {reason} ({_fallback_context(reason, probs)})",
        device,
    )


def _probe_device(device: torch.device) -> bool:
    """Record whether the kernel can run on ``device`` (first call on it); never raises."""
    supported = False
    try:
        if device.type == "cuda" and device.index is not None:
            capability = get_compute_capability(device)
            _device_capability[device.index] = capability
            supported = capability == _COMPUTE_CAPABILITY
    except Exception:  # noqa: BLE001 - an unreadable device keeps the stock path
        supported = False
    _device_supported[device] = supported
    return supported


def _report_allowlist_unavailable(probs: torch.Tensor) -> None:
    """Once per process: the allowlist failed to load, so no call is served."""
    error = _allowlist_error
    _report_once(
        "allowlist",
        lambda: f"{FALLBACK_MARKER}: allowlist_unavailable ({error})",
        probs.device,
        warning="top_p_renorm_probs: specialized kernel allowlist unavailable "
        f"({error}); using the AIR top-p path",
    )


def top_p_renorm_probs_specialized(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    is_deterministic: bool = False,
) -> Optional[torch.Tensor]:
    """Renormalised probabilities from the specialized kernel, or ``None`` when the call must
    take the stock path (any guard fails, the module cannot be built, or the launch is
    rejected). Never raises for an ineligible call."""
    # The calls the kernel can never serve return here, before the compile check, the
    # environment, the counters and the diagnostics below, so that the stock path's host cost is
    # unchanged for them (a few attribute reads and one dict lookup).
    try:
        if (
            is_deterministic
            or not isinstance(top_p, torch.Tensor)
            or probs.dtype is not torch.float32
            or top_p.dtype is not torch.float32
        ):
            return None
        supported = _device_supported.get(probs.device)
        if supported is None:  # the first call on this device
            if torch.compiler.is_compiling():
                return None
            supported = _probe_device(probs.device)
        if not supported:
            return None
        # Under torch.compile (dynamo) tracing keep the registered custom op of the stock path:
        # this check folds to a constant, so none of the host-side dispatch below is traced. It
        # precedes the allowlist check, which dynamo would specialize the batch size on.
        if torch.compiler.is_compiling():
            return None
        listed = probs.shape in _load_allowlist()
    except Exception:  # noqa: BLE001 - arguments these checks cannot read: the guard reports them
        if torch.compiler.is_compiling():
            return None
        listed = True
    if not listed:
        if _allowlist_error is not None:
            _report_allowlist_unavailable(probs)
        return None
    _stats["calls"] += 1
    try:
        reason = _reject_reason(probs, top_p, is_deterministic)
    except Exception as exc:  # noqa: BLE001 - a guard exception must never break the stock path
        reason = "guard_error"
        _stats["guard_error"] = f"{type(exc).__name__}: {exc}"[:500]
    module = None
    if reason is None:
        module = _get_module()
        if module is None:
            reason = "jit_unavailable"
    if reason is not None:
        _bump_fallback(reason, probs)
        return None

    out = torch.empty_like(probs, memory_format=torch.contiguous_format)
    try:
        module.top_p_renorm_probs_specialized(probs, top_p, out)
    except Exception as exc:  # noqa: BLE001 - a rejected launch falls back to the stock path
        error = f"{type(exc).__name__}: {exc}"[:500]
        _launch_failed_devices.add(probs.device.index)
        _stats["launch_errors"] += 1
        _stats["launch_error"] = error
        logger.warning(
            "top_p_renorm_probs: specialized kernel launch failed on %s, using the AIR "
            "top-p path for this device: %s",
            str(probs.device),
            error,
        )
        _bump_fallback("launch_error", probs)
        return None

    _stats["dispatch_count"] += 1
    if "dispatch" not in _markers_emitted:
        _emit_once(
            "dispatch",
            f"{DISPATCH_MARKER} (batch_size={int(probs.shape[0])}, "
            f"vocab_size={int(probs.shape[1])}, device={probs.device})",
        )
    return out


def _plan_index(batch_size: int, vocab_size: int) -> int:
    """Mirror of ``plan_launch`` in the kernel header: index into ``_LAUNCH_CONFIGS``."""
    b, v = batch_size, vocab_size
    if (v + 15) // 16 < 4096:
        return 0
    if b <= 9:
        return 1 if v > 200000 else 2
    if b <= 20:
        return 3 if v > 200000 else 4
    if v <= 200000:
        return 5 if b <= 30 else 6
    return 7 if b <= 40 else 8


def _plan_config(batch_size: int, vocab_size: int) -> Tuple[int, int, int, int, bool]:
    """``(cluster_size, num_threads, min_blocks, unroll, vectorized)`` the host plan picks for a
    16-byte aligned row."""
    cluster_size, num_threads, min_blocks, unroll = _LAUNCH_CONFIGS[
        _plan_index(batch_size, vocab_size)
    ]
    return cluster_size, num_threads, min_blocks, unroll, vocab_size % 4 == 0


def _top_p_renorm_probs_specialized_stats() -> Dict[str, Any]:
    """Diagnostics: dispatch counters, JIT state and the compile footprint.

    ``calls`` and the fallback counters cover the calls that pass the first checks (float32
    inputs, a ``top_p`` tensor, ``is_deterministic=False``, a compute capability 10.3 device, an
    allowlisted shape); the calls the kernel can never serve return before any counter.

    One JIT module (``compiled_variants`` is 0 or 1) holds every kernel instantiation the host
    plan can pick; batch size and vocabulary size are runtime launch arguments, so no call shape
    triggers a compile and the compile-cache key is the module name alone.
    ``distinct_kernels_for_allowlist`` counts the instantiations the allowlisted rows reach.
    """
    allowlist = sorted(_load_allowlist(), key=lambda row: (row[1], row[0]))
    configs = sorted({_plan_config(b, v) for b, v in allowlist})
    return {
        **_stats,
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "launch_failed_devices": sorted(_launch_failed_devices),
        "module_name": TOP_P_RENORM_SPECIALIZED_MODULE_NAME,
        "compile_cache_key": TOP_P_RENORM_SPECIALIZED_MODULE_NAME
        + " (no shape key: batch_size and vocab_size are runtime arguments)",
        "compiled_variants": int(_module is not None),
        "kernel_instantiations": _KERNEL_INSTANTIATIONS,
        "distinct_kernels_for_allowlist": len(configs),
        "allowlist_kernel_configs": [
            {
                "cluster_size": c,
                "num_threads": nt,
                "min_blocks": mb,
                "unroll": unr,
                "vectorized": vec,
            }
            for c, nt, mb, unr, vec in configs
        ],
        "precompiled": _module is not None,
        "allowlist_rows": len(allowlist),
        "allowlist_fields": list(_WORKLOAD_FIELDS),
        "allowlist": [list(row) for row in allowlist],
        "allowlist_error": _allowlist_error,
        "compute_capability": list(_COMPUTE_CAPABILITY),
        "dispatch_marker": DISPATCH_MARKER,
        "kill_switch_marker": KILL_SWITCH_MARKER,
        "fallback_marker": FALLBACK_MARKER,
        "markers_emitted": sorted(_markers_emitted),
    }
