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

The "specialized" backend of ``mm_mxfp8``: an agent-generated, SM107 small-M
MXFP8 GEMM kernel (``csrc/mm_mxfp8_smallm_sm107.cu``) behind a strict
(M, N, K) allowlist.

Dispatch contract (the specialized-backend pattern):

- ``backend="auto"`` routes to "specialized" only on the exact shapes of the
  package allowlist (``mm_mxfp8_specialized_sm107_workloads.json``); every
  other call keeps the stock path, with the reason recorded.
- An explicitly selected backend is never silently swapped AND never falls
  back: only ``backend="specialized"`` (or "auto") can reach this kernel, and
  explicit ``backend="specialized"`` that cannot serve a call raises
  ``NotImplementedError`` with the refusal reason (on a non-SM107 device the
  backend checker in ``gemm_base`` raises its standard unsupported-backend
  error before this module is reached). Only ``"auto"`` may route around a
  refusal (to the stock ranking).
- Guard order: GPU-free checks (kill switch, dtypes, geometry, contiguity,
  allowlist) first, then scale-layout guards, then device guards.
- A JIT build failure is logged once; auto falls through to the stock path,
  explicit ``backend="specialized"`` raises.
- ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read at call time) disables
  the kernel: auto keeps the stock path, explicit ``backend="specialized"``
  raises (reason ``kill_switch``).
- ``fallback_reasons`` and per-backend dispatch counters are recorded; see
  ``_mm_mxfp8_specialized_stats``.
"""

import functools
import json
import logging
import os
import threading
from importlib import resources
from typing import Any, Dict, Optional

import torch

from ..autotuner import TunableRunner
from ..jit.gemm.mxfp8_smallm_sm107 import gen_mm_mxfp8_smallm_sm107_module
from ..utils import get_compute_capability

logger = logging.getLogger(__name__)

# Kill switch shared by FlashInfer's specialized kernels: read at CALL time.
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Dispatch surface of the specialized backend (package data).
_WORKLOAD_FILE = "mm_mxfp8_specialized_sm107_workloads.json"


@functools.cache
def _load_allowlist() -> Optional[dict]:
    """Load the specialized backend's dispatch surface; ``None`` => never dispatch."""
    try:
        payload = json.loads(
            resources.files("flashinfer").joinpath(_WORKLOAD_FILE).read_text()
        )
        return {
            "cc": tuple(int(v) for v in payload["compute_capability"]),
            "shapes": frozenset(
                tuple(int(v) for v in shape) for shape in payload["shapes_mnk"]
            ),
        }
    except (
        OSError,
        ModuleNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            "flashinfer.mm_mxfp8: specialized-backend allowlist unavailable "
            "(%s: %s); using the stock path",
            type(exc).__name__,
            str(exc)[:200],
        )
        return None


_backend_modules: Dict[str, Any] = {}
_module_errors: Dict[str, str] = {}
_module_lock = threading.Lock()
_stats: Dict[str, Any] = {
    # Calls that considered the specialized backend (backend "auto" or
    # "specialized"); explicit other backends never enter this dispatch.
    "calls": 0,
    "backend_dispatches": {"specialized": 0},
    # Calls the specialized backend declined: auto ones continue into the
    # stock path, explicit ones raise with the recorded reason.
    "fallback_dispatches": 0,
    "fallback_reasons": {},
    "explicit_refusals": 0,
    "module_loaded": False,
    "module_error": None,
}
_marker_logged = False


def _bump_fallback(reason: str) -> None:
    _stats["fallback_dispatches"] += 1
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _get_specialized_module():
    """Build/load the specialized kernel's JIT module once; never inside a CUDA-graph capture."""
    key = "specialized"
    module = _backend_modules.get(key)
    if module is not None:
        return module
    with _module_lock:
        if key not in _backend_modules and key not in _module_errors:
            try:
                _backend_modules[key] = (
                    gen_mm_mxfp8_smallm_sm107_module().build_and_load()
                )
                _stats["module_loaded"] = True
            except Exception as exc:  # noqa: BLE001 - guard must never break the stock path
                error = f"{type(exc).__name__}: {exc}"[:500]
                _module_errors[key] = error
                _stats["module_error"] = error
                logger.warning(
                    "flashinfer.mm_mxfp8: JIT build of the specialized SM107 "
                    "kernel failed, using the stock path: %s",
                    error,
                )
    return _backend_modules.get(key)


def _mm_mxfp8_specialized_stats() -> dict:
    """Diagnostics: dispatch counters, fallback reasons and JIT state."""
    allowlist = _load_allowlist()
    return {
        **_stats,
        "backend_dispatches": dict(_stats["backend_dispatches"]),
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "module_errors": dict(_module_errors),
        "allowlist_loaded": allowlist is not None,
        "allowlist_size": len(allowlist["shapes"]) if allowlist else 0,
        "kill_switch_env": _DISABLE_ENV,
    }


def _swizzled_sf_numel(rows: int, k: int) -> int:
    """Element count of the 1D swizzled 128x4 scale buffer for (rows, k)."""
    return ((rows + 127) // 128 * 128) * ((k // 32 + 3) // 4 * 4)


# ---------------------------------------------------------------------------
# Guards. Each returns None when it passes, else the fallback reason.
# Order: GPU-free (kill switch, dtype, geometry, contiguity, allowlist) ->
# scale-layout guards -> device guards.
# ---------------------------------------------------------------------------


def _common_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: torch.Tensor,
    out_dtype: torch.dtype,
    use_8x4_sf_layout: bool,
) -> Optional[str]:
    """GPU-free guards of the specialized backend."""
    if os.environ.get(_DISABLE_ENV) == "1":
        return "kill_switch"
    if out_dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        return "out_dtype"
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        return "input_dtype"
    if a.shape[1] % 32 != 0:
        return "geometry"
    if not a.is_contiguous() or not b.t().is_contiguous() or not out.is_contiguous():
        return "non_contiguous"
    # Vector-access alignment. Contiguity does not establish it: a contiguous
    # view into offset storage keeps the offset address. The kernel issues
    # 32-byte operand loads (ldg256) and relies on 8-byte-aligned output rows
    # for its vector stores, so misaligned pointers would be undefined
    # behavior. Fresh torch allocations are 256-byte aligned; only offset
    # storage views fail these, and the stock path serves them.
    if a.data_ptr() % 32 or b.data_ptr() % 32 or out.data_ptr() % 16:
        return "pointer_alignment"
    # The kernel loads the operands and scale buffers from global memory
    # while CTAs store `out`, with no grid-level ordering between CTAs: an
    # output that shares storage bytes with any input is a data race (the
    # stock route's aliasing behavior is its own contract).
    out_lo = out.data_ptr()
    out_hi = out_lo + out.numel() * out.element_size()
    for t in (a, b, a_descale, b_descale):
        t_lo = t.data_ptr()
        if max(out_lo, t_lo) < min(out_hi, t_lo + t.numel() * t.element_size()):
            return "overlap"
    allowlist = _load_allowlist()
    if allowlist is None:
        return "allowlist_unavailable"
    if (a.shape[0], b.shape[1], a.shape[1]) not in allowlist["shapes"]:
        return "shape_not_allowlisted"
    return None


def _layout_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    use_8x4_sf_layout: bool,
) -> Optional[str]:
    """Scale-layout guards: the kernel decodes the 1D swizzled 128x4 layout only."""
    if use_8x4_sf_layout:
        return "scale_layout"
    if a_descale.ndim != 1 or b_descale.ndim != 1:
        return "scale_layout"
    if a_descale.dtype != torch.uint8 or b_descale.dtype != torch.uint8:
        return "scale_layout"
    # The kernel reads the raw scale bytes with 16-byte vector loads: the
    # buffers must be dense (a strided 1D view passes the ndim/dtype/numel
    # checks but interleaves foreign bytes) and 16-byte aligned.
    if not a_descale.is_contiguous() or not b_descale.is_contiguous():
        return "scale_layout"
    if a_descale.data_ptr() % 16 or b_descale.data_ptr() % 16:
        return "pointer_alignment"
    m, k = a.shape[0], a.shape[1]
    n = b.shape[1]
    if a_descale.numel() < _swizzled_sf_numel(m, k) or b_descale.numel() < (
        _swizzled_sf_numel(n, k)
    ):
        return "scale_numel"
    return None


def _device_supported(a: torch.Tensor) -> Optional[str]:
    """Device guards, checked after the layout guards."""
    if not a.is_cuda:
        return "device"
    allowlist = _load_allowlist()
    if allowlist is None or get_compute_capability(a.device) != allowlist["cc"]:
        return "compute_capability"
    # The capture state that matters is the one of the stream the kernel will
    # launch on — the current stream OF a.device (the FFI binding launches on
    # that device's stream under a device guard) — not the ambient device's.
    # The device context is entered only on an actual mismatch: this guard
    # runs on a host-latency-bound dispatch path.
    a_index = a.device.index
    if a_index is not None and a_index != torch.cuda.current_device():
        with torch.cuda.device(a_index):
            capturing = torch.cuda.is_current_stream_capturing()
    else:
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing and "specialized" not in _backend_modules:
        # Never compile/load inside a capture; the stock path is capture-safe.
        return "capturing_before_jit"
    return None


def _specialized_supported(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: torch.Tensor,
    out_dtype: torch.dtype,
    use_8x4_sf_layout: bool,
) -> Optional[str]:
    """Return ``None`` when the specialized backend serves this call, else the reason."""
    reason = _common_supported(
        a, b, a_descale, b_descale, out, out_dtype, use_8x4_sf_layout
    )
    if reason is not None:
        return reason
    reason = _layout_supported(a, b, a_descale, b_descale, use_8x4_sf_layout)
    if reason is not None:
        return reason
    return _device_supported(a)


def specialized_mm_mxfp8_eligible(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    use_8x4_sf_layout: bool,
) -> Optional[str]:
    """Eligibility probe for auto/autotune candidacy; ``None`` = eligible.

    Identical to the dispatch guards, but tolerates ``out=None`` (the probe
    runs before ``mm_mxfp8`` allocates its output, which is a fresh
    contiguous, aligned tensor of ``out_dtype`` and therefore passes the
    output guards by construction).
    """
    if out is None:
        if out_dtype != torch.bfloat16:
            return "out_dtype"
        out = _PROBE_OUT
    reason = _specialized_supported(
        a, b, a_descale, b_descale, out, out_dtype, use_8x4_sf_layout
    )
    if reason is not None:
        return reason
    # Candidacy requires a loadable kernel: the JIT module builds once here
    # (cached; failures are logged once) so backend="auto" never selects a
    # candidate whose forward would fail.
    if _get_specialized_module() is None:
        return "jit_unavailable"
    return None


# Stands in for a fresh allocation in the ``out=None`` eligibility probe:
# bf16, contiguous, allocator-aligned — exactly what mm_mxfp8 will allocate.
_PROBE_OUT = torch.empty(1, dtype=torch.bfloat16)


def _launch_specialized(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Run the kernel: ``out[M, N] = dequant(a[M, K]) @ dequant(b[K, N])``.

    ``b`` arrives in the ``mm_mxfp8`` convention, (K, N) column-major; the
    kernel consumes the row-major (N, K) view ``b.t()``.
    """
    module = _get_specialized_module()
    if module is None:
        return False
    module.mm_mxfp8_smallm_sm107_run(a, b.t(), a_descale, b_descale, out)
    return True


class SpecializedMxfp8SmallMRunner(TunableRunner):
    """``TunableRunner`` for the specialized kernel: a first-class autotune
    candidate of ``mm_mxfp8``.

    The kernel has a single fixed tactic (``-1``); ``get_valid_tactics``
    returns it only when the synthesized/real inputs are on the measured
    allowlist (and the JIT module loads), so the autotuner races the kernel
    as itself exactly where it is dispatchable and skips it elsewhere.
    ``supports_inputs`` / ``is_tactic_compatible`` let bucketed cache
    entries recorded on an allowlisted shape fall back on shapes the kernel
    does not serve.
    """

    def __init__(self, use_8x4_sf_layout: bool = False):
        self._use_8x4_sf_layout = use_8x4_sf_layout

    def get_cache_key_extras(self, inputs) -> tuple:
        a, b, _, _, _, out, _ = inputs
        return (a.dtype, b.dtype, out.dtype)

    def _eligible(self, inputs) -> bool:
        a, b, a_descale, b_descale, _, out, _ = inputs
        reason = _specialized_supported(
            a, b, a_descale, b_descale, out, out.dtype, self._use_8x4_sf_layout
        )
        return reason is None and _get_specialized_module() is not None

    def supports_inputs(self, inputs) -> bool:
        return self._eligible(inputs)

    def is_tactic_compatible(self, inputs, tactic) -> bool:
        return tactic in (-1, 0) and self._eligible(inputs)

    def get_valid_tactics(self, inputs, profile):
        return [-1] if self._eligible(inputs) else []

    def precompile_tactics(self, inputs, tactics, profile, **kwargs) -> bool:
        _get_specialized_module()
        return True

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, a_descale, b_descale, _, out, _ = inputs
        # The autotuner's v2 default-racing prepends tactic -1 for runners[0]
        # even when get_valid_tactics returned [], and bucketed profiles can
        # synthesize shapes off the allowlist. The kernel must never run
        # outside its measured contract, so forward re-checks eligibility and
        # refuses: the profiling loop records a refused tactic as failed
        # (time inf) and moves on, and the ranking path's direct
        # forward(tactic=-1, do_preparation=True) call cannot reach this
        # runner because precompile_tactics handles preparation.
        reason = _specialized_supported(
            a, b, a_descale, b_descale, out, out.dtype, self._use_8x4_sf_layout
        )
        if reason is not None:
            raise NotImplementedError(
                f"mm_mxfp8 specialized SM107 kernel cannot serve this profile ({reason})"
            )
        if not _launch_specialized(a, b, a_descale, b_descale, out):
            raise RuntimeError(
                "mm_mxfp8 specialized SM107 kernel: JIT module unavailable "
                f"({_stats['module_error']})"
            )
        _stats["backend_dispatches"]["specialized"] += 1
        return out


# ---------------------------------------------------------------------------
# Reference helpers: unswizzle + dequant (used by the test suite as the
# quantized-exact reference for the kernel's scale decode). Explicit
# backend="specialized" calls the kernel declines RAISE — there is no
# fallback path on the explicit route.
# ---------------------------------------------------------------------------


def _unswizzle_sf(sf: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Linear (rows, k//32) view of a 1D swizzled 128x4 UE8M0 scale buffer."""
    k4 = (k // 32 + 3) // 4 * 4
    r_pad = (rows + 127) // 128 * 128
    t = sf[: r_pad * k4].view(r_pad // 128, k4 // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    return t.reshape(r_pad, k4)[:rows, : k // 32]


def _dequant(q: torch.Tensor, sf_lin: torch.Tensor) -> torch.Tensor:
    rows, k = q.shape
    scale = torch.pow(2.0, sf_lin.float() - 127.0)
    return (q.float().view(rows, k // 32, 32) * scale.unsqueeze(-1)).view(rows, k)


def dispatch_mm_mxfp8_specialized(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: torch.Tensor,
    out_dtype: torch.dtype,
    use_8x4_sf_layout: bool,
    explicit: bool,
) -> bool:
    """Specialized-backend dispatch for ``mm_mxfp8``.

    Returns True when ``out`` was written by the specialized kernel. Returns
    False only for ``backend="auto"`` calls the backend declined, which then
    continue into the stock path. Explicit ``backend="specialized"`` calls
    the backend declines raise ``NotImplementedError`` with the refusal
    reason — an explicitly selected backend never falls back. The decline
    reason is recorded in ``fallback_reasons`` either way.
    """
    global _marker_logged
    _stats["calls"] += 1
    reason = _specialized_supported(
        a, b, a_descale, b_descale, out, out_dtype, use_8x4_sf_layout
    )
    if reason is None:
        if _launch_specialized(a, b, a_descale, b_descale, out):
            _stats["backend_dispatches"]["specialized"] += 1
            if not _marker_logged:
                _marker_logged = True
                logger.info(
                    "flashinfer.mm_mxfp8: specialized SM107 kernel dispatched "
                    "(M=%d, N=%d, K=%d)",
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                )
            return True
        reason = "jit_unavailable"
    _bump_fallback(reason)
    if explicit:
        # An explicitly selected backend never falls back: refuse loudly with
        # the reason instead of serving the call on any other implementation.
        _stats["explicit_refusals"] += 1
        raise NotImplementedError(
            f"mm_mxfp8 backend='specialized' cannot serve this call ({reason}); "
            "an explicitly selected backend never falls back — use "
            "backend='auto' to route among the supported backends"
        )
    return False
