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
"""

import functools
import json
import logging
import os
import threading
from importlib import resources
from typing import Any, Dict, Optional, Tuple

import torch

from . import cake_concat_mla_kv_quant_fp8
from .api_logging import flashinfer_api
from .jit.cake_concat_mla_kv_quant_fp8 import (
    ROUTES,
    concat_mla_kv_quant_fp8_route_record,
    concat_mla_kv_quant_fp8_target_for_capability,
    load_concat_mla_kv_quant_fp8_build,
    route_key,
)
from .jit.mla_kv_pack import gen_mla_kv_pack_fp8_module
from .trace.templates.attention import concat_mla_kv_quant_fp8_trace
from .utils import get_compute_capability

logger = logging.getLogger(__name__)

# Kill switch shared by FlashInfer's specialized kernels: read at CALL time.
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Fixed head geometry of the fused kernels (DeepSeek/Kimi MLA).
NOPE_DIM = 128
ROPE_DIM = 64
V_DIM = 128
_KV_DIM = NOPE_DIM + V_DIM
# Largest finite float8_e4m3fn magnitude; the fused kernels saturate to it.
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_QK_DIM = NOPE_DIM + ROPE_DIM

# Two fused backends behind one API, selected per call:
#   "specialized": the hand-written CUDA kernel (include/flashinfer/mla_kv_pack_fp8.cuh),
#                  one warp per token, compiled for every compute capability >= 10.0;
#                  carries a fully unrolled variant for 12 local heads.
#   "cake":        generated Cake programs (one per head group), exact targets
#                  sm_100a / sm_103a.
#   "auto":        the backend measured fastest for the shape: "specialized" for
#                  the head counts in _SPECIALIZED_AUTO_HEADS, "cake" otherwise;
#                  the other backend is tried when the preferred one cannot serve
#                  the call (layout, target), and the torch path serves the rest.
BACKENDS = ("auto", "specialized", "cake")
_FUSED_BACKENDS = ("specialized", "cake")
# Head counts at which "auto" prefers the specialized kernel. GB300 / B200
# kernel-level A/B (cold L2, 18 serving shapes): the unrolled 12-head variant is
# 0-3 % faster than the Cake program at every token count, while for every other
# head count the Cake programs are 1.03-1.4x faster (more bytes in flight per
# warp). 12 = Kimi-K3's 96 heads over TP8.
_SPECIALIZED_AUTO_HEADS = frozenset({12})

# Host plan of the generated Cake programs (a replica of the Cake production
# launcher's ``plan_head_group`` / ``plan_grid``): one warp group per token,
# every warp owns ``head_group`` consecutive head pairs, eight warps per CTA.
_WARPS_PER_CTA = 8
_THREADS_PER_CTA = 32 * _WARPS_PER_CTA
# Below this token count prefer two pairs per warp (more CTAs in the one-wave
# regime) whenever the head-pair count is even.
_SMALL_T_TOKENS = 2048
# Most head pairs (4 KB of input) a warp keeps in flight within the register budget.
_MAX_HEAD_GROUP = 4

# Dispatch surface of the fused kernels (package data; see the file's note).
_WORKLOAD_FILE = "mla_kv_pack_fp8_workloads.json"


@functools.cache
def _load_allowlist() -> Optional[dict]:
    """Load the fused kernels' dispatch surface; ``None`` => never dispatch."""
    try:
        payload = json.loads(
            resources.files("flashinfer").joinpath(_WORKLOAD_FILE).read_text()
        )
        return {
            "min_cc": tuple(int(v) for v in payload["min_compute_capability"]),
            "heads": (
                int(payload["num_heads"]["min"]),
                int(payload["num_heads"]["max"]),
            ),
            "tokens": (
                int(payload["num_tokens"]["min"]),
                int(payload["num_tokens"]["max"]),
            ),
        }
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            "flashinfer.concat_mla_kv_quant_fp8: workload allowlist unavailable "
            "(%s: %s); using the composable torch path",
            type(exc).__name__,
            str(exc)[:200],
        )
        return None


def _plan_head_group(num_tokens: int, num_heads: int) -> int:
    """Head pairs per warp: the fewest warps per token that keep at most four
    pairs (4 KB of loads in flight) per warp, with the pairs spread evenly over
    those warps; two for one-wave shapes.

    Exact replica of the Cake production launcher's rule, which selects the
    generated program: ``T <= 2048`` with an even pair count takes two pairs per
    warp (more, smaller CTAs for a one-wave launch); otherwise ``ceil(P / 4)``
    warps share the token's ``P = ceil(H / 2)`` pairs and each warp takes
    ``ceil(P / warps)`` of them (one pair for H <= 2, three for H in {5, 6}, for
    H in {9, 10}: 3 + 2 rather than 4 + 1, for H in {17, 18}: 3 + 3 + 3, and
    for H in {11, 12}: 3 + 3 rather than 4 + 2; four for H = 24 .. 128).
    """
    head_pairs = (int(num_heads) + 1) // 2
    if int(num_tokens) <= _SMALL_T_TOKENS and head_pairs % 2 == 0:
        return 2
    warps_per_token = (head_pairs + _MAX_HEAD_GROUP - 1) // _MAX_HEAD_GROUP
    return (head_pairs + warps_per_token - 1) // warps_per_token


def _plan_grid(num_tokens: int, num_heads: int, head_group: int) -> Tuple[int, int]:
    """``(grid_x, warps_per_token)``: ``ceil(P / head_group)`` warps per token, eight warps per CTA."""
    head_pairs = (int(num_heads) + 1) // 2
    warps_per_token = (head_pairs + int(head_group) - 1) // int(head_group)
    grid = max(
        1, (int(num_tokens) * warps_per_token + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA
    )
    return grid, warps_per_token


# Cake modules, keyed by (exact target, head group).
_modules: Dict[Tuple[str, int], Any] = {}
_module_errors: Dict[Any, str] = {}
_module_lock = threading.Lock()
# The specialized kernel's single module (both head-count variants in one build).
_specialized_module: Any = None
_SPECIALIZED_MODULE_KEY = "specialized"
_stats: Dict[str, Any] = {
    "calls": 0,
    # Every fused dispatch, whichever backend served it.
    "specialized_dispatches": 0,
    "backend_dispatches": {"specialized": 0, "cake": 0},
    "fallback_dispatches": 0,
    "fallback_reasons": {},
    "module_loaded": False,
    "module_error": None,
}
_marker_logged = False


def _bump_fallback(reason: str) -> None:
    _stats["fallback_dispatches"] += 1
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _get_module(target: str, head_group: int):
    """Build/load the Cake (target, head group) JIT module once; never inside a CUDA-graph capture."""
    key = (target, int(head_group))
    module = _modules.get(key)
    if module is not None:
        return module
    with _module_lock:
        if key not in _modules and key not in _module_errors:
            try:
                _modules[key] = load_concat_mla_kv_quant_fp8_build(
                    target, int(head_group)
                )
                _stats["module_loaded"] = True
            except Exception as exc:  # noqa: BLE001 - guard must never break the stock path
                error = f"{type(exc).__name__}: {exc}"[:500]
                _module_errors[key] = error
                _stats["module_error"] = error
                logger.warning(
                    "flashinfer.concat_mla_kv_quant_fp8: JIT build failed for %s head group %d, "
                    "using the composable torch path: %s",
                    target,
                    int(head_group),
                    error,
                )
    return _modules.get(key)


def _get_specialized_module():
    """Build/load the specialized kernel's JIT module once; never inside a CUDA-graph capture."""
    global _specialized_module
    if _specialized_module is not None:
        return _specialized_module
    with _module_lock:
        if (
            _specialized_module is None
            and _SPECIALIZED_MODULE_KEY not in _module_errors
        ):
            try:
                _specialized_module = gen_mla_kv_pack_fp8_module().build_and_load()
                _stats["module_loaded"] = True
            except Exception as exc:  # noqa: BLE001 - guard must never break the stock path
                error = f"{type(exc).__name__}: {exc}"[:500]
                _module_errors[_SPECIALIZED_MODULE_KEY] = error
                _stats["module_error"] = error
                logger.warning(
                    "flashinfer.concat_mla_kv_quant_fp8: JIT build of the specialized "
                    "kernel failed, trying the other backends: %s",
                    error,
                )
    return _specialized_module


def _concat_mla_kv_quant_fp8_stats() -> dict:
    """Diagnostics: dispatch counters, JIT state and the compile footprint.

    ``specialized_dispatches`` counts every fused dispatch; ``backend_dispatches``
    splits it by backend. The Cake backend is one generated program per head
    group (1, 2, 3 or 4 head pairs per warp, planned from ``(num_tokens,
    num_heads)`` on the host); the specialized backend is one module holding
    the 12-head unrolled variant and the runtime-head-count variant. Token and
    head counts are runtime launch arguments on both, so nothing is compiled
    per shape. ``compiled_variants`` counts the modules loaded so far;
    ``distinct_kernels_for_allowlist`` is the Cake route table's program count.
    """
    allowlist = _load_allowlist()
    return {
        **_stats,
        "backend_dispatches": dict(_stats["backend_dispatches"]),
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "module_errors": dict(_module_errors),
        "allowlist_loaded": allowlist is not None,
        "allowlist": allowlist,
        "backends": BACKENDS,
        "auto_specialized_heads": tuple(sorted(_SPECIALIZED_AUTO_HEADS)),
        "compiled_variants": len(_modules) + (1 if _specialized_module else 0),
        "specialized_module_loaded": _specialized_module is not None,
        "distinct_kernels_for_allowlist": len(ROUTES),
        "distinct_kernels": "cake: one generated program per head group "
        + ", ".join(sorted(ROUTES))
        + " (head group planned from num_tokens and num_heads); specialized: "
        "num_heads==12 (unrolled) + runtime-heads variant (T and H are runtime "
        "arguments on both)",
        "precompiled": bool(_modules) or _specialized_module is not None,
    }


def _fallback(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    nope_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Composable stock path: two casts + two strided copies (any geometry).

    Byte-identical to the fused kernels' saturating cast on every supported
    torch: NaN payloads lose their sign bit (torch < 2.13's software cast keeps
    it and encodes ``-NaN`` as ``0xFF``; the kernels and torch >= 2.13 emit the
    canonical ``0x7F``) and finite values are clamped to the e4m3fn range
    before ``Tensor.to`` (torch < 2.13 encodes finite overflow and ``+-inf`` as
    NaN, while the kernels saturate to ``+-448``).
    """
    num_tokens = kv_nope.shape[0]
    kv_fp8 = _saturating_cast(kv_nope, key.dtype)
    key[..., :nope_dim].copy_(kv_fp8[..., :nope_dim])
    key[..., nope_dim:].copy_(
        _saturating_cast(k_pe, key.dtype)
        .reshape(num_tokens, 1, k_pe.shape[-1])
        .expand(num_tokens, kv_nope.shape[1], k_pe.shape[-1])
    )
    value.copy_(kv_fp8[..., nope_dim:])
    return key, value


def _saturating_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x.to(dtype)`` with the fused kernels' NaN and overflow encoding on any
    torch build: every NaN (either sign) -> canonical ``0x7F``, finite overflow
    and ``+-inf`` -> ``+-448``."""
    if x.dtype == dtype:
        return x
    x = torch.where(torch.isnan(x), x.abs(), x)
    return x.clamp(-_FP8_MAX, _FP8_MAX).to(dtype)


def _k_pe_rows_admissible(k_pe: torch.Tensor) -> bool:
    """Cake backend: ``k_pe`` ``[T, 64]`` may be a column slice of a wider
    row-major workspace (vLLM's non-DCP prefill passes the last 64 columns of
    the ``[T, 576]`` latent): unit last stride and a 32-byte-aligned row stride
    (a multiple of 16 elements) of at least one row, because the generated
    kernels issue 256-bit lane loads.  They take the row stride in elements."""
    if k_pe.dim() != 2:
        return False
    if k_pe.shape[0] == 1:
        return k_pe.stride(1) == 1
    return (
        k_pe.stride(1) == 1 and k_pe.stride(0) % 16 == 0 and k_pe.stride(0) >= ROPE_DIM
    )


def _common_supported(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    nope_dim: int,
) -> Optional[str]:
    """GPU-free guards shared by both fused backends; ``None`` when they pass,
    else the reason. Device guards follow the per-backend layout guards."""
    if os.environ.get(_DISABLE_ENV) == "1":
        return "kill_switch"
    if kv_nope.dtype != torch.bfloat16 or k_pe.dtype != torch.bfloat16:
        return "input_dtype"
    if key.dtype != torch.float8_e4m3fn or value.dtype != torch.float8_e4m3fn:
        return "output_dtype"
    if (
        nope_dim != NOPE_DIM
        or kv_nope.shape[-1] != _KV_DIM
        or k_pe.shape[-1] != ROPE_DIM
    ):
        return "head_geometry"
    if key.shape[-1] != _QK_DIM or value.shape[-1] != V_DIM:
        return "output_geometry"
    if not (kv_nope.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        return "non_contiguous"
    if k_pe.dim() != 2 or k_pe.stride(1) != 1:
        return "non_contiguous"
    allowlist = _load_allowlist()
    if allowlist is None:
        return "allowlist_unavailable"
    num_tokens, num_heads = kv_nope.shape[0], kv_nope.shape[1]
    if not allowlist["heads"][0] <= num_heads <= allowlist["heads"][1]:
        return "num_heads_not_allowlisted"
    if not allowlist["tokens"][0] <= num_tokens <= allowlist["tokens"][1]:
        return "num_tokens_not_allowlisted"
    return None


def _device_supported(kv_nope: torch.Tensor) -> Optional[str]:
    """Device guards shared by both backends, checked after their layout guards."""
    if not kv_nope.is_cuda:
        return "device"
    allowlist = _load_allowlist()
    if (
        allowlist is None
        or get_compute_capability(kv_nope.device) < allowlist["min_cc"]
    ):
        return "compute_capability"
    return None


def _cake_supported(
    kv_nope: torch.Tensor, k_pe: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Optional[str]:
    """Cake-backend guards (after ``_common_supported``): layout first, device last."""
    if not _k_pe_rows_admissible(k_pe):
        return "non_contiguous"
    # 256-bit lane loads: every base must be 32-byte aligned (torch allocations
    # are; a 16-byte-aligned slice is not and takes the fallback).
    if any(t.data_ptr() % 32 for t in (kv_nope, k_pe, key, value)):
        return "alignment"
    num_tokens, num_heads = kv_nope.shape[0], kv_nope.shape[1]
    if route_key(_plan_head_group(num_tokens, num_heads)) not in ROUTES:
        return "head_group_route"
    reason = _device_supported(kv_nope)
    if reason is not None:
        return reason
    target = concat_mla_kv_quant_fp8_target_for_capability(
        get_compute_capability(kv_nope.device)
    )
    if target is None:
        # A 10.x / 12.x part the exact sm_100a / sm_103a programs are not built for.
        return "exact_target"
    if (
        torch.cuda.is_current_stream_capturing()
        and _modules.get((target, _plan_head_group(num_tokens, num_heads))) is None
    ):
        # Never compile/load inside a capture; the stock path is capture-safe.
        return "capturing_before_jit"
    return None


def _specialized_backend_supported(
    kv_nope: torch.Tensor, k_pe: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> Optional[str]:
    """Specialized-backend guards (after ``_common_supported``): contiguous
    ``k_pe`` and 16-byte-aligned bases (128-bit vector loads/stores), then any
    compute capability >= 10.0; layout first, device last."""
    if not k_pe.is_contiguous():
        return "non_contiguous"
    if any(t.data_ptr() % 16 for t in (kv_nope, k_pe, key, value)):
        return "alignment"
    reason = _device_supported(kv_nope)
    if reason is not None:
        return reason
    if torch.cuda.is_current_stream_capturing() and _specialized_module is None:
        return "capturing_before_jit"
    return None


_BACKEND_CHECKS = {
    "specialized": _specialized_backend_supported,
    "cake": _cake_supported,
}


def _backend_order(backend: str, num_heads: int) -> Tuple[str, ...]:
    """Fused backends to try, most preferred first."""
    if backend != "auto":
        return (backend,)
    if int(num_heads) in _SPECIALIZED_AUTO_HEADS:
        return ("specialized", "cake")
    return ("cake", "specialized")


def _specialized_supported(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    nope_dim: int,
    backend: str = "auto",
) -> Optional[str]:
    """Return ``None`` when a fused backend serves this call, else the reason.

    With ``backend="auto"`` the reason is the one the last backend in the
    preference order gave, so a call that only one backend declines reads as
    that backend's reason."""
    reason = _common_supported(kv_nope, k_pe, key, value, nope_dim)
    if reason is not None:
        return reason
    for candidate in _backend_order(backend, kv_nope.shape[1]):
        reason = _BACKEND_CHECKS[candidate](kv_nope, k_pe, key, value)
        if reason is None:
            return None
    return reason


def _launch_cake(
    kv_nope: torch.Tensor, k_pe: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> bool:
    num_tokens, num_heads = int(kv_nope.shape[0]), int(kv_nope.shape[1])
    head_group = _plan_head_group(num_tokens, num_heads)
    route = concat_mla_kv_quant_fp8_route_record(kv_nope.device, head_group)
    module = _get_module(route["target"], head_group)
    if module is None:
        return False
    grid, warps_per_token = _plan_grid(num_tokens, num_heads, head_group)
    cake_concat_mla_kv_quant_fp8.launch(
        kv_nope,
        k_pe,
        key,
        value,
        head_group=head_group,
        warps_per_token=warps_per_token,
        grid=grid,
        module=module,
        record=route["module"],
    )
    return True


def _launch_specialized(
    kv_nope: torch.Tensor, k_pe: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> bool:
    module = _get_specialized_module()
    if module is None:
        return False
    module.concat_mla_kv_quant_fp8(kv_nope, k_pe, key, value)
    return True


_BACKEND_LAUNCH = {"specialized": _launch_specialized, "cake": _launch_cake}


@flashinfer_api(trace=concat_mla_kv_quant_fp8_trace)
def concat_mla_kv_quant_fp8(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    *,
    nope_dim: int = NOPE_DIM,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused MLA context K/V pack with fp8 e4m3 quantization.

    Turns the per-head ``kv_b_proj`` output ``kv_nope = [k_nope | v]`` and the
    shared, already-RoPE'd ``k_pe`` into the fp8 ``key = [k_nope | k_pe]`` /
    ``value = v`` buffers a ragged MLA prefill kernel consumes (e.g.
    :func:`trtllm_ragged_attention_deepseek` with separate Q/K/V), in one
    memory pass instead of two casts and two strided copies.

    Numerics: the cast is the saturating e4m3 conversion (round to nearest
    even; finite overflow and ``+-inf`` -> ``+-448``; NaN -> ``0x7F``), which
    is what ``Tensor.to(torch.float8_e4m3fn)`` produces on the GPU since
    torch 2.13. Output is bit-exact with that cast plus slice copies, on every
    backend.

    Two fused backends serve bf16 inputs with head geometry ``nope_dim=128,
    rope_dim=64, v_dim=128`` within the head-count and token-count surface of
    the package's ``mla_kv_pack_fp8_workloads.json`` (1..128 heads, 1..131072
    tokens) on compute capability 10.0 and newer:

    - ``"specialized"``: the hand-written CUDA kernel (one warp per token, a
      fully unrolled variant for 12 local heads, runtime head count otherwise),
      compiled for every compute capability >= 10.0. Needs contiguous,
      16-byte-aligned tensors.
    - ``"cake"``: generated Cake programs (one per head group, i.e. the number
      of head pairs a warp owns, planned from the token and head counts) built
      for the exact compute capabilities 10.0 and 10.3. Needs 32-byte-aligned
      tensors; ``k_pe`` may be a row-strided column slice of a wider row-major
      workspace (unit last stride, row stride a multiple of 16 elements).
    - ``"auto"`` (default): ``"specialized"`` for 12 local heads (96 heads over
      TP8, where it measured 0-3 % faster on GB300/B200 at every token count),
      ``"cake"`` for every other head count (1.03-1.4x faster there); if the
      preferred backend cannot serve the call the other one is tried.

    Every call no fused backend serves takes the composable torch path.
    ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` forces the composable path
    (read at call time).

    Parameters
    ----------
    kv_nope : torch.Tensor
        ``[num_tokens, num_heads, nope_dim + v_dim]`` bf16 (contiguous for the
        fused paths).
    k_pe : torch.Tensor
        ``[num_tokens, rope_dim]`` or ``[num_tokens, 1, rope_dim]`` bf16,
        shared across heads; contiguous, or for the Cake backend a column
        slice of a row-major workspace whose row stride is a multiple of 16
        elements (e.g. ``latent[:, 512:]`` of a ``[num_tokens, 576]`` buffer).
    key : Optional[torch.Tensor]
        ``[num_tokens, num_heads, nope_dim + rope_dim]`` float8_e4m3fn output;
        allocated when ``None``.
    value : Optional[torch.Tensor]
        ``[num_tokens, num_heads, v_dim]`` float8_e4m3fn output; allocated
        when ``None``.
    nope_dim : int
        Split point of ``kv_nope``'s last dim (default 128).
    backend : str
        ``"auto"`` (default), ``"specialized"`` or ``"cake"``. An explicit
        fused backend that cannot serve the call takes the composable path.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(key, value)``.
    """
    global _marker_logged
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if kv_nope.dim() != 3:
        raise ValueError("kv_nope must be [num_tokens, num_heads, nope_dim + v_dim]")
    num_tokens, num_heads, kv_dim = kv_nope.shape
    if k_pe.dim() == 3:
        if k_pe.shape[1] != 1:
            raise ValueError(
                "k_pe must be [num_tokens, rope_dim] or [num_tokens, 1, rope_dim]"
            )
        k_pe = k_pe.reshape(num_tokens, k_pe.shape[-1])
    elif k_pe.dim() != 2:
        raise ValueError(
            "k_pe must be [num_tokens, rope_dim] or [num_tokens, 1, rope_dim]"
        )
    if k_pe.shape[0] != num_tokens:
        raise ValueError("k_pe and kv_nope must have the same num_tokens")
    if not 0 < nope_dim < kv_dim:
        raise ValueError("nope_dim must split kv_nope's last dim")
    rope_dim = k_pe.shape[-1]
    v_dim = kv_dim - nope_dim
    fp8 = torch.float8_e4m3fn
    if key is None:
        key = torch.empty(
            num_tokens, num_heads, nope_dim + rope_dim, dtype=fp8, device=kv_nope.device
        )
    if value is None:
        value = torch.empty(
            num_tokens, num_heads, v_dim, dtype=fp8, device=kv_nope.device
        )
    if key.shape != (num_tokens, num_heads, nope_dim + rope_dim):
        raise ValueError("key has the wrong shape")
    if value.shape != (num_tokens, num_heads, v_dim):
        raise ValueError("value has the wrong shape")
    if key.dtype != fp8 or value.dtype != fp8:
        raise ValueError("key and value must have dtype torch.float8_e4m3fn")

    _stats["calls"] += 1
    reason = _common_supported(kv_nope, k_pe, key, value, nope_dim)
    served = None
    if reason is None and num_tokens > 0:
        for candidate in _backend_order(backend, num_heads):
            reason = _BACKEND_CHECKS[candidate](kv_nope, k_pe, key, value)
            if reason is not None:
                continue
            if _BACKEND_LAUNCH[candidate](kv_nope, k_pe, key, value):
                served = candidate
                break
            reason = "jit_unavailable"
    if served is None:
        if reason is not None:
            _bump_fallback(reason)
            return _fallback(kv_nope, k_pe, key, value, nope_dim)
        return key, value  # num_tokens == 0
    _stats["specialized_dispatches"] += 1
    _stats["backend_dispatches"][served] += 1
    if not _marker_logged:
        _marker_logged = True
        logger.info(
            "flashinfer.concat_mla_kv_quant_fp8: fused kernel dispatched "
            "(backend=%s, num_heads=%d, num_tokens=%d)",
            served,
            num_heads,
            num_tokens,
        )
    return key, value
