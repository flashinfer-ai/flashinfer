# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Specialized SM90 decode-M dense fp8 block-scale GEMM backend.

An agent-generated kernel for the decode-M classes of
:func:`flashinfer.gemm.fp8_blockscale_gemm_sm90` in its BF16-activation /
FP8-weight / 128x128-block-scale form: the activation 1x128 per-token-group
quantization happens inside the kernel. It uses the FP32 multiplier
448/amax without an epsilon floor, or 1 for an all-zero group, and stores
the reciprocal multiplier for dequantization. Weights and their
(ceil(N/128), K/128) float32 scale grid are consumed as passed.

Backend surface (#5667 semantics), driven from
``fp8_blockscale_gemm_sm90(..., backend=...)``:

- ``"auto"`` (the public default) routes here only on the packaged
  allowlist (``flashinfer/fp8bs_gemm_decodem_sm90_dispatch.json``): M is
  interpolated within the decode band for measured (N, K) pairs only.
  Otherwise it is exactly the stock route.
- ``"specialized"`` selects this kernel explicitly; a call it cannot serve
  raises ``NotImplementedError`` with the refusal reason (also recorded in
  the dispatch stats) — an explicitly selected backend never falls back;
  only ``"auto"`` routes around a refusal.
- ``"deepgemm"`` pins the stock route; an explicitly selected backend is
  never silently swapped.
- ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read per call; shared by
  FlashInfer's specialized kernels) disables this kernel: ``"auto"`` keeps
  the stock route, explicit ``"specialized"`` raises.

Introspection: :func:`_fp8bs_decodem_stats` reports per-backend dispatch
counts and a ``fallback_reasons`` census. Guard order is GPU-free checks →
layout guards → device guards; a JIT build failure is logged once — under
``"auto"`` the call is served by the stock route, explicit ``"specialized"``
raises.
"""

import functools
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

SPECIALIZED_KERNEL_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

FP8BS_GEMM_BACKENDS = ("auto", "specialized", "deepgemm")

_DISPATCH_FILE = "fp8bs_gemm_decodem_sm90_dispatch.json"

# Kernel coverage limit, independent of the packaged auto policy.
_M_MAX = 24
# Retained for launch ABI stability; fused quantization does not use epsilon.
_QUANT_EPS = 0.0


class GraphCaptureJitError(RuntimeError):
    """The specialized kernel is not built and JIT compilation cannot run
    under CUDA graph capture; warm the allowlisted shape up eagerly first.
    Kept distinct from other :class:`RuntimeError`\\ s (e.g. a JIT build
    failure) so only this error surfaces instead of falling back."""


_stats: Dict[str, Any] = {
    "backend_dispatches": {"specialized": 0, "deepgemm": 0},
    "fallback_reasons": {},
    "jit_build": "not attempted",
}

# Serializes the cold JIT build and its `_stats["jit_build"]` bookkeeping:
# `functools.cache` does not prevent two concurrent cold calls, and a failing
# call racing a succeeding one must not record `"failed"` over `"built"`
# (which would strand the already-cached module behind the failure guard).
_jit_lock = threading.Lock()


def _fp8bs_decodem_stats() -> Dict[str, Any]:
    out = dict(_stats)
    out["backend_dispatches"] = dict(_stats["backend_dispatches"])
    out["fallback_reasons"] = dict(_stats["fallback_reasons"])
    out["backends"] = FP8BS_GEMM_BACKENDS
    return out


def _reset_fp8bs_decodem_stats() -> None:
    _stats["backend_dispatches"] = {"specialized": 0, "deepgemm": 0}
    _stats["fallback_reasons"] = {}


def _fallback(reason: str) -> None:
    fr = _stats["fallback_reasons"]
    fr[reason] = fr.get(reason, 0) + 1


def _disabled() -> bool:
    return os.environ.get(SPECIALIZED_KERNEL_DISABLE_ENV, "0") not in ("", "0")


def _positive_pair(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(d) is int and d > 0 for d in value)
    )


@functools.cache
def _dispatch_table() -> Optional[Dict[str, Any]]:
    """The packaged allowlist that gates ``backend="auto"`` routing (the
    compiled-in guards in :func:`_eligibility` stay authoritative)."""
    try:
        import importlib.resources
        import json

        data = (importlib.resources.files("flashinfer") / _DISPATCH_FILE).read_text()
        table = json.loads(data)
        if table.get("backends") != list(FP8BS_GEMM_BACKENDS):
            return None
        op = table["ops"]["fp8_blockscale_gemm_sm90"]
        band = op["m_band"]
        if not _positive_pair(band) or not 1 <= band[0] <= band[1] <= _M_MAX:
            return None
        alignment = op["alignment"]
        n_step, k_step = alignment["n_multiple_of"], alignment["k_multiple_of"]
        if not all(type(v) is int and v > 0 for v in (n_step, k_step)):
            return None
        if n_step % 64 or k_step % 128:
            return None
        pairs = op["measured_nk_classes"]
        if (
            not isinstance(pairs, list)
            or not pairs
            or not all(
                _positive_pair(pair) and pair[0] % n_step == 0 and pair[1] % k_step == 0
                for pair in pairs
            )
        ):
            return None
        if len(set(map(tuple, pairs))) != len(pairs):
            return None
        return table
    except Exception:
        return None


def _auto_refusal(m: int, n: int, k: int) -> Optional[str]:
    table = _dispatch_table()
    if table is None:
        return "no_dispatch_table"
    op = table["ops"]["fp8_blockscale_gemm_sm90"]
    lo, hi = op["m_band"]
    if not lo <= m <= hi:
        return "m_band"
    alignment = op["alignment"]
    if n % alignment["n_multiple_of"] or k % alignment["k_multiple_of"]:
        return "shape_alignment"
    if [n, k] not in op["measured_nk_classes"]:
        return "unmeasured_nk"
    return None


def _eligibility(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    out: torch.Tensor,
) -> Tuple[bool, str]:
    """(eligible, reason). Runs AFTER the stock entry's own consistency
    validation, so shapes/dtypes here are already self-consistent; this
    checks only the specialized kernel's coverage."""
    # GPU-free checks
    if _disabled():
        return False, "kill_switch"
    if input.dtype != torch.bfloat16:
        return False, "input_dtype"
    if weight.dtype != torch.float8_e4m3fn:
        return False, "weight_dtype"
    if input_scale is not None:
        return False, "external_input_scale"
    M, K = input.shape
    N = weight.shape[0]
    if K == 0:
        # the stock route owns the K==0 contract (it rejects null inputs)
        return False, "k_zero"
    if N == 0:
        # a zero-width grid is not launchable; the stock route owns the
        # N==0 contract too (it rejects a null weight pointer)
        return False, "n_zero"
    if weight_scale is None or weight_scale.shape != (
        (N + 127) // 128,
        K // 128,
    ):
        return False, "weight_scale_granularity"
    if not (1 <= M <= _M_MAX):
        return False, "m_band"
    if out.dtype != torch.bfloat16:
        return False, "out_dtype"
    # layout guards
    if not (
        input.is_contiguous()
        and weight.is_contiguous()
        and weight_scale.is_contiguous()
        and out.is_contiguous()
    ):
        return False, "layout"
    # the kernel issues 16-byte vector loads from the activation and weight
    # base pointers; a contiguous view with a storage offset can break that
    if input.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0:
        return False, "alignment"
    # the kernel reads its operands while CTAs write `out`; storage that
    # overlaps an input is a data race (the stock route's aliasing behavior
    # is its own contract)
    out_lo = out.data_ptr()
    out_hi = out_lo + out.numel() * out.element_size()
    for t in (input, weight, weight_scale):
        t_lo = t.data_ptr()
        t_hi = t_lo + t.numel() * t.element_size()
        if max(out_lo, t_lo) < min(out_hi, t_hi):
            return False, "overlap"
    # device guards
    if not input.is_cuda:
        return False, "device"
    if (
        weight.device != input.device
        or weight_scale.device != input.device
        or out.device != input.device
    ):
        return False, "device"
    major, minor = torch.cuda.get_device_capability(input.device)
    if major != 9:
        return False, "device"
    return True, "ok"


@functools.cache
def _get_module() -> Any:
    from ..jit.gemm import gen_fp8bs_gemm_decodem_sm90_module

    spec = gen_fp8bs_gemm_decodem_sm90_module()
    if torch.cuda.is_current_stream_capturing():
        # Loading a packaged or current on-disk module needs no compiler.
        # Never enter build_and_load's compilation fallback during capture.
        module = spec.try_load()
        if module is None:
            raise GraphCaptureJitError(
                "fp8bs_gemm_decodem_sm90: the specialized kernel is not built "
                "yet and JIT compilation is not allowed under CUDA graph "
                "capture. Call fp8_blockscale_gemm_sm90 once eagerly on an "
                "allowlisted shape first."
            )
    else:
        module = spec.build_and_load()
    _stats["jit_build"] = "built"
    return module


def _fp8bs_decodem_try_dispatch(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    requested: str,
) -> Optional[torch.Tensor]:
    """Serve the call with the specialized kernel, or return None (with the
    reason recorded) so an ``"auto"`` caller continues on the stock route.
    Explicit ``requested="specialized"`` calls the kernel declines raise
    ``NotImplementedError`` with the refusal reason — an explicitly selected
    backend never falls back."""
    eligible, reason = _eligibility(input, weight, input_scale, weight_scale, out)
    if eligible and requested == "auto":
        refusal = _auto_refusal(input.shape[0], weight.shape[0], input.shape[1])
        if refusal is not None:
            eligible, reason = False, refusal
    if not eligible:
        _fallback(f"{requested}: {reason}")
        if requested == "specialized":
            raise NotImplementedError(
                "fp8_blockscale_gemm_sm90 backend='specialized' cannot serve "
                f"this call ({reason}); an explicitly selected backend never "
                "falls back — use backend='auto' or 'deepgemm'"
            )
        return None
    with _jit_lock:
        state = _stats["jit_build"]
        if state.startswith("failed"):
            _fallback(f"{requested}: jit_build_failure")
            if requested == "specialized":
                raise NotImplementedError(
                    "fp8_blockscale_gemm_sm90 backend='specialized' JIT build "
                    f"previously failed ({state}); an explicitly selected backend "
                    "never falls back — use backend='auto' or 'deepgemm'"
                )
            return None
        try:
            # Check graph capture on the device whose stream runs the kernel.
            with torch.cuda.device(input.device):
                module = _get_module()
        except GraphCaptureJitError:
            raise
        except Exception as e:
            if state == "not attempted":
                _stats["jit_build"] = f"failed ({str(e)[:120]})"
                logger.warning(
                    "fp8bs_gemm_decodem_sm90: specialized kernel JIT build failed%s",
                    "; serving via the stock route"
                    if requested == "auto"
                    else "; explicit backend='specialized' raises",
                    exc_info=True,
                )
            _fallback(f"{requested}: jit_build_failure")
            if requested == "specialized":
                raise NotImplementedError(
                    "fp8_blockscale_gemm_sm90 backend='specialized' JIT build "
                    f"failed ({str(e)[:200]}); an explicitly selected backend "
                    "never falls back — use backend='auto' or 'deepgemm'"
                ) from e
            return None
    # Outside the build handler: native errors are never build failures or
    # fallback signals. False specifically means no kernel was submitted.
    if not module.try_run(input, weight, weight_scale, _QUANT_EPS, out):
        reason = "stream_cluster_capacity"
        _fallback(f"{requested}: {reason}")
        if requested == "specialized":
            raise NotImplementedError(
                "fp8_blockscale_gemm_sm90 backend='specialized' cannot serve "
                f"this call ({reason}); use backend='auto' or 'deepgemm'"
            )
        return None
    _stats["backend_dispatches"]["specialized"] += 1
    return out


__all__ = [
    "FP8BS_GEMM_BACKENDS",
    "SPECIALIZED_KERNEL_DISABLE_ENV",
]
