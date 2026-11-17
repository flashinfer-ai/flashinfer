# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Specialized SM90 decode-M dense fp8 block-scale GEMM backend.

An agent-generated kernel for the decode-M classes of
:func:`flashinfer.gemm.fp8_blockscale_gemm_sm90` in its BF16-activation /
FP8-weight / 128x128-block-scale form: the activation 1x128 per-token-group
quantization happens inside the kernel (amax/448 convention, matching the
stock route's internal quantization), weights and their
(ceil(N/128), K/128) float32 scale grid are consumed as passed.

Backend surface (#5667 semantics), driven from
``fp8_blockscale_gemm_sm90(..., backend=...)``:

- ``"auto"`` (the public default) routes here only on the packaged
  allowlist (``flashinfer/fp8bs_gemm_decodem_sm90_dispatch.json``);
  otherwise it is exactly the stock route.
- ``"specialized"`` selects this kernel explicitly; a call it cannot serve
  falls back to the stock route (this op's composable path) with the
  reason recorded — never to another fused backend.
- ``"stock"`` pins the stock route; an explicitly selected backend is
  never silently swapped.
- ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read per call; shared by
  FlashInfer's specialized kernels) keeps every call on the stock route.

Introspection: :func:`_fp8bs_decodem_stats` reports per-backend dispatch
counts and a ``fallback_reasons`` census. Guard order is GPU-free checks →
layout guards → device guards; a JIT build failure is logged once and the
call served by the stock route.
"""

import functools
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

SPECIALIZED_KERNEL_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

FP8BS_GEMM_BACKENDS = ("auto", "specialized", "stock")

_DISPATCH_FILE = "fp8bs_gemm_decodem_sm90_dispatch.json"

# The measured decode-M envelope (captured serve rows span M in [1, 24];
# interior points admitted by band interpolation, the study's banded form).
_M_MAX = 24
# The kernel's internal 1x128 activation quantization epsilon (the
# campaign definition's value, carried by the packaged allowlist too).
_QUANT_EPS = 1e-4

_stats: Dict[str, Any] = {
    "backend_dispatches": {"specialized": 0, "stock": 0},
    "fallback_reasons": {},
    "jit_build": "not attempted",
}


def _fp8bs_decodem_stats() -> Dict[str, Any]:
    out = dict(_stats)
    out["backend_dispatches"] = dict(_stats["backend_dispatches"])
    out["fallback_reasons"] = dict(_stats["fallback_reasons"])
    out["backends"] = FP8BS_GEMM_BACKENDS
    return out


def _reset_fp8bs_decodem_stats() -> None:
    _stats["backend_dispatches"] = {"specialized": 0, "stock": 0}
    _stats["fallback_reasons"] = {}


def _fallback(reason: str) -> None:
    fr = _stats["fallback_reasons"]
    fr[reason] = fr.get(reason, 0) + 1


def _disabled() -> bool:
    return os.environ.get(SPECIALIZED_KERNEL_DISABLE_ENV, "0") not in ("", "0")


@functools.cache
def _dispatch_table() -> Optional[Dict[str, Any]]:
    """The packaged allowlist that gates ``backend="auto"`` routing (the
    compiled-in guards in :func:`_eligibility` stay authoritative)."""
    try:
        import importlib.resources
        import json

        data = (importlib.resources.files("flashinfer") / _DISPATCH_FILE).read_text()
        table = json.loads(data)
        if not isinstance(table.get("backends"), list):
            return None
        if "fp8_blockscale_gemm_sm90" not in table.get("ops", {}):
            return None
        return table
    except Exception:
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
    # device guards
    if not input.is_cuda:
        return False, "device"
    major, minor = torch.cuda.get_device_capability(input.device)
    if major != 9:
        return False, "device"
    return True, "ok"


@functools.cache
def _get_module() -> Any:
    from ..jit.gemm import gen_fp8bs_gemm_decodem_sm90_module

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "fp8bs_gemm_decodem_sm90: the specialized kernel is not built "
            "yet and JIT compilation is not allowed under CUDA graph "
            "capture. Call fp8_blockscale_gemm_sm90 once eagerly on an "
            "allowlisted shape first."
        )
    module = gen_fp8bs_gemm_decodem_sm90_module().build_and_load()
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
    reason recorded for auto/specialized) so the caller continues on the
    stock route."""
    eligible, reason = _eligibility(input, weight, input_scale, weight_scale, out)
    if eligible and requested == "auto" and _dispatch_table() is None:
        eligible, reason = False, "no_dispatch_table"
    if not eligible:
        _fallback(f"{requested}: {reason}")
        return None
    try:
        module = _get_module()
    except RuntimeError:
        raise  # the capture-without-warmup error above must surface
    except Exception as e:
        if _stats["jit_build"] == "not attempted":
            logger.warning(
                "fp8bs_gemm_decodem_sm90: specialized kernel JIT build "
                "failed; serving via the stock route",
                exc_info=True,
            )
        _stats["jit_build"] = f"failed ({str(e)[:120]})"
        _fallback(f"{requested}: jit_build_failure")
        return None
    module.run(input, weight, weight_scale, _QUANT_EPS, out)
    _stats["backend_dispatches"]["specialized"] += 1
    return out


__all__ = [
    "FP8BS_GEMM_BACKENDS",
    "SPECIALIZED_KERNEL_DISABLE_ENV",
]
