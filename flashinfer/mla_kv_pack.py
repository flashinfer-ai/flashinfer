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
from .trace.templates.attention import concat_mla_kv_quant_fp8_trace
from .utils import get_compute_capability

logger = logging.getLogger(__name__)

# Kill switch shared by FlashInfer's specialized kernels: read at CALL time.
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Fixed head geometry of the fused kernel (DeepSeek/Kimi MLA).
NOPE_DIM = 128
ROPE_DIM = 64
V_DIM = 128
_KV_DIM = NOPE_DIM + V_DIM
# Largest finite float8_e4m3fn magnitude; the fused kernel saturates to it.
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_QK_DIM = NOPE_DIM + ROPE_DIM

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

# Dispatch surface of the fused kernel (package data; see the file's note).
_WORKLOAD_FILE = "mla_kv_pack_fp8_workloads.json"


@functools.cache
def _load_allowlist() -> Optional[dict]:
    """Load the fused kernel's dispatch surface; ``None`` => never dispatch."""
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


_modules: Dict[Tuple[str, int], Any] = {}
_module_errors: Dict[Tuple[str, int], str] = {}
_module_lock = threading.Lock()
_stats: Dict[str, Any] = {
    "calls": 0,
    "specialized_dispatches": 0,
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
    """Build/load the (target, head group) JIT module once; never inside a CUDA-graph capture."""
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


def _concat_mla_kv_quant_fp8_stats() -> dict:
    """Diagnostics: dispatch counters, JIT state and the compile footprint.

    The fused kernel is one generated Cake program per head group (1, 2, 3 or 4
    head pairs per warp, planned from ``(num_tokens, num_heads)`` on the host);
    token and head counts are runtime launch arguments, so nothing is compiled
    per shape.  ``compiled_variants`` counts the (target, head group) modules
    loaded so far; ``distinct_kernels_for_allowlist`` is the delivered program
    count of the route table.
    """
    allowlist = _load_allowlist()
    return {
        **_stats,
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "module_errors": dict(_module_errors),
        "allowlist_loaded": allowlist is not None,
        "allowlist": allowlist,
        "compiled_variants": len(_modules),
        "distinct_kernels_for_allowlist": len(ROUTES),
        "distinct_kernels": "one generated Cake program per head group "
        + ", ".join(sorted(ROUTES))
        + " (head group planned from num_tokens and num_heads; T and H are runtime arguments)",
        "precompiled": bool(_modules),
    }


def _fallback(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    nope_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Composable stock path: two casts + two strided copies (any geometry).

    Finite values are clamped to the e4m3fn range before ``Tensor.to``: torch
    < 2.13 encodes finite overflow as NaN, while the fused kernel saturates to
    +/-448. NaN passes through the clamp unchanged (both encode it as 0x7F), so
    the fallback is byte-identical to the kernel on every supported torch.
    """
    num_tokens = kv_nope.shape[0]
    kv_fp8 = kv_nope.clamp(-_FP8_MAX, _FP8_MAX).to(key.dtype)
    key[..., :nope_dim].copy_(kv_fp8[..., :nope_dim])
    key[..., nope_dim:].copy_(
        k_pe.clamp(-_FP8_MAX, _FP8_MAX)
        .reshape(num_tokens, 1, k_pe.shape[-1])
        .to(key.dtype)
        .expand(num_tokens, kv_nope.shape[1], k_pe.shape[-1])
    )
    value.copy_(kv_fp8[..., nope_dim:])
    return key, value


def _specialized_supported(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    nope_dim: int,
) -> Optional[str]:
    """Return None when the fused kernel serves this call, else the reason."""
    if os.environ.get(_DISABLE_ENV) == "1":
        return "kill_switch"
    # Semantics / geometry first (GPU-free), device last.
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
    if not (
        kv_nope.is_contiguous()
        and k_pe.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
    ):
        return "non_contiguous"
    if any(t.data_ptr() % 16 for t in (kv_nope, k_pe, key, value)):
        return "alignment"
    allowlist = _load_allowlist()
    if allowlist is None:
        return "allowlist_unavailable"
    num_tokens, num_heads = kv_nope.shape[0], kv_nope.shape[1]
    if not allowlist["heads"][0] <= num_heads <= allowlist["heads"][1]:
        return "num_heads_not_allowlisted"
    if not allowlist["tokens"][0] <= num_tokens <= allowlist["tokens"][1]:
        return "num_tokens_not_allowlisted"
    if route_key(_plan_head_group(num_tokens, num_heads)) not in ROUTES:
        return "head_group_route"
    if not kv_nope.is_cuda:
        return "device"
    capability = get_compute_capability(kv_nope.device)
    if capability < allowlist["min_cc"]:
        return "compute_capability"
    target = concat_mla_kv_quant_fp8_target_for_capability(capability)
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


@flashinfer_api(trace=concat_mla_kv_quant_fp8_trace)
def concat_mla_kv_quant_fp8(
    kv_nope: torch.Tensor,
    k_pe: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    *,
    nope_dim: int = NOPE_DIM,
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
    torch 2.13. Output is bit-exact with that cast plus slice copies.

    The fused kernel is a generated Cake program (one per head group, i.e. the
    number of head pairs a warp owns, planned from the token and head counts)
    built for the exact compute capabilities 10.0 and 10.3. It serves bf16
    inputs with head geometry ``nope_dim=128, rope_dim=64, v_dim=128`` on
    contiguous, 16-byte-aligned tensors within the head-count and token-count
    surface of the package's ``mla_kv_pack_fp8_workloads.json`` (1..128 heads,
    1..131072 tokens); every other call takes the composable torch path.
    ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` forces the composable path
    (read at call time).

    Parameters
    ----------
    kv_nope : torch.Tensor
        ``[num_tokens, num_heads, nope_dim + v_dim]`` bf16 (contiguous for the
        fused path).
    k_pe : torch.Tensor
        ``[num_tokens, rope_dim]`` or ``[num_tokens, 1, rope_dim]`` bf16,
        shared across heads.
    key : Optional[torch.Tensor]
        ``[num_tokens, num_heads, nope_dim + rope_dim]`` float8_e4m3fn output;
        allocated when ``None``.
    value : Optional[torch.Tensor]
        ``[num_tokens, num_heads, v_dim]`` float8_e4m3fn output; allocated
        when ``None``.
    nope_dim : int
        Split point of ``kv_nope``'s last dim (default 128).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(key, value)``.
    """
    global _marker_logged
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
    reason = _specialized_supported(kv_nope, k_pe, key, value, nope_dim)
    module = record = None
    head_group = 0
    if reason is None and num_tokens > 0:
        head_group = _plan_head_group(int(num_tokens), int(num_heads))
        route = concat_mla_kv_quant_fp8_route_record(kv_nope.device, head_group)
        record = route["module"]
        module = _get_module(route["target"], head_group)
        if module is None:
            reason = "jit_unavailable"
    if reason is not None:
        _bump_fallback(reason)
        return _fallback(kv_nope, k_pe, key, value, nope_dim)
    if num_tokens == 0:
        return key, value
    grid, warps_per_token = _plan_grid(int(num_tokens), int(num_heads), head_group)
    cake_concat_mla_kv_quant_fp8.launch(
        kv_nope,
        k_pe,
        key,
        value,
        head_group=head_group,
        warps_per_token=warps_per_token,
        grid=grid,
        module=module,
        record=record,
    )
    _stats["specialized_dispatches"] += 1
    if not _marker_logged:
        _marker_logged = True
        logger.info(
            "flashinfer.concat_mla_kv_quant_fp8: fused Cake kernel dispatched "
            "(num_heads=%d, num_tokens=%d, head_group=%d)",
            num_heads,
            num_tokens,
            head_group,
        )
    return key, value
