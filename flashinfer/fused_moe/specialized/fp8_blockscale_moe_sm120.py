"""Specialized fused MoE for fp8 e4m3 weights with DeepSeek-style [128, 128]
fp32 block scales on SM120 (RTX PRO 6000), bf16 activations in/out,
silu-gated.

This is an agent-generated kernel set (CUDA, tvm-ffi DPS entry), produced
and benchmarked against
the Qwen3.6-35B-A3B-FP8 serving contract (``num_experts=256, top_k=8,
hidden=2048, intermediate=512``) with captured serving routing. FlashInfer's
cutlass path refuses this contract on SM120 ("FP8 block scaling not yet
implemented for Blackwell"); this backend is the first FlashInfer-resident
path for it.

This module is the kernels' OWN dispatch surface, and its sole public home
is the unified MoE API: ``SpecializedFp8BlockConfig`` (``api.py``, a default
backend candidate on SM120) and ``SpecializedFp8BlockRunner``
(``runners.py``) call :func:`maybe_run_specialized` directly. The kernels
are not CUTLASS kernels and are not reachable through ``cutlass_fused_moe``
or any other backend's entry point; ``MoELayer``'s ``backend=auto``
cross-backend shootout times them as their own named candidate
(``"specialized_fp8_block"``) and routes to them where they win. The
dispatch surface (token allowlist, device profile) lives in
``fp8_blockscale_moe_sm120_workload.json`` next to this module.
``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read at call time) disables the
backend entirely. Guard order: GPU-free checks (kill switch, dtype,
geometry, contiguity, device placement, pointer alignment, token
allowlist) -> device guards (compute capability, SM count, CUDA-graph warm
state). Every refusal is counted in ``fallback_reasons``.
"""

import functools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...jit import env as jit_env
from ...jit.core import JitSpec, gen_jit_spec
from ...utils import get_compute_capability

logger = logging.getLogger(__name__)


# Kill switch shared by FlashInfer's specialized kernels: read at CALL time.
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

_WORKLOAD_FILE = "fp8_blockscale_moe_sm120_workload.json"

_E, _TOPK, _H, _I = 256, 8, 2048, 512

# (device index, num_tokens) pairs already run eagerly in this process:
# only these variants may be dispatched under CUDA-graph capture.
_warmed: set = set()
_allowlist_cache: Optional[dict] = None

_stats: Dict[str, Any] = {
    "calls": 0,
    "specialized_dispatches": 0,
    "fallback_dispatches": 0,
    "fallback_reasons": {},
}


def _bump_fallback(reason: str) -> str:
    _stats["fallback_dispatches"] += 1
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    return reason


def _load_allowlist() -> Optional[dict]:
    """Load the dispatch surface; ``None`` => never dispatch."""
    global _allowlist_cache
    if _allowlist_cache is not None:
        return _allowlist_cache
    try:
        payload = json.loads((Path(__file__).parent / _WORKLOAD_FILE).read_text())
        _allowlist_cache = {
            "min_cc": tuple(int(v) for v in payload["min_compute_capability"]),
            "max_cc": tuple(int(v) for v in payload["max_compute_capability"]),
            # The kernels hard-code cooperative grids and barrier arrival
            # counts for this exact SM count; other SM120 parts (e.g. RTX
            # 5080/5090) must be refused, not served.
            "sm_count": int(payload["multi_processor_count"]),
            "tokens": frozenset(int(v) for v in payload["num_tokens_allowlist"]),
        }
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            "flashinfer.fused_moe specialized fp8-block-scale: workload "
            "allowlist unavailable, backend disabled: %s",
            exc,
        )
        _allowlist_cache = None
    return _allowlist_cache


def specialized_moe_fp8_blockscale_stats() -> dict:
    """Diagnostics: dispatch counters and the dispatch surface."""
    return {
        **_stats,
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "allowlist": _load_allowlist(),
        "warmed_variants": sorted(_warmed),  # (device index, num_tokens)
        "kill_switch_env": _DISABLE_ENV,
    }


def gen_specialized_fused_moe_fp8_blockscale_sm120_module() -> JitSpec:
    src_dir = jit_env.FLASHINFER_CSRC_DIR / "specialized_fused_moe_fp8_blockscale_sm120"
    sources = sorted(Path(src_dir).glob("*.cu"))
    return gen_jit_spec(
        "specialized_fused_moe_fp8_blockscale_sm120",
        sources,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            # single-token form REQUIRED: cpp_ext only treats "-gencode=..."
            # as a module arch override; the sm_120f family target caps
            # static smem at 48KB (sm_120a permits more), so the module
            # builds for sm_120a exactly.
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def _get_cuda_module():
    return gen_specialized_fused_moe_fp8_blockscale_sm120_module().build_and_load()


def _kernel_entry():
    """The DPS kernel entry for the active implementation (JIT/module load
    happens here — never under CUDA-graph capture, which the guard's
    warmed-variant rule already prevents)."""
    return _get_cuda_module().run


def _guard(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    quant_scales: List[torch.Tensor],
    context_ok: bool,
) -> Optional[str]:
    """Return None when the specialized backend can serve the call, else the
    fallback reason (recorded)."""
    # --- GPU-free checks first ---
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        return "kill_switch"
    if not context_ok:
        return "unsupported_call_context"
    allow = _load_allowlist()
    if allow is None:
        return "allowlist_unavailable"
    if input.ndim != 2 or input.dtype != torch.bfloat16:
        return "input_dtype_or_rank"
    # Every operand the kernels dereference must live on the input's CUDA
    # device; the TensorView entry would otherwise pass a host (or
    # wrong-device) pointer straight to the GPU.
    if not input.is_cuda:
        return "operand_device"
    for t in (
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        *quant_scales,
    ):
        if t is not None and t.device != input.device:
            return "operand_device"
    # Tensor-map (TMA) descriptors and vectorized loads require 16-byte
    # base alignment; a contiguous view at an odd storage offset passes the
    # shape checks but must be refused.
    for t in (
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        *quant_scales,
    ):
        if t is not None and t.data_ptr() % 16 != 0:
            return "operand_alignment"
    T = input.shape[0]
    if input.shape[1] != _H or not input.is_contiguous():
        return "input_geometry"
    if T not in allow["tokens"]:
        return "num_tokens_off_allowlist"
    if (
        token_selected_experts.shape != (T, _TOPK)
        or token_selected_experts.dtype != torch.int32
        or not token_selected_experts.is_contiguous()
    ):
        return "topk_ids_form"
    if (
        token_final_scales is None
        or token_final_scales.shape != (T, _TOPK)
        or token_final_scales.dtype != torch.float32
        or not token_final_scales.is_contiguous()
    ):
        return "topk_scales_form"
    if (
        fc1_expert_weights.shape != (_E, 2 * _I, _H)
        or fc1_expert_weights.dtype != torch.float8_e4m3fn
        or not fc1_expert_weights.is_contiguous()
    ):
        return "fc1_weights_form"
    if (
        fc2_expert_weights.shape != (_E, _H, _I)
        or fc2_expert_weights.dtype != torch.float8_e4m3fn
        or not fc2_expert_weights.is_contiguous()
    ):
        return "fc2_weights_form"
    if len(quant_scales) != 2:
        return "quant_scales_count"
    w13_s, w2_s = quant_scales
    if (
        w13_s.shape != (_E, (2 * _I) // 128, _H // 128)
        or w13_s.dtype != torch.float32
        or not w13_s.is_contiguous()
    ):
        return "fc1_scales_form"
    if (
        w2_s.shape != (_E, _H // 128, _I // 128)
        or w2_s.dtype != torch.float32
        or not w2_s.is_contiguous()
    ):
        return "fc2_scales_form"
    # --- device guards ---
    cc = get_compute_capability(input.device)
    if not (allow["min_cc"] <= tuple(cc) <= allow["max_cc"]):
        return "compute_capability"
    # The kernels' cooperative grids and barrier arrival counts are sized
    # for the validated device profile; a CC match alone (e.g. RTX
    # 5080/5090, also SM120) is not sufficient.
    props = torch.cuda.get_device_properties(input.device)
    if props.multi_processor_count != allow["sm_count"]:
        return "device_sm_count"
    # CUDA-graph rule: never dispatch a variant under capture unless it has
    # been run eagerly on this device (first call JIT-compiles on the
    # host). Query capture state on the input device's current stream.
    with torch.cuda.device(input.device):
        if (
            torch.cuda.is_current_stream_capturing()
            and (input.device.index, T) not in _warmed
        ):
            return "capture_unwarmed_variant"
    return None


def maybe_run_specialized(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    quant_scales: List[torch.Tensor],
    output: Optional[torch.Tensor],
    context_ok: bool,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Dispatch to the specialized backend when the guard passes.

    Returns ``(output, None)`` when served, else ``(None, reason)`` with the
    reason recorded in ``fallback_reasons``.
    """
    _stats["calls"] += 1
    reason = _guard(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        quant_scales,
        context_ok,
    )
    if reason is not None:
        return None, _bump_fallback(reason)
    dispatch_run = _kernel_entry()

    T = input.shape[0]
    if output is None:
        output = torch.empty((T, _H), dtype=torch.bfloat16, device=input.device)
    elif (
        output.shape != (T, _H)
        or output.dtype != torch.bfloat16
        or output.device != input.device
        or not output.is_contiguous()
        or output.data_ptr() % 16 != 0
    ):
        # The early specialized return bypasses core.py's output check, so a
        # bad caller buffer must be refused here (fail closed, never write
        # through an undersized or misplaced pointer).
        return None, _bump_fallback("output_form")
    output.zero_()  # kernels may accumulate partials into the output
    with torch.cuda.device(input.device):
        dispatch_run(
            input,
            fc1_expert_weights,
            quant_scales[0],
            fc2_expert_weights,
            quant_scales[1],
            token_selected_experts,
            token_final_scales,
            output,
        )
    _warmed.add((input.device.index, T))
    _stats["specialized_dispatches"] += 1
    return output, None


def warmup(
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    quant_scales: List[torch.Tensor],
    device=None,
    t_values=None,
) -> None:
    """Pre-compile the allowlisted token-count variants (CUDA-graph rule)."""
    allow = _load_allowlist()
    if allow is None:
        return
    device = torch.device(device or fc1_expert_weights.device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    dispatch_run = _kernel_entry()

    with torch.cuda.device(device_index):
        for T in sorted(t_values or allow["tokens"]):
            x = torch.zeros((T, _H), dtype=torch.bfloat16, device=device)
            ids = torch.zeros((T, _TOPK), dtype=torch.int32, device=device)
            tw = torch.zeros((T, _TOPK), dtype=torch.float32, device=device)
            out = torch.zeros((T, _H), dtype=torch.bfloat16, device=device)
            dispatch_run(
                x,
                fc1_expert_weights,
                quant_scales[0],
                fc2_expert_weights,
                quant_scales[1],
                ids,
                tw,
                out,
            )
            _warmed.add((device_index, T))
    torch.cuda.synchronize(device)
