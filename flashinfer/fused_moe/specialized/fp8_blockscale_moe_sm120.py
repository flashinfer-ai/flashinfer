# SPDX-License-Identifier: Apache-2.0

"""Specialized fused MoE for fp8 e4m3 weights with DeepSeek-style [128, 128]
fp32 block scales on SM120 (RTX PRO 6000), bf16 activations in/out,
silu-gated.

This is an agent-generated kernel set (CUDA, tvm-ffi DPS entry), produced
and benchmarked against
the Qwen3.6-35B-A3B-FP8 serving contract (``num_experts=256, top_k=8,
hidden=2048, intermediate=512``) with captured serving routing. FlashInfer's
cutlass path refuses this contract on SM120 ("FP8 block scaling not yet
implemented for Blackwell"); this backend serves it as a DEFAULT candidate
(the opt-in CuTe-DSL route, ``SM12xFp8Config``, also serves DeepSeek FP8 on
SM120).

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
backend entirely. The switch gates fresh Python dispatch only: a CUDA graph
captured while it was unset still launches the captured specialized kernels
on every replay (replay never re-runs the Python guard), so deterministic
disable requires retiring those graphs and recapturing with the switch set.
Guard order: GPU-free checks (kill switch, dtype,
geometry, contiguity, device placement, pointer alignment, token
allowlist) -> device guards (compute capability, SM count, CUDA-graph warm
state). Every refusal is counted in ``fallback_reasons``.

Scratch ownership (same model as the SM90 small-T grouped path): the
kernels take an explicit workspace tensor sized by the module's
``plan_workspace_bytes``; nothing mutable is cached per device inside the
CUDA module. Eager calls use one workspace per (device index, CUDA stream)
— the native kernels launch on the caller's current stream, so two streams
sharing one buffer would overwrite each other's intermediates. During
capture each call allocates scratch on the current stream and submits all
uses on that stream. PyTorch's graph pool preserves replay addresses and
manages ordered reuse; no Python workspace owner survives the call. If the
caller explicitly shares a pool between graphs, the usual PyTorch replay
ordering and non-concurrency requirements apply.
"""

import functools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import tvm_ffi

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
    "workspace_allocs": 0,  # eager per-(device, stream) (re)allocations
    "graph_workspace_allocs": 0,  # captured-call allocations
}

# Eager only: captured calls never look up or retain tensors in this cache.
_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}

# The workspace's first bytes are the kernels' barrier/epoch counter region
# (kWorkspaceDataOffset in the CUDA sources — keep in sync). It is zeroed
# ONCE when a buffer is allocated: every counter is epoch-based and each
# fixed slot is used by exactly one launch variant, so stream-ordered reuse
# needs no per-launch reset. For a capture-owned buffer the zeroing is
# recorded into the graph, so every replay restarts its epochs.
_COUNTER_PREFIX_BYTES = 2048


@functools.lru_cache(maxsize=None)
def _ffi_device(index: int):
    """tvm-ffi device handle, cached: building it parses a device string."""
    return tvm_ffi.device(f"cuda:{index}")


@functools.lru_cache(maxsize=None)
def _plan_bytes(T: int) -> int:
    """Workspace bytes for a token count (cached: the module's answer is a
    pure function of T, and the FFI hop is measurable at decode sizes)."""
    return int(_get_cuda_module().plan_workspace_bytes(T))


def _new_workspace(need: int, device: torch.device) -> torch.Tensor:
    ws = torch.empty(need, dtype=torch.uint8, device=device)
    ws[:_COUNTER_PREFIX_BYTES].zero_()
    return ws


def _get_workspace(
    device: torch.device,
    dev_index: int,
    stream_handle: int,
    need: int,
    capturing: bool,
) -> torch.Tensor:
    if capturing:
        ws = _new_workspace(need, device)
        _stats["graph_workspace_allocs"] += 1
        return ws
    ws_key = (dev_index, stream_handle)
    ws = _workspaces.get(ws_key)
    if ws is None or ws.numel() < need or ws.device != device:
        ws = _new_workspace(need, device)
        _workspaces[ws_key] = ws
        _stats["workspace_allocs"] += 1
    return ws


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
    allow = _load_allowlist()
    return {
        **_stats,
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        # A shallow copy: the cached dict is what dispatch reads, so a
        # diagnostics caller mutating the report must not change the
        # dispatch surface (the values themselves are immutable).
        "allowlist": dict(allow) if allow is not None else None,
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


def _weight_guard(device, w13, w2, quant_scales) -> Optional[str]:
    if len(quant_scales) != 2:
        return "quant_scales_count"
    for t, shape, dtype, reason in (
        (w13, (_E, 2 * _I, _H), torch.float8_e4m3fn, "fc1_weights_form"),
        (w2, (_E, _H, _I), torch.float8_e4m3fn, "fc2_weights_form"),
        (
            quant_scales[0],
            (_E, 2 * _I // 128, _H // 128),
            torch.float32,
            "fc1_scales_form",
        ),
        (quant_scales[1], (_E, _H // 128, _I // 128), torch.float32, "fc2_scales_form"),
    ):
        if t.device != device:
            return "operand_device"
        if t.data_ptr() % 16:
            return "operand_alignment"
        if t.shape != shape or t.dtype != dtype or not t.is_contiguous():
            return reason
    return None


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
    for t in (token_selected_experts, token_final_scales):
        if t is not None and t.device != input.device:
            return "operand_device"
    # Tensor-map (TMA) descriptors and vectorized loads require 16-byte
    # base alignment; a contiguous view at an odd storage offset passes the
    # shape checks but must be refused.
    for t in (input, token_selected_experts, token_final_scales):
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
    reason = _weight_guard(
        input.device, fc1_expert_weights, fc2_expert_weights, quant_scales
    )
    if reason is not None:
        return reason
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
    with torch.cuda.device(input.device):
        dispatch_run = _kernel_entry()
        dev_index = (
            input.device.index
            if input.device.index is not None
            else torch.cuda.current_device()
        )
        stream = torch.cuda.current_stream()
        need = _plan_bytes(T)
        capturing = torch.cuda.is_current_stream_capturing()
        ws = _get_workspace(
            input.device, dev_index, stream.cuda_stream, need, capturing
        )
        output.zero_()  # kernels may accumulate partials into the output
        # StreamContext syncs the FFI environment stream to the caller's
        # current stream for this call (what tvm_ffi.use_torch_stream does,
        # minus its per-call device-string parse): without it the kernels
        # launch on whatever stream the FFI environment last held (the
        # default stream on first use), so a caller dispatching from a side
        # stream or capturing a graph would have its kernels ESCAPE its
        # stream — unordered against the output zeroing and against the
        # per-stream workspace ownership established above.
        with tvm_ffi.StreamContext(_ffi_device(dev_index), stream.cuda_stream):
            dispatch_run(
                input,
                fc1_expert_weights,
                quant_scales[0],
                fc2_expert_weights,
                quant_scales[1],
                token_selected_experts,
                token_final_scales,
                output,
                ws,
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
    # The kill switch disables the backend ENTIRELY: warmup must not JIT-load
    # the module or launch anything while it is set (and nothing is recorded
    # as warmed, so a later capture without the switch still requires a real
    # eager warmup).
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        logger.debug(
            "flashinfer.fused_moe specialized fp8-block-scale: warmup skipped, %s=1",
            _DISABLE_ENV,
        )
        return
    allow = _load_allowlist()
    if allow is None:
        return
    device = torch.device(fc1_expert_weights.device if device is None else device)
    if device.type != "cuda":
        raise ValueError("warmup requires a CUDA device")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    # Validate everything BEFORE JIT-loading the module or launching: warmup
    # takes the same dispatch surface as _guard for what it does not fabricate
    # itself (token list, device profile, capture state, weight operands).
    ts = sorted(t_values or allow["tokens"])
    off = [t for t in ts if t not in allow["tokens"]]
    if off:
        raise ValueError(
            f"warmup t_values {off} are off the shipped num_tokens allowlist; "
            "ordinary dispatch would refuse these variants."
        )
    warm_device = torch.device("cuda", device_index)
    cc = tuple(get_compute_capability(warm_device))
    if not (allow["min_cc"] <= cc <= allow["max_cc"]):
        raise RuntimeError(
            f"warmup device compute capability {cc} is outside the validated "
            f"range {allow['min_cc']}..{allow['max_cc']}."
        )
    props = torch.cuda.get_device_properties(device_index)
    if props.multi_processor_count != allow["sm_count"]:
        raise RuntimeError(
            f"warmup device has {props.multi_processor_count} SMs; the "
            f"kernels are validated only for the {allow['sm_count']}-SM "
            "profile."
        )
    reason = _weight_guard(
        warm_device, fc1_expert_weights, fc2_expert_weights, quant_scales
    )
    if reason is not None:
        raise ValueError(f"invalid warmup operands: {reason}")
    with torch.cuda.device(device_index):
        # Warmup ends with a device synchronization and exists to make
        # capture legal later; it must itself run eagerly.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warmup must not run under CUDA-graph capture.")
    pending = set()
    with torch.cuda.device(device_index), tvm_ffi.use_torch_stream():
        dispatch_run = _kernel_entry()
        for T in ts:
            x = torch.zeros((T, _H), dtype=torch.bfloat16, device=warm_device)
            ids = torch.zeros((T, _TOPK), dtype=torch.int32, device=warm_device)
            tw = torch.zeros((T, _TOPK), dtype=torch.float32, device=warm_device)
            out = torch.zeros((T, _H), dtype=torch.bfloat16, device=warm_device)
            # Warmup runs eagerly (checked above), so this primes the
            # (device, current stream) workspace at the largest warmed size.
            ws = _get_workspace(
                warm_device,
                device_index,
                torch.cuda.current_stream().cuda_stream,
                _plan_bytes(T),
                False,
            )
            dispatch_run(
                x,
                fc1_expert_weights,
                quant_scales[0],
                fc2_expert_weights,
                quant_scales[1],
                ids,
                tw,
                out,
                ws,
            )
            pending.add((device_index, T))
    # Publish warmup receipts only after the device work completes: if the
    # synchronization raises, nothing is recorded, so the capture guard never
    # treats a failed warmup as proof of preparation. Receipts from earlier
    # successful warmups are preserved.
    torch.cuda.synchronize(device_index)
    _warmed.update(pending)
