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

Specialized SM90 small-batch (decode-step) fused-MoE kernel for the
DeepSeek-style fp8 [128,128]-block-scale contract
(csrc/moe_fp8_grouped_smallt_sm90.cu,
flashinfer/jit/moe_fp8_grouped_smallt_sm90.py).

This module is the kernel's OWN dispatch surface. Its sole public home is
the unified MoE API: ``SpecializedFp8BlockSmallTConfig`` (``api.py``, a
default backend candidate on SM90) and ``SpecializedFp8BlockSmallTRunner``
(``runners.py``) call :func:`_moe_smallt_sm90_dispatch` directly. The kernel
is NOT a CUTLASS kernel and is not reachable through ``cutlass_fused_moe``
or any other backend's entry point: ``MoELayer``'s ``backend=auto``
per-token-bucket shootout times it as its own named candidate and routes to
it where it wins, and an explicitly selected other backend is never served
by it.

ALLOWLIST (the dispatch declines otherwise; the runner never serves a
declined call with a different kernel): SM90 device, DeepSeek-style fp8
[128,128]-block-scale weights (``quant_scales == [fc1_block_scales,
fc2_block_scales]``), bf16 unquantized activations, bf16 output, gated SiLU
(Swiglu) activation, no biases / input_sf / swiglu parameters / expert
parallelism / min-latency mode, the measured kernel geometry per launch
class (T == 1: hidden 512-aligned, intermediate 256, top_k 9; the grouped
band: hidden 256-aligned, intermediate 256, top_k 9, at most 288 experts),
and token counts inside the measured won bands **T <= 16 or
256 <= T <= 4096** — the measured envelope where this kernel beats the
serving engines' dispatched triton fused-MoE on the captured GLM-class
routing. At the measured T >= 64 classes the kernel LOSES to that incumbent
(0.55-0.81x), so batches in the 17..255 gap are declined by design, not by
omission (``MoELayer`` routes them to the stock dsfp8 backend when it is
configured, as it is by default). The kernel consumes the stock dsfp8 fc1
stacking as-is (rows [0:I] = up, rows [I:2I] = gate).

``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read per call; the kill switch
shared by FlashInfer's specialized kernels) makes the runner decline every
call, so ``MoELayer`` selects among the remaining backends;
``FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90=1`` is a deprecated alias
kept for one release.

The dispatch never allocates on the hot path except the output tensor when
the caller did not pass one: the intermediate workspace is cached per
(device, stream) at the serving band's capacity. A CUDA-graph capture
adopts a primed buffer and takes PER-CAPTURE ownership of it (keyed by the
CUDA capture-sequence id, so two independent graphs never share scratch;
``FLASHINFER_MOE_SMALLT_SHARE_GRAPH_WORKSPACE=1`` opts back into
per-capture-stream sharing for callers that serialize their replays), and
the JIT build happens at layer init or on the first eligible call outside
any CUDA-graph capture (a capture without a built module is declined).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..jit.core import logger
from ..utils import get_compute_capability

# Kill switch shared by FlashInfer's specialized kernels (read per call).
_SPECIALIZED_KERNEL_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
# Deprecated alias (kept for one release).
_DISABLE_ENV = "FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90"
# Explicit opt-in: share ONE graph workspace among every graph captured on
# the same capture stream (the pre-fix behavior). Only safe when the caller
# guarantees the graphs never replay concurrently; by default each capture
# owns its own scratch (see _graph_workspaces).
_SHARE_GRAPH_WS_ENV = "FLASHINFER_MOE_SMALLT_SHARE_GRAPH_WORKSPACE"
_MAX_TOKENS = 16  # small-path band cap (measured won rows T=1/4/16)
# Expansion (measured vs FI's own grouped-fp8 route at contract): the
# candidate also wins the T=256/1024/4096 classes against FI's stock cutlass
# dsfp8 path, so a second band [256, 4096] is admitted; interior points
# (e.g. T=512/2048) are admitted by BAND INTERPOLATION between the measured
# classes, the same semantics as the shipped 1..16 band. The 17..255 gap —
# containing the measured T=64 class, where FI's own kernel held at contract
# (and in-venv) — is declined (served by the stock dsfp8 backend).
_BAND2_LO = 256
_BAND2_HI = 4096
_COMPUTE_CAPABILITY = (9, 0)
# Per-block dynamic shared memory available WITHOUT the explicit opt-in the
# launcher performs only for the grouped band's constant-size kernels. The
# small path's launches are data-dependent (fc1 stages H halves (+ the fc1
# scale grid for T >= 2), fc2 stages top_k * I halves), so geometry whose
# requirement exceeds this must be declined before launch.
_SMEM_LIMIT = 48 * 1024
# The kernels read activations, weights and the fp16 intermediate through
# uint4 (16-byte) vectors; CUDA requires natural alignment, which a
# contiguous view with a storage offset does not guarantee.
_VEC_ALIGN = 16

_backend_dispatches: Dict[str, int] = {"specialized": 0, "declined": 0}
_fallback_reasons: Dict[str, int] = {}


def _fallback(reason: str) -> None:
    _fallback_reasons[reason] = _fallback_reasons.get(reason, 0) + 1


_counters: Dict[str, int] = {
    "calls_seen": 0,  # dsfp8 cutlass calls that reached the hook
    "calls_disabled": 0,  # disable switch set
    "calls_ineligible": 0,  # dsfp8 route, but geometry/arguments differ
    "build_failed": 0,  # JIT build/load failed (falls back, logged once)
    "workspace_allocs": 0,  # per-device workspace (re)allocations
    "runs_dispatched": 0,  # specialized kernel launched
    "runs_dispatched_capturing": 0,  # ... of which during CUDA-graph capture
    "runs_capture_unready": 0,  # capturing without a built module: fallback
}
_module: Any = None
_module_error: Optional[str] = None
# Intermediate workspaces keyed by (device index, CUDA stream handle): the
# native run launches on the caller's current stream, so two streams sharing
# one buffer would overwrite each other's intermediates. A CUDA-graph
# capture adopts the device's largest primed workspace (the torch.cuda.graph
# warmup pattern primes the cache on the warmup stream) and takes OWNERSHIP
# of it: the buffer moves to _graph_workspaces and never returns to eager
# use — a replay is ordered only against work on its own replay stream, so
# scratch shared with eager calls could be written by two executions
# concurrently.
#
# _graph_workspaces is keyed by (device index, CUDA capture-sequence id):
# every independent capture owns its own scratch. The capture stream is NOT
# an ownership key — torch.cuda.graph reuses one default capture stream for
# every capture in the process, so two independent graphs (e.g. two
# MoELayer instances captured back to back) would silently share scratch
# and corrupt each other when replayed concurrently, even with separate
# output allocations and no shared graph pool. Launches inside one graph
# serialize at replay, so reuse within a single capture stays safe (and
# keeps an N-layer model capture at one buffer). Callers that guarantee
# their graphs never replay concurrently can opt back into per-capture-
# stream sharing with _SHARE_GRAPH_WS_ENV=1 to reclaim the per-graph
# buffers.
_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}
_graph_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}


def _current_capture_id(device: torch.device) -> Optional[int]:
    """Unique id of the capture sequence the current stream is recording
    into, or ``None`` when it cannot be established. ``cudaStreamGetCaptureInfo``
    is explicitly legal during capture; the id is unique per capture within
    the process, which is exactly the ownership boundary graph scratch
    needs (the capture stream is reused across independent captures)."""
    try:
        from ..cuda_utils import runtime

        stream = torch.cuda.current_stream(device).cuda_stream
        res = runtime.cudaStreamGetCaptureInfo(stream)
        if int(res[0]) != 0:
            return None
        status, capture_id = res[1], res[2]
        if status != runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
            return None
        return int(capture_id)
    except Exception:
        return None


# Workspaces replaced by larger ones are retained for the life of the
# process: a CUDA graph captured earlier may still replay into the old
# tensor's address, so releasing it to the allocator would let unrelated
# allocations alias a buffer a replay still writes. Bounded by the few
# distinct plan sizes a process uses.
_retired_workspaces: List[torch.Tensor] = []
_announced = False


def _disabled() -> bool:
    if os.environ.get(_SPECIALIZED_KERNEL_DISABLE_ENV, "") not in ("", "0"):
        return True
    if os.environ.get(_DISABLE_ENV, "") not in ("", "0"):
        logger.warning_once(
            "%s is deprecated and will be removed in a future release; set "
            "%s=1 instead (the kill switch shared by FlashInfer's "
            "specialized kernels)",
            _DISABLE_ENV,
            _SPECIALIZED_KERNEL_DISABLE_ENV,
        )
        return True
    return False


def _moe_smallt_sm90_stats() -> Dict[str, Any]:
    """Introspection for benchmarks and tests: per-process call census and
    build state of the specialized SM90 small-T dsfp8 MoE kernel."""
    stats: Dict[str, Any] = dict(_counters)
    stats["module_loaded"] = _module is not None
    stats["module_error"] = _module_error
    stats["backend_dispatches"] = dict(_backend_dispatches)
    stats["fallback_reasons"] = dict(_fallback_reasons)
    return stats


def _reset_moe_smallt_sm90_stats() -> None:
    global _announced
    for key in _counters:
        _counters[key] = 0
    for key in _backend_dispatches:
        _backend_dispatches[key] = 0
    _fallback_reasons.clear()
    _announced = False


def _get_module() -> Any:
    global _module, _module_error
    if _module is not None:
        return _module
    if _module_error is not None:
        return None
    try:
        from ..jit.moe_fp8_grouped_smallt_sm90 import (
            gen_moe_fp8_grouped_smallt_sm90_module,
        )

        _module = gen_moe_fp8_grouped_smallt_sm90_module().build_and_load()
    except Exception as exc:  # build or load failure: decline, say why
        _module_error = f"{type(exc).__name__}: {exc}"[:800]
        _counters["build_failed"] += 1
        logger.warning_once(
            "moe_fp8_grouped_smallt_sm90: JIT build failed (%s); the "
            "specialized backend declines every call",
            _module_error,
        )
        return None
    return _module


def _smallt_smem_reason(T: int, H: int, I: int, KTOP: int) -> Optional[str]:
    """Dynamic shared-memory capacity of the small-path launches; ``None``
    when the geometry fits. The SINGLE predicate used by both ``_eligibility``
    (dispatch) and ``SpecializedFp8BlockSmallTRunner._band_geometry_ok``
    (acceptance), so the two checks cannot drift apart: a runner that accepts
    a call its dispatch declines would raise out of the auto shootout."""
    if T == 1:
        if 2 * H > _SMEM_LIMIT:
            return f"fc1 staging smem {2 * H}B > {_SMEM_LIMIT}B (H={H})"
    elif T <= _MAX_TOKENS:
        fc1_smem = 2 * H + 4 * ((2 * I + 127) // 128) * ((H + 127) // 128)
        if fc1_smem > _SMEM_LIMIT:
            return f"fc1 smem {fc1_smem}B > {_SMEM_LIMIT}B (H={H}, I={I})"
        if 2 * KTOP * I > _SMEM_LIMIT:
            return f"fc2 smem {2 * KTOP * I}B > {_SMEM_LIMIT}B (top_k={KTOP}, I={I})"
    return None


def _eligibility(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    output: Optional[torch.Tensor],
    flags: Dict[str, Any],
) -> Optional[str]:
    """Return None when the call is the measured small-T dsfp8 contract, else
    a reason string (logged once; the call is declined).
    Semantics first, then shapes/dtypes, device last; no synchronisation."""
    for name, val in flags.items():
        if val:
            return name
    if input.dtype != torch.bfloat16 or input.dim() != 2:
        return f"input {input.dtype} dim={input.dim()}"
    T = input.shape[0]
    if not (1 <= T <= _MAX_TOKENS or _BAND2_LO <= T <= _BAND2_HI):
        return (
            f"num_tokens={T} outside the measured won bands "
            f"[1,{_MAX_TOKENS}] u [{_BAND2_LO},{_BAND2_HI}]"
        )
    if output_dtype != torch.bfloat16:
        return f"output_dtype={output_dtype}"
    if not isinstance(quant_scales, (list, tuple)) or len(quant_scales) != 2:
        return "quant_scales is not the dsfp8 [fc1_block_scales, fc2_block_scales] pair"
    s1, s2 = quant_scales
    w1, w2 = fc1_expert_weights, fc2_expert_weights
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return f"weights {w1.dtype}/{w2.dtype}"
    if w1.dim() != 3 or w2.dim() != 3:
        return f"weights dim {w1.dim()}/{w2.dim()}"
    E, N1, H = w1.shape
    if w2.shape[0] != E or w2.shape[1] != H:
        return f"fc2 shape {tuple(w2.shape)} vs fc1 {tuple(w1.shape)}"
    I = w2.shape[2]
    if N1 != 2 * I:
        return f"fc1 rows {N1} != 2 * intermediate {I} (not gated)"
    if input.shape[1] != H:
        return f"hidden {input.shape[1]} != {H}"
    if H % 128 or I % 128:
        return f"H={H} / I={I} not 128-aligned (block-scale grid)"
    if not (
        isinstance(s1, torch.Tensor)
        and isinstance(s2, torch.Tensor)
        and s1.dtype == torch.float32
        and s2.dtype == torch.float32
        and tuple(s1.shape) == (E, (N1 + 127) // 128, (H + 127) // 128)
        and tuple(s2.shape) == (E, (H + 127) // 128, (I + 127) // 128)
    ):
        return "block scales are not the [128,128] f32 grids"
    ids, tw = token_selected_experts, token_final_scales
    if (
        ids.dtype not in (torch.int32, torch.int64)
        or ids.dim() != 2
        or ids.shape[0] != T
    ):
        return f"token_selected_experts {ids.dtype} {tuple(ids.shape)}"
    if tw is None or tw.dim() != 2 or tuple(tw.shape) != tuple(ids.shape):
        return "token_final_scales shape"
    # Per-launch-class geometry of the measured contract (mirrored by the
    # native validation in the .cu): T == 1 runs the 512-half swizzled fc1
    # staging and the fixed 256-column / 9-slot fc2 pair; the grouped band
    # runs the 288-thread prefix scan, 256-column fp16 fc2 tiles, the
    # 2x128-block fp8 fc2 and the 9-slot reductions. T in [2, 16] keeps the
    # generic small-path envelope (any top_k / experts, 128-aligned H / I).
    KTOP = ids.shape[1]
    if T == 1:
        if H % 512:
            return f"H={H} not 512-aligned (T=1 fc1 staging)"
        if I != 256:
            return f"I={I} != 256 (T=1 fc2 tile width)"
        if KTOP != 9:
            return f"top_k={KTOP} != 9 (T=1 fc2 routing slots)"
    if T >= _BAND2_LO:
        if KTOP != 9:
            return f"top_k={KTOP} != 9 (grouped combine slots)"
        if E > 288:
            return f"num_experts={E} > 288 (grouped prefix CTA)"
        if H % 256:
            return f"H={H} not 256-aligned (grouped fc2 tiles)"
        if I != 256:
            return f"I={I} != 256 (grouped fp8 fc2 K blocks)"
    # Dynamic shared-memory capacity of the small-path launches (mirrored by
    # the native validation; the grouped band's kernels use launch-class
    # constants and opt in where they exceed the default limit). Shared with
    # the unified runner's _band_geometry_ok so acceptance cannot admit a
    # geometry the dispatch declines.
    smem_reason = _smallt_smem_reason(T, H, I, KTOP)
    if smem_reason is not None:
        return smem_reason
    if output is not None and (
        output.dtype != torch.bfloat16 or tuple(output.shape) != (T, H)
    ):
        return f"output {output.dtype} {tuple(output.shape)}"
    for name, t in (
        ("input", input),
        ("fc1", w1),
        ("fc2", w2),
        ("s1", s1),
        ("s2", s2),
        ("ids", ids),
        ("tw", tw),
    ):
        if not t.is_contiguous():
            return f"{name} not contiguous"
        if t.device != input.device:
            return f"{name} device {t.device}"
    # Contiguity does not imply address alignment: a contiguous view with a
    # storage offset can be only element-aligned, and the kernels read these
    # tensors through 16-byte (uint4) vectors.
    for name, t in (("input", input), ("fc1", w1), ("fc2", w2)):
        if t.data_ptr() % _VEC_ALIGN:
            return f"{name} not {_VEC_ALIGN}-byte aligned (vectorized loads)"
    if output is not None and output.data_ptr() % _VEC_ALIGN:
        return f"output not {_VEC_ALIGN}-byte aligned (vectorized stores)"
    if input.device.type != "cuda":
        return f"device {input.device}"
    try:
        if tuple(get_compute_capability(input.device)) != _COMPUTE_CAPABILITY:
            return "not SM90"
    except Exception as exc:
        return f"compute capability query failed: {exc}"[:120]
    return None


def _moe_smallt_sm90_dispatch(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    output: Optional[torch.Tensor],
    flags: Dict[str, Any],
) -> Optional[torch.Tensor]:
    """Serve the call with the specialized small-T kernel, or return None
    (nothing launched; the call is DECLINED and the reason recorded — the
    caller, ``SpecializedFp8BlockSmallTRunner``, decides what a declined
    call means; it never serves one with a different kernel)."""
    global _announced
    _counters["calls_seen"] += 1
    if _disabled():
        _counters["calls_disabled"] += 1
        _fallback("kill switch")
        _backend_dispatches["declined"] += 1
        logger.info_once(
            "moe_fp8_grouped_smallt_sm90: disabled (%s, or its deprecated "
            "alias %s); every call is declined",
            _SPECIALIZED_KERNEL_DISABLE_ENV,
            _DISABLE_ENV,
        )
        return None
    try:
        reason = _eligibility(
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
            output_dtype,
            quant_scales,
            output,
            flags,
        )
    except Exception as exc:  # a guard failure must never break the caller
        reason = f"guard_error:{type(exc).__name__}: {exc}"[:300]
    if reason is not None:
        _counters["calls_ineligible"] += 1
        _fallback(reason)
        _backend_dispatches["declined"] += 1
        logger.info_once(
            "moe_fp8_grouped_smallt_sm90: not dispatched (%s); the call is declined",
            reason,
        )
        return None
    capturing = bool(
        hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    )
    if capturing and _module is None:
        # never compile inside a capture
        _counters["runs_capture_unready"] += 1
        _fallback("uncompiled under capture")
        _backend_dispatches["declined"] += 1
        return None
    module = _get_module()
    if module is None:
        _fallback("JIT build failed")
        _backend_dispatches["declined"] += 1
        return None

    T = input.shape[0]
    KTOP = token_selected_experts.shape[1]
    I = fc2_expert_weights.shape[2]
    dev_index = (
        input.device.index
        if input.device.index is not None
        else torch.cuda.current_device()
    )
    H_sz = input.shape[1]
    E_sz = fc1_expert_weights.shape[0]
    band_cap = _MAX_TOKENS if T <= _MAX_TOKENS else _BAND2_HI
    need = int(module.plan_workspace_bytes(band_cap, KTOP, I, H_sz, E_sz))
    ws_key = (dev_index, torch.cuda.current_stream(input.device).cuda_stream)
    if capturing:
        # Scratch ownership boundary: the CAPTURE, not the capture stream.
        # torch.cuda.graph reuses one default capture stream, so a stream
        # key would hand two independent graphs the same scratch — correct
        # in isolation, corrupted under concurrent replay. The capture-
        # sequence id is unique per capture; _SHARE_GRAPH_WS_ENV=1 restores
        # per-capture-stream sharing for callers that explicitly agree to
        # serialize their replays.
        if os.environ.get(_SHARE_GRAPH_WS_ENV, "") not in ("", "0"):
            graph_key = ws_key
        else:
            capture_id = _current_capture_id(input.device)
            if capture_id is None:
                _counters["runs_capture_unready"] += 1
                _fallback("capture id query failed under capture")
                _backend_dispatches["declined"] += 1
                return None
            graph_key = (dev_index, capture_id)
        ws = _graph_workspaces.get(graph_key)
        if ws is not None and (ws.numel() < need or ws.device != input.device):
            ws = None
        if ws is None:
            # Never allocate (or cache) inside a capture: adopt the device's
            # largest primed eager workspace — the torch.cuda.graph warmup
            # pattern primes it on the warmup stream — and take ownership:
            # the buffer leaves the eager cache so no eager call or capture
            # on another stream can race the graph's replays over it (see
            # the _graph_workspaces comment).
            #
            # ORDERING CONTRACT: adoption covers only work queued AFTER it;
            # work already queued on the donor stream must have completed
            # before capture begins. No ordering can be established from
            # inside an active capture (a stream/device synchronize here
            # would invalidate the capture, and a wait on an event recorded
            # outside the capture cannot be recorded into it), so the caller
            # owns it: torch.cuda.graph's capture entry synchronizes after
            # its warmup iterations, which quiesces the donor stream and is
            # the pattern that primes this cache. A capture started while an
            # eager small-T call is still in flight on another stream is
            # outside this contract.
            adopt_key = None
            for key, cand in _workspaces.items():
                if (
                    key[0] == dev_index
                    and cand.numel() >= need
                    and (ws is None or cand.numel() > ws.numel())
                ):
                    ws, adopt_key = cand, key
            if ws is None:
                _counters["runs_capture_unready"] += 1
                _fallback("workspace alloc under capture")
                _backend_dispatches["declined"] += 1
                return None
            del _workspaces[adopt_key]
            old = _graph_workspaces.get(graph_key)
            if old is not None:
                # an earlier adoption under this key (same capture, or the
                # shared-stream opt-in) may still be replayed into
                _retired_workspaces.append(old)
            _graph_workspaces[graph_key] = ws
    else:
        ws = _workspaces.get(ws_key)
        if ws is not None and (ws.numel() < need or ws.device != input.device):
            ws = None
        if ws is None:
            old = _workspaces.get(ws_key)
            ws = torch.empty(need, dtype=torch.uint8, device=input.device)
            if old is not None:
                # a captured graph may still replay into the old buffer
                _retired_workspaces.append(old)
            _workspaces[ws_key] = ws
            _counters["workspace_allocs"] += 1

    ids = token_selected_experts
    if ids.dtype != torch.int32:
        if capturing:
            _counters["runs_capture_unready"] += 1
            _fallback("int64 ids cast under capture")
            _backend_dispatches["declined"] += 1
            return None
        ids = ids.to(torch.int32)
    out = output
    if out is None:
        out = torch.empty(
            (T, input.shape[1]), dtype=torch.bfloat16, device=input.device
        )
    module.run(
        input,
        fc1_expert_weights,
        quant_scales[0],
        fc2_expert_weights,
        quant_scales[1],
        ids,
        token_final_scales,
        out,
        ws,
    )
    _counters["runs_dispatched"] += 1
    _backend_dispatches["specialized"] += 1
    if capturing:
        _counters["runs_dispatched_capturing"] += 1
    if not _announced:
        # one line per process so a measurement can name the kernel behind
        # its numbers (normal operation, hence INFO)
        _announced = True
        logger.info(
            "moe_fp8_grouped_smallt_sm90: serving small-T dsfp8 fused MoE "
            "(num_tokens=%d, experts=%d, top_k=%d)",
            T,
            fc1_expert_weights.shape[0],
            KTOP,
        )
    return out
