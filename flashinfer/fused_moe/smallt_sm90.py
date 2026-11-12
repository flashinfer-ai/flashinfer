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

Specialized SM90 small-batch (decode-step) fast-path for
``cutlass_fused_moe(..., use_deepseek_fp8_block_scale=True)``
(csrc/moe_fp8_grouped_smallt_sm90.cu,
flashinfer/jit/moe_fp8_grouped_smallt_sm90.py).

ALLOWLIST (fails closed to the stock cutlass path otherwise): SM90 device,
DeepSeek-style fp8 [128,128]-block-scale weights (``quant_scales ==
[fc1_block_scales, fc2_block_scales]``), bf16 unquantized activations, bf16
output, gated SiLU (Swiglu) activation, no biases / input_sf / swiglu
parameters / expert parallelism / min-latency mode, and token counts inside the measured won bands **T <= 16 or 256 <= T <= 4096**
— the measured envelope where this kernel beats the serving engines'
dispatched triton fused-MoE on the captured GLM-class routing. At the
measured T >= 64 classes the kernel LOSES to that incumbent (0.55-0.81x),
so batches above the bound keep the stock path by design, not by omission.
The kernel consumes the cutlass_fused_moe fc1 stacking as-is (rows [0:I] =
up, rows [I:2I] = gate).

``FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90=1`` (read per call)
disables the fast-path and restores the stock cutlass route bitwise.

The dispatch never allocates on the hot path except the output tensor when
the caller did not pass one (the stock path allocates it in the same
place): the fp16 intermediate workspace is cached per device at its T<=16
capacity and the JIT build happens on the first eligible call outside any
CUDA-graph capture (a capture without a built module takes the stock path).
"""

import os
from typing import Any, Dict, List, Optional

import torch

from ..jit.core import logger
from ..utils import get_compute_capability

_DISABLE_ENV = "FLASHINFER_DISABLE_MOE_FP8_GROUPED_SMALLT_SM90"
_MAX_TOKENS = 16  # small-path band cap (measured won rows T=1/4/16)
# Expansion (measured vs FI's own grouped-fp8 route at contract): the
# candidate also wins the T=256/1024/4096 classes against the stock cutlass
# dsfp8 path, so a second band [256, 4096] is admitted; interior points
# (e.g. T=512/2048) are admitted by BAND INTERPOLATION between the measured
# classes, the same semantics as the shipped 1..16 band. The 17..255 gap —
# containing the measured T=64 class, where FI's own kernel held at contract
# (and in-venv) — stays on the stock path.
_BAND2_LO = 256
_BAND2_HI = 4096
_COMPUTE_CAPABILITY = (9, 0)

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
_workspaces: Dict[int, torch.Tensor] = {}
_announced = False


def _disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "") not in ("", "0")


def _moe_smallt_sm90_stats() -> Dict[str, Any]:
    """Introspection for benchmarks and tests: per-process call census and
    build state of the specialized SM90 small-T dsfp8 MoE kernel."""
    stats: Dict[str, Any] = dict(_counters)
    stats["module_loaded"] = _module is not None
    stats["module_error"] = _module_error
    return stats


def _reset_moe_smallt_sm90_stats() -> None:
    global _announced
    for key in _counters:
        _counters[key] = 0
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
    except Exception as exc:  # build or load failure: stock path, say why
        _module_error = f"{type(exc).__name__}: {exc}"[:800]
        _counters["build_failed"] += 1
        logger.warning_once(
            "moe_fp8_grouped_smallt_sm90: JIT build failed (%s); small-T "
            "dsfp8 calls are served by the stock cutlass path",
            _module_error,
        )
        return None
    return _module


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
    a reason string (logged once; the stock cutlass path serves the call).
    Semantics first, then shapes/dtypes, device last; no synchronisation."""
    for name, val in flags.items():
        if val:
            return name
    if input.dtype != torch.bfloat16 or input.dim() != 2:
        return f"input {input.dtype} dim={input.dim()}"
    T = input.shape[0]
    if not (1 <= T <= _MAX_TOKENS or _BAND2_LO <= T <= _BAND2_HI):
        return (f"num_tokens={T} outside the measured won bands "
                f"[1,{_MAX_TOKENS}] u [{_BAND2_LO},{_BAND2_HI}]")
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
    if ids.dtype not in (torch.int32, torch.int64) or ids.dim() != 2 or ids.shape[0] != T:
        return f"token_selected_experts {ids.dtype} {tuple(ids.shape)}"
    if tw is None or tw.dim() != 2 or tuple(tw.shape) != tuple(ids.shape):
        return "token_final_scales shape"
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
    if input.device.type != "cuda":
        return f"device {input.device}"
    try:
        if tuple(get_compute_capability(input.device)) != _COMPUTE_CAPABILITY:
            return "not SM90"
    except Exception as exc:
        return f"compute capability query failed: {exc}"[:120]
    return None


def _moe_smallt_sm90_try_dispatch(
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
    """Serve the call with the specialized small-T kernel, or return None so
    the stock cutlass path runs (nothing launched)."""
    global _announced
    _counters["calls_seen"] += 1
    if _disabled():
        _counters["calls_disabled"] += 1
        logger.info_once(
            "moe_fp8_grouped_smallt_sm90: disabled by %s=%s; the stock "
            "cutlass path serves small-T dsfp8 calls",
            _DISABLE_ENV,
            os.environ.get(_DISABLE_ENV, ""),
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
    except Exception as exc:  # a guard failure must never break the stock path
        reason = f"guard_error:{type(exc).__name__}: {exc}"[:300]
    if reason is not None:
        _counters["calls_ineligible"] += 1
        logger.info_once(
            "moe_fp8_grouped_smallt_sm90: not dispatched (%s); the stock "
            "cutlass path serves this call",
            reason,
        )
        return None
    capturing = bool(
        hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    )
    if capturing and _module is None:
        # never compile inside a capture; the stock path is baked
        _counters["runs_capture_unready"] += 1
        return None
    module = _get_module()
    if module is None:
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
    ws = _workspaces.get(dev_index)
    if ws is None or ws.numel() < need or ws.device != input.device:
        if capturing:
            # never allocate the cache inside a capture
            _counters["runs_capture_unready"] += 1
            return None
        ws = torch.empty(need, dtype=torch.uint8, device=input.device)
        _workspaces[dev_index] = ws
        _counters["workspace_allocs"] += 1

    ids = token_selected_experts
    if ids.dtype != torch.int32:
        if capturing:
            _counters["runs_capture_unready"] += 1
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
