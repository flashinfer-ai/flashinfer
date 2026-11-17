# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The "specialized" backend of ``b12x_fused_moe``: agent-generated,
shape-specialized NVFP4 W4A16 (bf16-activation) routed-MoE kernels for
SM120 at the contract E=128, top_k=6, hidden=2688, intermediate=1856,
non-gated ReLU^2 (the Nemotron-3.5-Lightning serving geometry).

Dispatch model (the ``backend`` kwarg of :func:`b12x_fused_moe` and of
``B12xMoEWrapper`` — the wrapper is what the unified ``MoELayer`` path's
``B12xW4A16Runner`` drives, so both the flat entry and ``MoELayer`` reach
the same :func:`maybe_run_specialized` gate):
  * ``backend="auto"`` routes to "specialized" only on the allowlisted
    shapes/contract (``specialized_moe_nvfp4_w4a16_sm120_allowlist.json``);
    everything else keeps the stock path. An explicitly selected backend is
    never silently swapped.
  * ``backend="specialized"`` raises with the recorded refusal reasons when
    the call cannot be served (this module has no composable fallback of
    its own; nothing is silently swapped to another fused backend).
  * ``backend="b12x"`` forces the stock path; the specialized kernels are
    never consulted.
  * ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` disables the backend
    entirely (``auto`` behaves as stock; explicit "specialized" raises;
    :func:`warmup` refuses without loading or launching the module).

Guard order: GPU-free checks (kill switch, geometry, dtypes, allowlist)
-> layout checks (shapes, contiguity, alignment) -> device checks
(compute capability, graph-capture / autotune state). Refusals are
recorded in :data:`fallback_reasons` and per-backend dispatch counts in
:data:`dispatch_stats`.

Weight layout, consumed in place (no repack): ``w1 uint8 [E, I, H/2]``
(two E2M1 codes per byte, low nibble = element 2i), ``w1_scale
float8_e4m3fn [E, I, H/16]`` row-major group-16 (UNswizzled), ``w1_alpha
float32 [E]``; same pattern for ``w2``.

CUDA-graph contract: call :func:`warmup` (largest T first) on the capture
stream before capturing — the module keeps a process-global workspace
(cudaMalloc growth) and synchronizes on stream transitions, both illegal
inside capture; the guard refuses capture until warmup has run, when the
captured token count exceeds the largest token count already run on the
device (``graph_capture_token_count_exceeds_warmed``), and when the
capture stream or device differs from the one the backend last ran on
(``graph_capture_stream_mismatch`` — re-run :func:`warmup` on the capture
stream). Replay is single-flight: captured graphs bake the shared
process-global workspace addresses in, so two captured graphs must not
replay concurrently, and the workspace must not grow (no call with a
larger token count, on another device) while a captured graph may still
be replayed. The T=1 path uses a cooperative launch (stream-capturable;
verified). All launches go to the caller's current stream, re-queried
per call.
"""

import collections
import functools
import json
import os
import pathlib
from typing import List, Optional, Tuple

import torch

with open(
    pathlib.Path(__file__).with_name("specialized_moe_nvfp4_w4a16_sm120_allowlist.json")
) as _f:
    _ALLOWLIST = json.load(_f)

_C = _ALLOWLIST["contract"]
_T_EXACT = tuple(_ALLOWLIST["token_counts_exact"])
_T_MIN_LARGE = int(_ALLOWLIST["token_counts_min_large"])

ENV_DISABLE = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Selectable backends of the b12x W4A16 entry points (the canonical tuple;
# ``flashinfer.fused_moe.cute_dsl.b12x_moe`` re-exports it). "b12x" is the
# stock micro/static/dynamic dispatch; "specialized" is this agent-generated
# shape-specialized SM120 W4A16 backend; "auto" picks "specialized" only on
# its allowlisted shapes/contract and the stock path everywhere else. An
# explicitly selected backend is never silently swapped.
BACKENDS = ("auto", "specialized", "b12x")

# Observability: per-backend dispatch counts and recent refusal reasons.
dispatch_stats: collections.Counter = collections.Counter()
fallback_reasons: collections.deque = collections.deque(maxlen=64)

# Warmup / last-run state for the CUDA-graph capture guard: the largest
# token count run per current device (the process-global workspace grows
# monotonically with it) and the stream/device of the last backend run.
_warmed = {"done": False, "max_t": 0, "device": None, "stream": None}


def _note_run(device: torch.device, t: int) -> None:
    if _warmed["device"] != device:
        # The native workspace is freed and re-created on a device change.
        _warmed["max_t"] = 0
    _warmed["device"] = device
    _warmed["stream"] = torch.cuda.current_stream(device).cuda_stream
    _warmed["max_t"] = max(_warmed["max_t"], t)


def _disabled() -> bool:
    return os.environ.get(ENV_DISABLE, "0") == "1"


def _is_capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _autotune_active() -> bool:
    try:
        from ..autotuner import AutoTuner

        return AutoTuner.get().is_tuning_mode
    except Exception:  # noqa: BLE001
        return False


@functools.cache
def _module():
    from ..jit.specialized_moe_nvfp4_w4a16_sm120 import (
        gen_specialized_moe_nvfp4_w4a16_sm120_module,
    )

    return gen_specialized_moe_nvfp4_w4a16_sm120_module().build_and_load()


def specialized_support_reasons(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    *,
    w1_alpha: Optional[torch.Tensor] = None,
    w2_alpha: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    activation: str = "relu2",
    quant_mode: Optional[str] = "w4a16",
    source_format: str = "modelopt",
) -> Tuple[bool, List[str]]:
    """Guard, in the required order. Returns (supported, refusal_reasons)."""
    reasons: List[str] = []
    E = _C["num_experts"]
    TOPK = _C["top_k"]
    H = _C["hidden_size"]
    I = _C["intermediate_size"]
    # --- GPU-free checks
    if _disabled():
        reasons.append("kill_switch")
    if quant_mode != _C["quant_mode"]:
        reasons.append(f"quant_mode:{quant_mode}")
    if source_format != _C["source_format"]:
        reasons.append(f"source_format:{source_format}")
    if activation != _C["activation"]:
        reasons.append(f"activation:{activation}")
    if num_experts != E or top_k != TOPK:
        reasons.append(f"geometry:E{num_experts}/k{top_k}")
    if x.dim() != 2 or x.shape[1] != H:
        reasons.append(f"hidden_size:{tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        reasons.append(f"x_dtype:{x.dtype}")
    T = x.shape[0] if x.dim() == 2 else -1
    if T not in _T_EXACT and T < _T_MIN_LARGE:
        reasons.append(f"token_count:{T}")
    if reasons:
        return False, reasons
    # --- layout checks
    if w1_weight.shape != (E, I, H // 2) or w1_weight.dtype != torch.uint8:
        reasons.append("w1_layout")
    if w2_weight.shape != (E, H, I // 2) or w2_weight.dtype != torch.uint8:
        reasons.append("w2_layout")
    if (
        w1_weight_sf.shape != (E, I, H // 16)
        or w1_weight_sf.dtype != torch.float8_e4m3fn
    ):
        reasons.append("w1_scale_layout_rowmajor_required")
    if (
        w2_weight_sf.shape != (E, H, I // 16)
        or w2_weight_sf.dtype != torch.float8_e4m3fn
    ):
        reasons.append("w2_scale_layout_rowmajor_required")
    if (
        token_selected_experts.shape != (T, TOPK)
        or token_selected_experts.dtype != torch.int32
    ):
        reasons.append("topk_ids_layout")
    if token_final_scales.shape != (T, TOPK):
        reasons.append("topk_weights_layout")
    if not all(
        t.is_contiguous()
        for t in (
            x,
            w1_weight,
            w1_weight_sf,
            w2_weight,
            w2_weight_sf,
            token_selected_experts,
            token_final_scales,
        )
    ):
        reasons.append("non_contiguous")
    # The activation vector loads (all paths) need 32-byte alignment; the
    # weight/scale loads need 16.
    if x.data_ptr() % 32 != 0 or not all(
        t.data_ptr() % 16 == 0
        for t in (w1_weight, w1_weight_sf, w2_weight, w2_weight_sf)
    ):
        reasons.append("alignment")
    for name, a in (("w1_alpha", w1_alpha), ("w2_alpha", w2_alpha)):
        # The kernels consume both per-expert alphas; a missing one is a
        # refusal, not a crash inside the native runner.
        if a is None:
            reasons.append(f"{name}_missing")
        elif a.dtype != torch.float32 or a.shape != (E,) or not a.is_contiguous():
            reasons.append(f"{name}_layout")
    if output is not None and (
        output.dtype != torch.bfloat16
        or output.shape != (T, H)
        or not output.is_contiguous()
        or output.data_ptr() % 16 != 0
    ):
        reasons.append("output_layout")
    devices = {
        t.device
        for t in (
            x,
            w1_weight,
            w1_weight_sf,
            w2_weight,
            w2_weight_sf,
            token_selected_experts,
            token_final_scales,
            w1_alpha,
            w2_alpha,
            output,
        )
        if t is not None
    }
    if len(devices) != 1:
        reasons.append("device_mismatch")
    if reasons:
        return False, reasons
    # --- device checks
    if list(torch.cuda.get_device_capability(x.device)) != _C["compute_capability"]:
        reasons.append("compute_capability")
    # cvt.rn.bf16x2.e2m1x2 needs a CUDA >= 13.1-line ptxas (the shared b12x
    # gate only requires CUDA 13); refuse here so ``auto`` keeps the stock
    # path on a 13.0 toolkit instead of failing the JIT build.
    try:
        from ..jit.cpp_ext import get_cuda_version

        _v = get_cuda_version()
        if (_v.major, _v.minor) < (13, 1):
            reasons.append(f"cuda_version:{_v.major}.{_v.minor}")
    except Exception:  # noqa: BLE001
        reasons.append("cuda_version:unknown")
    if _autotune_active():
        reasons.append("autotune_capture")
    if _is_capturing():
        if not _warmed["done"]:
            reasons.append("graph_capture_before_warmup")
        else:
            if _warmed["max_t"] < T:
                # Larger T grows the process-global workspace (cudaMalloc),
                # which is illegal inside capture.
                reasons.append("graph_capture_token_count_exceeds_warmed")
            if (
                _warmed["device"] != x.device
                or _warmed["stream"] != torch.cuda.current_stream(x.device).cuda_stream
            ):
                # The native runner synchronizes the previous stream on a
                # stream/device transition, which is illegal inside capture.
                # Re-run warmup on the capture stream.
                reasons.append("graph_capture_stream_mismatch")
    if not reasons:
        # The module must be buildable/loadable (e.g. FLASHINFER_DISABLE_JIT
        # with a cold JIT cache raises); record a refusal so ``auto`` falls
        # back to the stock path instead of propagating a build error.
        try:
            _module()
        except Exception as e:  # noqa: BLE001
            reasons.append(f"module_unavailable:{type(e).__name__}")
    return not reasons, reasons


def maybe_run_specialized(
    backend: str,
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    num_experts: int,
    top_k: int,
    *,
    w1_alpha: Optional[torch.Tensor],
    w2_alpha: Optional[torch.Tensor],
    output: torch.Tensor,
    activation: str,
    quant_mode: Optional[str],
    source_format: str,
    expert_parallel: bool = False,
) -> Optional[torch.Tensor]:
    """The single dispatch gate of the "specialized" backend, shared by both
    call sites (:func:`b12x_fused_moe` and ``B12xMoEWrapper.run`` — the
    latter is what the unified ``MoELayer`` path traverses) so the guard
    semantics cannot drift between them.

    Returns ``output`` (written in place) when the specialized backend
    served the call; ``None`` when the call keeps the stock path
    (``backend="b12x"``, or ``backend="auto"`` with the refusal recorded in
    :data:`fallback_reasons`). ``backend="specialized"`` raises with the
    refusal reasons instead of falling back — an explicitly selected backend
    is never silently swapped, and this module has no composable fallback of
    its own. ``quant_mode`` must already be normalized (e.g. ``"w4a16"``).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "b12x":
        return None
    if expert_parallel:
        ok, reasons = False, ["expert_parallelism"]
    else:
        ok, reasons = specialized_support_reasons(
            x,
            w1_weight,
            w1_weight_sf,
            w2_weight,
            w2_weight_sf,
            token_selected_experts,
            token_final_scales,
            num_experts,
            top_k,
            w1_alpha=w1_alpha,
            w2_alpha=w2_alpha,
            output=output,
            activation=activation,
            quant_mode=quant_mode,
            source_format=source_format,
        )
    if ok:
        dispatch_stats["specialized"] += 1
        return run_specialized_moe_nvfp4_w4a16(
            x,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            token_selected_experts,
            token_final_scales,
            output,
        )
    fallback_reasons.append(tuple(reasons))
    if backend == "specialized":
        # never silently swapped to another fused backend; this module has
        # no composable fallback of its own
        raise ValueError(
            "backend='specialized' cannot serve this call: " + ", ".join(reasons)
        )
    dispatch_stats["b12x"] += 1
    return None


def run_specialized_moe_nvfp4_w4a16(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the specialized kernels (DPS: writes ``output`` [T, H] bf16 in
    place). Callers go through :func:`specialized_support_reasons` first."""
    tw = token_final_scales
    if tw.dtype != torch.float32:
        tw = tw.float()
    _note_run(x.device, int(x.shape[0]))
    output.zero_()
    _module().run(
        x,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        token_selected_experts,
        tw,
        output,
    )
    return output


def warmup(
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    token_counts=(16384, 4096, 1024, 64, 16, 4, 1),
    device: str = "cuda",
) -> None:
    """Build the module, grow the workspace to its ladder maximum and run
    each admitted shape once. Call BEFORE CUDA graph capture, on the
    stream that will be captured (largest T first keeps the workspace
    monotone)."""
    if _disabled():
        raise RuntimeError(
            f"specialized NVFP4-W4A16 SM120 warmup refused: {ENV_DISABLE}=1 "
            "disables the backend entirely"
        )
    TOPK, H = _C["top_k"], _C["hidden_size"]
    for T in token_counts:
        if T not in _T_EXACT and T < _T_MIN_LARGE:
            continue
        x = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
        ids = torch.zeros(T, TOPK, dtype=torch.int32, device=device)
        tw = torch.full((T, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device)
        out = torch.empty(T, H, dtype=torch.bfloat16, device=device)
        run_specialized_moe_nvfp4_w4a16(
            x,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            ids,
            tw,
            out,
        )
    torch.cuda.synchronize()
    _warmed["done"] = True
