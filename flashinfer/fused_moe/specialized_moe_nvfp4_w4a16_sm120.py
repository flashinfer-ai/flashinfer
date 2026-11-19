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

"""Specialized agent-generated, shape-specialized NVFP4 W4A16
(bf16-activation) routed-MoE kernels for SM120 at the contract E=128,
top_k=6, hidden=2688, intermediate=1856, non-gated ReLU^2 (the
Nemotron-3.5-Lightning serving geometry).

This module is the kernels' OWN dispatch surface, and its sole public home
is the unified MoE API: ``SpecializedW4A16Config`` (``api.py``, a default
backend candidate on SM120) and ``SpecializedW4A16Runner`` (``runners.py``)
call :func:`specialized_support_reasons` / :func:`run_specialized_moe_nvfp4_w4a16`
directly. The kernels are not b12x kernels and are not reachable through
``b12x_fused_moe`` or ``B12xMoEWrapper``: ``MoELayer``'s ``backend=auto``
cross-backend shootout times them as their own named candidate
(``"specialized_w4a16"``) next to the stock ``"b12x_w4a16"`` backend and
routes to them where they win. An explicitly selected other backend is
never served by them, and the runner never serves a call with a different
kernel. ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` rejects the backend at
layer construction and makes the runner decline every call
(:func:`warmup` refuses without loading or launching the module).

Guard order: GPU-free checks (kill switch, geometry, dtypes, allowlist)
-> layout checks (shapes, contiguity, alignment) -> device checks
(compute capability, graph-capture state). Refusals are recorded in
:data:`fallback_reasons` and dispatch counts in :data:`dispatch_stats`.

Weight layout, consumed in place (no repack): ``w1 uint8 [E, I, H/2]``
(two E2M1 codes per byte, low nibble = element 2i), ``w1_scale
float8_e4m3fn [E, I, H/16]`` row-major group-16 (UNswizzled), ``w1_alpha
float32 [E]``; same pattern for ``w2``.

Build before capture. Scratch is a temporary allocated on the launch
stream: during torch.cuda.graph capture it belongs to the active graph
pool. Independent graphs with private pools can replay concurrently;
as for any PyTorch graph, callers explicitly sharing a pool must order
its uses. Eager launches do not borrow captured addresses.
"""

import collections
import functools
import json
import os
import pathlib
from typing import Any, List, Optional, Tuple, Union

import torch

with open(
    pathlib.Path(__file__).with_name("specialized_moe_nvfp4_w4a16_sm120_allowlist.json")
) as _f:
    _ALLOWLIST = json.load(_f)

_C = _ALLOWLIST["contract"]
_T_EXACT = tuple(_ALLOWLIST["token_counts_exact"])
_T_MIN_LARGE = int(_ALLOWLIST["token_counts_min_large"])

ENV_DISABLE = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Observability: per-backend dispatch counts and recent refusal reasons.
dispatch_stats: collections.Counter = collections.Counter()
fallback_reasons: collections.deque = collections.deque(maxlen=64)


def _workspace_bytes(t: int) -> int:
    """Byte slices used by Run's selected family. Native checks the bound before launch."""
    e, h, i = _C["num_experts"], _C["hidden_size"], _C["intermediate_size"]
    r = t * _C["top_k"]

    def aligned(n: int) -> int:
        return (n + 255) & ~255

    if t == 64:
        groups = (r + e) // 2 + 1
        return r * i * 2 + (r + groups) * 4
    size = aligned(r * i * 2)
    if t < 2048:
        size += aligned(r * h * 2)
    bins = e if t == 1024 or t >= 2048 else 1
    size += aligned(r * bins * 4) + 5 * aligned((e + 1) * 4)
    if t > 64 and t != 1024:
        mm = 64 if t < 8192 else 96
        tiles = (r + mm - 1) // mm + e
        size += sum(aligned(tiles * ((n + 127) // 128) * 4) for n in (i, h))
    if t == 1 or t > 64:
        size += 256  # barrier slots
    return size


def _disabled() -> bool:
    return os.environ.get(ENV_DISABLE, "0") == "1"


def _is_capturing(device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _launch_key(device, t):
    # Mirror native Run, not the autotuner's token buckets.
    variant = t if t in (1, 4, 16, 64, 1024) else ("wideM64" if t < 8192 else "wideM96")
    return device, variant


# A receipt retains the very module that launched. Capture only reads this
# table: it never enters the builder, even on a cache miss. Tensor devices
# here are indexed; an unqualified 'cuda' is not a receipt key.
_warmed: dict[tuple[torch.device, Union[int, str]], Any] = {}


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
    """Guard, in the required order. Returns (supported, refusal_reasons).

    Runs under autotune tuning mode by design: the unified
    ``SpecializedW4A16Runner`` is a first-class cross-backend autotune
    candidate, so when it is the thing being timed the specialized kernel
    is what runs."""
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
    try:
        capability = torch.cuda.get_device_capability(x.device)
    except (AssertionError, ValueError, RuntimeError):
        return False, ["compute_capability:unavailable"]
    if list(capability) != _C["compute_capability"]:
        reasons.append("compute_capability")
    # cvt.rn.bf16x2.e2m1x2 needs a CUDA >= 13.2-line ptxas (the shared b12x
    # gate only requires CUDA 13); refuse here so ``auto`` keeps the stock
    # path on a 13.0/13.1 toolkit instead of failing the JIT build.
    try:
        from ..jit.cpp_ext import get_cuda_version

        _v = get_cuda_version()
        if (_v.major, _v.minor) < (13, 2):
            reasons.append(f"cuda_version:{_v.major}.{_v.minor}")
    except Exception:  # noqa: BLE001
        reasons.append("cuda_version:unknown")
    if reasons:
        return False, reasons
    try:
        capturing = _is_capturing(x.device)
    except (ValueError, RuntimeError):
        return False, ["graph_capture_state_unknown"]
    if capturing:
        if _launch_key(x.device, T) not in _warmed:
            return False, ["graph_capture_before_warmup"]
        return True, []
    if not reasons:
        # The module must be buildable/loadable (e.g. FLASHINFER_DISABLE_JIT
        # with a cold JIT cache raises); record a refusal so ``auto`` falls
        # back to the stock path instead of propagating a build error.
        try:
            _module()
        except Exception as e:  # noqa: BLE001
            reasons.append(f"module_unavailable:{type(e).__name__}")
    return not reasons, reasons


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
    place). The unified ``SpecializedW4A16Runner`` goes through
    :func:`specialized_support_reasons` first."""
    return _launch(
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
        _warmed,
    )


def _launch(
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
    receipts,
):
    with torch.cuda.device(x.device):
        capturing = _is_capturing(x.device)
        t = int(x.shape[0])
        key = _launch_key(x.device, t)
        if capturing:
            module = _warmed.get(key)
            if module is None or receipts is not _warmed:
                raise RuntimeError(
                    "specialized W4A16: eager warmup required before CUDA-graph capture"
                )
        else:
            module = _module()
        tw = token_final_scales.float()
        size = _workspace_bytes(t)
        # A fresh lease, not a shared tensor. Eager reuse is stream ordered
        # by the allocator; geometric size classes avoid a new large block
        # for every +1 token. Capture needs only the actual layout size.
        if not capturing:
            size = 1 << (size - 1).bit_length()
        scratch = torch.empty(size, dtype=torch.uint8, device=x.device)
        output.zero_()
        module.run(
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
            scratch,
        )
        if not capturing:
            receipts[key] = module
    return output


def warmup(
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    token_counts=(16384, 4096, 1024, 64, 16, 4, 1),
    device: Union[str, torch.device] = "cuda",
) -> None:
    """Build the module and exercise the requested admitted shapes before capture."""
    if _disabled():
        raise RuntimeError(
            f"specialized NVFP4-W4A16 SM120 warmup refused: {ENV_DISABLE}=1 "
            "disables the backend entirely"
        )
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("specialized W4A16 warmup requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if _is_capturing(device):
        raise RuntimeError(
            "specialized W4A16 warmup is not allowed during CUDA-graph capture"
        )
    weights = (w1_weight, w1_weight_sf, w1_alpha, w2_weight, w2_weight_sf, w2_alpha)
    if not all(isinstance(w, torch.Tensor) and w.device == device for w in weights):
        raise ValueError(
            "warmup requires all six weight operands on the requested device"
        )
    counts = tuple(t for t in token_counts if t in _T_EXACT or t >= _T_MIN_LARGE)
    pending: dict[tuple[torch.device, Union[int, str]], Any] = {}
    top_k, h = _C["top_k"], _C["hidden_size"]
    # Validate operands even for an empty filtered request, without launching.
    for t in counts or (1,):
        x = torch.zeros(t, h, dtype=torch.bfloat16, device=device)
        ids = torch.zeros(t, top_k, dtype=torch.int32, device=device)
        tw = torch.full((t, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
        out = torch.empty_like(x)
        ok, reasons = specialized_support_reasons(
            x,
            w1_weight,
            w1_weight_sf,
            w2_weight,
            w2_weight_sf,
            ids,
            tw,
            _C["num_experts"],
            top_k,
            w1_alpha=w1_alpha,
            w2_alpha=w2_alpha,
            output=out,
        )
        if not ok:
            raise RuntimeError(f"specialized W4A16 warmup refused: {reasons}")
        if counts:
            _launch(x, *weights, ids, tw, out, pending)
    torch.cuda.synchronize(device)
    _warmed.update(pending)
