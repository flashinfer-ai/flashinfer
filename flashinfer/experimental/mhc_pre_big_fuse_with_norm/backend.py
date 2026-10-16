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

# mHC pre big-fuse with the fused output RMSNorm (``norm_weight`` variant).
#
# The split-K partials of the mHC mix projection (``gemm_out_mul`` [NS, T, 24],
# ``gemm_out_sqrsum`` [NS, T]) plus the residual streams [T, 4, H] and the next
# sublayer's RMSNorm weight [H] go in; ``post_mix`` [T, 4], ``comb_mix`` [T, 16]
# (row-major 4x4) and the RMSNorm'd ``layer_input`` [T, H] (bf16) come out. For
# H = 4096 on SM100 / SM103 a thread-block-cluster kernel spreads every token
# over 4 CTAs (2 CTAs for T > 24; decode regime, T <= 32, NS in {8, 64}); every
# other call runs the composable fp32 torch implementation below, which is the
# arithmetic contract of the op.

from __future__ import annotations

import os
import struct
import threading
from typing import Optional

import torch

from ...jit.core import logger
from .jit import gen_mhc_pre_big_fuse_with_norm_module

# ``1`` makes every call skip the cluster kernel and run the composable
# implementation (read at call time, for A/B testing the kernel).
KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# What the cluster kernel is compiled for. The model constants are compile-time
# constants of the kernel and are compared as fp32; calls outside this set run
# the composable implementation.
HIDDEN_SIZE = 4096
HC_MULT = 4
MIX = HC_MULT * (2 + HC_MULT)
NUM_SPLITS = frozenset({8, 64})
MIN_TOKENS = 1
MAX_TOKENS = 32
COMPUTE_CAPABILITIES = frozenset({(10, 0), (10, 3)})
CONSTANTS = {
    "rms_eps": 1e-5,
    "hc_pre_eps": 1e-6,
    "hc_sinkhorn_eps": 1e-6,
    "hc_post_mult_value": 2.0,
    "sinkhorn_repeat": 20,
    "norm_eps": 1e-5,
}


def _fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


_CONSTANTS_FP32 = {k: _fp32(v) for k, v in CONSTANTS.items() if k != "sinkhorn_repeat"}

_lock = threading.Lock()
_state: dict = {"module": None, "module_error": None}
_device_capability: dict = {}
# Call counters (kernel calls, composable calls and the reason of every
# rejected support check), for debugging and tests.
STATS: dict = {"kernel_calls": 0, "composable_calls": 0, "rejects": {}}


def _reject(reason: str) -> bool:
    rejects = STATS["rejects"]
    rejects[reason] = rejects.get(reason, 0) + 1
    return False


def _is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _load_module():
    """Build / load the cluster kernel once (never during CUDA-graph capture)."""
    if _state["module"] is not None:
        return _state["module"]
    if _state["module_error"] is not None:
        return None
    with _lock:
        if _state["module"] is None and _state["module_error"] is None:
            try:
                _state["module"] = (
                    gen_mhc_pre_big_fuse_with_norm_module().build_and_load()
                )
            except Exception as exc:  # the composable path keeps the op working
                message = f"{type(exc).__name__}: {exc}"
                _state["module_error"] = message
                logger.warning(
                    "mhc_pre_big_fuse_with_norm: cluster kernel unavailable, "
                    f"using the composable implementation ({message[:2000]})"
                )
    return _state["module"]


def _capability_for(device: torch.device, capturing: bool) -> Optional[tuple]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = _device_capability.get(index)
    if capability is None and not capturing:
        capability = tuple(torch.cuda.get_device_capability(index))
        _device_capability[index] = capability
    return capability


def _dense(t: Optional[torch.Tensor], shape: tuple, dtype: torch.dtype) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == dtype
        and tuple(t.shape) == shape
        and t.is_contiguous()
    )


def supported(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    post_mix: Optional[torch.Tensor] = None,
    comb_mix: Optional[torch.Tensor] = None,
    layer_input: Optional[torch.Tensor] = None,
) -> bool:
    """Return True when :func:`run` launches the cluster kernel for these arguments."""
    if os.environ.get(KILL_SWITCH_ENV, "") == "1":
        return _reject("kill_switch")
    # cheap shape rejects first (callers may probe every call, decode or not)
    if not isinstance(gemm_out_mul, torch.Tensor) or gemm_out_mul.dim() != 3:
        return _reject("shape")
    ns, t_tokens, mix = (int(v) for v in gemm_out_mul.shape)
    if mix != MIX:
        return _reject("shape")
    if ns not in NUM_SPLITS:
        return _reject("num_splits")
    if not MIN_TOKENS <= t_tokens <= MAX_TOKENS:
        return _reject("num_tokens")
    if norm_weight is None:
        return _reject("no_norm_weight")
    if (
        _fp32(rms_eps) != _CONSTANTS_FP32["rms_eps"]
        or _fp32(hc_pre_eps) != _CONSTANTS_FP32["hc_pre_eps"]
        or _fp32(hc_sinkhorn_eps) != _CONSTANTS_FP32["hc_sinkhorn_eps"]
        or _fp32(hc_post_mult_value) != _CONSTANTS_FP32["hc_post_mult_value"]
        or _fp32(norm_eps) != _CONSTANTS_FP32["norm_eps"]
        or int(sinkhorn_repeat) != CONSTANTS["sinkhorn_repeat"]
    ):
        return _reject("constants")
    if not (
        _dense(gemm_out_mul, (ns, t_tokens, MIX), torch.float32)
        and _dense(gemm_out_sqrsum, (ns, t_tokens), torch.float32)
        and _dense(hc_scale, (3,), torch.float32)
        and _dense(hc_base, (MIX,), torch.float32)
        and _dense(residual, (t_tokens, HC_MULT, HIDDEN_SIZE), torch.bfloat16)
        and _dense(norm_weight, (HIDDEN_SIZE,), torch.bfloat16)
    ):
        return _reject("input_layout")
    if post_mix is not None and not _dense(
        post_mix, (t_tokens, HC_MULT), torch.float32
    ):
        return _reject("output_layout")
    if comb_mix is not None and not (
        _dense(comb_mix, (t_tokens, HC_MULT * HC_MULT), torch.float32)
        or _dense(comb_mix, (t_tokens, HC_MULT, HC_MULT), torch.float32)
    ):
        return _reject("output_layout")
    if layer_input is not None and not _dense(
        layer_input, (t_tokens, HIDDEN_SIZE), torch.bfloat16
    ):
        return _reject("output_layout")
    # 16-byte vector loads / stores
    for t in (gemm_out_mul, hc_base, residual, norm_weight, comb_mix, layer_input):
        if t is not None and t.data_ptr() % 16 != 0:
            return _reject("alignment")
    # device last
    tensors = [gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, norm_weight]
    tensors += [t for t in (post_mix, comb_mix, layer_input) if t is not None]
    device = residual.device
    if device.type != "cuda" or any(t.device != device for t in tensors):
        return _reject("device")
    capturing = _is_capturing()
    capability = _capability_for(device, capturing)
    if capability is None:
        return _reject("capture_unready")
    if capability not in COMPUTE_CAPABILITIES:
        return _reject("arch")
    if _state["module"] is None:
        if capturing:
            return _reject("capture_unready")
        if _load_module() is None:
            return _reject("module_unavailable")
    return True


def composable(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input: torch.Tensor,
) -> None:
    """fp32 torch implementation of the op contract (any NS / T / H, HC = 4)."""
    t_tokens, hc, hidden = residual.shape
    f32 = torch.float32
    sq = gemm_out_sqrsum.to(f32).sum(dim=0)
    rms = torch.rsqrt(sq / (hc * hidden) + rms_eps)
    mixes = gemm_out_mul.to(f32).sum(dim=0) * rms.unsqueeze(-1)
    scale = hc_scale.to(f32)
    base = hc_base.to(f32)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + hc_pre_eps
    post = (
        torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
        * hc_post_mult_value
    )
    comb = mixes[:, 2 * hc :].view(t_tokens, hc, hc) * scale[2] + base[2 * hc :].view(
        1, hc, hc
    )
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(int(sinkhorn_repeat) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    ol = pre[:, 0:1] * residual[:, 0, :].to(f32)
    for j in range(1, hc):
        ol = ol + pre[:, j : j + 1] * residual[:, j, :].to(f32)
    ob = ol.to(torch.bfloat16)
    r = torch.rsqrt(ol.square().mean(dim=-1, keepdim=True) + norm_eps)
    post_mix.copy_(post.view(post_mix.shape))
    comb_mix.copy_(comb.reshape(comb_mix.shape))
    layer_input.copy_((ob.to(f32) * r * norm_weight.to(f32)).to(torch.bfloat16))


def run(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    post_mix: Optional[torch.Tensor] = None,
    comb_mix: Optional[torch.Tensor] = None,
    layer_input: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_tokens, hc, hidden = residual.shape
    device = residual.device
    if post_mix is None:
        post_mix = torch.empty((t_tokens, hc), dtype=torch.float32, device=device)
    if comb_mix is None:
        comb_mix = torch.empty((t_tokens, hc * hc), dtype=torch.float32, device=device)
    if layer_input is None:
        layer_input = torch.empty(
            (t_tokens, hidden), dtype=torch.bfloat16, device=device
        )
    args = (
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        post_mix,
        comb_mix,
        layer_input,
    )
    if supported(*args):
        _state["module"].mhc_pre_big_fuse_with_norm_cluster(
            post_mix,
            comb_mix,
            layer_input,
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
            bool(enable_pdl),
        )
        STATS["kernel_calls"] += 1
    else:
        composable(*args)
        STATS["composable_calls"] += 1
    return post_mix, comb_mix, layer_input


def warmup(device: Optional[torch.device] = None) -> bool:
    """Load the cluster kernel (and the device memo) outside CUDA-graph capture."""
    if _is_capturing():
        return False
    device = torch.device("cuda") if device is None else torch.device(device)
    if _capability_for(device, False) not in COMPUTE_CAPABILITIES:
        return False
    return _load_module() is not None
