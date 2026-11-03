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

# Backend of the experimental MNNVL all-reduce fused with the mHC post-map and the split-K
# pre-norm partials: the supported surface, the module build and its cache, the support query,
# the fused launch and the composable path used outside the supported surface.

from __future__ import annotations

import functools
import threading
from typing import Any, Dict, Optional, Tuple

import torch

from ...comm.trtllm_mnnvl_ar import (
    MNNVLAllReduceFusionWorkspace,
    MNNVLAllreduceFusionStrategy,
    trtllm_mnnvl_allreduce,
)
from ...jit.core import logger
from .jit import gen_allreduce_mhc_post_prenorm_module

# Supported surface of the fused kernel; the constants of csrc/allreduce_mhc_post_prenorm.cuh.
WORLD_SIZE = 8
HIDDEN_SIZE = 4096
HC_MULT = 4
MIX = HC_MULT * (2 + HC_MULT)  # rows of the pre-norm projection fn
NUM_SPLITS = 8  # contiguous hidden splits of the split-K partials
MAX_TOKENS = 32
DTYPE = torch.bfloat16
COMPUTE_CAPABILITY_MAJORS = frozenset({10})  # SM100 / SM103
# From this token count every residual element is read and written by the same thread. Below it
# several CTAs read the residual slice that one of them overwrites, so an in-place update
# (residual_out sharing memory with residual) goes through a separate buffer.
IN_PLACE_MIN_TOKENS = 5

_lock = threading.Lock()
_module: Optional[Any] = None
_stats: Dict[str, Any] = {
    "module_loaded": False,
    "load_error": None,
    "dispatch_count": 0,
    "dispatch_by_num_tokens": {},
    "fallback_count": 0,
    "capture_unready_count": 0,
}
_dispatch_logged = False


@functools.cache
def _device_cc_major(device_index: int) -> int:
    # Keyed per device: a value cached for one device must not answer for another.
    return torch.cuda.get_device_capability(device_index)[0]


def _build_module() -> Any:
    return gen_allreduce_mhc_post_prenorm_module().build_and_load()


def _get_module(allow_build: bool) -> Optional[Any]:
    """The loaded module; builds and loads it only when ``allow_build``."""
    global _module
    if _module is not None:
        return _module
    if not allow_build or _stats["load_error"] is not None:
        return None
    with _lock:
        if _module is None and _stats["load_error"] is None:
            try:
                _module = _build_module()
                _stats["module_loaded"] = True
            except Exception as exc:  # the composable path stays available
                _stats["load_error"] = str(exc)[:2000]
                logger.warning(
                    "allreduce_mhc_post_prenorm: fused module unavailable, "
                    "using the composable path: %s",
                    _stats["load_error"],
                )
    return _module


def prepare() -> bool:
    """Build and load the fused module outside CUDA-graph capture; True when it is loaded.

    A call made during capture never builds; it reports whether the module is already loaded.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return _module is not None
    return _get_module(allow_build=True) is not None


def _shape_supported(
    workspace: Any,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    if not isinstance(workspace, MNNVLAllReduceFusionWorkspace):
        return False
    if (
        dtype != DTYPE
        or workspace.tp_size != WORLD_SIZE
        or hidden_size != HIDDEN_SIZE
        or hc_mult != HC_MULT
        or not 1 <= num_tokens <= MAX_TOKENS
    ):
        return False
    if not workspace.is_buffer_size_sufficient(
        workspace.tp_size,
        num_tokens,
        hidden_size,
        dtype,
        MNNVLAllreduceFusionStrategy.ONESHOT,
    ):
        return False
    if device.type != "cuda":
        return False
    device_index = device.index if device.index is not None else 0
    return _device_cc_major(device_index) in COMPUTE_CAPABILITY_MAJORS


def is_supported(
    workspace: Any,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> bool:
    """Whether :func:`run` launches the fused kernel for this problem.

    Checks the supported surface, the workspace's one-shot capacity and the architecture, and
    loads the module (building it outside CUDA-graph capture only). The answer is the same on
    every rank for the same arguments and process state.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not _shape_supported(workspace, num_tokens, hidden_size, hc_mult, dtype, device):
        return False
    capturing = torch.cuda.is_current_stream_capturing()
    if _get_module(allow_build=not capturing) is None:
        if capturing:
            _stats["capture_unready_count"] += 1
        return False
    return True


def _composable(
    input: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    launch_with_pdl: bool,
    strategy: MNNVLAllreduceFusionStrategy,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stock MNNVL all-reduce, then the epilogue in fp32 (post-map) and fp64 (partials)."""
    # The stock all-reduce reads dense rows.
    x = trtllm_mnnvl_allreduce(
        input.contiguous(), workspace, launch_with_pdl, strategy=strategy
    )
    num_tokens, hc_mult, hidden_size = residual.shape
    mix = fn.shape[0]
    new_r = post_layer_mix.float().unsqueeze(-1) * x.float().unsqueeze(1)
    for k in range(hc_mult):
        new_r = new_r + comb_res_mix[:, k, :].float().unsqueeze(-1) * residual[
            :, k, :
        ].float().unsqueeze(1)
    residual_out = new_r.to(residual.dtype)
    width = hidden_size // NUM_SPLITS
    r64 = new_r.double().view(num_tokens, hc_mult, NUM_SPLITS, width)
    f64 = fn.double().view(mix, hc_mult, NUM_SPLITS, width)
    yp = torch.einsum("mjsh,njsh->smn", r64, f64).float().contiguous()
    rp = r64.square().sum(dim=(1, 3)).transpose(0, 1).float().contiguous()
    return yp, rp, residual_out


def _dtypes_supported(
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    yp: torch.Tensor,
    rp: torch.Tensor,
    residual_out: torch.Tensor,
) -> bool:
    """The kernel's dtypes for the tensors :func:`is_supported` does not see."""
    return (
        residual.dtype == DTYPE
        and residual_out.dtype == DTYPE
        and post_layer_mix.dtype == torch.float32
        and comb_res_mix.dtype == torch.float32
        and fn.dtype == torch.float32
        and yp.dtype == torch.float32
        and rp.dtype == torch.float32
    )


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the memory spans of ``a`` and ``b`` intersect (conservative for strided views)."""
    if a.numel() == 0 or b.numel() == 0 or a.device != b.device:
        return False

    def span(t: torch.Tensor) -> Tuple[int, int]:
        last = sum((n - 1) * s for n, s in zip(t.shape, t.stride(), strict=True))
        return t.data_ptr(), t.data_ptr() + (last + 1) * t.element_size()

    a_begin, a_end = span(a)
    b_begin, b_end = span(b)
    return a_begin < b_end and b_begin < a_end


def _dense_output(
    out: torch.Tensor, avoid: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out`` if the kernel can write it directly, else a contiguous buffer of its shape.

    With ``avoid``, an ``out`` that shares memory with it also gets a separate buffer.
    """
    if out.is_contiguous() and (avoid is None or not _overlaps(out, avoid)):
        return out
    return torch.empty_like(out, memory_format=torch.contiguous_format)


def run(
    input: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    workspace: MNNVLAllReduceFusionWorkspace,
    launch_with_pdl: bool = True,
    yp: Optional[torch.Tensor] = None,
    rp: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fused kernel on the supported surface, the composable path elsewhere."""
    global _dispatch_logged
    num_tokens, hc_mult, hidden_size = residual.shape
    mix = fn.shape[0]
    post_flat = post_layer_mix.reshape(num_tokens, hc_mult)
    if yp is None:
        yp = torch.empty(
            (NUM_SPLITS, num_tokens, mix), dtype=torch.float32, device=input.device
        )
    if rp is None:
        rp = torch.empty(
            (NUM_SPLITS, num_tokens), dtype=torch.float32, device=input.device
        )
    if residual_out is None:
        residual_out = torch.empty_like(residual)

    if _dtypes_supported(
        residual, post_flat, comb_res_mix, fn, yp, rp, residual_out
    ) and is_supported(
        workspace, num_tokens, hidden_size, hc_mult, input.dtype, input.device
    ):
        workspace._require_handles_attached()
        module = _get_module(allow_build=False)
        assert module is not None
        # The kernel takes dense row-major tensors. Strided inputs (e.g. post / comb mix
        # sliced from one packed mix tensor) are copied; contiguous ones pass through.
        # Strided preallocated outputs get a dense buffer and receive a copy of the result.
        # Both copies are plain stream work and are valid during CUDA-graph capture.
        residual_in = residual.contiguous()
        fn_in = fn.contiguous()
        yp_dense = _dense_output(yp)
        rp_dense = _dense_output(rp)
        residual_out_dense = _dense_output(
            residual_out,
            avoid=residual_in if num_tokens < IN_PLACE_MIN_TOKENS else None,
        )
        # The kernel loads fn before griddepcontrol.wait, as a weight independent of the
        # preceding kernel. A copied fn is written by the kernel right before the launch, so
        # that launch goes without programmatic dependent launch.
        module.allreduce_mhc_post_prenorm(
            input.contiguous(),
            residual_in,
            post_flat.contiguous(),
            comb_res_mix.contiguous(),
            fn_in,
            yp_dense,
            rp_dense,
            residual_out_dense,
            workspace.mc_ptr,
            workspace.uc_ptr_local,
            workspace.buffer_flags,
            workspace.tp_size,
            workspace.rank,
            launch_with_pdl and fn_in is fn,
        )
        for out, dense in (
            (yp, yp_dense),
            (rp, rp_dense),
            (residual_out, residual_out_dense),
        ):
            if dense is not out:
                out.copy_(dense)
        _stats["dispatch_count"] += 1
        per_m = _stats["dispatch_by_num_tokens"]
        per_m[num_tokens] = per_m.get(num_tokens, 0) + 1
        if not _dispatch_logged:
            _dispatch_logged = True
            logger.info(
                "allreduce_mhc_post_prenorm: fused MNNVL all-reduce + mHC post-map / "
                "pre-norm kernel dispatched (world_size=%d, num_tokens=%d)",
                workspace.tp_size,
                num_tokens,
            )
        return yp, rp, residual_out

    _stats["fallback_count"] += 1
    # Each rank decides on its own: a rank whose module failed to load takes this path while the
    # others launch the fused one-shot kernel. Inside the fused kernel's surface this path
    # all-reduces with the stock one-shot kernel too, so every rank runs the same protocol on the
    # workspace whatever it chose; outside it every rank is here and the stock choice applies.
    if _shape_supported(
        workspace, num_tokens, hidden_size, hc_mult, input.dtype, input.device
    ):
        strategy = MNNVLAllreduceFusionStrategy.ONESHOT
    else:
        strategy = MNNVLAllreduceFusionStrategy.AUTO
    ref_yp, ref_rp, ref_res = _composable(
        input,
        residual,
        post_flat,
        comb_res_mix,
        fn,
        workspace,
        launch_with_pdl,
        strategy,
    )
    yp.copy_(ref_yp)
    rp.copy_(ref_rp)
    residual_out.copy_(ref_res)
    return yp, rp, residual_out


def stats() -> Dict[str, Any]:
    """Counters of this process: module state, fused dispatches and composable fallbacks."""
    snapshot = dict(_stats)
    snapshot["dispatch_by_num_tokens"] = dict(_stats["dispatch_by_num_tokens"])
    return snapshot
