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

# Implementation of flashinfer.mhc.mhc_fused_post_pre: the boundary between two mHC
# blocks -- the post map of block i followed by the pre map (projection, RMS scaling,
# gates, Sinkhorn) and the input RMSNorm of block i + 1 -- in one launch. The
# specialized SM103 kernels (csrc/) serve SUPPORTED_SURFACE; every other call runs the
# composable PyTorch implementation (or, with ``fallback=False``, is declined so the
# caller keeps its own path).

import functools
import math
from typing import Optional

import torch

from ...jit.core import logger
from ...mhc import _check_mhc_post_inputs
from .jit import gen_mhc_fused_post_pre_module
from .reference import mhc_fused_post_pre_reference

# Surface served by the specialized kernels. num_tokens is a runtime kernel argument:
# one compiled module covers the whole range. Operand dtypes (bf16 x / residual /
# norm_weight, fp32 mixes / fn / hc_scale / hc_base), shapes and layouts are checked by
# the module itself on every call.
SUPPORTED_SURFACE: dict = {
    "archs": ("sm_103",),
    "hidden_size": 4096,
    "hc_mult": 4,
    "mix_width": 24,
    "constants": {
        "rms_eps": 1e-5,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "norm_eps": 1e-5,
    },
    "num_tokens_range": (33, 16384),
}
_CONSTANTS_KEY = tuple(SUPPORTED_SURFACE["constants"].values())

_DISPATCH_MESSAGE = "mhc_fused_post_pre: specialized SM103 kernel dispatched"
# Status codes of the module's mhc_fused_post_pre_run (0 = launched).
_STATUS = {
    1: "dtype",
    2: "shape",
    3: "layout",
    4: "alignment",
    5: "aliasing",
    6: "device",
    7: "not_prepared",
    8: "no_kernel_for_shape",
}
VARIANTS = (
    "decode_narrow",
    "decode_wide",
    "stream",
    "stream_long_regs",
    "short_stream",
    "crossover",
    "long_stream",
)

_stats: dict = {
    "specialized_dispatches": 0,
    "fallback_calls": 0,
    "declined_calls": 0,
    "capture_unready": 0,
    "warmup_calls": 0,
    "guard_rejects": {},
    "module_loaded": False,
    "precompile_errors": [],
}
# module: loaded tvm-ffi module or None; build_failed: a build error was recorded;
# devices: device index -> cluster limits once prepared; failed_devices: prepare errors.
_state: dict = {
    "module": None,
    "build_failed": False,
    "devices": {},
    "failed_devices": set(),
    "dispatch_logged": False,
}


@functools.cache
def _device_arch(device_index: int) -> str:
    # Keyed per device: one process may drive several GPUs.
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm_{major}{minor}"


def _reject_reason(
    residual: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> Optional[str]:
    """Support checks that need no tensor data (semantics first, device last).

    Operand dtypes, shapes, strides, alignment, aliasing and devices are checked by the
    module itself on every call (it declines instead of launching)."""
    surface = SUPPORTED_SURFACE
    if norm_weight is None:
        return "norm_weight"
    if (
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
    ) != _CONSTANTS_KEY:
        return "constants"
    shape = residual.shape
    if (
        len(shape) != 3
        or shape[1] != surface["hc_mult"]
        or shape[2] != surface["hidden_size"]
    ):
        return "shape"
    lo, hi = surface["num_tokens_range"]
    if not lo <= shape[0] <= hi:
        return "num_tokens"
    device = residual.device
    if device.type != "cuda":
        return "device"
    if _device_arch(device.index) not in surface["archs"]:
        return "arch"
    return None


def _load_module(device_index: int):
    """The loaded module, prepared for ``device_index``; None while unavailable.

    Builds / loads the module and prepares the device (kernel attributes, cluster
    capacity) on the first call outside CUDA graph capture; never under capture."""
    state = _state
    module = state["module"]
    if module is not None and device_index in state["devices"]:
        return module
    stats = _stats
    if state["build_failed"] or device_index in state["failed_devices"]:
        return None
    if torch.cuda.is_current_stream_capturing():
        stats["capture_unready"] += 1
        return None
    if module is None:
        try:
            module = gen_mhc_fused_post_pre_module().build_and_load()
        except Exception as exc:  # noqa: BLE001 - any build failure falls back
            state["build_failed"] = True
            message = str(exc)[:4000]
            stats["precompile_errors"].append(message)
            logger.warning(
                "mhc_fused_post_pre: specialized kernel build failed, using the "
                "fallback path: " + message
            )
            return None
        state["module"] = module
        stats["module_loaded"] = True
    try:
        module.mhc_fused_post_pre_prepare(device_index)
        limits = [int(v) for v in module.mhc_fused_post_pre_limits(device_index)]
    except Exception as exc:  # noqa: BLE001 - a device that cannot run it falls back
        state["failed_devices"].add(device_index)
        message = f"cuda:{device_index}: " + str(exc)[:4000]
        stats["precompile_errors"].append(message)
        logger.warning(
            "mhc_fused_post_pre: specialized kernel unavailable, using the "
            "fallback path: " + message
        )
        return None
    state["devices"][device_index] = dict(
        zip(
            ("decode", "stream", "short_stream", "crossover", "long_stream"),
            limits[1:],
            strict=True,
        )
    )
    return module


def _view_as(t: torch.Tensor, shape: tuple) -> torch.Tensor:
    # Skips the view (about 1 us of host time each) when the shape already matches:
    # eager prefill steps are host-launch bound.
    return t if t.shape == shape else t.view(shape)


def _flatten_args(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: Optional[tuple],
    allocate: bool = True,
):
    hc, hidden_size, outer_shape = _check_mhc_post_inputs(
        x, residual, post_layer_mix, comb_res_mix
    )
    if residual.ndim == 3 and x.ndim == 2 and post_layer_mix.ndim == 2:
        m = residual.shape[0]
        x2, res3, pm2, comb3 = x, residual, post_layer_mix, comb_res_mix
    else:
        m = math.prod(outer_shape)
        x2 = x.reshape(m, hidden_size)
        res3 = residual.reshape(m, hc, hidden_size)
        pm2 = post_layer_mix.reshape(m, hc)
        comb3 = comb_res_mix.reshape(m, hc, hc)
    if out is not None:
        if len(out) != 4:
            raise ValueError(
                "out must be (residual_cur, post_mix_cur, comb_mix_cur, "
                "layer_input_cur)"
            )
        outs = (
            _view_as(out[0], (m, hc, hidden_size)),
            _view_as(out[1], (m, hc)),
            _view_as(out[2], (m, hc, hc)),
            _view_as(out[3], (m, hidden_size)),
        )
    elif allocate:
        device = residual.device
        outs = (
            torch.empty((m, hc, hidden_size), dtype=residual.dtype, device=device),
            torch.empty((m, hc), dtype=torch.float32, device=device),
            torch.empty((m, hc, hc), dtype=torch.float32, device=device),
            torch.empty((m, hidden_size), dtype=residual.dtype, device=device),
        )
    else:
        outs = None
    return outer_shape, hc, hidden_size, x2, res3, pm2, comb3, outs


def _try_specialized(
    x2,
    res3,
    pm2,
    comb3,
    fn,
    hc_scale,
    hc_base,
    norm_weight,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    norm_eps,
    outs,
    dry_run: bool = False,
) -> Optional[str]:
    """Launch the specialized kernel; return None when it ran (or would run), else
    the reason it did not."""
    reason = _reject_reason(
        res3,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
    )
    if reason is not None:
        return reason
    module = _load_module(res3.device.index)
    if module is None:
        return "not_ready"
    if outs is None:
        # Support query without caller outputs: the outputs the op would allocate always
        # pass the native output checks, so check the inputs only and allocate nothing.
        if not dry_run:
            raise ValueError("a launch needs output tensors")
        status = module.mhc_fused_post_pre_check(
            x2, res3, pm2, comb3, fn, hc_scale, hc_base, norm_weight
        )
    else:
        status = module.mhc_fused_post_pre_run(
            x2,
            res3,
            pm2,
            comb3,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            *outs,
            dry_run,
        )
    if status:
        return _STATUS.get(int(status), f"status_{int(status)}")
    return None


def warmup(
    device: Optional[torch.device] = None,
    hidden_size: int = 4096,
    hc_mult: int = 4,
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-5,
) -> bool:
    """See :func:`flashinfer.mhc.mhc_fused_post_pre_warmup`."""
    _stats["warmup_calls"] += 1
    surface = SUPPORTED_SURFACE
    if (
        hidden_size != surface["hidden_size"]
        or hc_mult != surface["hc_mult"]
        or (
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
        )
        != _CONSTANTS_KEY
    ):
        return False
    if not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    if _device_arch(index) not in surface["archs"]:
        return False
    return _load_module(index) is not None


def supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-5,
    out: Optional[tuple] = None,
) -> bool:
    """See :func:`flashinfer.mhc.mhc_fused_post_pre_supported`."""
    try:
        # allocate=False: without caller outputs only the inputs are checked, so a support
        # query never allocates output-sized buffers (also not under CUDA graph capture).
        _, _, _, x2, res3, pm2, comb3, outs = _flatten_args(
            x, residual, post_layer_mix, comb_res_mix, out, allocate=False
        )
    except (ValueError, RuntimeError):
        return False
    return (
        _try_specialized(
            x2,
            res3,
            pm2,
            comb3,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            outs,
            dry_run=True,
        )
        is None
    )


def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float = 1e-5,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-5,
    out: Optional[tuple] = None,
    fallback: bool = True,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """See :func:`flashinfer.mhc.mhc_fused_post_pre`."""
    outer_shape, hc, hidden_size, x2, res3, pm2, comb3, outs = _flatten_args(
        x, residual, post_layer_mix, comb_res_mix, out
    )
    stats = _stats
    reason = _try_specialized(
        x2,
        res3,
        pm2,
        comb3,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        outs,
    )
    if reason is None:
        stats["specialized_dispatches"] += 1
        state = _state
        if not state["dispatch_logged"]:
            state["dispatch_logged"] = True
            logger.info(
                f"{_DISPATCH_MESSAGE} ({_device_arch(res3.device.index)}, "
                f"H={hidden_size}, HC={hc}, first num_tokens={res3.shape[0]})"
            )
    else:
        rejects = stats["guard_rejects"]
        rejects[reason] = rejects.get(reason, 0) + 1
        if not fallback:
            stats["declined_calls"] += 1
            return None
        stats["fallback_calls"] += 1
        if norm_weight is None:
            raise ValueError("mhc_fused_post_pre requires norm_weight")
        for dst, src in zip(
            outs,
            mhc_fused_post_pre_reference(
                x2,
                res3,
                pm2,
                comb3,
                fn,
                hc_scale,
                hc_base,
                norm_weight,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                norm_eps=norm_eps,
            ),
            strict=True,
        ):
            dst.copy_(src.view(dst.shape))
    residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = outs
    return (
        _view_as(residual_cur, outer_shape + (hc, hidden_size)),
        _view_as(post_mix_cur, outer_shape + (hc, 1)),
        _view_as(comb_mix_cur, outer_shape + (hc, hc)),
        _view_as(layer_input_cur, outer_shape + (hidden_size,)),
    )


def stats() -> dict:
    """Dispatch / build counters of the specialized kernels.

    ``specialized_dispatches`` / ``fallback_calls`` / ``declined_calls`` count calls by
    the path that served them, ``guard_rejects`` the reason each non-specialized call
    was not served. Once the module is loaded, ``dispatch_by_variant`` counts launches
    per kernel variant, ``tensor_map_cache`` the TMA descriptor cache hits and misses,
    and ``variants_by_num_tokens`` maps each prepared device to the token-count span
    every kernel variant serves on it.
    """
    report = dict(_stats)
    report["guard_rejects"] = dict(report["guard_rejects"])
    report["precompile_errors"] = list(report["precompile_errors"])
    state = _state
    report["compiled_variants"] = len(VARIANTS)
    report["devices"] = {k: dict(v) for k, v in state["devices"].items()}
    report["supported_surface"] = dict(SUPPORTED_SURFACE)
    module = state["module"]
    if module is not None:
        counters = [int(v) for v in module.mhc_fused_post_pre_counters()]
        report["tensor_map_cache"] = {"hits": counters[0], "misses": counters[1]}
        report["dispatch_by_variant"] = dict(zip(VARIANTS, counters[2:], strict=True))
        # Kernel variants the supported token counts reach on each prepared device
        # (one compiled module covers them all: num_tokens is a runtime argument).
        lo, hi = SUPPORTED_SURFACE["num_tokens_range"]
        reached = {}
        for index, limits in state["devices"].items():
            variants: dict = {}
            for m in range(lo, hi + 1):
                v = int(module.mhc_fused_post_pre_plan(m, *limits.values())[0])
                name = VARIANTS[v] if v >= 0 else "no_kernel_for_shape"
                span = variants.setdefault(name, [m, m])
                span[1] = m
            reached[index] = variants
        report["variants_by_num_tokens"] = reached
        report["distinct_kernels_for_surface"] = max(
            (
                len([n for n in v if n != "no_kernel_for_shape"])
                for v in reached.values()
            ),
            default=0,
        )
    return report


__all__ = ["SUPPORTED_SURFACE", "VARIANTS", "run", "stats", "supported", "warmup"]
