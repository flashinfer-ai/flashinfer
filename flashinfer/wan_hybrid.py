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
import math
from typing import Callable, NamedTuple, Optional

import torch

from .utils import register_custom_op, register_fake_op


_WAN_HYBRID_SHAPE = (1, 4800, 40, 128)
_WAN_HYBRID_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_WAN_HYBRID_PADDED_SEQUENCE = 5120
_WAN_HYBRID_VALUE_ROWS = 40 * 128
_WAN_HYBRID_PACKED_VALUE_SHAPE = (
    _WAN_HYBRID_VALUE_ROWS,
    _WAN_HYBRID_PADDED_SEQUENCE // 2,
)
_WAN_HYBRID_SCALE_PLANE_SHAPE = (25_600, 32)
_WAN_HYBRID_TENSOR_MAP_COUNT = 8
_WAN_HYBRID_TENSOR_MAP_BYTES = 128
_WAN_HYBRID_UNAVAILABLE_MESSAGE = (
    "wan_hybrid attention is not available in this FlashInfer installation"
)


class _WanHybridAttentionABIViews(NamedTuple):
    """Allocation-free views prepared for the optional attention binding."""

    vt: torch.Tensor
    sfq: torch.Tensor
    sfk: torch.Tensor
    sfvt_lo: torch.Tensor
    sfvt_hi: torch.Tensor
    qk_correction: torch.Tensor


class WanHybridAttentionWorkspace:
    r"""Reusable caller-owned storage for :func:`wan_hybrid_attention`.

    Construct one workspace on the CUDA device used by ``q``, ``k``, ``v``,
    and ``out``, then retain it across calls. The workspace is opaque: its
    storage belongs to the selected implementation and callers must not rely
    on its internal tensors.

    Parameters
    ----------
    device : torch.device or str
        CUDA device on which the attention call will run.
    """

    def __init__(self, device: torch.device | str) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError("WanHybridAttentionWorkspace requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required to create a WanHybridAttentionWorkspace"
            )
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device

        self._v_levels = torch.empty(
            (2, *_WAN_HYBRID_PACKED_VALUE_SHAPE),
            dtype=torch.uint8,
            device=normalized_device,
        )
        self._v_scale_lo_levels = torch.empty(
            (2, *_WAN_HYBRID_SCALE_PLANE_SHAPE),
            dtype=torch.uint8,
            device=normalized_device,
        )
        self._v_scale_hi_levels = torch.empty(
            (2, *_WAN_HYBRID_SCALE_PLANE_SHAPE),
            dtype=torch.uint8,
            device=normalized_device,
        )
        self._qk_correction = torch.zeros(
            (1,), dtype=torch.float32, device=normalized_device
        )
        self._descriptor_storage = torch.empty(
            (_WAN_HYBRID_TENSOR_MAP_COUNT, _WAN_HYBRID_TENSOR_MAP_BYTES),
            dtype=torch.uint8,
            device=normalized_device,
        )
        self._descriptor_signature: Optional[tuple[int, ...]] = None
        self._buffers = {
            "v_base": self._v_levels[0],
            "v_residual": self._v_levels[1],
            "v_scale_base_lo": self._v_scale_lo_levels[0],
            "v_scale_base_hi": self._v_scale_hi_levels[0],
            "v_scale_residual_lo": self._v_scale_lo_levels[1],
            "v_scale_residual_hi": self._v_scale_hi_levels[1],
        }
        self._attention_views = _WanHybridAttentionABIViews(
            vt=self._v_levels.view(
                2 * _WAN_HYBRID_VALUE_ROWS,
                _WAN_HYBRID_PADDED_SEQUENCE // 2,
            ),
            sfq=self._buffers["v_scale_base_lo"],
            sfk=self._buffers["v_scale_base_hi"],
            sfvt_lo=self._v_scale_lo_levels.view(
                2 * _WAN_HYBRID_SCALE_PLANE_SHAPE[0],
                _WAN_HYBRID_SCALE_PLANE_SHAPE[1],
            ),
            sfvt_hi=self._v_scale_hi_levels.view(
                2 * _WAN_HYBRID_SCALE_PLANE_SHAPE[0],
                _WAN_HYBRID_SCALE_PLANE_SHAPE[1],
            ),
            qk_correction=self._qk_correction,
        )

    @property
    def _attention_abi_views(self) -> _WanHybridAttentionABIViews:
        """Return stable views without allocating or materializing tensors."""

        return self._attention_views


@functools.cache
def _get_wan_hybrid_quantization_module(target: str):
    from .jit.wan_hybrid import gen_wan_hybrid_quantization_module

    return gen_wan_hybrid_quantization_module(target).build_and_load()


def _wan_hybrid_quantization_target(device: torch.device | str | int) -> str:
    from .jit.wan_hybrid import _wan_hybrid_target

    target, _ = _wan_hybrid_target(device)
    return target


@register_custom_op(
    "flashinfer::wan_hybrid_quantize_value",
    mutates_args=(
        "base",
        "residual",
        "base_scale_lo",
        "base_scale_hi",
        "residual_scale_lo",
        "residual_scale_hi",
    ),
)
def _wan_hybrid_quantize_value_impl(
    value: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    base_scale_lo: torch.Tensor,
    base_scale_hi: torch.Tensor,
    residual_scale_lo: torch.Tensor,
    residual_scale_hi: torch.Tensor,
) -> None:
    target = _wan_hybrid_quantization_target(value.device)
    module = _get_wan_hybrid_quantization_module(target)
    module.wan_hybrid_quantize_value(
        value,
        base,
        residual,
        base_scale_lo,
        base_scale_hi,
        residual_scale_lo,
        residual_scale_hi,
    )


@register_fake_op("flashinfer::wan_hybrid_quantize_value")
def _fake_wan_hybrid_quantize_value_impl(
    value: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    base_scale_lo: torch.Tensor,
    base_scale_hi: torch.Tensor,
    residual_scale_lo: torch.Tensor,
    residual_scale_hi: torch.Tensor,
) -> None:
    pass


_WanHybridAttentionImpl = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        WanHybridAttentionWorkspace,
        float,
    ],
    None,
]
# The implementation must enqueue asynchronously on the current PyTorch CUDA
# stream for the input device. The public wrapper does not synchronize.

_wan_hybrid_attention_impl: Optional[_WanHybridAttentionImpl] = None
try:
    from ._wan_hybrid import wan_hybrid_attention_impl as _loaded_wan_hybrid_impl
except ModuleNotFoundError as error:
    if error.name != f"{__package__}._wan_hybrid":
        raise
else:
    _wan_hybrid_attention_impl = _loaded_wan_hybrid_impl


def _normalize_cuda_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def is_wan_hybrid_attention_available(
    device: torch.device | str | int | None = None,
) -> bool:
    r"""Return whether the explicit ``wan_hybrid`` implementation is usable.

    This probe fails closed. It returns ``False`` when the implementation is
    not present, CUDA is unavailable, ``device`` is not a CUDA device, or the
    device is not one of the implementation's supported architectures. When
    ``device`` is omitted, the result only reports implementation availability.

    Parameters
    ----------
    device : torch.device, str, int, or None, optional
        CUDA device to query. An integer is interpreted as a CUDA device index.

    Returns
    -------
    bool
        Whether ``wan_hybrid_attention`` can be selected explicitly.
    """

    if _wan_hybrid_attention_impl is None:
        return False
    if device is None:
        return True
    try:
        normalized_device = _normalize_cuda_device(device)
    except (TypeError, ValueError, RuntimeError):
        return False
    if normalized_device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability(normalized_device)
    except (ValueError, RuntimeError):
        return False
    return capability in _WAN_HYBRID_SUPPORTED_COMPUTE_CAPABILITIES


def _validate_tensor_metadata(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != _WAN_HYBRID_SHAPE:
        raise ValueError(
            f"{name} must have NHD shape {_WAN_HYBRID_SHAPE}, "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.bfloat16:
        raise ValueError(
            f"{name} must have dtype torch.bfloat16, got {tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")


def _validate_wan_hybrid_attention_contract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    workspace: WanHybridAttentionWorkspace,
    *,
    sm_scale: Optional[float],
    qkv_layout: str,
    causal: bool,
) -> float:
    if qkv_layout != "NHD":
        raise ValueError(f"qkv_layout must be 'NHD', got {qkv_layout!r}")
    if causal:
        raise ValueError("wan_hybrid_attention only supports noncausal attention")

    if sm_scale is None:
        normalized_sm_scale = _WAN_HYBRID_SHAPE[-1] ** -0.5
    else:
        try:
            normalized_sm_scale = float(sm_scale)
        except (TypeError, ValueError) as error:
            raise TypeError("sm_scale must be a finite real number") from error
        if not math.isfinite(normalized_sm_scale):
            raise ValueError("sm_scale must be finite")

    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        _validate_tensor_metadata(name, tensor)

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if out is tensor:
            raise ValueError(f"caller-owned out must not alias {name}")

    if not isinstance(workspace, WanHybridAttentionWorkspace):
        raise TypeError("workspace must be a WanHybridAttentionWorkspace")

    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        if not tensor.is_cuda:
            raise ValueError(
                f"{name} must be a CUDA tensor, got device={tensor.device}"
            )
        if tensor.device != q.device:
            raise ValueError(
                f"{name} must be on the same device as q, "
                f"got {tensor.device} and {q.device}"
            )
    if workspace.device != q.device:
        raise ValueError(
            "workspace must be on the same device as q, "
            f"got {workspace.device} and {q.device}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if out.data_ptr() == tensor.data_ptr():
            raise ValueError(f"caller-owned out must not alias {name}")
    return normalized_sm_scale


def _quantize_wan_hybrid_value(
    value: torch.Tensor,
    workspace: WanHybridAttentionWorkspace,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Populate the reusable two-level value representation in ``workspace``."""

    _validate_tensor_metadata("v", value)
    if not value.is_cuda:
        raise ValueError(f"v must be a CUDA tensor, got device={value.device}")
    if not isinstance(workspace, WanHybridAttentionWorkspace):
        raise TypeError("workspace must be a WanHybridAttentionWorkspace")
    if workspace.device != value.device:
        raise ValueError(
            "workspace must be on the same device as v, "
            f"got {workspace.device} and {value.device}"
        )
    if (
        torch.cuda.get_device_capability(value.device)
        not in _WAN_HYBRID_SUPPORTED_COMPUTE_CAPABILITIES
    ):
        major, minor = torch.cuda.get_device_capability(value.device)
        raise NotImplementedError(
            "wan_hybrid value quantization does not support compute capability "
            f"{major}.{minor}"
        )

    base = workspace._buffers["v_base"]
    residual = workspace._buffers["v_residual"]
    base_scale_lo = workspace._buffers["v_scale_base_lo"]
    base_scale_hi = workspace._buffers["v_scale_base_hi"]
    residual_scale_lo = workspace._buffers["v_scale_residual_lo"]
    residual_scale_hi = workspace._buffers["v_scale_residual_hi"]
    _wan_hybrid_quantize_value_impl(
        value,
        base,
        residual,
        base_scale_lo,
        base_scale_hi,
        residual_scale_lo,
        residual_scale_hi,
    )
    return (
        base,
        residual,
        base_scale_lo,
        base_scale_hi,
        residual_scale_lo,
        residual_scale_hi,
    )


def wan_hybrid_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: torch.Tensor,
    workspace: WanHybridAttentionWorkspace,
    sm_scale: Optional[float] = None,
    qkv_layout: str = "NHD",
    causal: bool = False,
) -> torch.Tensor:
    r"""Run the explicit Wan BF16-QK + NVFP4-P/V attention path.

    This API is intentionally limited to the serving contract
    ``B=1, S=4800, H=40, D=128`` in contiguous ``NHD`` layout. ``q``, ``k``,
    and ``v`` are raw post-RoPE BF16 tensors. The implementation retains Q and
    K in BF16 and quantizes the probability/value path internally. It writes
    the result into the required caller-owned BF16 ``out`` tensor and returns
    that same tensor.

    The function is explicit-only and is not selected by an automatic
    attention dispatcher. Call :func:`is_wan_hybrid_attention_available`
    before selecting it. Both component kernels enqueue asynchronously on the
    caller's current PyTorch CUDA stream; this function does not synchronize.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Contiguous post-RoPE BF16 tensors with shape ``(1, 4800, 40, 128)``.
    out : torch.Tensor
        Caller-owned contiguous BF16 output tensor with the same shape and
        device as ``q``.
    workspace : WanHybridAttentionWorkspace
        Reusable workspace on the same CUDA device as the input tensors.
    sm_scale : float, optional
        Scale applied to QK scores before softmax. Defaults to ``1 / sqrt(128)``.
    qkv_layout : str, optional
        Must be ``"NHD"``.
    causal : bool, optional
        Must be ``False``.

    Returns
    -------
    torch.Tensor
        The same tensor passed as ``out``.

    Raises
    ------
    NotImplementedError
        If the optional implementation is unavailable for this installation.
    """

    normalized_sm_scale = _validate_wan_hybrid_attention_contract(
        q,
        k,
        v,
        out,
        workspace,
        sm_scale=sm_scale,
        qkv_layout=qkv_layout,
        causal=causal,
    )
    if _wan_hybrid_attention_impl is None:
        raise NotImplementedError(_WAN_HYBRID_UNAVAILABLE_MESSAGE)
    if not is_wan_hybrid_attention_available(q.device):
        major, minor = torch.cuda.get_device_capability(q.device)
        raise NotImplementedError(
            f"wan_hybrid attention does not support compute capability {major}.{minor}"
        )

    _quantize_wan_hybrid_value(v, workspace)
    _wan_hybrid_attention_impl(q, k, v, out, workspace, normalized_sm_scale)
    return out
