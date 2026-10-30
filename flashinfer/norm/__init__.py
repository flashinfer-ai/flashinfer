"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

FlashInfer Normalization Kernels
================================

This package provides high-performance normalization kernels:

- RMSNorm: Root Mean Square Normalization
- LayerNorm: Layer Normalization
- Fused Add + RMSNorm: Combined residual add and RMSNorm
- Quantized variants with FP8/FP4 output
"""

import functools
import os
import warnings
from typing import Literal, Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.norm import (
    fused_add_rmsnorm_quant_trace,
    fused_add_rmsnorm_trace,
    fused_rmsnorm_silu_trace,
    gated_residual_next_rmsnorm_trace_dispatch,
    gated_residual_rmsnorm_trace,
    gemma_fused_add_rmsnorm_trace,
    gemma_rmsnorm_trace,
    layernorm_quant_trace,
    layernorm_trace,
    rmsnorm_quant_trace,
    rmsnorm_sigmoid_gate_trace,
    rmsnorm_trace,
)
from ..utils import (
    backend_requirement,
    device_support_pdl,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

# Always import gen_norm_module for JIT warmup and CUDA fallback
from ..jit.norm import gen_norm_module

# Use CUDA JIT implementation instead of CuTe DSL (for debugging/fallback)
# Also fallback to CUDA JIT if nvidia-cutlass-dsl is not installed
_USE_CUDA_NORM = os.environ.get("FLASHINFER_USE_CUDA_NORM", "0") == "1"

if not _USE_CUDA_NORM:
    try:
        from .kernels import (
            rmsnorm_cute,
            qk_rmsnorm_cute,
            rmsnorm_quant_cute,
            fused_add_rmsnorm_cute,
            fused_add_rmsnorm_quant_cute,
            layernorm_cute,
            layernorm_quant_cute,
        )
    except (ImportError, AttributeError):
        # nvidia-cutlass-dsl not installed or incompatible version
        _USE_CUDA_NORM = True

try:
    from .kernels.gated_residual_rmsnorm import (
        gated_residual_rmsnorm as _gated_residual_rmsnorm_cute,
    )
    from .kernels.gated_residual_next_rmsnorm import (
        gated_residual_next_rmsnorm as _gated_residual_next_rmsnorm_cute,
    )
    from .kernels.rmsnorm_sigmoid_gate import (
        rmsnorm_sigmoid_gate as _rmsnorm_sigmoid_gate_cute,
    )
    from .kernels.rmsnorm_sigmoid_gate import (
        supports_rmsnorm_sigmoid_gate as _cute_supports_rmsnorm_sigmoid_gate,
    )

    _HAS_CUTE_DSL_FUSED_NORM = True
except (ImportError, AttributeError):
    _HAS_CUTE_DSL_FUSED_NORM = False

    def _cute_supports_rmsnorm_sigmoid_gate(
        dtype: torch.dtype,
        head_dim: int,
        device: torch.device | str | None = None,
    ) -> bool:
        return False


try:
    from ..cake_muse_spark_norm_backend import (
        cake_gated_residual_next_rmsnorm as _cake_gated_residual_next_rmsnorm,
    )
    from ..cake_muse_spark_norm_backend import (
        cake_gated_residual_rmsnorm as _cake_gated_residual_rmsnorm,
    )
    from ..cake_muse_spark_norm_backend import (
        cake_rmsnorm_sigmoid_gate as _cake_rmsnorm_sigmoid_gate,
    )
    from ..cake_muse_spark_norm_backend import (
        is_cake_gated_residual_next_rmsnorm_supported as _cake_gated_residual_next_rmsnorm_supported,
    )
    from ..cake_muse_spark_norm_backend import (
        is_cake_gated_residual_rmsnorm_supported as _cake_gated_residual_rmsnorm_supported,
    )
    from ..cake_muse_spark_norm_backend import (
        is_cake_rmsnorm_sigmoid_gate_supported as _cake_rmsnorm_sigmoid_gate_supported,
    )

    _HAS_CAKE_FUSED_NORM = True
except (ImportError, AttributeError):
    _HAS_CAKE_FUSED_NORM = False


@functools.cache
def _cute_dsl_supports_arch(major: int, minor: int) -> bool:
    """Whether the installed CuTe DSL can target this compute capability."""
    try:
        from ..cute_dsl.availability import is_cute_dsl_arch_supported

        return is_cute_dsl_arch_supported(major, minor)
    except Exception:
        # Never let the capability probe itself break norm dispatch.
        return True


def _use_cuda_norm(device: torch.device) -> bool:
    """Return ``True`` when the CUDA JIT norm kernels should be used instead
    of the CuTe DSL ones.

    Besides the explicit ``FLASHINFER_USE_CUDA_NORM`` opt-in, this covers
    devices whose architecture the installed CuTe DSL cannot target (e.g.
    Rubin/sm_107 against a DSL without ``sm_107a``, unless
    ``CUTE_DSL_ARCH=sm_100f`` was exported before the process started).
    The CUDA JIT kernels are functionally equivalent, so norm falls back
    instead of raising like the DSL-only kernels do."""
    if _USE_CUDA_NORM:
        return True
    return not _cute_dsl_supports_arch(*torch.cuda.get_device_capability(device))


@functools.cache
def get_norm_module():
    """Get or compile the CUDA JIT norm module.

    Defined unconditionally because fused DIT kernels only have a CUDA JIT
    implementation and no CuTe DSL alternative.
    """
    return gen_norm_module().build_and_load()


def _normalize_scale_tensor(
    scale: Union[float, torch.Tensor], ref_tensor: torch.Tensor
) -> torch.Tensor:
    """Normalize quantization scale to 1D tensor of shape (1,) on target device."""
    if not isinstance(scale, torch.Tensor):
        warnings.warn(
            "Passing scale as a float is deprecated and will be removed in a future "
            "release. Use a torch.Tensor of shape (1,) instead.",
            FutureWarning,
            stacklevel=3,
        )
        scale = torch.tensor([scale], dtype=torch.float32, device=ref_tensor.device)
    if scale.device != ref_tensor.device:
        scale = scale.to(ref_tensor.device)
    if scale.dtype != torch.float32:
        scale = scale.to(torch.float32)
    if scale.ndim == 0:
        scale = scale.view(1)
    elif scale.ndim == 1 and scale.numel() == 1:
        pass
    else:
        raise ValueError(
            f"scale must be a scalar tensor or shape (1,), got shape {tuple(scale.shape)}"
        )
    return scale.contiguous()


_GATED_RESIDUAL_HIDDEN_SIZE = 8192
_GATED_RESIDUAL_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
_RMSNORM_SIGMOID_GATE_HEAD_DIM = 64
_FUSED_NORM_ALIGNMENT = 16
_DUAL_NORM_ALIGNMENT = 32
#: The Cake package reaches rows and vectors in accesses of up to 256 bits: it serves storage aligned to 32 bytes
#: alone (``gated_residual_rmsnorm``'s tensors; ``rmsnorm_sigmoid_gate``'s tensors and, in bytes, their token
#: strides).  Under ``"auto"``, storage the 16-byte contract admits but the package does not runs the CuTe DSL kernel.
_CAKE_NORM_ALIGNMENT = 32


def _fused_norm_arch_supported(device: torch.device) -> bool:
    cap = torch.cuda.get_device_capability(device)
    return cap in _GATED_RESIDUAL_COMPUTE_CAPABILITIES and _cute_dsl_supports_arch(*cap)


def _is_aligned(tensor: torch.Tensor, alignment: int = _FUSED_NORM_ALIGNMENT) -> bool:
    return tensor.data_ptr() % alignment == 0


def _cake_norm_layout(*tensors: Optional[torch.Tensor]) -> bool:
    """Whether every tensor given (``None`` skipped) sits in storage the Cake package serves: 32-byte aligned."""
    return all(
        tensor is None or _is_aligned(tensor, _CAKE_NORM_ALIGNMENT)
        for tensor in tensors
    )


def _cake_gate_layout(input: torch.Tensor, gate: torch.Tensor) -> bool:
    """Whether a ``rmsnorm_sigmoid_gate`` call's tensors sit in storage the Cake package serves: 32-byte aligned
    storage and token strides that are 32-byte multiples (16 elements)."""
    return (
        _cake_norm_layout(input, gate)
        and (input.stride(0) * input.element_size()) % _CAKE_NORM_ALIGNMENT == 0
        and (gate.stride(0) * gate.element_size()) % _CAKE_NORM_ALIGNMENT == 0
    )


_GATED_RESIDUAL_NEXT_BACKENDS = ("auto", "cake", "cute")


def _cute_gated_residual_next_rmsnorm_available(device: torch.device) -> bool:
    return _HAS_CUTE_DSL_FUSED_NORM and _fused_norm_arch_supported(device)


def _cake_gated_residual_next_rmsnorm_available(
    device: torch.device,
    *,
    tensor_alpha: bool,
    has_post_weight: bool,
    rows: Optional[int] = None,
) -> bool:
    return _HAS_CAKE_FUSED_NORM and _cake_gated_residual_next_rmsnorm_supported(
        device, tensor_alpha=tensor_alpha, has_post_weight=has_post_weight, rows=rows
    )


def _gated_residual_next_rmsnorm_backend(
    backend: str,
    device: torch.device,
    *,
    tensor_alpha: bool,
    has_post_weight: bool,
    rows: Optional[int] = None,
) -> Optional[str]:
    """The backend one call runs on: ``"cake"`` when the Cake package ships a
    program for the device, the call's variant and (with ``rows``) its row count
    (preferred under ``"auto"``; a row count in a gap of the package's windows
    falls to the CuTe DSL kernel), else ``"cute"`` when the CuTe DSL kernel
    targets the device; ``None`` when the requested backend is not available."""
    if backend not in _GATED_RESIDUAL_NEXT_BACKENDS:
        raise ValueError(
            f"backend must be one of {_GATED_RESIDUAL_NEXT_BACKENDS}, got {backend!r}"
        )
    if backend != "cute" and _cake_gated_residual_next_rmsnorm_available(
        device,
        tensor_alpha=tensor_alpha,
        has_post_weight=has_post_weight,
        rows=None if backend == "cake" else rows,
    ):
        return "cake"
    if backend != "cake" and _cute_gated_residual_next_rmsnorm_available(device):
        return "cute"
    return None


def _resolved_pdl(device: torch.device, enable_pdl: bool) -> bool:
    """The programmatic dependent launch flag a call runs with: the caller's, where the device supports it."""
    return bool(enable_pdl) and device_support_pdl(device)


def _cute_gated_residual_rmsnorm_available(device: torch.device) -> bool:
    return _HAS_CUTE_DSL_FUSED_NORM and _fused_norm_arch_supported(device)


def _cake_gated_residual_rmsnorm_available(
    device: torch.device,
    *,
    weight_before_norm: bool,
    enable_pdl: bool,
    rows: Optional[int] = None,
) -> bool:
    return _HAS_CAKE_FUSED_NORM and _cake_gated_residual_rmsnorm_supported(
        device, weight_before_norm=weight_before_norm, enable_pdl=enable_pdl, rows=rows
    )


def _gated_residual_rmsnorm_backend(
    backend: str,
    device: torch.device,
    *,
    weight_before_norm: bool,
    enable_pdl: bool,
    rows: Optional[int] = None,
    cake_layout: bool = True,
) -> Optional[str]:
    """The backend one call runs on: ``"cake"`` when the Cake package ships a
    program for the device, the call's variant (``weight`` applied before the
    norm or absent, programmatic dependent launch or not) and (with ``rows``)
    its row count (preferred under ``"auto"``; a row count in a gap of the
    package's windows falls to the CuTe DSL kernel) and the call's tensors sit
    in storage the package serves (``cake_layout``: 32-byte aligned; other
    storage falls to the CuTe DSL kernel too), else ``"cute"`` when the CuTe DSL
    kernel targets the device; ``None`` when the requested backend is not
    available."""
    if backend not in _GATED_RESIDUAL_NEXT_BACKENDS:
        raise ValueError(
            f"backend must be one of {_GATED_RESIDUAL_NEXT_BACKENDS}, got {backend!r}"
        )
    if (
        backend != "cute"
        and cake_layout
        and _cake_gated_residual_rmsnorm_available(
            device,
            weight_before_norm=weight_before_norm,
            enable_pdl=enable_pdl,
            rows=None if backend == "cake" else rows,
        )
    ):
        return "cake"
    if backend != "cake" and _cute_gated_residual_rmsnorm_available(device):
        return "cute"
    return None


def _cake_rmsnorm_sigmoid_gate_available(
    device: torch.device,
    *,
    heads: Optional[int] = None,
    tokens: Optional[int] = None,
    enable_pdl: bool = False,
) -> bool:
    return _HAS_CAKE_FUSED_NORM and _cake_rmsnorm_sigmoid_gate_supported(
        device, heads=heads, tokens=tokens, enable_pdl=enable_pdl
    )


def _rmsnorm_sigmoid_gate_backend(
    backend: str,
    device: torch.device,
    *,
    heads: Optional[int] = None,
    tokens: Optional[int] = None,
    enable_pdl: bool = False,
    cake_layout: bool = True,
) -> Optional[str]:
    """The backend one call runs on: ``"cake"`` when the Cake package ships a
    program for the device, the call's launch form (its PDL route under
    ``enable_pdl``) and (with ``heads`` and ``tokens``) the call's head count
    and token count (preferred under ``"auto"``; a head count the package's
    windows do not name, or a token count in a gap of them, falls to the CuTe
    DSL kernel) and the call's tensors sit in storage the package serves
    (``cake_layout``: 32-byte aligned rows and token strides; other storage
    falls to the CuTe DSL kernel too),
    else ``"cute"`` when the CuTe DSL kernel targets the device; ``None`` when
    the requested backend is not available."""
    if backend not in _GATED_RESIDUAL_NEXT_BACKENDS:
        raise ValueError(
            f"backend must be one of {_GATED_RESIDUAL_NEXT_BACKENDS}, got {backend!r}"
        )
    if (
        backend != "cute"
        and cake_layout
        and _cake_rmsnorm_sigmoid_gate_available(
            device,
            heads=None if backend == "cake" else heads,
            tokens=None if backend == "cake" else tokens,
            enable_pdl=enable_pdl,
        )
    ):
        return "cake"
    if backend != "cake" and _cute_supports_rmsnorm_sigmoid_gate(
        torch.bfloat16, _RMSNORM_SIGMOID_GATE_HEAD_DIM, device=device
    ):
        return "cute"
    return None


def supports_rmsnorm_sigmoid_gate(
    dtype: torch.dtype,
    head_dim: int,
    device: torch.device | str | None = None,
    *,
    backend: Literal["auto", "cake", "cute"] = "auto",
    enable_pdl: bool = True,
) -> bool:
    """Whether a BF16, head-dimension-64 call can run on ``device`` (the current
    one when ``None``) through ``backend`` (``"auto"``: the Cake package or the
    CuTe DSL kernel) in the launch form ``enable_pdl`` selects.  The package
    routes on the head count and the token count of a call as well and serves
    32-byte aligned storage alone; ``rmsnorm_sigmoid_gate`` falls to the CuTe
    DSL kernel where they lie outside its windows or its alignment."""
    if dtype != torch.bfloat16 or head_dim != _RMSNORM_SIGMOID_GATE_HEAD_DIM:
        return False
    if device is None:
        if not torch.cuda.is_available():
            return False
        resolved = torch.device("cuda", torch.cuda.current_device())
    else:
        resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return False
    return (
        _rmsnorm_sigmoid_gate_backend(
            backend, resolved, enable_pdl=_resolved_pdl(resolved, enable_pdl)
        )
        is not None
    )


def supports_gated_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    enable_pdl: bool = True,
    backend: Literal["auto", "cake", "cute"] = "auto",
) -> bool:
    """Return whether tensors satisfy the SM100/SM103/SM107 fused-kernel contract
    and ``backend`` (``"auto"``: the Cake package or the CuTe DSL kernel) can run
    the call on their device.  ``"auto"`` answers for the call's row count as
    well: a row count in a gap of the Cake package's windows runs on the CuTe DSL
    kernel, so the answer is false there only where that kernel is unavailable.
    ``"cake"`` answers whether the package ships a program for the device, the
    variant and the tensors' storage; an explicit cake call still raises for a
    row count in a gap (``is_cake_gated_residual_rmsnorm_supported`` of
    :mod:`flashinfer.cake_muse_spark_norm_backend` answers for the row count)."""
    return (
        _gated_residual_rmsnorm_tensors_ok(branch, residual, weight, alpha, beta)
        and _gated_residual_rmsnorm_backend(
            backend,
            branch.device,
            weight_before_norm=weight is not None,
            enable_pdl=_resolved_pdl(branch.device, enable_pdl),
            rows=branch.shape[0] or None,
            cake_layout=_cake_norm_layout(branch, residual, weight, alpha, beta),
        )
        is not None
    )


def _gated_residual_rmsnorm_tensors_ok(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> bool:
    """The tensor contract of ``gated_residual_rmsnorm`` (shapes, dtypes, one CUDA
    device, contiguous, aligned storage), without the backend decision; one flat
    check, since every call pays for it."""
    shape = branch.shape
    if len(shape) != 2 or shape[1] != _GATED_RESIDUAL_HIDDEN_SIZE or not branch.is_cuda:
        return False
    device = branch.device
    vector = (_GATED_RESIDUAL_HIDDEN_SIZE,)
    alignment = _FUSED_NORM_ALIGNMENT
    return (
        residual.shape == shape
        and alpha.shape == vector
        and beta.shape == vector
        and branch.dtype == torch.bfloat16
        and residual.dtype == torch.float32
        and alpha.dtype == torch.float32
        and beta.dtype == torch.float32
        and residual.device == device
        and alpha.device == device
        and beta.device == device
        and branch.is_contiguous()
        and residual.is_contiguous()
        and alpha.is_contiguous()
        and beta.is_contiguous()
        and (
            branch.data_ptr() | residual.data_ptr() | alpha.data_ptr() | beta.data_ptr()
        )
        % alignment
        == 0
        and (
            weight is None
            or (
                weight.shape == vector
                and weight.dtype == torch.bfloat16
                and weight.device == device
                and weight.is_contiguous()
                and weight.data_ptr() % alignment == 0
            )
        )
    )


def supports_gated_residual_next_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    next_weight: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: torch.Tensor,
    *,
    post_weight: Optional[torch.Tensor] = None,
    backend: Literal["auto", "cake", "cute"] = "auto",
) -> bool:
    """Return whether inputs satisfy the SM100/SM103/SM107 two-norm kernel contract
    and ``backend`` (``"auto"``: the Cake package or the CuTe DSL kernel) can run
    the call on their device.  ``"auto"`` answers for the call's row count as
    well: a row count in a gap of the Cake package's windows runs on the CuTe DSL
    kernel, so the answer is false there only where that kernel is unavailable.
    ``"cake"`` answers whether the package ships a program for the device and the
    variant; an explicit cake call still raises for a row count in a gap
    (``is_cake_gated_residual_next_rmsnorm_supported`` of
    :mod:`flashinfer.cake_muse_spark_norm_backend` answers for the row count)."""
    tensors: tuple[torch.Tensor, ...] = (branch, residual, next_weight, beta)
    if isinstance(alpha, torch.Tensor):
        if (
            alpha.shape != (_GATED_RESIDUAL_HIDDEN_SIZE,)
            or alpha.dtype != torch.float32
        ):
            return False
        tensors += (alpha,)
    elif not isinstance(alpha, (int, float)):
        return False
    if post_weight is not None:
        tensors += (post_weight,)
    return (
        branch.ndim == 2
        and branch.shape[1] == _GATED_RESIDUAL_HIDDEN_SIZE
        and residual.shape == branch.shape
        and next_weight.shape == (_GATED_RESIDUAL_HIDDEN_SIZE,)
        and (
            post_weight is None
            or (
                post_weight.shape == (_GATED_RESIDUAL_HIDDEN_SIZE,)
                and post_weight.dtype == torch.bfloat16
            )
        )
        and beta.shape == (_GATED_RESIDUAL_HIDDEN_SIZE,)
        and branch.dtype == torch.bfloat16
        and residual.dtype == torch.float32
        and next_weight.dtype == torch.bfloat16
        and beta.dtype == torch.float32
        and all(
            tensor.is_cuda
            and tensor.is_contiguous()
            and tensor.device == branch.device
            and _is_aligned(tensor, _DUAL_NORM_ALIGNMENT)
            for tensor in tensors
        )
        and _gated_residual_next_rmsnorm_backend(
            backend,
            branch.device,
            tensor_alpha=isinstance(alpha, torch.Tensor),
            has_post_weight=post_weight is not None,
            rows=branch.shape[0] or None,
        )
        is not None
    )


@flashinfer_api(trace=gated_residual_next_rmsnorm_trace_dispatch)
def gated_residual_next_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    next_weight: torch.Tensor,
    alpha: Union[torch.Tensor, float],
    beta: torch.Tensor,
    post_eps: float = 1e-6,
    next_eps: float = 1e-6,
    next_gain_center: float = 0.0,
    residual_out: Optional[torch.Tensor] = None,
    ffn_input: Optional[torch.Tensor] = None,
    *,
    post_weight: Optional[torch.Tensor] = None,
    post_gain_center: float = 0.0,
    enable_pdl: bool = True,
    backend: Literal["auto", "cake", "cute"] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated residual RMSNorm plus the next RMSNorm in one launch, for hidden size 8192.

    This computes ``residual_out = alpha * residual + beta * RMSNorm(branch)``
    in FP32. It then computes ``ffn_input = RMSNorm(residual_out) *
    (next_weight + next_gain_center)`` and rounds only that output to BF16.
    When ``post_weight`` is supplied, ``branch`` is multiplied by
    ``post_weight + post_gain_center`` in FP32 before its RMSNorm.
    The optimized implementation supports contiguous hidden-size-8192 tensors
    on SM100, SM103, and SM107 GPUs.

    Parameters
    ----------
    branch : torch.Tensor
        Two-dimensional BF16 branch tensor.
    residual : torch.Tensor
        FP32 residual stream with the same shape as ``branch``.
    next_weight : torch.Tensor
        BF16 weight for the second RMSNorm, with shape ``(8192,)``.
    alpha : Union[torch.Tensor, float]
        FP32 stream gate with shape ``(8192,)``, or a scalar coefficient.
        Use ``1.0`` for an ungated stream. Scalar values specialize the kernel
        without allocating or loading an alpha tensor.
    beta : torch.Tensor
        FP32 branch gate with shape ``(8192,)``.
    post_eps, next_eps : float
        Epsilons for the branch and residual-stream RMSNorms.
    next_gain_center : float
        Constant added to ``next_weight`` in FP32.
    residual_out : Optional[torch.Tensor]
        Optional FP32 destination with the same shape as ``residual``.
    ffn_input : Optional[torch.Tensor]
        Optional BF16 destination with the same shape as ``branch``.
    post_weight : Optional[torch.Tensor]
        Optional BF16 input scale for the first RMSNorm, with shape ``(8192,)``.
    post_gain_center : float
        Constant added to ``post_weight`` in FP32; unused when it is ``None``.
    enable_pdl : bool
        Whether to enable programmatic dependent launch on the CuTe DSL kernel.
        Defaults to ``True``; pass ``False`` to disable it. The Cake kernels
        launch without the attribute and ignore this argument.
    backend : str
        ``"auto"`` (default) runs the Cake kernels
        (:mod:`flashinfer.cake_muse_spark_norm_backend`) where the package ships
        a program for the device, the call's inputs and its row count (the
        package's routes may leave row-count gaps where the CuTe DSL kernel
        measured ahead), else the CuTe DSL kernel; ``"cake"`` and ``"cute"``
        select one and raise ``ValueError`` when it is not available (``"cake"``
        at a row count in a gap raises too). The Cake package ships programs for a tensor ``alpha``
        with ``post_weight`` and for a scalar ``alpha`` with or without it; a
        tensor ``alpha`` without ``post_weight`` runs its ``post_weight`` programs
        with a zero vector and ``post_gain_center = 1.0``, which is exact.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The FP32 residual stream and BF16 normalized FFN input.
    """
    if not supports_gated_residual_next_rmsnorm(
        branch, residual, next_weight, alpha, beta, post_weight=post_weight
    ):
        raise ValueError(
            "tensors do not satisfy the gated_residual_next_rmsnorm contract"
        )
    if (
        _gated_residual_next_rmsnorm_backend(
            backend,
            branch.device,
            tensor_alpha=isinstance(alpha, torch.Tensor),
            has_post_weight=post_weight is not None,
        )
        is None
    ):
        raise ValueError(
            f"gated_residual_next_rmsnorm: backend {backend!r} is not available "
            f"on {branch.device}"
        )
    if residual_out is None:
        residual_out = torch.empty_like(residual)
    if ffn_input is None:
        ffn_input = torch.empty_like(branch)
    outputs = (
        ("residual_out", residual_out, torch.float32),
        ("ffn_input", ffn_input, torch.bfloat16),
    )
    for name, output, dtype in outputs:
        if (
            output.shape != branch.shape
            or output.dtype != dtype
            or output.device != branch.device
            or not output.is_contiguous()
            or not _is_aligned(output, _DUAL_NORM_ALIGNMENT)
        ):
            raise ValueError(
                f"{name} must match branch shape, expected dtype, device, and layout"
            )
    # Scalar specializations never read the alpha tensor argument; reuse beta
    # to preserve the tensor ABI without allocating a constant gate vector.
    if isinstance(alpha, torch.Tensor):
        alpha_tensor, alpha_scalar = alpha, None
    else:
        alpha_tensor, alpha_scalar = beta, float(alpha)
    _gated_residual_next_rmsnorm_impl(
        branch,
        residual,
        next_weight,
        alpha_tensor,
        beta,
        residual_out,
        ffn_input,
        post_eps,
        next_eps,
        next_gain_center,
        post_weight,
        post_gain_center,
        alpha_scalar,
        enable_pdl,
        backend,
    )
    return residual_out, ffn_input


@register_custom_op(
    "flashinfer::gated_residual_next_rmsnorm",
    mutates_args=("residual_out", "ffn_input"),
)
def _gated_residual_next_rmsnorm_impl(
    branch: torch.Tensor,
    residual: torch.Tensor,
    next_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual_out: torch.Tensor,
    ffn_input: torch.Tensor,
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    post_weight: Optional[torch.Tensor],
    post_gain_center: float,
    alpha_scalar: Optional[float] = None,
    enable_pdl: bool = True,
    backend: str = "auto",
) -> None:
    chosen = _gated_residual_next_rmsnorm_backend(
        backend,
        branch.device,
        tensor_alpha=alpha_scalar is None,
        has_post_weight=post_weight is not None,
        rows=branch.shape[0],
    )
    if chosen == "cake":
        # The Cake programs launch without programmatic dependent launch; a scalar
        # gate travels as alpha_scalar and the alpha slot (beta) is unread.
        _cake_gated_residual_next_rmsnorm(
            branch,
            residual,
            next_weight,
            alpha,
            beta,
            residual_out,
            ffn_input,
            post_eps=post_eps,
            next_eps=next_eps,
            next_gain_center=next_gain_center,
            post_weight=post_weight,
            post_gain_center=post_gain_center,
            alpha_scalar=alpha_scalar,
        )
        return
    if chosen is None:
        raise ValueError(
            f"gated_residual_next_rmsnorm: backend {backend!r} is not available "
            f"on {branch.device}"
        )
    if enable_pdl:
        enable_pdl = device_support_pdl(branch.device)
    _gated_residual_next_rmsnorm_cute(
        branch,
        residual,
        alpha,
        beta,
        next_weight,
        post_eps,
        next_eps,
        next_gain_center,
        residual_out,
        ffn_input,
        post_weight,
        post_gain_center,
        enable_pdl,
        alpha_scalar,
    )


@register_fake_op("flashinfer::gated_residual_next_rmsnorm")
def _gated_residual_next_rmsnorm_fake(
    branch: torch.Tensor,
    residual: torch.Tensor,
    next_weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    residual_out: torch.Tensor,
    ffn_input: torch.Tensor,
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    post_weight: Optional[torch.Tensor],
    post_gain_center: float,
    alpha_scalar: Optional[float] = None,
    enable_pdl: bool = True,
    backend: str = "auto",
) -> None:
    return None


@flashinfer_api(trace=gated_residual_rmsnorm_trace)
def gated_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
    gain_center: float = 0.0,
    out: Optional[torch.Tensor] = None,
    *,
    enable_pdl: bool = True,
    backend: Literal["auto", "cake", "cute"] = "auto",
) -> torch.Tensor:
    r"""Gated residual RMSNorm for hidden size 8192: blend a normalized branch into the stream.

    The operation is ``alpha * residual + beta * RMSNorm(branch)``. When
    ``weight`` is supplied, ``weight + gain_center`` is applied to ``branch``
    before its RMS reduction. When it is ``None``, the branch is unweighted.

    This optimized implementation currently supports contiguous BF16 tensors
    with hidden size 8192 on SM100, SM103, and SM107 GPUs.

    Parameters
    ----------
    branch, residual : torch.Tensor
        Two-dimensional BF16 and FP32 tensors, respectively, with shape
        ``(rows, 8192)``.
    weight : Optional[torch.Tensor]
        Optional BF16 pre-normalization weight with shape ``(8192,)``.
    alpha, beta : torch.Tensor
        FP32 gate tensors with shape ``(8192,)``.
    eps : float
        Epsilon added to the RMS denominator.
    gain_center : float
        Constant added to ``weight`` in FP32.
    out : Optional[torch.Tensor]
        Optional FP32 output tensor with the same shape and layout as ``residual``.
    enable_pdl : bool
        Whether to enable programmatic dependent launch. Defaults to ``True``;
        pass ``False`` to disable it. The Cake package ships a plain and a
        programmatic-dependent-launch build of each program and runs the one
        the flag selects.
    backend : str
        ``"auto"`` (default) runs the Cake kernels
        (:mod:`flashinfer.cake_muse_spark_norm_backend`) where the package ships
        a program for the device, the call's inputs (``weight`` present or not,
        ``enable_pdl``) and its row count (the package's routes may leave
        row-count gaps where the CuTe DSL kernel measured ahead) and the tensors
        sit in 32-byte aligned storage (the package's programs read and write
        256-bit vectors; storage aligned to 16 bytes only runs the CuTe DSL
        kernel), else the CuTe DSL kernel; ``"cake"`` and ``"cute"`` select one
        and raise ``ValueError`` when it is not available (``"cake"`` at a row
        count in a gap, or on storage aligned to 16 bytes only, raises too).

    Returns
    -------
    torch.Tensor
        The fused result.
    """
    if not _gated_residual_rmsnorm_tensors_ok(branch, residual, weight, alpha, beta):
        raise ValueError("tensors do not satisfy the gated_residual_rmsnorm contract")
    device = branch.device
    weight_before_norm = weight is not None
    if out is not None and (
        out.shape != branch.shape
        or out.dtype != residual.dtype
        or out.device != device
        or not out.is_contiguous()
        or not _is_aligned(out)
    ):
        raise ValueError("out must match branch shape, dtype, device, and layout")
    # One backend decision per call, handed to the implementation as the resolved backend; the Cake package
    # serves 32-byte aligned storage alone (its programs read and write 256-bit vectors).
    chosen = _gated_residual_rmsnorm_backend(
        backend,
        device,
        weight_before_norm=weight_before_norm,
        enable_pdl=_resolved_pdl(device, enable_pdl),
        rows=branch.shape[0],
        cake_layout=_cake_norm_layout(branch, residual, weight, alpha, beta, out),
    )
    if chosen is None:
        raise ValueError(
            f"gated_residual_rmsnorm: backend {backend!r} is not available on {device}"
        )
    if out is None:
        out = torch.empty_like(residual)
    if weight is not None:
        kernel_weight = weight
    elif chosen == "cute":
        kernel_weight = branch.reshape(-1)[:_GATED_RESIDUAL_HIDDEN_SIZE]
    else:
        # The Cake package binds the unread weight slot of its mode-0 programs itself.
        kernel_weight = branch
    _gated_residual_rmsnorm_impl(
        branch,
        residual,
        kernel_weight,
        alpha,
        beta,
        out,
        eps,
        gain_center,
        weight_before_norm,
        enable_pdl,
        chosen,
    )
    return out


@register_custom_op("flashinfer::gated_residual_rmsnorm", mutates_args=("out",))
def _gated_residual_rmsnorm_impl(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gain_center: float,
    apply_weight_before_norm: bool,
    enable_pdl: bool = True,
    backend: str = "auto",
) -> None:
    if enable_pdl:
        enable_pdl = device_support_pdl(branch.device)
    if backend == "cake" or backend == "cute":
        chosen = backend  # resolved by the public wrapper (or requested explicitly)
    else:
        chosen = _gated_residual_rmsnorm_backend(
            backend,
            branch.device,
            weight_before_norm=apply_weight_before_norm,
            enable_pdl=enable_pdl,
            rows=branch.shape[0],
            cake_layout=_cake_norm_layout(branch, residual, weight, alpha, beta, out),
        )
    if chosen == "cake":
        # The package binds the unread weight slot of its mode-0 programs itself.
        _cake_gated_residual_rmsnorm(
            branch,
            residual,
            weight if apply_weight_before_norm else None,
            alpha,
            beta,
            out,
            eps=eps,
            gain_center=gain_center,
            enable_pdl=enable_pdl,
        )
        return
    if chosen is None:
        raise ValueError(
            f"gated_residual_rmsnorm: backend {backend!r} is not available on {branch.device}"
        )
    _gated_residual_rmsnorm_cute(
        branch,
        residual,
        weight,
        alpha,
        beta,
        eps,
        gain_center,
        int(apply_weight_before_norm),
        out,
        enable_pdl,
    )


@register_fake_op("flashinfer::gated_residual_rmsnorm")
def _gated_residual_rmsnorm_fake(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gain_center: float,
    apply_weight_before_norm: bool,
    enable_pdl: bool = True,
    backend: str = "auto",
) -> None:
    return None


@flashinfer_api(trace=rmsnorm_sigmoid_gate_trace)
def rmsnorm_sigmoid_gate(
    input: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-5,
    *,
    enable_pdl: bool = True,
    backend: Literal["auto", "cake", "cute"] = "auto",
) -> None:
    r"""Per-head RMSNorm and sigmoid gate, in place, for head dim 64.

    The last dimension is normalized independently, multiplied by
    ``sigmoid(gate)``, and written back to ``input``. The optimized kernel
    supports contiguous BF16 head rows of width 64 on SM100, SM103, and SM107 GPUs.

    Parameters
    ----------
    input : torch.Tensor
        BF16 tensor of shape ``(rows, 64)`` or ``(rows, heads, 64)``, normalized
        in place. The last dimension must be contiguous, heads must be packed,
        and the row stride must be a multiple of 8 elements.
    gate : torch.Tensor
        BF16 gate tensor with the same shape and layout constraints as ``input``.
    eps : float
        Epsilon added to the RMS denominator.
    enable_pdl : bool
        Whether to launch with programmatic dependent launch. Defaults to
        ``True``; pass ``False`` to disable it. The CuTe DSL kernel takes the
        attribute directly; the Cake package serves the call from its PDL
        route, whose windows name a PDL build, a late-signal PDL build or a
        plain build (launched without the attribute where the PDL build
        measured slower) or leave a token count to the CuTe DSL kernel under
        PDL (a declared gap).  The package ships that route on SM100, SM103
        and SM107.
    backend : str
        ``"auto"`` (default) runs the Cake kernels
        (:mod:`flashinfer.cake_muse_spark_norm_backend`) where the package ships
        a program for the device, the call's launch form, head count and token
        count (the package routes 128, 32 and 16 heads; its windows leave
        token-count gaps where the CuTe DSL kernel measured ahead) and the
        tensors sit in 32-byte aligned storage with token strides that are
        multiples of 16 elements (the package's 16-head programs on SM107 read
        and write 256-bit vectors; other storage runs the CuTe DSL kernel),
        else the CuTe DSL kernel; ``"cake"`` and ``"cute"`` select one and raise
        ``ValueError`` when it is not available (``"cake"`` at a head count or
        token count outside the package's windows, or on storage the package
        does not serve, raises too).
    """
    if input.ndim not in (2, 3) or gate.shape != input.shape:
        raise ValueError("input and gate must have matching 2-D or 3-D shapes")
    device = input.device
    if (
        input.dtype != torch.bfloat16
        or input.shape[-1] != _RMSNORM_SIGMOID_GATE_HEAD_DIM
        or device.type != "cuda"
    ):
        raise ValueError("tensors do not satisfy the rmsnorm_sigmoid_gate contract")
    # One backend decision per call, handed to the implementation as the resolved backend; the implementation resolves
    # the launch form against the device once (the package ships its PDL route on PDL-capable architectures alone, so
    # the decision reads the requested form).
    enable_pdl = bool(enable_pdl)
    chosen = _rmsnorm_sigmoid_gate_backend(
        backend,
        device,
        heads=input.shape[1] if input.ndim == 3 else 1,
        tokens=input.shape[0],
        enable_pdl=enable_pdl,
        cake_layout=_cake_gate_layout(input, gate),
    )
    if chosen is None:
        raise ValueError(
            f"rmsnorm_sigmoid_gate: backend {backend!r} is not available on {device}"
        )
    if (
        gate.dtype != torch.bfloat16
        or not gate.is_cuda
        or gate.device != input.device
        or input.stride(-1) != 1
        or gate.stride(-1) != 1
        or input.stride(0) % 8 != 0
        or gate.stride(0) % 8 != 0
        or (input.ndim == 3 and input.stride(1) != input.shape[-1])
        or (gate.ndim == 3 and gate.stride(1) != gate.shape[-1])
        or not _is_aligned(input)
        or not _is_aligned(gate)
    ):
        raise ValueError("tensors do not satisfy the rmsnorm_sigmoid_gate contract")
    _rmsnorm_sigmoid_gate_impl(input, gate, eps, enable_pdl, chosen)


@register_custom_op("flashinfer::rmsnorm_sigmoid_gate", mutates_args=("input",))
def _rmsnorm_sigmoid_gate_impl(
    input: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    enable_pdl: bool = True,
    backend: str = "auto",
) -> None:
    if enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if backend == "cake" or backend == "cute":
        chosen = backend  # resolved by the public wrapper (or requested explicitly)
    else:
        chosen = _rmsnorm_sigmoid_gate_backend(
            backend,
            input.device,
            heads=input.shape[1] if input.ndim == 3 else 1,
            tokens=input.shape[0],
            enable_pdl=enable_pdl,
            cake_layout=_cake_gate_layout(input, gate),
        )
    if chosen == "cake":
        # The package's PDL route decides per window whether the program launches with the attribute.
        _cake_rmsnorm_sigmoid_gate(input, gate, eps=eps, enable_pdl=enable_pdl)
        return
    if chosen is None:
        raise ValueError(
            f"rmsnorm_sigmoid_gate: backend {backend!r} is not available on {input.device}"
        )
    _rmsnorm_sigmoid_gate_cute(input, gate, eps, enable_pdl=enable_pdl)


@register_fake_op("flashinfer::rmsnorm_sigmoid_gate")
def _rmsnorm_sigmoid_gate_fake(
    input: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    enable_pdl: bool = True,
    backend: str = "auto",
) -> None:
    return None


@flashinfer_api(trace=rmsnorm_trace)
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor in place.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm_impl(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
def _rmsnorm_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if _use_cuda_norm(input.device):
        get_norm_module().rmsnorm(out, input, weight, eps, enable_pdl)
    else:
        if input.dim() == 3:
            qk_rmsnorm_cute(
                input, weight, out, eps, weight_bias=0.0, enable_pdl=enable_pdl
            )
        else:
            rmsnorm_cute(
                input, weight, out, eps, weight_bias=0.0, enable_pdl=enable_pdl
            )


@register_fake_op("flashinfer::rmsnorm")
def _rmsnorm_impl_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api(trace=rmsnorm_quant_trace)
@register_custom_op("flashinfer::rmsnorm_quant", mutates_args=("out",))
def rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: Union[float, torch.Tensor],
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Root mean square normalization + fp8 quantization.

    ``out[i] = ((input[i] / RMS(input)) * weight[i]).to(fp8)``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, will quantize the output to the dtype of this tensor,
        which must be float8_e4m3fn or float8_e5m2.
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    scale: torch.Tensor
        Scale factor for quantization, shape (1,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>`_

    """
    scale = _normalize_scale_tensor(scale, input)
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if _use_cuda_norm(input.device):
        get_norm_module().rmsnorm_quant(out, input, weight, scale, eps, enable_pdl)
    else:
        rmsnorm_quant_cute(
            out, input, weight, scale, eps, weight_bias=0.0, enable_pdl=enable_pdl
        )


@register_fake_op("flashinfer::rmsnorm_quant")
def _rmsnorm_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api(trace=fused_add_rmsnorm_trace)
@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"))
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>`_
    """
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if _use_cuda_norm(input.device):
        get_norm_module().fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
    else:
        fused_add_rmsnorm_cute(
            input, residual, weight, eps, weight_bias=0.0, enable_pdl=enable_pdl
        )


@register_fake_op("flashinfer::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api(trace=fused_add_rmsnorm_quant_trace)
@register_custom_op(
    "flashinfer::fused_add_rmsnorm_quant", mutates_args=("out", "residual")
)
def fused_add_rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: Union[float, torch.Tensor],
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization + fp8 quantization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``out[i] = (((residual[i] / RMS(residual)) * weight[i]) / scale).to(fp8)``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, will quantize the output to the dtype of this tensor,
        which must be float8_e4m3fn or float8_e5m2.
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    scale: torch.Tensor
        Scale factor for quantization, shape (1,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>`_
    """
    scale = _normalize_scale_tensor(scale, input)
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if _use_cuda_norm(input.device):
        get_norm_module().fused_add_rmsnorm_quant(
            out, input, residual, weight, scale, eps, enable_pdl
        )
    else:
        fused_add_rmsnorm_quant_cute(
            out,
            input,
            residual,
            weight,
            scale,
            eps,
            weight_bias=0.0,
            enable_pdl=enable_pdl,
        )


@register_fake_op("flashinfer::fused_add_rmsnorm_quant")
def _fused_add_rmsnorm_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
@register_custom_op(
    "flashinfer::fused_add_rmsnorm_fp8_block_quant",
    mutates_args=("out", "block_scale", "normed_out", "residual"),
)
def fused_add_rmsnorm_fp8_block_quant(
    out: torch.Tensor,
    block_scale: torch.Tensor,
    normed_out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add-residual + RMSNorm + 1x128 fp8 block quantization, in one pass.

    Replaces the two-kernel producer (:func:`fused_add_rmsnorm` then a per-token-group 1x128 fp8
    quant) used before fp8 block-scaled GEMMs. Unlike :func:`fused_add_rmsnorm_quant` (scalar
    per-tensor scale) this emits a **dynamic per-1x128-block** fp32 scale in the column-major
    TMA-aligned layout that :mod:`flashinfer.deep_gemm` consumes.

    Step 1: ``residual = input + residual`` (pre-norm, written in place).
    Step 2: ``normed = RMSNorm(residual) * weight`` (bf16, written to ``normed_out``).
    Step 3: ``out, block_scale = quant_1x128_fp8(normed)`` (fp8 from the bf16-rounded normed, so
    ``out`` matches re-quantizing ``normed_out``).

    Parameters
    ----------
    out : torch.Tensor
        fp8 output, shape ``(batch_size, hidden_size)``, dtype float8_e4m3fn (the activation
        dtype the fp8 block-scaled GEMMs consume).
    block_scale : torch.Tensor
        fp32 block scales, shape ``(hidden_size // 128, round_up(batch_size, 4))``, contiguous.
        This is the column-major (MN-major, TMA-aligned) buffer deep_gemm expects: the logical
        ``(batch_size, hidden_size // 128)`` scale is ``block_scale.transpose(0, 1)[:batch_size]``.
    normed_out : torch.Tensor
        bf16/fp16 normed output, shape ``(batch_size, hidden_size)`` (for consumers that need the
        pre-quant activation, e.g. an MoE router).
    input, residual : torch.Tensor
        Shape ``(batch_size, hidden_size)``. ``residual`` is updated in place with ``input+residual``.
    weight : torch.Tensor
        RMSNorm weight, shape ``(hidden_size,)``. ``hidden_size`` must be a multiple of 128
        (and of 256 for hidden_size<=8192, 512 for 8192<hidden_size<=16384).
    eps : float
        Epsilon for numerical stability.
    enable_pdl : bool
        Whether to enable programmatic dependent launch.
    """
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().fused_add_rmsnorm_fp8_block_quant(
        out, block_scale, normed_out, input, residual, weight, eps, enable_pdl
    )


@register_fake_op("flashinfer::fused_add_rmsnorm_fp8_block_quant")
def _fused_add_rmsnorm_fp8_block_quant_fake(
    out: torch.Tensor,
    block_scale: torch.Tensor,
    normed_out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api(trace=gemma_rmsnorm_trace)
def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Gemma-style root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor in place.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>`_

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    _gemma_rmsnorm_impl(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::gemma_rmsnorm", mutates_args=("out",))
def _gemma_rmsnorm_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if _use_cuda_norm(input.device):
        get_norm_module().gemma_rmsnorm(out, input, weight, eps, enable_pdl)
    else:
        if input.dim() == 3:
            qk_rmsnorm_cute(
                input, weight, out, eps, weight_bias=1.0, enable_pdl=enable_pdl
            )
        else:
            rmsnorm_cute(
                input, weight, out, eps, weight_bias=1.0, enable_pdl=enable_pdl
            )


@register_fake_op("flashinfer::gemma_rmsnorm")
def _gemma_rmsnorm_impl_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api(trace=gemma_fused_add_rmsnorm_trace)
@register_custom_op(
    "flashinfer::gemma_fused_add_rmsnorm", mutates_args=("input", "residual")
)
def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Gemma-style fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html>`_
    """
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if _use_cuda_norm(input.device):
        get_norm_module().gemma_fused_add_rmsnorm(
            input, residual, weight, eps, enable_pdl
        )
    else:
        fused_add_rmsnorm_cute(
            input, residual, weight, eps, weight_bias=1.0, enable_pdl=enable_pdl
        )


@register_fake_op("flashinfer::gemma_fused_add_rmsnorm")
def _gemma_fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api(trace=layernorm_trace)
@register_custom_op("flashinfer::layernorm", mutates_args=())
def layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Layer normalization.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size). Need to be bfloat16.
    gemma: torch.Tensor
        Gemma tensor, shape (hidden_size,). Need to be float32.
    beta: torch.Tensor
        Beta tensor, shape (hidden_size,). Need to be float32.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    output: torch.Tensor
        Layer Normalized tensor, shape (batch_size, hidden_size). Same dtype as input.
    """
    out = torch.empty_like(input)
    if _use_cuda_norm(input.device):
        get_norm_module().layernorm(out, input, gemma, beta, eps)
    else:
        layernorm_cute(out, input, gemma, beta, eps)
    return out


@register_fake_op("flashinfer::layernorm")
def _layernorm_fake(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    b, k = input.shape
    return input.new_empty([b, k])


@flashinfer_api(trace=layernorm_quant_trace)
@register_custom_op("flashinfer::layernorm_quant", mutates_args=("out",))
def layernorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    scale: Union[float, torch.Tensor],
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Layer normalization + fp8 quantization.

    ``out[i] = (((input[i] - E[input]) / sqrt(Var[input] + eps)) * gemma[i] + beta[i]) / scale``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, shape (batch_size, hidden_size). Need to be contiguous.
        The output is quantized to the dtype of this tensor, which must be
        float8_e4m3fn or float8_e5m2.
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size). Need to be bfloat16 and contiguous.
    gemma: torch.Tensor
        Gemma tensor, shape (hidden_size,). Need to be float32.
    beta: torch.Tensor
        Beta tensor, shape (hidden_size,). Need to be float32.
    scale: Union[float, torch.Tensor]
        Scale factor for quantization; a tensor scale must have shape ``(1,)``.
        The normalized output is divided by this scale before the fp8 cast.
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_.
        Only honored by the CuTe-DSL backend; the CUDA fallback ignores it.
    """
    scale = _normalize_scale_tensor(scale, input)
    if enable_pdl is None or enable_pdl:
        enable_pdl = device_support_pdl(input.device)
    if _use_cuda_norm(input.device):
        get_norm_module().layernorm_quant(out, input, gemma, beta, scale, eps)
    else:
        if input.dtype != torch.bfloat16:
            raise RuntimeError("input must be bfloat16")
        if gemma.dtype != torch.float32:
            raise RuntimeError("gamma must be float32")
        if beta.dtype != torch.float32:
            raise RuntimeError("beta must be float32")
        if not input.is_contiguous():
            raise RuntimeError("input must be contiguous")
        if not out.is_contiguous():
            raise RuntimeError("output must be contiguous")
        layernorm_quant_cute(out, input, gemma, beta, scale, eps, enable_pdl=enable_pdl)


@register_fake_op("flashinfer::layernorm_quant")
def _layernorm_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


# CuTe-DSL fused RMSNorm + FP4 Quantization kernels
# These require SM100+ (Blackwell) GPUs and nvidia-cutlass-dsl
try:
    from ..cute_dsl import rmsnorm_fp4quant as rmsnorm_fp4quant
    from ..cute_dsl import add_rmsnorm_fp4quant as add_rmsnorm_fp4quant
except ImportError:
    # nvidia-cutlass-dsl not installed, these functions will not be available
    pass


# ============================================================
# Fused RMSNorm + SiLU kernel (SM100 optimized)
# ============================================================

from ..jit.rmsnorm_silu import (
    gen_rmsnorm_silu_module,
    select_knobs,
    _estimate_ctas_per_row,
)


@functools.cache
def _get_rmsnorm_silu_sm_count(device_id: int):
    """Cache the SM count per device."""
    props = torch.cuda.get_device_properties(device_id)
    return props.multi_processor_count


@functools.cache
def _get_rmsnorm_silu_max_threads_per_sm(device_id: int):
    """Cache the resident-thread limit per SM (2048 on SM100/SM103, 1024 on SM107)."""
    props = torch.cuda.get_device_properties(device_id)
    return props.max_threads_per_multi_processor


@functools.cache
def _get_rmsnorm_silu_module(
    C, output_dtype, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
):
    return gen_rmsnorm_silu_module(
        C, output_dtype, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
    ).build_and_load()


def _compute_rmsnorm_silu_workspace_size(
    rows, cols, output_dtype, warps_m, ctas_per_row, kernel_cfg, occupancy, sm_count
):
    """Compute workspace size matching the engine's layout."""
    # rs
    ws = rows * 4  # sizeof(float)
    ws = ((ws + 127) // 128) * 128
    # fp8_scale
    ws += 4
    ws = ((ws + 127) // 128) * 128
    # cooperative workspace (multi-CTA)
    if ctas_per_row > 1:
        ctas_per_col_max = (rows + warps_m - 1) // warps_m
        if kernel_cfg == 2:
            ctas_per_col = ctas_per_col_max
        else:
            ctas_per_col = min(sm_count * occupancy // ctas_per_row, ctas_per_col_max)
        ctas_per_col = max(ctas_per_col, 1)
        ws += ctas_per_col * warps_m * ctas_per_row * 8 * 2  # sizeof(float2) * 2
        ws = ((ws + 127) // 128) * 128
        ws += 2 * ctas_per_col * 4  # sizeof(int32_t)
        ws = ((ws + 127) // 128) * 128
    ws += 128  # final alignment padding
    return ws


def _torch_dtype_to_str(dtype):
    if dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float8_e4m3fn:
        return "fp8"
    elif hasattr(torch, "float4_e2m1fn_x2") and dtype == torch.float4_e2m1fn_x2:
        return "nvfp4"
    raise ValueError(
        "Unsupported output dtype for fused_rmsnorm_silu: "
        f"{dtype}. Supported dtypes: bfloat16, float8_e4m3fn, float4_e2m1fn_x2"
    )


@flashinfer_api(trace=fused_rmsnorm_silu_trace)
def fused_rmsnorm_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    block_scale: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple]:
    r"""Fused RMSNorm + SiLU activation.

    ``out[i] = SiLU(RMSNorm(input[i], weight, eps))``

    where ``SiLU(x) = x / (1 + exp(-x))``

    Optimized for SM100 (B200) for WAN VAE decoder problem sizes.
    Other shapes and architectures (SM80+) use conservative fallback heuristics.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape ``(num_tokens, hidden_size)``, dtype ``bfloat16``.
    weight: torch.Tensor
        Scale (gamma) tensor, shape ``(hidden_size,)``, dtype ``bfloat16``.
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        Output tensor. If ``None``, allocated as ``bfloat16`` matching input.
        The dtype of ``out`` selects the output format:

        - ``torch.bfloat16``: shape ``(num_tokens, hidden_size)``.
        - ``torch.float8_e4m3fn``: FP8 E4M3 output, shape ``(num_tokens, hidden_size)``.
          Requires SM89+ (Ada/Hopper).
        - ``torch.float4_e2m1fn_x2``: NVFP4 block-scaled output, shape
          ``(num_tokens, hidden_size // 2)``. Requires SM100+ (Blackwell)
          and ``hidden_size`` divisible by 16.
    block_scale: Optional[torch.Tensor]
        Pre-allocated output tensor for per-block scale factors (NVFP4 only).
        Shape ``(num_tokens, hidden_size // 16)``, dtype ``torch.float8_e4m3fn``.
        If ``None``, allocated automatically when ``out`` is NVFP4.
        Ignored for bf16/fp8 output.

    Returns
    -------
    output: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        For bf16/fp8: normalized + SiLU activated tensor,
        shape ``(num_tokens, hidden_size)``.

        For NVFP4: a tuple ``(y_fp4, block_scale)`` following the same
        convention as :func:`rmsnorm_fp4quant`. ``y_fp4`` has shape
        ``(num_tokens, hidden_size // 2)`` with dtype ``float4_e2m1fn_x2``,
        and ``block_scale`` has shape ``(num_tokens, hidden_size // 16)``
        with dtype ``float8_e4m3fn`` (one E4M3 scale per 16-element block).

    Notes
    -----
    Kernel tuning knobs are sweep-optimized on B200 (SM100) for WAN VAE
    decoder problem sizes: ``hidden_size`` in {64, 128, 160, 256, 320, 512,
    640, 1024} and ``num_tokens`` in {1560, 6240, 24960, 99840, 399360}.
    Other problem sizes use conservative fallback heuristics that are
    functionally correct but may not achieve peak throughput. Performance
    on non-SM100 architectures uses the same fallback path.
    """
    if input.device.type != "cuda":
        raise ValueError("fused_rmsnorm_silu requires CUDA tensors")
    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must be torch.bfloat16, got {input.dtype}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be torch.bfloat16, got {weight.dtype}")
    if input.ndim != 2:
        raise ValueError(
            f"input must be 2D [num_tokens, hidden_size], got ndim={input.ndim}"
        )
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1D [hidden_size], got ndim={weight.ndim}")
    if weight.device != input.device:
        raise ValueError("weight must be on the same device as input")

    if out is None:
        out = torch.empty_like(input)
    if out.device != input.device:
        raise ValueError("out must be on the same device as input")

    num_tokens = input.size(0)
    C = input.size(1)
    if weight.size(0) != C:
        raise ValueError(
            f"weight shape mismatch: expected [{C}], got {tuple(weight.shape)}"
        )
    output_dtype_str = _torch_dtype_to_str(out.dtype)

    if output_dtype_str in ("bf16", "fp8"):
        if tuple(out.shape) != tuple(input.shape):
            raise ValueError(
                f"out shape mismatch for {output_dtype_str}: expected {tuple(input.shape)}, got {tuple(out.shape)}"
            )
    elif output_dtype_str == "nvfp4":
        expected_shape = (num_tokens, C // 2)
        if C % 16 != 0:
            raise ValueError(
                f"nvfp4 output requires hidden_size divisible by 16, got C={C}"
            )
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out shape mismatch for nvfp4: expected {expected_shape}, got {tuple(out.shape)}"
            )

    major, minor = get_compute_capability(input.device)
    sm_version = major * 10 + minor
    if sm_version < 80:
        raise RuntimeError("fused_rmsnorm_silu requires SM80+")
    if output_dtype_str == "fp8" and sm_version < 89:
        raise RuntimeError("FP8 output requires SM89+ (Ada/Hopper)")
    if output_dtype_str == "nvfp4" and sm_version < 100:
        raise RuntimeError("NVFP4 output requires SM100+ (Blackwell)")

    knobs = select_knobs(C, num_tokens, output_dtype_str, sm_version)
    if knobs is None:
        raise ValueError(
            f"Unsupported problem size for fused_rmsnorm_silu: "
            f"C={C}, num_tokens={num_tokens}, dtype={output_dtype_str}"
        )

    warps_m, split_cols, kernel_cfg, occupancy, bytes_per_ldg = knobs
    # The LUT is tuned for 2048 threads/SM; cap occupancy on SM107 (1024).
    max_threads_per_sm = _get_rmsnorm_silu_max_threads_per_sm(input.device.index)
    if max_threads_per_sm <= 1024:
        occupancy = max(1, min(occupancy, max_threads_per_sm // (warps_m * 32)))
    ctas_per_row = _estimate_ctas_per_row(C, split_cols, kernel_cfg, bytes_per_ldg)
    sm_count = _get_rmsnorm_silu_sm_count(input.device.index)

    module = _get_rmsnorm_silu_module(
        C, output_dtype_str, warps_m, ctas_per_row, bytes_per_ldg, kernel_cfg, occupancy
    )

    ws_size = _compute_rmsnorm_silu_workspace_size(
        num_tokens,
        C,
        output_dtype_str,
        warps_m,
        ctas_per_row,
        kernel_cfg,
        occupancy,
        sm_count,
    )
    workspace = torch.empty(ws_size, dtype=torch.uint8, device=input.device)

    if output_dtype_str == "nvfp4":
        num_blocks = C // 16
        if block_scale is None:
            block_scale = torch.empty(
                num_tokens, num_blocks, dtype=torch.float8_e4m3fn, device=input.device
            )
        else:
            expected_shape = (num_tokens, num_blocks)
            if tuple(block_scale.shape) != expected_shape:
                raise ValueError(
                    f"block_scale shape mismatch: expected {expected_shape}, "
                    f"got {tuple(block_scale.shape)}"
                )
            if block_scale.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"block_scale must be float8_e4m3fn, got {block_scale.dtype}"
                )
        scale_row_out = block_scale
    else:
        scale_row_out = torch.empty(0, dtype=torch.uint8, device=input.device)

    module.rmsnorm_silu(out, input, weight, eps, workspace, scale_row_out, sm_count)

    if output_dtype_str == "nvfp4":
        return out, block_scale

    return out


####################################################################################################
# Fused DIT LayerNorm for Diffusion Transformer Self-Attention
#
# Three fused modes for WAN 2.2 and similar DIT architectures.
# Primary targets: SM90 (Hopper), SM100/SM103 (Blackwell).
# BF16 output: SM80+. NVFP4/MXFP8 output: SM100+ (Blackwell) only.
# Hidden dim restricted to 3072 (WAN 2.2 5B target).
#
# Gate/scale/shift tensors use stride = 6 * hidden_dim in the row dimension,
# matching WAN's temb.chunk(6, dim=2) pattern.
####################################################################################################

_DIT_LN_MODE_GATE_RESIDUAL_GAMMA_BETA = 0
_DIT_LN_MODE_RESIDUAL_SCALE_SHIFT = 1
_DIT_LN_MODE_GATE_RESIDUAL_SCALE_SHIFT = 2

_DIT_LN_OUTPUT_BF16 = 0
_DIT_LN_OUTPUT_NVFP4 = 1
_DIT_LN_OUTPUT_MXFP8 = 2

_DIT_LN_SUPPORTED_HIDDEN_DIM = 3072


def _dit_ln_resolve_output_format(use_nvfp4: bool, use_mxfp8: bool) -> int:
    if use_nvfp4 and use_mxfp8:
        raise ValueError("Cannot use both NVFP4 and MXFP8 output quantization")
    if use_nvfp4:
        return _DIT_LN_OUTPUT_NVFP4
    if use_mxfp8:
        return _DIT_LN_OUTPUT_MXFP8
    return _DIT_LN_OUTPUT_BF16


def _dit_ln_check_common_inputs(
    input: torch.Tensor,
    residual: Optional[torch.Tensor],
    use_nvfp4: bool,
    use_mxfp8: bool,
) -> None:
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must be bfloat16, got {input.dtype}")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if input.ndim != 3:
        raise ValueError(
            f"input must be 3D [batch, num_rows, hidden_dim], got {input.ndim}D"
        )
    hidden_dim = input.shape[2]
    if hidden_dim != _DIT_LN_SUPPORTED_HIDDEN_DIM:
        raise ValueError(
            f"hidden_dim must be {_DIT_LN_SUPPORTED_HIDDEN_DIM} (WAN 2.2 5B target), "
            f"got {hidden_dim}"
        )
    if input.shape[0] > 2**31 - 1:
        raise ValueError(f"batch_size must fit in int32, got {input.shape[0]}")
    if input.shape[1] > 2**31 - 1:
        raise ValueError(f"num_rows must fit in int32, got {input.shape[1]}")

    if residual is not None:
        if residual.shape != input.shape:
            raise ValueError(
                f"residual shape {tuple(residual.shape)} must match "
                f"input shape {tuple(input.shape)}"
            )
        if residual.dtype != torch.bfloat16:
            raise ValueError(f"residual must be bfloat16, got {residual.dtype}")
        if not residual.is_contiguous():
            raise ValueError("residual must be contiguous")

    major, minor = get_compute_capability(input.device)
    sm = major * 10 + minor
    if sm < 80:
        raise RuntimeError("Fused DIT LayerNorm requires SM80+")
    if (use_nvfp4 or use_mxfp8) and sm < 100:
        raise RuntimeError(
            f"NVFP4/MXFP8 output requires SM100+ (Blackwell), got SM{sm}"
        )


def _dit_ln_check_strided_tensor(
    tensor: torch.Tensor, input: torch.Tensor, name: str
) -> None:
    """Validate gate/scale/shift tensors.

    These tensors come from WAN's ``temb.chunk(6, dim=2)`` pattern and have
    stride ``(seq_len * 6 * hidden_dim, 6 * hidden_dim, 1)`` in the row
    dimension instead of contiguous stride. The kernel handles this via
    ``gate_shift_scale_stride = 6``.

    A contiguous tensor (stride == hidden_dim in the row dim) will cause
    illegal memory access because the kernel reads with stride 6.
    """
    if tensor.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D, got {tensor.ndim}D")
    hidden_dim = input.shape[2]
    if tensor.shape[-1] != hidden_dim:
        raise ValueError(
            f"{name} last dim must match hidden_dim {hidden_dim}, "
            f"got {tensor.shape[-1]}"
        )
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}")

    # Validate stride matches WAN's temb.chunk(6, dim=2) pattern.
    # The kernel hardcodes gate_shift_scale_stride = 6, so the row stride
    # must be 6 * hidden_dim (not hidden_dim as in a contiguous tensor).
    expected_row_stride = 6 * hidden_dim
    if tensor.ndim == 3:
        actual_row_stride = tensor.stride(1)
    else:
        actual_row_stride = tensor.stride(0)

    if tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} must have stride 1 in the last dimension, "
            f"got stride {tensor.stride(-1)}"
        )
    if actual_row_stride != expected_row_stride:
        raise ValueError(
            f"{name} row stride must be {expected_row_stride} "
            f"(6 * hidden_dim, from WAN's temb.chunk(6, dim=2) pattern), "
            f"got {actual_row_stride}. "
            f"Passing a contiguous tensor will cause incorrect results. "
            f"Use temb.chunk(6, dim=2)[i].squeeze(2) to get the correct stride."
        )


def _dit_ln_check_bias_tensor(
    tensor: Optional[torch.Tensor], input: torch.Tensor, name: str
) -> None:
    if tensor is None:
        return
    if tensor.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D, got {tensor.ndim}D")
    if tensor.shape[-1] != input.shape[2]:
        raise ValueError(
            f"{name} last dim must match hidden_dim {input.shape[2]}, "
            f"got {tensor.shape[-1]}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {tensor.dtype}")


def _dit_ln_check_scaling_factor(
    tensor: Optional[torch.Tensor], device: torch.device, name: str
) -> None:
    if tensor is None:
        return
    if tensor.shape != (1,):
        raise ValueError(f"{name} must have shape (1,), got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _dit_ln_prepare_outputs(
    input: torch.Tensor,
    output_format: int,
    global_scaling_factor: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    sf_out: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate or validate pre-allocated output tensors."""
    batch_size, num_rows, hidden_size = input.shape
    device = input.device

    # Validate global_scaling_factor if provided
    if global_scaling_factor is not None:
        if global_scaling_factor.shape != (1,):
            raise ValueError(
                f"global_scaling_factor must have shape (1,), "
                f"got {tuple(global_scaling_factor.shape)}"
            )
        if global_scaling_factor.dtype != torch.float32:
            raise ValueError(
                f"global_scaling_factor must be float32, "
                f"got {global_scaling_factor.dtype}"
            )
        if global_scaling_factor.device != device:
            raise ValueError(
                f"global_scaling_factor must be on {device}, "
                f"got {global_scaling_factor.device}"
            )

    # residual_out: always [batch, num_rows, hidden_size] BF16
    if residual_out is None:
        residual_out = torch.empty_like(input)
    else:
        if residual_out.shape != input.shape:
            raise ValueError(
                f"residual_out shape {tuple(residual_out.shape)} must match "
                f"input shape {tuple(input.shape)}"
            )
        if residual_out.dtype != torch.bfloat16:
            raise ValueError(f"residual_out must be bfloat16, got {residual_out.dtype}")
        if residual_out.device != device:
            raise ValueError(
                f"residual_out must be on {device}, got {residual_out.device}"
            )
        if not residual_out.is_contiguous():
            raise ValueError("residual_out must be contiguous")

    if output_format == _DIT_LN_OUTPUT_NVFP4:
        expected_norm_shape = (batch_size, num_rows, hidden_size // 8)
        expected_norm_dtype = torch.int32
        numMTiles = (num_rows + 127) // 128
        numKTiles = (hidden_size + 63) // 64
        expected_sf_shape = (batch_size, numMTiles, numKTiles, 32, 4, 4)
        sf_scale = (
            global_scaling_factor
            if global_scaling_factor is not None
            else torch.ones(1, dtype=torch.float32, device=device)
        )
    elif output_format == _DIT_LN_OUTPUT_MXFP8:
        expected_norm_shape = (batch_size, num_rows, hidden_size // 4)
        expected_norm_dtype = torch.int32
        numMTiles = (num_rows + 127) // 128
        numKTiles = (hidden_size + 127) // 128
        expected_sf_shape = (batch_size, numMTiles, numKTiles, 32, 4, 4)
        sf_scale = torch.zeros(1, dtype=torch.float32, device=device)
    else:
        expected_norm_shape = (batch_size, num_rows, hidden_size)
        expected_norm_dtype = torch.bfloat16
        expected_sf_shape = None
        sf_scale = torch.empty(0, dtype=torch.float32, device=device)

    # norm_out
    if norm_out is None:
        norm_out = torch.empty(
            *expected_norm_shape, dtype=expected_norm_dtype, device=device
        )
    else:
        if tuple(norm_out.shape) != expected_norm_shape:
            raise ValueError(
                f"norm_out shape {tuple(norm_out.shape)} must be {expected_norm_shape}"
            )
        if norm_out.dtype != expected_norm_dtype:
            raise ValueError(
                f"norm_out dtype must be {expected_norm_dtype}, got {norm_out.dtype}"
            )
        if norm_out.device != device:
            raise ValueError(f"norm_out must be on {device}, got {norm_out.device}")
        if not norm_out.is_contiguous():
            raise ValueError("norm_out must be contiguous")

    # sf_out (only for NVFP4/MXFP8)
    if expected_sf_shape is not None:
        if sf_out is None:
            sf_out = torch.empty(*expected_sf_shape, dtype=torch.uint8, device=device)
        else:
            if tuple(sf_out.shape) != expected_sf_shape:
                raise ValueError(
                    f"sf_out shape {tuple(sf_out.shape)} must be {expected_sf_shape}"
                )
            if sf_out.dtype != torch.uint8:
                raise ValueError(f"sf_out must be uint8, got {sf_out.dtype}")
            if sf_out.device != device:
                raise ValueError(f"sf_out must be on {device}, got {sf_out.device}")
            if not sf_out.is_contiguous():
                raise ValueError("sf_out must be contiguous")
    else:
        sf_out = torch.empty(0, dtype=torch.uint8, device=device)

    return residual_out, norm_out, sf_out, sf_scale


def _dit_ln_empty_bf16(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.bfloat16, device=device)


def _dit_ln_empty_f32(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32, device=device)


@flashinfer_api
def fused_dit_gate_residual_layernorm_gamma_beta(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    *,
    gate_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused gate + residual + LayerNorm(gamma, beta) for DIT self-attention.

    Computes in a single kernel:

    .. code-block:: python

        residual_out = residual + input * (gate + gate_bias)
        norm_out = LayerNorm(residual_out, gamma, beta)

    Parameters
    ----------
    input : torch.Tensor
        Input tensor ``[batch, num_rows, 3072]``, BF16, contiguous.
    residual : torch.Tensor
        Residual tensor, same shape as input, BF16, contiguous.
    gate : torch.Tensor
        Gating tensor ``[batch, num_rows, 3072]``, BF16.
        May be non-contiguous with stride ``6 * hidden_dim`` in dim 1
        (from WAN's ``temb.chunk(6, dim=2)`` pattern).
    gamma : torch.Tensor
        LayerNorm weight ``[3072]``, FP32.
    beta : torch.Tensor
        LayerNorm bias ``[3072]``, FP32.
    gate_bias : Optional[torch.Tensor]
        Bias for gate ``[3072]`` or ``[1, 3072]``, FP32.
    epsilon : float
        LayerNorm epsilon.
    use_nvfp4 : bool
        Quantize norm output to NVFP4 (SM100+ only).
    use_mxfp8 : bool
        Quantize norm output to MXFP8 (SM100+ only).
    global_scaling_factor : Optional[torch.Tensor]
        Global scale for NVFP4 output ``[1]``, FP32. Required when ``use_nvfp4=True``.
    input_global_scaling_factor : Optional[torch.Tensor]
        Scale applied to input before gating (for pre-quantized NVFP4 inputs) ``[1]``, FP32.
    residual_out : Optional[torch.Tensor]
        Pre-allocated residual output ``[batch, num_rows, 3072]``, BF16.
    norm_out : Optional[torch.Tensor]
        Pre-allocated norm output. Shape depends on output format.
    sf_out : Optional[torch.Tensor]
        Pre-allocated scale factor output for NVFP4/MXFP8.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(residual_out, norm_out)``.
        For BF16: both are ``[batch, num_rows, 3072]`` BF16.
        For NVFP4/MXFP8: ``residual_out`` is BF16, ``norm_out`` is packed int32.

    Note
    ----
    This kernel targets WAN 2.2 5B (hidden_dim=3072 only).
    Primary targets: SM90 (Hopper), SM100/SM103 (Blackwell).
    BF16 output compatible with SM80+. NVFP4/MXFP8 requires SM100+.
    """
    output_format = _dit_ln_resolve_output_format(use_nvfp4, use_mxfp8)
    _dit_ln_check_common_inputs(input, residual, use_nvfp4, use_mxfp8)
    _dit_ln_check_strided_tensor(gate, input, "gate")
    _dit_ln_check_bias_tensor(gate_bias, input, "gate_bias")

    if gamma.ndim != 1 or gamma.shape[0] != input.shape[2]:
        raise ValueError(f"gamma must be [hidden_dim], got {tuple(gamma.shape)}")
    if beta.shape != gamma.shape:
        raise ValueError(f"beta shape must match gamma, got {tuple(beta.shape)}")
    if gamma.dtype != torch.float32:
        raise ValueError(f"gamma must be float32, got {gamma.dtype}")
    if beta.dtype != torch.float32:
        raise ValueError(f"beta must be float32, got {beta.dtype}")

    if use_nvfp4 and global_scaling_factor is None:
        raise ValueError("global_scaling_factor required for NVFP4 output")
    _dit_ln_check_scaling_factor(
        global_scaling_factor, input.device, "global_scaling_factor"
    )
    _dit_ln_check_scaling_factor(
        input_global_scaling_factor, input.device, "input_global_scaling_factor"
    )

    device = input.device
    residual_out, norm_out, sf_out, sf_scale = _dit_ln_prepare_outputs(
        input, output_format, global_scaling_factor, residual_out, norm_out, sf_out
    )

    get_norm_module().fused_dit_layernorm(
        input,
        residual,
        gate,
        gate_bias if gate_bias is not None else _dit_ln_empty_f32(device),
        gamma,
        beta,
        _dit_ln_empty_bf16(device),
        _dit_ln_empty_f32(device),
        _dit_ln_empty_bf16(device),
        _dit_ln_empty_f32(device),
        residual_out,
        norm_out,
        sf_out,
        sf_scale,
        (
            input_global_scaling_factor
            if input_global_scaling_factor is not None
            else _dit_ln_empty_f32(device)
        ),
        float(epsilon),
        _DIT_LN_MODE_GATE_RESIDUAL_GAMMA_BETA,
        output_format,
    )

    return residual_out, norm_out


@flashinfer_api
def fused_dit_gate_residual_layernorm_scale_shift(
    input: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    gate_bias: Optional[torch.Tensor] = None,
    scale_bias: Optional[torch.Tensor] = None,
    shift_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused gate + residual + LayerNorm + scale/shift for DIT self-attention.

    Computes in a single kernel:

    .. code-block:: python

        residual_out = residual + input * (gate + gate_bias)
        norm_out = LayerNorm(residual_out) * (1 + scale + scale_bias) + (shift + shift_bias)

    Parameters
    ----------
    input : torch.Tensor
        Input tensor ``[batch, num_rows, 3072]``, BF16, contiguous.
    residual : torch.Tensor
        Residual tensor, same shape as input, BF16, contiguous.
    gate : torch.Tensor
        Gating tensor, BF16. Stride ``6 * hidden_dim`` from ``temb.chunk(6, dim=2)``.
    scale : torch.Tensor
        Scale tensor, BF16. Same stride convention as gate.
    shift : torch.Tensor
        Shift tensor, BF16. Same stride convention as gate.
    gate_bias, scale_bias, shift_bias : Optional[torch.Tensor]
        Biases ``[3072]`` or ``[1, 3072]``, FP32.
    epsilon : float
        LayerNorm epsilon.
    use_nvfp4 : bool
        Quantize norm output to NVFP4 (SM100+ only).
    use_mxfp8 : bool
        Quantize norm output to MXFP8 (SM100+ only).
    global_scaling_factor : Optional[torch.Tensor]
        Global scale for NVFP4 output ``[1]``, FP32.
    input_global_scaling_factor : Optional[torch.Tensor]
        Scale for pre-quantized NVFP4 inputs ``[1]``, FP32.
    residual_out : Optional[torch.Tensor]
        Pre-allocated residual output ``[batch, num_rows, 3072]``, BF16.
    norm_out : Optional[torch.Tensor]
        Pre-allocated norm output. Shape depends on output format.
    sf_out : Optional[torch.Tensor]
        Pre-allocated scale factor output for NVFP4/MXFP8.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(residual_out, norm_out)``.

    Note
    ----
    This kernel targets WAN 2.2 5B (hidden_dim=3072 only).
    Primary targets: SM90 (Hopper), SM100/SM103 (Blackwell).
    BF16 output compatible with SM80+. NVFP4/MXFP8 requires SM100+.
    """
    output_format = _dit_ln_resolve_output_format(use_nvfp4, use_mxfp8)
    _dit_ln_check_common_inputs(input, residual, use_nvfp4, use_mxfp8)
    _dit_ln_check_strided_tensor(gate, input, "gate")
    _dit_ln_check_strided_tensor(scale, input, "scale")
    _dit_ln_check_strided_tensor(shift, input, "shift")
    _dit_ln_check_bias_tensor(gate_bias, input, "gate_bias")
    _dit_ln_check_bias_tensor(scale_bias, input, "scale_bias")
    _dit_ln_check_bias_tensor(shift_bias, input, "shift_bias")

    if use_nvfp4 and global_scaling_factor is None:
        raise ValueError("global_scaling_factor required for NVFP4 output")
    _dit_ln_check_scaling_factor(
        global_scaling_factor, input.device, "global_scaling_factor"
    )
    _dit_ln_check_scaling_factor(
        input_global_scaling_factor, input.device, "input_global_scaling_factor"
    )

    device = input.device
    residual_out, norm_out, sf_out, sf_scale = _dit_ln_prepare_outputs(
        input, output_format, global_scaling_factor, residual_out, norm_out, sf_out
    )

    get_norm_module().fused_dit_layernorm(
        input,
        residual,
        gate,
        gate_bias if gate_bias is not None else _dit_ln_empty_f32(device),
        _dit_ln_empty_f32(device),
        _dit_ln_empty_f32(device),
        scale,
        scale_bias if scale_bias is not None else _dit_ln_empty_f32(device),
        shift,
        shift_bias if shift_bias is not None else _dit_ln_empty_f32(device),
        residual_out,
        norm_out,
        sf_out,
        sf_scale,
        (
            input_global_scaling_factor
            if input_global_scaling_factor is not None
            else _dit_ln_empty_f32(device)
        ),
        float(epsilon),
        _DIT_LN_MODE_GATE_RESIDUAL_SCALE_SHIFT,
        output_format,
    )

    return residual_out, norm_out


@flashinfer_api
def fused_dit_residual_layernorm_scale_shift(
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    residual: Optional[torch.Tensor] = None,
    scale_bias: Optional[torch.Tensor] = None,
    shift_bias: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    use_nvfp4: bool = False,
    use_mxfp8: bool = False,
    global_scaling_factor: Optional[torch.Tensor] = None,
    input_global_scaling_factor: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    sf_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused residual + LayerNorm + scale/shift for DIT self-attention.

    Computes in a single kernel:

    .. code-block:: python

        residual_out = residual + input  # (or just input if residual is None)
        norm_out = LayerNorm(residual_out) * (1 + scale + scale_bias) + (shift + shift_bias)

    Parameters
    ----------
    input : torch.Tensor
        Input tensor ``[batch, num_rows, 3072]``, BF16, contiguous.
    scale : torch.Tensor
        Scale tensor, BF16. Stride ``6 * hidden_dim`` from ``temb.chunk(6, dim=2)``.
    shift : torch.Tensor
        Shift tensor, BF16. Same stride convention as scale.
    residual : Optional[torch.Tensor]
        Residual tensor, BF16, contiguous. If None, no residual addition.
    scale_bias, shift_bias : Optional[torch.Tensor]
        Biases ``[3072]`` or ``[1, 3072]``, FP32.
    epsilon : float
        LayerNorm epsilon.
    use_nvfp4 : bool
        Quantize norm output to NVFP4 (SM100+ only).
    use_mxfp8 : bool
        Quantize norm output to MXFP8 (SM100+ only).
    global_scaling_factor : Optional[torch.Tensor]
        Global scale for NVFP4 output ``[1]``, FP32.
    input_global_scaling_factor : Optional[torch.Tensor]
        Scale for pre-quantized NVFP4 inputs ``[1]``, FP32.
    residual_out : Optional[torch.Tensor]
        Pre-allocated residual output ``[batch, num_rows, 3072]``, BF16.
    norm_out : Optional[torch.Tensor]
        Pre-allocated norm output. Shape depends on output format.
    sf_out : Optional[torch.Tensor]
        Pre-allocated scale factor output for NVFP4/MXFP8.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(residual_out, norm_out)``.

    Note
    ----
    This kernel targets WAN 2.2 5B (hidden_dim=3072 only).
    Primary targets: SM90 (Hopper), SM100/SM103 (Blackwell).
    BF16 output compatible with SM80+. NVFP4/MXFP8 requires SM100+.
    """
    output_format = _dit_ln_resolve_output_format(use_nvfp4, use_mxfp8)
    _dit_ln_check_common_inputs(input, residual, use_nvfp4, use_mxfp8)
    _dit_ln_check_strided_tensor(scale, input, "scale")
    _dit_ln_check_strided_tensor(shift, input, "shift")
    _dit_ln_check_bias_tensor(scale_bias, input, "scale_bias")
    _dit_ln_check_bias_tensor(shift_bias, input, "shift_bias")

    if use_nvfp4 and global_scaling_factor is None:
        raise ValueError("global_scaling_factor required for NVFP4 output")
    _dit_ln_check_scaling_factor(
        global_scaling_factor, input.device, "global_scaling_factor"
    )
    _dit_ln_check_scaling_factor(
        input_global_scaling_factor, input.device, "input_global_scaling_factor"
    )

    device = input.device
    residual_out, norm_out, sf_out, sf_scale = _dit_ln_prepare_outputs(
        input, output_format, global_scaling_factor, residual_out, norm_out, sf_out
    )

    get_norm_module().fused_dit_layernorm(
        input,
        residual if residual is not None else _dit_ln_empty_bf16(device),
        _dit_ln_empty_bf16(device),
        _dit_ln_empty_f32(device),
        _dit_ln_empty_f32(device),
        _dit_ln_empty_f32(device),
        scale,
        scale_bias if scale_bias is not None else _dit_ln_empty_f32(device),
        shift,
        shift_bias if shift_bias is not None else _dit_ln_empty_f32(device),
        residual_out,
        norm_out,
        sf_out,
        sf_scale,
        (
            input_global_scaling_factor
            if input_global_scaling_factor is not None
            else _dit_ln_empty_f32(device)
        ),
        float(epsilon),
        _DIT_LN_MODE_RESIDUAL_SCALE_SHIFT,
        output_format,
    )

    return residual_out, norm_out


####################################################################################################
# Fused QK RMSNorm + 3D RoPE for Video Generation DIT Self-Attention
####################################################################################################


@functools.cache
def _get_fused_qk_rmsnorm_rope_module():
    """Compile the norm module for fused_qk_rmsnorm_rope.

    Separate from get_norm_module() so this kernel can be used regardless
    of _USE_CUDA_NORM (which gates CuTe DSL vs CUDA JIT for standard norms).
    The underlying .so is the same norm module, but this accessor is always
    available and only triggers compilation when fused_qk_rmsnorm_rope is
    actually called.
    """
    return gen_norm_module().build_and_load()


@supported_compute_capability([80, 86, 89, 90, 100, 103, 107, 110, 120, 121])
def _check_fused_qk_rmsnorm_rope(
    qkv,
    q_weight,
    k_weight,
    **kwargs,
):
    """Validate inputs for fused QK RMSNorm + 3D RoPE.

    Architecture notes:
    - SM80+ (Ampere): Full support for BF16 path; FP8 output uses software emulation
    - SM89+ (Ada): Native FP8 E4M3 conversion instructions (faster FP8 output)
    - SM90 (Hopper): Primary target architecture
    - SM100/103 (Blackwell B200, B300): Native float2 packed math (FFMA2); primary target
    All SM100+/SM89+ features have scalar fallbacks, so SM80 is the true minimum.
    """
    if not qkv.is_cuda:
        raise ValueError("qkv must be a CUDA tensor")
    if qkv.dtype != torch.bfloat16:
        raise ValueError("qkv must be bfloat16")
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous")
    if qkv.ndim not in (2, 3):
        raise ValueError(
            f"qkv must be 2D [num_tokens, hidden] or 3D [batch, seq_len, hidden], "
            f"got {qkv.ndim}D"
        )

    head_dim = kwargs.get("head_dim")
    if head_dim not in (64, 128, 256):
        raise ValueError(f"head_dim must be 64, 128, or 256, got {head_dim}")

    num_heads_q = kwargs.get("num_heads_q")
    num_heads_k = kwargs.get("num_heads_k")
    num_heads_v = kwargs.get("num_heads_v")
    max_heads = max(num_heads_q, num_heads_k, num_heads_v)
    if max_heads > 32:
        raise ValueError(
            f"max(num_heads_q, num_heads_k, num_heads_v) must be <= 32, got {max_heads}"
        )

    num_frame_channels = kwargs.get("num_frame_channels")
    num_height_channels = kwargs.get("num_height_channels")
    num_width_channels = kwargs.get("num_width_channels")
    if num_frame_channels + num_height_channels + num_width_channels != head_dim:
        raise ValueError(
            f"num_frame_channels ({num_frame_channels}) + num_height_channels "
            f"({num_height_channels}) + num_width_channels ({num_width_channels}) "
            f"must equal head_dim ({head_dim})"
        )
    if (
        num_frame_channels % 2 != 0
        or num_height_channels % 2 != 0
        or num_width_channels % 2 != 0
    ):
        raise ValueError(
            f"Channel counts must all be even (freq table uses count/2), got "
            f"frame={num_frame_channels}, height={num_height_channels}, "
            f"width={num_width_channels}"
        )

    ppf = kwargs.get("ppf")
    pph = kwargs.get("pph")
    ppw = kwargs.get("ppw")
    if ppf <= 0 or pph <= 0 or ppw <= 0:
        raise ValueError(f"ppf, pph, ppw must be positive, got ({ppf}, {pph}, {ppw})")
    expected_seq_len = ppf * pph * ppw
    if qkv.ndim == 3:
        actual_seq_len = qkv.shape[1]
        if actual_seq_len != expected_seq_len:
            raise ValueError(
                f"qkv seq_len ({actual_seq_len}) != ppf*pph*ppw ({expected_seq_len})"
            )
    else:
        num_tokens = qkv.shape[0]
        if num_tokens % expected_seq_len != 0:
            raise ValueError(
                f"qkv num_tokens ({num_tokens}) must be divisible by "
                f"ppf*pph*ppw ({expected_seq_len})"
            )

    return True


@flashinfer_api
@backend_requirement(backend_checks={}, common_check=_check_fused_qk_rmsnorm_rope)
def fused_qk_rmsnorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    ppf: int,
    pph: int,
    ppw: int,
    num_frame_channels: int,
    num_height_channels: int,
    num_width_channels: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float = 1e-6,
    base: float = 10000.0,
    interleave: bool = True,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    output_fp8: bool = False,
    output_quant_scale: float = 1.0,
    v_quant_scale: float = 1.0,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fused QK RMSNorm + 3D RoPE + V copy for video generation DIT self-attention.

    Applies across-heads RMSNorm to Q and K, then rotary position embeddings
    with 3D spatial decomposition (frame/height/width), and copies V to a
    contiguous output buffer. Optionally quantizes all outputs to FP8 E4M3.

    Parameters
    ----------
    qkv : torch.Tensor
        Combined QKV input, BF16, contiguous. Accepted shapes:
        - 3D: ``[batch, seq_len, (num_heads_q+num_heads_k+num_heads_v)*head_dim]``
        - 2D: ``[num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]``
          where ``num_tokens`` must be divisible by ``ppf*pph*ppw``.
    q_weight : torch.Tensor
        RMSNorm weight for Q ``[num_heads_q * head_dim]``, BF16.
    k_weight : torch.Tensor
        RMSNorm weight for K ``[num_heads_k * head_dim]``, BF16.
    ppf : int
        Number of patches in frame dimension.
    pph : int
        Number of patches in height dimension.
    ppw : int
        Number of patches in width dimension.
        ``seq_len = ppf * pph * ppw``.
    num_frame_channels : int
        RoPE frequency channels for the frame dimension (must be even).
    num_height_channels : int
        RoPE frequency channels for the height dimension (must be even).
    num_width_channels : int
        RoPE frequency channels for the width dimension (must be even).
        ``num_frame_channels + num_height_channels + num_width_channels == head_dim``.
    num_heads_q : int
        Number of query heads.
    num_heads_k : int
        Number of key heads.
    num_heads_v : int
        Number of value heads.
    head_dim : int
        Dimension per head (must be 64, 128, or 256).
    eps : float
        RMSNorm epsilon.
    base : float
        RoPE base frequency.
    interleave : bool
        True for interleaved RoPE (non-NeoX style), False for NeoX-style.
    factor : float
        YARN RoPE scaling factor. 1.0 disables YARN.
    low : float
        YARN low frequency threshold.
    high : float
        YARN high frequency threshold.
    attention_factor : float
        YARN attention factor applied to cos/sin. Must be 1.0 when factor is 1.0.
    is_qk_norm : bool
        Whether to apply RMSNorm (False = RoPE only, skip normalization).
    output_fp8 : bool
        Quantize Q, K, V outputs to FP8 E4M3.
    output_quant_scale : float
        FP8 quantization scale for Q and K outputs.
    v_quant_scale : float
        FP8 quantization scale for V output.
    q_out : Optional[torch.Tensor]
        Pre-allocated Q output tensor (destination-passing style).
    k_out : Optional[torch.Tensor]
        Pre-allocated K output tensor.
    v_out : Optional[torch.Tensor]
        Pre-allocated V output tensor.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(q_out, k_out, v_out)``. If input is 3D, each has shape
        ``[batch, seq_len, num_heads_x, head_dim]``. If input is 2D,
        each has shape ``[num_tokens, num_heads_x, head_dim]``.
    """
    out_dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
    seq_len = ppf * pph * ppw

    out_shape_q: tuple[int, ...]
    out_shape_k: tuple[int, ...]
    out_shape_v: tuple[int, ...]
    if qkv.ndim == 3:
        batch_size = qkv.shape[0]
        num_tokens = batch_size * seq_len
        out_shape_q = (batch_size, seq_len, num_heads_q, head_dim)
        out_shape_k = (batch_size, seq_len, num_heads_k, head_dim)
        out_shape_v = (batch_size, seq_len, num_heads_v, head_dim)
    else:
        num_tokens = qkv.shape[0]
        out_shape_q = (num_tokens, num_heads_q, head_dim)
        out_shape_k = (num_tokens, num_heads_k, head_dim)
        out_shape_v = (num_tokens, num_heads_v, head_dim)

    # Validate weights
    expected_q_weight_numel = num_heads_q * head_dim
    expected_k_weight_numel = num_heads_k * head_dim
    if q_weight.numel() != expected_q_weight_numel:
        raise ValueError(
            f"q_weight size {q_weight.numel()} != num_heads_q*head_dim ({expected_q_weight_numel})"
        )
    if k_weight.numel() != expected_k_weight_numel:
        raise ValueError(
            f"k_weight size {k_weight.numel()} != num_heads_k*head_dim ({expected_k_weight_numel})"
        )
    if q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16:
        raise ValueError("q_weight and k_weight must be bfloat16")
    if not q_weight.is_contiguous() or not k_weight.is_contiguous():
        raise ValueError("q_weight and k_weight must be contiguous")

    if q_out is None:
        q_out = torch.empty(*out_shape_q, dtype=out_dtype, device=qkv.device)
    if k_out is None:
        k_out = torch.empty(*out_shape_k, dtype=out_dtype, device=qkv.device)
    if v_out is None:
        v_out = torch.empty(*out_shape_v, dtype=out_dtype, device=qkv.device)

    # Validate user-supplied output buffers
    for name, buf, expected_shape in [
        ("q_out", q_out, out_shape_q),
        ("k_out", k_out, out_shape_k),
        ("v_out", v_out, out_shape_v),
    ]:
        if tuple(buf.shape) != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(buf.shape)} != expected {expected_shape}"
            )
        if buf.dtype != out_dtype:
            raise ValueError(f"{name} dtype {buf.dtype} != expected {out_dtype}")
        if not buf.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if buf.device != qkv.device:
            raise ValueError(f"{name} device {buf.device} != qkv device {qkv.device}")

    qkv_flat = qkv.view(num_tokens, -1)
    q_out_flat = q_out.view(num_tokens, -1)
    k_out_flat = k_out.view(num_tokens, -1)
    v_out_flat = v_out.view(num_tokens, -1)

    _get_fused_qk_rmsnorm_rope_module().fused_qk_rmsnorm_rope(
        qkv_flat,
        q_weight,
        k_weight,
        q_out_flat,
        k_out_flat,
        v_out_flat,
        num_tokens,
        seq_len,
        ppf,
        pph,
        ppw,
        num_frame_channels,
        num_height_channels,
        num_width_channels,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        float(eps),
        float(base),
        interleave,
        float(factor),
        float(low),
        float(high),
        float(attention_factor),
        is_qk_norm,
        output_fp8,
        float(output_quant_scale),
        float(v_quant_scale),
    )

    return q_out, k_out, v_out


# Public API exports
__all__ = [
    # JIT module generator (always available)
    "gen_norm_module",
    # Public APIs
    "rmsnorm",
    "rmsnorm_quant",
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_quant",
    "gemma_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "layernorm",
    "layernorm_quant",
    "fused_rmsnorm_silu",
    # Fused DIT LayerNorm (diffusion transformer)
    "fused_dit_gate_residual_layernorm_gamma_beta",
    "fused_dit_gate_residual_layernorm_scale_shift",
    "fused_dit_residual_layernorm_scale_shift",
    "fused_qk_rmsnorm_rope",
]
