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
from typing import Final

import torch

from ..api_logging import flashinfer_api
from ..jit.cpp_ext import is_cuda_version_at_least
from ..jit.minimax_h3 import gen_minimax_h3_bf16_pre_attention_module
from ..trace.templates.diffusion import minimax_h3_bf16_pre_attention_trace
from ..utils import (
    get_compute_capability,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)


_HIDDEN: Final = 5376
_NUM_HEADS: Final = 56
_HEAD_DIM: Final = 128
_QKV_KINDS: Final = 3
_QKV_WIDTH: Final = _NUM_HEADS * _QKV_KINDS * _HEAD_DIM
_ROPE_DIM: Final = 96
_ADALN_ROWS: Final = 9
_EPS: Final = 1.0e-5
_SUPPORTED_ULYSSES_DEGREES: Final = frozenset((1, 2, 4, 8))


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {shape}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype {tensor.dtype} != expected {dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} device {tensor.device} != x device {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_input_contract(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    out: torch.Tensor,
    *,
    ulysses_degree: int,
    eps: float,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2 or x.shape[1] != _HIDDEN:
        raise ValueError(f"x must have shape [M, {_HIDDEN}], got {tuple(x.shape)}")
    if x.shape[0] <= 0:
        raise ValueError("x.shape[0] (M) must be positive")
    if x.dtype != torch.bfloat16:
        raise ValueError("x must be bfloat16")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if (
        isinstance(ulysses_degree, bool)
        or ulysses_degree not in _SUPPORTED_ULYSSES_DEGREES
    ):
        raise ValueError("ulysses_degree must be one of 1, 2, 4, or 8")
    if float(eps) != _EPS:
        raise ValueError(f"eps must be {_EPS} for this kernel")

    m = x.shape[0]
    p = ulysses_degree
    device = x.device
    _require_tensor(
        "x_norm_weight",
        x_norm_weight,
        shape=(_HIDDEN,),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        "adaln_scale",
        adaln_scale,
        shape=(_ADALN_ROWS, _HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        "adaln_shift",
        adaln_shift,
        shape=(_ADALN_ROWS, _HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        "adaln_index",
        adaln_index,
        shape=(m,),
        dtype=torch.int32,
        device=device,
    )
    _require_tensor(
        "qkv_weight",
        qkv_weight,
        shape=(_QKV_WIDTH, _HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        "q_norm_weight",
        q_norm_weight,
        shape=(_HEAD_DIM,),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        "k_norm_weight",
        k_norm_weight,
        shape=(_HEAD_DIM,),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        "rope_cos_sin",
        rope_cos_sin,
        shape=(m, _ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    _require_tensor(
        "out",
        out,
        shape=(p, m, _NUM_HEADS // p, _QKV_KINDS, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )


def _check_runtime_support(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("MiniMax-H3 BF16 pre-attention requires CUDA tensors")
    if get_compute_capability(device) != (10, 3):
        raise RuntimeError(
            "MiniMax-H3 BF16 pre-attention requires compute capability 10.3"
        )
    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError("MiniMax-H3 BF16 pre-attention requires CUDA 12.9 or newer")


@functools.cache
def _get_module():
    return gen_minimax_h3_bf16_pre_attention_module().build_and_load()


@register_custom_op(
    "flashinfer::minimax_h3_bf16_pre_attention",
    mutates_args=("out",),
)
def _minimax_h3_bf16_pre_attention_impl(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    out: torch.Tensor,
    m: int,
    ulysses_degree: int,
    eps: float,
) -> None:
    _get_module().minimax_h3_bf16_pre_attention(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        out,
        m,
        ulysses_degree,
        eps,
    )


@register_fake_op("flashinfer::minimax_h3_bf16_pre_attention")
def _minimax_h3_bf16_pre_attention_fake(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    out: torch.Tensor,
    m: int,
    ulysses_degree: int,
    eps: float,
) -> None:
    pass


@supported_compute_capability([103])
@flashinfer_api(trace=minimax_h3_bf16_pre_attention_trace)
def minimax_h3_bf16_pre_attention(
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    *,
    ulysses_degree: int,
    out: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    r"""Run the fused BF16 pre-attention projection for MiniMax-H3.

    The operation applies input RMSNorm, indexed AdaLN, a BF16 QKV
    projection, per-head Q/K RMSNorm, partial 3-D split-half NeoX RoPE, and a
    destination-major output pack. The collective that consumes ``out`` is
    outside this operation.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous BF16 input with shape ``[M, 5376]``.
    x_norm_weight : torch.Tensor
        Contiguous BF16 input RMSNorm weight with shape ``[5376]``.
    adaln_scale, adaln_shift : torch.Tensor
        Contiguous BF16 AdaLN tables with shape ``[9, 5376]``.
    adaln_index : torch.Tensor
        Contiguous int32 row indices with shape ``[M]`` and values in
        ``[0, 8]``. A malformed index is guarded in the CUDA kernel and makes
        its corresponding output row all-zero instead of addressing outside
        the AdaLN tables.
    qkv_weight : torch.Tensor
        Contiguous BF16 checkpoint weight with physical shape
        ``[21504, 5376]`` and row order ``[head, qkv_kind, head_dim]``.
    q_norm_weight, k_norm_weight : torch.Tensor
        Contiguous BF16 per-head RMSNorm weights with shape ``[128]``.
    rope_cos_sin : torch.Tensor
        Contiguous BF16 cache with shape ``[M, 96]``. Columns ``[0, 48)``
        hold frame/height/width cosine values and columns ``[48, 96)`` hold
        the corresponding sine values. RoPE transforms Q/K dimensions
        ``[0, 96)``; dimensions ``[96, 128)`` pass through.
    ulysses_degree : int
        Destination count, one of ``1``, ``2``, ``4``, or ``8``.
    out : torch.Tensor
        Caller-owned contiguous BF16 destination with shape
        ``[P, M, 56 // P, 3, 128]``.
    eps : float
        RMSNorm epsilon. This kernel supports ``1e-5``.

    Returns
    -------
    torch.Tensor
        The same tensor passed as ``out``.

    Notes
    -----
    The CUDA kernel independently guards its AdaLN table loads, so malformed
    indices cannot form an out-of-bounds address. Valid indices preserve the
    MiniMax-H3 checkpoint semantics without a synchronizing host reduction.

    This is a direct kernel entry point for all supported destination counts.
    The measured performance promotion range is ``P in {2, 4, 8}``; callers
    that dispatch by ``P`` should retain their segmented fallback for ``P=1``.
    """
    _validate_input_contract(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        out,
        ulysses_degree=ulysses_degree,
        eps=eps,
    )
    _check_runtime_support(x.device)
    _minimax_h3_bf16_pre_attention_impl(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        out,
        x.shape[0],
        ulysses_degree,
        float(eps),
    )
    return out


__all__ = ["minimax_h3_bf16_pre_attention"]
