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

from typing import Dict, Final, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.cpp_ext import is_cuda_version_at_least
from ..trace.templates.cake_minimax_h3_bf16_pre_attention import (
    minimax_h3_bf16_pre_attention_trace,
)
from ..utils import (
    get_compute_capability,
    supported_compute_capability,
)

from .cake_minimax_h3_bf16_pre_attention import (
    get_minimax_h3_bf16_pre_attention_backend,
)


_HIDDEN: Final = 5376
_NUM_HEADS: Final = 56
_HEAD_DIM: Final = 128
_QKV_KINDS: Final = 3
_QKV_WIDTH: Final = _NUM_HEADS * _QKV_KINDS * _HEAD_DIM
_ROPE_DIM: Final = 96
_EPS: Final = 1.0e-5
_SUPPORTED_ULYSSES_DEGREES: Final = frozenset((1, 2, 4, 8))
# AdaLN tables are read with 16-byte vector loads: the row pitch must be a
# multiple of 8 BF16 elements and the base pointer 16-byte aligned.
_TABLE_ALIGN_ELEMENTS: Final = 8
_TABLE_ALIGN_BYTES: Final = 16

# Identity RoPE positions (``rope_positions=None``): one int64 ``arange(M)`` per
# ``(M, device index)``, allocated on first use and reused by every later call.
_IDENTITY_ROPE_POSITIONS: Dict[Tuple[int, int], torch.Tensor] = {}


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


def _require_index(
    name: str, tensor: torch.Tensor, *, m: int, device: torch.device
) -> None:
    """Contiguous int64 ``[M]`` row index (AdaLN rows, RoPE positions)."""
    _require_tensor(name, tensor, shape=(m,), dtype=torch.int64, device=device)


def _require_table(name: str, tensor: torch.Tensor, *, device: torch.device) -> None:
    """BF16 ``[rows, 5376]`` table with any ``rows >= 1`` and a 16-byte row pitch.

    Column chunks of a wider projection (``stride(0) > 5376``) are accepted as
    they are; the kernel receives ``rows`` and ``stride(0)``.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2 or tensor.shape[1] != _HIDDEN or tensor.shape[0] < 1:
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} != expected [rows >= 1, {_HIDDEN}]"
        )
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} dtype {tensor.dtype} != expected torch.bfloat16")
    if tensor.device != device:
        raise ValueError(f"{name} device {tensor.device} != x device {device}")
    if tensor.stride(1) != 1:
        raise ValueError(
            f"{name} must have a unit last stride (stride(1) == 1), got strides "
            f"{tuple(tensor.stride())}"
        )
    if tensor.stride(0) % _TABLE_ALIGN_ELEMENTS != 0:
        raise ValueError(
            f"{name} row pitch stride(0) = {tensor.stride(0)} elements must be a "
            f"multiple of {_TABLE_ALIGN_ELEMENTS} elements ({_TABLE_ALIGN_BYTES} bytes)"
        )
    span = (tensor.shape[0] - 1) * tensor.stride(0) + _HIDDEN
    if span >= 2**32:
        raise ValueError(
            f"{name} addressed span (rows - 1) * stride(0) + {_HIDDEN} = {span} elements "
            f"must fit 32 bits (the kernel forms table offsets as u32 row * stride + k)"
        )
    if tensor.device.type == "cuda" and tensor.data_ptr() % _TABLE_ALIGN_BYTES != 0:
        raise ValueError(
            f"{name} data pointer must be {_TABLE_ALIGN_BYTES}-byte aligned"
        )


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
    rope_positions: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    ulysses_degree: int,
    eps: float,
    qk_eps: Optional[float],
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
    float(eps)
    if qk_eps is not None:
        float(qk_eps)

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
    _require_table("adaln_scale", adaln_scale, device=device)
    _require_table("adaln_shift", adaln_shift, device=device)
    _require_index("adaln_index", adaln_index, m=m, device=device)
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
    if not isinstance(rope_cos_sin, torch.Tensor):
        raise TypeError("rope_cos_sin must be a torch.Tensor")
    if (
        rope_cos_sin.ndim != 2
        or rope_cos_sin.shape[1] != _ROPE_DIM
        or rope_cos_sin.shape[0] < 1
    ):
        raise ValueError(
            f"rope_cos_sin shape {tuple(rope_cos_sin.shape)} != expected "
            f"[S >= 1, {_ROPE_DIM}]"
        )
    cache_rows = int(rope_cos_sin.shape[0])
    _require_tensor(
        "rope_cos_sin",
        rope_cos_sin,
        shape=(cache_rows, _ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    if rope_positions is None:
        if cache_rows < m:
            raise ValueError(
                f"rope_cos_sin has {cache_rows} rows but rope_positions=None (identity) "
                f"needs at least M = {m} rows"
            )
    else:
        _require_index("rope_positions", rope_positions, m=m, device=device)
    _require_tensor(
        "out",
        out,
        shape=(p, m, _NUM_HEADS // p, _QKV_KINDS, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )


def _identity_rope_positions(m: int, device: torch.device) -> torch.Tensor:
    """Cached int64 ``arange(M)`` on ``device`` (no allocation after the first call)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (m, index)
    positions = _IDENTITY_ROPE_POSITIONS.get(key)
    if positions is None:
        positions = torch.arange(m, dtype=torch.int64, device=device)
        _IDENTITY_ROPE_POSITIONS[key] = positions
    return positions


def _check_runtime_support(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("MiniMax-H3 BF16 pre-attention requires CUDA tensors")
    if get_compute_capability(device) not in {(10, 0), (10, 3)}:
        raise RuntimeError(
            "MiniMax-H3 BF16 pre-attention requires compute capability 10.0 or 10.3"
        )
    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError("MiniMax-H3 BF16 pre-attention requires CUDA 12.9 or newer")


@supported_compute_capability([100, 103])
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
    qk_eps: Optional[float] = None,
    rope_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Run the fused BF16 pre-attention projection for MiniMax-H3.

    The operation applies input RMSNorm, indexed AdaLN, a BF16 QKV
    projection, per-head Q/K RMSNorm, partial 3-D split-half NeoX RoPE, and a
    destination-major output pack. The collective that consumes ``out`` is
    outside this operation.

    The operands are the engine's own tensors: AdaLN tables may be column
    chunks of a wider projection, indices are int64, and RoPE is a shared
    ``(cos_sin_cache, positions)`` pair. No copies are made on the host.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous BF16 input with shape ``[M, 5376]``.
    x_norm_weight : torch.Tensor
        Contiguous BF16 input RMSNorm weight with shape ``[5376]``.
    adaln_scale, adaln_shift : torch.Tensor
        BF16 AdaLN tables with shape ``[rows, 5376]`` for any ``rows >= 1``.
        ``stride(1)`` must be ``1``; ``stride(0)`` may exceed ``5376`` (for
        example ``6 * 5376`` for a column chunk of the engine's ``[rows,
        6 * 5376]`` modulation projection) and must be a multiple of 8
        elements (16 bytes); the data pointer must be 16-byte aligned. The
        tensors are passed to the kernel as they are, with their row count
        and row stride.
    adaln_index : torch.Tensor
        Contiguous int64 row indices with shape ``[M]``. Values in
        ``[0, rows)`` select a table row; any other int64 value is guarded in
        the CUDA kernel and makes its output row all-zero instead of
        addressing outside the tables.
    qkv_weight : torch.Tensor
        Contiguous BF16 checkpoint weight with physical shape
        ``[21504, 5376]`` and row order ``[head, qkv_kind, head_dim]``.
    q_norm_weight, k_norm_weight : torch.Tensor
        Contiguous BF16 per-head RMSNorm weights with shape ``[128]``.
    rope_cos_sin : torch.Tensor
        Contiguous BF16 cache with shape ``[S, 96]`` and ``S >= 1``. Columns
        ``[0, 48)`` hold frame/height/width cosine values and columns
        ``[48, 96)`` hold the corresponding sine values. Row ``m`` of the
        projection uses cache row ``rope_positions[m]``. RoPE transforms Q/K
        dimensions ``[0, 96)``; dimensions ``[96, 128)`` pass through.
    ulysses_degree : int
        Destination count, one of ``1``, ``2``, ``4``, or ``8``.
    out : torch.Tensor
        Caller-owned contiguous BF16 destination with shape
        ``[P, M, 56 // P, 3, 128]``.
    eps : float
        Epsilon of the input RMSNorm (runtime argument).
    qk_eps : Optional[float]
        Epsilon of the per-head Q/K RMSNorms. ``None`` uses ``eps``.
    rope_positions : Optional[torch.Tensor]
        Contiguous int64 ``[M]`` cache row per token. Positions outside
        ``[0, S)`` are clamped on the device. ``None`` means the identity
        (row ``m`` uses cache row ``m``, so ``S >= M`` is required); the
        identity is passed as a cached per-``(M, device)`` ``arange`` tensor
        that is allocated on the first call for that ``M`` and reused
        afterwards, so the identity path is CUDA-graph safe once warmed up
        with one eager call of the same ``M``.

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
    On SM103a, the measured performance promotion range is ``P in {2, 4, 8}``.
    Callers that dispatch by ``P`` should retain their segmented fallback for
    ``P=1``.
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
        rope_positions,
        out,
        ulysses_degree=ulysses_degree,
        eps=eps,
        qk_eps=qk_eps,
    )
    _check_runtime_support(x.device)
    m = x.shape[0]
    if rope_positions is None:
        rope_positions = _identity_rope_positions(m, x.device)
    get_minimax_h3_bf16_pre_attention_backend(backend="cake")(
        x,
        x_norm_weight,
        adaln_scale,
        adaln_shift,
        adaln_index,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        rope_cos_sin,
        rope_positions,
        out,
        m,
        ulysses_degree,
        float(eps),
        float(eps if qk_eps is None else qk_eps),
    )
    return out


__all__ = ["minimax_h3_bf16_pre_attention"]
