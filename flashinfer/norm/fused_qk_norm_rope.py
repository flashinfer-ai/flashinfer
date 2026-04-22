"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Fused QKNorm + 3D RoPE for Video Generation DIT Self-Attention
===============================================================

Fuses across-heads RMSNorm on Q and K, 3D rotary position embeddings
(frame/height/width decomposition), and V passthrough copy into a single
CUDA kernel. Optionally quantizes outputs to FP8 E4M3.

Optimized for WAN 2.2 and similar video generation DIT architectures.
"""

from typing import Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.norm import gen_norm_module
from ..utils import backend_requirement, supported_compute_capability

from . import get_norm_module


@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120, 121])
def _check_fused_qk_norm_rope(
    qkv,
    q_weight,
    k_weight,
    **kwargs,
):
    """Validate inputs for fused QKNorm + 3D RoPE.

    Architecture notes:
    - SM80+ (Ampere): Full support for BF16 path; FP8 output uses software emulation
    - SM89+ (Ada): Native FP8 E4M3 conversion instructions (faster FP8 output)
    - SM90+ (Hopper): Target architecture, tested by kernel author
    - SM100+ (Blackwell): Native float2 packed math (FFMA2); __CUDA_ARCH__ >= 1000
    All SM100+/SM89+ features have scalar fallbacks, so SM80 is the true minimum.
    """
    if not qkv.is_cuda:
        raise ValueError("qkv must be a CUDA tensor")
    if qkv.dtype != torch.bfloat16:
        raise ValueError("qkv must be bfloat16")
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous")
    if qkv.ndim != 3:
        raise ValueError(f"qkv must be 3D [batch, seq_len, hidden], got {qkv.ndim}D")

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
    actual_seq_len = qkv.shape[1]
    if actual_seq_len != expected_seq_len:
        raise ValueError(
            f"qkv seq_len ({actual_seq_len}) != ppf*pph*ppw ({expected_seq_len})"
        )

    return True


@flashinfer_api
@backend_requirement(backend_checks={}, common_check=_check_fused_qk_norm_rope)
def fused_qk_norm_rope(
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
    r"""Fused QKNorm + 3D RoPE + V copy for video generation DIT self-attention.

    Applies across-heads RMSNorm to Q and K, then rotary position embeddings
    with 3D spatial decomposition (frame/height/width), and copies V to a
    contiguous output buffer. Optionally quantizes all outputs to FP8 E4M3.

    Parameters
    ----------
    qkv : torch.Tensor
        Combined QKV input ``[batch, seq_len, (num_heads_q+num_heads_k+num_heads_v)*head_dim]``,
        BF16, contiguous.
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
        ``(q_out, k_out, v_out)``, each ``[batch, seq_len, num_heads_x, head_dim]``.
    """
    batch_size = qkv.shape[0]
    seq_len = qkv.shape[1]
    num_tokens = batch_size * seq_len
    out_dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16

    if q_out is None:
        q_out = torch.empty(
            batch_size, seq_len, num_heads_q, head_dim,
            dtype=out_dtype, device=qkv.device,
        )
    if k_out is None:
        k_out = torch.empty(
            batch_size, seq_len, num_heads_k, head_dim,
            dtype=out_dtype, device=qkv.device,
        )
    if v_out is None:
        v_out = torch.empty(
            batch_size, seq_len, num_heads_v, head_dim,
            dtype=out_dtype, device=qkv.device,
        )

    qkv_flat = qkv.view(num_tokens, -1)
    q_out_flat = q_out.view(num_tokens, -1)
    k_out_flat = k_out.view(num_tokens, -1)
    v_out_flat = v_out.view(num_tokens, -1)

    get_norm_module().fused_qk_norm_rope(
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
