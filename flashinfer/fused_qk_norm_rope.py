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
"""

import functools
from typing import Optional

import torch

from .api_logging import flashinfer_api
from .jit.fused_qk_norm_rope import gen_fused_qk_norm_rope_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_fused_qk_norm_rope_module():
    return gen_fused_qk_norm_rope_module().build_and_load()


@register_custom_op("flashinfer::fused_qk_norm_rope", mutates_args=("q", "k"))
def _fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    eps: float,
    rope_theta: float,
    interleave: bool,
    yarn_factor: float,
    yarn_low: float,
    yarn_high: float,
    yarn_attention_factor: float,
    is_qk_norm: bool,
) -> None:
    get_fused_qk_norm_rope_module().fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        pos_ids,
        rotary_dim,
        eps,
        rope_theta,
        interleave,
        yarn_factor,
        yarn_low,
        yarn_high,
        yarn_attention_factor,
        is_qk_norm,
    )


@register_fake_op("flashinfer::fused_qk_norm_rope")
def _fake_fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    eps: float,
    rope_theta: float,
    interleave: bool,
    yarn_factor: float,
    yarn_low: float,
    yarn_high: float,
    yarn_attention_factor: float,
    is_qk_norm: bool,
) -> None:
    pass


@flashinfer_api
def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    eps: float = 1e-6,
    rope_theta: float = 1e4,
    interleave: bool = False,
    is_qk_norm: bool = True,
    yarn_factor: float = 1.0,
    yarn_low: float = 0.0,
    yarn_high: float = 0.0,
    yarn_attention_factor: float = 1.0,
) -> None:
    r"""Fused per-head QK RMSNorm + Rotary Position Embedding, updated in place.

    Combines (optional) per-head RMSNorm with separate ``q_weight`` / ``k_weight``
    and RoPE (with on-the-fly cos/sin derived from ``pos_ids`` and
    ``rope_theta``) in a single CUDA kernel launch. Optionally applies YaRN
    frequency correction when ``yarn_factor != 1.0``. This is the flashinfer
    port of TensorRT-LLM's ``trtllm::fused_qk_norm_rope`` torch op and is
    intended for Qwen3-style / ExaOne4-style attention layers that place
    QK-norm before RoPE.

    Both ``q`` and ``k`` are updated **in place**.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``(nnz, num_q_heads, head_dim)``, dtype
        ``torch.bfloat16`` or ``torch.float16``. Last dim must be contiguous.
        Updated in place.
    k : torch.Tensor
        Key tensor, shape ``(nnz, num_kv_heads, head_dim)``, same dtype as ``q``.
        Last dim must be contiguous. Updated in place.
    q_weight : torch.Tensor
        Per-head RMSNorm weights for Q, shape ``(head_dim,)``, same dtype as
        ``q``. Ignored when ``is_qk_norm=False``.
    k_weight : torch.Tensor
        Per-head RMSNorm weights for K, shape ``(head_dim,)``, same dtype as
        ``q``. Ignored when ``is_qk_norm=False``.
    pos_ids : torch.Tensor
        Position indices, shape ``(nnz,)``, dtype ``torch.int32``.
    rotary_dim : Optional[int]
        Number of leading dimensions RoPE is applied to. Defaults to
        ``head_dim`` (full rope). Must be even and ``<= head_dim``.
    eps : float
        RMSNorm epsilon. Default ``1e-6``.
    rope_theta : float
        RoPE base frequency (``base`` in TRT-LLM). Default ``1e4``.
    interleave : bool
        If ``True``, RoPE rotates GPT-J-style pairs
        ``(x[..., 2i], x[..., 2i+1])``. If ``False`` (Neox / Llama style), RoPE
        rotates ``(x[..., :rotary_dim//2], x[..., rotary_dim//2:rotary_dim])``.
        ``interleave = not is_neox`` relative to TRT-LLM's wrapper.
        Default ``False``.
    is_qk_norm : bool
        If ``False``, skip the RMSNorm stage and only apply RoPE. Default
        ``True``.
    yarn_factor : float
        YaRN scaling factor. ``1.0`` disables YaRN (standard RoPE frequency).
    yarn_low : float
        YaRN low bound of the ramp (in half-dim units).
    yarn_high : float
        YaRN high bound of the ramp.
    yarn_attention_factor : float
        YaRN post-processing scale applied to ``cos``/``sin``. Must be ``1.0``
        when ``yarn_factor == 1.0``.

    Notes
    -----
    Supports ``head_dim`` in ``{64, 128, 256}``. For ``interleave=False``
    (Neox) with partial RoPE (``rotary_dim < head_dim``), the value
    ``rotary_dim / (2 * head_dim / 32)`` must be a positive power of 2; this
    constraint comes from the intra-warp ``__shfl_xor_sync`` pair-swap and is
    validated in the kernel launcher. The stricter check is adopted from
    SGLang's port of this kernel.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> nnz, H_q, H_k, D = 4, 32, 8, 128
    >>> q = torch.randn(nnz, H_q, D, dtype=torch.bfloat16, device="cuda")
    >>> k = torch.randn(nnz, H_k, D, dtype=torch.bfloat16, device="cuda")
    >>> q_w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    >>> k_w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    >>> pos = torch.arange(nnz, dtype=torch.int32, device="cuda")
    >>> flashinfer.fused_qk_norm_rope(q, k, q_w, k_w, pos)
    """
    if rotary_dim is None:
        rotary_dim = q.size(-1)
    if yarn_factor == 1.0 and yarn_attention_factor != 1.0:
        raise ValueError(
            "yarn_attention_factor must be 1.0 when yarn_factor == 1.0 (YaRN disabled)"
        )
    _fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        pos_ids,
        int(rotary_dim),
        float(eps),
        float(rope_theta),
        bool(interleave),
        float(yarn_factor),
        float(yarn_low),
        float(yarn_high),
        float(yarn_attention_factor),
        bool(is_qk_norm),
    )
