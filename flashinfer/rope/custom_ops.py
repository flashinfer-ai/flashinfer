"""
Copyright (c) 2024-2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CUDA Custom Op Registrations for RoPE
=====================================

This module contains the custom op registrations for the CUDA RoPE kernels.
These register PyTorch custom ops that dispatch to the JIT-compiled CUDA module.
"""

import torch

from ..utils import register_custom_op, register_fake_op
from .utils import get_rope_module


# ============================================================================
# Standard RoPE with indptr/offsets
# ============================================================================


@register_custom_op("flashinfer::apply_rope", mutates_args=("q_rope", "k_rope"))
def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    """Apply standard RoPE using indptr/offsets for batched sequences."""
    get_rope_module().apply_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


@register_fake_op("flashinfer::apply_rope")
def _fake_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass


# ============================================================================
# Llama 3.1 style RoPE with indptr/offsets
# ============================================================================


@register_custom_op("flashinfer::apply_llama31_rope", mutates_args=("q_rope", "k_rope"))
def _apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    """Apply Llama 3.1 style RoPE with adaptive frequency scaling."""
    get_rope_module().apply_llama31_rope(
        q,
        k,
        q_rope,
        k_rope,
        indptr,
        offsets,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope")
def _fake_apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass


# ============================================================================
# Standard RoPE with position IDs
# ============================================================================


@register_custom_op("flashinfer::apply_rope_pos_ids", mutates_args=("q_rope", "k_rope"))
def _apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    """Apply standard RoPE using explicit position IDs."""
    get_rope_module().apply_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids")
def _fake_apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
) -> None:
    pass


# ============================================================================
# Llama 3.1 style RoPE with position IDs
# ============================================================================


@register_custom_op(
    "flashinfer::apply_llama31_rope_pos_ids", mutates_args=("q_rope", "k_rope")
)
def _apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    """Apply Llama 3.1 style RoPE using explicit position IDs."""
    get_rope_module().apply_llama31_rope_pos_ids(
        q,
        k,
        q_rope,
        k_rope,
        pos_ids,
        rotary_dim,
        interleave,
        rope_scale,
        rope_theta,
        low_freq_factor,
        high_freq_factor,
        old_context_len,
    )


@register_fake_op("flashinfer::apply_llama31_rope_pos_ids")
def _fake_apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    interleave: bool,
    rope_scale: float,
    rope_theta: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: float,
) -> None:
    pass


# ============================================================================
# RoPE with cos/sin cache
# ============================================================================


@register_custom_op(
    "flashinfer::apply_rope_pos_ids_cos_sin_cache", mutates_args=("q_rope", "k_rope")
)
def _apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    """Apply RoPE using precomputed cos/sin cache values."""
    get_rope_module().apply_rope_pos_ids_cos_sin_cache(
        q,
        k,
        q_rope,
        k_rope,
        cos_sin_cache,
        pos_ids,
        interleave,
    )


@register_fake_op("flashinfer::apply_rope_pos_ids_cos_sin_cache")
def _fake_apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    interleave: bool,
) -> None:
    pass


# ============================================================================
# RoPE + Quantize combined kernels
# ============================================================================


@register_custom_op(
    "flashinfer::rope_quantize",
    mutates_args=("q_rope_out", "k_rope_out", "q_nope_out", "k_nope_out"),
)
def _rope_quantize(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    """Combined RoPE application and FP8 quantization."""
    get_rope_module().rope_quantize(
        q_rope_in,
        k_rope_in,
        q_nope_in,
        k_nope_in,
        q_rope_out,
        k_rope_out,
        q_nope_out,
        k_nope_out,
        cos_sin_cache,
        pos_ids,
        quant_scale_q,
        quant_scale_kv,
        interleave,
        enable_pdl,
    )


@register_fake_op("flashinfer::rope_quantize")
def _fake_rope_quantize(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    pass


# ============================================================================
# RoPE + Quantize + Append to paged KV cache
# ============================================================================


@register_custom_op(
    "flashinfer::rope_quantize_append_paged_kv_cache",
    mutates_args=(
        "q_rope_out",
        "q_nope_out",
        "k_cache",
        "v_cache",
        "ckv_cache",
        "kpe_cache",
    ),
)
def _rope_quantize_fp8_append_paged_kv_cache(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    v_in: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_layout_code: int,
    page_size: int,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    """Combined RoPE + quantize + paged KV cache append."""
    get_rope_module().rope_quantize_append_paged_kv_cache(
        q_rope_in,
        k_rope_in,
        q_nope_in,
        k_nope_in,
        v_in,
        q_rope_out,
        q_nope_out,
        cos_sin_cache,
        pos_ids,
        k_cache,
        v_cache,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
        kv_layout_code,
        page_size,
        quant_scale_q,
        quant_scale_kv,
        interleave,
        enable_pdl,
    )


@register_fake_op("flashinfer::rope_quantize_append_paged_kv_cache")
def _fake_rope_quantize_fp8_append_paged_kv_cache(
    q_rope_in: torch.Tensor,
    k_rope_in: torch.Tensor,
    q_nope_in: torch.Tensor,
    k_nope_in: torch.Tensor,
    v_in: torch.Tensor,
    q_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_layout_code: int,
    page_size: int,
    quant_scale_q: float,
    quant_scale_kv: float,
    interleave: bool,
    enable_pdl: bool,
) -> None:
    pass
