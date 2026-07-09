# Copyright (c) 2026 by FlashInfer team.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# This file is adapted from the diffusers WanTransformer3DModel implementation:
#   https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
# The original code is licensed under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FlashInfer-optimized implementation of WanTransformer3DModel.

This module provides an inference-optimized version of the Wan video transformer
using FlashInfer kernels for attention, normalization, and GEMM operations.

Original model: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py

Optimizations:
- RMSNorm: torch.nn.RMSNorm -> flashinfer.rmsnorm
- Attention:
  - single_prefill_with_kv_cache for single-batch
  - cudnn_batch_prefill_with_kv_cache for multi-batch (cuDNN backend)
  - trtllm_batch_context_with_kv_cache for multi-batch (TRT-LLM, supports skip-softmax sparse)
- Activations: Fused GELU when applicable
- Linear: nn.Linear -> FlashInferLinear with multiple GEMM backends:
  - mm_bf16, mm_fp8, mm_fp4, bmm_fp8, bmm_bf16, mm_mxfp8, bmm_mxfp8
  - gemm_fp8_nt_groupwise, gemm_fp8_nt_blockscaled (SM100+ CUTLASS NT GEMM)
  - batch_deepgemm_fp8_nt_groupwise (SM100/103 DeepGEMM)
  - fp8_blockscale_gemm_sm90 (Hopper-optimized)
- Activation quantization: Online (compute scale from data) or offline (fixed default scale)
- Sparse Attention: Optional skip-softmax sparse attention via trtllm_batch_context_with_kv_cache

Note: The 3D RoPE (WanRotaryPosEmbed) is kept as-is since it's specialized for video
(combining temporal, height, width frequencies) and not directly supported by FlashInfer's
standard RoPE APIs.
"""

import logging
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_EXAMPLES_PYTORCH_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_PYTORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_PYTORCH_DIR))

from flashinfer_modules import (
    GEMMBackend,
    FlashInferAttentionDispatcher,
    FlashInferFP32LayerNorm,
    FlashInferFeedForward,
    FlashInferLinear,
    FlashInferRMSNorm,
    apply_rotary_emb,
    create_linear_layer,
    get_1d_rotary_pos_embed,
)


@dataclass
class WanTransformer3DConfig:
    """Configuration for WanTransformer3DModel.

    Added fields for FlashInfer optimizations:
    - gemm_backend: GEMM backend for linear layers (see GEMMBackend enum, default "torch")
    - online_act_quant: True for online activation scale computation, False for fixed default scale
    - attention_backend: "auto", "single", "cudnn", or "trtllm"
    - use_skip_softmax_sparse: Whether to use skip-softmax sparse attention (trtllm backend)
    - skip_softmax_threshold_scale_factor: Threshold scale factor for skip-softmax sparsity
    """

    patch_size: Tuple[int, ...] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: Optional[int] = None
    added_kv_proj_dim: Optional[int] = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: Optional[int] = None

    # FlashInfer optimization options. Both backend strings accept an optional
    # ``"-<kernel>"`` suffix forwarded to the corresponding FlashInfer API's
    # own ``backend`` kwarg — e.g. ``"fp4-cudnn"``, ``"fp4-cutlass"``,
    # ``"mxfp8-cute-dsl"``, ``"bmm_fp8-cublas"``, ``"single-fa3"``,
    # ``"single-fa2"``. See ``FlashInferLinear`` / ``FlashInferAttentionDispatcher``
    # in ``examples/pytorch/flashinfer_modules.py`` for the full list of bases
    # and the legal suffixes per kernel.
    gemm_backend: str = "torch"
    online_act_quant: bool = (
        True  # True: compute activation scale from data; False: use default scale
    )
    attention_backend: str = "auto"
    use_skip_softmax_sparse: bool = (
        False  # Enable skip-softmax sparse attention (trtllm only)
    )
    skip_softmax_threshold_scale_factor: float = 1.0  # Threshold scale factor for skip-softmax (higher = more sparse, less accurate)
    # Text encoder context length used to split image-vs-text tokens in I2V
    # cross-attention. The default of 512 matches Wan 2.x text encoders; users
    # with custom text encoders should set this to their own context length.
    text_encoder_context_length: int = 512

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


_FLASHINFER_ENV_OVERRIDES = {
    "gemm_backend": "FLASHINFER_GEMM_BACKEND",
    "online_act_quant": "FLASHINFER_ONLINE_ACT_QUANT",
    "use_skip_softmax_sparse": "FLASHINFER_USE_SKIP_SOFTMAX_SPARSE",
    "skip_softmax_threshold_scale_factor": "FLASHINFER_SKIP_SOFTMAX_THRESHOLD",
    "attention_backend": "FLASHINFER_ATTENTION_BACKEND",
}


def _env_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value (1/0, true/false, yes/no, on/off), got {value!r}."
    )


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


# Mapping of param-key infixes between diffusers' FeedForward (nn.ModuleList
# named ``net``) and FlashInferFeedForward (named submodules ``proj_up`` /
# ``proj_down``). Each tuple is (old_infix, new_infix); applied as a substring
# replace, so it transparently handles both ``ffn.net.0.proj`` (block FFN) and
# ``ff.net.0.proj`` (image-embedding FFN), and any future module that wraps a
# diffusers ``FeedForward`` inside another sub-module.
_FFN_KEY_REMAP: tuple[tuple[str, str], ...] = (
    (".net.0.proj.", ".proj_up."),
    (".net.2.", ".proj_down."),
)


def _remap_diffusers_state_dict(state_dict: dict) -> dict:
    """Rename diffusers FFN keys to FlashInferFeedForward's naming."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_infix, new_infix in _FFN_KEY_REMAP:
            if old_infix in new_key:
                new_key = new_key.replace(old_infix, new_infix)
        remapped[new_key] = value
    return remapped


def _config_to_flashinfer_config(
    config: Any, overrides: Optional[dict] = None
) -> WanTransformer3DConfig:
    overrides = dict(overrides or {})
    values = {
        field: _config_value(config, field, getattr(WanTransformer3DConfig, field))
        for field in WanTransformer3DConfig.__dataclass_fields__
        if field != "inner_dim"
    }

    for field, env_name in _FLASHINFER_ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        if field in {"online_act_quant", "use_skip_softmax_sparse"}:
            values[field] = _env_bool(env_name)
        elif field == "skip_softmax_threshold_scale_factor":
            values[field] = float(raw)
        else:
            values[field] = raw

    values.update({k: v for k, v in overrides.items() if v is not None})
    return WanTransformer3DConfig(**values)


class WanRotaryPosEmbed(nn.Module):
    """3D Rotary Position Embedding for video transformers.

    This is kept from the original implementation as it's specialized for video
    (combining temporal, height, width position encodings) and not directly
    supported by FlashInfer's standard RoPE APIs.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _batch_size, _num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )

        return freqs_cos, freqs_sin


class FlashInferWanAttention(nn.Module):
    """Wan attention module that delegates FlashInfer kernel dispatch."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        is_cross_attention: Optional[bool] = None,
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
        attention_backend: str = "auto",
        use_skip_softmax_sparse: bool = False,
        skip_softmax_threshold_scale_factor: float = 1.0,
        text_encoder_context_length: int = 512,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )
        self.text_encoder_context_length = text_encoder_context_length
        self.is_cross_attention = (
            is_cross_attention
            if is_cross_attention is not None
            else cross_attention_dim_head is not None
        )

        self.to_q = create_linear_layer(
            dim,
            self.inner_dim,
            bias=True,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.to_k = create_linear_layer(
            dim,
            self.kv_inner_dim,
            bias=True,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.to_v = create_linear_layer(
            dim,
            self.kv_inner_dim,
            bias=True,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.to_out = nn.ModuleList(
            [
                create_linear_layer(
                    self.inner_dim,
                    dim,
                    bias=True,
                    gemm_backend=gemm_backend,
                    online_act_quant=online_act_quant,
                ),
                nn.Dropout(dropout),
            ]
        )

        self.norm_q = FlashInferRMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.norm_k = FlashInferRMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = create_linear_layer(
                added_kv_proj_dim,
                self.inner_dim,
                bias=True,
                gemm_backend=gemm_backend,
                online_act_quant=online_act_quant,
            )
            self.add_v_proj = create_linear_layer(
                added_kv_proj_dim,
                self.inner_dim,
                bias=True,
                gemm_backend=gemm_backend,
                online_act_quant=online_act_quant,
            )
            self.norm_added_k = FlashInferRMSNorm(
                dim_head * heads, eps=eps, elementwise_affine=False
            )

        self.attention_dispatcher = FlashInferAttentionDispatcher(
            heads=heads,
            dim_head=dim_head,
            attention_backend=attention_backend,
            use_skip_softmax_sparse=use_skip_softmax_sparse,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if self.add_k_proj is not None and encoder_hidden_states is not None:
            image_context_length = (
                encoder_hidden_states.shape[1] - self.text_encoder_context_length
            )
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.norm_q(self.to_q(hidden_states))
        key = self.norm_k(self.to_k(encoder_hidden_states))
        value = self.to_v(encoder_hidden_states)

        batch_size = hidden_states.shape[0]
        seq_len_q = hidden_states.shape[1]
        seq_len_kv = encoder_hidden_states.shape[1]

        query = query.view(batch_size, seq_len_q, self.heads, self.dim_head)
        key = key.view(batch_size, seq_len_kv, self.heads, self.dim_head)
        value = value.view(batch_size, seq_len_kv, self.heads, self.dim_head)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        orig_dtype = query.dtype
        needs_cast = orig_dtype == torch.float32
        if needs_cast:
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = self.add_k_proj(encoder_hidden_states_img)
            value_img = self.add_v_proj(encoder_hidden_states_img)
            key_img = self.norm_added_k(key_img)

            seq_len_img = encoder_hidden_states_img.shape[1]
            key_img = key_img.view(batch_size, seq_len_img, self.heads, self.dim_head)
            value_img = value_img.view(
                batch_size, seq_len_img, self.heads, self.dim_head
            )
            if needs_cast:
                key_img = key_img.to(torch.bfloat16)
                value_img = value_img.to(torch.bfloat16)

            img_backend, _ = self.attention_dispatcher._resolve_attention_backend(
                batch_size, query.device
            )
            hidden_states_img = self.attention_dispatcher._dispatch_attention(
                img_backend,
                query,
                key_img,
                value_img,
                batch_size,
                seq_len_q,
                seq_len_img,
                use_sparse=False,
            )

        attn_backend, use_sparse = self.attention_dispatcher._resolve_attention_backend(
            batch_size, query.device
        )
        hidden_states = self.attention_dispatcher._dispatch_attention(
            attn_backend,
            query,
            key,
            value,
            batch_size,
            seq_len_q,
            seq_len_kv,
            use_sparse=use_sparse,
        )

        if needs_cast:
            hidden_states = hidden_states.to(orig_dtype)
            if hidden_states_img is not None:
                hidden_states_img = hidden_states_img.to(orig_dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class FlashInferWanTransformerBlock(nn.Module):
    """Transformer block using FlashInfer optimizations.

    Args:
        dim: Hidden dimension
        ffn_dim: Feed-forward hidden dimension
        num_heads: Number of attention heads
        qk_norm: QK normalization type
        cross_attn_norm: Whether to use cross attention normalization
        eps: Epsilon for normalization
        added_kv_proj_dim: Dimension for additional KV projection
        gemm_backend: GEMM backend for linear layers
        use_skip_softmax_sparse: Whether to use skip-softmax sparse attention
        skip_softmax_threshold_scale_factor: Threshold scale factor for skip-softmax
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
        attention_backend: str = "auto",
        use_skip_softmax_sparse: bool = False,
        skip_softmax_threshold_scale_factor: float = 1.0,
        text_encoder_context_length: int = 512,
    ):
        super().__init__()

        # 1. Self-attention (with optional skip-softmax sparse attention)
        self.norm1 = FlashInferFP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = FlashInferWanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
            attention_backend=attention_backend,
            use_skip_softmax_sparse=use_skip_softmax_sparse,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            text_encoder_context_length=text_encoder_context_length,
        )

        # 2. Cross-attention (always full attention, no sparse attention)
        self.attn2 = FlashInferWanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
            attention_backend=attention_backend,
            use_skip_softmax_sparse=False,  # No sparse attention for cross-attention
            text_encoder_context_length=text_encoder_context_length,
        )
        self.norm2 = (
            FlashInferFP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward with FlashInfer GELU fusion and optimized GEMM
        self.ffn = FlashInferFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu-approximate",
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.norm3 = FlashInferFP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)

        return hidden_states


class WanImageEmbedding(nn.Module):
    """Image embedding module with FlashInfer GEMM support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pos_embed_seq_len: Optional[int] = None,
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
    ):
        super().__init__()

        self.norm1 = FlashInferFP32LayerNorm(in_features)
        self.ff = FlashInferFeedForward(
            in_features,
            out_features,
            activation_fn="gelu",
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.norm2 = FlashInferFP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_seq_len, in_features)
            )
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            _batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class Timesteps(nn.Module):
    """Timestep embedding."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            0, half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = timesteps[:, None].float() * exponent[None, :].exp()
        # Match diffusers get_timestep_embedding: [sin, cos] BEFORE the flip,
        # so flip_sin_to_cos=True yields [cos, sin]. (Starting from [cos, sin]
        # here would make the flip produce [sin, cos] -- swapped halves that
        # silently scramble the timestep conditioning.)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding projection with FlashInfer GEMM support."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
    ):
        super().__init__()
        self.linear_1 = create_linear_layer(
            in_channels,
            time_embed_dim,
            bias=True,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.act = nn.SiLU()
        self.linear_2 = create_linear_layer(
            time_embed_dim,
            time_embed_dim,
            bias=True,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaTextProjection(nn.Module):
    """Text projection with GELU-tanh activation and FlashInfer GEMM support."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        act_fn: str = "gelu_tanh",
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
    ):
        super().__init__()
        self.linear_1 = create_linear_layer(
            in_features,
            hidden_size,
            bias=True,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.linear_2 = create_linear_layer(
            hidden_size,
            hidden_size,
            bias=True,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    """Combined time, text, and image embedding with FlashInfer GEMM support."""

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
        gemm_backend: str = "torch",
        online_act_quant: bool = True,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.act_fn = nn.SiLU()
        self.time_proj = create_linear_layer(
            dim,
            time_proj_dim,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim,
            dim,
            act_fn="gelu_tanh",
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim,
                dim,
                pos_embed_seq_len=pos_embed_seq_len,
                gemm_backend=gemm_backend,
                online_act_quant=online_act_quant,
            )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None and self.image_embedder is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class FlashInferWanTransformer3DModel(nn.Module):
    """
    FlashInfer-optimized Wan Transformer for video generation.

    This is an inference-optimized implementation of WanTransformer3DModel
    using FlashInfer kernels for:
    - RMSNorm normalization
    - Scaled dot-product attention
    - Fused GELU activations
    - Optimized GEMM backends (bf16/fp8/fp4) for linear layers
    - Optional skip-softmax sparse attention

    Args:
        config: WanTransformer3DConfig or compatible config object

    Config options for optimization:
        gemm_backend: "torch", "bf16", "fp8", "fp8_sm90", "bmm_fp8", "fp8_groupwise", "fp8_blockscaled", "batch_deepgemm_fp8", "fp4", etc.
        use_skip_softmax_sparse: Whether to use skip-softmax sparse attention
        skip_softmax_threshold_scale_factor: Threshold scale factor for skip-softmax
    """

    def __init__(self, config: Union[WanTransformer3DConfig, Any]):
        super().__init__()

        # Normalize external configs once. Pre-normalized local configs should
        # keep their explicit values.
        if not isinstance(config, WanTransformer3DConfig):
            config = _config_to_flashinfer_config(config)
        self.config = config

        patch_size = getattr(config, "patch_size", (1, 2, 2))
        num_attention_heads = getattr(config, "num_attention_heads", 40)
        attention_head_dim = getattr(config, "attention_head_dim", 128)
        in_channels = getattr(config, "in_channels", 16)
        out_channels = getattr(config, "out_channels", in_channels)
        text_dim = getattr(config, "text_dim", 4096)
        freq_dim = getattr(config, "freq_dim", 256)
        ffn_dim = getattr(config, "ffn_dim", 13824)
        num_layers = getattr(config, "num_layers", 40)
        cross_attn_norm = getattr(config, "cross_attn_norm", True)
        qk_norm = getattr(config, "qk_norm", "rms_norm_across_heads")
        eps = getattr(config, "eps", 1e-6)
        image_dim = getattr(config, "image_dim", None)
        added_kv_proj_dim = getattr(config, "added_kv_proj_dim", None)
        rope_max_seq_len = getattr(config, "rope_max_seq_len", 1024)
        pos_embed_seq_len = getattr(config, "pos_embed_seq_len", None)

        # FlashInfer optimization options
        gemm_backend = getattr(config, "gemm_backend", "torch")
        online_act_quant = getattr(config, "online_act_quant", True)
        attention_backend = getattr(config, "attention_backend", "auto")
        use_skip_softmax_sparse = getattr(config, "use_skip_softmax_sparse", False)
        skip_softmax_threshold_scale_factor = getattr(
            config, "skip_softmax_threshold_scale_factor", 1.0
        )
        text_encoder_context_length = getattr(
            config, "text_encoder_context_length", 512
        )

        inner_dim = num_attention_heads * attention_head_dim

        # Store config values for forward pass
        self.patch_size = patch_size
        self.gemm_backend = gemm_backend
        self.online_act_quant = online_act_quant
        self.attention_backend = attention_backend
        self.use_skip_softmax_sparse = use_skip_softmax_sparse
        self.skip_softmax_threshold_scale_factor = skip_softmax_threshold_scale_factor

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings with optimized GEMM
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )

        # 3. Transformer blocks with FlashInfer optimizations
        self.blocks = nn.ModuleList(
            [
                FlashInferWanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    gemm_backend=gemm_backend,
                    online_act_quant=online_act_quant,
                    attention_backend=attention_backend,
                    use_skip_softmax_sparse=use_skip_softmax_sparse,
                    skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                    text_encoder_context_length=text_encoder_context_length,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FlashInferFP32LayerNorm(
            inner_dim, eps, elementwise_affine=False
        )
        self.proj_out = create_linear_layer(
            inner_dim,
            out_channels * math.prod(patch_size),
            gemm_backend=gemm_backend,
            online_act_quant=online_act_quant,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

    def prepare_weights(self):
        """Prepare all linear layer weights for optimized inference.

        Call this after loading pretrained weights to convert weights
        to the appropriate format for the selected GEMM backend.
        """
        for module in self.modules():
            if isinstance(module, FlashInferLinear):
                module.prepare_weights()

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def cache_context(self, _name: str):
        """Compatibility shim for diffusers pipelines that cache cond/uncond calls."""
        return nullcontext()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[dict] = None,
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass of the FlashInfer-optimized Wan Transformer.

        Args:
            hidden_states: Input video latents, shape (B, C, T, H, W)
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings
            encoder_hidden_states_image: Optional image embeddings for I2V
            return_dict: Whether to return a dict or tuple

        Returns:
            Output tensor or dict with 'sample' key
        """
        batch_size, _num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    use_reentrant=False,
                )
        else:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (
                self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (
                self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return {"sample": output}

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load model from a pretrained HuggingFace checkpoint.

        Note: This requires the original diffusers model to be installed
        for config loading.
        """
        flashinfer_kwargs = {}
        for name in WanTransformer3DConfig.__dataclass_fields__:
            if name in kwargs:
                flashinfer_kwargs[name] = kwargs.pop(name)

        try:
            from diffusers.models.transformers.transformer_wan import (
                WanTransformer3DModel as OriginalModel,
            )
        except ImportError as e:
            raise ImportError(
                "Please install diffusers to load from pretrained: "
                "pip install diffusers"
            ) from e

        # Load original model to get config and weights
        original_model = OriginalModel.from_pretrained(model_path, **kwargs)

        # Create FlashInfer model with same config
        config = _config_to_flashinfer_config(original_model.config, flashinfer_kwargs)
        model = cls(config)

        # Diffusers' FeedForward stores the two linear layers under
        # ``net.0.proj`` (GELU/GEGLU's projection) and ``net.2`` (output linear),
        # while FlashInferFeedForward names them ``proj_up`` and ``proj_down``.
        # Without renaming, load_state_dict(..., strict=False) silently drops
        # all FFN weights and the model produces meaningless outputs.
        state_dict = _remap_diffusers_state_dict(original_model.state_dict())

        # Copy weights. ``strict=False`` is required because some FlashInfer
        # GEMM backends (FP8, FP4, MXFP8, ...) register additional buffers
        # that don't exist in the diffusers model. We log any unmatched keys
        # so users can spot weight-shape mismatches early.
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.info(
                "from_pretrained: %d missing keys (likely FlashInfer-only buffers): %s",
                len(result.missing_keys),
                result.missing_keys[:10]
                + (["..."] if len(result.missing_keys) > 10 else []),
            )
        if result.unexpected_keys:
            logger.warning(
                "from_pretrained: %d unexpected keys (present in checkpoint but not in model): %s",
                len(result.unexpected_keys),
                result.unexpected_keys[:10]
                + (["..."] if len(result.unexpected_keys) > 10 else []),
            )

        return model


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Load a Hugging Face Wan transformer checkpoint and run FlashInfer forward."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Hugging Face repo id or local checkpoint path.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="transformer",
        help="Subfolder containing the WanTransformer3DModel checkpoint.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model and input dtype.",
    )
    parser.add_argument(
        "--gemm-backend",
        type=str,
        default=os.getenv("FLASHINFER_GEMM_BACKEND", "torch"),
        help=(
            "GEMM backend for linear layers. Base names: "
            + ", ".join(b.value for b in GEMMBackend)
            + ". An optional '-<kernel>' suffix is forwarded to the "
            "underlying FlashInfer kernel's backend kwarg (e.g. 'fp4-cudnn', "
            "'fp4-cutlass', 'mxfp8-cute-dsl', 'bmm_fp8-cublas')."
        ),
    )
    parser.add_argument(
        "--offline-act-quant",
        action="store_true",
        help="Use offline (fixed default) activation quantization instead of online",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=os.getenv("FLASHINFER_ATTENTION_BACKEND", "auto"),
        help=(
            "Attention backend. Base names: auto, single, cudnn, trtllm, torch "
            "(torch is the F.scaled_dot_product_attention fallback). An "
            "optional '-<kernel>' suffix on 'single' is forwarded to "
            "single_prefill_with_kv_cache's backend kwarg, e.g. 'single-fa3', "
            "'single-fa2', 'single-cudnn'."
        ),
    )
    parser.add_argument(
        "--skip-softmax-sparse",
        action="store_true",
        help="Enable skip-softmax sparse attention (trtllm backend)",
    )
    parser.add_argument(
        "--skip-softmax-threshold",
        type=float,
        default=1.0,
        help="Threshold scale factor for skip-softmax (higher = more sparse)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument(
        "--text-seq-len",
        type=int,
        default=512,
        help="Synthetic encoder_hidden_states sequence length.",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=1)
    parser.add_argument(
        "--prepare-weights",
        action="store_true",
        help="Pre-convert FlashInferLinear weights for the selected GEMM backend.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Capture and replay the forward pass with a CUDA graph. Eliminates "
            "per-layer kernel-launch overhead at the cost of fixing input shapes."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Wrap the forward pass with torch.compile. Fuses elementwise pre/post "
            "processing (activation quant, bias add, dtype casts) around FlashInfer "
            "kernels. Mutually exclusive with --cuda-graph (use --torch-compile-mode "
            "reduce-overhead to also get CUDA-graph replay)."
        ),
    )
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode (default | reduce-overhead | max-autotune).",
    )
    parser.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile. Likely to fail on FlashInfer custom ops.",
    )
    parser.add_argument(
        "--torch-compile-dynamic",
        action="store_true",
        help="Pass dynamic=True to torch.compile. Default (False) specializes on shape.",
    )
    args = parser.parse_args()

    # --cuda-graph (manual outer capture) and --torch-compile (Inductor
    # fusion) compose, provided torch.compile isn't itself inserting CUDA
    # graphs around its own captured regions. The "reduce-overhead" /
    # "max-autotune" modes do exactly that, which conflicts with the outer
    # capture (and runs into stale-pointer errors when FlashInfer kernels
    # cause graph breaks between compiled regions). Force the non-cudagraph
    # mode in that case.
    if (
        args.cuda_graph
        and args.torch_compile
        and args.torch_compile_mode
        in (
            "reduce-overhead",
            "max-autotune",
        )
    ):
        raise ValueError(
            f"--cuda-graph cannot combine with --torch-compile-mode "
            f"{args.torch_compile_mode!r} (which already manages its own CUDA "
            f"graph capture). Use --torch-compile-mode default or "
            f"max-autotune-no-cudagraphs alongside --cuda-graph."
        )

    if args.benchmark_iters < 1:
        raise ValueError("--benchmark-iters must be >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this FlashInfer example.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading checkpoint: {args.model_id} subfolder={args.subfolder!r}")
    print(
        "FlashInfer config: "
        f"gemm_backend={args.gemm_backend}, "
        f"online_act_quant={not args.offline_act_quant}, "
        f"attention_backend={args.attention_backend}, "
        f"use_skip_softmax_sparse={args.skip_softmax_sparse}"
    )

    load_kwargs = {
        "subfolder": args.subfolder,
        "torch_dtype": dtype,
        "gemm_backend": args.gemm_backend,
        "online_act_quant": not args.offline_act_quant,
        "attention_backend": args.attention_backend,
        "use_skip_softmax_sparse": args.skip_softmax_sparse,
        "skip_softmax_threshold_scale_factor": args.skip_softmax_threshold,
    }
    if args.revision is not None:
        load_kwargs["revision"] = args.revision
    if args.variant is not None:
        load_kwargs["variant"] = args.variant

    model = FlashInferWanTransformer3DModel.from_pretrained(
        args.model_id, **load_kwargs
    )
    model = model.to(device="cuda", dtype=dtype).eval()
    if args.prepare_weights:
        model.prepare_weights()

    # Print model info
    linear_count = sum(1 for m in model.modules() if isinstance(m, FlashInferLinear))
    torch_linear_count = sum(1 for m in model.modules() if type(m) is nn.Linear)
    print(f"FlashInferLinear layers: {linear_count}")
    print(f"Standard nn.Linear layers: {torch_linear_count}")

    if args.torch_compile:
        print(
            f"torch.compile: mode={args.torch_compile_mode}, "
            f"fullgraph={args.torch_compile_fullgraph}, dynamic={args.torch_compile_dynamic}"
        )
        # FlashInfer kernel wrappers are not registered as custom ops, so dynamo
        # will graph-break around them; that's fine — the wins we want here are
        # fusing the per-layer activation-quant / bias-add / dtype-cast prologue
        # that BENCHMARK.md identified as the wrapper bottleneck. Compile each
        # transformer block individually so the same compiled artifact is reused
        # across all 30/40 blocks (one compile vs. 30/40), and keep the outer
        # forward in eager so RoPE / patch_embedding / output projection don't
        # force a recompile when only a block changes.
        compile_kwargs = {
            "mode": args.torch_compile_mode,
            "fullgraph": args.torch_compile_fullgraph,
            "dynamic": args.torch_compile_dynamic,
        }
        for i, block in enumerate(model.blocks):
            model.blocks[i] = torch.compile(block, **compile_kwargs)

    in_channels = model.config.in_channels
    text_dim = model.config.text_dim

    hidden_states = torch.randn(
        args.batch_size,
        in_channels,
        args.num_frames,
        args.height,
        args.width,
        device="cuda",
        dtype=dtype,
    )
    timestep = torch.randint(0, 1000, (args.batch_size,), device="cuda")
    encoder_hidden_states = torch.randn(
        args.batch_size, args.text_seq_len, text_dim, device="cuda", dtype=dtype
    )

    if args.cuda_graph:
        # CUDA-graph path: stream-warmup so JIT kernels are compiled and any
        # autotune passes settle, capture once, then time replays.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(max(args.warmup_iters, 3)):
                _ = model(
                    hidden_states, timestep, encoder_hidden_states, return_dict=False
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            graph_output = model(
                hidden_states, timestep, encoder_hidden_states, return_dict=False
            )

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.benchmark_iters):
            graph.replay()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        output = graph_output
    else:
        with torch.no_grad():
            for _ in range(args.warmup_iters):
                _ = model(
                    hidden_states, timestep, encoder_hidden_states, return_dict=False
                )

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(args.benchmark_iters):
                output = model(
                    hidden_states, timestep, encoder_hidden_states, return_dict=False
                )
        torch.cuda.synchronize()
        elapsed = time.time() - start

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output[0].shape}")
    if args.cuda_graph:
        mode_tag = " (cuda graph)"
    elif args.torch_compile:
        mode_tag = f" (torch.compile {args.torch_compile_mode})"
    else:
        mode_tag = ""
    print(
        f"Average time per forward pass{mode_tag}: "
        f"{elapsed / args.benchmark_iters * 1000:.2f} ms"
    )
