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

import torch

from ..jit.cake_minimax_h3_bf16_pre_attention import (
    gen_minimax_h3_bf16_pre_attention_module,
)
from ..utils import register_custom_op, register_fake_op


@functools.cache
def _get_module():
    return gen_minimax_h3_bf16_pre_attention_module().build_and_load()


@register_custom_op(
    "flashinfer::minimax_h3_bf16_pre_attention",
    mutates_args=("workspace", "out"),
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
    rope_positions: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    m: int,
    ulysses_degree: int,
    eps: float,
    qk_eps: float,
) -> None:
    # TVM-FFI argument order of the generated host tail: the AdaLN row count
    # and row stride are read from the ``adaln_scale`` / ``adaln_shift``
    # TensorViews, the RoPE cache row count from ``rope_cos_sin``; the host
    # tail launches the norm/AdaLN kernel into ``workspace`` and the QKV GEMM
    # (with the fused Q/K RMSNorm, RoPE and pack epilogue) into ``out``.
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
        rope_positions,
        workspace,
        out,
        m,
        ulysses_degree,
        eps,
        qk_eps,
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
    rope_positions: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    m: int,
    ulysses_degree: int,
    eps: float,
    qk_eps: float,
) -> None:
    pass


def get_minimax_h3_bf16_pre_attention_backend(*, backend: str):
    if backend == "cake":
        return _minimax_h3_bf16_pre_attention_impl
    raise ValueError(f"Unsupported MiniMax-H3 BF16 pre-attention backend: {backend}")
