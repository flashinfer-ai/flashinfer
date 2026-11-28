# Copyright (c) 2026 by FlashInfer team.
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

"""Forward-input validation and FP4 activation staging for the Cake SM120 W4A4 MegaMoE backends.

Kernel input contract (per rank, ``0 <= rows <= max_tokens_per_rank``):

* ``x``  ``uint8 [rows, H // 2]``  packed E2M1 (low nibble first);
* ``x_sf`` ``uint8 [rows, H // B]`` scale bytes, one per ``B`` values: E4M3 with ``B = 16``
  (NVFP4, ``sm120_nvfp4_nvfp4_bf16_cake``) or UE8M0 with ``B = 32`` (MXFP4,
  ``sm120_mxfp4_mxfp4_bf16_cake``);
* routing as in the MXFP8 backend (``topk_idx`` int64 ``-1`` = masked, ``topk_weights`` float32).

With ``quantize_input`` the BF16/FP16/FP32 activation is quantized by the codec's rule
(:func:`~.cake_nvfp4_weights.nvfp4_quantize_torch` as ``x * input_norm_const``, or
``mxfp4_quantize_torch``); otherwise the caller passes packed ``uint8`` / ``float4_e2m1fn_x2``
``[T, H // 2]`` and the scales ``[T, H // B]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..mxfp4_mxfp8_bf16_cake.cake_staging import (
    MoEEpConfigError,
    stage_inputs,
    validate_forward_inputs,
)
from .cake_nvfp4_quant import _HAVE_TRITON, nvfp4_quantize_fused
from .cake_nvfp4_weights import NVFP4_BLOCK, nvfp4_quantize_torch

# The fused Triton quantizers when Triton is available, else the pure-torch references (bit-identical).
FUSED_QUANT = _HAVE_TRITON


@dataclass(frozen=True)
class Fp4ActivationCodec:
    """How one FP4 activation format is validated, quantized and staged."""

    name: str
    block: int  # values per scale byte
    scale_dtypes: tuple[torch.dtype, ...]
    # (values, norm) -> (packed codes, scale bytes), the reference rule
    quantize_torch: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    # (values, norm, codes=, sf=) -> in-place fused twin (BF16/FP32 CUDA input)
    quantize_fused: Callable[..., tuple[torch.Tensor, torch.Tensor]]


NVFP4_CODEC = Fp4ActivationCodec(
    name="NVFP4",
    block=NVFP4_BLOCK,
    scale_dtypes=(torch.uint8, torch.float8_e4m3fn),
    quantize_torch=nvfp4_quantize_torch,
    quantize_fused=nvfp4_quantize_fused,
)


def validate_fp4_forward_inputs(
    codec: Fp4ActivationCodec,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None,
) -> None:
    if quantize_input:
        validate_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            fleet_params,
            top_k=top_k,
            quantize_input=True,
            scales=None,
        )
        return
    tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    if tuple(hidden_states.shape) != (
        tokens,
        hidden // 2,
    ) or hidden_states.dtype not in (
        torch.uint8,
        torch.float4_e2m1fn_x2,
    ):
        raise MoEEpConfigError(
            f"pre-quantized {codec.name} activation must be packed uint8/float4_e2m1fn_x2 of shape "
            f"({tokens}, {hidden // 2})"
        )
    if (
        scales is None
        or scales.dtype not in codec.scale_dtypes
        or tuple(scales.shape) != (tokens, hidden // codec.block)
    ):
        raise MoEEpConfigError(
            f"pre-quantized {codec.name} activation requires {codec.scale_dtypes[-1]} scales of shape "
            f"({tokens}, {hidden // codec.block})"
        )
    if tuple(topk_ids.shape) != (tokens, top_k) or topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("routing tensors do not match token count/top-k")
    if topk_ids.dtype != torch.int64 or topk_weights.dtype != torch.float32:
        raise MoEEpConfigError("topk_ids must be int64 and topk_weights float32")
    if tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError("token count exceeds max_tokens_per_rank")


def stage_fp4_inputs(
    codec: Fp4ActivationCodec,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    quantize_input: bool,
    scales: torch.Tensor | None,
    input_norm_const: float = 1.0,
) -> int:
    """Copy/quantize one step's inputs into the workspace's FP4 staging buffers; returns the row count."""

    if quantize_input:
        if (
            FUSED_QUANT
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float32)
        ):
            # the fused quantizer writes straight into the workspace's staging rows
            rows = int(hidden_states.shape[0])
            codec.quantize_fused(
                hidden_states, float(input_norm_const), codes=x, sf=x_sf
            )
            topk_idx_out[:rows].copy_(topk_ids)
            topk_weights_out[:rows].copy_(topk_weights)
            return rows
        codes, sf = codec.quantize_torch(hidden_states, input_norm_const)
    else:
        codes, sf = hidden_states.view(torch.uint8), scales.view(torch.uint8)  # type: ignore[union-attr]
    return stage_inputs(
        codes,
        topk_ids,
        topk_weights,
        x_fp8=x,
        x_sf=x_sf,
        topk_idx_out=topk_idx_out,
        topk_weights_out=topk_weights_out,
        quantize_input=False,
        scales=sf,
    )


__all__ = [
    "FUSED_QUANT",
    "NVFP4_CODEC",
    "Fp4ActivationCodec",
    "stage_fp4_inputs",
    "validate_fp4_forward_inputs",
]
