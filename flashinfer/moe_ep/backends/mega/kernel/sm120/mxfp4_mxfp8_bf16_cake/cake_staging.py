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

"""Forward-input validation and activation staging for the Cake SM120 MegaMoE.

Kernel input contract (per rank, ``0 <= rows <= max_tokens_per_rank``; ranks may
hold different row counts in the same call, including zero, as long as one rank
holds a token):

* ``x_fp8``  ``uint8 [rows, H]``      e4m3 codes;
* ``x_sf``   ``uint8 [rows, H // 32]`` UE8M0 exponents, one per 32-wide K block;
* ``topk_idx`` ``int64 [rows, top_k]`` global expert ids, ``-1`` = masked route;
* ``topk_weights`` ``float32 [rows, top_k]`` (already normalized/scaled by the caller).

A call in which no rank holds a token is skipped by every rank.
"""

from __future__ import annotations

import torch

try:  # imported as part of flashinfer
    from ......core.validation.common import (
        MoEEpConfigError,
        validate_mega_forward_inputs,
    )
except ImportError:  # host-only unit tests

    class MoEEpConfigError(ValueError):  # type: ignore[no-redef]
        pass

    validate_mega_forward_inputs = None  # type: ignore[assignment]

MASKED_ROUTE = -1
K_BLOCK = 32
FP8_MAX = 448.0


def quantize_activations_k32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16/FP32 ``[rows, H]`` -> (e4m3 codes ``uint8 [rows, H]``, UE8M0 ``uint8 [rows, H // 32]``).

    The scale of each 32-wide block is the UE8M0 power of two ``>= amax / 448``
    (round the FP32 ratio up to the next exponent); values are divided by
    that scale and rounded to e4m3.  This is the exact rule the kernel uses
    for its in-kernel requantization, so a pure-torch reference built on the
    same helper reproduces the kernel's operand values.
    """

    if x.dim() != 2 or x.shape[1] % K_BLOCK:
        raise ValueError(
            f"activations must be [rows, H] with H % {K_BLOCK} == 0, got {tuple(x.shape)}"
        )
    rows, hidden = (int(v) for v in x.shape)
    grouped = x.float().reshape(rows, hidden // K_BLOCK, K_BLOCK)
    amax = grouped.abs().amax(dim=2)
    bits = (amax / FP8_MAX).contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)).clamp(
        min=0, max=254
    )
    inverse = ((254 - exponent) << 23).view(torch.float32)
    codes = (grouped * inverse.unsqueeze(2)).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8).reshape(rows, hidden), exponent.to(torch.uint8)


def dequantize_activations_k32(
    codes: torch.Tensor, exponents: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`quantize_activations_k32` (float32 output)."""

    scales = (exponents.to(torch.int32) << 23).view(torch.float32)
    return codes.view(torch.float8_e4m3fn).float() * scales.repeat_interleave(
        K_BLOCK, dim=1
    )


def validate_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None,
) -> None:
    if topk_ids.dtype != torch.int64:
        raise MoEEpConfigError(
            f"topk_ids must be int64 (-1 = masked route), got {topk_ids.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        raise MoEEpConfigError(
            f"topk_weights must be float32, got {topk_weights.dtype}"
        )
    if quantize_input:
        if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise MoEEpConfigError(
                f"hidden_states must be bf16/fp16/fp32 when quantize_input=True, got {hidden_states.dtype}"
            )
        if validate_mega_forward_inputs is not None and hidden_states.shape[0] > 0:
            validate_mega_forward_inputs(
                hidden_states,
                topk_ids,
                topk_weights,
                fleet_params,
                top_k=top_k,
                quantize_input=True,
            )
        return
    tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    if tuple(hidden_states.shape) != (tokens, hidden) or hidden_states.dtype not in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        raise MoEEpConfigError(
            f"pre-quantized activation must be uint8/e4m3 codes with shape ({tokens}, {hidden})"
        )
    if (
        scales is None
        or scales.dtype not in (torch.uint8, torch.float8_e8m0fnu)
        or tuple(scales.shape) != (tokens, hidden // K_BLOCK)
    ):
        raise MoEEpConfigError(
            f"pre-quantized activation requires UE8M0 scales of shape ({tokens}, {hidden // K_BLOCK})"
        )
    if tuple(topk_ids.shape) != (tokens, top_k) or topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("routing tensors do not match token count/top-k")
    if tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError("token count exceeds max_tokens_per_rank")


def stage_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    quantize_input: bool,
    scales: torch.Tensor | None,
) -> int:
    """Copy/quantize one step's inputs into the workspace's fixed-capacity buffers.

    Returns the staged row count.  Runs on the current stream; no allocation
    beyond the quantization temporaries when ``quantize_input`` is set.
    """

    rows = int(hidden_states.shape[0])
    if quantize_input:
        codes, exponents = quantize_activations_k32(hidden_states)
    else:
        codes, exponents = hidden_states.view(torch.uint8), scales.view(torch.uint8)  # type: ignore[union-attr]
    x_fp8[:rows].copy_(codes)
    x_sf[:rows].copy_(exponents)
    topk_idx_out[:rows].copy_(topk_ids)
    topk_weights_out[:rows].copy_(topk_weights)
    return rows


__all__ = [
    "MASKED_ROUTE",
    "dequantize_activations_k32",
    "quantize_activations_k32",
    "stage_inputs",
    "validate_forward_inputs",
]
