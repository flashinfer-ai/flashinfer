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
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from ..jit import gen_alphamoe_sm100_module
from ..utils import register_custom_op


def _interleave_gated_rows(tensor: torch.Tensor, rep: int) -> torch.Tensor:
    """Interleave the gate half and the up half of dim 1 in ``rep``-row chunks."""
    num_experts, rows, columns = tensor.shape
    if rows % (2 * rep):
        raise ValueError(f"row count {rows} must be divisible by {2 * rep}")
    gate = tensor[:, : rows // 2].reshape(num_experts, rows // (2 * rep), rep, columns)
    up = tensor[:, rows // 2 :].reshape(num_experts, rows // (2 * rep), rep, columns)
    return (
        torch.stack((gate, up), dim=2).reshape(num_experts, rows, columns).contiguous()
    )


def alphamoe_interleave_gated_weights(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interleave gate/up weight rows into the alphamoe_sm100 device layout.

    :func:`alphamoe_fp8_block_scale_aligned_moe` consumes gemm1 (gate+up)
    weights whose row axis alternates eight gate rows with eight up rows, so
    one 128-row TMA box carries matching gate and up features. This offline
    helper converts the conventional ``[gate; up]``-stacked layout:

    - ``gemm1_weights`` ``(num_experts, 2 * intermediate_size, hidden_size)``
      FP8 with gate rows first: interleaved in eight-row chunks.
    - ``gemm1_weights_scale`` ``(num_experts, 2 * intermediate_size / 128,
      hidden_size / 128)`` float32 block scales: interleaved one row at a
      time (each scale row covers 128 weight rows).

    Returns the ``(weights, scales)`` pair in device layout. Run once at
    weight-load time; the kernel must never see un-interleaved weights.
    """
    return (
        _interleave_gated_rows(gemm1_weights, rep=8),
        _interleave_gated_rows(gemm1_weights_scale, rep=1),
    )


@functools.cache
def get_alphamoe_sm100_module() -> SimpleNamespace:
    module = gen_alphamoe_sm100_module().build_and_load()

    @register_custom_op(
        "flashinfer::alphamoe_fp8_block_scale_aligned_moe",
        mutates_args=["out"],
    )
    def fp8_block_scale_aligned_moe(
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
        top_k: int,
        block_m: int,
        routed_scaling_factor: float,
    ) -> None:
        module.fp8_block_scale_aligned_moe_op(
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            out,
            top_k,
            block_m,
            routed_scaling_factor,
        )

    return SimpleNamespace(fp8_block_scale_aligned_moe=fp8_block_scale_aligned_moe)


def alphamoe_fp8_block_scale_aligned_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    top_k: int,
    block_m: int = 8,
    routed_scaling_factor: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Fused W8A8 block-scale MoE over a pre-aligned routing plan (SM100/SM103).

    Runs the generated Alpha-MoE megakernel: routed gate/up projection,
    SwiGLU with per-token FP8 requantization of the intermediate, down
    projection, routing-weight/scaling application, and asynchronous BF16
    reduce-add into ``out`` — one kernel, no global intermediate tensor.

    The routing plan is the vLLM/SGLang ``moe_align_block_size`` contract:
    flattened ``token * top_k + route`` positions grouped by expert, padded
    per expert to a multiple of ``block_m`` with the sentinel
    ``num_tokens * top_k``. ``expert_ids`` carries one expert index per
    ``block_m``-sized block, and ``num_tokens_post_padded`` is a device-side
    scalar naming the valid plan extent; blocks past it are skipped, so
    worst-case-sized plan buffers are fine.

    Parameters
    ----------
    hidden_states: torch.Tensor
        ``(num_tokens, hidden_size)`` ``float8_e4m3fn`` activations,
        quantized per token in groups of 128 along ``hidden_size``. Only the
        innermost stride must be 1 (row-sliced activations are accepted).
    hidden_states_scale: torch.Tensor
        ``(num_tokens, hidden_size // 128)`` float32 per-token-group scales.
    gemm1_weights: torch.Tensor
        ``(num_experts, 2 * intermediate_size, hidden_size)``
        ``float8_e4m3fn`` gate+up weights **in the interleaved device
        layout** produced by :func:`alphamoe_interleave_gated_weights`.
        ``2 * intermediate_size`` must be a multiple of 256 and
        ``hidden_size`` a multiple of 128.
    gemm1_weights_scale: torch.Tensor
        ``(num_experts, 2 * intermediate_size // 128, hidden_size // 128)``
        float32 128x128 block scales, interleaved by the same helper.
    gemm2_weights: torch.Tensor
        ``(num_experts, hidden_size, intermediate_size)`` ``float8_e4m3fn``
        down weights (natural layout, no interleaving).
    gemm2_weights_scale: torch.Tensor
        ``(num_experts, hidden_size // 128, intermediate_size // 128)``
        float32 128x128 block scales.
    sorted_token_ids: torch.Tensor
        int32 ``moe_align_block_size`` plan positions (see above).
    expert_ids: torch.Tensor
        int32, one expert index per ``block_m`` plan entries.
    num_tokens_post_padded: torch.Tensor
        int32 device tensor with one element: the valid plan extent.
    topk_weights: torch.Tensor
        ``(num_tokens, top_k)`` float32 routing weights.
    top_k: int
        Routes per token.
    block_m: int
        Routing-plan block size; a positive multiple of 8. Defaults to 8.
    routed_scaling_factor: float
        Extra scalar applied to every routed contribution. Defaults to 1.0.
    out: Optional[torch.Tensor]
        ``(num_tokens, hidden_size)`` bfloat16 accumulator. When omitted, a
        zeroed tensor is allocated and returned. When provided, the kernel
        **accumulates into it** (BF16 reduce-add); the caller is responsible
        for zeroing it (or seeding it with the intended initial values)
        before the call.

    Returns
    -------
    out: torch.Tensor
        The ``(num_tokens, hidden_size)`` bfloat16 accumulator.

    Notes
    -----
    Requires an SM100/SM103 (B200/B300 class) device; the kernel is a frozen
    generated Loom schedule (see ``csrc/alphamoe_sm100/``).
    """
    if out is None:
        out = torch.zeros(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
    get_alphamoe_sm100_module().fp8_block_scale_aligned_moe(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        routed_scaling_factor,
    )
    return out
