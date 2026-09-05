# Copyright (c) 2025 by FlashInfer team.
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
"""High-level MXFP8 activation x MXFP4 weight fused-MoE API for SM12x (SM120/SM121)."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ...api_logging import flashinfer_api
from ...tllm_enums import DEFAULT_SITU_BETA, DEFAULT_SITU_LINEAR_BETA, ActivationType
from ...utils import supported_compute_capability
from .blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1 import (
    cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4,
    out_sf_shape,
)
from .blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize import (
    cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4,
)
from .blackwell_sm12x.moe_mxfp8_q0_route_triton import (
    Mxfp8Q0RouteWorkspace,
    mxfp8_q0_route_triton,
)

_SUPPORTED_ACTIVATIONS = (ActivationType.Swiglu, ActivationType.Situ)


def _shares_storage(tensor: torch.Tensor, others) -> bool:
    storage_ptr = tensor.untyped_storage().data_ptr()
    return any(
        other is not None and storage_ptr == other.untyped_storage().data_ptr()
        for other in others
    )


def _validate(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    num_experts: int,
    activation: ActivationType,
    w1_alpha: Optional[torch.Tensor],
    w2_alpha: Optional[torch.Tensor],
) -> None:
    if x.device.type != "cuda":
        raise ValueError("SM12x MXFP8 x MXFP4 fused MoE inputs must be CUDA tensors")
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"x must be bfloat16, got {x.dtype}")
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    if w1_alpha is not None or w2_alpha is not None:
        raise NotImplementedError(
            "per-expert alpha is not wired into the SM12x kernels"
        )
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if token_selected_experts.shape != token_final_scales.shape:
        raise ValueError(
            "token_selected_experts and token_final_scales must have equal shape"
        )
    if token_selected_experts.shape[0] != x.shape[0]:
        raise ValueError("routing tensors must have one row per token")
    if w1_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError(
            "w1_weight and w2_weight must be 3D grouped packed-weight tensors"
        )
    hidden_size = x.shape[1]
    if w1_weight.shape[2] * 2 != hidden_size:
        raise ValueError(
            f"w1_weight K extent {w1_weight.shape[2] * 2} != hidden size {hidden_size}"
        )
    if w2_weight.shape[1] != hidden_size:
        raise ValueError(
            f"w2_weight N extent {w2_weight.shape[1]} != hidden size {hidden_size}"
        )


@supported_compute_capability([120, 121])
@flashinfer_api
def cute_dsl_sm12x_fused_moe_mxfp8_mxfp4(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    num_experts: int,
    w1_alpha: Optional[torch.Tensor] = None,
    w2_alpha: Optional[torch.Tensor] = None,
    moe_output: Optional[torch.Tensor] = None,
    activation_type: int = ActivationType.Swiglu.value,
    situ_beta: float = DEFAULT_SITU_BETA,
    situ_linear_beta: float = DEFAULT_SITU_LINEAR_BETA,
    tactic: Optional[Any] = None,
    workspace13: Optional[torch.Tensor] = None,
    workspace2: Optional[torch.Tensor] = None,
    q0_route_workspace: Optional[Mxfp8Q0RouteWorkspace] = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Run fused MoE with BF16 activations, MXFP8 activation quant and packed MXFP4 weights.

    Parameters
    ----------
    x : torch.Tensor
        BF16 activations with shape ``[num_tokens, hidden_size]``.
    token_selected_experts, token_final_scales : torch.Tensor
        Per-token expert indices and routing scales.
    w1_weight, w2_weight : torch.Tensor
        Packed MXFP4 expert weights; ``w1_weight`` is laid out ``[linear | gate]``.
    w1_weight_sf, w2_weight_sf : torch.Tensor
        Block-32 weight scales in the MN-major UE8M0 SFB layout.
    num_experts : int
        Number of experts spanned by the routing indices.
    moe_output : torch.Tensor, optional
        Caller-owned BF16 output tensor.
    activation_type : int
        Fused activation identifier; SwiGLU and SiTU are supported.
    enable_pdl : bool
        Enable programmatic dependent launch across q0, FC1, and FC2. Defaults to ``True``.
    """
    activation = ActivationType(activation_type)
    _validate(
        x,
        token_selected_experts,
        token_final_scales,
        w1_weight,
        w2_weight,
        num_experts,
        activation,
        w1_alpha,
        w2_alpha,
    )
    if tactic is not None:
        raise NotImplementedError("SM12x fused MoE does not expose tactics yet")

    num_tokens = x.shape[0]
    total_pairs = num_tokens * token_selected_experts.shape[1]
    intermediate_size = w1_weight.shape[1] // 2
    w1_weight_sf = w1_weight_sf.contiguous()
    w2_weight_sf = w2_weight_sf.contiguous()
    q1 = torch.empty(
        total_pairs, intermediate_size, dtype=torch.float8_e4m3fn, device=x.device
    )
    sf1 = torch.zeros(
        out_sf_shape(total_pairs, intermediate_size, num_experts),
        dtype=torch.int32,
        device=x.device,
    )
    q0_workspace_tensors = (
        () if q0_route_workspace is None else tuple(vars(q0_route_workspace).values())
    )
    if moe_output is None:
        out = torch.zeros(num_tokens, x.shape[1], dtype=torch.bfloat16, device=x.device)
    elif _shares_storage(
        moe_output,
        (
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w2_weight,
            w2_weight_sf,
            w1_alpha,
            w2_alpha,
            workspace13,
            workspace2,
            *q0_workspace_tensors,
        ),
    ):
        out = torch.zeros_like(moe_output)
    else:
        moe_output.zero_()
        out = moe_output
    offsets, token_map, token_weights, a_q, a_scale = mxfp8_q0_route_triton(
        x,
        token_selected_experts,
        token_final_scales,
        num_experts,
        workspace=q0_route_workspace,
        workspace13=workspace13,
        workspace2=workspace2,
        enable_pdl=enable_pdl,
    )
    q1, sf1 = cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
        a_q,
        a_scale,
        w1_weight,
        w1_weight_sf,
        offsets,
        tune=False,
        activation=activation,
        enable_pdl=enable_pdl,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        out_q=q1,
        out_sf=sf1,
    )
    out = cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
        q1,
        sf1,
        w2_weight,
        w2_weight_sf,
        offsets,
        token_map,
        token_weights,
        num_tokens,
        tune=False,
        enable_pdl=enable_pdl,
        out=out,
    )
    stream = torch.cuda.current_stream(x.device)
    q1.record_stream(stream)
    sf1.record_stream(stream)
    if moe_output is not None and out is not moe_output:
        moe_output.copy_(out)
        return moe_output
    return out


__all__ = ["cute_dsl_sm12x_fused_moe_mxfp8_mxfp4"]
