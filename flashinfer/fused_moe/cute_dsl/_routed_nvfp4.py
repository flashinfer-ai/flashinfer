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

"""Private implementation of pre-FC2 routing weighting for NVFP4 MoE."""

import torch

from ...utils import get_compute_capability
from ...tllm_enums import ActivationType
from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_act_fusion as gather_gemm,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion as grouped_gemm,
)
from .moe_utils import (
    NVFP4_BLOCK_SIZE,
    NVFP4_VALUES_PER_BYTE,
    PRE_FC2_DIM_ALIGNMENT,
    PRE_FC2_MMA_SHAPE,
    PRE_FC2_CLUSTER_SHAPE,
    moe_unpermute,
)

_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))


def _nvfp4_moe_with_pre_fc2_routing_weights(
    x,
    x_sf,
    weights,
    ids,
    w1,
    w1_sf,
    alpha1,
    scale2,
    w2,
    w2_sf,
    alpha2,
    tiles,
    limits,
    inverse,
    mapping,
    active,
    clamp,
    output,
    use_fused_finalize,
    enable_pdl=False,
):
    """Keep BF16 rounding and routing weighting before FC2 activation quantization.

    The gather GEMM1 applies the BF16-rounded ClampedRelu2, the route weight and
    the NVFP4 quantization in its epilogue, so no BF16 FC1 output is stored.
    """
    tokens, packed_hidden = x.shape
    hidden = packed_hidden * NVFP4_VALUES_PER_BYTE
    top_k = ids.shape[1]
    experts = w1.shape[0]
    common = dict(
        tile_idx_to_expert_idx=tiles,
        num_non_exiting_tiles=active,
        tile_idx_to_mn_limit=limits,
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        sf_vec_size=NVFP4_BLOCK_SIZE,
        mma_tiler_mn=PRE_FC2_MMA_SHAPE,
        cluster_shape_mn=PRE_FC2_CLUSTER_SHAPE,
        enable_pdl=enable_pdl,
    )
    # Zero the atomically finalized output first so GEMM1 -> GEMM2 stay
    # adjacent and can overlap through programmatic dependent launch.
    if use_fused_finalize:
        output.zero_()
        final_output = output
    else:
        final_output = torch.empty(
            (tokens * top_k, hidden), dtype=torch.bfloat16, device=x.device
        )
    # FC2 must not reapply the weights already consumed by the FC1 epilogue.
    # Allocated before GEMM1 so no other launch sits between the two GEMMs.
    final_scales = torch.ones_like(weights, dtype=torch.float32)
    # The epilogue reads one FC2 input scale per local expert and FP32 weights.
    scale2 = scale2.reshape(-1)
    expert_scales = (
        scale2 if scale2.numel() == experts else scale2.expand(experts).contiguous()
    )
    quantized, quantized_sf = gather_gemm(
        a=x,
        b=w1,
        a_scale=x_sf,
        b_scale=w1_sf,
        alpha=alpha1,
        token_id_mapping=mapping,
        global_scale=expert_scales,
        topk=top_k,
        c_dtype="float4_e2m1fn",
        quantize_output=True,
        activation_type=ActivationType.ClampedRelu2.value,
        gated=False,
        clamped_relu2_limit=clamp,
        routing_weights=weights.reshape(-1).float(),
        **common,
    )
    grouped_gemm(
        a=quantized,
        b=w2,
        a_scale=quantized_sf,
        b_scale=w2_sf,
        alpha=alpha2,
        token_final_scales=final_scales,
        out=final_output,
        permuted_idx_to_expanded_idx=mapping,
        use_fused_finalize=use_fused_finalize,
        **common,
    )
    if not use_fused_finalize:
        moe_unpermute(
            final_output,
            output,
            inverse,
            final_scales,
            tokens,
            top_k,
            input_is_expanded=True,
            enable_pdl=enable_pdl,
        )
    return output


def _validate_pre_fc2_output(output, x, tokens, hidden):
    """Check the output buffer before any GPU routing or expert launches."""
    if output.device != x.device:
        raise ValueError("moe_output must be on the same device as x")
    if output.dtype != torch.bfloat16:
        raise TypeError("moe_output must have dtype torch.bfloat16")
    if output.shape != (tokens, hidden):
        raise ValueError(f"moe_output must have shape ({tokens}, {hidden})")
    if not output.is_contiguous():
        raise ValueError("moe_output must be contiguous")


def _validate_pre_fc2_inputs(
    x, x_sf, ids, weights, w1, w1_sf, alpha1, w2, w2_sf, alpha2, scale2
):
    """Validate packed NVFP4 operands and metadata consumed through raw pointers."""
    if get_compute_capability(x.device) not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise NotImplementedError("Pre-FC2 routing weighting requires SM100 or SM103")
    if x.dtype != torch.uint8 or w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise TypeError(
            "Pre-FC2 routing weighting requires packed NVFP4 inputs and weights"
        )
    if w1.device != x.device or w2.device != x.device:
        raise ValueError("Expert weights must be on the same device as x")
    if not w1.is_contiguous() or not w2.is_contiguous():
        raise ValueError("Expert weights must be contiguous")
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("x must be rank 2 and FC1/FC2 expert weights must be rank 3")
    if (
        x.shape[1] * NVFP4_VALUES_PER_BYTE % PRE_FC2_DIM_ALIGNMENT
        or w1.shape[1] % PRE_FC2_DIM_ALIGNMENT
    ):
        raise ValueError("Pre-FC2 routing weighting requires K and N divisible by 128")
    if (
        w1.shape[0] != w2.shape[0]
        or w1.shape[2] != x.shape[1]
        or w2.shape[1] != x.shape[1] * NVFP4_VALUES_PER_BYTE
        or w2.shape[2] * NVFP4_VALUES_PER_BYTE != w1.shape[1]
    ):
        raise ValueError(
            "Packed FC1 and FC2 weights must match the hidden and intermediate widths"
        )
    if ids.dtype != torch.int32 or weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Expert ids must be int32 and routing weights BF16 or float32")
    if ids.shape != weights.shape or ids.ndim != 2 or ids.shape[0] != x.shape[0]:
        raise ValueError("Routing ids and weights must have shape (tokens, top_k)")
    if (
        scale2 is None
        or scale2.dtype != torch.float32
        or scale2.numel() not in (1, w1.shape[0])
    ):
        raise ValueError(
            "fc2_input_scale must contain one float32 value or one per local expert"
        )
    if x_sf is None or tuple(x_sf.shape) not in (
        (x.shape[0], x.shape[1] * NVFP4_VALUES_PER_BYTE // NVFP4_BLOCK_SIZE),
        (x.shape[0], x.shape[1] * NVFP4_VALUES_PER_BYTE // NVFP4_BLOCK_SIZE, 1),
    ):
        raise ValueError("x_sf must contain row-major block-16 input scales")
    if x_sf.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError("x_sf must have dtype uint8 or float8_e4m3fn")
    if any(
        not t.is_contiguous() or t.device != x.device
        for t in (x, x_sf, ids, weights, scale2)
    ):
        raise ValueError(
            "Inputs, routing metadata and FC2 scales must be contiguous on the same device"
        )
    experts, intermediate = w1.shape[0], w1.shape[1]
    hidden = x.shape[1] * NVFP4_VALUES_PER_BYTE
    # The grouped GEMMs read these through raw pointers.
    for name, sf, numel in (
        ("w1_weight_sf", w1_sf, experts * intermediate * hidden // NVFP4_BLOCK_SIZE),
        ("w2_weight_sf", w2_sf, experts * hidden * intermediate // NVFP4_BLOCK_SIZE),
    ):
        if sf is None or sf.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise TypeError(f"{name} must have dtype uint8 or float8_e4m3fn")
        if sf.numel() < numel:
            raise ValueError(f"{name} must hold at least {numel} block scales")
    for name, alpha in (("w1_alpha", alpha1), ("w2_alpha", alpha2)):
        if alpha is None or alpha.dtype != torch.float32 or alpha.numel() != experts:
            raise ValueError(f"{name} must contain one float32 value per local expert")
    # Weight scales may be strided MMA-layout views; the GEMMs take their layout.
    for name, sf in (("w1_weight_sf", w1_sf), ("w2_weight_sf", w2_sf)):
        if sf.device != x.device:
            raise ValueError(f"{name} must be on the same device as x")
    for name, alpha in (("w1_alpha", alpha1), ("w2_alpha", alpha2)):
        if not alpha.is_contiguous() or alpha.device != x.device:
            raise ValueError(f"{name} must be contiguous on the same device as x")
