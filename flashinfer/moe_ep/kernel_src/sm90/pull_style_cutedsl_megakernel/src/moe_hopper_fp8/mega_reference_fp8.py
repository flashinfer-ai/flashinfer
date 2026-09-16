# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch MoE reference for the multi-rank FP8 MegaMoE kernel.

FP8 counterpart of ``moe_nvfp4_swapab.mega_reference`` (the NVFP4 ground
truth). It takes already-FP8 inputs plus a routing table
``topk_idx`` and computes
``combine_output[r, t, k] = fc12(input[r, t], expert[topk_idx[r, t, k]])`` for
every ``(rank, token, topk_slot)``.

Topk weighting follows the existing NVFP4/MXFP8 compute graphs. ``deepgemm``
multiplies each routing weight into the SwiGLU output before the FC1-output FP8
quantization; ``transformers`` leaves per-topk FC2 terms unweighted and applies
the routing weights in the standalone top-k reducer. ``norm_const`` remains
hard-coded to 1.0 in the kernel epilogue.

The per-expert chain dispatches to either static per-tensor FP32 scales or
DeepGEMM-style blockwise FP32 scales.

Zero cuTeDSL / NVSHMEM dependency: importable on CPU-only hosts (the helpers
run on whatever device the input tensors live on).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch

from common.megamoe_constants import (
    Fp8BlockScaleK,
    Fp8Fc2ActivationScaleK,
    Fp8GateUpInterleave,
    Fp8WeightScaleBlockK,
    Fp8WeightScaleBlockN,
)
from moe_nvfp4_swapab.runner_common import (
    _swiglu_pair_hw_match_cuda,
)
from moe_hopper_fp8.hopper_moe_utils import (
    FP8_ACCUM_MODE_CHOICES,
    compute_fp8_per_tensor_output_dequant_scale_from_absmax,
    fp8_block_scaled_reference_mm,
    fp8_per_tensor_wgmma_reference_mm,
    make_fp8_per_tensor_dequant_scale,
    quantize_fp8_per_token_block,
    quantize_fp8_with_per_token_block_scale,
)


def compute_megamoe_reference_fp8(
    # FP8 tensors carry LOGICAL shape (fp8 = 1 byte/element, no packing).
    input_activation: torch.Tensor,        # (num_ranks, num_tokens_per_rank, hidden) fp8
    input_activation_sf: torch.Tensor,     # retained for ABI compatibility; unused
    input_topk_idx: torch.Tensor,          # (num_ranks, num_tokens_per_rank, num_topk) int64
    input_topk_weights: torch.Tensor,      # (num_ranks, num_tokens_per_rank, num_topk) fp32
    fc1_weight: torch.Tensor,              # (num_ranks, num_experts_per_rank, hidden, intermediate) fp8, hidden stride-1
    fc1_weight_sf: torch.Tensor,           # retained for ABI compatibility; unused
    fc2_weight: torch.Tensor,              # (num_ranks, num_experts_per_rank, intermediate//2, hidden) fp8, inter//2 stride-1
    fc2_weight_sf: torch.Tensor,           # retained for ABI compatibility; unused
    ab_dtype: torch.dtype,                 # torch.float8_e4m3fn or torch.float8_e5m2
    fc1_activation_dequant_scale: Optional[torch.Tensor] = None,
    fc1_weight_dequant_scale: Optional[torch.Tensor] = None,
    fc2_activation_dequant_scale: Optional[torch.Tensor] = None,
    fc2_weight_dequant_scale: Optional[torch.Tensor] = None,
    norm_const: float = 1.0,
    ref_compute_graph: Literal["transformers", "deepgemm"] = "deepgemm",
    fp8_accum_mode: Literal["1xacc", "2xacc"] = "1xacc",
    mma_tiler_k: int = 128,
    fc2_output_dtype: torch.dtype = torch.bfloat16,
    gate_up_clamp: Optional[float] = None,
    return_fc2_activation_dequant_scale: bool = False,
    fp8_scale_mode: Literal["per_tensor", "blockwise"] = "per_tensor",
    fc1_activation_block_scale: Optional[torch.Tensor] = None,
    fc1_weight_block_scale: Optional[torch.Tensor] = None,
    fc2_activation_block_scale: Optional[torch.Tensor] = None,
    fc2_weight_block_scale: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(num_ranks, num_tokens_per_rank, num_topk, hidden)`` combine reference.

    ``ref_compute_graph='deepgemm'`` follows ``fp8_accum_mode``: 1xacc uses
    full-K fast accumulation, while 2xacc promotes once per ``mma_tiler_k``.
    ``norm_const`` is accepted for API parity with the NVFP4 reference.
    ``deepgemm`` applies topk weights before FC1-output quantization;
    ``transformers`` leaves terms unweighted for the standalone reducer.
    ``fp8_scale_mode='blockwise'`` means
    DeepGEMM-style blockwise FP8 scaling: activation scales are per token/K
    block, weight scales are per N/K block, and raw FP8 partial GEMMs are
    promoted by those scales.
    """
    if fc2_output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"fc2_output_dtype must be torch.bfloat16 or torch.float16, "
            f"got {fc2_output_dtype}."
        )
    if fp8_scale_mode not in ("per_tensor", "blockwise"):
        raise ValueError(
            f"fp8_scale_mode must be 'per_tensor' or 'blockwise', "
            f"got {fp8_scale_mode!r}."
        )
    if fp8_accum_mode not in FP8_ACCUM_MODE_CHOICES:
        raise ValueError(
            f"fp8_accum_mode must be one of {FP8_ACCUM_MODE_CHOICES}, "
            f"got {fp8_accum_mode!r}."
        )
    if ref_compute_graph not in ("transformers", "deepgemm"):
        raise ValueError(
            "ref_compute_graph must be 'transformers' or 'deepgemm', "
            f"got {ref_compute_graph!r}."
        )

    num_ranks, num_tokens_per_rank, num_topk = input_topk_idx.shape
    num_experts_per_rank = fc1_weight.shape[1]
    num_total_experts = num_ranks * num_experts_per_rank

    hidden = fc2_weight.shape[-1]
    intermediate = fc1_weight.shape[-1]
    intermediate_downproj = intermediate // 2

    if fc1_weight.shape[0] != num_ranks or fc2_weight.shape[0] != num_ranks:
        raise ValueError(
            f"fc1_weight / fc2_weight must have leading dim num_ranks={num_ranks}, "
            f"got {tuple(fc1_weight.shape)}, {tuple(fc2_weight.shape)}."
        )
    if input_activation.shape[-1] != hidden:
        raise ValueError(
            f"input_activation last dim ({input_activation.shape[-1]}) "
            f"!= hidden ({hidden})."
        )
    if fc1_weight.shape[2] != hidden:
        raise ValueError(
            f"fc1_weight K dim ({fc1_weight.shape[2]}) != hidden ({hidden})."
        )
    if fc2_weight.shape[2] != intermediate_downproj:
        raise ValueError(
            f"fc2_weight K dim ({fc2_weight.shape[2]}) != intermediate_downproj "
            f"({intermediate_downproj}); fc2's K is intermediate//2 "
            f"(post-SwiGLU-fold)."
        )

    if fp8_scale_mode == "blockwise":
        return _compute_megamoe_reference_fp8_blockwise(
            input_activation=input_activation,
            input_activation_sf=input_activation_sf,
            input_topk_idx=input_topk_idx,
            input_topk_weights=input_topk_weights,
            fc1_weight=fc1_weight,
            fc1_weight_sf=fc1_weight_sf,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            ab_dtype=ab_dtype,
            fc1_activation_block_scale=fc1_activation_block_scale,
            fc1_weight_block_scale=fc1_weight_block_scale,
            fc2_activation_block_scale=fc2_activation_block_scale,
            fc2_weight_block_scale=fc2_weight_block_scale,
            fc2_output_dtype=fc2_output_dtype,
            gate_up_clamp=gate_up_clamp,
            ref_compute_graph=ref_compute_graph,
            return_fc2_activation_block_scale=return_fc2_activation_dequant_scale,
        )

    return _compute_megamoe_reference_fp8_per_tensor(
        input_activation=input_activation,
        input_topk_idx=input_topk_idx,
        input_topk_weights=input_topk_weights,
        fc1_weight=fc1_weight,
        fc2_weight=fc2_weight,
        ab_dtype=ab_dtype,
        fc1_activation_dequant_scale=fc1_activation_dequant_scale,
        fc1_weight_dequant_scale=fc1_weight_dequant_scale,
        fc2_activation_dequant_scale=fc2_activation_dequant_scale,
        fc2_weight_dequant_scale=fc2_weight_dequant_scale,
        ref_compute_graph=ref_compute_graph,
        fp8_accum_mode=fp8_accum_mode,
        mma_tiler_k=mma_tiler_k,
        fc2_output_dtype=fc2_output_dtype,
        gate_up_clamp=gate_up_clamp,
        return_fc2_activation_dequant_scale=return_fc2_activation_dequant_scale,
    )


def _compute_megamoe_reference_fp8_per_tensor(
    *,
    input_activation: torch.Tensor,
    input_topk_idx: torch.Tensor,
    input_topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ab_dtype: torch.dtype,
    fc1_activation_dequant_scale: Optional[torch.Tensor],
    fc1_weight_dequant_scale: Optional[torch.Tensor],
    fc2_activation_dequant_scale: Optional[torch.Tensor],
    fc2_weight_dequant_scale: Optional[torch.Tensor],
    ref_compute_graph: Literal["transformers", "deepgemm"],
    fp8_accum_mode: Literal["1xacc", "2xacc"],
    mma_tiler_k: int,
    fc2_output_dtype: torch.dtype,
    gate_up_clamp: Optional[float],
    return_fc2_activation_dequant_scale: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """FP8 per-tensor reference path."""
    num_ranks, num_tokens_per_rank, num_topk = input_topk_idx.shape
    num_experts_per_rank = fc1_weight.shape[1]
    num_total_experts = num_ranks * num_experts_per_rank
    hidden = fc2_weight.shape[-1]

    if fc1_activation_dequant_scale is None:
        fc1_activation_dequant_scale = make_fp8_per_tensor_dequant_scale(
            input_activation
        )
    if fc1_weight_dequant_scale is None:
        fc1_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
            fc1_weight, reduce_dims=(2, 3)
        )
    if fc2_weight_dequant_scale is None:
        fc2_weight_dequant_scale = make_fp8_per_tensor_dequant_scale(
            fc2_weight, reduce_dims=(2, 3)
        )

    fc1_activation_dequant_scale = fc1_activation_dequant_scale.to(
        device=input_activation.device, dtype=torch.float32
    )
    fc1_weight_dequant_scale = fc1_weight_dequant_scale.to(
        device=input_activation.device, dtype=torch.float32
    )
    fc2_weight_dequant_scale = fc2_weight_dequant_scale.to(
        device=input_activation.device, dtype=torch.float32
    )

    # 1. Convert all input activations once (flatten (rank, token) to 2D) and
    # apply the activation scale used by the transformers-style reference.
    input_act_fp32 = (
        input_activation.reshape(num_ranks * num_tokens_per_rank, hidden).to(torch.float32)
        * fc1_activation_dequant_scale[0]
    ).reshape(num_ranks, num_tokens_per_rank, hidden)

    combine_ref = torch.zeros(
        (num_ranks, num_tokens_per_rank, num_topk, hidden),
        dtype=fc2_output_dtype,
        device=input_activation.device,
    )

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if fc2_activation_dequant_scale is None:
            swiglu_absmax = torch.zeros(
                (), dtype=torch.float32, device=input_activation.device
            )
            for global_expert in range(num_total_experts):
                target_rank = global_expert // num_experts_per_rank
                local_expert = global_expert % num_experts_per_rank

                routing_mask = input_topk_idx == global_expert
                if not routing_mask.any():
                    continue
                routed = routing_mask.nonzero(as_tuple=False)
                source_ranks = routed[:, 0]
                source_tokens = routed[:, 1]
                source_topk_slots = routed[:, 2]

                if ref_compute_graph == "deepgemm":
                    gathered_act = input_activation[source_ranks, source_tokens]
                    fc1_output_fp32 = fp8_per_tensor_wgmma_reference_mm(
                        gathered_act,
                        fc1_weight[target_rank, local_expert],
                        accum_mode=fp8_accum_mode,
                        k_chunk=mma_tiler_k,
                    )
                    fc1_output_fp32 = (
                        fc1_output_fp32
                        * fc1_activation_dequant_scale[0]
                        * fc1_weight_dequant_scale[target_rank, local_expert]
                    )
                else:
                    gathered_act = input_act_fp32[source_ranks, source_tokens]
                    fc1_weight_fp32 = (
                        fc1_weight[target_rank, local_expert].to(torch.float32)
                        * fc1_weight_dequant_scale[target_rank, local_expert]
                    )
                    fc1_output_fp32 = gathered_act @ fc1_weight_fp32

                _M, _N = fc1_output_fp32.shape
                _n_pairs = _N // (2 * Fp8GateUpInterleave)
                _reshaped = fc1_output_fp32.view(
                    _M, _n_pairs, 2, Fp8GateUpInterleave
                )
                _gate = _reshaped[:, :, 0, :]
                _up = _reshaped[:, :, 1, :]
                if gate_up_clamp is not None:
                    limit = abs(float(gate_up_clamp))
                    _gate = _gate.clamp(max=limit)
                    _up = _up.clamp(min=-limit, max=limit)
                swiglu_output = _swiglu_pair_hw_match_cuda(_gate, _up).reshape(
                    _M, _N // 2
                )
                if ref_compute_graph == "deepgemm":
                    topk_weight = input_topk_weights[
                        source_ranks, source_tokens, source_topk_slots
                    ].to(torch.float32).unsqueeze(-1)
                    swiglu_output = swiglu_output * topk_weight
                if swiglu_output.numel() > 0:
                    swiglu_absmax = torch.maximum(
                        swiglu_absmax,
                        swiglu_output.to(torch.float32).abs().amax(),
                    )

            fc2_activation_dequant_scale = (
                compute_fp8_per_tensor_output_dequant_scale_from_absmax(
                    swiglu_absmax,
                    ab_dtype,
                    device=input_activation.device,
                )
            )
        else:
            fc2_activation_dequant_scale = fc2_activation_dequant_scale.to(
                device=input_activation.device, dtype=torch.float32
            )

        # 2. Per-expert GEMM chain over routed tokens.
        for global_expert in range(num_total_experts):
            target_rank = global_expert // num_experts_per_rank
            local_expert = global_expert % num_experts_per_rank

            routing_mask = input_topk_idx == global_expert
            if not routing_mask.any():
                continue
            routed = routing_mask.nonzero(as_tuple=False)
            source_ranks = routed[:, 0]
            source_tokens = routed[:, 1]
            source_topk_slots = routed[:, 2]

            if ref_compute_graph == "deepgemm":
                gathered_act = input_activation[source_ranks, source_tokens]
                fc1_output_fp32 = fp8_per_tensor_wgmma_reference_mm(
                    gathered_act,
                    fc1_weight[target_rank, local_expert],
                    accum_mode=fp8_accum_mode,
                    k_chunk=mma_tiler_k,
                )
                fc1_output_fp32 = (
                    fc1_output_fp32
                    * fc1_activation_dequant_scale[0]
                    * fc1_weight_dequant_scale[target_rank, local_expert]
                )
            else:
                gathered_act = input_act_fp32[source_ranks, source_tokens]  # (R, hidden)
                fc1_weight_fp32 = (
                    fc1_weight[target_rank, local_expert].to(torch.float32)
                    * fc1_weight_dequant_scale[target_rank, local_expert]
                )                                                          # (hidden, intermediate)
                fc1_output_fp32 = gathered_act @ fc1_weight_fp32           # (R, intermediate)

            # SwiGLU fold: gate/up interleaved at Fp8GateUpInterleave (=8)
            # granularity, matching the kernel's PostSwigluHalf interleave.
            _M, _N = fc1_output_fp32.shape
            _n_pairs = _N // (2 * Fp8GateUpInterleave)
            _reshaped = fc1_output_fp32.view(
                _M, _n_pairs, 2, Fp8GateUpInterleave
            )
            _gate = _reshaped[:, :, 0, :]
            _up = _reshaped[:, :, 1, :]
            if gate_up_clamp is not None:
                limit = abs(float(gate_up_clamp))
                _gate = _gate.clamp(max=limit)
                _up = _up.clamp(min=-limit, max=limit)
            swiglu_output = _swiglu_pair_hw_match_cuda(_gate, _up).reshape(
                _M, _N // 2
            )                                                          # (R, intermediate//2)
            if ref_compute_graph == "deepgemm":
                topk_weight = input_topk_weights[
                    source_ranks, source_tokens, source_topk_slots
                ].to(torch.float32).unsqueeze(-1)
                swiglu_output = swiglu_output * topk_weight

            fc2_act_fp8 = (swiglu_output / fc2_activation_dequant_scale[0]).to(ab_dtype)

            if ref_compute_graph == "deepgemm":
                fc2_output_fp32 = fp8_per_tensor_wgmma_reference_mm(
                    fc2_act_fp8,
                    fc2_weight[target_rank, local_expert],
                    accum_mode=fp8_accum_mode,
                    k_chunk=mma_tiler_k,
                )
                fc2_output_fp32 = (
                    fc2_output_fp32
                    * fc2_activation_dequant_scale[0]
                    * fc2_weight_dequant_scale[target_rank, local_expert]
                )
            else:
                fc1_dequant = (
                    fc2_act_fp8.to(torch.float32) * fc2_activation_dequant_scale[0]
                )
                fc2_weight_fp32 = (
                    fc2_weight[target_rank, local_expert].to(torch.float32)
                    * fc2_weight_dequant_scale[target_rank, local_expert]
                )                                                         # (intermediate//2, hidden)
                fc2_output_fp32 = fc1_dequant @ fc2_weight_fp32            # (R, hidden)

            combine_ref[source_ranks, source_tokens, source_topk_slots, :] = (
                fc2_output_fp32.to(fc2_output_dtype)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    if return_fc2_activation_dequant_scale:
        return combine_ref, fc2_activation_dequant_scale
    return combine_ref


def _check_block_scale_shape(
    name: str,
    tensor: torch.Tensor,
    expected: Tuple[int, ...],
) -> torch.Tensor:
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}."
        )
    return tensor.to(dtype=torch.float32)


def _compute_megamoe_reference_fp8_blockwise(
    *,
    input_activation: torch.Tensor,
    input_activation_sf: torch.Tensor,
    input_topk_idx: torch.Tensor,
    input_topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    ab_dtype: torch.dtype,
    fc1_activation_block_scale: Optional[torch.Tensor],
    fc1_weight_block_scale: Optional[torch.Tensor],
    fc2_activation_block_scale: Optional[torch.Tensor],
    fc2_weight_block_scale: Optional[torch.Tensor],
    fc2_output_dtype: torch.dtype,
    gate_up_clamp: Optional[float],
    ref_compute_graph: Literal["transformers", "deepgemm"],
    return_fc2_activation_block_scale: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """DeepGEMM-style block-scale reference.

    The FC2 activation block scale is naturally aligned with routed
    ``(rank, token, topk_slot)`` rows in this pure global reference.  Kernel
    workspace code can later remap it to physical FC1 output rows.
    """
    num_ranks, num_tokens_per_rank, num_topk = input_topk_idx.shape
    num_experts_per_rank = fc1_weight.shape[1]
    num_total_experts = num_ranks * num_experts_per_rank
    hidden = fc2_weight.shape[-1]
    intermediate = fc1_weight.shape[-1]
    intermediate_downproj = intermediate // 2

    if hidden % Fp8BlockScaleK != 0:
        raise ValueError(f"hidden={hidden} must be divisible by {Fp8BlockScaleK}.")
    if intermediate % Fp8WeightScaleBlockN != 0:
        raise ValueError(
            f"intermediate={intermediate} must be divisible by "
            f"{Fp8WeightScaleBlockN}."
        )
    if intermediate_downproj % Fp8Fc2ActivationScaleK != 0:
        raise ValueError(
            f"intermediate_downproj={intermediate_downproj} must be divisible by "
            f"{Fp8Fc2ActivationScaleK}."
        )
    if intermediate_downproj % Fp8WeightScaleBlockK != 0:
        raise ValueError(
            f"intermediate_downproj={intermediate_downproj} must be divisible by "
            f"{Fp8WeightScaleBlockK}."
        )

    if fc1_activation_block_scale is None:
        fc1_activation_block_scale = input_activation_sf
    if fc1_weight_block_scale is None:
        fc1_weight_block_scale = fc1_weight_sf
    if fc2_weight_block_scale is None:
        fc2_weight_block_scale = fc2_weight_sf

    fc1_activation_block_scale = _check_block_scale_shape(
        "fc1_activation_block_scale",
        fc1_activation_block_scale,
        (num_ranks, num_tokens_per_rank, hidden // Fp8BlockScaleK),
    ).to(device=input_activation.device)
    fc1_weight_block_scale = _check_block_scale_shape(
        "fc1_weight_block_scale",
        fc1_weight_block_scale,
        (
            num_ranks,
            num_experts_per_rank,
            intermediate // Fp8WeightScaleBlockN,
            hidden // Fp8WeightScaleBlockK,
        ),
    ).to(device=input_activation.device)
    fc2_weight_block_scale = _check_block_scale_shape(
        "fc2_weight_block_scale",
        fc2_weight_block_scale,
        (
            num_ranks,
            num_experts_per_rank,
            hidden // Fp8WeightScaleBlockN,
            intermediate_downproj // Fp8WeightScaleBlockK,
        ),
    ).to(device=input_activation.device)

    fc2_act_scale_shape = (
        num_ranks,
        num_tokens_per_rank,
        num_topk,
        intermediate_downproj // Fp8Fc2ActivationScaleK,
    )
    if fc2_activation_block_scale is not None:
        fc2_activation_block_scale_ref = _check_block_scale_shape(
            "fc2_activation_block_scale",
            fc2_activation_block_scale,
            fc2_act_scale_shape,
        ).to(device=input_activation.device)
        fc2_activation_scale_is_provided = True
    else:
        fc2_activation_block_scale_ref = torch.zeros(
            fc2_act_scale_shape,
            dtype=torch.float32,
            device=input_activation.device,
        )
        fc2_activation_scale_is_provided = False

    combine_ref = torch.zeros(
        (num_ranks, num_tokens_per_rank, num_topk, hidden),
        dtype=fc2_output_dtype,
        device=input_activation.device,
    )

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for global_expert in range(num_total_experts):
            target_rank = global_expert // num_experts_per_rank
            local_expert = global_expert % num_experts_per_rank

            routing_mask = input_topk_idx == global_expert
            if not routing_mask.any():
                continue
            routed = routing_mask.nonzero(as_tuple=False)
            source_ranks = routed[:, 0]
            source_tokens = routed[:, 1]
            source_topk_slots = routed[:, 2]

            gathered_act = input_activation[source_ranks, source_tokens]
            gathered_act_scale = fc1_activation_block_scale[
                source_ranks, source_tokens
            ]
            fc1_output_fp32 = fp8_block_scaled_reference_mm(
                gathered_act,
                fc1_weight[target_rank, local_expert],
                gathered_act_scale,
                fc1_weight_block_scale[target_rank, local_expert],
                a_scale_block_k=Fp8BlockScaleK,
                b_scale_block_n=Fp8WeightScaleBlockN,
                b_scale_block_k=Fp8WeightScaleBlockK,
            )

            _M, _N = fc1_output_fp32.shape
            _n_pairs = _N // (2 * Fp8GateUpInterleave)
            _reshaped = fc1_output_fp32.view(
                _M, _n_pairs, 2, Fp8GateUpInterleave
            )
            _gate = _reshaped[:, :, 0, :]
            _up = _reshaped[:, :, 1, :]
            if gate_up_clamp is not None:
                limit = abs(float(gate_up_clamp))
                _gate = _gate.clamp(max=limit)
                _up = _up.clamp(min=-limit, max=limit)
            swiglu_output = _swiglu_pair_hw_match_cuda(_gate, _up).reshape(
                _M, _N // 2
            )
            if ref_compute_graph == "deepgemm":
                topk_weight = input_topk_weights[
                    source_ranks, source_tokens, source_topk_slots
                ].to(torch.float32).unsqueeze(-1)
                swiglu_output = swiglu_output * topk_weight

            # Both kernel variants reuse one FP32 reciprocal per scale block.
            if fc2_activation_scale_is_provided:
                fc2_act_scale = fc2_activation_block_scale_ref[
                    source_ranks, source_tokens, source_topk_slots
                ]
                fc2_act_fp8 = quantize_fp8_with_per_token_block_scale(
                    swiglu_output,
                    fc2_act_scale,
                    ab_dtype,
                    block_k=Fp8Fc2ActivationScaleK,
                    use_reciprocal_multiply=True,
                )
            else:
                fc2_act_fp8, fc2_act_scale = quantize_fp8_per_token_block(
                    swiglu_output,
                    ab_dtype,
                    block_k=Fp8Fc2ActivationScaleK,
                    use_reciprocal_multiply=True,
                )
                fc2_activation_block_scale_ref[
                    source_ranks, source_tokens, source_topk_slots
                ] = fc2_act_scale

            fc2_output_fp32 = fp8_block_scaled_reference_mm(
                fc2_act_fp8,
                fc2_weight[target_rank, local_expert],
                fc2_act_scale,
                fc2_weight_block_scale[target_rank, local_expert],
                a_scale_block_k=Fp8Fc2ActivationScaleK,
                b_scale_block_n=Fp8WeightScaleBlockN,
                b_scale_block_k=Fp8WeightScaleBlockK,
            )
            combine_ref[source_ranks, source_tokens, source_topk_slots, :] = (
                fc2_output_fp32.to(fc2_output_dtype)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    if return_fc2_activation_block_scale:
        return combine_ref, fc2_activation_block_scale_ref
    return combine_ref

__all__ = [
    "compute_megamoe_reference_fp8",
    "Fp8GateUpInterleave",
]
