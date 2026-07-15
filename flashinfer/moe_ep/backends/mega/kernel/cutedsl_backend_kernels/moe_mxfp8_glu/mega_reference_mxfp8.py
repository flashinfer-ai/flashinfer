# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch MoE reference for the multi-rank MXFP8 MegaMoE kernel.

Standalone MXFP8 counterpart of ``moe_nvfp4_swapab.mega_reference`` (the NVFP4
ground truth).  It takes already-MXFP8 inputs (1-byte fp8 data + E8M0 block
scales, ``sf_vec_size = 32``) plus a routing table ``topk_idx`` and computes
``combine_output[r, t, k] = fc12(input[r, t], expert[topk_idx[r, t, k]])`` for
every ``(rank, token, topk_slot)``.

Topk weighting: the MXFP8 fused fc1+fc2 epilogue applies **no** topk weighting
(``swiglu_act`` uses a 1.0 scale and the per-token score is not multiplied into
the swiglu output -- mirroring the MXFP8 fc12 standalone tester, which inherits
the no-op pre/post topk hooks).  ``norm_const`` is hard-coded to 1.0 in the
kernel epilogue.  This reference therefore produces *unweighted* per-(token,
topk) fc12 outputs; callers must multiply by topk weights when reducing form-A
outputs to final per-token outputs.

The per-expert chain (dequant input -> gather -> fc1 -> swiglu fold + fc1-out
MXFP8 round-trip -> fc2 -> scatter back) mirrors the MXFP8 fc12 reference in
``runner_fc12_common._build_reference`` so kernel-vs-reference disagreement is
bounded by the fc1-out MXFP8 RTNE quant and fp32 GEMM accumulation noise.

Zero cuTeDSL / NVSHMEM dependency: importable on CPU-only hosts (the helpers
run on whatever device the input tensors live on).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from common.megamoe_constants import Mxfp8BlockSize
from moe_nvfp4_swapab.runner_common import (
    dequant_block_scale_to_fp32,
    transpose_rhs_for_block_dequant,
    _swiglu_pair_hw_match_cuda,
)
from common.host_utils import mxfp8_quantize_per_block_32


def compute_megamoe_reference_mxfp8(
    # MXFP8 tensors carry LOGICAL shape (fp8 = 1 byte/element, no packing).
    input_activation: torch.Tensor,  # (num_ranks, num_tokens_per_rank, hidden) fp8
    input_activation_sf: torch.Tensor,  # (num_ranks, num_tokens_per_rank, hidden//32) E8M0
    input_topk_idx: torch.Tensor,  # (num_ranks, num_tokens_per_rank, num_topk) int64
    input_topk_weights: torch.Tensor,  # (num_ranks, num_tokens_per_rank, num_topk) fp32
    fc1_weight: torch.Tensor,  # (num_ranks, num_experts_per_rank, hidden, intermediate) fp8, hidden stride-1
    fc1_weight_sf: torch.Tensor,  # (num_ranks, num_experts_per_rank, intermediate, hidden//32) E8M0
    fc2_weight: torch.Tensor,  # (num_ranks, num_experts_per_rank, intermediate//2, hidden) fp8, inter//2 stride-1
    fc2_weight_sf: torch.Tensor,  # (num_ranks, num_experts_per_rank, hidden, (intermediate//2)//32) E8M0
    ab_dtype: torch.dtype,  # torch.float8_e4m3fn or torch.float8_e5m2
    norm_const: float = 1.0,
    ref_compute_graph: Literal["transformers", "deepgemm"] = "deepgemm",
    fc2_output_dtype: torch.dtype = torch.bfloat16,
    gate_up_clamp: Optional[float] = None,
) -> torch.Tensor:
    """Return ``(num_ranks, num_tokens_per_rank, num_topk, hidden)`` combine reference.

    ``norm_const`` / ``ref_compute_graph`` are accepted for API parity with the
    NVFP4 reference. Top-k weighting is intentionally left to callers reducing
    the returned form-A tensor; ``norm_const`` is not applied to the fc1-out
    quant because the MXFP8 kernel hard-codes a 1.0 norm const and
    ``mxfp8_quantize_per_block_32`` takes no norm-const argument.
    """
    if fc2_output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"fc2_output_dtype must be torch.bfloat16 or torch.float16, "
            f"got {fc2_output_dtype}."
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

    # 1. Dequant all input activations once (flatten (rank, token) to 2D).
    input_act_fp32 = dequant_block_scale_to_fp32(
        input_activation.reshape(num_ranks * num_tokens_per_rank, hidden),
        input_activation_sf.reshape(num_ranks * num_tokens_per_rank, -1),
        Mxfp8BlockSize,
        global_scale=None,
    ).reshape(num_ranks, num_tokens_per_rank, hidden)

    combine_ref = torch.zeros(
        (num_ranks, num_tokens_per_rank, num_topk, hidden),
        dtype=fc2_output_dtype,
        device=input_activation.device,
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

        gathered_act = input_act_fp32[source_ranks, source_tokens]  # (R, hidden)

        # fc1: dequant weight (transpose so K=hidden trailing for block dequant,
        # transpose back to (hidden, intermediate)) + GEMM.
        fc1_weight_t_fp32 = dequant_block_scale_to_fp32(
            transpose_rhs_for_block_dequant(fc1_weight[target_rank, local_expert]),
            fc1_weight_sf[target_rank, local_expert],
            Mxfp8BlockSize,
            global_scale=None,
        )
        fc1_weight_fp32 = fc1_weight_t_fp32.transpose(0, 1)  # (hidden, intermediate)
        fc1_output_fp32 = gathered_act @ fc1_weight_fp32  # (R, intermediate)

        # SwiGLU fold: gate/up interleaved at Mxfp8BlockSize (=32) granularity,
        # matching the kernel's PostSwigluHalf interleave for MXFP8.
        _M, _N = fc1_output_fp32.shape
        _n_pairs = _N // (2 * Mxfp8BlockSize)
        _reshaped = fc1_output_fp32.view(_M, _n_pairs, 2, Mxfp8BlockSize)
        _gate = _reshaped[:, :, 0, :]
        _up = _reshaped[:, :, 1, :]
        if gate_up_clamp is not None:
            limit = abs(float(gate_up_clamp))
            _gate = _gate.clamp(max=limit)
            _up = _up.clamp(min=-limit, max=limit)
        swiglu_output = _swiglu_pair_hw_match_cuda(_gate, _up).reshape(
            _M, _N // 2
        )  # (R, intermediate//2)

        # fc1-out MXFP8 round-trip (the only step that introduces kernel-vs-ref
        # disagreement above fp32 accumulation noise).
        fc1_quant, fc1_sf = mxfp8_quantize_per_block_32(swiglu_output, ab_dtype)
        fc1_dequant = dequant_block_scale_to_fp32(
            fc1_quant, fc1_sf, Mxfp8BlockSize, global_scale=None
        )

        fc2_weight_t_fp32 = dequant_block_scale_to_fp32(
            transpose_rhs_for_block_dequant(fc2_weight[target_rank, local_expert]),
            fc2_weight_sf[target_rank, local_expert],
            Mxfp8BlockSize,
            global_scale=None,
        )
        fc2_weight_fp32 = fc2_weight_t_fp32.transpose(0, 1)  # (intermediate//2, hidden)
        fc2_output_fp32 = fc1_dequant @ fc2_weight_fp32  # (R, hidden)

        combine_ref[source_ranks, source_tokens, source_topk_slots, :] = (
            fc2_output_fp32.to(fc2_output_dtype)
        )

    return combine_ref


__all__ = ["compute_megamoe_reference_mxfp8", "Mxfp8BlockSize"]
