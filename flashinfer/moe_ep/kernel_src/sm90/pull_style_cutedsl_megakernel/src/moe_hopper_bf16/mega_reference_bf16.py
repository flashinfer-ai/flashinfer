# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch MoE reference for the multi-rank BF16 MegaMoE kernel.

BF16 counterpart of ``moe_nvfp4_swapab.mega_reference`` (the NVFP4 ground
truth). It takes already-BF16 inputs plus a routing table ``topk_idx`` and
computes ``combine_output[r, t, k] = fc12(input[r, t], expert[topk_idx[r, t, k]])``
for every ``(rank, token, topk_slot)``.

Topk weighting follows the existing NVFP4/MXFP8 compute graphs. ``deepgemm``
multiplies each routing weight into the SwiGLU output before the FC1 output is
rounded to BF16; ``transformers`` leaves per-topk FC2 terms unweighted and
applies the routing weights in the standalone top-k reducer. ``norm_const``
remains hard-coded to 1.0 in the kernel epilogue.

Numerics: both GEMMs read BF16 operands and accumulate in FP32, and the FC1
result is rounded back to BF16 before FC2 consumes it -- that round models the
kernel's BF16 ``fc1_output`` staging buffer, so the reference stays faithful to
the data path at the hand-off point.

Zero cuTeDSL / NVSHMEM dependency: importable on CPU-only hosts (the helpers
run on whatever device the input tensors live on).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from common.megamoe_constants import (
    Fp8GateUpInterleave,
)
from moe_nvfp4_swapab.runner_common import (
    _swiglu_pair_hw_match_cuda,
)
from moe_hopper_bf16.hopper_moe_utils import (
    bf16_reference_mm,
)


def compute_megamoe_reference_bf16(
    input_activation: torch.Tensor,        # (num_ranks, num_tokens_per_rank, hidden) bf16
    input_topk_idx: torch.Tensor,          # (num_ranks, num_tokens_per_rank, num_topk) int64
    input_topk_weights: torch.Tensor,      # (num_ranks, num_tokens_per_rank, num_topk) fp32
    fc1_weight: torch.Tensor,              # (num_ranks, num_experts_per_rank, hidden, intermediate) bf16, hidden stride-1
    fc2_weight: torch.Tensor,              # (num_ranks, num_experts_per_rank, intermediate//2, hidden) bf16, inter//2 stride-1
    norm_const: float = 1.0,
    ref_compute_graph: Literal["transformers", "deepgemm"] = "deepgemm",
    fc2_output_dtype: torch.dtype = torch.bfloat16,
    gate_up_clamp: Optional[float] = None,
    return_fc1_gateup: bool = False,
):
    """Return ``(num_ranks, num_tokens_per_rank, num_topk, hidden)`` combine reference.

    ``norm_const`` is accepted for API parity with the NVFP4 reference.
    ``deepgemm`` applies topk weights before the FC1 output is rounded to BF16;
    ``transformers`` leaves terms unweighted for the standalone reducer.

    When ``return_fc1_gateup`` is True, additionally returns a
    ``{global_expert: (routed_tokens, intermediate_gateup) BF16}`` map of the
    raw pre-SwiGLU gate+up accumulators (consumed by the ``generate_c``
    check).  Columns keep the interleaved raw GEMM order -- that map is
    exactly ``x @ fc1_weight[e]`` before any clamp or SwiGLU fold -- and rows
    follow ``routing_mask.nonzero()``, i.e. (rank, token, topk) row-major.
    """
    if fc2_output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"fc2_output_dtype must be torch.bfloat16 or torch.float16, "
            f"got {fc2_output_dtype}."
        )
    if ref_compute_graph not in ("transformers", "deepgemm"):
        raise ValueError(
            "ref_compute_graph must be 'transformers' or 'deepgemm', "
            f"got {ref_compute_graph!r}."
        )

    num_ranks, num_tokens_per_rank, num_topk = input_topk_idx.shape
    num_experts_per_rank = fc1_weight.shape[1]

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

    return _compute_megamoe_reference_bf16(
        input_activation=input_activation,
        input_topk_idx=input_topk_idx,
        input_topk_weights=input_topk_weights,
        fc1_weight=fc1_weight,
        fc2_weight=fc2_weight,
        ref_compute_graph=ref_compute_graph,
        fc2_output_dtype=fc2_output_dtype,
        gate_up_clamp=gate_up_clamp,
        return_fc1_gateup=return_fc1_gateup,
    )


def _compute_megamoe_reference_bf16(
    *,
    input_activation: torch.Tensor,
    input_topk_idx: torch.Tensor,
    input_topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ref_compute_graph: Literal["transformers", "deepgemm"],
    fc2_output_dtype: torch.dtype,
    gate_up_clamp: Optional[float],
    return_fc1_gateup: bool = False,
):
    """BF16 reference path."""
    num_ranks, num_tokens_per_rank, num_topk = input_topk_idx.shape
    num_experts_per_rank = fc1_weight.shape[1]
    num_total_experts = num_ranks * num_experts_per_rank
    hidden = fc2_weight.shape[-1]

    combine_ref = torch.zeros(
        (num_ranks, num_tokens_per_rank, num_topk, hidden),
        dtype=fc2_output_dtype,
        device=input_activation.device,
    )
    fc1_gateup_per_expert = {} if return_fc1_gateup else None

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Per-expert GEMM chain over routed tokens.
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

            gathered_act = input_activation[source_ranks, source_tokens]  # (R, hidden) bf16
            fc1_output_fp32 = bf16_reference_mm(
                gathered_act, fc1_weight[target_rank, local_expert],
            )                                                          # (R, intermediate)

            if return_fc1_gateup:
                # Raw pre-SwiGLU gate+up snapshot in the kernel's raw N order
                # (kernel c_dtype is BFloat16).
                fc1_gateup_per_expert[global_expert] = fc1_output_fp32.to(
                    torch.bfloat16
                )

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

            # The kernel stages FC1's result in a BF16 workspace before FC2
            # reads it back, so round here too.
            fc2_activation_bf16 = swiglu_output.to(torch.bfloat16)

            fc2_output_fp32 = bf16_reference_mm(
                fc2_activation_bf16, fc2_weight[target_rank, local_expert],
            )                                                          # (R, hidden)

            combine_ref[source_ranks, source_tokens, source_topk_slots, :] = (
                fc2_output_fp32.to(fc2_output_dtype)
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    if return_fc1_gateup:
        return combine_ref, fc1_gateup_per_expert
    return combine_ref


__all__ = [
    "compute_megamoe_reference_bf16",
    "Fp8GateUpInterleave",
]
