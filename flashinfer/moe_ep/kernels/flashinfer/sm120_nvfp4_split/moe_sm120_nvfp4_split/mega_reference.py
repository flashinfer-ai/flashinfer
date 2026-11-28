# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVFP4-weight/NVFP4-activation references for SM120 MegaMoE."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from moe_sm120_nvfp4_split.runner_common import (
    dequant_block_scale_to_fp32,
    nvfp4_quantize_per_block_16,
    swiglu_fold_interleave,
)
from moe_sm120_nvfp4_split.sm120_mma import SWAP_AB_INTERLEAVE


def reference_expert_fc12(
    *,
    ref_scaled_mm,
    quantize_fn,
    act_packed: torch.Tensor,
    act_sf: torch.Tensor,
    fc1_weight_packed: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight_packed: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    intermediate: int,
    hidden: int,
    fc1_alpha: float,
    fc2_alpha: float,
    fc1_norm_const: float,
    gate_up_interleave: int,
    gate_up_clamp: Optional[float],
    topk_weights: Optional[torch.Tensor],
    ref_compute_graph: Literal["transformers", "deepgemm"],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-expert fused fc1+fc2 reference shared by the single-rank tester
    and the multi-rank MegaMoE reference.

    Both GEMMs run on the blockscaled launcher (``ref_scaled_mm``) straight off
    the NVFP4 operands.  The launcher's K-major ``b`` and raw SF
    formats are identical for the per-expert single-rank and gathered
    multi-rank tensors.  Returns the fc2 fp32 output (``deepgemm``: topk
    pre-multiplied into SwiGLU; ``transformers``: left unweighted for the caller
    to apply), plus the fc1 hand-off ``(fc1_q, fc1_sf)`` used by the fc1-phase
    ablation.
    """
    intermediate_downproj = intermediate // 2

    fc1_fp32 = ref_scaled_mm(
        a=act_packed, sfa=act_sf,
        b=fc1_weight_packed, sfb=fc1_weight_sf,
        n=intermediate, k=hidden,
    )
    fc1_fp32 = fc1_fp32 * fc1_alpha

    swiglu = swiglu_fold_interleave(
        fc1_fp32, gate_up_interleave, gate_up_clamp=gate_up_clamp,
    )
    if ref_compute_graph == "deepgemm" and topk_weights is not None:
        swiglu = swiglu * topk_weights.unsqueeze(-1)

    fc1_q, fc1_sf_out = quantize_fn(swiglu, fc1_norm_const)

    fc2_fp32 = ref_scaled_mm(
        a=fc1_q, sfa=fc1_sf_out,
        b=fc2_weight_packed, sfb=fc2_weight_sf,
        n=hidden, k=intermediate_downproj,
    )
    fc2_fp32 = fc2_fp32 * fc2_alpha
    return fc2_fp32, fc1_q, fc1_sf_out


class _BlockScaledGemmReferenceLauncher:
    """SM120-safe host wrapper for dense blockscaled GEMM reference calls.

    Keep the same call contract as the device blockscaled GEMM path, while
    computing the reference through CUDA PyTorch dequantization:

        C[M, N] = dequant(A[M, K], SFA[M, K / vec])
                @ dequant(B[N, K], SFB[N, K / vec]).T

    ``B`` is supplied in the same K-major storage used by the kernel
    (physically ``(K, N)``); ``transpose_rhs`` converts
    it to logical ``(N, K)`` before scale expansion.
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        compute_dtype: torch.dtype = torch.float64,
    ):
        self.compute_dtype = compute_dtype
        self.sf_vec_size = sf_vec_size
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn

    def __call__(
        self,
        *,
        a: torch.Tensor,
        sfa: torch.Tensor,
        b: torch.Tensor,
        sfb: torch.Tensor,
        n: int,
        k: int,
    ) -> torch.Tensor:
        """Run C[M,N] = blockscaled(A[M,K], B[N,K]) and return fp32 C."""
        if a.dim() != 2 or b.dim() != 2 or sfa.dim() != 2 or sfb.dim() != 2:
            raise ValueError(
                "blockscaled reference GEMM expects 2D A/B/SFA/SFB tensors; "
                f"got A={a.dim()}D B={b.dim()}D SFA={sfa.dim()}D SFB={sfb.dim()}D."
            )
        m = a.shape[0]
        expected_sf_cols = (k + self.sf_vec_size - 1) // self.sf_vec_size
        if sfa.shape != (m, expected_sf_cols):
            raise ValueError(
                f"SFA must have raw shape {(m, expected_sf_cols)}, got {tuple(sfa.shape)}."
            )
        if sfb.shape != (n, expected_sf_cols):
            raise ValueError(
                f"SFB must have raw shape {(n, expected_sf_cols)}, got {tuple(sfb.shape)}."
            )

        # Runtime weights are logical (N, K), physically packed along K.
        # ``dequant_block_scale_to_fp32`` expands that packed axis in place.
        b_nk = b

        a_fp32 = dequant_block_scale_to_fp32(
            a, sfa, self.sf_vec_size, global_scale=None
        )
        b_fp32 = dequant_block_scale_to_fp32(
            b_nk, sfb, self.sf_vec_size, global_scale=None
        )
        if tuple(a_fp32.shape) != (m, k):
            raise ValueError(
                f"A dequantized shape must be {(m, k)}, got {tuple(a_fp32.shape)}."
            )
        if tuple(b_fp32.shape) != (n, k):
            raise ValueError(
                f"B dequantized shape must be {(n, k)}, got "
                f"{tuple(b_fp32.shape)} from physical {tuple(b.shape)}."
            )

        # Independent dense reference: FP64 accumulation prevents a reference
        # FP32 GEMM from crossing a later BF16 halfway point. Return FP32 to
        # preserve the existing activation/quantization and epilogue contract.
        return (a_fp32.to(self.compute_dtype) @ b_fp32.to(self.compute_dtype).transpose(0, 1)).float()


def _byte_gather_rank_token(
    tensor: torch.Tensor,
    ranks: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    return tensor.view(torch.uint8)[ranks, tokens].view(tensor.dtype)


def _byte_select_expert(
    tensor: torch.Tensor,
    rank: int,
    expert: int,
) -> torch.Tensor:
    return tensor.view(torch.uint8)[rank, expert].view(tensor.dtype)


def compute_megamoe_reference_nvfp4(
    *,
    input_activation: torch.Tensor,
    input_activation_sf: torch.Tensor,
    input_topk_idx: torch.Tensor,
    input_topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_alpha: torch.Tensor,
    fc1_norm_const: torch.Tensor,
    ref_compute_graph: Literal["transformers", "deepgemm"],
    fc2_output_dtype: torch.dtype,
    gate_up_clamp: Optional[float] = None,
    gemm_backend: str = "torch_fp64",
) -> torch.Tensor:
    """Eyeball-reviewable per-expert NVFP4 dispatch/FC1/FC2 reference."""
    num_ranks, tokens_per_rank, num_topk = input_topk_idx.shape
    experts_per_rank = fc1_weight.shape[1]
    total_experts = num_ranks * experts_per_rank
    hidden = fc2_weight.shape[-2]
    intermediate = fc1_weight.shape[-2]
    if gemm_backend == "native_qmma":
        from .native_qmma_reference import NativeQmmaReferenceLauncher
        launcher = NativeQmmaReferenceLauncher()
    elif gemm_backend in ("torch_fp64", "torch_fp32"):
        dtype = torch.float64 if gemm_backend == "torch_fp64" else torch.float32
        launcher = _BlockScaledGemmReferenceLauncher(16, (64, 128), (1, 1), dtype)
    else:
        raise ValueError(f"Unknown GEMM reference backend: {gemm_backend}")
    output = torch.zeros(
        (num_ranks, tokens_per_rank, num_topk, hidden),
        dtype=fc2_output_dtype,
        device=input_activation.device,
    )
    for global_expert in range(total_experts):
        owner = global_expert // experts_per_rank
        local_expert = global_expert % experts_per_rank
        routed = (input_topk_idx == global_expert).nonzero(as_tuple=False)
        if routed.numel() == 0:
            continue
        src_rank, src_token, src_slot = routed.unbind(dim=1)
        fc2, _q, _sf = reference_expert_fc12(
            ref_scaled_mm=launcher,
            quantize_fn=nvfp4_quantize_per_block_16,
            act_packed=_byte_gather_rank_token(
                input_activation, src_rank, src_token
            ),
            act_sf=_byte_gather_rank_token(
                input_activation_sf, src_rank, src_token
            ),
            fc1_weight_packed=_byte_select_expert(
                fc1_weight, owner, local_expert
            ),
            fc1_weight_sf=fc1_weight_sf[owner, local_expert],
            fc2_weight_packed=_byte_select_expert(
                fc2_weight, owner, local_expert
            ),
            fc2_weight_sf=fc2_weight_sf[owner, local_expert],
            intermediate=intermediate,
            hidden=hidden,
            fc1_alpha=float(fc1_alpha[owner, local_expert]),
            fc2_alpha=float(fc2_alpha[owner, local_expert]),
            fc1_norm_const=float(fc1_norm_const[owner, local_expert]),
            gate_up_interleave=SWAP_AB_INTERLEAVE,
            gate_up_clamp=gate_up_clamp,
            topk_weights=input_topk_weights[src_rank, src_token, src_slot],
            ref_compute_graph=ref_compute_graph,
        )
        output[src_rank, src_token, src_slot] = fc2.to(fc2_output_dtype)
    return output


__all__ = [
    "_BlockScaledGemmReferenceLauncher",
    "compute_megamoe_reference_nvfp4",
    "reference_expert_fc12",
]
