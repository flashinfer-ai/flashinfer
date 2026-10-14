# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MXFP8 references for the SM120 single-rank and MegaMoE runners."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from moe_sm120_mxfp8_swapab.runner_common import (
    dequant_block_scale_to_fp32,
    swiglu_fold_interleave,
    transpose_rhs_for_block_dequant,
)

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
    the MXFP8 operands.  The launcher's K-major ``b`` and raw SF
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
    ):
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
        if a.shape[1] != k:
            raise ValueError(
                f"A inner dim ({a.shape[1]}) must equal logical K ({k})."
            )
        expected_sf_cols = (k + self.sf_vec_size - 1) // self.sf_vec_size
        if sfa.shape != (m, expected_sf_cols):
            raise ValueError(
                f"SFA must have raw shape {(m, expected_sf_cols)}, got {tuple(sfa.shape)}."
            )
        if sfb.shape != (n, expected_sf_cols):
            raise ValueError(
                f"SFB must have raw shape {(n, expected_sf_cols)}, got {tuple(sfb.shape)}."
            )

        b_nk = transpose_rhs_for_block_dequant(b)
        if tuple(b_nk.shape) != (n, k):
            raise ValueError(
                f"B K-major storage must map to logical {(n, k)}, got "
                f"{tuple(b_nk.shape)} from physical {tuple(b.shape)}."
            )

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

        return a_fp32 @ b_fp32.transpose(0, 1)



def compute_megamoe_reference_mxfp8(*args, **kwargs) -> torch.Tensor:
    """SM120 Swap-A/B MXFP8 reference with register interleave=8."""
    from moe_mxfp8_glu.mega_reference_mxfp8 import (
        compute_megamoe_reference_mxfp8 as _generic_mxfp8_reference,
    )
    from moe_sm120_mxfp8_swapab.sm120_mma import SWAP_AB_INTERLEAVE

    kwargs["gate_up_interleave"] = SWAP_AB_INTERLEAVE
    kwargs["apply_topk_in_fc1"] = True
    return _generic_mxfp8_reference(*args, **kwargs)




__all__ = [
    "_BlockScaledGemmReferenceLauncher",
    "compute_megamoe_reference_mxfp8",
    "reference_expert_fc12",
]
