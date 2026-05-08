"""MoEMicroKernel - direct routed W4A16 micro MoE kernel for SM120/SM121.

Ported from the b12x kernel library to FlashInfer.

The compact W4A16 path dequantizes whole FP4 B tiles into BF16 shared memory
and then runs BF16 MMA.  For tiny routed batches that structure is dominated
by CTA barriers and underfilled producer/consumer phases.  The direct micro
kernel keeps the W4A4 scheduler, but stages 16-bit activations and activated
intermediates directly before dotting with FP4 weights.
"""

from __future__ import annotations

from typing import Tuple

from .moe_direct_micro_kernel import (
    _MAX_DIRECT_K_SEGMENTS,
    _direct_k_segments_supported,
    MoEDirectMicroKernel,
)


class MoEMicroKernel(MoEDirectMicroKernel):
    """Low-latency direct W4A16 path for micro routed batches."""

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        if m not in (1, 2, 4, 8, 10, 12, 16, 24, 32):
            return False
        # CUTLASS 4.5 rejects the wide multi-token direct path at this envelope.
        if m >= 4 and n >= 4096:
            return False
        if k <= 0 or k % (32 * 16) != 0 or k % 128 != 0:
            return False
        if k // 16 > 32 * _MAX_DIRECT_K_SEGMENTS:
            return False
        if n <= 0 or n % 16 != 0:
            return False
        rows_per_warp = max(1, m)
        fc1_chunks = max(1, n // (16 * rows_per_warp))
        if n % fc1_chunks != 0:
            return False
        i_chunk = n // fc1_chunks
        if i_chunk % 16 != 0:
            return False
        k_segments = k // (32 * 16)
        return (
            _direct_k_segments_supported(k_segments)
            and 0 < num_topk <= 32
            and weight_E > 0
        )

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        output_tile_count_n: int,
        *,
        fast_math: bool = False,
        activation: str = "silu",
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
        dynamic_down_scale: bool = False,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            output_tile_count_n,
            fast_math=fast_math,
            activation=activation,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
            dynamic_down_scale=dynamic_down_scale,
            w4a16_mode=True,
        )


__all__ = ["MoEMicroKernel"]
