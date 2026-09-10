"""Materialize canonical EP weights into fused_moe native views."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .....weights import MoEWeightPack

if TYPE_CHECKING:
    from ......fused_moe.api import MoEConfig, MoEWeightPack as FusedMoEWeightPack


# Shape-keyed permute-index cache shared across experts/layers (mirrors
# flashinfer.fused_moe.prepare._TRTLLM_PERMUTE_CACHE for the fp4 path).
_TRTLLM_BF16_PERMUTE_CACHE: dict = {}


def _block_major_k_weights(
    w13: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffled BlockMajorK weights for the trtllm bf16 routed runner.

    Mirrors the upstream trtllm-gen bf16 prep (tests/moe
    ``trtllm_gen_fused_moe_utils``): gemm1 takes the gated-act w3/w1 row
    reorder + MMA shuffle (``is_gated_act_gemm=True``), gemm2 the plain w2
    shuffle, both with the kernel-internal ``epilogue_tile_m=128``, then
    BlockMajorK conversion.  The previous plain ``shuffle_matrix_a`` with
    tile 64 skipped the gated-act reorder and fed the kernel misordered
    gate/up rows — undetected by EP-vs-kernel tests because both sides
    shared this prep; caught by the torch-oracle tests.
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128
    device = w13.device

    g1, g2 = [], []
    for i in range(w13.shape[0]):
        p = _maybe_get_cached_w3_w1_permute_indices(
            _TRTLLM_BF16_PERMUTE_CACHE,
            w13[i].view(torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=True,
        )
        s = w13[i].view(torch.uint8)[p.to(device)].contiguous()
        g1.append(convert_to_block_layout(s, block_k))

        p = get_w2_permute_indices_with_cache(
            _TRTLLM_BF16_PERMUTE_CACHE,
            w2[i].view(torch.uint8),
            epilogue_tile_m,
        )
        s = w2[i].view(torch.uint8)[p.to(device)].contiguous()
        g2.append(convert_to_block_layout(s, block_k))

    return (
        torch.stack(g1).view(torch.bfloat16),
        torch.stack(g2).view(torch.bfloat16),
    )


def materialize_fused_moe_weights(
    weights: MoEWeightPack,
    moe_config: "MoEConfig",
) -> "FusedMoEWeightPack":
    """Convert canonical :class:`MoEWeightPack` into a fused_moe weight pack."""
    from ......fused_moe.api import (
        CuteDslConfig,
        MoEWeightPack as FusedMoEWeightPack,
        QuantFormat,
        QuantVariant,
        TrtllmBf16Config,
        TrtllmFp4Config,
    )

    pair = moe_config.quant.pair
    experts = moe_config.experts
    routing = moe_config.routing
    num_local = experts.local_num_experts or routing.num_experts
    hidden = weights.w13.shape[-1]
    intermediate = experts.intermediate_size

    pack = FusedMoEWeightPack()
    cute_dsl_prepare_variant = {
        (QuantFormat.NVFP4, QuantFormat.NVFP4): QuantVariant.NVFP4,
        (QuantFormat.MXFP4, QuantFormat.MXFP8): QuantVariant.MXFP4,
        (QuantFormat.NVFP4, QuantFormat.BF16): QuantVariant.W4A16,
    }

    for backend_cfg in moe_config.backend:
        if pair == (QuantFormat.BF16, QuantFormat.BF16) and isinstance(
            backend_cfg, TrtllmBf16Config
        ):
            g1, g2 = _block_major_k_weights(weights.w13, weights.w2)
            pack.prepare_for(
                "trtllm_bf16_routed",
                {
                    "gemm1_weights": g1,
                    "gemm2_weights": g2,
                },
            )
            return pack

        if pair == (QuantFormat.NVFP4, QuantFormat.NVFP4) and isinstance(
            backend_cfg, TrtllmFp4Config
        ):
            view = TrtllmFp4Config.prepare_weights(
                weights.w13,
                weights.w2,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("trtllm_fp4_routed", view)
            return pack

        prepare_variant = cute_dsl_prepare_variant.get(pair)
        if prepare_variant is not None and isinstance(backend_cfg, CuteDslConfig):
            view = CuteDslConfig.prepare_weights(
                weights.w13,
                weights.w2,
                variant=prepare_variant,
                num_local_experts=num_local,
                hidden_size=hidden,
                intermediate_size=intermediate,
                activation=moe_config.activation,
                device=weights.w13.device,
            )
            pack.prepare_for("cute_dsl", view)
            return pack

    raise ValueError(
        f"No fused_moe backend in MoEConfig matches quant pair "
        f"weight={pair[0].name}, activation={pair[1].name}. "
        f"Configured backends: {[type(c).__name__ for c in moe_config.backend]}"
    )
