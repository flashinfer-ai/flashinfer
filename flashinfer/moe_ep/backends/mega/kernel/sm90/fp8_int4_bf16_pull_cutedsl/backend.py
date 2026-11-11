# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""INT4 format adapter over the shared SM90 MegaMoE lifecycle."""

from typing import Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.registry import register_mega_kernel
from ......weights import MoEWeightPack
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend
from .config import Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig


@register_mega_kernel("sm90_fp8_int4_bf16_pull_cutedsl")
class Sm90PullInt4MegaKernelBackend(Sm90PullMxfp4MegaKernelBackend):
    _kernel_config: Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig
    _weight_format = "INT4"
    _frontend_format = "int4"
    _autotune_output_error = (
        "compute(output=None) is incompatible with INT4 knobs='auto'"
    )
    _autotune_warning = (
        "INT4 knobs='auto' runs a COLLECTIVE compile+timing sweep at the first "
        "forward. Every EP rank must participate; tune before CUDA Graph capture. "
        "knobs=None reads the INT4 cache with heuristic fallback."
    )

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_fp8_int4_bf16_pull_cutedsl"

    def preprocess_weights(self, weights: MoEWeightPack, fleet_params: FleetParams):
        from ......kernel_src.sm90.pull_style_cutedsl_megakernel import (
            Int4WeightPack,
            preprocess_int4_mega_weights,
        )

        if weights.w13_scale is None or weights.w2_scale is None:
            raise ValueError("INT4 requires prequantized weights and BF16 K128 scales")
        return preprocess_int4_mega_weights(
            Int4WeightPack(
                weights.w13, weights.w13_scale, weights.w2, weights.w2_scale
            ),
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            expert_chunk_size=self._kernel_config.preprocess_expert_chunk_size,
        )

    def validate_transformed_weights(
        self,
        transformed_weights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        experts = fleet_params.num_experts // bootstrap.world_size
        hidden = fleet_params.token_hidden_size
        intermediate = self._kernel_config.intermediate_size
        for leg, (rows, k) in zip(
            transformed_weights,
            ((2 * intermediate, hidden), (hidden, intermediate)),
            strict=True,
        ):
            payload, tables, _, _ = leg
            if (
                payload.dtype != torch.uint8
                or tuple(payload.shape) != (experts, k // 2, rows)
                or payload.stride(1) != 1
                or tables.dtype != torch.uint64
                or tuple(tables.shape) != (experts, rows // 64, k // 128, 8, 8)
                or not tables.is_contiguous()
            ):
                raise ValueError(
                    "invalid prepared INT4 payload or register-table layout"
                )

    def _workspace_pool_key_from_config(self, resolved_config: Any) -> Any:
        return (
            self.kernel_name(),
            "signed_int4_bf16_k128_xor_reglut_linear_m64",
            torch.cuda.current_device(),
            id(self._ep_comm_group),
            resolved_config,
        )


__all__ = ["Sm90PullInt4MegaKernelBackend"]
