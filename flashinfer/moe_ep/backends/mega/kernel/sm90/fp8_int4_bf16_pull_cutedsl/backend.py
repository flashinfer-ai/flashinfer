# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""INT4 format adapter over the shared SM90 MegaMoE lifecycle."""

from typing import Any

import torch

from ......config import FleetParams
from ......core.kernel.registry import register_mega_kernel
from ......weights import MoEWeightPack
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend
from .config import Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)


@register_mega_kernel("sm90_fp8_int4_bf16_pull_cutedsl")
class Sm90PullInt4MegaKernelBackend(Sm90PullMxfp4MegaKernelBackend):
    _kernel_config: Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig
    _weight_format = "INT4"
    _frontend_format = "int4"
    _validate_transformed = staticmethod(validate_transformed_mega_weights)
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

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedMegaWeights:
        return preprocess_mega_weights(
            weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            expert_chunk_size=self._kernel_config.preprocess_expert_chunk_size,
        )

    def _workspace_pool_key_from_config(self, resolved_config: Any) -> Any:
        return (
            self.kernel_name(),
            "signed_int4_bf16_k128_xor_reglut_pair_m16_presigned_v2",
            torch.cuda.current_device(),
            id(self._ep_comm_group),
            resolved_config,
        )


__all__ = ["Sm90PullInt4MegaKernelBackend"]
