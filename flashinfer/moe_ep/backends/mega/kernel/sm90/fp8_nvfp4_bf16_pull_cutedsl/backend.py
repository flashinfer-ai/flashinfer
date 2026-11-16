"""SM90 pull-style NVFP4 x FP8 MegaMoE backend."""

from __future__ import annotations

import functools
from typing import Any

import torch

from ......config import FleetParams
from ......core.kernel.registry import register_mega_kernel
from ......weights import MoEWeightPack
from ..fp8_mxfp4_bf16_pull_cutedsl.backend import Sm90PullMxfp4MegaKernelBackend
from .config import Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig
from .weights import (
    NVFP4_EPILOGUE_COMPENSATION,
    NVFP4_FOLD_K,
    NVFP4_FOLD_M,
    NVFP4_GROUP_SIZE,
    NVFP4_GATE_UP_INTERLEAVE,
    TransformedMegaWeights,
    copy_nvfp4_kperm_hidden,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

_NVFP4_WEIGHT_FORMAT_ID = "nvfp4_e2m1"
_NVFP4_ACTIVATION_FORMAT_ID = "fp8_e4m3_per_token_full_hidden"
_NVFP4_LAYOUT_ID = "nvfp4_sm90_m64_k128_gateup8_global_x8_single_v1"
_FUSED_EXECUTION_MODE = "fused"


@register_mega_kernel("sm90_fp8_nvfp4_bf16_pull_cutedsl")
class Sm90PullNvfp4MegaKernelBackend(Sm90PullMxfp4MegaKernelBackend):
    """NV format and workspace identity over the shared packed-FP4 lifecycle."""

    _kernel_config: Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig
    _autotune_output_error = (
        "compute(output=None) is incompatible with NVFP4 knobs='auto'"
    )
    _weight_format = "NVFP4"
    _frontend_format = "nvfp4"
    _autotune_warning = (
        "NVFP4 knobs='auto' runs a COLLECTIVE compile+timing sweep "
        "at the first forward. Every EP rank must participate; "
        "knobs=None selects a tuned default profile when applicable."
    )
    _validate_transformed = staticmethod(validate_transformed_mega_weights)

    @classmethod
    def kernel_name(cls) -> str:
        return "sm90_fp8_nvfp4_bf16_pull_cutedsl"

    def _knob_resolver_kwargs(self) -> dict[str, Any]:
        k = self._kernel_config
        return {
            "nvfp4_kperm": k.nvfp4_kperm,
            "nvfp4_fc1_k256_overlap": k.nvfp4_fc1_k256_overlap,
        }

    def _format_config_kwargs(self) -> dict[str, Any]:
        return self._knob_resolver_kwargs()

    def _validate_transformed_kwargs(self) -> dict[str, Any]:
        return {"nvfp4_kperm": self._kernel_config.nvfp4_kperm}

    def _staging_kwargs(self) -> dict[str, Any]:
        kperm = self._kernel_config.nvfp4_kperm
        return {
            "write_payload": functools.partial(copy_nvfp4_kperm_hidden, kperm=kperm)
            if kperm
            else None
        }

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        k = self._kernel_config
        return preprocess_mega_weights(
            weights,
            intermediate_size=k.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            expert_chunk_size=k.preprocess_expert_chunk_size,
            nvfp4_kperm=k.nvfp4_kperm,
        )

    def _workspace_pool_key_from_config(self, resolved_config: Any) -> Any:
        return (
            "sm90_fp8_nvfp4_bf16_pull_cutedsl",
            _FUSED_EXECUTION_MODE,
            _NVFP4_WEIGHT_FORMAT_ID,
            _NVFP4_ACTIVATION_FORMAT_ID,
            _NVFP4_LAYOUT_ID,
            NVFP4_GROUP_SIZE,
            NVFP4_FOLD_M,
            NVFP4_FOLD_K,
            NVFP4_GATE_UP_INTERLEAVE,
            NVFP4_EPILOGUE_COMPENSATION,
            torch.cuda.current_device(),
            id(self._ep_comm_group),
            resolved_config,
        )


__all__ = ["Sm90PullNvfp4MegaKernelBackend"]
