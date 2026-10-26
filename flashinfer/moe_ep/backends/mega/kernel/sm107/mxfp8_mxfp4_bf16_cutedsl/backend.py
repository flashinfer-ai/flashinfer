"""Rubin MegaMoE with MXFP4 weights and MXFP8 E4M3 activations."""

from ......core.kernel.registry import register_mega_kernel
from ..mxfp8_mxfp8_bf16_cutedsl.backend import Sm107Mxfp8BlockScaledMegaKernelBackend
from .config import Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
from .weights import preprocess_mega_weights, validate_transformed_mega_weights


@register_mega_kernel("sm107_mxfp8_mxfp4_bf16_cutedsl")
class Sm107Mxfp8Mxfp4MegaKernelBackend(Sm107Mxfp8BlockScaledMegaKernelBackend):
    """Reuse MXFP8 staging and communication with packed FP4 expert weights."""

    def __init__(self, config: Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)

    @classmethod
    def kernel_name(cls) -> str:
        return "sm107_mxfp8_mxfp4_bf16_cutedsl"

    def preprocess_weights(self, weights, fleet_params):
        return preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
        )

    def validate_transformed_weights(
        self, transformed_weights, bootstrap, fleet_params
    ):
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )
