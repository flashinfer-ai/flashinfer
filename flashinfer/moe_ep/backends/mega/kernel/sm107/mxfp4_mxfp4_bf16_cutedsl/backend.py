"""Rubin MegaMoE with MXFP4 weights and activations."""

from ......core.kernel.registry import register_mega_kernel
from ..mxfp8_mxfp4_bf16_cutedsl.backend import Sm107Mxfp8Mxfp4MegaKernelBackend
from .config import Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig


@register_mega_kernel("sm107_mxfp4_mxfp4_bf16_cutedsl")
class Sm107Mxfp4MegaKernelBackend(Sm107Mxfp8Mxfp4MegaKernelBackend):
    """Reuse MXFP4 weight preparation and the block-scaled execution path."""

    def __init__(self, config: Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)

    @classmethod
    def kernel_name(cls) -> str:
        return "sm107_mxfp4_mxfp4_bf16_cutedsl"
