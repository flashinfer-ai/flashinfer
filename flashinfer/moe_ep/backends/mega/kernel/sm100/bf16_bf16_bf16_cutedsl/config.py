"""CuTeDSL BF16 MegaMoE kernel configuration."""

from __future__ import annotations

from dataclasses import dataclass

from flashinfer.moe_ep._compat import legacy_finalize
from flashinfer.moe_ep.config import FinalizeMode, validate_finalize_mode

from ..common.bf16_config import Sm100_Bf16_Cutedsl_MegaMoeConfigBase


@legacy_finalize
@dataclass
class Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(Sm100_Bf16_Cutedsl_MegaMoeConfigBase):
    """Parameters for the SM100 BF16 CuTeDSL MegaMoE kernel."""

    kernel_name: str = "sm100_bf16_bf16_bf16_cutedsl"
    # NONE makes MoEEpMegaLayer.forward return per-route outputs.
    finalize_mode: FinalizeMode = FinalizeMode.AUTO

    def __post_init__(self) -> None:
        validate_finalize_mode(
            self.finalize_mode,
            (FinalizeMode.AUTO, FinalizeMode.DEFAULT, FinalizeMode.NONE),
        )
        if self.finalize_mode is FinalizeMode.NONE:
            if self.enable_in_kernel_fc2_reduce:
                raise ValueError(
                    "finalize_mode=NONE requires enable_in_kernel_fc2_reduce=False."
                )
            if self.knobs == "auto":
                raise ValueError(
                    "finalize_mode=NONE requires fixed/offline-tuned knobs."
                )
