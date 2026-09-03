"""CuTeDSL BF16 MegaMoE kernel configuration."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Literal

from ..common.bf16_config import Sm100_Bf16_Cutedsl_MegaMoeConfigBase


@dataclass
class Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(Sm100_Bf16_Cutedsl_MegaMoeConfigBase):
    """Parameters for the SM100 BF16 CuTeDSL MegaMoE kernel."""

    kernel_name: str = "sm100_bf16_bf16_bf16_cutedsl"
    token_back_mode: (
        Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"] | None
    ) = None

    def __post_init__(self) -> None:
        if self.token_back_mode is None:
            # MegaMoEBf16Config rejects ikr + epi_warps, so an ikr session must
            # take a dispatch-warp carrier (reuse is the measured/tested one).
            self.token_back_mode = (
                "reuse_dispatch_warps" if self.in_kernel_fc2_reduce else "epi_warps"
            )
