"""CuTeDSL BF16 MegaMoE kernel configuration."""

from __future__ import annotations

from dataclasses import dataclass

from ..common.bf16_config import Sm100_Bf16_Cutedsl_MegaMoeConfigBase


@dataclass
class Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(Sm100_Bf16_Cutedsl_MegaMoeConfigBase):
    """Parameters for the SM100 BF16 CuTeDSL MegaMoE kernel."""

    kernel_name: str = "sm100_bf16_bf16_bf16_cutedsl"
    # False exposes per-route outputs through MoEEpMegaLayer.forward_unfinalized().
    do_finalize: bool = True

    def __post_init__(self) -> None:
        if not self.do_finalize:
            if self.enable_in_kernel_fc2_reduce:
                raise ValueError(
                    "do_finalize=False requires enable_in_kernel_fc2_reduce=False."
                )
            if self.knobs == "auto":
                raise ValueError(
                    "do_finalize=False requires fixed/offline-tuned knobs."
                )
