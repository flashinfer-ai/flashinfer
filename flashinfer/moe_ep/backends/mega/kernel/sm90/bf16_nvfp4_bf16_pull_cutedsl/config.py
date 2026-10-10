"""SM90 (Hopper) pull-style BF16 x NVFP4 (W4A16) mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ..bf16_bf16_bf16_pull_cutedsl.config import (
    Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig,
)

if TYPE_CHECKING:
    import torch


@dataclass
class Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
    Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig
):
    """W4A16 on the SM90 pull-style megakernel: BF16 activations, NVFP4 weights.

    Packed NVFP4 expert weights stay in HBM (about 0.56 B/element) and are
    decoded inside the GEMM: each consumer thread turns its slice of the
    packed E2M1 tile and E4M3 per-16 scales into the register-sourced BF16
    WGMMA operand (exact; E2M1 x E4M3 has at most 6 significant bits).  The
    FP32 per-expert global scale ``fc1_alpha`` / ``fc2_alpha`` (default one)
    is applied to the FP32 accumulator in the epilogue; runtime
    ``MoEEpTensors.fc1_alpha`` / ``fc2_alpha`` override it per forward.
    Dispatch, the FC1 handoff and combine are BF16, as on the BF16 backend.

    Swap-AB only (the weights must be WGMMA operand A to be
    register-sourced): ``swap_ab=False`` raises, and manual geometry defaults
    to swap-AB.  Other fields mean the same as on the BF16 config.
    """

    kernel_name: str = "sm90_bf16_nvfp4_bf16_pull_cutedsl"
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None

    weight_format: ClassVar[str] = "nvfp4"

    def __post_init__(self) -> None:
        if self.swap_ab is False:
            from ......core.validation.common import MoEEpConfigError

            raise MoEEpConfigError(
                "sm90_bf16_nvfp4_bf16_pull_cutedsl is swap-AB only (the NVFP4 "
                "weights are the register-sourced WGMMA A operand)"
            )
        if self.swap_ab is None and (
            self.pingpong is not None
            or self.mma_tiler_mnk is not None
            or self.cluster_shape_mnk is not None
        ):
            # Manual geometry would otherwise fall back to the non-swap default.
            self.swap_ab = True
