"""SM90 FP8-activation / NVFP4-weight pull-style MegaMoE config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ..fp8_mxfp4_bf16_pull_cutedsl.config import (
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from .weights import validate_nvfp4_kperm


@dataclass
class Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
):
    """NVFP4 K16 E4M3 scales with the shared packed-FP4 execution contract.

    ``nvfp4_kperm`` (0/32/64, default 64) applies to weight preparation, input staging,
    and the kernel. K32 is byte-identical to K0; K64 changes accumulation order.
    Compact pull storage is a NV tactic field, rather than a backend selector.
    """

    kernel_name: str = "sm90_fp8_nvfp4_bf16_pull_cutedsl"
    # Remove MX-only constructor fields while retaining the inherited validation.
    # dataclasses supports ClassVar removal; mypy treats it as a mutable override.
    humming_max_range: ClassVar[Literal[11]] = 11  # type: ignore[misc]
    compact_pull_buffer: ClassVar[bool] = False  # type: ignore[misc]
    nvfp4_kperm: int = field(default=64, kw_only=True)
    nvfp4_fc1_k256_overlap: bool | None = field(default=None, kw_only=True)
    _weight_format: ClassVar[str] = "NVFP4"
    _kind_contract: ClassVar[str] = "NVFP4 x FP8"

    def _validate_weight_format(self) -> None:
        validate_nvfp4_kperm(self.nvfp4_kperm)
        if self.nvfp4_fc1_k256_overlap is not None and not isinstance(
            self.nvfp4_fc1_k256_overlap, bool
        ):
            raise ValueError("nvfp4_fc1_k256_overlap must be None or a bool")

    def _validate_active_dispatch_warps(self) -> None:
        if self.active_dispatch_warps is not None and (
            type(self.active_dispatch_warps) is not int
            or self.active_dispatch_warps not in (1, 2, 4)
        ):
            raise ValueError("active_dispatch_warps must be 1, 2, or 4")


__all__ = ["Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig"]
